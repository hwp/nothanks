#!/usr/bin/env python

import argparse
import json
import logging
import os
import random
import string
import time

import numpy as np
import torch
import torch.optim as optim

from bot import Bot
from utils import (
    build_model_from_json,
    compute_score,
    feature_from_player_state,
    result_and_score_reward,
    to_binary_vector,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_DEFAULT_SERVER_URL = "http://localhost:3000"
_DEFAULT_NAMESPACE = "/bots"
_DEFAULT_BOT_COUNT = 1
_DEFAULT_BOT_NAME = "NNBot"
_DEFAULT_CHECKPOINT_EVERY = 50  # number of updates between saving checkpoints
_DEFAULT_MODEL_DIR = "models"
_DEFAULT_REWARD_CONFIG = "reward_config.json"


class NeuralNetworkBot(Bot):
    def __init__(
        self,
        name,
        server_url,
        namespace,
        model_dir,
        reward_config,
        init_model: str = None,
        arch_json: str = None,
        lr=0.001,
        checkpoint_every=100,
    ):
        super().__init__(name, server_url, namespace, sequential=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.checkpoint_every = checkpoint_every
        self.update_count = 0
        with open(reward_config, "r") as f:
            self.reward_config = json.load(f)

        if init_model is not None and arch_json is not None:
            raise ValueError("init_model and arch_json are mutually exclusive")

        if init_model is not None:
            # Load pretrained model
            model_path = os.path.join(model_dir, f"{init_model}.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Pretrained model not found: {model_path}")
            self.model = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            logger.info(f"[{name}] Loaded pretrained model from {model_path}")
        elif arch_json is not None:
            # Build model from JSON architecture
            with open(arch_json, "r") as f:
                arch_def = json.load(f)
            self.model = build_model_from_json(arch_def).to(self.device)
            logger.info(f"[{name}] Built model from architecture {arch_json}")
        else:
            raise ValueError("Either init_model or arch_json must be provided")

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def init_match(self):
        return {"chosen_logits": [], "score_history": []}

    def extract_feature(self, turn_state):
        return torch.tensor(
            [
                *to_binary_vector([turn_state.current]),
                (turn_state.pot - 3.0) / 3.0,
                *feature_from_player_state(turn_state.you),
                *[f for p in turn_state.others for f in feature_from_player_state(p)],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def sample_action_from_probs(self, probs):
        if probs[0] < 0.1:
            probs = [0.1, 0.9]
        elif probs[0] > 0.9:
            probs = [0.9, 0.1]
        return np.random.choice(len(probs), p=probs)

    def choose_action(self, turn_state, match_state) -> str:
        x = self.extract_feature(turn_state)

        logits = self.model(x)
        log_probs = torch.log_softmax(logits, dim=0)
        probs = torch.exp(log_probs).detach().cpu().numpy()

        action_idx = self.sample_action_from_probs(probs)
        action = "take" if action_idx == 0 else "pass"

        match_state["chosen_logits"].append(logits[action_idx])

        scores = {
            "score": compute_score(turn_state.you.cards, turn_state.you.chips),
            "others": [compute_score(p.cards, p.chips) for p in turn_state.others],
        }
        match_state["score_history"].append(scores)

        return action

    def compute_reward(self, score_history, result):
        if self.reward_config["type"] == "result_and_score":
            return result_and_score_reward(
                self.reward_config["config"], score_history, result
            )
        else:
            raise ValueError(f"Unknown {self.reward_config['type']=}")

    def match_end_feedback(self, match_state, result, score, others):
        if not match_state["chosen_logits"]:
            logger.warning(f"[{self.name}] no history of logits found")
            return

        match_state["score_history"].append({"score": score, "others": others})
        rewards = self.compute_reward(match_state["score_history"], result)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        chosen_logits = torch.stack(match_state["chosen_logits"])
        loss = -(chosen_logits * rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # checkpoint model every N updates
        self.update_count += 1
        if self.update_count % self.checkpoint_every == 0:
            checkpoint_path = os.path.join(self.model_dir, f"{self.name}.pt")
            torch.save(self.model, checkpoint_path)
            logger.info(f"[{self.name}] saved checkpoint at {checkpoint_path}")

        logger.info(
            f"[{self.name}] updated model, avg reward={rewards.mean().item():.3f}"
        )


def main(
    name,
    server_url,
    namespace,
    n_bots,
    model_dir,
    arch_json,
    checkpoint_every,
    init_model,
    reward_config,
):
    torch.autograd.set_detect_anomaly(True)

    suffixes = [
        "".join(random.choices(string.ascii_lowercase, k=3)) for _ in range(n_bots)
    ]
    bots = [
        NeuralNetworkBot(
            f"{name}-{s}",
            server_url,
            namespace,
            model_dir,
            reward_config=reward_config,
            init_model=init_model,
            arch_json=arch_json,
            checkpoint_every=checkpoint_every,
        )
        for s in suffixes
    ]

    try:
        logger.info("All bots: Connecting ...")
        for bot in bots:
            bot.connect()
        logger.info("All bots: Connected ...")

        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            pass
    finally:
        logger.info("All bots: Disconnecting ...")
        for bot in bots:
            bot.disconnect()
        logger.info("All bots: Disconnected ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NeuralNetworkBots")
    parser.add_argument("--server-url", type=str, default=_DEFAULT_SERVER_URL)
    parser.add_argument("--namespace", type=str, default=_DEFAULT_NAMESPACE)
    parser.add_argument("--n-bots", type=int, default=_DEFAULT_BOT_COUNT)
    parser.add_argument("--name", type=str, default=_DEFAULT_BOT_NAME)
    parser.add_argument("--model-dir", type=str, default=_DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--checkpoint-every-updates", type=int, default=_DEFAULT_CHECKPOINT_EVERY
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model-arch", type=str, help="JSON file specifying model architecture"
    )
    group.add_argument(
        "--init-model", type=str, help="Name of pretrained model to load from model_dir"
    )

    parser.add_argument(
        "--reward-config",
        type=str,
        default=_DEFAULT_REWARD_CONFIG,
        help="JSON file specifying reward function",
    )

    args = parser.parse_args()

    main(
        args.name,
        args.server_url,
        args.namespace,
        args.n_bots,
        args.model_dir,
        arch_json=args.model_arch,
        checkpoint_every=args.checkpoint_every_updates,
        init_model=args.init_model,
        reward_config=args.reward_config,
    )
