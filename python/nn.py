import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bot import Bot
from utils import (
    ACTIONS,
    compute_score,
    feature_from_player_state,
    result_and_score_reward,
    to_binary_vector,
)

logger = logging.getLogger(__name__)


def build_model_from_json(arch_def):
    """Recursively build nn.Module from JSON list structure"""
    module_type, module_args = arch_def

    if module_type == "Sequential":
        return nn.Sequential(*[build_model_from_json(a) for a in module_args])
    elif hasattr(nn, module_type):
        return getattr(nn, module_type)(**module_args)
    else:
        raise ValueError(f"Unknown layer type: {module_type}")


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
        action = ACTIONS[action_idx]

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
