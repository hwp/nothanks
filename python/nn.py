import json
import logging
import os

import numpy as np
import torch
import torch.optim as optim

from bot import Bot
from modules import build_model_from_json
from utils import (
    ACTIONS,
    TAKE,
    feature_from_player_state,
    result_and_score_reward,
    result_reward,
    to_binary_vector,
)

logger = logging.getLogger(__name__)


class NeuralNetworkBot(Bot):
    def __init__(
        self,
        name,
        server_url,
        namespace,
        model_dir,
        eval_mode: bool,
        train_config: str,
        init_model: str = None,
        arch_json: str = None,
    ):
        super().__init__(
            name=name, server_url=server_url, namespace=namespace, sequential=True
        )
        self.eval_mode = eval_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir

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

        if not eval_mode:
            with open(train_config, "r") as f:
                train_config_json = json.load(f)
            self.load_train_config(train_config_json)

    def load_train_config(self, train_config):
        self.checkpoint_every = train_config.get("checkpoint_every", 100)
        lr = train_config.get("lr", 0.001)
        self.lr_decay = train_config.get("lr_decay", 1_000_000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = optim.lr_scheduler.MultiplicativeLR(
            self.optimizer, lambda epoch: 0.5**epoch
        )
        self.clip_grad = train_config.get("clip_grad")
        self.update_count = 0

        self.log_loss_every = train_config.get("log_loss_every", 100)
        self.loss_count = 0
        self.loss_sum = 0.0

        self.load_reward_config(train_config.get("reward_config"))

    def load_reward_config(self, reward_config):
        reward_type = reward_config["type"]
        if reward_type == "result_and_score":
            self.compute_reward = lambda score_history, result: result_and_score_reward(
                reward_config["config"], score_history, result
            )
        elif reward_type == "result":
            self.compute_reward = lambda _, result: result_reward(result)
        else:
            raise ValueError(f"Unknown {reward_type=}")

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

    def choose_action_train(self, x, turn_state, match_state):
        self.model.train()
        logits = self.model(x)
        log_probs = torch.log_softmax(logits, dim=0)
        probs = torch.exp(log_probs).detach().cpu().numpy()

        if turn_state.you.chips <= 0:
            action_idx = TAKE
        else:
            action_idx = self.sample_action_from_probs(probs)

        match_state["chosen_logits"].append(logits[action_idx])
        scores = {
            "score": turn_state.you.score(),
            "others": [p.score() for p in turn_state.others],
        }
        match_state["score_history"].append(scores)

        return action_idx

    def choose_action_eval(self, x, turn_state, match_state):
        if turn_state.you.chips <= 0:
            action_idx = TAKE
        else:
            with torch.no_grad():
                self.model.eval()
                logits = self.model(x)
                action_idx = logits.argmax().item()
        return action_idx

    def choose_action(self, turn_state, match_state) -> str:
        x = self.extract_feature(turn_state)

        if self.eval_mode:
            action_idx = self.choose_action_eval(x, turn_state, match_state)
        else:
            action_idx = self.choose_action_train(x, turn_state, match_state)

        return ACTIONS[action_idx]

    def match_end_feedback(self, match_state, result, score, others):
        if self.eval_mode:
            return

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
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        self.record_loss(loss)

        # checkpoint model every N updates
        self.update_count += 1
        if self.update_count % self.checkpoint_every == 0:
            checkpoint_path = os.path.join(self.model_dir, f"{self.name}.pt")
            torch.save(self.model, checkpoint_path)
            logger.info(f"[{self.name}] saved checkpoint at {checkpoint_path}")

        logger.info(
            f"[{self.name}] updated model, avg reward={rewards.mean().item():.3f}"
        )

        if self.update_count % self.log_loss_every == 0:
            avg_loss = self.log_loss_to_file()
            logger.info(f"[{self.name}] loss@{self.update_count}: {avg_loss}")

        if self.update_count % self.lr_decay == 0:
            self.lr_scheduler.step()

    def record_loss(self, loss):
        self.loss_count += 1
        self.loss_sum += loss.detach()

    def log_loss_to_file(self):
        avg_loss = self.loss_sum.cpu().item() / self.loss_count
        loss_log_path = os.path.join(self.model_dir, f"{self.name}.loss")
        with open(loss_log_path, "a") as f:
            print(f"{self.update_count} {avg_loss}", file=f)

        self.loss_count = 0
        self.loss_sum = 0.0
        return avg_loss
