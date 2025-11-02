import contextlib
import copy
import logging
import os
import random
import threading
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from nn import NeuralNetworkBot
from utils import ACTIONS

logger = logging.getLogger(__name__)


class DQNBot(NeuralNetworkBot):
    def __init__(
        self,
        name,
        server_url,
        namespace,
        eval_mode: bool,
        model_dir,
        train_config,
        init_model=None,
        arch_json=None,
    ):
        super().__init__(
            name=name,
            server_url=server_url,
            namespace=namespace,
            model_dir=model_dir,
            eval_mode=eval_mode,
            train_config=train_config,
            init_model=init_model,
            arch_json=arch_json,
        )

        if eval_mode:
            self.lock = contextlib.nullcontext()
        else:
            # thread lock to avoid concurent model update
            self.lock = threading.Lock()

    def load_train_config(self, train_config):
        super().load_train_config(train_config)

        self.batch_size = train_config.get("batch_size", 64)
        buffer_size = train_config.get("buffer_size", 5000)
        self.replay = deque(maxlen=buffer_size)

        self.gamma = train_config.get("gamma", 0.99)
        self.epsilon = train_config.get("epsilon_start", 1.0)
        self.epsilon_end = train_config.get("epsilon_end", 0.05)
        self.epsilon_decay = train_config.get("epsilon_decay", 0.9999)

        # Target network
        self.target_model = copy.deepcopy(self.model)
        self.target_model.to(self.device)
        self.target_model.eval()

    def load_reward_config(self, reward_config):
        self.reward_map = reward_config

    def init_match(self):
        return {}

    def choose_action_train(self, x, turn_state, match_state):
        if random.random() < self.epsilon:
            action_idx = np.random.randint(len(ACTIONS))
        else:
            with torch.no_grad() and self.lock:
                self.model.eval()
                q_values = self.model(x.unsqueeze(0)).squeeze(0)
                action_idx = q_values.argmax().item()

        if "last_state" in match_state:
            self.observe(match_state, next_state=x, reward=0.0, final=0.0)

        match_state["last_state"] = x
        match_state["last_action"] = action_idx

        return action_idx

    def choose_action_eval(self, x, turn_state, match_state):
        with torch.no_grad() and self.lock:
            self.model.eval()
            q_values = self.model(x.unsqueeze(0)).squeeze(0)
            action_idx = q_values.argmax().item()
        return action_idx

    def observe(self, match_state, next_state, reward, final):
        s = match_state.pop("last_state")
        a = match_state.pop("last_action")
        self.replay.append((s, a, reward, next_state, final))

        if len(self.replay) >= self.batch_size:
            self.train_step()

    def train_step(self):
        batch = random.sample(self.replay, self.batch_size)
        s, a, r, s2, f = zip(*batch)
        s = torch.stack(s)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.stack(s2)
        f = torch.tensor(f, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_q = self.target_model(s2).max(1, keepdim=True)[0]
            target = r + self.gamma * (1 - f) * next_q

        with self.lock:
            self.model.train()
            q_vals = self.model(s).gather(1, a)
            loss = nn.functional.smooth_l1_loss(q_vals, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.update_count += 1
        # Update epsilon and target network
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if self.update_count % self.checkpoint_every == 0:
            with self.lock:
                self.target_model.load_state_dict(self.model.state_dict())
                checkpoint_path = os.path.join(self.model_dir, f"{self.name}.pt")
                torch.save(self.model, checkpoint_path)
            logger.info(
                f"[{self.name}] saved checkpoint%{self.update_count} at "
                f"{checkpoint_path} and target network updated"
            )

    def match_end_feedback(self, match_state, result, score, others):
        if self.eval_mode:
            return

        if "last_state" in match_state:
            self.observe(
                match_state,
                next_state=match_state["last_state"],  # this is just a place holder
                reward=self.reward_map[result],
                final=1.0,
            )
