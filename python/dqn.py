import contextlib
import copy
import logging
import os
import random
import threading
from collections import deque

import torch
import torch.nn as nn

from nn import NeuralNetworkBot
from utils import PASS, TAKE

logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.DEBUG)


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
        self.explore_p_pass = train_config.get("explore_p_pass", 0.5)

        self.clip_target = train_config.get("clip_target")

        # Target network
        self.target_model = copy.deepcopy(self.model)
        self.target_model.to(self.device)
        self.target_model.eval()

    def load_reward_config(self, reward_config):
        self.reward_config = reward_config

    def init_match(self):
        return {}

    def choose_action_train(self, x, turn_state, match_state):
        if turn_state.you.chips <= 0:
            action_idx = TAKE
        elif random.random() < self.epsilon:
            action_idx = PASS if random.random() < self.explore_p_pass else TAKE
        else:
            with torch.no_grad() and self.lock:
                self.model.eval()
                q_values = self.model(x.unsqueeze(0)).squeeze(0)
                action_idx = q_values.argmax().item()

        reward = self.turn_reward(
            match_state,
            turn_state.you.score(),
            [p.score() for p in turn_state.others],
        )

        if "last_state" in match_state:
            self.observe(
                match_state,
                next_state=x,
                reward=reward,
                final=0.0,
                broke=1.0 if turn_state.you.chips <= 0 else 0.0,  # broke at this turn
            )

        match_state["last_state"] = x
        match_state["last_action"] = action_idx

        return action_idx

    def choose_action_eval(self, x, turn_state, match_state):
        if turn_state.you.chips <= 0:
            action_idx = TAKE
        else:
            with torch.no_grad() and self.lock:
                self.model.eval()
                q_values = self.model(x.unsqueeze(0)).squeeze(0)
                action_idx = q_values.argmax().item()
        return action_idx

    def observe(self, match_state, next_state, reward, final, broke):
        """
        Args:
           next_state: input for next state
           reward: reward seen on next turn
           final: if next state is ending state (match end)
           broke: if next state is broke (no chips) == action has to be TAKE
        """
        s = match_state.pop("last_state")
        a = match_state.pop("last_action")
        self.replay.append((s, a, reward, next_state, final, broke))

        if len(self.replay) >= self.batch_size:
            self.train_step()

    def train_step(self):
        batch = random.sample(self.replay, self.batch_size)
        s, a, r, s2, f, b = zip(*batch)
        s = torch.stack(s)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.stack(s2)
        f = torch.tensor(f, dtype=torch.float32, device=self.device).unsqueeze(1)
        b = torch.tensor(b, dtype=torch.float32, device=self.device).unsqueeze(1)

        logger.debug(f"{s.shape=} {a.shape=} {r.shape=} {f.shape=} {b.shape=}")

        with torch.no_grad():
            next_qa = self.target_model(s2)
            if self.clip_target is not None:
                next_qa = next_qa.clip(min=-self.clip_target, max=self.clip_target)
            next_q_broke = next_qa[:, [TAKE]]
            next_q = next_qa.max(1, keepdim=True)[0]
            target = r + self.gamma * (1 - f) * ((1 - b) * next_q + b * next_q_broke)

        logger.debug(
            f"{next_qa.shape=} {next_q_broke.shape=} {next_q.shape=} {target.shape=}"
        )

        with self.lock:
            self.model.train()
            q_vals = self.model(s).gather(1, a)
            logger.debug(f"{q_vals.shape=}")

            loss = nn.functional.smooth_l1_loss(q_vals, target)

            logger.debug(
                f"{torch.max(loss)=} {torch.max(torch.abs(q_vals))=} {torch.max(torch.abs(target))=}"
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
            self.record_loss(loss)

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

        if self.update_count % self.log_loss_every == 0:
            avg_loss = self.log_loss_to_file()
            logger.info(f"[{self.name}] loss@{self.update_count}: {avg_loss}")

        if self.update_count % self.lr_decay == 0:
            self.lr_scheduler.step()

    def match_end_feedback(self, match_state, result, score, others):
        if self.eval_mode:
            return

        if "last_state" in match_state:
            self.observe(
                match_state,
                next_state=match_state["last_state"],  # this is just a place holder
                reward=self.match_end_reward(match_state, result, score, others),
                final=1.0,
                broke=0.0,  # doesn't matter for the ending state
            )

    def turn_reward(self, match_state, score, others):
        rel_score = min(others) - score
        rel_score_delta = rel_score - match_state.get("rel_score", 0)
        match_state["rel_score"] = rel_score
        return self.reward_config["rel_score"] * rel_score_delta

    def match_end_reward(self, match_state, result, score, others):
        return self.reward_config[result] + self.turn_reward(match_state, score, others)
