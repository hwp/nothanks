#!/usr/bin/env python

import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import string
import logging
import time

from bot import Bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_DEFAULT_SERVER_URL = "http://localhost:3000"
_DEFAULT_NAMESPACE = "/bots"
_DEFAULT_BOT_COUNT = "3"
_DEFAULT_BOT_NAME = "NNBot"


def compute_score(cards, chips):
    """Compute No Thanks! score: sum of lowest cards in each sequence - chips."""
    if not cards:
        return -chips
    cards = sorted(cards)
    total = cards[0]
    for i in range(1, len(cards)):
        if cards[i] != cards[i - 1] + 1:
            total += cards[i]
    return total - chips


class NeuralNetworkBot(Bot):
    def __init__(self, name, server_url, namespace, lr=0.001):
        super().__init__(name, server_url, namespace, sequential=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    # === Interface methods ===

    def init_match(self):
        """Initialize per-match memory."""
        return {
            "chosen_log_probs": [],
            "score_history": []
        }

    def extract_feature(self, turn_state):
        """Convert turn state to model input tensor."""
        return torch.tensor([
            turn_state.current,
            turn_state.pot,
            turn_state.you.chips,
            len(turn_state.others),
        ], dtype=torch.float32, device=self.device)

    def sample_action_from_probs(self, probs):
        """
        Sample action index from probability distribution.
        Returns:
            int: 0 = take, 1 = pass
        """
        return np.random.choice(len(probs), p=probs)

    # === Core game logic ===

    def choose_action(self, turn_state, match_state) -> str:
        x = self.extract_feature(turn_state)

        logits = self.model(x)
        log_probs = torch.log_softmax(logits, dim=0)
        probs = torch.exp(log_probs).detach().cpu().numpy()

        action_idx = self.sample_action_from_probs(probs)
        action = "take" if action_idx == 0 else "pass"

        # store chosen log-prob for REINFORCE update
        chosen_log_prob = log_probs[action_idx]
        match_state["chosen_log_probs"].append(chosen_log_prob)

        # record per-player scores
        scores = {
            "score": compute_score(turn_state.you.cards, turn_state.you.chips),
            "others": [
                compute_score(p.cards, p.chips) for p in turn_state.others
            ]
        }
        match_state["score_history"].append(scores)

        return action

    def compute_reward(self, score_history, result):
        """Compute per-step rewards based on score progression."""
        others_scores = np.array([s["others"] for s in score_history])
        self_scores = np.array([s["score"] for s in score_history])
        rel = others_scores.mean(axis=1) - self_scores
        score_delta = np.diff(rel)

        result_reward = {"win": 1.0, "draw": 0.0, "loss": -1.0}[result]
        rewards = score_delta + result_reward * 20.0
        return rewards

    def match_end_feedback(self, match_state, result, score, others):
        match_state["score_history"].append({"score": score, "others": others})
        rewards = self.compute_reward(match_state["score_history"], result)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        chosen_log_probs = torch.stack(match_state["chosen_log_probs"])
        loss = -(chosen_log_probs * rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logger.info(f"[{self.name}] updated model, avg reward={rewards.mean().item():.3f}")

def main(name, server_url, namespace, n_bots):
    torch.autograd.set_detect_anomaly(True)

    suffixes = [''.join(random.choices(string.ascii_lowercase,k=3)) for _ in range(n_bots)]
    bots = [ NeuralNetworkBot(f"{name}-{s}", server_url, namespace) for s in suffixes ]

    try:
        logger.info("All bots: Connecting ...")
        for bot in bots:
            bot.connect()
        logger.info("All bots: Connected ...")

        try:
            while True:
                time.sleep(10)  # Wait 1 second in each loop iteration
        except KeyboardInterrupt:
            pass
    finally:
        logger.info("All bots: Disconnecting ...")
        for bot in bots:
            bot.disconnect()
        logger.info("All bots: Disconnected ...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Example bots")
    parser.add_argument("--server-url", type=str, default=_DEFAULT_SERVER_URL)
    parser.add_argument("--namespace", type=str, default=_DEFAULT_NAMESPACE)
    parser.add_argument("--n-bots", type=int, default=_DEFAULT_BOT_COUNT)
    parser.add_argument("--name", type=str, default=_DEFAULT_BOT_NAME)

    args = parser.parse_args()

    main(args.name, args.server_url, args.namespace, args.n_bots)
