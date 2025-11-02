#!/usr/bin/env python

import argparse
import logging
import random
import string
import time

import torch

from dqn import DQNBot

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_DEFAULT_SERVER_URL = "http://localhost:3000"
_DEFAULT_NAMESPACE = "/bots"
_DEFAULT_BOT_COUNT = 1
_DEFAULT_BOT_NAME = "DQNBot"
_DEFAULT_CHECKPOINT_EVERY = 50
_DEFAULT_MODEL_DIR = "models"
_DEFAULT_REWARD_CONFIG = "reward_config.json"


def main(
    name,
    eval_mode,
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
        DQNBot(
            f"{name}-{s}",
            server_url,
            namespace,
            eval_mode=eval_mode,
            model_dir=model_dir,
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
    parser = argparse.ArgumentParser("DQNBots")
    parser.add_argument("--eval-mode", action='store_true')
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
        args.eval_mode,
        args.server_url,
        args.namespace,
        args.n_bots,
        args.model_dir,
        arch_json=args.model_arch,
        checkpoint_every=args.checkpoint_every_updates,
        init_model=args.init_model,
        reward_config=args.reward_config,
    )
