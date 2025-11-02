#!/usr/bin/env python

import argparse
import logging
import random
import string
import time

from bot import Bot

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_DEFAULT_SERVER_URL = "http://localhost:3000"
_DEFAULT_NAMESPACE = "/bots"
_DEFAULT_BOT_COUNT = 1
_DEFAULT_BOT_NAME = "SamplePythonBot"


class RandomBot(Bot):
    def __init__(self, name, server_url, namespace, p_pass):
        super().__init__(name, server_url, namespace)
        self.p_pass = p_pass

    def choose_action(self, turn_state, match_state) -> str:
        if turn_state.you.chips <= 0:
            return "take"

        if turn_state.current - turn_state.pot <= 0:
            return "take"

        return "pass" if random.random() < self.p_pass else "take"


def bot_factory(n_bots, name, **kwargs):
    suffixes = [
        "".join(random.choices(string.ascii_lowercase, k=3)) for _ in range(n_bots)
    ]
    if n_bots > 1:
        p_pass_list = [0.3 + 0.6 * i / (n_bots - 1) for i in range(n_bots)]
    else:
        p_pass_list = [0.5]
    return [
        RandomBot(f"{name}-{s}-p{p:.2f}", p_pass=p, **kwargs)
        for s, p in zip(suffixes, p_pass_list)
    ]


def get_parser(description, default_name):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--server-url", type=str, default=_DEFAULT_SERVER_URL)
    parser.add_argument("--namespace", type=str, default=_DEFAULT_NAMESPACE)
    parser.add_argument("--n-bots", type=int, default=_DEFAULT_BOT_COUNT)
    parser.add_argument("--name", type=str, default=default_name)
    return parser


def main(bot_factory, **kwargs):
    bots = bot_factory(**kwargs)

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
    parser = get_parser("Launch example bots", default_name=_DEFAULT_BOT_NAME)
    args = parser.parse_args()

    main(
        bot_factory,
        name=args.name,
        server_url=args.server_url,
        namespace=args.namespace,
        n_bots=args.n_bots,
    )
