#!/usr/bin/env python

import logging

from dqn import DQNBot
from example import main
from nn_bot import BotFactory, get_parser

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_DEFAULT_BOT_NAME = "DQNBot"
_DEFAULT_CHECKPOINT_EVERY = 50


if __name__ == "__main__":
    parser = get_parser("Launch DQN-based bots", default_name=_DEFAULT_BOT_NAME)
    args = parser.parse_args()

    main(
        BotFactory(DQNBot),
        name=args.name,
        server_url=args.server_url,
        namespace=args.namespace,
        n_bots=args.n_bots,
        model_dir=args.model_dir,
        arch_json=args.model_arch,
        init_model=args.init_model,
        eval_mode=args.eval_mode,
        train_config=args.train_config,
    )
