#!/usr/bin/env python

import logging
import random
import string

from example import get_parser as get_base_parser
from example import main
from nn import NeuralNetworkBot

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_DEFAULT_BOT_NAME = "NNBot"
_DEFAULT_MODEL_DIR = "models"


class BotFactory:
    def __init__(self, get_one):
        self.get_one = get_one

    def __call__(self, n_bots, name, **kwargs):
        suffixes = [
            "".join(random.choices(string.ascii_lowercase, k=3)) for _ in range(n_bots)
        ]
        bots = [self.get_one(f"{name}-{s}", **kwargs) for s in suffixes]
        return bots


def get_parser(description, default_name):
    parser = get_base_parser(description=description, default_name=default_name)
    parser.add_argument("--model-dir", type=str, default=_DEFAULT_MODEL_DIR)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model-arch", type=str, help="JSON file specifying model architecture"
    )
    group.add_argument(
        "--init-model", type=str, help="Name of pretrained model to load from model_dir"
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--eval-mode", action="store_true")
    group.add_argument(
        "--train-config",
        type=str,
        help="JSON file specifying training config. Not used for eval.",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser("Launch NN-based bots", default_name=_DEFAULT_BOT_NAME)
    args = parser.parse_args()

    main(
        BotFactory(NeuralNetworkBot),
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
