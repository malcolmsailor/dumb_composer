import argparse
import random
import sys

from dumb_composer.exec.incremental_contrapuntist_runner import run_incremental_composer
from dumb_composer.exec.script_helpers import custom_excepthook, setup_logging

sys.excepthook = custom_excepthook


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-config", "-R", type=str, default=None)
    parser.add_argument("--contrapuntist-config", "-C", type=str, default=None)
    parser.add_argument("--seed", "-S", type=int, default=42)
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="warning",
        help="Set the debugging level",
    )
    parser.add_argument("rntxt_paths", type=str, nargs="+")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    setup_logging(args.log_level)
    random.seed(42)
    for rntxt_path in args.rntxt_paths:
        run_incremental_composer(
            args.runner_config, args.contrapuntist_config, rntxt_path
        )


if __name__ == "__main__":
    main()
