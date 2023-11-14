import argparse
import math
import random
import sys
from functools import partial
from multiprocessing import Pool

from dumb_composer.exec.incremental_contrapuntist_with_accomps_runner import (
    run_contrapuntist_with_accomps,
)
from dumb_composer.exec.script_helpers import custom_excepthook, setup_logging

sys.excepthook = custom_excepthook


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-config", "-R", type=str, default=None)
    parser.add_argument("--contrapuntist-config", "-C", type=str, default=None)
    parser.add_argument("--accomp-config", "-A", type=str, default=None)
    parser.add_argument("--seed", "-S", type=int, default=42)
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="warning",
        help="Set the debugging level",
    )
    parser.add_argument("rntxt_paths", type=str, nargs="+")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args


def sub(
    i_and_path,
    n_paths: int,
    runner_config,
    contrapuntist_config,
    accomp_config,
    base_seed,
):
    i, rntxt_path = i_and_path
    random.seed(base_seed + i)
    print(f"{i}/{n_paths}: {rntxt_path} ")
    try:
        run_contrapuntist_with_accomps(
            runner_config,
            contrapuntist_config,
            accomp_config,
            rntxt_path,
        )
    except TimeoutError:
        print(f"Timeout on {rntxt_path}")
    except Exception as exc:
        # TODO: (Malcolm 2023-10-18) debug these exceptions
        print(f"Exception on {rntxt_path}: {repr(exc)}")


def main():
    args = get_args()
    setup_logging(args.log_level)
    random.seed(args.seed)
    if args.num_workers > 1:
        paths = args.rntxt_paths
        chunk_size = math.ceil(len(paths) / args.num_workers)
        with Pool(args.num_workers) as pool:
            pool.map(
                partial(
                    sub,
                    n_paths=len(paths),
                    runner_config=args.runner_config,
                    contrapuntist_config=args.contrapuntist_config,
                    accomp_config=args.accomp_config,
                    base_seed=args.seed,
                ),
                enumerate(paths),
                chunksize=chunk_size,
            )
    else:
        for i, rntxt_path in enumerate(args.rntxt_paths, start=1):
            print(f"{i}/{len(args.rntxt_paths)}: {rntxt_path} ")
            try:
                run_contrapuntist_with_accomps(
                    args.runner_config,
                    args.contrapuntist_config,
                    args.accomp_config,
                    rntxt_path,
                )
            except TimeoutError:
                print("Timeout")

            except Exception as exc:
                if args.debug:
                    raise
                print(f"Exception on {rntxt_path}: {repr(exc)}")


if __name__ == "__main__":
    main()
