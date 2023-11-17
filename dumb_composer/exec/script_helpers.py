import argparse
import json
import logging
import math
import os
import pdb
import random
import sys
import time
import traceback
from functools import partial
from multiprocessing import Pool
from typing import Any, Iterable


def custom_excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
    pdb.post_mortem(exc_traceback)


def get_default_output_folder(base_folder: str):
    return os.path.join(base_folder, f"{int(time.time())}")


def setup_logging(debug_level):
    # Map debug level string to logging level
    debug_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    # Set up the logging system
    logging.basicConfig(level=debug_levels.get(debug_level, logging.WARNING))


# (Malcolm 2023-11-15) since some attributes are not JSON serializable,
#   this is a little more complicated than I supposed
# def save_settings(dataclasses: dict[str, Any], outpath: str):
#     with open(outpath, "w") as outf:
#         json.dump(dataclasses, outf, indent=2)


def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--basename-prefix", type=str, default=None)
    parser.add_argument("--runner-config", "-R", type=str, default=None)
    parser.add_argument("--seed", "-S", type=int, default=42)
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="warning",
        help="Set the debugging level",
    )
    parser.add_argument("rntxt_paths", type=str, nargs="+")
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--shuffle-input-paths", action="store_true")
    return parser


def worker_func(i_and_path, *, f, n_paths: int, base_seed, **kwargs):
    i, rntxt_path = i_and_path
    random.seed(base_seed + i)
    print(f"{i}/{n_paths}: {rntxt_path} ")
    try:
        f(rntxt_path=rntxt_path, **kwargs)
    except TimeoutError:
        print(f"Timeout on {rntxt_path}")
    except Exception as exc:
        print(f"Exception on {rntxt_path}: {repr(exc)}")


def run(f, cli_args: argparse.Namespace, **f_kwargs):
    setup_logging(cli_args.log_level)
    random.seed(cli_args.seed)
    output_folder = cli_args.output_folder

    paths = cli_args.rntxt_paths
    if cli_args.shuffle_input_paths:
        random.shuffle(paths)
    if cli_args.max_files is not None:
        paths = paths[: cli_args.max_files]

    if cli_args.num_workers > 1:
        chunk_size = math.ceil(len(paths) / cli_args.num_workers)
        with Pool(cli_args.num_workers) as pool:
            pool.map(
                partial(
                    worker_func,
                    f=f,
                    n_paths=len(paths),
                    runner_settings_path=cli_args.runner_config,
                    base_seed=cli_args.seed,
                    output_folder=output_folder,
                    basename_prefix=cli_args.basename_prefix,
                    **f_kwargs,
                ),
                enumerate(paths),
                chunksize=chunk_size,
            )
    else:
        for i, rntxt_path in enumerate(paths, start=1):
            print(f"{i}/{len(paths)}: {rntxt_path} ")
            try:
                f(
                    rntxt_path=rntxt_path,
                    output_folder=output_folder,
                    runner_settings_path=cli_args.runner_config,
                    basename_prefix=cli_args.basename_prefix,
                    **f_kwargs,
                )
            except TimeoutError:
                print("Timeout")

            except Exception as exc:
                if cli_args.debug:
                    raise
                print(f"Exception on {rntxt_path}: {repr(exc)}")
