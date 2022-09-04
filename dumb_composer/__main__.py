import argparse
import logging
import os
import random

from midi_to_notes import df_to_midi

from dumb_composer.dumb_composer import PrefabComposer


def configure_logging(log_file_path, log_level, append_to_log):
    if log_file_path is None:
        return
    if not append_to_log:
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
    loglevel = getattr(logging, log_level.upper())
    logging.basicConfig(filename=log_file_path, level=loglevel)


SEED = 42  # TODO eventually remove

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="path to roman text input")
    parser.add_argument("-o", "--output-file", help="path to midi output")
    parser.add_argument("-l", "--log-file", help="path to log file")
    parser.add_argument(
        "-L",
        "--log-level",
        choices=("debug", "info", "warn"),
        default="warn",
        help="log level",
    )
    parser.add_argument(
        "--append-to-log",
        action="store_true",
        help="append to log file (if it exists)",
    )
    parser.add_argument("-s", "--seed", type=int, default=SEED)
    args = parser.parse_args()
    configure_logging(args.log_file, args.log_level, args.append_to_log)
    if args.output_file is None:
        logging.warn("No output file provided, skipping output")
    if args.seed is not None:
        logging.debug(f"Setting seed {args.seed}")
        random.seed(args.seed)
    composer = PrefabComposer()
    print(f"Building score from {args.input_file}")
    out, ts = composer(args.input_file, return_ts=True)
    if args.output_file is not None:
        print(f"Writing {args.output_file}")
        df_to_midi(out, args.output_file, ts)
