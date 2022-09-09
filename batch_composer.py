import argparse
import os
import random

from dumb_composer.composer_wrangler import ComposerWrangler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "-n",
        "--n-output-files",
        type=int,
        default=0,
        help="if 0, set to the number of files in --input-dir",
    )
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-M", "--no-midi", action="store_true")
    parser.add_argument("-C", "--no-csv", action="store_true")
    parser.add_argument("-R", "--no-rntxt", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    paths = [
        os.path.join(args.input_dir, file)
        for file in os.listdir(args.input_dir)
        if file.endswith("txt")
    ]
    if args.n_output_files == 0:
        args.n_output_files = len(paths)

    ComposerWrangler().call_n_times(
        args.n_output_files,
        args.output_dir,
        paths,
        write_midi=(not args.no_midi),
        write_csv=(not args.no_csv),
        write_romantext=(not args.no_rntxt),
        _log_wo_pytest=args.log,
    )
