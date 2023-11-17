import os
import sys

from dumb_composer.exec.incremental_contrapuntist_with_prefabs_runner import (
    run_contrapuntist_with_prefabs,
)
from dumb_composer.exec.script_helpers import custom_excepthook, get_base_parser, run

sys.excepthook = custom_excepthook


def get_args():
    parser = get_base_parser()
    parser.add_argument("--contrapuntist-config", "-C", type=str, default=None)
    parser.add_argument("--prefab-config", "-P", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    run(
        f=run_contrapuntist_with_prefabs,
        cli_args=args,
        contrapuntist_settings_path=args.contrapuntist_config,
        prefab_applier_settings_path=args.prefab_config,
    )


if __name__ == "__main__":
    main()
