"""This file is not meant to be run on its own but should be called by 
test_cacher_across_runs() in test_cache_lib.py."""

import argparse
from dumb_composer.utils.cache_lib import cacher

SCRAP = {"ran": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_base")
    parser.add_argument("path")
    args = parser.parse_args()

    @cacher(cache_base=args.cache_base)
    def f(path):
        with open(path) as inf:
            pass
        SCRAP["ran"] = True

    f(args.path)
    print(SCRAP["ran"])
