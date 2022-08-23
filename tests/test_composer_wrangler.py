import os
import random

from dumb_composer.composer_wrangler import ComposerWrangler
from dumb_composer.utils.recursion import RecursionFailed
from test_helpers import write_df

WHEN_IN_ROME_DIR = os.environ["WHEN_IN_ROME_DIR"]

N_FILES = 5


def test_composer_wrangler(romantext):
    cw = ComposerWrangler()
    if not romantext:
        paths = cw._get_paths(WHEN_IN_ROME_DIR, basename_startswith="analysis")
    else:
        paths = [
            romantext,
        ]
    failures = []
    i = 0
    for path in paths:
        if not romantext and "Haydn" not in path:
            continue
        i += 1
        if i == N_FILES:
            break
        random.seed(43)
        try:
            out, ts = cw(path)
        except RecursionFailed as exc:
            failures.append((path, str(exc)))
            continue
        midi_basename = os.path.join(
            "wrangler",
            os.path.dirname(path)
            .replace(WHEN_IN_ROME_DIR, "")
            .lstrip(os.path.sep)
            .replace(os.path.sep, "_")
            + ".mid",
        )
        write_df(
            out,
            midi_basename,
            ts=ts,
        )
    for path, failure in failures:
        print(f"FAILED on {path}")
        print(failure)
