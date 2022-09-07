import os
import random

from dumb_composer.composer_wrangler import ComposerWrangler
from dumb_composer.utils.recursion import RecursionFailed
from test_helpers import write_df
from tests.test_helpers import get_funcname, TEST_OUT_DIR

WHEN_IN_ROME_DIR = os.environ["WHEN_IN_ROME_DIR"]

N_FILES = 5


def test_composer_wrangler(pytestconfig):
    funcname = get_funcname()
    test_out_dir = os.path.join(TEST_OUT_DIR, funcname)
    cw = ComposerWrangler()
    paths = list(
        cw._get_paths(WHEN_IN_ROME_DIR, basename_startswith="analysis")
    )
    random.seed(42)
    path_formatter = (
        lambda path, i, transpose: os.path.dirname(path)
        .replace(WHEN_IN_ROME_DIR, "")
        .lstrip(os.path.sep)
        .replace(os.path.sep, "_")
        + f"_transpose={transpose}_{i+1:09d}"
    )
    cw.call_n_times(
        20,
        test_out_dir,
        paths,
        path_formatter=path_formatter,
        _pytestconfig=pytestconfig,
    )


# def test_composer_wrangler(romantext, pytestconfig):
#     funcname = get_funcname()
#     test_out_dir = os.path.join(TEST_OUT_DIR, funcname)
#     os.makedirs(test_out_dir, exist_ok=True)
#     cw = ComposerWrangler()
#     if not romantext:
#         paths = list(
#             cw._get_paths(WHEN_IN_ROME_DIR, basename_startswith="analysis")
#         )
#         random.seed(42)
#         random.shuffle(paths)
#     else:
#         paths = [romantext]
#     failures = []
#     i = 0
#     for path in paths:
#         if not romantext and "Mozart" not in path:
#             continue
#         path_wo_ext = os.path.join(
#             test_out_dir,
#             os.path.dirname(path)
#             .replace(WHEN_IN_ROME_DIR, "")
#             .lstrip(os.path.sep)
#             .replace(os.path.sep, "_"),
#         )
#         mid_path = path_wo_ext + ".mid"
#         log_path = path_wo_ext + ".log"
#         logging_plugin = pytestconfig.pluginmanager.get_plugin("logging-plugin")
#         logging_plugin.set_log_path(log_path)
#         if i == N_FILES:
#             break
#         i += 1
#         print(path)
#         # random.seed(43)
#         try:
#             out, ts = cw(path)
#         except RecursionFailed as exc:
#             failures.append((path, str(exc)))
#             continue

#         write_df(
#             out,
#             mid_path,
#             ts=ts,
#         )
#     for path, failure in failures:
#         print(f"FAILED on {path}")
#         print(failure)
