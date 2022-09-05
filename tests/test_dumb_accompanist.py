import inspect
import os
import random

from dumb_composer.dumb_accompanist import (
    AccompAnnots,
    DumbAccompanist,
    DumbAccompanistSettings,
)
from dumb_composer.patterns import PatternMaker
from midi_to_notes import df_to_midi


TEST_OUT_DIR = os.path.join(
    os.path.dirname((os.path.realpath(__file__))), "test_out"
)
if not os.path.exists(TEST_OUT_DIR):
    os.makedirs(TEST_OUT_DIR)


def get_funcname():
    return inspect.currentframe().f_back.f_code.co_name


def test_dumb_accompanist(quick, pytestconfig):
    # rn_format = """Time signature: {}
    #     m1 C: I
    #     m2 V65
    #     m3 vi
    #     m4 V
    #     m5 I
    #     m6 bII6
    #     m7 V
    #     m8 bVI
    #     """
    test_cases = [
        (
            (4, 4),
            """Time signature: 4/4
    m1 Bb: I
    m2 F: ii b2 ii
    m3 I64
    m4 V7
    m5 I b3 I6
    m6 ii6 b4 V65/V
    m7 V7
    m8 I b2 vi b3 ii b4 Ger65
    m9 I64
    m10 V7 b4.5 viio7/vi
    m11 vi b1.5 V b3 I
    m12 V43
    m13 V42
    m14 I6
    """,
        ),
        (
            (3, 4),
            """Time signature: 3/4
    m1 Bb: I
    m2 F: ii b2 ii
    m3 I64
    m4 V7
    m5 I b3 I6
    m6 ii6 b3.5 V65/V
    m7 V7
    m8 I b2 vi b2.5 ii b3 Ger65
    m9 I64
    m10 V7 b3.75 viio7/vi
    m11 vi b1.5 V b3 I
    m12 V43
    m13 V42
    m14 I6
    """,
        ),
    ]
    funcname = get_funcname()
    test_out_dir = os.path.join(TEST_OUT_DIR, funcname)
    os.makedirs(test_out_dir, exist_ok=True)
    for (numer, denom), rntxt in test_cases:
        ts = f"{numer}/{denom}"
        random.seed(42)
        for pattern in PatternMaker._all_patterns:
            path_wo_ext = os.path.join(
                test_out_dir, f"ts={ts.replace('/', '-')}_pattern={pattern}"
            )
            mid_path = path_wo_ext + ".mid"
            log_path = path_wo_ext + ".log"
            logging_plugin = pytestconfig.pluginmanager.get_plugin(
                "logging-plugin"
            )
            logging_plugin.set_log_path(log_path)
            settings = DumbAccompanistSettings(
                pattern=pattern, accompaniment_annotations=AccompAnnots.ALL
            )
            dc = DumbAccompanist(settings)
            out_df = dc(rntxt)
            print(f"writing {mid_path}")
            df_to_midi(out_df, mid_path, ts=(numer, denom))
            if quick:
                return


if __name__ == "__main__":
    test_dumb_accompanist()
