import inspect
import os
import random

from dumb_composer.dumb_accompanist import (
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


def test_dumb_accompanist(quick):
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
    rn_format = """Time signature: {}
    m1 Bb: I
    m2 F: ii
    m3 I64
    m4 V7
    m5 I
    m6 ii6
    m7 V7
    m8 I
    m9 I64
    m10 V7
    m11 I53
    m12 V43
    m13 V42
    m14 I6
    """
    time_sigs = [(4, 4), (3, 4)]
    for numer, denom in time_sigs:
        ts = f"{numer}/{denom}"
        rn_temp = rn_format.format(ts)
        random.seed(42)
        for pattern in PatternMaker._all_patterns:
            settings = DumbAccompanistSettings(
                pattern=pattern, text_annotations="chord"
            )
            dc = DumbAccompanist(settings)
            out_df = dc(rn_temp)
            funcname = get_funcname()
            mid_path = os.path.join(
                TEST_OUT_DIR,
                funcname + f"_ts={ts.replace('/', '-')}_pattern={pattern}.mid",
            )
            print(f"writing {mid_path}")
            df_to_midi(out_df, mid_path, ts=(numer, denom))
            if quick:
                return


if __name__ == "__main__":
    test_dumb_accompanist()
