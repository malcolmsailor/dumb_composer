import random
import os

from dumb_composer.dumb_composer import PrefabComposer
from test_helpers import write_df

TEST_OUT_DIR = os.path.join(
    os.path.dirname((os.path.realpath(__file__))), "test_out"
)
if not os.path.exists(TEST_OUT_DIR):
    os.makedirs(TEST_OUT_DIR)


def test_prefab_composer():
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
    time_sigs = [(4, 4)]
    for numer, denom in time_sigs:
        random.seed(42)
        ts = f"{numer}/{denom}"
        rn_temp = rn_format.format(ts)
        pfc = PrefabComposer()
        out_df = pfc(rn_temp)
        write_df(out_df, "prefab_composer.mid")
        print(pfc.get_missing_prefab_str())
