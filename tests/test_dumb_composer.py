import random
import os

from dumb_composer.dumb_composer import PrefabComposer, PrefabComposerSettings
from dumb_composer.utils.recursion import RecursionFailed
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
    time_sigs = [(3, 4), (4, 4)]
    for numer, denom in time_sigs:
        for prefab_voice in ("tenor", "bass", "soprano"):
            random.seed(42)
            for i in range(1):
                ts = f"{numer}/{denom}"
                rn_temp = rn_format.format(ts)
                settings = PrefabComposerSettings(prefab_voice=prefab_voice)
                pfc = PrefabComposer(settings)
                out_df = pfc(rn_temp)
                write_df(
                    out_df,
                    f"prefab_composer{i + 1}_ts={ts.replace('/', '-')}_"
                    f"prefab_voice={prefab_voice}.mid",
                    ts=(numer, denom),
                )
