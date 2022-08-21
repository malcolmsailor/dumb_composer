import random
import os
import dumb_composer.melodist as mod
from dumb_composer.rhythmist import DumbRhythmist

from midi_to_notes import df_to_midi

TEST_OUT_DIR = os.path.join(
    os.path.dirname((os.path.realpath(__file__))), "test_out"
)
if not os.path.exists(TEST_OUT_DIR):
    os.makedirs(TEST_OUT_DIR)


def test_melodist():
    rhythmist = DumbRhythmist(4.0)
    melodist = mod.Melodist()
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
    # rhythm = [
    #     (0.0, 2.0),
    #     (2.0, 3.0),
    #     (3.0, 3.5),
    #     (3.5, 4.0),
    #     (4.0, 5.0),
    #     (6.0, 7.5),
    #     (7.5, 8.0),
    #     (8.0, 11.0),
    #     (11.0, 11.25),
    #     (11.25, 11.5),
    #     (11.5, 11.75),
    #     (11.75, 12.0),
    #     (12.0, 16.0),
    # ]

    for numer, denom in time_sigs:
        random.seed(42)
        rhythm = rhythmist(14)
        ts = f"{numer}/{denom}"
        rn_temp = rn_format.format(ts)
        out = melodist(rn_temp, rhythm)
        mid_path = os.path.join(
            TEST_OUT_DIR,
            f"melodist_ts={ts.replace('/', '-')}.mid",
        )
        print(f"writing {mid_path}")
        df_to_midi(out, mid_path, ts=(numer, denom))
