import os
import random

import pandas as pd
from midi_to_notes import df_to_midi

from dumb_composer.four_part_composer import FourPartComposer, FourPartComposerSettings
from tests.test_helpers import TEST_OUT_DIR


def test_four_part_composer(time_sig=(4, 4)):
    numer, denom = time_sig
    ts = f"{numer}/{denom}"
    # TODO: (Malcolm 2023-07-18) pedal points in bass?
    rn_txt = f"""Time signature: {ts}
    m1 Bb: I
    m2 V7/IV
    m3 IV64
    Note: TODO try a pedal point here
    m4 V65
    m5 I b3 V43
    m6 I6 b3 I
    m7 F: viio64 b3 viio6/ii
    m8 ii b3 ii42
    m9 V65 b3 V7
    m10 vi b3 viio7/V
    m11 V b3 Cad64
    m12 V b3 V7
    m13 I
    """
    dfs = []
    settings = FourPartComposerSettings()
    for seed in range(42, 42 + 10):
        fpc = FourPartComposer(settings)
        random.seed(seed)
        out_df = fpc(rn_txt)

        dfs.append(out_df)

    time_adjustment = 0
    for df in dfs:
        df["onset"] += time_adjustment
        df["release"] += time_adjustment
        time_adjustment = df["release"].max() + numer
    mid_path = os.path.join(
        TEST_OUT_DIR,
        f"four_part_composer={ts.replace('/', '-')}.mid",
    )
    out_df = pd.concat(dfs, axis=0)
    print(f"writing {mid_path}")
    df_to_midi(out_df, mid_path, ts=(numer, denom))
