import os
import random

import pandas as pd
from music_df.midi_parser import df_to_midi

from dumb_composer.four_part_composer import FourPartComposer, FourPartComposerSettings
from dumb_composer.time import Meter
from tests.test_helpers import TEST_OUT_DIR, merge_dfs


def test_four_part_composer(pytestconfig, time_sig=(4, 4)):
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

    path_wo_ext = os.path.join(
        TEST_OUT_DIR,
        f"four_part_composer={ts.replace('/', '-')}",
    )
    mid_path = path_wo_ext + ".mid"
    log_path = path_wo_ext + ".log"
    logging_plugin = pytestconfig.pluginmanager.get_plugin("logging-plugin")
    logging_plugin.set_log_path(log_path)

    dfs = []
    deadend_dfs = []
    settings = FourPartComposerSettings()
    initial_seed = 42
    number_of_tries = 10
    for seed in range(initial_seed, initial_seed + number_of_tries):
        fpc = FourPartComposer(settings)
        random.seed(seed)
        out_df, these_deadends = fpc._debug(rn_txt)

        dfs.append(out_df)
        deadend_dfs.extend(these_deadends)

    out_df = merge_dfs(dfs, ts)
    deadend_df = merge_dfs(deadend_dfs, ts)

    print(f"writing {mid_path}")
    print(f"log_path = {log_path}")
    df_to_midi(out_df, mid_path, ts=(numer, denom))

    if len(deadend_df):
        deadend_path = os.path.join(
            TEST_OUT_DIR,
            f"four_part_composer={ts.replace('/', '-')}_deadends.mid",
        )
        print(f"writing {deadend_path}")
        df_to_midi(deadend_df, deadend_path, ts=(numer, denom))
