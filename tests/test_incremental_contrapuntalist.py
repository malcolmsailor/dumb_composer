import os
import random

import pandas as pd
import pytest
from midi_to_notes import df_to_midi

from dumb_composer.incremental_contrapuntist import (
    IncrementalContrapuntist,
    IncrementalContrapuntistSettings,
)
from dumb_composer.pitch_utils.types import (
    ALTO,
    BASS,
    MELODY,
    TENOR,
    TENOR_AND_ALTO,
    voice_enum_to_string,
)
from dumb_composer.shared_classes import FourPartScore
from tests.test_helpers import TEST_OUT_DIR, merge_dfs


@pytest.mark.parametrize(
    "voices",
    (
        (BASS,),
        (MELODY,),
        (BASS, MELODY),
        (MELODY, BASS),
        (BASS, MELODY, TENOR),
        (MELODY, BASS, ALTO),
        (BASS, MELODY, TENOR_AND_ALTO),
    ),
)
def test_incremental_contrapuntalist(pytestconfig, voices, time_sig=(4, 4)):
    numer, denom = time_sig
    ts = f"{numer}/{denom}"

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
    voice_strs = "-".join([voice_enum_to_string[v] for v in voices])
    path_wo_ext = os.path.join(
        TEST_OUT_DIR,
        f"incremental_contapuntalist={ts.replace('/', '-')}_{voice_strs}",
    )
    mid_path = path_wo_ext + ".mid"
    log_path = path_wo_ext + ".log"
    logging_plugin = pytestconfig.pluginmanager.get_plugin("logging-plugin")
    logging_plugin.set_log_path(log_path)

    dfs = []
    deadend_dfs = []
    settings = IncrementalContrapuntistSettings()
    initial_seed = 42
    number_of_tries = 10
    for seed in range(initial_seed, initial_seed + number_of_tries):
        score = FourPartScore(rn_txt)
        contrapuntist = IncrementalContrapuntist(
            score=score, voices=voices, settings=settings
        )
        random.seed(seed)
        result = contrapuntist()
        out_df = result._score.structural_df()

        dfs.append(out_df)
        # deadend_dfs.extend(these_deadends)

    out_df = merge_dfs(dfs, ts)
    # deadend_df = merge_dfs(deadend_dfs, ts)

    print(f"writing {mid_path}")
    print(f"log_path = {log_path}")
    df_to_midi(out_df, mid_path, ts=(numer, denom))

    # if len(deadend_df):
    #     deadend_path = os.path.join(
    #         TEST_OUT_DIR,
    #         f"four_part_composer={ts.replace('/', '-')}_deadends.mid",
    #     )
    #     print(f"writing {deadend_path}")
    #     df_to_midi(deadend_df, deadend_path, ts=(numer, denom))
