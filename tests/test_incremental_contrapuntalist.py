import itertools
import os
import random
from copy import deepcopy

import pandas as pd
import pytest
from music_df.midi_parser import df_to_midi

from dumb_composer.classes.scores import FourPartScore
from dumb_composer.incremental_contrapuntist import (
    IncrementalContrapuntist,
    IncrementalContrapuntistSettings,
    build_incrementally,
)
from dumb_composer.pitch_utils.types import (
    ALTO,
    BASS,
    MELODY,
    TENOR,
    TENOR_AND_ALTO,
    Voice,
    voice_enum_to_string,
)
from dumb_composer.utils.iterables import slice_into_sublists
from dumb_composer.utils.recursion import DeadEnd, append_attempt
from tests.test_helpers import TEST_OUT_DIR, merge_dfs

VALID_VOICE_ORDERS = [
    # TODO: (Malcolm 2023-08-16) currently, BASS-only or MELODY-only doesn't
    #   work because the _structural defaultdict will have an entry
    #   for the voice that isn't being made (e.g., MELODY, if we are doing BASS
    #   only) and this will cause the worker to return ready=False infinitely
    #   since the MELODY entry in the defaultdict will always be shorter
    #   than the BASS entry
    # (BASS,),
    # (MELODY,),
    (BASS, MELODY),
    (MELODY, BASS),
    (BASS, MELODY, TENOR),
    (MELODY, BASS, ALTO),
    (BASS, MELODY, TENOR_AND_ALTO),
]


@pytest.fixture(params=VALID_VOICE_ORDERS)
def voices(request) -> tuple[Voice, ...]:
    return request.param


VALID_VOICE_DIVISIONS = []
for voice_order in VALID_VOICE_ORDERS:
    VALID_VOICE_DIVISIONS.extend(list(slice_into_sublists(voice_order)))


@pytest.fixture(params=VALID_VOICE_DIVISIONS)
def voice_tups(request):
    return request.param


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
        out_df = score.get_df(["structural", "annotations"])

        dfs.append(out_df)
        # deadend_dfs.extend(these_deadends)

    out_df = merge_dfs(dfs, ts)
    # deadend_df = merge_dfs(deadend_dfs, ts)

    print(f"writing {mid_path}")
    print(f"log_path = {log_path}")
    df_to_midi(out_df, mid_path, ts=(numer, denom))


def test_incremental_contrapuntalist_with_prior_voices(
    pytestconfig, voice_tups: tuple[tuple[Voice]]
):
    rn_txt = """Time signature: 4/4
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
    voice_tup_str = "+".join(
        [
            "-".join([voice_enum_to_string[v] for v in voice_tup])
            for voice_tup in voice_tups
        ]
    )
    path_wo_ext = os.path.join(
        TEST_OUT_DIR,
        f"incremental_contapuntalist_with_prior_voices={voice_tup_str}",
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
        score2 = deepcopy(score)
        random.seed(seed)
        build_incrementally(voice_tups, score, settings)
        out_df = score.get_df(["structural", "annotations"])

        dfs.append(out_df)
        # deadend_dfs.extend(these_deadends)

    out_df = merge_dfs(dfs, "4/4")
    # deadend_df = merge_dfs(deadend_dfs, ts)

    print(f"writing {mid_path}")
    print(f"log_path = {log_path}")
    df_to_midi(out_df, mid_path, ts=(4, 4))
