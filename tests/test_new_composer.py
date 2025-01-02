import os
import random

import pandas as pd
import pytest
from music_df.midi_parser import df_to_midi

from dumb_composer.new_composer import NewComposer, NewComposerSettings
from tests.test_helpers import TEST_OUT_DIR, get_funcname, merge_dfs, write_df


@pytest.mark.skip(reason="WIP")
@pytest.mark.parametrize(
    "time_sig",
    [
        (4, 4),
        # (3, 4),
    ],
)
def test_new_composer(quick, pytestconfig, time_sig):
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
    funcname = get_funcname()
    test_out_dir = os.path.join(TEST_OUT_DIR, funcname)
    os.makedirs(test_out_dir, exist_ok=True)

    path_wo_ext = os.path.join(test_out_dir, f"ts={ts.replace('/', '-')}")
    mid_path = path_wo_ext + ".mid"
    log_path = path_wo_ext + ".log"
    logging_plugin = pytestconfig.pluginmanager.get_plugin("logging-plugin")
    logging_plugin.set_log_path(log_path)

    dfs = []
    initial_seed = 48
    number_of_tries = 1
    for seed in range(initial_seed, initial_seed + number_of_tries):
        random.seed(seed)
        settings = NewComposerSettings()
        pfc = NewComposer(settings)
        result = pfc(rn_txt)
        # TODO: (Malcolm 2023-08-09) get df in better way
        accompaniment_df = result.accompaniments_as_df
        melody_df = result.as_df("structural_soprano")
        out_df = pd.concat([accompaniment_df, melody_df]).sort_values(by="onset")
        # out_df = result.get_df()
        dfs.append(out_df)

    out_df = merge_dfs(dfs, ts)

    write_df(
        out_df,
        mid_path,
        ts=(numer, denom),
    )
    print(f"log_path = {log_path}")

    if quick:
        raise NotImplementedError()
