import inspect
import os
import typing as t
from copy import deepcopy

import pandas as pd
from music_df.midi_parser import df_to_midi

from dumb_composer.time import Meter

TEST_OUT_DIR = os.path.join(os.path.dirname((os.path.realpath(__file__))), "test_out")
if not os.path.exists(TEST_OUT_DIR):
    os.makedirs(TEST_OUT_DIR)


def get_funcname():
    return inspect.currentframe().f_back.f_code.co_name  # type:ignore


def merge_dfs(dfs: t.Iterable[pd.DataFrame], ts_str: str) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    dfs = deepcopy(dfs)
    meter = Meter(ts_str)
    time_adjustment = 0
    for df in dfs:
        df["onset"] += time_adjustment
        df["release"] += time_adjustment
        time_adjustment = (df["release"].max() // meter.bar_dur + 1) * meter.bar_dur
    return pd.concat(dfs, axis=0)


def write_df(
    out_df: pd.DataFrame, midi_basename: str, ts: t.Optional[t.Tuple[int, int]]
) -> None:
    mid_path = os.path.join(TEST_OUT_DIR, midi_basename)
    os.makedirs(os.path.dirname(mid_path), exist_ok=True)
    print(f"writing {mid_path}")
    df_to_midi(out_df, mid_path, ts=ts)
