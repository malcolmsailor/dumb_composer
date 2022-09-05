import typing as t
import os
import inspect

import pandas as pd
from midi_to_notes import df_to_midi

TEST_OUT_DIR = os.path.join(
    os.path.dirname((os.path.realpath(__file__))), "test_out"
)
if not os.path.exists(TEST_OUT_DIR):
    os.makedirs(TEST_OUT_DIR)


def get_funcname():
    return inspect.currentframe().f_back.f_code.co_name


def write_df(
    out_df: pd.DataFrame, midi_basename: str, ts: t.Optional[t.Tuple[int, int]]
) -> None:
    mid_path = os.path.join(TEST_OUT_DIR, midi_basename)
    os.makedirs(os.path.dirname(mid_path), exist_ok=True)
    print(f"writing {mid_path}")
    df_to_midi(out_df, mid_path, ts=ts)
