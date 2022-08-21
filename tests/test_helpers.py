import os

import pandas as pd
from midi_to_notes import df_to_midi

TEST_OUT_DIR = os.path.join(
    os.path.dirname((os.path.realpath(__file__))), "test_out"
)
if not os.path.exists(TEST_OUT_DIR):
    os.makedirs(TEST_OUT_DIR)


def write_df(out_df: pd.DataFrame, midi_basename: str) -> None:
    mid_path = os.path.join(TEST_OUT_DIR, midi_basename)
    print(f"writing {mid_path}")
    df_to_midi(out_df, mid_path)
