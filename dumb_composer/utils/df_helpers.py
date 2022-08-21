import typing as t
import pandas as pd


def sort_note_df(
    note_df: pd.DataFrame,
    pitch_sort_asc: t.Optional[bool] = False,
    track_sort_asc: t.Optional[bool] = False,
) -> pd.DataFrame:
    out = note_df.sort_values(
        by="release",
        axis=0,
        ignore_index=True,
        key=lambda x: 0 if x is None else x,
    )
    if pitch_sort_asc is not None:
        out.sort_values(
            by="pitch",
            axis=0,
            inplace=True,
            ignore_index=True,
            ascending=pitch_sort_asc,
            key=lambda x: 128 if x is None else x,
            kind="mergesort",  # default sort is not stable
        )
    if track_sort_asc is not None:
        out.sort_values(
            by="track",
            axis=0,
            inplace=True,
            ignore_index=True,
            ascending=track_sort_asc,
            kind="mergesort",  # default sort is not stable
        )
    out.sort_values(
        by="type",
        axis=0,
        inplace=True,
        ignore_index=True,
        key=lambda col: col.where(col != "note", "~~~note"),
        kind="mergesort",  # default sort is not stable
    )
    out.sort_values(
        by="onset",
        axis=0,
        inplace=True,
        ignore_index=True,
        kind="mergesort",  # default sort is not stable
    )
    return out


def merge_note_dfs(
    note_df1: pd.DataFrame, note_df2: pd.DataFrame
) -> pd.DataFrame:
    out_df = pd.concat([note_df1, note_df2], ignore_index=True)
    out_df = sort_note_df(out_df)
    return out_df
