import os
import shutil
from pathlib import Path
from typing import Iterable

from midi_to_note_table import df_to_midi

from dumb_composer.classes.scores import _ScoreBase
from dumb_composer.exec.runner_settings_base import RunnerSettingsBase
from dumb_composer.pitch_utils.types import Voice, voice_string_to_enum


def path_formatter(
    path: str | Path,
    i: int | None = None,
    transpose: int | None = None,
    prefix: str | None = None,
) -> str:
    out = os.path.splitext(os.path.basename(path))[0]
    if prefix is not None:
        out = f"{prefix}_{out}"
    if transpose is not None:
        out += f"_transpose={transpose}"
    if i is not None:
        out += f"_{i+1:03d}"
    return out


def voice_strs_to_enums(voice_strs: Iterable[str]) -> tuple[Voice, ...]:
    return tuple(voice_string_to_enum[v] for v in voice_strs)


def write_output(
    output_folder: str | Path,
    rntxt_path: str | Path,
    score: _ScoreBase,
    settings: RunnerSettingsBase,
    basename_prefix: str | None = None,
):
    os.makedirs(output_folder, exist_ok=True)
    output_path_wo_ext = os.path.join(
        output_folder, path_formatter(rntxt_path, prefix=basename_prefix)
    )
    out_df = score.get_df(settings.get_df_keys)

    if settings.write_midi:
        mid_path = f"{output_path_wo_ext}.mid"
        df_to_midi(out_df, mid_path, ts=score.ts.ts_str)
        print(f"Wrote {mid_path}")

    if settings.write_csv:
        csv_path = f"{output_path_wo_ext}.csv"
        out_df.to_csv(csv_path)
        print(f"Wrote {csv_path}")

    if settings.write_rntxt:
        output_rntxt_path = f"{output_path_wo_ext}.txt"
        # copy rntxt_path to output_rntxt_path
        shutil.copyfile(rntxt_path, output_rntxt_path)
