import os
from dataclasses import dataclass
from pathlib import Path

from midi_to_note_table import df_to_midi

from dumb_composer.classes.scores import PrefabScoreWithAccompaniments
from dumb_composer.config.read_config import (
    load_config_from_yaml,
    load_config_from_yaml_basic,
)
from dumb_composer.dumb_accompanist import DumbAccompanist2, DumbAccompanistSettings
from dumb_composer.exec.runner_helpers import path_formatter
from dumb_composer.exec.runner_settings_base import RunnerSettingsBase
from dumb_composer.incremental_contrapuntist import (
    IncrementalContrapuntist,
    IncrementalContrapuntistSettings,
)
from dumb_composer.pitch_utils.types import BASS, MELODY, TENOR_AND_ALTO, Voice
from dumb_composer.utils.composer_helpers import chain_steps

DEFAULT_OUTPUT_FOLDER = os.path.expanduser(
    "~/output/run_incremental_contrapuntist_with_accomps/"
)


@dataclass
class ContrapuntistWithAccompsSettings(RunnerSettingsBase):
    # TODO: (Malcolm 2023-08-15) allow choices
    contrapuntist_voices: tuple[Voice, ...] = (BASS, MELODY, TENOR_AND_ALTO)
    get_df_keys: tuple[str, ...] = ("annotations", "accompaniments")
    output_folder: str = DEFAULT_OUTPUT_FOLDER
    timeout: int = 10


def run_contrapuntist_with_accomps(
    runner_settings_path: str | Path | None,
    contrapuntist_settings_path: str | Path | None,
    dumb_accompanist_settings_path: str | Path | None,
    rntxt_path: str | Path,
):
    settings: ContrapuntistWithAccompsSettings = load_config_from_yaml_basic(
        ContrapuntistWithAccompsSettings, runner_settings_path
    )
    contrapuntist_settings: IncrementalContrapuntistSettings = load_config_from_yaml(
        IncrementalContrapuntistSettings, contrapuntist_settings_path
    )
    dumb_accompanist_settings: DumbAccompanistSettings = load_config_from_yaml(
        DumbAccompanistSettings, dumb_accompanist_settings_path
    )
    os.makedirs(settings.output_folder, exist_ok=True)
    output_path_wo_ext = os.path.join(
        settings.output_folder, path_formatter(rntxt_path, 0, 0)
    )

    with open(rntxt_path) as inf:
        rntxt = inf.read()

    # TODO: (Malcolm 2023-11-14) should I define a class that doesn't take prefabs?
    score = PrefabScoreWithAccompaniments(rntxt)

    contrapuntist = IncrementalContrapuntist(
        score=score,
        voices=settings.contrapuntist_voices,
        settings=contrapuntist_settings,
    )
    dumb_accompanist = DumbAccompanist2(
        score=score, voices_to_accompany=[], settings=dumb_accompanist_settings
    )

    chain_steps([contrapuntist, dumb_accompanist], timeout=settings.timeout)

    out_df = score.get_df(settings.get_df_keys)

    if settings.write_midi:
        mid_path = f"{output_path_wo_ext}.mid"
        df_to_midi(out_df, mid_path, ts=score.ts.ts_str)
        print(f"Wrote {mid_path}")

    if settings.write_csv:
        csv_path = f"{output_path_wo_ext}.csv"
        out_df.to_csv(csv_path)
        print(f"Wrote {csv_path}")
