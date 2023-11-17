import os
from dataclasses import dataclass, field
from pathlib import Path

from midi_to_note_table import df_to_midi

from dumb_composer.classes.scores import PrefabScoreWithAccompaniments
from dumb_composer.config.read_config import (
    load_config_from_yaml,
    load_config_from_yaml_basic,
)
from dumb_composer.dumb_accompanist import DumbAccompanist2, DumbAccompanistSettings
from dumb_composer.exec.runner_helpers import (
    path_formatter,
    voice_strs_to_enums,
    write_output,
)
from dumb_composer.exec.runner_settings_base import RunnerSettingsBase
from dumb_composer.incremental_contrapuntist import (
    IncrementalContrapuntist,
    IncrementalContrapuntistSettings,
)
from dumb_composer.pitch_utils.types import BASS, MELODY, TENOR_AND_ALTO, Voice
from dumb_composer.utils.composer_helpers import chain_steps


@dataclass
class ContrapuntistWithAccompsSettings(RunnerSettingsBase):
    contrapuntist_voices: tuple[str, ...] = ("bass", "melody", "tenor_and_alto")
    get_df_keys: tuple[str, ...] = ("annotations", "accompaniments")
    timeout: int = 10


def run_contrapuntist_with_accomps(
    runner_settings_path: str | Path | None,
    contrapuntist_settings_path: str | Path | None,
    dumb_accompanist_settings_path: str | Path | None,
    rntxt_path: str | Path,
    output_folder: str | Path,
    basename_prefix: str | None,
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
    # os.makedirs(output_folder, exist_ok=True)
    # output_path_wo_ext = os.path.join(output_folder, path_formatter(rntxt_path, 0, 0))

    with open(rntxt_path) as inf:
        rntxt = inf.read()

    # (Malcolm 2023-11-14) we use the PrefabScoreWithAccompaniments that also
    #   allows for prefabs, although we are not using them
    score = PrefabScoreWithAccompaniments(rntxt)

    contrapuntist = IncrementalContrapuntist(
        score=score,
        voices=voice_strs_to_enums(settings.contrapuntist_voices),
        settings=contrapuntist_settings,
    )
    dumb_accompanist = DumbAccompanist2(
        score=score, voices_to_accompany=[], settings=dumb_accompanist_settings
    )

    chain_steps([contrapuntist, dumb_accompanist], timeout=settings.timeout)

    if basename_prefix is None:
        basename_prefix = "accomps"
    else:
        basename_prefix = f"{basename_prefix}_accomps"
    write_output(
        output_folder, rntxt_path, score, settings, basename_prefix=basename_prefix
    )
    # out_df = score.get_df(settings.get_df_keys)

    # if settings.write_midi:
    #     mid_path = f"{output_path_wo_ext}.mid"
    #     df_to_midi(out_df, mid_path, ts=score.ts.ts_str)
    #     print(f"Wrote {mid_path}")

    # if settings.write_csv:
    #     csv_path = f"{output_path_wo_ext}.csv"
    #     out_df.to_csv(csv_path)
    #     print(f"Wrote {csv_path}")
