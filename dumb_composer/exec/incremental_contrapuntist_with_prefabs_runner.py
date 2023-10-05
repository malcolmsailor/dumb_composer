import os
from dataclasses import dataclass
from pathlib import Path

from dumb_composer.classes.scores import PrefabScore
from dumb_composer.config.read_config import (
    load_config_from_yaml,
    load_config_from_yaml_basic,
)
from dumb_composer.exec.runner_helpers import path_formatter
from dumb_composer.exec.runner_settings_base import RunnerSettingsBase
from dumb_composer.incremental_contrapuntist import (
    IncrementalContrapuntist,
    IncrementalContrapuntistSettings,
)
from dumb_composer.pitch_utils.types import BASS, MELODY, TENOR_AND_ALTO, Voice
from dumb_composer.prefab_applier import PrefabApplier, PrefabApplierSettings
from dumb_composer.utils.composer_helpers import chain_steps
from midi_to_note_table import df_to_midi

DEFAULT_OUTPUT_FOLDER = os.path.expanduser(
    "~/output/run_incremental_contrapuntist_with_prefabs/"
)


@dataclass
class ContrapuntistWithPrefabsSettings(RunnerSettingsBase):
    # TODO: (Malcolm 2023-08-15) allow choices
    contrapuntist_voices: tuple[Voice, ...] = (BASS, MELODY, TENOR_AND_ALTO)
    get_df_keys: tuple[str, ...] = ("structural", "annotations", "prefabs")
    output_folder: str = DEFAULT_OUTPUT_FOLDER


def run_contrapuntist_with_prefabs(
    runner_settings_path: str | Path | None,
    contrapuntist_settings_path: str | Path | None,
    prefab_applier_settings_path: str | Path | None,
    rntxt_path: str | Path,
):
    settings: ContrapuntistWithPrefabsSettings = load_config_from_yaml_basic(
        ContrapuntistWithPrefabsSettings, runner_settings_path
    )
    contrapuntist_settings: IncrementalContrapuntistSettings = load_config_from_yaml(
        IncrementalContrapuntistSettings, contrapuntist_settings_path
    )
    prefab_applier_settings: PrefabApplierSettings = load_config_from_yaml(
        PrefabApplierSettings, prefab_applier_settings_path
    )
    os.makedirs(settings.output_folder, exist_ok=True)
    output_path_wo_ext = os.path.join(
        settings.output_folder, path_formatter(rntxt_path, 0, 0)
    )

    with open(rntxt_path) as inf:
        rntxt = inf.read()

    score = PrefabScore(rntxt)

    contrapuntist = IncrementalContrapuntist(
        score=score,
        voices=settings.contrapuntist_voices,
        settings=contrapuntist_settings,
    )
    prefab_applier = PrefabApplier(score=score, settings=prefab_applier_settings)

    chain_steps([contrapuntist, prefab_applier])

    out_df = score.get_df(settings.get_df_keys)

    if settings.write_midi:
        mid_path = f"{output_path_wo_ext}.mid"
        df_to_midi(out_df, mid_path, ts=score.ts.ts_str)
        print(f"Wrote {mid_path}")

    if settings.write_csv:
        csv_path = f"{output_path_wo_ext}.csv"
        out_df.to_csv(csv_path)
        print(f"Wrote {csv_path}")
