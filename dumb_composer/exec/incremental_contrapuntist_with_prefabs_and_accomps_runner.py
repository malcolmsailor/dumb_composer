import os
import random
from dataclasses import dataclass, field
from functools import cached_property
from itertools import accumulate
from pathlib import Path

from music_df.midi_parser import df_to_midi

from dumb_composer.classes.scores import PrefabScore, PrefabScoreWithAccompaniments
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
from dumb_composer.pitch_utils.types import Voice, voice_enum_to_string
from dumb_composer.prefab_applier import PrefabApplier, PrefabApplierSettings
from dumb_composer.utils.composer_helpers import chain_steps


@dataclass
class ContrapuntistWithPrefabsSettings(RunnerSettingsBase):
    prefab_voices_and_weights: dict[float, str | tuple[str, ...]] = field(
        default_factory=lambda: {
            4.0: "melody",
            2.0: "bass",
            1.0: ("melody", "bass"),
            1.0: ("melody", "alto"),
            1.0: ("tenor", "bass"),
        }
    )
    contrapuntist_voices: tuple[str, ...] = ("bass", "melody", "tenor_and_alto")
    get_df_keys: tuple[str, ...] = ("annotations", "prefabs", "accompaniments")
    # get_df_keys: tuple[str, ...] = ("structural", "annotations", "prefabs")
    timeout: int = 10

    @cached_property
    def prefab_weights(self):
        return list(accumulate([w for w in self.prefab_voices_and_weights.keys()]))

    @cached_property
    def prefab_voice_choices(self):
        return [v for v in self.prefab_voices_and_weights.values()]

    def choose_prefab_voice(self) -> tuple[Voice, ...]:
        out = random.choices(
            self.prefab_voice_choices, cum_weights=self.prefab_weights, k=1
        )[0]
        if isinstance(out, str):
            out = (out,)
        return voice_strs_to_enums(out)


def run_contrapuntist_with_prefabs_and_accomps(
    runner_settings_path: str | Path | None,
    contrapuntist_settings_path: str | Path | None,
    prefab_applier_settings_path: str | Path | None,
    dumb_accompanist_settings_path: str | Path | None,
    rntxt_path: str | Path,
    output_folder: str | Path,
    basename_prefix: str | None,
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
    prefab_voice_or_voices = settings.choose_prefab_voice()
    prefab_applier_settings.prefab_voices = [
        voice_enum_to_string[v] for v in prefab_voice_or_voices
    ]
    dumb_accompanist_settings: DumbAccompanistSettings = load_config_from_yaml(
        DumbAccompanistSettings, dumb_accompanist_settings_path
    )

    with open(rntxt_path) as inf:
        rntxt = inf.read()

    score = PrefabScoreWithAccompaniments(rntxt)

    contrapuntist = IncrementalContrapuntist(
        score=score,
        voices=voice_strs_to_enums(settings.contrapuntist_voices),
        settings=contrapuntist_settings,
    )
    prefab_applier = PrefabApplier(score=score, settings=prefab_applier_settings)
    dumb_accompanist = DumbAccompanist2(
        score=score,
        voices_to_accompany=prefab_voice_or_voices,
        settings=dumb_accompanist_settings,
    )

    chain_steps(
        [contrapuntist, prefab_applier, dumb_accompanist], timeout=settings.timeout
    )

    if basename_prefix is None:
        basename_prefix = "prefabs_and_accomps"
    else:
        basename_prefix = f"{basename_prefix}_prefabs_and_accomps"
    write_output(
        output_folder,
        rntxt_path,
        score,
        settings,
        basename_prefix=basename_prefix,
    )
    # os.makedirs(output_folder, exist_ok=True)
    # output_path_wo_ext = os.path.join(output_folder, path_formatter(rntxt_path, 0, 0))
    # out_df = score.get_df(settings.get_df_keys)

    # if settings.write_midi:
    #     mid_path = f"{output_path_wo_ext}.mid"
    #     df_to_midi(out_df, mid_path, ts=score.ts.ts_str)
    #     print(f"Wrote {mid_path}")

    # if settings.write_csv:
    #     csv_path = f"{output_path_wo_ext}.csv"
    #     out_df.to_csv(csv_path)
    #     print(f"Wrote {csv_path}")

    # if settings.write_rntxt:
    #     output_rntxt_path = f"{output_path_wo_ext}.txt"
    #     with open(output_rntxt_path, "w") as outf:
    #         outf.write(rntxt)
