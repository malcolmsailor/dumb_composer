import os
import random
from dataclasses import dataclass, field
from functools import cached_property
from itertools import accumulate
from pathlib import Path

from midi_to_note_table import df_to_midi

from dumb_composer.classes.scores import PrefabScore
from dumb_composer.config.read_config import (
    load_config_from_yaml,
    load_config_from_yaml_basic,
)
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
from dumb_composer.prefab_applier import PrefabApplier, PrefabApplierSettings
from dumb_composer.utils.composer_helpers import chain_steps


# TODO: (Malcolm 2023-10-18) I'd like to write "structural" separately from "prefabs"
@dataclass
class ContrapuntistWithPrefabsSettings(RunnerSettingsBase):
    # contrapuntist_voices: tuple[str, ...] = ("bass", "melody", "tenor_and_alto")
    contrapuntist_voices_and_weights: dict[float, tuple[str, ...]] = field(
        default_factory=lambda: {
            8.0: ("bass", "melody", "tenor_and_alto"),
            2.0: ("bass", "melody", "alto"),
            2.0: ("bass", "melody", "tenor"),
        }
    )
    get_df_keys: tuple[str, ...] = ("annotations", "prefabs")
    # get_df_keys: tuple[str, ...] = ("structural", "annotations", "prefabs")
    timeout: int = 10

    @cached_property
    def contrapuntist_weights(self):
        return list(
            accumulate([w for w in self.contrapuntist_voices_and_weights.keys()])
        )

    @cached_property
    def contrapuntist_voice_choices(self):
        return [v for v in self.contrapuntist_voices_and_weights.values()]

    def choose_contrapuntist_voice(self) -> tuple[str, ...]:
        out = random.choices(
            self.contrapuntist_voice_choices,
            cum_weights=self.contrapuntist_weights,
            k=1,
        )[0]
        return out


def run_contrapuntist_with_prefabs(
    runner_settings_path: str | Path | None,
    contrapuntist_settings_path: str | Path | None,
    prefab_applier_settings_path: str | Path | None,
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
    # os.makedirs(output_folder, exist_ok=True)
    # output_path_wo_ext = os.path.join(output_folder, path_formatter(rntxt_path, 0, 0))

    with open(rntxt_path) as inf:
        rntxt = inf.read()

    score = PrefabScore(rntxt)

    contrapuntist_voices = settings.choose_contrapuntist_voice()
    individual_voices = contrapuntist_voices
    for voice_pair in ("tenor_and_alto", "alto_and_tenor"):
        if voice_pair in individual_voices:
            i = individual_voices.index(voice_pair)
            individual_voices = (
                individual_voices[:i] + individual_voices[i + 1 :] + ("tenor", "alto")
            )

    prefab_applier_settings.prefab_voices = tuple(
        v
        for v in prefab_applier_settings.prefab_voices
        if (
            (v in individual_voices)
            # (Malcolm 2023-11-15) it's very annoying to need to do this
            or (v == "melody" and "soprano" in individual_voices)
            or (v == "soprano" and "melody" in individual_voices)
        )
    )

    contrapuntist = IncrementalContrapuntist(
        score=score,
        voices=voice_strs_to_enums(contrapuntist_voices),
        settings=contrapuntist_settings,
    )
    prefab_applier = PrefabApplier(score=score, settings=prefab_applier_settings)

    chain_steps([contrapuntist, prefab_applier], timeout=settings.timeout)

    if basename_prefix is None:
        basename_prefix = "prefabs"
    else:
        basename_prefix = f"{basename_prefix}_prefabs"
    write_output(
        output_folder, rntxt_path, score, settings, basename_prefix=basename_prefix
    )
