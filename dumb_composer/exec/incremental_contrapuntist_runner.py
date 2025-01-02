import os
import random
from dataclasses import dataclass, field
from functools import cached_property
from itertools import accumulate
from pathlib import Path

from music_df.midi_parser import df_to_midi

from dumb_composer.classes.scores import FourPartScore
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
from dumb_composer.utils.composer_helpers import chain_steps

# parser = argparse.ArgumentParser()
# parser.add_argument()
# args = parser.parse_args()


def get_random_transpose(random_transpose: bool):
    if not random_transpose:
        return 0
    return random.randrange(12)


@dataclass
class IncrementalComposerRunnerSettings(RunnerSettingsBase):
    contrapuntist_voices_and_weights: dict[float, tuple[str, ...]] = field(
        default_factory=lambda: {
            8.0: ("bass", "melody", "tenor_and_alto"),
            2.0: ("bass", "melody", "alto"),
            2.0: ("bass", "melody", "tenor"),
        }
    )
    get_df_keys: tuple[str, ...] = ("structural", "annotations")
    timeout: int = 10

    @cached_property
    def contrapuntist_weights(self):
        return list(
            accumulate([w for w in self.contrapuntist_voices_and_weights.keys()])
        )

    @cached_property
    def contrapuntist_voice_choices(self):
        return [v for v in self.contrapuntist_voices_and_weights.values()]

    def choose_contrapuntist_voice(self) -> tuple[Voice, ...]:
        out = random.choices(
            self.contrapuntist_voice_choices,
            cum_weights=self.contrapuntist_weights,
            k=1,
        )[0]
        return voice_strs_to_enums(out)


def run_incremental_composer(
    runner_settings_path: str | Path | None,
    contrapuntist_settings_path: str | Path | None,
    rntxt_path: str | Path,
    output_folder: str | Path,
    basename_prefix: str | None,
):
    settings: IncrementalComposerRunnerSettings = load_config_from_yaml_basic(
        IncrementalComposerRunnerSettings, runner_settings_path
    )
    contrapuntist_settings: IncrementalContrapuntistSettings = load_config_from_yaml(
        IncrementalContrapuntistSettings, contrapuntist_settings_path
    )
    # TODO: (Malcolm 2023-08-11) log settings

    with open(rntxt_path) as inf:
        rntxt = inf.read()

    score = FourPartScore(rntxt)

    contrapuntist = IncrementalContrapuntist(
        score=score,
        voices=settings.choose_contrapuntist_voice(),
        settings=contrapuntist_settings,
    )
    chain_steps([contrapuntist], timeout=settings.timeout)
    if basename_prefix is None:
        basename_prefix = "structural"
    else:
        basename_prefix = f"{basename_prefix}_structural"
    write_output(
        output_folder, rntxt_path, score, settings, basename_prefix=basename_prefix
    )
