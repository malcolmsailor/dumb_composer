import os
import random
from dataclasses import dataclass
from pathlib import Path

from midi_to_notes import df_to_midi

from dumb_composer.config.read_config import (
    load_config_from_yaml,
    load_config_from_yaml_basic,
)
from dumb_composer.incremental_contrapuntist import (
    IncrementalContrapuntist,
    IncrementalContrapuntistSettings,
)
from dumb_composer.pitch_utils.types import (
    BASS,
    MELODY,
    TENOR_AND_ALTO,
    SettingsBase,
    Voice,
)
from dumb_composer.shared_classes import FourPartScore

# parser = argparse.ArgumentParser()
# parser.add_argument()
# args = parser.parse_args()

DEFAULT_OUTPUT_FOLDER = os.path.expanduser("~/tmp/run_incremental_contrapuntist/")


# TODO: (Malcolm 2023-08-11) add voices
def path_formatter(path, i, transpose):
    # TODO: (Malcolm 2023-08-11) set i or remove
    return (
        os.path.splitext(os.path.basename(path))[0]
        + f"_transpose={transpose}_{i+1:03d}"
    )


def get_random_transpose(random_transpose: bool):
    if not random_transpose:
        return 0
    return random.randrange(12)


@dataclass
class IncrementalComposerRunnerSettings(SettingsBase):
    write_midi: bool = True
    write_csv: bool = True
    # TODO: (Malcolm 2023-08-11) allow choices here
    voices: tuple[Voice, ...] = (BASS, MELODY, TENOR_AND_ALTO)
    get_df_keys: tuple[str, ...] = ("structural", "annotations")
    output_folder: str = DEFAULT_OUTPUT_FOLDER


def run_incremental_composer(
    runner_settings_path: str | Path | None,
    contrapuntist_settings_path: str | Path | None,
    rntxt_path: str | Path,
):
    settings: IncrementalComposerRunnerSettings = load_config_from_yaml_basic(
        IncrementalComposerRunnerSettings, runner_settings_path
    )
    contrapuntist_settings: IncrementalContrapuntistSettings = load_config_from_yaml(
        IncrementalContrapuntistSettings, contrapuntist_settings_path
    )
    # TODO: (Malcolm 2023-08-11) log settings

    # transpose_by = get_random_transpose(random_transpose)
    os.makedirs(settings.output_folder, exist_ok=True)
    output_path_wo_ext = os.path.join(
        settings.output_folder, path_formatter(rntxt_path, 0, 0)
    )

    with open(rntxt_path) as inf:
        rntxt = inf.read()

    score = FourPartScore(rntxt)

    contrapuntist = IncrementalContrapuntist(
        score=score, voices=settings.voices, settings=contrapuntist_settings
    )
    contrapuntist()

    out_df = score.get_df(settings.get_df_keys)

    if settings.write_midi:
        mid_path = f"{output_path_wo_ext}.mid"
        df_to_midi(out_df, mid_path, ts=score.ts.ts_str)

    if settings.write_csv:
        csv_path = f"{output_path_wo_ext}.csv"
        out_df.to_csv(csv_path)
