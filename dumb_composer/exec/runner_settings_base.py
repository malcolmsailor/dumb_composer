import os
from dataclasses import dataclass

from dumb_composer.pitch_utils.types import (
    BASS,
    MELODY,
    TENOR_AND_ALTO,
    SettingsBase,
    Voice,
)


@dataclass
class RunnerSettingsBase(SettingsBase):
    get_df_keys: tuple[str, ...]
    output_folder: str
    write_midi: bool = True
    write_csv: bool = True
