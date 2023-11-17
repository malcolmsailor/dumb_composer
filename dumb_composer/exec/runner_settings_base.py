import os
import time
from dataclasses import dataclass
from functools import cached_property

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
    write_midi: bool = True
    write_csv: bool = True
    write_rntxt: bool = True
