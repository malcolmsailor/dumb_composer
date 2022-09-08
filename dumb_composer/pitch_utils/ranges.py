from dataclasses import dataclass
from functools import cached_property
import logging
import random
import sys
import typing as t

import numpy as np
from scipy.stats import truncnorm

from dumb_composer.constants import unspeller
from dumb_composer.utils.math_ import softmax_from_slope


@dataclass
class Ranger:
    min_nadir: t.Union[str, int] = "C2"
    max_nadir: t.Union[str, int] = "C3"
    min_apogee: t.Union[str, int] = "G5"
    max_apogee: t.Union[str, int] = "C7"
    min_accomp_nadir: t.Union[str, int] = "Bb2"
    max_accomp_apogee: t.Union[str, int] = "G5"
    min_mel_ambitus: int = 15
    max_mel_ambitus: int = 24
    min_bass_ambitus: int = 15
    max_bass_ambitus: int = 24
    tenor_mel_min_nadir: t.Union[str, int] = "Bb2"
    tenor_mel_max_apogee: t.Union[str, int] = "C5"

    slope_scale: float = 0.1

    def __post_init__(self):
        for attr in (
            "min_nadir",
            "max_nadir",
            "min_apogee",
            "max_apogee",
            "min_accomp_nadir",
            "max_accomp_apogee",
            "tenor_mel_min_nadir",
            "tenor_mel_max_apogee",
        ):
            val = getattr(self, attr)
            if isinstance(val, str):
                setattr(self, attr, unspeller(val))
        self._max_ambitus = self.max_apogee - self.min_nadir
        self._min_ambitus = self.min_apogee - self.max_nadir
        self._mel_ambitus_delta = self.max_mel_ambitus - self.min_mel_ambitus
        self._bass_ambitus_delta = self.max_bass_ambitus - self.min_bass_ambitus

    @cached_property
    def _truncnorm(self):
        out = truncnorm(-1, 1)
        out.random_state = np.random.default_rng(random.randrange(10000))
        return out

    def within(self, top, bottom, dist="normal"):
        if dist != "normal":
            raise NotImplementedError
        loc = (top + bottom) / 2
        scale = (top - bottom) / 2
        return int(round(self._truncnorm.rvs() * scale + loc))

    def _get_slope(
        self, min_val, max_val, actual_val, scale=0.1, sign: int = 1
    ):
        prop = (actual_val - min_val) / (max_val - min_val)
        centered = prop - 0.5
        scaled = centered * 2 * self.slope_scale
        return scaled * sign

    def _choose_ambitus(self, ambitus_delta, slope, ambitus_min):
        weights = softmax_from_slope(ambitus_delta, slope)
        ambitus = (
            random.choices(range(ambitus_delta + 1), weights=weights)[0]
            + ambitus_min
        )
        return ambitus

    def _soprano_call(
        self, nadir, apogee, slope
    ) -> t.Dict[str, t.Tuple[int, int]]:
        # start by choosing the extremes of the range ("nadir" and "apogee")
        #   then:
        #       melody ambitus tends to get smaller as total ambitus gets smaller
        #       bass ambitus tends to get smaller as total ambitus gets smaller

        mel_ambitus = self._choose_ambitus(
            self._mel_ambitus_delta, slope, self.min_mel_ambitus
        )
        mel_range = (apogee - mel_ambitus, apogee)

        bass_ambitus = self._choose_ambitus(
            self._bass_ambitus_delta, slope, self.min_bass_ambitus
        )
        bass_range = (nadir, nadir + bass_ambitus)
        accomp_nadir = max(
            self.min_accomp_nadir, int(round(sum(bass_range) / 2))
        )
        accomp_apogee = min(
            self.max_accomp_apogee, int(round(sum(mel_range) / 2))
        )
        accomp_range = (accomp_nadir, accomp_apogee)
        return {
            "mel_range": mel_range,
            "bass_range": bass_range,
            "accomp_range": accomp_range,
        }

    def _bass_call(
        self, nadir, apogee, slope
    ) -> t.Dict[str, t.Tuple[int, int]]:
        bass_ambitus = self._choose_ambitus(
            self._bass_ambitus_delta, slope, self.min_bass_ambitus
        )
        bass_range = (nadir, nadir + bass_ambitus)
        accomp_nadir = max(
            self.min_accomp_nadir, int(round(sum(bass_range) / 2))
        )
        accomp_range = (accomp_nadir, apogee)
        return {
            "mel_range": bass_range,
            "bass_range": bass_range,
            "accomp_range": accomp_range,
        }

    def _tenor_call(
        self, nadir, apogee, slope
    ) -> t.Dict[str, t.Tuple[int, int]]:
        bass_ambitus = self._choose_ambitus(
            self._bass_ambitus_delta, slope, self.min_bass_ambitus
        )
        bass_range = (nadir, nadir + bass_ambitus)
        mel_ambitus = self._choose_ambitus(
            self._mel_ambitus_delta, slope, self.min_mel_ambitus
        )
        increments = range(
            (self.tenor_mel_max_apogee - self.tenor_mel_min_nadir)
            - mel_ambitus
            + 1
        )
        tenor_nadir = self.tenor_mel_min_nadir + random.choice(increments)
        tenor_apogee = tenor_nadir + mel_ambitus
        mel_range = (tenor_nadir, tenor_apogee)
        accomp_nadir = max(
            self.min_accomp_nadir, int(round(sum(mel_range) / 2))
        )
        accomp_range = (accomp_nadir, apogee)
        return {
            "mel_range": mel_range,
            "bass_range": bass_range,
            "accomp_range": accomp_range,
        }

    def __call__(
        self, melody_part: str = "soprano"
    ) -> t.Dict[str, t.Tuple[int, int]]:
        nadir = self.within(self.min_nadir, self.max_nadir)
        apogee = self.within(self.min_apogee, self.max_apogee)
        ambitus = apogee - nadir
        slope = self._get_slope(self._min_ambitus, self._max_ambitus, ambitus)
        if melody_part == "soprano":
            out = self._soprano_call(nadir, apogee, slope)
        elif melody_part == "bass":
            out = self._bass_call(nadir, apogee, slope)
        else:
            out = self._tenor_call(nadir, apogee, slope)
        logging.debug(f"ranges={out}")
        return out
