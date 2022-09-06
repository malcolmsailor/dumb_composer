from dataclasses import dataclass
from functools import cached_property
import random
import typing as t

from scipy.stats import truncnorm

from dumb_composer.constants import unspeller
from dumb_composer.utils.math_ import softmax_from_slope


@dataclass
class Ranger:
    min_nadir: t.Union[str, int] = "A1"
    max_nadir: t.Union[str, int] = "Bb2"
    min_apogee: t.Union[str, int] = "G5"
    max_apogee: t.Union[str, int] = "C7"
    min_accomp_nadir: t.Union[str, int] = "Bb2"
    max_accomp_apogee: t.Union[str, int] = "G5"
    min_mel_ambitus: int = 15
    max_mel_ambitus: int = 24
    min_bass_ambitus: int = 15
    max_bass_ambitus: int = 24

    slope_scale: float = 0.1

    def __post_init__(self):
        for attr in (
            "min_nadir",
            "max_nadir",
            "min_apogee",
            "max_apogee",
            "min_accomp_nadir",
            "max_accomp_apogee",
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
        return truncnorm(-1, 1)

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

    def __call__(self, melody_part: str = "soprano"):
        # start by choosing the extremes of the range ("nadir" and "apogee")
        #   then:
        #       melody ambitus tends to get smaller as total ambitus gets smaller
        #       bass ambitus tends to get smaller as total ambitus gets smaller
        if melody_part != "soprano":
            raise NotImplementedError
            # TODO
        nadir = self.within(self.min_nadir, self.max_nadir)
        apogee = self.within(self.min_apogee, self.max_apogee)
        ambitus = apogee - nadir
        slope = self._get_slope(self._min_ambitus, self._max_ambitus, ambitus)

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
        # TODO change "mel_range" to the more accurate "soprano_range"
        return {
            "mel_range": mel_range,
            "bass_range": bass_range,
            "accomp_range": accomp_range,
        }
