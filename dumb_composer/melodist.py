from dataclasses import dataclass
from functools import partial
import random
from numbers import Number
import math
import typing as t
import warnings

import pandas as pd
import numpy as np

from dumb_composer.pitch_utils.chords import get_chords_from_rntxt, Chord
from dumb_composer.pitch_utils.put_in_range import put_in_range
from dumb_composer.pitch_utils.intervals import interval_finder

from dumb_composer.shared_classes import Note

from dumb_composer.pitch_utils.interval_chooser import (
    IntervalChooser,
    IntervalChooserSettings,
)


# TODO move this elsewhere
# some ideas for "musically informed data augmentation":
#   transform homophonic accompaniments into arpeggios etc.
#   transform arpeggios into homophonic accompaniments...

"""Some ideas:
melodic inertia
"arpeggio inertia" (i.e., if there have been two chord tones in a row, make it more likely that there is another chord tone)
"""


@dataclass
class MelodistSettings(IntervalChooserSettings):
    range_: t.Tuple[int, int] = (60, 78)
    max_interval: int = 12
    initial_chord_tone_prob: float = 1.0
    unison_weighted_as: int = 5


def choose_whether_chordtone(n_since_chord_tone: int) -> bool:
    # TODO set rate default
    def _chord_tone_prob(n_since_chord_tone: int, rate: float = 0.5) -> float:
        # non_chord_tone_prob has exponential distribution (sort of, because
        #   n_since_chord_tone is discrete)
        non_chord_tone_prob = rate * math.exp(-rate * n_since_chord_tone)
        return 1 - non_chord_tone_prob

    return random.random() < _chord_tone_prob(n_since_chord_tone)


class Melodist:
    def __init__(
        self,
        settings: t.Optional[MelodistSettings] = None,
        # range_: t.Tuple[int, int] = None,
        # max_interval: int = 12,
        # initial_chord_tone_prob: float = 1.0,
        # unison_weighted_as: int = 5,
    ):
        if settings is None:
            settings = MelodistSettings()
        self._range = settings.range_
        self._max_interval = settings.max_interval
        self._ic = IntervalChooser(settings)
        self._initial_chord_tone_prob = settings.initial_chord_tone_prob
        self._interval_finder = partial(
            interval_finder,
            min_pitch=self._range[0],
            max_pitch=self._range[1],
            max_interval=self._max_interval,
        )

    def _recurse(
        self,
        chord_data: t.Union[str, pd.DataFrame],
        rhythm: t.Sequence[Number], # TODO: (Malcolm) fix type annot
        _melody: t.Optional[t.List[Note]] = None,
        _melody_i: int = 0,
        _n_since_chord_tone: int = 2**31,
        _chord_i: int = 0,
    ) -> pd.DataFrame:
        def _update_chord_i():
            nonlocal _chord_i
            for _chord_i in range(_chord_i, len(chord_data)):
                if rhythm.loc[_melody_i, "onset"] < chord_data[_chord_i].release:
                    break

        def _get_eligible_pcs(chord_tone: bool):
            attr_name = "pcs" if chord_tone else "scale_pcs"
            # TODO handle altered tones in scales
            return getattr(chord_data[_chord_i], attr_name)

        def _update_n_since_chord_tone():
            nonlocal _n_since_chord_tone
            last_p_is_chord_tone = _melody[-1].pitch % 12 in chord_data[_chord_i].pcs
            if last_p_is_chord_tone:
                _n_since_chord_tone = 0
            else:
                _n_since_chord_tone += 1

        def _proceed():
            _update_n_since_chord_tone()
            return self._recurse(
                chord_data,
                rhythm,
                _melody,
                _melody_i + 1,
                _n_since_chord_tone,
                _chord_i,
            )

        if _melody_i == len(rhythm):
            return pd.DataFrame(_melody)
        _update_chord_i()
        if _melody is None:
            assert _melody_i == 0
            _melody = []
            # choose initial note
            chord_tone = random.random() < self._initial_chord_tone_prob
            pc = random.choice(_get_eligible_pcs(chord_tone))
            pitch = put_in_range(pc, *self._range)
            _melody = [Note(pitch, *rhythm.loc[_melody_i])]

        else:
            # choose note based on previous note
            prev_pitch = _melody[-1].pitch
            chord_tone = choose_whether_chordtone(_n_since_chord_tone)
            eligible_intervals = self._interval_finder(
                prev_pitch, _get_eligible_pcs(chord_tone)
            )
            chosen_interval = self._ic(eligible_intervals)
            _melody.append(Note(prev_pitch + chosen_interval, *rhythm.loc[_melody_i]))

        return _proceed()

    def __call__(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        rhythm: t.Union[t.Sequence[t.Tuple[Number, Number]], pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Args:
            chord_data:
                If string, should be in roman-text format.
                If a list, should be the output of the get_chords_from_rntxt
                    function or similar.
            rhythm:
                If sequence, consists of 2-tuples of form (onset, release).
                If Pandas DataFrame, must have "onset" and "release" columns.
        """
        if isinstance(chord_data, str):
            chord_data, _ = get_chords_from_rntxt(chord_data)
        if not isinstance(rhythm, pd.DataFrame):
            rhythm = pd.DataFrame(rhythm, columns=["onset", "release"])
        # we use numeric indexing above so we need to make sure the indexes
        #   are default range indices
        for idx in (rhythm.index,):
            assert isinstance(idx, pd.RangeIndex)
            assert idx.start == 0
            assert idx.step == 1
        if chord_data[-1].release < rhythm.release.max():
            warnings.warn(
                f"chord_data[-1].release is {chord_data[-1].release} but "
                f"rhythm.release.max() is {rhythm.release.max()}"
            )
        return self._recurse(chord_data, rhythm)
