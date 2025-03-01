import math
import random
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from numbers import Number
from types import MappingProxyType

from dumb_composer.classes.scores import Score, _ScoreBase  # Score used in doctests
from dumb_composer.pitch_utils.types import SettingsBase, TimeStamp
from dumb_composer.time import Meter
from dumb_composer.time_utils import (
    get_barline_times_within_duration,
    get_onset_closest_to_middle_of_duration,
)
from dumb_composer.utils.math_ import linear_arc, quadratic_arc, softmax


class Shape(Enum):
    LINEAR = auto()
    QUADRATIC = auto()


@dataclass
class StructuralPartitionerSettings(SettingsBase):
    never_split_dur_in_beats: float = 1.0
    always_split_dur_in_bars: float = 2.0

    # If we are splitting a chord that does not overlap with a barline, we get
    # the metric weights and than apply a softmax to turn this into a
    # probability distribution we can sample from. The higher the temperature,
    # the more evenly the probability mass is spread out.
    # (The temperature should always be > 0.)
    candidates_softmax_temperature: float = 0.5
    arc_shape: Shape = Shape.QUADRATIC


# this could maybe be replaced with flatten_iterables
def _flatten_list_sub(x: t.Union[t.Any, t.List[t.Any]]):
    if isinstance(x, list):
        out = []
        for y in x:
            out += _flatten_list_sub(y)
        return out
    else:
        return [x]


def flatten_list(l: t.List[t.Any]):
    """
    >>> flatten_list([1, 2, 3])
    [1, 2, 3]
    >>> flatten_list([])
    []
    >>> flatten_list([1, [2, 3], [[4], [], [[5, 6]]]])
    [1, 2, 3, 4, 5, 6]
    """
    return _flatten_list_sub(l)


class StructuralPartitioner:
    """
    Partitions long chords so that they will receive multiple "structural melody" notes.

    >>> rntxt = '''m1 C: I
    ... m5 V6
    ... m9 I
    ... '''
    >>> score = Score(chord_data=rntxt)
    >>> structural_partitioner = StructuralPartitioner()  # default settings
    >>> structural_partitioner(score)

    The output is random.
    >>> [chord.token for chord in score.chords]  # doctest: +SKIP
    ['C:I', 'C:I', 'C:I', 'C:I', 'C:I', 'C:I', 'V6', 'V6', 'V6', 'V6', 'V6', 'I']
    """

    arcs = MappingProxyType({Shape.LINEAR: "_linear", Shape.QUADRATIC: "_quadratic"})

    def __init__(self, settings: t.Optional[StructuralPartitionerSettings] = None):
        if settings is None:
            settings = StructuralPartitionerSettings()
        self.settings = settings
        self._ts: None | Meter = None

    def _linear(self):
        return linear_arc(
            min_x=self.settings.never_split_dur_in_beats
            * self._ts.beat_dur,  # type:ignore
            max_x=self.settings.always_split_dur_in_bars
            * self._ts.bar_dur,  # type:ignore
        )

    def _quadratic(self):
        return quadratic_arc(
            min_x=self.settings.never_split_dur_in_beats
            * self._ts.beat_dur,  # type:ignore
            max_x=self.settings.always_split_dur_in_bars
            * self._ts.bar_dur,  # type:ignore
        )

    def step(
        self, chord_onset: TimeStamp, chord_release: TimeStamp
    ) -> t.Union[t.Tuple[Number, Number], t.List]:
        assert self._ts is not None
        chord_dur = chord_release - chord_onset
        split = random.random() < self._arc(chord_dur)
        if not split:
            return (chord_onset, chord_release)

        barline_times = get_barline_times_within_duration(
            chord_onset,
            chord_release,
            self._ts.bar_dur,  # type:ignore
            include_start=False,
        )
        if barline_times:
            # if we are splitting the chord and it overlaps with at least one
            #   barline, then we split at the barline closest to the middle of the
            #   chord.
            split_point = get_onset_closest_to_middle_of_duration(
                barline_times, chord_onset, chord_release
            )
        else:
            # Otherwise, we choose a split point with probability proportional to
            #   the metric strength of each point in the bar, down to
            #   never_split_dur_in_beats
            # TODO debug the weights here
            candidates = self._ts.weights_between(
                (math.ceil(self.settings.never_split_dur_in_beats) * self._ts.beat_dur),
                chord_onset,
                chord_release,
                include_start=False,
            )
            probs = softmax(
                [candidate["weight"] for candidate in candidates],
                temperature=self.settings.candidates_softmax_temperature,
            )
            choice = random.choices(candidates, weights=probs, k=1)[0]
            split_point = choice["onset"]
        return [
            self.step(chord_onset, split_point),
            self.step(split_point, chord_release),
        ]

    def __call__(self, score: _ScoreBase):
        """Updates score in place, stores original chords in score.misc."""
        self._ts = score.ts
        self._arc = getattr(self, self.arcs[self.settings.arc_shape])()

        smallest_duration_already_processed = 2**31
        chords = list(score.chords)

        # We split chords one duration at a time, from longest to shortest.
        # The reason is because chords that are the product of a split
        # should themselves be eligible to be split, but these chords
        # don't exist until we've split the longer chords.

        while True:
            chord_durs = {chord.release - chord.onset for chord in chords}
            eligible_durations = [
                chord_dur
                for chord_dur in chord_durs
                if chord_dur < smallest_duration_already_processed
            ]
            if not eligible_durations:
                break
            new_chords = []
            duration_to_process = max(eligible_durations)
            for chord in chords:
                if chord.suspensions is not None:
                    # TODO: (Malcolm 2023-08-24) remove this condition. For now, we
                    #   are skipping chords that contain suspensions.
                    new_chords.append(chord)
                    breakpoint()
                elif chord.release - chord.onset == duration_to_process:
                    splits = self.step(chord.onset, chord.release)
                    for start, stop in flatten_list(splits):  # type:ignore
                        new_chord = chord.copy()
                        new_chord.onset = start
                        new_chord.release = stop
                        new_chords.append(new_chord)
                else:
                    new_chords.append(chord)

            smallest_duration_already_processed = duration_to_process
            chords = new_chords

        score.misc["original_chords"] = score.chords
        score.chords = chords
