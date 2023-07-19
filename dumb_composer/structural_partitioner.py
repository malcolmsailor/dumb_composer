import math
import random
import typing as t
from dataclasses import dataclass
from enum import Enum, auto
from numbers import Number
from types import MappingProxyType

from .shared_classes import Score, _ScoreBase
from .time import Meter
from .time_utils import (
    get_barline_times_within_duration,
    get_onset_closest_to_middle_of_duration,
)
from .utils.math_ import linear_arc, quadratic_arc, softmax


class Shape(Enum):
    LINEAR = auto()
    QUADRATIC = auto()


@dataclass
class StructuralPartitionerSettings:
    never_split_dur_in_beats: float = 1.0
    always_split_dur_in_bars: float = 2.0

    # If we are splitting a chord that does not overlap with a barline, we get
    # the metric weights and than apply a softmax to turn this into a
    # probability distribution we can sample from. The higher the temperature,
    # the more evenly the probability mass is spread out.
    # (The temperature should always be > 0.)
    # TODO changing this doesn't seem to be having an effect
    candidates_softmax_temperature: float = 0.5
    arc_shape: Shape = Shape.QUADRATIC


# TODO: (Malcolm) can I replace this with flatten_iterables ?
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

    def _step(
        self, chord_onset: Number, chord_release: Number
    ) -> t.Union[t.Tuple[Number, Number], t.List]:
        # TODO for now, we just split once. That means a two-bar chord is
        # guaranteed to be split into two one-bar chords, but these
        # one-bar chords will *not* be split further. Whereas each original
        # one-bar chord has a ~50% chance of being split. That seems wrong.
        chord_dur = chord_release - chord_onset  # type:ignore
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
            candidates = self._ts.weights_between(  # type:ignore
                (
                    math.ceil(self.settings.never_split_dur_in_beats)
                    * self._ts.beat_dur  # type:ignore
                ),
                chord_onset,
                chord_release,
                include_start=False,
                out_format="dict",
            )
            candidate_onsets, candidate_weights = zip(
                *candidates.items()  # type:ignore
            )
            probs = softmax(
                candidate_weights,
                temperature=self.settings.candidates_softmax_temperature,
            )
            split_point = random.choices(candidate_onsets, weights=probs)[0]
        return [
            self._step(chord_onset, split_point),
            self._step(split_point, chord_release),
        ]

    def __call__(self, score: _ScoreBase):
        """Updates score in place.

        TODO it would be good to preserve the original "score.chords" somehow.
        """
        self._ts = score.ts
        self._arc = getattr(self, self.arcs[self.settings.arc_shape])()
        split_chords = []
        for chord in score.chords:
            splits = self._step(chord.onset, chord.release)
            for start, stop in flatten_list(splits):  # type:ignore
                new_chord = chord.copy()
                new_chord.onset = start
                new_chord.release = stop
                split_chords.append(new_chord)
        score.chords = split_chords


# if __name__ == "__main__":
#     settings = StructuralPartitionerSettings(ts="3/4")
#     sp = StructuralPartitioner(settings)
#     for x in range(0, 100):
#         print(x * 0.1, sp._linear(x * 0.1))
