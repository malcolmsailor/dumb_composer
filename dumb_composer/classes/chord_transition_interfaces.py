import collections.abc
import logging
import typing as t
from collections import defaultdict

from dumb_composer.chords.chords import Allow, is_same_harmony
from dumb_composer.classes.scores import (
    PrefabScore,
    ScoreWithAccompaniments,
    _ScoreBase,
)
from dumb_composer.pitch_utils.types import (
    BASS,
    AllowLiteral,
    Note,
    OuterVoice,
    Pitch,
    ScalarInterval,
    Suspension,
    TimeStamp,
    Voice,
)
from dumb_composer.time import Meter

LOGGER = logging.getLogger(__name__)


class _ChordTransitionInterface:
    def __init__(self, reference_score: _ScoreBase, get_len: t.Callable[[], int]):
        self._score = reference_score
        self._get_len = get_len

    @property
    def score(self):  # pylint: disable=missing-docstring
        return self._score

    @property
    def ts(self) -> Meter:
        return self._score.ts

    @property
    def structural_voices(self) -> t.Iterator[Voice]:
        yield from self._score._structural.keys()

    @property
    def i(self) -> int:
        # We first need to wait until there are at least 2 items made.
        # Perhaps that should be a parameter.

        # TODO: (Malcolm 2023-08-15) I feel that we should be able to offset by just 1
        #   here
        bass_len = len(self._score._structural[BASS]) - 2
        if bass_len < 0:
            return bass_len
        return self._get_len()

    @property
    def departure_time(self) -> TimeStamp:
        assert self.i >= 0
        return self._score.chords[self.i].onset

    @property
    def arrival_time(self) -> TimeStamp:
        assert self.i >= 0
        return self._score.chords[self.i].release

    def structural_interval_from_departure_to_arrival(
        self, voice: Voice
    ) -> ScalarInterval:
        return self.departure_scale.get_interval(
            self.departure_pitch(voice),
            self.arrival_pitch(voice),
            scale2=self.arrival_scale,
        )

    def _validate_state(self) -> bool:
        raise NotImplementedError

    def validate_state(self) -> bool:
        # TODO: (Malcolm 2023-08-04) not sure this is correct any more
        return self._validate_state() and self._score.validate_state()

    def departure_pitch(self, voice: Voice) -> Pitch:
        if self.i < 0:
            raise ValueError
        return self._score._structural[voice][self.i]

    def arrival_pitch(self, voice: Voice):
        if self.i < 0:
            raise ValueError
        return self._score._structural[voice][self.i + 1]

    def departure_interval_above_bass(self, voice: Voice):
        if voice is OuterVoice.BASS:
            return 0
        return self.departure_scale.get_reduced_scalar_interval(
            self.departure_pitch(OuterVoice.BASS), self.departure_pitch(voice)
        )

    @property
    def departure_chord(self):
        assert self.i >= 0
        return self._score.chords[self.i]

    @property
    def arrival_chord(self):
        assert self.i >= 0
        return self._score.chords[self.i + 1]

    @property
    def departure_scale(self):
        assert self.i >= 0
        return self._score.scales[self.i]

    @property
    def arrival_scale(self):
        assert self.i >= 0
        return self._score.scales[self.i + 1]

    def departure_is_preparation(self, voice: Voice) -> bool:
        assert self.i >= 0
        return self._score.is_suspension_at(self.arrival_time, voice)
        # return self._score.prev_is_suspension(voice)

    def departure_suspension(self, voice: Voice) -> Suspension | None:
        assert self.i >= 0
        # return self._score.prev_suspension(voice=voice, offset_from_current=2)
        return self._score.get_suspension_at(self.departure_time, voice)

    def departure_is_suspension(self, voice: Voice) -> bool:
        assert self.i >= 0
        return self._score.is_suspension_at(self.departure_time, voice)
        # return self._score.prev_is_resolution(voice)

    def departure_is_resolution(self, voice: Voice) -> bool:
        assert self.i >= 0
        # TODO: (Malcolm 2023-08-10) can we check if departure is in resolutions instead?
        return self._score.is_resolution_at(self.departure_time, voice)
        # return self._score.is_suspension_at(self.i - 1, voice)
        # return self._score._suspension_resolutions[voice].get(self.i, None) is None

    def at_chord_change(
        self,
        compare_scales: bool = True,
        compare_inversions: bool = True,
        allow_subsets: bool = False,
    ):
        return not is_same_harmony(
            self.departure_chord,
            self.arrival_chord,
            compare_scales,
            compare_inversions,
            allow_subsets,
        )

    def departure_attr(self, attr_name: str, voice: Voice):
        assert self.i >= 0
        return getattr(self._score, attr_name)[voice][self.i]

    def arrival_attr(self, attr_name: str, voice: Voice):
        assert self.i >= 0
        return getattr(self._score, attr_name)[voice][self.i + 1]


class PrefabInterface(_ChordTransitionInterface):
    def __init__(self, reference_score: PrefabScore):
        get_len = lambda: self.prefab_lengths[0]
        super().__init__(reference_score, get_len)
        self._tied_prefab_indices: defaultdict[Voice, set[int]] = defaultdict(set)
        self._allow_prefab_start_with_rest: defaultdict[
            Voice, dict[int, Allow]
        ] = defaultdict(dict)
        self._score: PrefabScore

    def _validate_state(self) -> bool:
        return len(set(self.prefab_lengths)) == 1

    def last_existing_prefab_note(self, voice: Voice) -> Note | None:
        prefabs = self._score.prefabs[voice]
        if not prefabs:
            return None
        last_prefab = prefabs[-1]
        if not last_prefab:
            return None
        return last_prefab[-1]

    def departure_follows_tie(self, voice: Voice) -> bool:
        if self.i < 1:
            return False
        return self.i - 1 in self._tied_prefab_indices[voice]

    def add_tie_from_departure(self, voice: Voice):
        assert self.i not in self._tied_prefab_indices[voice]
        self._tied_prefab_indices[voice].add(self.i)

    def remove_tie_from_departure(self, voice: Voice):
        self._tied_prefab_indices[voice].remove(self.i)

    def get_departure_can_start_with_rest(self, voice: Voice) -> AllowLiteral:
        out = self._allow_prefab_start_with_rest[voice].get(self.i, "NO")
        if out is Allow.YES or out == "YES":
            return "YES"
        if out is Allow.NO or out == "NO":
            return "NO"
        if out is Allow.ONLY or out == "ONLY":
            return "ONLY"
        raise ValueError

    def set_arrival_can_start_with_rest(self, voice: Voice, allow: Allow) -> None:
        self._allow_prefab_start_with_rest[voice][self.i + 1] = allow

    def unset_arrival_can_start_with_rest(self, voice: Voice) -> None:
        del self._allow_prefab_start_with_rest[voice][self.i + 1]

    @property
    def prefab_lengths(self) -> list[int]:
        out = [len(prefabs) for prefabs in self._score.prefabs.values()]
        if not out:
            return [0]
        return out


class AccompanimentInterface(_ChordTransitionInterface):
    def __init__(self, reference_score: ScoreWithAccompaniments):
        get_len = lambda: len(reference_score.accompaniments)
        super().__init__(reference_score, get_len)
