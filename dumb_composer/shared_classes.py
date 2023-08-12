import collections.abc
import logging
import math
import re
import textwrap
import typing as t
from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from copy import copy
from dataclasses import asdict, dataclass
from enum import Enum
from functools import cached_property
from numbers import Number

import pandas as pd

from dumb_composer.constants import TRACKS
from dumb_composer.pitch_utils.chords import (
    Allow,
    Chord,
    get_chords_from_rntxt,
    is_same_harmony,
)
from dumb_composer.pitch_utils.music21_handler import get_ts_from_rntxt
from dumb_composer.pitch_utils.put_in_range import put_in_range
from dumb_composer.pitch_utils.scale import Scale, ScaleDict
from dumb_composer.pitch_utils.spacings import RangeConstraints
from dumb_composer.pitch_utils.types import (
    BASS,
    GLOBAL,
    TIME_TYPE,
    Annotation,
    InnerVoice,
    Note,
    OuterVoice,
    Pitch,
    PitchClass,
    ScalarInterval,
    TimeStamp,
    Voice,
)
from dumb_composer.suspensions import Suspension, SuspensionCombo
from dumb_composer.time import Meter
from dumb_composer.utils.df_helpers import sort_note_df
from dumb_composer.utils.iterables import flatten_iterables

LOGGER = logging.getLogger(__name__)


# class Annotation(pd.Series):
#     def __init__(self, onset, text):
#         # TODO remove the next lines when I've figured out how to get
#         #   text annotations to display correctly
#         text = text.replace("_", "").replace(" ", "")
#         super().__init__(
#             {"onset": onset, "text": text, "type": "text", "track": 0}  # type:ignore
#         )


def notes(
    pitches: t.Sequence[int], onset: TimeStamp, release: TimeStamp, track: int = 1
) -> t.List[Note]:
    return [Note(pitch, onset, release, track=track) for pitch in pitches]


def print_notes(notes: t.Iterable[Note]) -> None:
    """Helper function to make printing notes convenient in doctests.

    >>> print_notes(
    ...     [
    ...         Note(pitch=48, onset=0, release=1),
    ...         Note(pitch=52, onset=1, release=2),
    ...         Note(pitch=55, onset=1, release=2),
    ...     ]
    ... )
    on  off   pitches
    0    1     (48,)
    1    2  (52, 55)
    """
    df = pd.DataFrame(notes)  # type:ignore
    df["on_off"] = [(row.onset, row.release) for _, row in df.iterrows()]
    df.sort_values("on_off", inplace=True, kind="mergesort")
    out = defaultdict(list)
    for on, off in df["on_off"].unique():
        out["on"].append(on)
        out["off"].append(off)
        out["pitches"].append(
            tuple(df[df["on_off"] == (on, off)]["pitch"])  # type:ignore
        )
    df = pd.DataFrame(out)
    for line in str(df).split("\n"):
        m = re.match(r"^\d* +(.*)$", line)
        if m:
            print(m.group(1))
        else:
            print(line)


def apply_ties(
    notes: t.Iterable[Note], check_correctness: bool = False
) -> t.List[Note]:
    """
    >>> len(apply_ties([Note(60, 0.0, 1.0), Note(60, 1.0, 2.0)]))
    2
    >>> len(apply_ties([Note(60, 0.0, 1.0, tie_to_next=True), Note(60, 1.0, 2.0)]))
    1
    >>> len(
    ...     apply_ties(
    ...         [
    ...             Note(60, 0.0, 1.0, tie_to_next=True),
    ...             Note(60, 1.0, 2.0, tie_to_next=True),
    ...             Note(60, 2.0, 3.0),
    ...             Note(60, 3.0, 4.0, tie_to_next=True),
    ...             Note(60, 4.0, 5.0),
    ...         ]
    ...     )
    ... )
    2

    An exception is always raised if the last note is a tie (because there is
    nothing for it to be tied to).

    >>> apply_ties([Note(60, 0.0, 1.0), Note(60, 1.0, 2.0, tie_to_next=True)])
    Traceback (most recent call last):
    ValueError: last Note has tie_to_next=True

    If check_correctness is True, we make sure that all tied notes have
    the same pitch and that the release of the first note == the onset of the
    second.

    >>> apply_ties(
    ...     [
    ...         Note(60, 0.0, 1.0, tie_to_next=True),
    ...         Note(60, 1.0, 2.0, tie_to_next=True),
    ...         Note(60, 2.5, 3.0),
    ...     ],
    ...     check_correctness=True,
    ... )
    Traceback (most recent call last):
    ValueError: Release of note at 2.0 != onset of note at 2.5

    >>> apply_ties(
    ...     [
    ...         Note(60, 0.0, 1.0, tie_to_next=True),
    ...         Note(60, 1.0, 2.0),
    ...         Note(60, 2.5, 3.0, tie_to_next=True),
    ...         Note(62, 3.0, 4.0),
    ...     ],
    ...     check_correctness=True,
    ... )
    Traceback (most recent call last):
    ValueError: Tied notes have different pitches 60 and 62
    """

    def _check_pair(note1: Note, note2: Note):
        if note1.pitch != note2.pitch:
            raise ValueError(
                f"Tied notes have different pitches {note1.pitch} " f"and {note2.pitch}"
            )
        if not math.isclose(note1.release, note2.onset):
            raise ValueError(
                f"Release of note at {note1.release} != "
                f"onset of note at {note2.onset}"
            )

    out = []
    queue = deque()
    for note in notes:
        if note.tie_to_next:
            queue.append(note)
        elif queue:
            first_note = queue.popleft()
            if check_correctness:
                note1 = first_note
                while queue:
                    note2 = queue.popleft()
                    _check_pair(note1, note2)
                    note1 = note2
                _check_pair(note1, note)
            new_note = first_note.copy()
            new_note.release = note.release
            out.append(new_note)
            queue.clear()
        else:
            out.append(note.copy())
    if queue:
        raise ValueError("last Note has tie_to_next=True")
    return out


# class Chord:
#     def __init__(
#         self,
#         pcs: t.Sequence[int],
#         scale_pcs: t.Sequence[int],
#         onset: Number,
#         release: Number,
#         foot: int,
#         token: t.Optional[str],
#     ):
#         # foot, bass, melody, pcs, inversion, scale, onset, release, inversion
#         pass


class ScaleGetter:
    """
    >>> C_major = (0, 2, 4, 5, 7, 9, 11)
    >>> D_major = (2, 4, 6, 7, 9, 11, 1)
    >>> Octatonic = (0, 2, 3, 5, 6, 8, 9, 11)
    >>> scale_getter = ScaleGetter([C_major, D_major, Octatonic])
    >>> len(scale_getter)
    3
    >>> scale_getter[0]
    Scale(pcs=(0, 2, 4, 5, 7, 9, 11), zero_pitch=0, tet=12)
    """

    def __init__(self, scale_pcs: t.Iterable[t.Tuple[PitchClass, ...]]):
        self._scale_pcs = list(scale for scale in scale_pcs)
        self._scales = ScaleDict()

    def __len__(self) -> int:
        return len(self._scale_pcs)

    def __getitem__(self, idx: int) -> Scale:
        return self._scales[self._scale_pcs[idx]]

    def insert_scale_pcs(self, i: int, scale_pcs: t.Tuple[PitchClass, ...]) -> None:
        self._scale_pcs.insert(i, scale_pcs)

    def pop_scale_pcs(self, i: int) -> t.Tuple[PitchClass, ...]:
        return self._scale_pcs.pop(i)


class StructuralMelodyIntervals(collections.abc.Mapping):
    """Provides a [] interface to retrieve structural melody intervals."""

    def __init__(
        self,
        scales: ScaleGetter,
        structural_soprano: t.List[int],
        structural_bass: t.List[int],
    ):
        # This class doesn't really have custody of its attributes, which
        #   should all be attributes of the PrefabScore that creates it. The only
        #   reason this class exists is to that we can provide a [] subscript
        #   syntax to PrefabScore.structural_soprano_intervals
        self.scales = scales
        self.structural_soprano = structural_soprano
        self.structural_bass = structural_bass

    def __getitem__(self, idx):
        return self.scales[idx].get_reduced_scalar_interval(
            self.structural_bass[idx], self.structural_soprano[idx]
        )

    def __len__(self):
        return min(len(self.structural_bass), len(self.structural_soprano))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class _ScoreBase:
    def __init__(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        ts: Meter | str | None = None,
        transpose: int = 0,
    ):
        if isinstance(chord_data, str):
            logging.debug(f"reading chords from {chord_data}")
            ts = get_ts_from_rntxt(chord_data)
            chord_data = get_chords_from_rntxt(chord_data)
        elif ts is None:
            raise ValueError(f"`ts` must be supplied if `chord_data` is not a string")
        if isinstance(ts, str):
            self.ts = Meter(ts)
        else:
            self.ts = ts
        self._chords = chord_data
        if transpose:
            self._chords = [chord.transpose(transpose) for chord in self._chords]
        self._scale_getter = ScaleGetter(chord.scale_pcs for chord in chord_data)
        # self.range_constraints = range_constraints
        self._structural: defaultdict[Voice, list[Pitch]] = defaultdict(list)
        self._suspensions: defaultdict[
            Voice, dict[TimeStamp, Suspension]
        ] = defaultdict(dict)
        # TODO: (Malcolm 2023-08-04) I'm not sure if using the "structural unit" as
        #   key for suspension resolutions (quasi-bar number) still works. What if
        #   a soprano suspension leads to splitting a chord and then a tenor
        #   suspension leads to splitting the chord still further?
        self._suspension_resolutions: t.DefaultDict[
            Voice, t.Dict[TimeStamp, int]
        ] = defaultdict(dict)
        # self.annotations: defaultdict[str, t.List[Annotation]] = defaultdict(list)
        self.annotations: defaultdict[
            Voice, defaultdict[str, dict[TimeStamp, Annotation]]
        ] = defaultdict(lambda: defaultdict(dict))
        self.misc: dict[str, t.Any] = {}

        self._structural_soprano_interval_getter = StructuralMelodyIntervals(
            self.scales, self.structural_soprano, self.structural_bass
        )
        # _split_chords indices are to the chord *before* the split. E.g., if we
        #   split chord i=3, the split chords will have i=3 and i=4. We then add 3
        #   to _split_chords.
        self._split_chords: Counter[int] = Counter()

    def __eq__(self, other):
        if type(other) is type(self):
            for k, v in self.__dict__.items():
                if k in ("_scale_getter"):
                    continue
                other_v = other.__dict__[k]
                if isinstance(v, defaultdict):
                    if v == other_v or (
                        not any(x for x in v.values())
                        and not any(x for x in other_v.values())
                    ):
                        continue
                    return False
                elif v != other.__dict__[k]:
                    return False
            return True
        return False

    # ----------------------------------------------------------------------------------
    # State validation etc.
    # ----------------------------------------------------------------------------------

    def validate_state(self) -> bool:
        return len({len(pitches) for pitches in self._structural.values()}) == 1

    # ----------------------------------------------------------------------------------
    # Properties and helper functions
    # ----------------------------------------------------------------------------------
    @cached_property
    def structural_soprano(self):
        return self._structural[OuterVoice.MELODY]

    @cached_property
    def structural_bass(self):
        return self._structural[OuterVoice.BASS]

    @cached_property
    def pc_bass(self) -> t.List[int]:
        return [chord.foot for chord in self._chords]  # type:ignore

    # TODO: (Malcolm 2023-08-10) possibly restore
    def get_suspension_at(self, time: TimeStamp, voice: Voice) -> Suspension | None:
        return self._suspensions[voice].get(time, None)

    def is_suspension_at(self, time: TimeStamp, voice: Voice) -> bool:
        return self.get_suspension_at(time, voice) is not None

    def is_resolution_at(self, time: TimeStamp, voice: Voice) -> bool:
        return time in self._suspension_resolutions[voice]

    @property
    def chords(self) -> t.List[Chord]:
        return self._chords

    @chords.setter
    def chords(self, new_chords: t.List[Chord]):
        self._chords = new_chords
        # deleting cached property self.pc_bass allows it to be regenerated next time it
        #   is needed
        if hasattr(self, "pc_bass"):
            del self.pc_bass
        # TODO: (Malcolm 2023-07-28) why is ScaleGetter needed? perhaps Scale
        #   should be composed within Chord?
        self._scale_getter = ScaleGetter(chord.scale_pcs for chord in self._chords)
        self._structural_soprano_interval_getter = StructuralMelodyIntervals(
            self.scales, self.structural_soprano, self.structural_bass
        )

    @property
    def scales(self) -> ScaleGetter:
        return self._scale_getter

    @property
    @abstractmethod
    def default_existing_pitch_attr_names(self) -> t.Tuple[str]:
        raise NotImplementedError

    def get_existing_pitches(
        self,
        idx: int,
        attr_names: t.Sequence[str] | None = None,
    ) -> t.Tuple[Pitch]:
        raise NotImplementedError
        if attr_names is None:
            attr_names = self.default_existing_pitch_attr_names

        return tuple(  # type:ignore
            flatten_iterables(
                getattr(self, attr_name)[idx]
                for attr_name in attr_names
                if len(getattr(self, attr_name)) > idx
            )
        )

    def _validate_split_ith_chord_at(self, i: int, chord: Chord, time: TimeStamp):
        assert chord.onset < time < chord.release

    def split_ith_chord_at(
        self, i: int, time: TimeStamp, check_correctness: bool = True
    ) -> None:
        """
        >>> rntxt = '''Time Signature: 4/4
        ... m1 C: I
        ... m2 V
        ... m3 I'''
        >>> score = PrefabScore(rntxt)
        >>> {float(chord.onset): chord.token for chord in score.chords}
        {0.0: 'C:I', 4.0: 'V', 8.0: 'I'}
        >>> score.split_ith_chord_at(1, 6.0)
        >>> {float(chord.onset): chord.token for chord in score.chords}
        {0.0: 'C:I', 4.0: 'V', 6.0: 'V', 8.0: 'I'}
        >>> score.merge_ith_chords(1)
        >>> {float(chord.onset): chord.token for chord in score.chords}
        {0.0: 'C:I', 4.0: 'V', 8.0: 'I'}
        """
        # just in case pc_bass has not been computed yet, we need
        #   to compute it now:
        LOGGER.debug(f"splitting chord {i=} at {time=}")
        self.pc_bass
        # TODO make debug flag for check_correctness
        chord = self.chords[i]
        if check_correctness:
            self._validate_split_ith_chord_at(i, chord, time)
        new_chord = chord.copy()
        chord.release = time
        new_chord.onset = time
        self.chords.insert(i + 1, new_chord)
        self.pc_bass.insert(i + 1, self.pc_bass[i])

        # Extend structural notes if necessary by duplicating the ith pitch
        for structural_notes in self._structural.values():
            if len(structural_notes) > i:
                structural_notes.insert(i + 1, structural_notes[i])

        # If we split a suspension, we want to copy it over to the new chord
        for suspensions in self._suspensions.values():
            if chord.onset in suspensions:
                suspensions[new_chord.onset] = suspensions[chord.onset]

        self._scale_getter.insert_scale_pcs(i + 1, new_chord.scale_pcs)
        self._split_chords[i] += 1

    def merge_ith_chords(self, i: int, check_correctness: bool = True) -> None:
        """
        Merge score.chords[i] and score.chords[i + 1]
        """
        # just in case pc_bass has not been computed yet, we need
        #   to compute it now:
        LOGGER.debug(f"merging chord {i=}")
        self.pc_bass

        chord1, chord2 = self.chords[i : i + 2]
        chord1.release = chord2.release
        self.chords.pop(i + 1)
        self.pc_bass.pop(i + 1)

        for structural_notes in self._structural.values():
            if len(structural_notes) > i + 1:
                structural_notes.pop(i + 1)

        # If there was a suspension at chord2 onset (e.g., because a suspension on
        #   chord1 was previously split), we need to remove it
        for suspensions in self._suspensions.values():
            if chord2.onset in suspensions:
                suspensions.pop(chord2.onset)

        # if len(self.structural_soprano) > i + 2:
        #     self.structural_soprano.pop(i + 1)

        self._scale_getter.pop_scale_pcs(i + 1)
        if check_correctness:
            assert self._split_chords[i]
            chord2.onset = chord1.onset
            assert chord1 == chord2

        self._split_chords[i] -= 1

    def get_flat_list_of_all_annotations(self) -> list[Annotation]:
        out = []
        for voice_annots in self.annotations.values():
            for annots_of_type in voice_annots.values():
                out.extend(list(annots_of_type.values()))

        return out

    @property
    def annotations_as_df(self) -> pd.DataFrame:
        out = pd.DataFrame(self.get_flat_list_of_all_annotations())
        # only one annotation per time-point appears in the kern files (or is it
        #   the verovio realizations?). Anyway, we merge them into one here. TODO
        #   is there a way around this constraint?
        if not len(out):
            return out
        temp = []
        out = out.sort_values(by=["onset", "track"])
        for (onset, track), rows in out.groupby(["onset", "track"]):
            new_annot = Annotation(onset, text="".join(rows["text"]), track=track)
            temp.append(new_annot)
        # for onset in sorted(out.onset.unique()):
        #     new_annot = Annotation(
        #         onset,
        #         "".join(annot.text for _, annot in out[out.onset == onset].iterrows()),
        #     )
        #     temp.append(new_annot)
        out = pd.DataFrame(temp)
        return out

    def as_df(self, attr_name: str, track: int = 0) -> pd.DataFrame:
        notes = getattr(self, attr_name)
        return pd.DataFrame(
            Note(
                melody_pitch,
                self.chords[i].onset,
                self.chords[i].release,
                track=track,
            )
            for i, melody_pitch in enumerate(notes)
        )

    @property
    def structural_melody_as_df(self):
        return self.as_df(
            "structural_melody", track=TRACKS["structural"][OuterVoice.MELODY]
        )

    @property
    def structural_bass_as_df(self):
        return self.as_df(
            "structural_bass", track=TRACKS["structural"][OuterVoice.BASS]
        )

    def get_df(self, contents: t.Union[str, t.Sequence[str]]) -> pd.DataFrame:
        if isinstance(contents, str):
            contents = [contents]
        dfs = []
        for name in contents:
            if hasattr(self, f"{name}_as_df"):
                dfs.append(getattr(self, f"{name}_as_df"))
            else:
                dfs.append(self.as_df(name))
        df = pd.concat(dfs)
        return sort_note_df(df)

    @property
    def structural_as_df(self) -> pd.DataFrame:
        dfs = []
        for voice, notes in self._structural.items():
            track = TRACKS["structural"][voice]
            sub_df = pd.DataFrame(
                Note(
                    pitch,
                    self.chords[i].onset,
                    self.chords[i].release,
                    track=track,
                )
                for i, pitch in enumerate(notes)
            )
            dfs.append(sub_df)
        df = pd.concat(dfs)
        return sort_note_df(df)


class Score(_ScoreBase):
    """This class provides a "shared working area" for the various classes and
    functions that build a score. It doesn't encapsulate much of anything.

    >>> rntxt = "m1 C: I b2 ii6 b3 V b4 I6"
    >>> score = Score(chord_data=rntxt)

    >>> [chord.pcs for chord in score.chords]
    [(0, 4, 7), (5, 9, 2), (7, 11, 2), (4, 7, 0)]

    >>> score.pc_bass  # Pitch classes
    [0, 5, 7, 4]
    """

    @property
    def default_existing_pitch_attr_names(self) -> t.Tuple[str]:
        raise NotImplementedError
        return "structural_bass", "structural_soprano"  # type:ignore


class FourPartScore(_ScoreBase):
    """
    >>> rntxt = "m1 C: I b2 ii6 b3 V b4 I6"
    >>> score = FourPartScore(chord_data=rntxt)

    >>> score.pc_bass  # Pitch-classes, not pcs
    [0, 5, 7, 4]

    Let's create a bass:
    >>> score.structural_bass.extend(put_in_range(pc, low=36) for pc in (score.pc_bass))
    >>> score.structural_bass
    [36, 41, 43, 40]

    Let's add a melody:
    >>> score.structural_soprano.extend([72, 74, 71, 72])

    # TODO: (Malcolm 2023-08-03) implement or change
    # >>> score.get_existing_pitches(0)
    # (36, 72)
    # >>> score.get_existing_pitches(1)
    # (41, 74)

    Let's add inner parts:
    >>> score.structural_tenor.extend([55, 57, 55, 55])
    >>> score.structural_alto.extend([64, 65, 62, 77])

    # >>> score.get_existing_pitches(0)
    # (36, 55, 64, 72)
    """

    def __init__(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        ts: t.Optional[t.Union[Meter, str]] = None,
        transpose: int = 0,
    ):
        super().__init__(
            chord_data,
            ts=ts,
            transpose=transpose,
        )

    @property
    def structural_tenor(self):
        return self._structural[InnerVoice.TENOR]

    @property
    def structural_alto(self):
        return self._structural[InnerVoice.ALTO]

    @property
    def structural_tenor_as_df(self):
        return self.as_df(
            "structural_tenor", track=TRACKS["structural"][InnerVoice.TENOR]
        )

    @property
    def structural_alto_as_df(self):
        return self.as_df(
            "structural_alto", track=TRACKS["structural"][InnerVoice.ALTO]
        )

    @property
    def default_existing_pitch_attr_names(self) -> t.Tuple[str, ...]:
        raise NotImplementedError
        return (
            "structural_bass",
            "structural_tenor",
            "structural_alto",
            "structural_soprano",
        )


class _ChordTransitionInterface:
    def __init__(self, reference_score: _ScoreBase):
        self._score = reference_score

    @property
    def ts(self) -> Meter:
        return self._score.ts

    @property
    def structural_voices(self) -> t.Iterator[Voice]:
        yield from self._score._structural.keys()

    @property
    def i(self) -> int:
        return len(self._score._structural[BASS]) - 2

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

    def validate_state(self) -> bool:
        # TODO: (Malcolm 2023-08-04) not sure this is correct any more
        return self._score.validate_state()

    def departure_pitch(self, voice: Voice) -> Pitch:
        if self.i < 0:
            raise ValueError
        return self._score._structural[voice][self.i]
        # return self._score.prev_pitch(voice, offset_from_current=2)

    def arrival_pitch(self, voice: Voice):
        if self.i < 0:
            raise ValueError
        return self._score._structural[voice][self.i + 1]
        # return self._score.prev_pitch(voice)

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


class ScoreWithAccompaniments(ABC, _ScoreBase):
    @property
    @abstractmethod
    def accompaniments(self) -> list[list[Note]]:
        raise NotImplementedError()


class PrefabScore(FourPartScore):
    """This class provides a "shared working area" for the various classes and
    functions that build a score. It doesn't encapsulate much of anything.

    >>> rntxt = "m1 C: I b2 ii6 b3 V b4 I6"
    >>> score = PrefabScore(chord_data=rntxt)

    >>> [chord.pcs for chord in score.chords]
    [(0, 4, 7), (5, 9, 2), (7, 11, 2), (4, 7, 0)]

    >>> score.pc_bass  # Pitch-classes
    [0, 5, 7, 4]
    """

    def __init__(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        ts: t.Optional[t.Union[Meter, str]] = None,
        transpose: int = 0,
    ):
        super().__init__(chord_data, ts=ts, transpose=transpose)

        self.prefabs: defaultdict[Voice, list[list[Note]]] = defaultdict(list)
        self.accompaniments: t.List[t.List[Note]] = []
        self._tied_prefab_indices: defaultdict[Voice, set[int]] = defaultdict(set)
        self._allow_prefab_start_with_rest: t.Dict[int, Allow] = {}

    @property
    def default_existing_pitch_attr_names(self) -> t.Tuple[str]:
        raise NotImplementedError
        return "structural_bass", "structural_soprano"  # type:ignore

    @property
    def prefabs_as_df(self) -> pd.DataFrame:
        sub_dfs = []
        for voice, notes in self.prefabs.items():
            track = TRACKS["prefabs"][voice]
            tied_notes = apply_ties(
                (note for prefab in notes for note in prefab),
                check_correctness=True,  # TODO remove this when I'm more confident
            )
            sub_df = pd.DataFrame(tied_notes)
            sub_df["track"] = track
            sub_dfs.append(sub_df)
        return pd.concat(sub_dfs)

    @property
    def accompaniments_as_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            note
            for accompaniment in self.accompaniments
            for note in accompaniment  # type:ignore
        )

    def _validate_split_ith_chord_at(self, i: int, chord: Chord, time: TIME_TYPE):
        super()._validate_split_ith_chord_at(i, chord, time)
        for voice in self.prefabs:
            assert len(self.prefabs[voice]) <= i
        assert len(self.accompaniments) <= i


class PrefabInterface(_ChordTransitionInterface):
    def __init__(self, reference_score: PrefabScore):
        super().__init__(reference_score)
        self._tied_prefab_indices: defaultdict[Voice, set[int]] = defaultdict(set)
        self._allow_prefab_start_with_rest: defaultdict[
            Voice, dict[int, Allow]
        ] = defaultdict(dict)
        self._score: PrefabScore

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

    def get_departure_can_start_with_rest(self, voice: Voice) -> Allow:
        return self._allow_prefab_start_with_rest[voice].get(self.i, Allow.NO)

    def set_arrival_can_start_with_rest(self, voice: Voice, allow: Allow) -> None:
        self._allow_prefab_start_with_rest[voice][self.i + 1] = allow

    def unset_arrival_can_start_with_rest(self, voice: Voice) -> None:
        del self._allow_prefab_start_with_rest[voice][self.i + 1]


class AccompanimentInterface(_ChordTransitionInterface):
    pass


class ScoreInterface:
    def __init__(
        self,
        score: _ScoreBase,
        get_i: t.Callable[[_ScoreBase], int],
        validate: t.Callable[[_ScoreBase], bool],
    ):
        self._score = score
        self._get_i = get_i
        # TODO: (Malcolm 2023-08-04) perhaps validation can be implemented more for
        #   user
        self._validate = validate

    @property
    def ts(self) -> Meter:
        return self._score.ts

    @property
    def score(self):  # pylint: disable=missing-docstring
        return self._score

    # ----------------------------------------------------------------------------------
    # State and validation
    # ----------------------------------------------------------------------------------

    def validate_state(self) -> bool:
        return self._validate(self._score)

    @property
    def i(self):
        return self._get_i(self._score)

    @property
    def empty(self) -> bool:
        # TODO: (Malcolm 2023-08-09) think about validation here.
        # assert self.validate_state()
        return self.i == 0

    @property
    def complete(self) -> bool:
        # assert self.validate_state()
        assert self.i <= len(self._score.chords)
        return self.i == len(self._score.chords)

    @property
    def structural_lengths(self) -> list[int]:
        return [len(v) for v in self._score._structural.values()]

    # ----------------------------------------------------------------------------------
    # Chords, etc.
    # ----------------------------------------------------------------------------------

    def annotate_chords(self):
        global_annots = self._score.annotations[GLOBAL]
        if "chords" in global_annots:
            LOGGER.warning("chords already appear to be annotated, skipping")
            return
        chord_track = TRACKS["abstract"][GLOBAL]
        for chord in self._score.chords:
            global_annots["chords"][chord.onset] = Annotation(
                chord.onset, chord.token, track=chord_track
            )

    @property
    def current_chord(self) -> Chord:
        return self._score.chords[self.i]

    @property
    def prev_chord(self) -> Chord:
        assert self.i > 0
        return self._score.chords[self.i - 1]

    @property
    def next_chord(self) -> Chord | None:
        if self.i + 1 >= len(self._score.chords):
            return None
        return self._score.chords[self.i + 1]

    @property
    def current_scale(self) -> Scale:
        return self._score.scales[self.i]

    @property
    def prev_scale(self) -> Scale:
        assert self.i > 0
        return self._score.scales[self.i - 1]

    @property
    def next_scale(self) -> Scale | None:
        if self.i + 1 >= len(self._score.scales):
            return None
        return self._score.scales[self.i + 1]

    @property
    def prev_foot_pc(self) -> PitchClass:
        """The "notional" bass pitch-class.

        It is "notional" because the pitch-class of the *actual* bass may be a
        suspension, etc. The "foot" differs from the "root" in that it isn't necessarily
        the root. E.g., in a V6 chord in C major, with a bass suspension C--B, on the C,
        the foot is the third B.
        """
        if self.i < 1:
            raise ValueError()
        return self._score.pc_bass[self.i - 1]

    @property
    def current_foot_pc(self) -> PitchClass:
        """The "notional" bass pitch-class.

        It is "notional" because the pitch-class of the *actual* bass may be a
        suspension, etc. The "foot" differs from the "root" in that it isn't necessarily
        the root. E.g., in a V6 chord in C major, with a bass suspension C--B, on the C,
        the foot is the third B.
        """
        return self._score.pc_bass[self.i]

    @property
    def next_foot_pc(self) -> PitchClass | None:
        """The "notional" bass pitch-class.

        It is "notional" because the pitch-class of the *actual* bass may be a
        suspension, etc. The "foot" differs from the "root" in that it isn't necessarily
        the root. E.g., in a V6 chord in C major, with a bass suspension C--B, on the C,
        the foot is the third B.
        """
        if self.i + 1 >= len(self._score.chords):
            return None
        return self._score.pc_bass[self.i + 1]

    # ----------------------------------------------------------------------------------
    # Pitches and intervals
    # ----------------------------------------------------------------------------------

    def prev_pitch(self, voice: Voice, offset_from_current: int = 1) -> Pitch:
        if self.i < offset_from_current:
            raise ValueError
        return self._score._structural[voice][self.i - offset_from_current]

    def current_pitch(self, voice: Voice) -> Pitch | None:
        try:
            return self._score._structural[voice][self.i]
        except IndexError:
            return None

    def prev_pitches(self) -> tuple[Pitch]:
        assert self.i > 0
        out = (pitches[self.i - 1] for pitches in self._score._structural.values())
        return tuple(sorted(out))

    def prev_structural_interval_above_bass(self, voice: Voice) -> ScalarInterval:
        if voice is OuterVoice.BASS:
            return 0
        return self.prev_scale.get_reduced_scalar_interval(
            self.prev_pitch(OuterVoice.BASS), self.prev_pitch(voice)
        )

    # ----------------------------------------------------------------------------------
    # Inspect suspensions
    # ----------------------------------------------------------------------------------

    def prev_suspension(
        self, voice: Voice  # , offset_from_current: int = 1
    ) -> Suspension | None:
        # if self.i < offset_from_current:
        #     raise ValueError
        if self.i < 1:
            raise ValueError()
        return self._score._suspensions[voice].get(self.prev_chord.onset, None)
        # return self._score._suspensions[voice].get(self.i - offset_from_current, None)

    def prev_is_suspension(self, voice: Voice) -> bool:
        return self.prev_suspension(voice) is not None

    def current_suspension(self, voice: Voice) -> Suspension | None:
        # return self._score._suspensions[voice].get(self.i, None)
        return self._score._suspensions[voice].get(self.current_chord.onset, None)

    def current_is_suspension(self, voice: Voice) -> bool:
        return self.current_suspension(voice) is not None

    def suspension_in_any_voice(self) -> bool:
        for suspensions in self._score._suspensions.values():
            if self.current_chord.onset in suspensions:
                return True
        return False

    # ----------------------------------------------------------------------------------
    # Suspension resolutions
    # ----------------------------------------------------------------------------------

    def prev_resolution(self, voice: Voice) -> Pitch | None:
        assert self.i > 0
        return self._score._suspension_resolutions[voice].get(
            self.prev_chord.onset, None
        )
        # return self._score._suspension_resolutions[voice].get(self.i - 1, None)

    def prev_is_resolution(self, voice: Voice) -> bool:
        return self.prev_resolution(voice) is not None

    def current_resolution(self, voice: Voice) -> Pitch | None:
        return self._score._suspension_resolutions[voice].get(
            self.current_chord.onset, None
        )
        # return self._score._suspension_resolutions[voice].get(self.i, None)

    def all_current_resolutions(self) -> dict[Voice, Pitch | None]:
        out = {}
        for voice, suspensions in self._score._suspension_resolutions.items():
            out[voice] = suspensions.get(self.current_chord.onset, None)
        return out

    def current_is_resolution(self, voice: Voice) -> bool:
        return self.current_resolution(voice) is not None

    def current_resolution_pcs(self):
        out = set()
        for voice in self._score._suspension_resolutions:
            if self.i in self._score._suspension_resolutions[voice]:
                out.add(self._score._suspension_resolutions[voice][self.i] % 12)
        return out

    def resolution_in_any_voice(self, time: TimeStamp) -> bool:
        for resolutions in self._score._suspension_resolutions.values():
            if time in resolutions:
                return True
        return False

    # ----------------------------------------------------------------------------------
    # Suspension preparations
    # ----------------------------------------------------------------------------------

    def prev_is_preparation(self, voice: Voice) -> bool:
        suspension = self.current_suspension
        if suspension is None:
            return False
        return self._score._structural[voice] == suspension.pitch

    # ----------------------------------------------------------------------------------
    # Alter chords
    # ----------------------------------------------------------------------------------
    def split_current_chord_at(self, time: TimeStamp):
        LOGGER.debug(
            f"Splitting chord at {time=} w/ duration "
            f"{self.current_chord.release - self.current_chord.onset}"
        )
        self._score.split_ith_chord_at(self.i, time)
        LOGGER.debug(
            f"After split chord has duration "
            f"{self.current_chord.release - self.current_chord.onset}"
        )

    def merge_current_chords_if_they_were_previously_split(self):
        if self._score._split_chords[self.i]:
            assert self.next_chord is not None
            assert is_same_harmony(self.current_chord, self.next_chord)
            self._score.merge_ith_chords(self.i)

    # ----------------------------------------------------------------------------------
    # Add and remove suspensions
    # ----------------------------------------------------------------------------------

    def _add_suspension(self, voice: Voice, suspension: Suspension, onset: TimeStamp):
        suspensions = self._score._suspensions[voice]
        assert onset not in suspensions
        suspensions[onset] = suspension

    def _remove_suspension(self, voice: Voice, onset: TimeStamp):
        suspensions = self._score._suspensions[voice]
        suspensions.pop(onset)  # raises KeyError if missing

    def _annotate_suspension(self, voice: Voice, onset: TimeStamp):
        annots = self._score.annotations[voice]["suspensions"]
        annots[onset] = Annotation(onset, "S", track=TRACKS["structural"][voice])

    def _remove_suspension_annotation(self, voice: Voice, onset: TimeStamp):
        annots = self._score.annotations[voice]["suspensions"]
        annots.pop(onset)

    def _add_suspension_resolution(
        self, voice: Voice, pitch: Pitch, release: TimeStamp
    ):
        assert not release in self._score._suspension_resolutions[voice]
        self._score._suspension_resolutions[voice][release] = pitch

    def _remove_suspension_resolution(self, voice: Voice, release: TimeStamp):
        assert release in self._score._suspension_resolutions[voice]
        del self._score._suspension_resolutions[voice][release]

    def apply_suspension(
        self,
        suspension: Suspension,
        suspension_release: TimeStamp,
        voice: Voice,
        annotate: bool = True,
    ) -> Pitch:
        LOGGER.debug(f"applying {suspension=} with {suspension_release=} at {self.i=}")
        # if self.i >= 10:
        #     breakpoint()
        if suspension_release < self.current_chord.release:
            # If the suspension resolves during the current chord, we need to split
            #   the current chord to permit that
            self.split_current_chord_at(suspension_release)
        else:
            # Otherwise, make sure the suspension resolves at the onset of the
            #   next chord
            assert (
                self.next_chord is not None
                and suspension_release == self.next_chord.onset
            )
        # TODO: (Malcolm 2023-08-10) I don't know why we were calculating the suspended
        #   pitch like this rather than retrieving it directly from the pitch attribute
        #   of the suspension
        # suspended_pitch = self.prev_pitch(voice)
        suspended_pitch = suspension.pitch
        self._add_suspension_resolution(
            voice, suspended_pitch + suspension.resolves_by, suspension_release
        )
        self._add_suspension(voice, suspension, self.current_chord.onset)
        if annotate:
            self._annotate_suspension(voice, self.current_chord.onset)
        return suspended_pitch

    def apply_suspensions(
        self,
        suspension_combo: SuspensionCombo,
        suspension_release: TimeStamp,
        annotate: bool = True,
    ):
        LOGGER.debug(f"applying {suspension_combo=} at {self.i=}")
        for voice, suspension in suspension_combo.items():
            self.apply_suspension(
                suspension,
                suspension_release=suspension_release,
                voice=voice,
                annotate=annotate,
            )

    def undo_suspension(
        self, voice: Voice, suspension_release: TimeStamp, annotate: bool = True
    ) -> None:
        LOGGER.debug(f"undoing suspension in {voice=} at {self.i=}")
        self._remove_suspension_resolution(voice, suspension_release)
        # TODO: (Malcolm 2023-08-10) double check this is the right onset
        self._remove_suspension(voice, self.current_chord.onset)
        if annotate:
            self._remove_suspension_annotation(voice, self.current_chord.onset)
        # if not self.suspension_in_any_voice():
        if not self.resolution_in_any_voice(self.current_chord.release):
            self.merge_current_chords_if_they_were_previously_split()

    def undo_suspensions(
        self,
        suspension_release: TimeStamp,
        suspension_combo: SuspensionCombo,
        annotate: bool = True,
    ):
        LOGGER.debug(f"undoing {suspension_combo=} at {self.i=}")
        for voice in suspension_combo:
            self.undo_suspension(voice, suspension_release, annotate)

    # TODO: (Malcolm 2023-08-04) prev_structural_interval_above_bass?

    def at_chord_change(
        self,
        compare_scales: bool = True,
        compare_inversions: bool = True,
        allow_subsets: bool = False,
    ) -> bool:
        return self.empty or not is_same_harmony(
            self.prev_chord,
            self.current_chord,
            compare_scales,
            compare_inversions,
            allow_subsets,
        )
