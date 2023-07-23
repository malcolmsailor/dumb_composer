import collections.abc
import logging
import math
import re
import textwrap
import typing as t
from abc import abstractmethod
from collections import defaultdict, deque
from enum import Enum
from functools import cached_property
from numbers import Number

import pandas as pd

from dumb_composer.constants import TIME_TYPE
from dumb_composer.pitch_utils.chords import (
    Allow,
    Chord,
    get_chords_from_rntxt,
    is_same_harmony,
)
from dumb_composer.pitch_utils.music21_handler import get_ts_from_rntxt
from dumb_composer.pitch_utils.spacings import RangeConstraints
from dumb_composer.pitch_utils.types import Pitch, PitchClass, TimeStamp
from dumb_composer.suspensions import Suspension
from dumb_composer.utils.iterables import flatten_iterables

from .pitch_utils.put_in_range import put_in_range
from .pitch_utils.scale import Scale, ScaleDict
from .time import Meter
from .utils.df_helpers import sort_note_df


class InnerVoice(Enum):
    TENOR = 2
    ALTO = 3


class Annotation(pd.Series):
    def __init__(self, onset, text):
        # TODO remove the next lines when I've figured out how to get
        #   text annotations to display correctly
        text = text.replace("_", "").replace(" ", "")
        super().__init__(
            {"onset": onset, "text": text, "type": "text", "track": 0}  # type:ignore
        )


class Note(pd.Series):
    def __init__(
        self,
        pitch: int,
        onset: Number,
        release: Number,
        track: int = 1,
        tie_to_next: bool = False,
    ):
        super().__init__(
            {  # type:ignore
                "pitch": pitch,
                "onset": onset,
                "release": release,
                "type": "note",
                "track": track,
                "tie_to_next": tie_to_next,
            }
        )

    def __str__(self):
        return f"{self.pitch}[{self.onset}-{self.release}]"


def notes(
    pitches: t.Sequence[int], onset: Number, release: Number, track: int = 1
) -> t.List[Note]:
    return [
        Note(pitch, onset, release, track=track) for pitch in pitches  # type:ignore
    ]


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
    df = pd.DataFrame(notes)
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
        structural_melody: t.List[int],
        structural_bass: t.List[int],
    ):
        # This class doesn't really have custody of its attributes, which
        #   should all be attributes of the PrefabScore that creates it. The only
        #   reason this class exists is to that we can provide a [] subscript
        #   syntax to PrefabScore.structural_melody_intervals
        self.scales = scales
        self.structural_melody = structural_melody
        self.structural_bass = structural_bass

    def __getitem__(self, idx):
        return self.scales[idx].get_reduced_scalar_interval(
            self.structural_bass[idx], self.structural_melody[idx]
        )

    def __len__(self):
        return min(len(self.structural_bass), len(self.structural_melody))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class _ScoreBase:
    def __init__(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        range_constraints: RangeConstraints = RangeConstraints(),
        ts: Meter | str | None = None,
        transpose: int = 0,
        melody_track: int = 1,
        bass_track: int = 2,
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
        self.range_constraints = range_constraints
        self.structural_bass: t.List[Pitch] = []
        self.structural_melody: t.List[Pitch] = []
        self.melody_suspensions: t.Dict[int, Suspension] = {}
        self.bass_suspensions: t.Dict[int, Suspension] = {}
        self.annotations: defaultdict[str, t.List[Annotation]] = defaultdict(list)
        self.misc: dict[str, t.Any] = {}

        self._structural_melody_interval_getter = StructuralMelodyIntervals(
            self.scales, self.structural_melody, self.structural_bass
        )

        self.melody_track = melody_track
        self.bass_track = bass_track

    @cached_property
    def pc_bass(self) -> t.List[int]:
        return [chord.foot for chord in self._chords]  # type:ignore

    @property
    def structural_melody_intervals(self) -> StructuralMelodyIntervals:
        return self._structural_melody_interval_getter

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
        self._scale_getter = ScaleGetter(chord.scale_pcs for chord in self._chords)
        self._structural_melody_interval_getter = StructuralMelodyIntervals(
            self.scales, self.structural_melody, self.structural_bass
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
        if len(self.structural_melody) > i:
            self.structural_melody.insert(i + 1, self.structural_melody[i])
        self._scale_getter.insert_scale_pcs(i + 1, new_chord.scale_pcs)

    def merge_ith_chords(self, i: int, check_correctness: bool = True) -> None:
        """
        Merge score.chords[i] and score.chords[i + 1]
        """
        # just in case pc_bass has not been computed yet, we need
        #   to compute it now:
        self.pc_bass
        # TODO make debug flag for check_correctness
        chord1, chord2 = self.chords[i : i + 2]
        chord1.release = chord2.release
        if check_correctness:
            chord2.onset = chord1.onset
            assert chord1 == chord2
        self.chords.pop(i + 1)
        self.pc_bass.pop(i + 1)
        if len(self.structural_melody) > i + 2:
            self.structural_melody.pop(i + 1)
        self._scale_getter.pop_scale_pcs(i + 1)

    def is_chord_change(
        self,
        i,
        compare_scales: bool = True,
        compare_inversions: bool = True,
        allow_subsets: bool = False,
    ) -> bool:
        """
        >>> rntxt = '''Time Signature: 4/4
        ... m1 I b4 V
        ... m2 V'''
        >>> score = PrefabScore(rntxt)
        >>> score.is_chord_change(0)
        True
        >>> score.is_chord_change(1)
        True
        >>> score.is_chord_change(2)
        False
        """
        return (i == 0) or not is_same_harmony(
            self.chords[i - 1],
            self.chords[i],
            compare_scales,
            compare_inversions,
            allow_subsets,
        )

    @property
    def annotations_as_df(self) -> pd.DataFrame:
        # return pd.DataFrame(self.annotations)
        out = pd.concat([pd.DataFrame(annots) for annots in self.annotations.values()])
        # only one annotation per time-point appears in the kern files (or is it
        #   the verovio realizations?). Anyway, we merge them into one here. TODO
        #   is there a way around this constraint?
        temp = []
        for onset in sorted(out.onset.unique()):
            new_annot = Annotation(
                onset,
                "".join(annot.text for _, annot in out[out.onset == onset].iterrows()),
            )
            temp.append(new_annot)
        out = pd.DataFrame(temp)
        return out

    @property
    def structural_bass_as_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            Note(  # type:ignore
                bass_pitch,
                self.chords[i].onset,
                self.chords[i].release,
                track=self.bass_track,
            )
            for i, bass_pitch in enumerate(self.structural_bass)
        )

    @property
    def structural_melody_as_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            Note(  # type:ignore
                melody_pitch,
                self.chords[i].onset,
                self.chords[i].release,
                track=self.melody_track,
            )
            for i, melody_pitch in enumerate(self.structural_melody)
        )

    def get_df(self, contents: t.Union[str, t.Sequence[str]]) -> pd.DataFrame:
        if isinstance(contents, str):
            contents = [contents]
        dfs = (getattr(self, f"{name}_as_df") for name in contents)
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
        return "structural_bass", "structural_melody"  # type:ignore


class PrefabScore(_ScoreBase):
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
        range_constraints: RangeConstraints = RangeConstraints(),
        prefab_track: int = 1,
        melody_track: int = 2,
        bass_track: int = 3,
        accompaniments_track: int = 2,
        ts: t.Optional[t.Union[Meter, str]] = None,
        transpose: int = 0,
    ):
        super().__init__(
            chord_data,
            range_constraints=range_constraints,
            ts=ts,
            transpose=transpose,
            melody_track=melody_track,
            bass_track=bass_track,
        )

        self.prefabs: t.List[t.List[Note]] = []
        self.accompaniments: t.List[t.List[Note]] = []
        self.prefab_track = prefab_track
        self.accompaniments_track = accompaniments_track
        self.tied_prefab_indices: t.Set[int] = set()
        self.allow_prefab_start_with_rest: t.Dict[int, Allow] = {}

    @property
    def default_existing_pitch_attr_names(self) -> t.Tuple[str]:
        return "structural_bass", "structural_melody"  # type:ignore

    @property
    def prefabs_as_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            apply_ties(
                (note for prefab in self.prefabs for note in prefab),
                check_correctness=True,  # TODO remove this when I'm more confident
            )
        )

    @property
    def accompaniments_as_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            note for accompaniment in self.accompaniments for note in accompaniment
        )

    def _validate_split_ith_chord_at(self, i: int, chord: Chord, time: TIME_TYPE):
        super()._validate_split_ith_chord_at(i, chord, time)
        assert len(self.prefabs) <= i
        assert len(self.accompaniments) <= i


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
    >>> score.structural_melody.extend([72, 74, 71, 72])
    >>> score.get_existing_pitches(0)
    (36, 72)
    >>> score.get_existing_pitches(1)
    (41, 74)

    Let's add inner parts:
    >>> score.inner_voices[InnerVoice.TENOR].extend([55, 57, 55, 55])
    >>> score.inner_voices[InnerVoice.ALTO].extend([64, 65, 62, 77])
    >>> score.get_existing_pitches(0)  # TODO: (Malcolm 2023-07-22) sort?
    (36, 72, 55, 64)
    """

    def __init__(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        range_constraints: RangeConstraints = RangeConstraints(),
        ts: t.Optional[t.Union[Meter, str]] = None,
        transpose: int = 0,
        melody_track: int = 1,
        bass_track: int = 3,
        inner_voices_track: int = 2,
    ):
        super().__init__(
            chord_data,
            range_constraints=range_constraints,
            ts=ts,
            transpose=transpose,
            melody_track=melody_track,
            bass_track=bass_track,
        )
        # self.inner_voices: t.List[t.Tuple[Pitch, Pitch]] = []
        self.inner_voices: defaultdict[InnerVoice, list[Pitch]] = defaultdict(list)
        self.inner_voices_track = inner_voices_track
        self.inner_voice_suspensions: defaultdict[
            InnerVoice, dict[int, Suspension]
        ] = defaultdict(lambda: {})

    def get_existing_pitches(
        self,
        idx: int,
        attr_names: t.Sequence[str] | None = None,
    ) -> t.Tuple[Pitch]:
        if attr_names is None:
            attr_names = self.default_existing_pitch_attr_names

        # Quite a hacky solution
        if "inner_voices" in attr_names:
            attr_names = [
                attr_name for attr_name in attr_names if attr_name != "inner_voices"
            ]
            inner_voices_pitches = [
                inner_voice_pitches[idx]
                for inner_voice_pitches in self.inner_voices.values()
            ]
        else:
            inner_voices_pitches = []

        return tuple(
            flatten_iterables(
                [super().get_existing_pitches(idx, attr_names), inner_voices_pitches]
            )
        )

    @property
    def default_existing_pitch_attr_names(self) -> t.Tuple[str]:
        return "structural_bass", "inner_voices", "structural_melody"  # type:ignore

    @property
    def inner_voices_as_df(self) -> pd.DataFrame:
        notes = []
        for chord_i, inner_voices in enumerate(zip(*self.inner_voices.values())):
            for pitch in inner_voices:
                notes.append(
                    Note(  # type:ignore
                        pitch,
                        self.chords[chord_i].onset,
                        self.chords[chord_i].release,
                        track=self.inner_voices_track,
                    )
                )
        return pd.DataFrame(notes)
