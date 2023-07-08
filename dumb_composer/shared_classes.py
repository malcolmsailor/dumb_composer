from collections import defaultdict, deque
from functools import cached_property
import logging
import math
from numbers import Number
import re
import textwrap
import pandas as pd
import typing as t

from .time import Meter

from .utils.df_helpers import sort_note_df

from .pitch_utils.scale import Scale, ScaleDict

from .pitch_utils.put_in_range import put_in_range

from .pitch_utils.chords import get_chords_from_rntxt, Allow, is_same_harmony
from dumb_composer.pitch_utils.chords import Chord


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

    >>> print_notes([
    ...     Note(pitch=48, onset=0, release=1),
    ...     Note(pitch=52, onset=1, release=2),
    ...     Note(pitch=55, onset=1, release=2),
    ... ])
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
    >>> len(apply_ties(
    ...     [Note(60, 0.0, 1.0, tie_to_next=True), Note(60, 1.0, 2.0)])
    ... )
    1
    >>> len(apply_ties(
    ...     [Note(60, 0.0, 1.0, tie_to_next=True),
    ...      Note(60, 1.0, 2.0, tie_to_next=True),
    ...      Note(60, 2.0, 3.0),
    ...      Note(60, 3.0, 4.0, tie_to_next=True),
    ...      Note(60, 4.0, 5.0)]
    ... ))
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
    ...     [Note(60, 0.0, 1.0, tie_to_next=True),
    ...      Note(60, 1.0, 2.0, tie_to_next=True),
    ...      Note(60, 2.5, 3.0)],
    ...     check_correctness=True,
    ... )
    Traceback (most recent call last):
    ValueError: Release of note at 2.0 != onset of note at 2.5

    >>> apply_ties(
    ...     [Note(60, 0.0, 1.0, tie_to_next=True),
    ...      Note(60, 1.0, 2.0),
    ...      Note(60, 2.5, 3.0, tie_to_next=True),
    ...      Note(62, 3.0, 4.0)],
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
    def __init__(self, scale_pcs: t.Iterable[t.Sequence[int]]):
        self._scale_pcs = list(scale for scale in scale_pcs)
        self._scales = ScaleDict()

    def __len__(self) -> int:
        return len(self._scale_pcs)

    def __getitem__(self, idx: int) -> Scale:
        return self._scales[self._scale_pcs[idx]]

    def insert_scale_pcs(self, i: int, scale_pcs: t.Sequence[int]) -> None:
        self._scale_pcs.insert(i, scale_pcs)

    def pop_scale_pcs(self, i: int) -> t.Sequence[int]:
        return self._scale_pcs.pop(i)


class StructuralMelodyIntervalGetter:
    def __init__(
        self,
        scales: ScaleGetter,
        structural_melody: t.List[int],
        structural_bass: t.List[int],
    ):
        # This class doesn't really have custody of its attributes, which
        #   should all be attributes of the Score that creates it. The only
        #   reason this class exists is to that we can provide a [] subscript
        #   syntax to Score.structural_melody_intervals
        self.scales = scales
        self.structural_melody = structural_melody
        self.structural_bass = structural_bass
        # self._melody_intervals: t.List[int] = []

    def __getitem__(self, idx):
        return self.scales[idx].get_interval_class(
            self.structural_bass[idx], self.structural_melody[idx]
        )
        # TODO remove---I don't think caching is practicable once we start
        #   splitting chords in the Score
        # try:
        #     return self._melody_intervals[idx]
        # except IndexError:
        #     for i in range(len(self._melody_intervals), idx + 1):
        #         self._melody_intervals.append(
        #             self.scales[i].get_interval_class(
        #                 self.structural_bass[i], self.structural_melody[i]
        #             )
        #         )
        #     return self._melody_intervals[idx]


class Score:
    """This class provides a "shared working area" for the various classes and
    functions that build a score. It doesn't encapsulate much of anything.

    >>> rntxt = "m1 C: I b2 ii6 b3 V b4 I6"
    >>> score = Score(chord_data=rntxt)

    >>> [chord.pcs for chord in score.chords]
    [(0, 4, 7), (5, 9, 2), (7, 11, 2), (4, 7, 0)]

    >>> score.structural_bass
    [36, 41, 31, 40]
    """

    def __init__(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        bass_range: t.Tuple[int, int] = (30, 50),
        mel_range: t.Tuple[int, int] = (60, 78),
        prefab_track: int = 1,
        bass_track: int = 2,
        accompaniments_track: int = 3,
        ts: t.Optional[t.Union[Meter, str]] = None,
        transpose: int = 0,
    ):
        if isinstance(chord_data, str):
            logging.debug(f"reading chords from {chord_data}")
            chord_data, ts = get_chords_from_rntxt(chord_data)
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
        self.bass_range = bass_range
        self.mel_range = mel_range
        self.structural_melody: t.List[int] = []
        self._structural_melody_interval_getter = StructuralMelodyIntervalGetter(
            self.scales, self.structural_melody, self.structural_bass
        )
        self.prefabs: t.List[t.List[Note]] = []
        self.accompaniments: t.List[t.List[Note]] = []
        self.annotations: defaultdict[str, t.List[Annotation]] = defaultdict(list)
        self.prefab_track = prefab_track
        self.bass_track = bass_track
        self.accompaniments_track = accompaniments_track
        self.suspension_indices: t.Set[int] = set()
        self.tied_prefab_indices: t.Set[int] = set()
        self.allow_prefab_start_with_rest: t.Dict[int, Allow] = {}

    @cached_property
    def structural_bass(self) -> t.List[int]:
        out = list(
            put_in_range((chord.foot for chord in self._chords), *self.bass_range)
        )
        logging.debug(
            textwrap.fill(
                f"Initializing {self.__class__.__name__}.structural_bass: {out}",
                subsequent_indent=" " * 4,
            )
        )
        return out

    @property
    def chords(self) -> t.List[Chord]:
        return self._chords

    @chords.setter
    def chords(self, new_chords: t.List[Chord]):
        self._chords = new_chords
        # deleting self.structural_bass allows it to be regenerated next
        #   time it is needed
        del self.structural_bass
        self._scale_getter = ScaleGetter(chord.scale_pcs for chord in self._chords)
        self._structural_melody_interval_getter = StructuralMelodyIntervalGetter(
            self.scales, self.structural_melody, self.structural_bass
        )

    @property
    def scales(self) -> ScaleGetter:
        return self._scale_getter

    @property
    def structural_melody_intervals(self) -> StructuralMelodyIntervalGetter:
        return self._structural_melody_interval_getter

    @property
    def structural_bass_as_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            Note(
                bass_pitch,
                self.chords[i].onset,
                self.chords[i].release,
                track=self.bass_track,
            )
            for i, bass_pitch in enumerate(self.structural_bass)
        )

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

    def get_existing_pitches(
        self,
        idx: int,
        attr_names: t.Sequence[str] = ("structural_bass", "structural_melody"),
    ):
        return tuple(
            getattr(self, attr_name)[idx]
            for attr_name in attr_names
            if len(getattr(self, attr_name)) > idx
        )

    def get_df(self, contents: t.Union[str, t.Sequence[str]]) -> pd.DataFrame:
        if isinstance(contents, str):
            contents = [contents]
        dfs = (getattr(self, f"{name}_as_df") for name in contents)
        df = pd.concat(dfs)
        return sort_note_df(df)

    def split_ith_chord_at(
        self, i: int, time: Number, check_correctness: bool = True
    ) -> None:
        """
        >>> rntxt = '''Time Signature: 4/4
        ... m1 C: I
        ... m2 V
        ... m3 I'''
        >>> score = Score(rntxt)
        >>> {float(chord.onset): chord.token for chord in score.chords}
        {0.0: 'C:I', 4.0: 'V', 8.0: 'I'}
        >>> score.split_ith_chord_at(1, 6.0)
        >>> {float(chord.onset): chord.token for chord in score.chords}
        {0.0: 'C:I', 4.0: 'V', 6.0: 'V', 8.0: 'I'}
        >>> score.merge_ith_chords(1)
        >>> {float(chord.onset): chord.token for chord in score.chords}
        {0.0: 'C:I', 4.0: 'V', 8.0: 'I'}
        """
        # just in case structural_bass has not been computed yet, we need
        #   to compute it now:
        self.structural_bass
        # TODO make debug flag for check_correctness
        chord = self.chords[i]
        if check_correctness:
            assert chord.onset < time < chord.release
            assert len(self.prefabs) <= i
            assert len(self.accompaniments) <= i
        new_chord = chord.copy()
        chord.release = time
        new_chord.onset = time
        self.chords.insert(i + 1, new_chord)
        self.structural_bass.insert(i + 1, self.structural_bass[i])
        if len(self.structural_melody) > i:
            self.structural_melody.insert(i + 1, self.structural_melody[i])
        self._scale_getter.insert_scale_pcs(i + 1, new_chord.scale_pcs)

    def merge_ith_chords(self, i: int, check_correctness: bool = True) -> None:
        # just in case structural_bass has not been computed yet, we need
        #   to compute it now:
        self.structural_bass
        # TODO make debug flag for check_correctness
        chord1, chord2 = self.chords[i : i + 2]
        chord1.release = chord2.release
        if check_correctness:
            chord2.onset = chord1.onset
            assert chord1 == chord2
        self.chords.pop(i + 1)
        self.structural_bass.pop(i + 1)
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
        >>> score = Score(rntxt)
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
