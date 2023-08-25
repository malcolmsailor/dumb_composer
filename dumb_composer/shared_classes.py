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

from dumb_composer.chords.chords import (
    Allow,
    Chord,
    get_chords_from_rntxt,
    is_same_harmony,
)
from dumb_composer.constants import TRACKS
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
    Suspension,
    TimeStamp,
    Voice,
)
from dumb_composer.suspensions import SuspensionCombo
from dumb_composer.time import Meter
from dumb_composer.utils.df_helpers import sort_note_df
from dumb_composer.utils.iterables import flatten_iterables

LOGGER = logging.getLogger(__name__)


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
