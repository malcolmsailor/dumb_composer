import logging
import random
import typing as t
from abc import abstractmethod
from enum import Enum, auto
from fractions import Fraction
from numbers import Number
from re import M
from types import MappingProxyType

import pandas as pd

from dumb_composer.chord_spacer import SpacingConstraints
from dumb_composer.pitch_utils.spacings import RangeConstraints
from dumb_composer.pitch_utils.types import Pitch, TimeStamp
from dumb_composer.shared_classes import Allow, Note, notes, print_notes
from dumb_composer.time import Meter, RhythmFetcher

LOGGER = logging.getLogger(__name__)

Pattern = t.Callable[..., list[Note]]


class ExtraPitches(Enum):
    BOTTOM = auto()
    MID = auto()
    TOP = auto()


def pattern_method(
    requires_bass: bool = False,
    allow_compound: Allow = Allow.YES,
    allow_triple: Allow = Allow.YES,
    min_dur: t.Optional[TimeStamp] = None,
    min_dur_fallback: str = "simple_chord",
    min_pitch_count: t.Optional[int] = None,
    min_pitch_count_fallback: str = "simple_chord",
    spacing_constraints: SpacingConstraints = SpacingConstraints(),
    range_constraints: RangeConstraints = RangeConstraints(),
    total_voice_count: int = 4,
):
    """
    Keyword args:
        min_dur: denominated in 'beats'
    """

    def wrap(f):
        setattr(f, "pattern_method", True)
        setattr(f, "requires_bass", requires_bass)
        setattr(f, "allow_compound", allow_compound)
        setattr(f, "allow_triple", allow_triple)
        setattr(f, "min_dur", min_dur)
        setattr(f, "min_dur_fallback", min_dur_fallback)
        setattr(f, "min_pitch_count", min_pitch_count)
        setattr(f, "min_pitch_count_fallback", min_pitch_count_fallback)
        setattr(f, "spacing_constraints", spacing_constraints)
        setattr(f, "range_constraints", range_constraints)
        setattr(f, "total_voice_count", total_voice_count)
        return f

    return wrap


# TODO: (Malcolm 2023-08-08) it would probably be better to rewrite all the
#   `patternmethod`s below as classes deriving from PatternBase as below.
# class PatternFallBack(Exception):
#     def __init__(self, fall_back_to: str, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.fall_back_to = fall_back_to


# class PatternBase:
#     requires_bass: bool = False
#     allow_compound: Allow = Allow.YES
#     allow_triple: Allow = Allow.YES
#     min_dur: TimeStamp | None = None
#     min_dur_fallback: str = "SimpleChord"
#     min_pitch_count: t.Optional[int] = None
#     min_pitch_count_fallback: str = "SimpleChord"
#     spacing_constraints: SpacingConstraints = field(default_factory=SpacingConstraints)()

#     @abstractmethod
#     @staticmethod
#     def _realize(
#         pitches: t.Sequence[int],
#         onset: TimeStamp,
#         release: TimeStamp,
#         track: int,
#         chord_change: bool,
#     ):
#         raise NotImplementedError()

#     @classmethod
#     def __call__(
#         cls,
#         pitches: t.Sequence[Pitch],
#         harmony_onset: TimeStamp,
#         harmony_release: TimeStamp,
#         onset: TimeStamp,
#         release: TimeStamp,
#         ts: Meter,
#         track: int,
#         chord_change: bool,
#     ):
#         if cls.min_dur is not None and (
#             (harmony_release - harmony_onset) / ts.beat_dur < cls.min_dur
#         ):
#             raise PatternFallBack(cls.min_dur_fallback)

#         if cls.min_pitch_count is not None and len(pitches) < cls.min_pitch_count:
#             raise PatternFallBack(cls.min_pitch_count_fallback)


# class SimpleChord(PatternBase):
#     min_dur = None
#     min_pitch_count = None

#     @staticmethod
#     def _realize(
#         pitches: t.Sequence[int],
#         onset: TimeStamp,
#         release: TimeStamp,
#         track: int,
#         chord_change: bool,
#     ):
#         return [Note(pitch, onset, release, track=track) for pitch in pitches]


class PatternMaker:
    def __init__(
        self,
        ts: t.Union[Meter, str],
        inertia: float = 0.75,
        include_bass: bool = True,
        pattern_changes_on_downbeats_only: bool = True,
    ):
        """Makes patterns from chords.

        Args:
            ts: str. Time signature, e.g., "4/4".

        Keyword args:
            inertia: probability of keeping same pattern at each chord change,
                as opposed to choosing a new one. (Though note that we pick at
                random so we can choose the same pattern twice in a row.)
                Default 0.75.
        """
        if isinstance(ts, str):
            ts = Meter(ts)
        self._ts = ts
        self._patterns = self._filter_patterns(ts, include_bass)
        self.rf = RhythmFetcher(ts)
        self._inertia = inertia
        self._prev_pattern: Pattern | None = None
        self._downbeats_only = pattern_changes_on_downbeats_only
        self._memo = {}

    def _filter_patterns(self, ts: Meter, include_bass: bool) -> t.List[str]:
        out = []
        for pattern_name in self._all_patterns:
            pattern_method = self.get_pattern_method_from_name(pattern_name)
            go_on = False
            for attr, constraint in (
                (ts.is_compound, pattern_method.allow_compound),
                (ts.is_triple, pattern_method.allow_triple),
            ):
                if attr and constraint is Allow.NO:
                    go_on = True
                    break
                elif not attr and constraint is Allow.ONLY:
                    go_on = True
                    break
            if go_on:
                continue
            if not include_bass and pattern_method.requires_bass:
                continue
            out.append(pattern_name)
        return out

    @pattern_method(requires_bass=True, min_dur=TimeStamp(2))
    def oompah(
        self,
        pitches: t.Sequence[int],
        onset: TimeStamp,
        release: TimeStamp,
        track: int,
        chord_change: bool,
    ):
        """
        >>> pm = PatternMaker("4/4")
        >>> print_notes(pm(pitches=[48, 64, 67], onset=0, release=4, pattern="oompah"))
        on off   pitches
        0   1     (48,)
        1   2  (64, 67)
        2   3     (48,)
        3   4  (64, 67)

        In 3/4 gives the same results as `oompahpah`.

        >>> pm = PatternMaker("3/4")
        >>> print_notes(pm(pitches=[48, 64, 67], onset=0, release=3, pattern="oompah"))
        on off   pitches
        0   1     (48,)
        1   2  (64, 67)
        2   3  (64, 67)

        The first beat is always given "oom". This may not be what you want so
        it may be best to avoid using this pattern starting on a weakbeat.

        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=0.5, release=4, pattern="oompah")
        ... )
        on off   pitches
        1   2     (48,)
        2   3     (48,)
        3   4  (64, 67)

        """
        beats = self.rf.beats(onset, release)
        beat_weight = self.rf.beat_weight
        foot, others = pitches[0], pitches[1:]
        out = []
        for i, beat in enumerate(beats):
            if (i == 0 and chord_change) or beat["weight"] > beat_weight:
                out.append(
                    Note(  # type:ignore
                        foot, beat["onset"], beat["release"], track=track
                    )
                )
            else:
                out.extend(notes(others, beat["onset"], beat["release"], track=track))
        return out

    @pattern_method(requires_bass=True, min_dur=TimeStamp(3), min_dur_fallback="oompah")
    def oompahpah(
        self,
        pitches: t.Sequence[int],
        onset: TimeStamp,
        release: TimeStamp,
        track: int,
        chord_change: bool,
    ):
        """
        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=0, release=4, pattern="oompahpah")
        ... )
        on off   pitches
        0   1     (48,)
        1   2  (64, 67)
        2   3  (64, 67)
        3   4  (64, 67)

        In 3/4 gives the same results as `oompah`.
        >>> pm = PatternMaker("3/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=0, release=3, pattern="oompahpah")
        ... )
        on off   pitches
        0   1     (48,)
        1   2  (64, 67)
        2   3  (64, 67)

        The first beat is always given "oom". This may not be what you want so
        it may be best to avoid using this pattern starting on a weakbeat.
        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=0.5, release=4, pattern="oompahpah")
        ... )
        on off   pitches
        1   2     (48,)
        2   3  (64, 67)
        3   4  (64, 67)
        """
        beats = self.rf.beats(onset, release)
        max_weight = self.rf.max_weight
        foot, others = pitches[0], pitches[1:]
        out = []
        for i, beat in enumerate(beats):
            if (i == 0 and chord_change) or beat["weight"] == max_weight:
                out.append(
                    Note(  # type:ignore
                        foot, beat["onset"], beat["release"], track=track
                    )
                )
            else:
                out.extend(notes(others, beat["onset"], beat["release"], track=track))
        return out

    @staticmethod
    def _apply_pitches_to_rhythms(rhythms, pitches, track):
        out = []
        for r in rhythms:
            out.extend(notes(pitches, r["onset"], r["release"], track=track))
        return out

    def _chords(
        self,
        rhythm_f,
        pitches,
        onset,
        release,
        track,
        chord_change: bool = True,
    ):
        rhythms = rhythm_f(onset, release, avoid_empty=chord_change)
        return self._apply_pitches_to_rhythms(rhythms, pitches, track)

    @pattern_method()
    def simple_chord(
        self,
        pitches: t.Sequence[int],
        onset: TimeStamp,
        release: TimeStamp,
        track: int,
        chord_change: bool,
    ):
        return [Note(pitch, onset, release, track=track) for pitch in pitches]

    @pattern_method()
    def off_beat_chords(
        self,
        pitches: t.Sequence[int],
        onset: TimeStamp,
        release: TimeStamp,
        track: int,
        chord_change: bool,
    ):
        """
        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=0, release=4, pattern="off_beat_chords")
        ... )
        on off       pitches
        1   2  (48, 64, 67)
        2   3  (48, 64, 67)
        3   4  (48, 64, 67)

        If the interval between onset and release would not otherwise contain a
        single chord, a chord will be inserted starting at onset:

        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=0, release=1, pattern="off_beat_chords")
        ... )
        on  off       pitches
        0    1  (48, 64, 67)

        This onset extends to release or until the next beat, whichever is sooner.

        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(
        ...         pitches=[48, 64, 67],
        ...         onset=3.5,
        ...         release=5,
        ...         pattern="off_beat_chords",
        ...     )
        ... )
        on off       pitches
        3.5   4  (48, 64, 67)

        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(
        ...         pitches=[48, 64, 67],
        ...         onset=3.5,
        ...         release=3.75,
        ...         pattern="off_beat_chords",
        ...     )
        ... )
        on   off       pitches
        3.5  3.75  (48, 64, 67)
        """
        return self._chords(
            self.rf.off_beats, pitches, onset, release, track, chord_change
        )

    @pattern_method()
    def off_semibeat_chords(
        self,
        pitches: t.Sequence[int],
        onset: TimeStamp,
        release: TimeStamp,
        track: int,
        chord_change: bool,
    ):
        """
        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(
        ...         pitches=[48, 64, 67],
        ...         onset=0,
        ...         release=2,
        ...         pattern="off_semibeat_chords",
        ...     )
        ... )
        on  off       pitches
        1/2    1  (48, 64, 67)
        1  3/2  (48, 64, 67)
        3/2    2  (48, 64, 67)
        """
        return self._chords(
            self.rf.off_semibeats, pitches, onset, release, track, chord_change
        )

    @pattern_method()
    def beat_chords(
        self,
        pitches: t.Sequence[int],
        onset: TimeStamp,
        release: TimeStamp,
        track: int,
        chord_change: bool,
    ):
        out = self._chords(self.rf.beats, pitches, onset, release, track, chord_change)
        return out

    @pattern_method()
    def semibeat_chords(
        self,
        pitches: t.Sequence[int],
        onset: TimeStamp,
        release: TimeStamp,
        track: int,
        chord_change: bool,
    ):
        return self._chords(
            self.rf.semibeats, pitches, onset, release, track, chord_change
        )

    @pattern_method()
    def superbeat_chords(
        self,
        pitches: t.Sequence[int],
        onset: TimeStamp,
        release: TimeStamp,
        track: int,
        chord_change: bool,
    ):
        return self._chords(
            self.rf.superbeats, pitches, onset, release, track, chord_change
        )

    # TODO compound alberti bass

    @pattern_method()
    def sustained_semibeat_arpeggio(
        self,
        pitches: t.Sequence[int],
        onset: TimeStamp,
        release: TimeStamp,
        track: int,
        chord_change: bool,
    ):
        # TODO accommodate chord_change
        onsets = self.rf.next_n("semibeat", onset, len(pitches), release)
        out = []
        for pitch_onset, pitch in zip(onsets, pitches):
            out.append(Note(pitch, pitch_onset, release, track=track))
        return out

    def _tremolo(
        self,
        onset_weight: int,
        pitches: t.Sequence[int],
        onset: TimeStamp,
        release: TimeStamp,
        track: int,
        chord_change: bool,
        extra_pitches_at: ExtraPitches = ExtraPitches.TOP,
    ):
        if extra_pitches_at is ExtraPitches.MID:
            raise ValueError(
                "for tremolo, extra_pitches_at must be either "
                "ExtraPitches.BOTTOM or ExtraPitches.TOP"
            )
        bass_weight = onset_weight + 1
        onsets = self.rf._regularly_spaced_by_weight(onset_weight, onset, release)
        out = []
        if extra_pitches_at is ExtraPitches.BOTTOM:
            for onset_dict in onsets:
                if onset_dict["weight"] >= bass_weight:
                    for pitch in pitches[:-1]:
                        out.append(
                            Note(
                                pitch,
                                onset_dict["onset"],
                                onset_dict["release"],
                                track=track,
                            )
                        )
                else:
                    out.append(
                        Note(
                            pitches[-1],
                            onset_dict["onset"],
                            onset_dict["release"],
                            track=track,
                        )
                    )
        else:
            for onset_dict in onsets:
                if onset_dict["weight"] >= bass_weight:
                    out.append(
                        Note(
                            pitches[0],
                            onset_dict["onset"],
                            onset_dict["release"],
                            track=track,
                        )
                    )
                else:
                    for pitch in pitches[1:]:
                        out.append(
                            Note(  # type:ignore
                                pitch,
                                onset_dict["onset"],
                                onset_dict["release"],
                                track=track,
                            )
                        )
        return out

    @pattern_method(min_dur=Fraction(1, 2))
    def demisemibeat_tremolo(self, *args, **kwargs):
        return self._tremolo(-2, *args, **kwargs)

    @pattern_method(min_dur=Fraction(1, 1))
    def semibeat_tremolo(self, *args, **kwargs):
        return self._tremolo(-1, *args, **kwargs)

    @pattern_method(min_dur=Fraction(2, 1))
    def beat_tremolo(self, *args, **kwargs):
        return self._tremolo(0, *args, **kwargs)

    def _three_pitch_arpeggio_pattern(
        self,
        bass_weight: int,
        mid_weight_increment: int,
        top_weight_increment: int,
        pitches: t.Sequence[int],
        onset: TimeStamp,
        release: TimeStamp,
        track: int,
        chord_change: bool,
        extra_pitches_at: ExtraPitches = ExtraPitches.TOP,
    ):
        mid_weight = bass_weight + mid_weight_increment
        top_weight = bass_weight + top_weight_increment
        onsets = self.rf._regularly_spaced_by_weight(
            min([bass_weight, mid_weight, top_weight]), onset, release
        )
        # bass_weight = weight + 2
        # mid_weight = weight + 1
        # onsets = self.rf._regularly_spaced_by_weight(weight, onset, release)
        if len(pitches) > 3:
            if extra_pitches_at is ExtraPitches.BOTTOM:
                bass_i, mid_i, top_i = (slice(-2), -2, -1)
            elif extra_pitches_at is ExtraPitches.MID:
                bass_i, mid_i, top_i = (0, slice(1, -1), -1)
            else:
                bass_i, mid_i, top_i = (0, 1, slice(2, None))
        else:
            bass_i, mid_i, top_i = range(3)
        bass_has_sounded = False
        out = []
        for onset_dict in onsets:
            if onset_dict["weight"] >= bass_weight or (
                not bass_has_sounded
                and chord_change
                and onset_dict["weight"] == bass_weight - 1
            ):
                if isinstance(bass_i, slice):
                    for pitch in pitches[bass_i]:
                        out.append(
                            Note(  # type:ignore
                                pitch,
                                onset_dict["onset"],
                                onset_dict["release"],
                                track=track,
                            )
                        )
                else:
                    out.append(
                        Note(  # type:ignore
                            pitches[bass_i],
                            onset_dict["onset"],
                            onset_dict["release"],
                            track=track,
                        )
                    )
                bass_has_sounded = True
            elif onset_dict["weight"] == mid_weight:
                if isinstance(mid_i, slice):
                    for pitch in pitches[mid_i]:
                        out.append(
                            Note(  # type:ignore
                                pitch,
                                onset_dict["onset"],
                                onset_dict["release"],
                                track=track,
                            )
                        )
                else:
                    out.append(
                        Note(  # type:ignore
                            pitches[mid_i],
                            onset_dict["onset"],
                            onset_dict["release"],
                            track=track,
                        )
                    )
            else:
                if isinstance(top_i, slice):
                    for pitch in pitches[top_i]:
                        out.append(
                            Note(  # type:ignore
                                pitch,
                                onset_dict["onset"],
                                onset_dict["release"],
                                track=track,
                            )
                        )
                else:
                    out.append(
                        Note(  # type:ignore
                            pitches[top_i],
                            onset_dict["onset"],
                            onset_dict["release"],
                            track=track,
                        )
                    )
        return out

    def _alberti(
        self,
        weight,
        pitches: t.Sequence[int],
        onset: TimeStamp,
        release: TimeStamp,
        track: int,
        chord_change: bool,
    ):
        return self._three_pitch_arpeggio_pattern(
            weight,
            -1,
            -2,
            pitches,
            onset,
            release,
            track,
            chord_change,
            extra_pitches_at=ExtraPitches.MID,
        )

    @pattern_method(
        min_dur=Fraction(1, 2),
        min_pitch_count=3,
        min_pitch_count_fallback="demisemibeat_tremolo",
    )
    def demisemibeat_alberti(self, *args, **kwargs):
        return self._alberti(0, *args, **kwargs)

    @pattern_method(
        allow_compound=Allow.NO,
        min_dur=Fraction(1, 1),
        min_pitch_count=3,
        min_pitch_count_fallback="semibeat_tremolo",
    )
    def semibeat_alberti(self, *args, **kwargs):
        return self._alberti(1, *args, **kwargs)

    @pattern_method(
        allow_triple=Allow.NO,
        min_dur=Fraction(2, 1),
        min_pitch_count=3,
        min_pitch_count_fallback="beat_tremolo",
    )
    def beat_alberti(self, *args, **kwargs):
        return self._alberti(2, *args, **kwargs)

    def _1_3_5(
        self,
        weight,
        pitches: t.Sequence[int],
        onset: TimeStamp,
        release: TimeStamp,
        track: int,
        chord_change: bool,
    ):
        return self._three_pitch_arpeggio_pattern(
            weight,
            -2,
            -1,
            pitches,
            onset,
            release,
            track,
            chord_change,
            extra_pitches_at=ExtraPitches.TOP,
        )

    @pattern_method(
        min_dur=Fraction(1, 2),
        min_pitch_count=3,
        min_pitch_count_fallback="demisemibeat_tremolo",
    )
    def demisemibeat_1_3_5(self, *args, **kwargs):
        return self._1_3_5(0, *args, **kwargs)

    @pattern_method(
        allow_compound=Allow.NO,
        min_dur=Fraction(2, 1),
        min_dur_fallback="semibeat_tremolo",
        min_pitch_count=3,
        min_pitch_count_fallback="semibeat_tremolo",
    )
    def semibeat_1_3_5(self, *args, **kwargs):
        return self._1_3_5(1, *args, **kwargs)

    @pattern_method(
        allow_triple=Allow.NO,
        min_dur=Fraction(2, 1),
        min_pitch_count=3,
        min_pitch_count_fallback="beat_tremolo",
    )
    def beat_1_3_5(self, *args, **kwargs):
        return self._1_3_5(2, *args, **kwargs)

    def get_pattern_method_from_name(self, name: str) -> Pattern:
        return getattr(self, name)

    def _get_pattern_options(
        self,
        pitches_or_pcs,
        harmony_onset,
        harmony_release,
        whitelist: t.Container | None = None,
    ) -> t.List[str]:
        params = (tuple(pitches_or_pcs), harmony_onset, harmony_release)
        if params in self._memo:
            return self._memo[params]
        patterns: list[str] = []
        for pattern_name in self._patterns:
            if whitelist is not None and pattern_name not in whitelist:
                continue
            pattern_method = self.get_pattern_method_from_name(pattern_name)
            pattern_fits = True
            for constraint in (
                lambda: (
                    pattern_method.min_dur is None
                    or (
                        (harmony_release - harmony_onset) / self._ts.beat_dur
                        >= pattern_method.min_dur
                    )
                ),
                lambda: (
                    pattern_method.min_pitch_count is None
                    or len(pitches_or_pcs) >= pattern_method.min_pitch_count
                ),
            ):
                if not constraint():
                    pattern_fits = False
                    break
            if pattern_fits:
                patterns.append(pattern_name)
        self._memo[params] = patterns
        return patterns

    def _call_pattern_method(
        self,
        pattern: Pattern,
        pitches,
        harmony_onset,
        harmony_release,
        onset,
        release,
        track,
        chord_change,
    ) -> list[Note]:
        if isinstance(pattern, str):
            pattern = self.get_pattern_method_from_name(pattern)
        # pattern_method = self.get_pattern_method_from_name(pattern_name)
        for constraint, fallback in (
            (
                lambda: (
                    pattern.min_dur is None
                    or (
                        (harmony_release - harmony_onset) / self._ts.beat_dur
                        >= pattern.min_dur
                    )
                ),
                pattern.min_dur_fallback,
            ),
            (
                lambda: (
                    pattern.min_pitch_count is None
                    or len(pitches) >= pattern.min_pitch_count
                ),
                pattern.min_pitch_count_fallback,
            ),
        ):
            if not constraint():
                LOGGER.debug(
                    f"{self.__class__.__name__} "
                    f"falling back from {pattern.__name__} to {fallback}"
                )
                return self._call_pattern_method(
                    fallback,
                    pitches,
                    harmony_onset,
                    harmony_release,
                    onset,
                    release,
                    track,
                    chord_change,
                )
        LOGGER.debug(
            f"{self.__class__.__name__}: "
            f"{pattern.__name__} time:{onset}--{release} "
            f"chord_change:{'yes' if chord_change else 'no'}"
        )
        out = pattern(pitches, onset, release, track, chord_change)
        return out

    def get_spacing_constraints(self, pattern: str | Pattern) -> SpacingConstraints:
        if isinstance(pattern, str):
            pattern_method = self.get_pattern_method_from_name(pattern)
        else:
            pattern_method = pattern
        return pattern_method.spacing_constraints

    def get_pattern(
        self,
        pitches_or_pcs,
        onset,
        harmony_onset,
        harmony_release,
        pattern: Pattern | None | str = None,
        whitelist: t.Container | None = None,
    ) -> Pattern:
        if pattern is not None:
            if isinstance(pattern, str):
                pattern = self.get_pattern_method_from_name(pattern)
            # assert isinstance(pattern, Pattern)  # Exception because Pattern is subscripted
            # just to get pyright not to complain
            assert pattern is not None
            assert not isinstance(pattern, str)
            self._prev_pattern = pattern
            return pattern
        try_to_keep_same_pattern = self._downbeats_only and (
            self._ts.weight(onset) != self._ts.max_weight
        )
        if self._prev_pattern is not None and (
            try_to_keep_same_pattern or random.random() <= self._inertia
        ):
            pattern = self._prev_pattern
            LOGGER.debug(f"{self.__class__.__name__} retrieving pattern {pattern}")
        else:
            pattern_options = self._get_pattern_options(
                pitches_or_pcs, harmony_onset, harmony_release, whitelist=whitelist
            )
            pattern_name = random.choice(pattern_options)
            self._prev_pattern = pattern  # type:ignore
            LOGGER.debug(f"{self.__class__.__name__} setting pattern {pattern}")
            pattern = self.get_pattern_method_from_name(pattern_name)
        # assert isinstance(pattern, Pattern) # Exception because Pattern is subscripted
        assert pattern is not None
        assert not isinstance(pattern, str)
        return pattern

    def __call__(
        self,
        pitches: t.Sequence[Pitch],
        onset: TimeStamp,
        release: TimeStamp,
        harmony_onset: TimeStamp | None = None,
        harmony_release: TimeStamp | None = None,
        pattern: Pattern | str | None = None,
        track=1,
        chord_change: bool = True,
    ) -> t.List[Note]:
        if harmony_onset is None:
            harmony_onset = onset
        if harmony_release is None:
            harmony_release = release

        pattern = self.get_pattern(
            pitches, onset, harmony_onset, harmony_release, pattern=pattern
        )

        return self._call_pattern_method(
            pattern,
            pitches,
            harmony_onset,
            harmony_release,
            onset,
            release,
            track,
            chord_change,
        )

    @property
    def prev_pattern(self):
        return self._prev_pattern

    # The definition of _all_patterns should be the last line in the class
    # definition
    _all_patterns = MappingProxyType(
        {name: f for name, f in locals().items() if getattr(f, "pattern_method", False)}
    )
