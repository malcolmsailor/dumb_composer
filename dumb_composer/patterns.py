import random
from re import M
import typing as t
from types import MappingProxyType
import pandas as pd

from dumb_composer.time import RhythmFetcher, Meter
from dumb_composer.shared_classes import Allow, Note, notes, print_notes


def pattern_method(
    requires_bass: bool = False, allow_compound: Allow = Allow.YES
):
    def wrap(f):
        setattr(f, "pattern_method", True)
        setattr(f, "requires_bass", requires_bass)
        setattr(f, "allow_compound", allow_compound)
        return f

    return wrap


class PatternMaker:
    def __init__(
        self,
        ts: t.Union[Meter, str],
        inertia: float = 0.75,
        include_bass: bool = True,
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
        self._patterns = self._filter_patterns(ts, include_bass)
        self.rf = RhythmFetcher(ts)
        self._inertia = inertia
        self._prev_pattern: t.Optional[str] = None

    def _filter_patterns(self, ts: Meter, include_bass: bool) -> t.List[str]:
        out = []
        for pattern_name in self._all_patterns:
            pattern_method = getattr(self, pattern_name)
            if ts.is_compound and pattern_method.allow_compound is Allow.NO:
                continue
            elif (
                not ts.is_compound
                and pattern_method.allow_compound is Allow.ONLY
            ):
                continue
            if not include_bass and pattern_method.requires_bass:
                continue
            out.append(pattern_name)
        return out

    @pattern_method(requires_bass=True)
    def oompah(self, pitches, onset, release, track):
        """
        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=0, release=4, pattern="oompah"))
        on off   pitches
        0   1     (48,)
        1   2  (64, 67)
        2   3     (48,)
        3   4  (64, 67)

        In 3/4 gives the same results as `oompahpah`.

        >>> pm = PatternMaker("3/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=0, release=3, pattern="oompah"))
        on off   pitches
        0   1     (48,)
        1   2  (64, 67)
        2   3  (64, 67)

        The first beat is always given "oom". This may not be what you want so
        it may be best to avoid using this pattern starting on a weakbeat.
        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=0.5, release=4, pattern="oompah"))
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
            if i == 0 or beat["weight"] > beat_weight:
                out.append(
                    Note(foot, beat["onset"], beat["release"], track=track)
                )
            else:
                out.extend(
                    notes(others, beat["onset"], beat["release"], track=track)
                )
        return out

    @pattern_method(requires_bass=True)
    def oompahpah(self, pitches, onset, release, track):
        """
        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=0, release=4, pattern="oompahpah"))
        on off   pitches
        0   1     (48,)
        1   2  (64, 67)
        2   3  (64, 67)
        3   4  (64, 67)

        In 3/4 gives the same results as `oompah`.
        >>> pm = PatternMaker("3/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=0, release=3, pattern="oompahpah"))
        on off   pitches
        0   1     (48,)
        1   2  (64, 67)
        2   3  (64, 67)

        The first beat is always given "oom". This may not be what you want so
        it may be best to avoid using this pattern starting on a weakbeat.
        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=0.5, release=4, pattern="oompahpah"))
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
            if i == 0 or beat["weight"] == max_weight:
                out.append(
                    Note(foot, beat["onset"], beat["release"], track=track)
                )
            else:
                out.extend(
                    notes(others, beat["onset"], beat["release"], track=track)
                )
        return out

    @staticmethod
    def _apply_pitches_to_rhythms(rhythms, pitches, track):
        out = []
        for r in rhythms:
            out.extend(notes(pitches, r["onset"], r["release"], track=track))
        return out

    def _chords(
        self, rhythm_f, pitches, onset, release, track, avoid_empty: bool = True
    ):
        rhythms = rhythm_f(onset, release, avoid_empty=avoid_empty)
        return self._apply_pitches_to_rhythms(rhythms, pitches, track)

    @pattern_method()
    def off_beat_chords(self, pitches, onset, release, track):
        """
        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=0, release=4, pattern="off_beat_chords"))
        on off       pitches
        1   2  (48, 64, 67)
        2   3  (48, 64, 67)
        3   4  (48, 64, 67)

        If the interval between onset and release would not otherwise contain a
        single chord, a chord will be inserted starting at onset:

        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=0, release=1, pattern="off_beat_chords"))
        on  off       pitches
        0    1  (48, 64, 67)

        This onset extends to release or until the next beat, whichever is sooner.

        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=3.5, release=5, pattern="off_beat_chords"))
        on off       pitches
        3.5   4  (48, 64, 67)

        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=3.5, release=3.75, pattern="off_beat_chords"))
        on   off       pitches
        3.5  3.75  (48, 64, 67)
        """
        return self._chords(self.rf.off_beats, pitches, onset, release, track)

    @pattern_method()
    def off_semibeat_chords(self, pitches, onset, release, track):
        """
        >>> pm = PatternMaker("4/4")
        >>> print_notes(
        ...     pm(pitches=[48, 64, 67], onset=0, release=2, pattern="off_semibeat_chords"))
        on  off       pitches
        1/2    1  (48, 64, 67)
        1  3/2  (48, 64, 67)
        3/2    2  (48, 64, 67)
        """
        return self._chords(
            self.rf.off_semibeats, pitches, onset, release, track
        )

    @pattern_method()
    def beat_chords(self, pitches, onset, release, track):
        return self._chords(self.rf.beats, pitches, onset, release, track)

    @pattern_method()
    def semibeat_chords(self, pitches, onset, release, track):
        return self._chords(self.rf.semibeats, pitches, onset, release, track)

    @pattern_method()
    def superbeat_chords(self, pitches, onset, release, track):
        return self._chords(self.rf.superbeats, pitches, onset, release, track)

    # TODO compound alberti bass

    @pattern_method(allow_compound=Allow.NO)
    def alberti(self, pitches, onset, release, track):
        beat_weight = self.rf.beat_weight
        sbs = self.rf.semibeats(onset, release)
        out = []
        for sb in sbs:
            if sb["weight"] > beat_weight:
                out.append(
                    Note(pitches[0], sb["onset"], sb["release"], track=track)
                )
            elif sb["weight"] < beat_weight:
                out.append(
                    Note(pitches[2], sb["onset"], sb["release"], track=track)
                )
            else:
                out.append(
                    Note(pitches[1], sb["onset"], sb["release"], track=track)
                )
        return out

    def __call__(
        self,
        pitches,
        onset,
        release=None,
        dur=None,
        pattern=None,
        track=1,
    ) -> t.List[Note]:
        if pattern is None:
            if self._prev_pattern is None or random.random() > self._inertia:
                pattern = random.choice(self._patterns)
                self._prev_pattern = pattern
            else:
                pattern = self._prev_pattern
        else:
            self._prev_pattern = pattern
        if release is None:
            release = onset + dur
        return getattr(self, pattern)(pitches, onset, release, track)

    @property
    def prev_pattern(self):
        return self._prev_pattern

    # The definition of _all_patterns should be the last line in the class
    # definition
    _all_patterns = MappingProxyType(
        {
            name: f
            for name, f in locals().items()
            if getattr(f, "pattern_method", False)
        }
    )
