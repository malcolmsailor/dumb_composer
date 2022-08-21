import random
import typing as t
from types import MappingProxyType


from dumb_composer.time import RhythmFetcher
from dumb_composer.shared_classes import Note, notes


def pattern_method(requires_bass: bool = False, **kwargs):
    def wrap(f):
        setattr(f, "pattern_method", True)
        setattr(f, "requires_bass", requires_bass)
        return f

    return wrap


class PatternMaker:
    def __init__(
        self, ts: str, inertia: float = 0.75, include_bass: bool = True
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
        # eventually we may want to filter _all_patterns further
        self._patterns = [name for name in self._all_patterns]
        if not include_bass:
            self._patterns = [
                name
                for name in self._all_patterns
                if not getattr(self, name).requires_bass
            ]
        self.rf = RhythmFetcher(ts)
        self._inertia = inertia
        self._prev_pattern = None
        self._include_bass = include_bass

    @pattern_method(requires_bass=True)
    def oompah(self, pitches, onset, release, track):
        beats = self.rf.beats(onset, release)
        beat_weight = self.rf.beat_weight
        foot, others = pitches[0], pitches[1:]
        out = []
        for beat in beats:
            if beat["weight"] > beat_weight:
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
        beats = self.rf.beats(onset, release)
        max_weight = self.rf.max_weight
        foot, others = pitches[0], pitches[1:]
        out = []
        for beat in beats:
            if beat["weight"] == max_weight:
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

    def _chords(self, rhythm_f, pitches, onset, release, track):
        off_beats = rhythm_f(onset, release)
        return self._apply_pitches_to_rhythms(off_beats, pitches, track)

    @pattern_method()
    def off_beat_chords(self, *args, **kwargs):
        return self._chords(self.rf.off_beats, *args, **kwargs)

    @pattern_method()
    def off_semibeat_chords(self, *args, **kwargs):
        return self._chords(self.rf.off_semibeats, *args, **kwargs)

    @pattern_method()
    def beat_chords(self, *args, **kwargs):
        return self._chords(self.rf.beats, *args, **kwargs)

    @pattern_method()
    def semibeat_chords(self, *args, **kwargs):
        return self._chords(self.rf.semibeats, *args, **kwargs)

    @pattern_method()
    def superbeat_chords(self, *args, **kwargs):
        return self._chords(self.rf.superbeats, *args, **kwargs)

    @pattern_method()
    def alberti(self, pitches, onset, release, track):
        if self.rf.is_compound:
            raise NotImplementedError
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
