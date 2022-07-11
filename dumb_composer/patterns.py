import random
from types import MappingProxyType
import pandas as pd


from .time import RhythmFetcher


class Note(pd.Series):
    def __init__(self, pitch, onset, release=None, dur=None):
        if release is None:
            release = onset + dur
        super().__init__(
            {"pitch": pitch, "onset": onset, "release": release, "type": "note"}
        )


def notes(pitches, onset, release=None, dur=None):
    return [Note(pitch, onset, release=release, dur=dur) for pitch in pitches]


def pattern_method(**kwargs):
    def wrap(f):
        setattr(f, "pattern_method", True)
        return f

    return wrap


class PatternMaker:
    def __init__(self, ts: str, inertia: float = 0.75):
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
        self.rf = RhythmFetcher(ts)
        self._inertia = inertia
        self._prev_pattern = None

    @pattern_method()
    def oompah(self, pitches, onset, release):
        beats = self.rf.beats(onset, release)
        beat_weight = self.rf.beat_weight
        foot, others = pitches[0], pitches[1:]
        out = []
        for beat in beats:
            if beat["weight"] > beat_weight:
                out.append(Note(foot, beat["onset"], beat["release"]))
            else:
                out.extend(notes(others, beat["onset"], beat["release"]))
        return out

    @pattern_method()
    def oompahpah(self, pitches, onset, release):
        beats = self.rf.beats(onset, release)
        max_weight = self.rf.max_weight
        foot, others = pitches[0], pitches[1:]
        out = []
        for beat in beats:
            if beat["weight"] == max_weight:
                out.append(Note(foot, beat["onset"], beat["release"]))
            else:
                out.extend(notes(others, beat["onset"], beat["release"]))
        return out

    @staticmethod
    def _apply_pitches_to_rhythms(rhythms, pitches):
        out = []
        for r in rhythms:
            out.extend(notes(pitches, r["onset"], r["release"]))
        return out

    def _chords(self, rhythm_f, pitches, onset, release):
        off_beats = rhythm_f(onset, release)
        return self._apply_pitches_to_rhythms(off_beats, pitches)

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
    def alberti(self, pitches, onset, release):
        if self.rf.is_compound:
            raise NotImplementedError
        beat_weight = self.rf.beat_weight
        sbs = self.rf.semibeats(onset, release)
        out = []
        for sb in sbs:
            if sb["weight"] > beat_weight:
                out.append(Note(pitches[0], sb["onset"], sb["release"]))
            elif sb["weight"] < beat_weight:
                out.append(Note(pitches[2], sb["onset"], sb["release"]))
            else:
                out.append(Note(pitches[1], sb["onset"], sb["release"]))
        return out

    def __call__(self, pitches, onset, release=None, dur=None, pattern=None):
        if pattern is None:
            if self._prev_pattern is None or random.random() > self._inertia:
                pattern = random.choice(self._patterns)
                self._prev_pattern = pattern
            else:
                pattern = self._prev_pattern
        if release is None:
            release = onset + dur
        return getattr(self, pattern)(pitches, onset, release)

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
