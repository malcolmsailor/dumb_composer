import typing as t
from contextlib import contextmanager
from dataclasses import dataclass

from dumb_composer.patterns import Pattern, PatternMaker
from dumb_composer.pitch_utils.types import Note, Pitch, SettingsBase
from dumb_composer.shared_classes import (
    ScoreInterface,
    ScoreWithAccompaniments,
    _ScoreBase,
)
from dumb_composer.utils.iterables import unique_items_in_order
from dumb_composer.utils.recursion import recursive_attempt


@dataclass
class PatternHandlerSettings(SettingsBase):
    whitelisted_patterns: t.Container[str] = ("demisemibeat_alberti",)


class PatternHandler:
    def __init__(
        self,
        score: _ScoreBase,
        settings: PatternHandlerSettings = PatternHandlerSettings(),
    ):
        get_i = lambda score: len(score.structural_bass) - 1
        validate = (
            lambda score: len({len(pitches) for pitches in score._structural.values()})
            == 1
        )
        # TODO: (Malcolm 2023-08-08) should we use AccompanimentInterface here?
        self._score = ScoreInterface(score, get_i=get_i, validate=validate)
        # TODO: (Malcolm 2023-08-03) allow include_bass to vary
        self._pattern_maker = PatternMaker(ts=score.ts, include_bass=True)
        self.settings = settings

    def choose_pattern(self) -> Pattern:
        chord = self._score.current_chord
        # TODO: (Malcolm 2023-08-08) filter pattern selections based on melody voice
        # TODO: (Malcolm 2023-08-03) rather than getting a new pattern every step, get a
        #   new pattern only every so often
        pattern: Pattern = self._pattern_maker.get_pattern(
            chord.pcs,
            chord.onset,
            chord.harmony_onset,
            chord.harmony_release,
            whitelist=self.settings.whitelisted_patterns,
        )
        return pattern

    def get_accompaniment(
        self, pitches: t.Sequence[Pitch], pattern: Pattern | None
    ) -> list[Note]:
        chord = self._score.current_chord
        unique_pitches = unique_items_in_order(pitches)
        out = self._pattern_maker(
            unique_pitches,
            onset=chord.onset,
            release=chord.release,
            harmony_onset=chord.harmony_onset,
            harmony_release=chord.harmony_release,
            pattern=pattern,
            # TODO: (Malcolm 2023-07-25) why do we need `track` here?`
            track=10,  # TODO: (Malcolm 2023-07-31) update track
            chord_change=self._score.at_chord_change(),
        )
        return out

    @contextmanager
    def append_attempt(self, notes: t.Sequence[Note]):
        with recursive_attempt(
            do_func=append_accompaniments,
            do_args=(notes, self._score.score),
            undo_func=pop_accompaniments,
            undo_args=(self._score.score,),
        ):
            yield


def append_accompaniments(notes: list[Note], score: ScoreWithAccompaniments):
    score.accompaniments.append(notes)


def pop_accompaniments(score: ScoreWithAccompaniments):
    score.accompaniments.pop()
