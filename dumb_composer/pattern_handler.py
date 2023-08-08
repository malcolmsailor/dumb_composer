from dumb_composer.patterns import PatternMaker
from dumb_composer.shared_classes import ScoreInterface, _ScoreBase


class PatternHandler:
    def __init__(self, score: _ScoreBase):
        get_i = lambda score: len(score.structural_bass)
        validate = (
            lambda score: len({len(pitches) for pitches in score._structural.values()})
            == 1
        )
        self._score = ScoreInterface(score, get_i=get_i, validate=validate)
        # TODO: (Malcolm 2023-08-03) allow include_bass to vary
        self._pattern_maker = PatternMaker(ts=score.ts, include_bass=True)

    def __call__(self):
        chord = self._score.current_chord
        # TODO: (Malcolm 2023-08-03) rather than getting a new pattern every step, get a
        #   new pattern only every so often
        pattern = self._pattern_maker.get_pattern(
            chord.pcs,
            chord.onset,
            chord.harmony_onset,
            chord.harmony_release,
        )
        return pattern
