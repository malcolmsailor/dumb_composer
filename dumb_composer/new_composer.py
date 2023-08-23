import logging
import typing as t
from dataclasses import dataclass

from dumb_composer.classes.score_interfaces import ScoreInterface
from dumb_composer.classes.scores import PrefabScore, PrefabScoreWithAccompaniments
from dumb_composer.incremental_contrapuntist import (
    IncrementalContrapuntist,
    append_structural,
    pop_structural,
)
from dumb_composer.pattern_handler import PatternHandler, PatternHandlerSettings
from dumb_composer.pitch_utils.spacings import RangeConstraints
from dumb_composer.pitch_utils.types import ALTO, BASS, MELODY, TENOR, TENOR_AND_ALTO
from dumb_composer.structural_partitioner import (
    StructuralPartitioner,
    StructuralPartitionerSettings,
)
from dumb_composer.utils.display import Spinner
from dumb_composer.utils.recursion import DeadEnd, RecursionFailed, recursive_attempt

LOGGER = logging.getLogger(__name__)

# TODO: (Malcolm 2023-08-08) eventually I should distinguish SOPRANO and MELODY
#   - MELODY carries the main melody and is generated before any other voices except
#       the bass. It may or may not be the top voice.
#   - SOPRANO is the top voice.
# Basically I guess we want to do this by setting MELODY as an alias for some other
#   voice (ALTO, TENOR, or SOPRANO). The only voice that can't be MELODY is the bass.
#   (In the case where the bass is the "melody" in the sense of carrying the tune,
#    it still will follow the harmonic bass.)


@dataclass
class NewComposerSettings(StructuralPartitionerSettings, PatternHandlerSettings):
    pass


class NewComposer:
    def __init__(self, settings: NewComposerSettings = NewComposerSettings()):
        self.settings = settings
        self.structural_partitioner = StructuralPartitioner(settings)
        self._pattern_handler: PatternHandler | None = None
        # TODO: (Malcolm 2023-08-08) write a melody voice chooser.
        #   Among other things, it should have a certain (perhaps quite high) "inertia",
        #   so that the same voice is chosen repeatedly.
        #   Another thing to do would be
        self._melody_voice_chooser = lambda: MELODY
        self._temp_melody_contrapuntist: IncrementalContrapuntist | None = None
        self._temp_melody_contrapuntist2: IncrementalContrapuntist | None = None
        self._score: ScoreInterface | None = None
        self._spinner = Spinner()

    def _recurse(self):
        self._spinner()
        assert self._score is not None
        LOGGER.debug(f"entering _recurse() with {self._score.i=}")
        if self._score.complete:
            return
        LOGGER.debug(f"entering _recurse() with {len(self._score._score.chords)=}")
        # if self._score.i > len(self._score._score.chords):
        #     breakpoint()

        assert self._pattern_handler is not None
        assert self._temp_melody_contrapuntist is not None
        assert self._temp_melody_contrapuntist2 is not None
        # TODO: (Malcolm 2023-08-07) each pattern should have a range associated with it
        # 1. choose melody voice (or voices?)
        # 2. choose pattern
        # 3. choose total number of voices
        #       - this is likely an attribute of the pattern
        # 4. set range for melody
        # 5. generate melody (and bass?)
        # 6. set range for other voices based on pattern and melody
        #       - move bass into range if necessary?
        # 7. generate inner voices
        # 8. double parts as necessary
        # 9. generate accompaniment

        # 1. choose melody voice
        melody_voice = self._melody_voice_chooser()
        LOGGER.debug(f"{melody_voice=}")

        # 2. choose pattern
        # TODO: (Malcolm 2023-08-08) pattern choice should maybe be iterated through
        #   in case we fail on one pattern but can succeed on others?
        pattern = self._pattern_handler.choose_pattern()
        LOGGER.debug(f"pattern={pattern.__name__}")

        # 3. choose total number of voices
        total_voice_count: int = pattern.total_voice_count
        LOGGER.debug(f"{total_voice_count=}")

        # 4. set range for melody
        # TODO: (Malcolm 2023-08-08) if there is more than some threshold difference
        #   between these range constraints and the last previous range constraints,
        #   space the voices anew rather than attempting to voice-lead (perhaps in
        #   this case add a special "break" symbol that causes a phrase break
        #   or something in the melody)
        # TODO: (Malcolm 2023-08-08) use these range constraints below
        range_constraints: RangeConstraints = pattern.range_constraints

        # 5. generate melody (and bass?)
        # TODO: (Malcolm 2023-08-08) we need to create contrapuntist in a flexible way
        #   allowing for the melody voice to change
        # TODO: (Malcolm 2023-08-08) can we merge the different contrapuntists or
        #   do they need to be separate instances?
        # TODO: (Malcolm 2023-08-08) rewrite in a way that doesn't require excessive
        #   nesting of for and with statements (probably refactor into separate functions?)
        for outer_pitches in self._temp_melody_contrapuntist.step():
            with self._temp_melody_contrapuntist.append_attempt(outer_pitches):
                # 6. set range for other voices based on pattern and melody
                # TODO: (Malcolm 2023-08-08) 6.

                # 7. generate inner voices
                # TODO: set inner voices based on pattern or something similar
                for inner_pitches in self._temp_melody_contrapuntist2.step():
                    with self._temp_melody_contrapuntist2.append_attempt(inner_pitches):
                        # 8. double parts as necessary
                        # TODO: (Malcolm 2023-08-08) 8

                        # 9. generate accompaniment
                        # TODO: (Malcolm 2023-08-08) get correct voices dynamically
                        pitches = (outer_pitches[BASS],) + (
                            inner_pitches[TENOR],
                            inner_pitches[ALTO],
                        )
                        accompaniment = self._pattern_handler.get_accompaniment(
                            pitches, pattern
                        )
                        with self._pattern_handler.append_attempt(accompaniment):
                            return self._recurse()

    def __call__(self, rntxt: str):
        assert self._score is None
        score = PrefabScoreWithAccompaniments(rntxt)
        get_i = lambda score: len(score._structural[BASS])
        validate = lambda score: True  # We rely on other views of the score to validate

        self._score = ScoreInterface(score, get_i=get_i, validate=validate)
        self.structural_partitioner(score)
        self._pattern_handler = PatternHandler(score, self.settings)
        self._temp_melody_contrapuntist = IncrementalContrapuntist(
            score=score, voices=(BASS, MELODY)
        )
        self._temp_melody_contrapuntist2 = IncrementalContrapuntist(
            score=score, voices=(TENOR_AND_ALTO,), prior_voices=(BASS, MELODY)
        )
        try:
            self._recurse()
        except DeadEnd:
            LOGGER.info(f"Recursion reached a terminal dead end.")
            raise RecursionFailed("Reached a terminal dead end")

        self._score = None
        return score
