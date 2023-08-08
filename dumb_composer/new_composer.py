import logging
import typing as t
from dataclasses import dataclass

from dumb_composer.pattern_handler import PatternHandler
from dumb_composer.pitch_utils.types import MELODY
from dumb_composer.shared_classes import PrefabScore, ScoreInterface
from dumb_composer.structural_partitioner import (
    StructuralPartitioner,
    StructuralPartitionerSettings,
)
from dumb_composer.utils.recursion import DeadEnd, RecursionFailed

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
class NewComposerSettings(StructuralPartitionerSettings):
    whitelisted_patterns: t.Container[str] = ("semibeat_alberti",)


class NewComposer:
    def __init__(self, settings: NewComposerSettings = NewComposerSettings()):
        self.settings = settings
        self.structural_partitioner = StructuralPartitioner(settings)
        self._pattern_handler: PatternHandler | None = None
        self._melody_voice_chooser = lambda: MELODY

    def _recurse(self, score: PrefabScore):
        assert self._pattern_handler is not None
        # TODO: (Malcolm 2023-08-07) each pattern should have a range associated with it
        # 1. choose melody voice (or voices?)
        # 2. choose pattern
        # 3. choose total number of voices
        # 4. set range for melody
        # 5. generate melody (and bass?)
        # 6. set range for other voices based on pattern and melody
        #       - move bass into range if necessary?
        # 7. generate inner voices
        # 8. double parts as necessary
        # 9. generate accompaniment
        pattern = self._pattern_handler()
        LOGGER.debug(f"{pattern=}")
        breakpoint()

    def __call__(self, rntxt: str):
        score = PrefabScore(rntxt)
        self.structural_partitioner(score)
        self._pattern_handler = PatternHandler(score)
        try:
            self._recurse(score)
        except DeadEnd:
            LOGGER.info(f"Recursion reached a terminal dead end.")
            raise RecursionFailed("Reached a terminal dead end")
