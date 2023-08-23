import itertools
from copy import deepcopy

import pytest

from dumb_composer.classes.score_interfaces import ScoreInterface
from dumb_composer.classes.scores import FourPartScore, _ScoreBase
from dumb_composer.pitch_utils.types import ALTO, BASS, MELODY, TENOR, Voice
from dumb_composer.suspensions import Suspension

RULE_OF_OCTAVE = """m1 C: I b2 V43 b3 I6 b4 ii65
m2 V b2 IV6 b3 V65 b4 I"""


def get_get_i(voice: Voice):
    def get_i(score):
        return len(score._structural[voice])

    return get_i


# Should we do all permutations?
VOICE_ORDERINGS = [
    (BASS, MELODY, TENOR, ALTO),
    (TENOR, ALTO, BASS, MELODY),
    (MELODY, BASS, TENOR, ALTO),
]

# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------


def structural_lengths(score: _ScoreBase):
    return [len(v) for v in score._structural.values()]


# ----------------------------------------------------------------------------------
# Score tests
# ----------------------------------------------------------------------------------


class TestScoreBase:
    @pytest.fixture
    @staticmethod
    def score():
        score = FourPartScore(RULE_OF_OCTAVE)
        return score

    @pytest.fixture(
        params=[
            {"voice_ordering": voice_ordering, "base_len": base_len}
            for voice_ordering, base_len in itertools.product(
                VOICE_ORDERINGS, [1, 2, 4]
            )
        ]
    )
    @staticmethod
    def ragged_score(request, score: _ScoreBase):
        base_len: int = request.param["base_len"]
        for i, voice in enumerate(request.param["voice_ordering"]):
            score._structural[voice].extend(
                list(range(min(i * base_len, len(score.chords))))
            )
        return score

    @staticmethod
    def test_split_and_merge_chords(ragged_score: _ScoreBase):
        original_lengths = structural_lengths(ragged_score)
        max_i = max(original_lengths)
        for i, chord in enumerate(ragged_score.chords):
            split_time = (chord.onset + chord.release) / 2
            ragged_score.split_ith_chord_at(i, split_time)
            new_lengths = structural_lengths(ragged_score)
            ragged_score.merge_ith_chords(i)
            restored_lengths = structural_lengths(ragged_score)
            assert original_lengths == restored_lengths
            if original_lengths == new_lengths:
                assert max_i <= i
            else:
                expected_lengths = [x if x <= i else x + 1 for x in original_lengths]
                assert new_lengths == expected_lengths


# ----------------------------------------------------------------------------------
# Score interface tests
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "voice_order",
    list(itertools.permutations([MELODY, BASS, TENOR])),  # type:ignore
)
class TestScoreInterface:
    @pytest.fixture
    def no_validate(self):
        return lambda: True

    @pytest.fixture(params=[BASS, MELODY, TENOR, ALTO])
    def get_i(self, request):
        return get_get_i(request.param)

    @pytest.fixture
    @staticmethod
    def score_interface(no_validate, get_i):
        score = _ScoreBase(RULE_OF_OCTAVE)
        score_interface = ScoreInterface(score, get_i=get_i, validate=no_validate)
        return score_interface

    @staticmethod
    def test_split_with_suspensions(
        score_interface: ScoreInterface, voice_order: tuple[Voice, ...]
    ):
        # As it currently stands, new suspensions overlapping with existing
        #   suspensions need to either:
        #       - end at the same time
        #       - end earlier
        # I should double-check that these conditions all work.

        suspension = Suspension(60, -2, True, 10)
        before_score = deepcopy(score_interface._score)
        times = [score_interface.current_chord.release / 2**i for i in range(1, 4)]

        # Add suspensions
        for time, voice in zip(times, voice_order):
            score_interface.apply_suspension(suspension, time, voice, annotate=False)

        # Verify that the suspensions are as expected
        for time_i, time in enumerate(times):
            for voice_i in range(time_i):
                voice = voice_order[voice_i]
                assert score_interface._score.is_suspension_at(time, voice)
            voice = voice_order[time_i]
            assert score_interface._score.is_resolution_at(time, voice)

        # Remove suspensions
        for time, voice in zip(reversed(times), reversed(voice_order)):
            score_interface.undo_suspension(voice, time, annotate=False)

        # Check that the score is unaltered
        after_score = score_interface._score
        assert before_score == after_score
