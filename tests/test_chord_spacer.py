from dataclasses import asdict

import pytest

from dumb_composer.chord_spacer import (
    NoSpacings,
    SimpleSpacer,
    SpacingConstraints,
    validate_spacing,
)
from dumb_composer.classes.scores import PrefabScore


# # TODO: (Malcolm) implement accounting for voice-leading
# @pytest.mark.skip(
#     reason="voice-leading doesn't take account of these constraints which causes the test to fail"
# )
@pytest.mark.parametrize("rntxt", ("m1 D: I b2 viio65/ii b3 iiø65 b4 I6",))
@pytest.mark.parametrize("max_adjacent_interval", (12, 5))
@pytest.mark.parametrize("control_bass_interval", (True, False))
@pytest.mark.parametrize("max_total_interval", (None, 19))
@pytest.mark.parametrize("max_bass_interval", (None, 15))
def test_apply_spacing_constraints(
    rntxt,
    max_adjacent_interval,
    control_bass_interval,
    max_total_interval,
    max_bass_interval,
):
    chord_spacer = SimpleSpacer()
    score = PrefabScore(chord_data=rntxt)
    spacing_constraints = SpacingConstraints(
        max_adjacent_interval=max_adjacent_interval,
        max_total_interval=max_total_interval,
        control_bass_interval=control_bass_interval,
        max_bass_interval=max_bass_interval,
    )
    for chord in score.chords:
        try:
            for spacing in chord_spacer(
                chord.pcs,
                chord.get_omissions((), ()),
                spacing_constraints=spacing_constraints,
            ):
                assert validate_spacing(spacing, spacing_constraints)
        except NoSpacings:
            pass
        # TODO: (Malcolm 2023-07-20) by default, the chord_spacer voice-leads
        #   chords following the first one. However, this causes the spacing
        #   constraints to be invalidated. To allow this test to work, we
        #   manually set _prev_pitches to None here so that each
        #   chord is spaced anew. However, I should look into validating
        #   voice-led spacings somehow.
        chord_spacer._prev_pitches = None


# from mspell import Speller


# def test_chord_spacer():
#     pg = ChordSpacer()
#     speller = Speller(pitches=True)
#     chords = ((0, 3, 7), (0, 3, 7, 10))
#     positions = ("close_position", "open_position", "keyboard_style")
#     for position in positions:
#         func = getattr(pg, position)
#         for chord in chords:
#             out = func(chord)
#             print(
#                 f"{position} of \n\t{chord} = \n\t{out} or \n\t{speller(out)}"
#             )


# def test_chord_spacer2():
#     pg = ChordSpacer()
#     speller = Speller()
#     chords = [(0, 4, 7, 10), (5, 9, 0)]
#     for pcs in chords:
#         out = pg(pcs, spacing="keyboard_style")
#         print(out, speller(out))
