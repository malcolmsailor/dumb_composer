from dataclasses import asdict
from dumb_composer.chord_spacer import (
    NoSpacings,
    SimpleSpacer,
    SpacingConstraints,
    validate_spacing,
)
import pytest

from dumb_composer.shared_classes import PrefabScore


# TODO: (Malcolm) implement accounting for voice-leading
@pytest.mark.skip(
    reason="voice-leading doesn't take account of these constraints which causes the test to fail"
)
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
        max_adjacent_interval,
        max_total_interval,
        control_bass_interval,
        max_bass_interval,
    )
    for chord in score.chords:
        try:
            for spacing in chord_spacer(
                chord.pcs,
                chord.get_omissions((), ()),
                spacing_constraints=spacing_constraints,
            ):
                assert validate_spacing(spacing, **asdict(spacing_constraints))
        except NoSpacings:
            pass


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
