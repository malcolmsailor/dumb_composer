from dumb_composer.chord_spacer import ChordSpacer
from mspell import Speller


def test_chord_spacer():
    pg = ChordSpacer()
    speller = Speller(pitches=True)
    chords = ((0, 3, 7), (0, 3, 7, 10))
    positions = ("close_position", "open_position", "keyboard_style")
    for position in positions:
        func = getattr(pg, position)
        for chord in chords:
            out = func(chord)
            print(
                f"{position} of \n\t{chord} = \n\t{out} or \n\t{speller(out)}"
            )


def test_chord_spacer2():
    pg = ChordSpacer()
    speller = Speller()
    chords = [(0, 4, 7, 10), (5, 9, 0)]
    for pcs in chords:
        out = pg(pcs, spacing="keyboard_style")
        print(out, speller(out))
