from dataclasses import dataclass, field
from numbers import Number
import textwrap
import typing as t
from dumb_composer.constants import TIME_TYPE
import dumb_composer.pitch_utils.chords as chords


def test_get_chords_from_rntxt_fit_scale():
    @dataclass
    class TestCase:
        rntxt: str
        expected_pcs: t.Tuple[int]
        expected_scale_pcs: t.Tuple[int]

    TC = TestCase
    test_cases = [
        TC("m1 C: V7/V", (2, 6, 9, 0), (7, 9, 11, 0, 2, 4, 6)),
        TC("m1 C: viio7", (11, 2, 5, 8), (0, 2, 4, 5, 7, 8, 11)),
        TC("m1 C: viio7/V", (6, 9, 0, 3), (7, 9, 11, 0, 2, 3, 6)),
        TC("m1 C: Ger65", (8, 0, 3, 6), (0, 2, 3, 6, 7, 8, 11)),
        TC("m1 C: V7/bII", (8, 0, 3, 6), (1, 3, 5, 6, 8, 10, 0)),
        TC("m1 C: viio7/bIII", (2, 5, 8, 11), (3, 5, 7, 8, 10, 11, 2)),
        TC("m1 C: V6/#i", (0, 3, 8), (1, 3, 4, 6, 8, 9, 0)),
        TC("m1 C: bII7", (1, 5, 8, 0), (0, 1, 4, 5, 7, 8, 11)),
    ]
    for test_case in test_cases:
        out_list, _, _ = chords.get_chords_from_rntxt(test_case.rntxt)
        chord = out_list[0]
        assert test_case.expected_pcs == chord.pcs
        assert test_case.expected_scale_pcs == chord.scale_pcs


def test_get_chords_from_rntxt_split_chords():
    @dataclass
    class TestCase:
        rntxt: str
        expected_onsets: t.Tuple[TIME_TYPE]

    test_cases = [
        TestCase("Time Signature: 2/4\nm1 I b1.25 V", (0, 0.25, 1)),
        # TODO next case should be (0, 0.25, 1, 2)
        TestCase("Time Signature: 4/4\nm1 I b1.25 V", (0, 0.25, 1, 2)),
        TestCase("m1 V b2.75 viio/V b3 V", (0, 1, 1.75, 2)),
    ]

    for test_case in test_cases:
        out_list, _, _ = chords.get_chords_from_rntxt(
            test_case.rntxt, split_chords_at_metric_strong_points=True
        )
        for chord in out_list:
            assert isinstance(chord.onset, TIME_TYPE)
            assert isinstance(chord.release, TIME_TYPE)
        onsets = tuple(chord.onset for chord in out_list)
        assert onsets == test_case.expected_onsets


def test_get_harmony_onsets_and_releases():
    @dataclass
    class DummyChord:
        onset: Number
        release: Number
        symbol: str
        harmony_onset: t.Optional[Number] = field(default=None, compare=False)
        harmony_release: t.Optional[Number] = field(default=None, compare=False)

    class TestCase:
        def __init__(
            self,
            onsets_releases_and_symbols: t.List[t.Tuple[Number, Number, str]],
        ):
            self.raw = onsets_releases_and_symbols
            self.chords = [
                DummyChord(*times) for times in onsets_releases_and_symbols
            ]

    test_cases = [
        TestCase(
            [
                (0, 1, "I"),
                (1, 3, "I"),
                (3, 7, "V"),
                (7, 10, "I"),
                (10, 10.5, "I"),
                (10.5, 11, "I"),
            ]
        )
    ]
    for test_case in test_cases:
        chords.get_harmony_onsets_and_releases(test_case.chords)
        prev_chord = None
        for chord, (onset, release, _) in zip(test_case.chords, test_case.raw):
            if prev_chord is not None:
                assert (
                    chord.harmony_onset == prev_chord.harmony_onset
                    and chord.harmony_release == prev_chord.harmony_release
                ) or (chord.onset == prev_chord.release)
            assert chord.onset == onset and chord.release == release
