from dataclasses import dataclass
import typing as t
import dumb_composer.pitch_utils.chords as chords


def test_get_chords_from_rntxt():
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
