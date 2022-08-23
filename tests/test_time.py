from dataclasses import dataclass
from numbers import Number
import random
from dumb_composer.time import Meter, RhythmFetcher
import typing as t


def test_meter():
    tests = {
        "4/4": {
            "weights": {
                0: 2,
                2: 1,
                1: 0,
                3: 0,
                0.5: -1,
                1.5: -1,
                0.25: -2,
                0.75: -2,
                15.75: -2,
                16: 2,
                34: 1,
            },
            "durs": (1, 0.5, 2),
            "weight_properties": (0, 2),
        },
        "4/2": {
            "weights": {
                0: 2,
                4: 1,
                2: 0,
                6: 0,
                1: -1,
                0.5: -2,
                0.25: -3,
                0.125: -3,
            },
        },
        "3/4": {
            "weights": {
                0: 1,
                1: 0,
                2: 0,
                0.5: -1,
                1.5: -1,
                0.25: -2,
                0.75: -2,
            },
        },
        "3/2": {
            "weights": {
                0: 1,
                2: 0,
                4: 0,
                1: -1,
                3: -1,
                0.5: -2,
                0.75: -3,
            },
            "durs": (2, 1, 6),
            "weight_properties": (0, 1),
        },
        "5/4": {
            "weights": {
                # I don't know if this is the ideal behavior but it is the expected
                # behavior as the algorithm stands. Don't expect to use this with
                # odd meters very much anyway.
                0: 3,
                2: 1,
                3: 0,
                4: 2,
            },
        },
        "6/8": {
            "weights": {
                0: 1,
                1.5: 0,
                0.5: -1,
                1: -1,
                2: -1,
                2.5: -1,
                0.25: -2,
            },
            "durs": (1.5, 0.5, 3),
            "weight_properties": (0, 1),
        },
        "6/4": {
            "weights": {
                0: 1,
                3: 0,
                1: -1,
                2: -1,
                4: -1,
                5: -1,
                3.5: -2,
            },
        },
        "9/8": {
            "weights": {
                0: 1,
                1.5: 0,
                3: 0,
                0.5: -1,
                1: -1,
                2: -1,
                2.5: -1,
                0.25: -2,
            },
        },
    }
    for ts, test_dict in tests.items():
        meter = Meter(ts)
        if "weights" in test_dict:
            for arg, result in test_dict["weights"].items():
                assert meter(arg) == result
        if "durs" in test_dict:
            beat_dur, semibeat_dur, superbeat_dur = test_dict["durs"]
            assert meter.beat_dur == beat_dur
            assert meter.semibeat_dur == semibeat_dur
            assert meter.superbeat_dur == superbeat_dur
        if "weight_properties" in test_dict:
            assert (meter.beat_weight, meter.max_weight) == test_dict[
                "weight_properties"
            ]


def test_rhythm_fetcher():
    class RTest:
        def __init__(
            self,
            ts,
            compound=False,
            triple=False,
            start=0,
            stop=16,
            length=8,
            increment=2.5,
        ):
            self.ts = ts
            self.compound = compound
            self.triple = triple
            self.start = start
            self.stop = stop
            self.length = length
            self.increment = increment

        def __call__(self):
            rf = RhythmFetcher(self.ts)
            assert self.compound == rf.is_compound
            assert self.triple == rf.is_triple
            onset = self.start
            while onset <= self.stop:
                release = onset + self.length
                out = rf(onset=onset, release=release)
                assert out[0]["onset"] >= onset
                assert out[-1]["onset"] < release
                assert out[-1]["release"] <= release
                onset += self.increment

    random.seed(42)
    tests = [
        RTest("3/4", triple=True),
        RTest("4/4"),
        RTest("2/8"),
        RTest("6/8", compound=True),
    ]
    for test in tests:
        test()
    rf = RhythmFetcher("4/4")
    foo = rf.trochee(0, 4)
    # bar = rf.iamb(0, 4)
    rf = RhythmFetcher("6/8")
    foo = rf.trochee(0, 3)
    bar = rf.iamb(0, 3)
    # for ts, compound in tests:
    #     rf = RhythmFetcher(ts)
    #     assert rf._compound == compound
    #     print(rf(onset=0, release=4))
    #     print(rf.beats(0, 4))
    #     print(rf.semibeats(0, 4))


def test_duration_is_odd():
    @dataclass
    class TestCase:
        meter: str
        odd: t.Tuple[Number]
        not_odd: t.Tuple[Number]

    testcases = [
        TestCase(
            "4/4",
            odd=(
                5 / 4,
                5 / 2,
                5,
                10,
                7 / 4,
                7 / 2,
                7,
                9 / 4,
                9 / 2,
                9,
                13 / 4,
                17 / 4,
                33,
                95,
                514,
            ),
            not_odd=(
                1 / 4,
                3 / 8,
                1 / 2,
                3 / 4,
                1,
                3 / 2,
                2,
                3,
                4,
                6,
                8,
                32,
                96,
                512,
            ),
        ),
        TestCase(
            "9/4",
            odd=(4, 5, 7, 8, 10, 12, 13, 15),
            not_odd=(3 / 4, 3 / 2, 3, 6, 9, 18, 27, 2),
        ),
        TestCase(
            "6/4",
            odd=(),
            not_odd=(1, 2, 3, 6, 9, 12),
        ),
    ]
    for testcase in testcases:
        meter = Meter(testcase.meter)
        for not_odd in testcase.not_odd:
            assert not meter.duration_is_odd(not_odd)
        for odd in testcase.odd:
            assert meter.duration_is_odd(odd)
