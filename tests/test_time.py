from dataclasses import dataclass
from numbers import Number
import random
import typing as t

from dumb_composer.constants import TIME_TYPE
from dumb_composer.time import Meter, RhythmFetcher


def test_meter():
    tests = {
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
        # "5/4": {
        #     "weights": {
        #         # I don't know if this is the ideal behavior but it is the expected
        #         # behavior as the algorithm stands. Don't expect to use this with
        #         # odd meters very much anyway.
        #         0: 3,
        #         2: 1,
        #         3: 0,
        #         4: 2,
        #     },
        # },
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
        "12/8": {
            "weights": {
                0: 2,
                3: 1,
                1.5: 0,
                0.5: -1,
                1.0: -1,
                2.5: -1,
                0.25: -2,
                4.5: 0,
                6: 2,
            }
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

    for ts in Meter._ts_dict:
        meter = Meter(ts)
        if meter.is_compound:
            assert meter.beat_dur / meter.semibeat_dur == 3
            assert sum(meter.stuttersemibeat_durs) == meter.beat_dur
        else:
            assert meter.beat_dur / meter.semibeat_dur == 2
        if meter.is_triple:
            assert meter.bar_dur / meter.beat_dur == 3
            assert meter.superbeat_dur == meter.bar_dur
            assert sum(meter.stutterbeat_durs) == meter.bar_dur
        else:

            assert meter.bar_dur / meter.beat_dur in (2, 4)
            if meter.bar_dur / meter.beat_dur == 2:
                assert meter.superbeat_dur == meter.bar_dur
            else:
                assert meter.superbeat_dur * 2 == meter.bar_dur
        if meter.is_compound and meter.is_triple:
            assert sorted(meter.non_odd_lengths) == [
                meter.scale_factor * dur
                for dur in (0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 3.0, 4.5, 7.5, 9)
            ]
        elif meter.is_compound:
            if meter.n_beats == 2:
                # test not implemented for 4
                assert sorted(meter.non_odd_lengths) == [
                    meter.scale_factor * dur
                    for dur in (0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 3.0, 4.5, 6)
                ]
        elif meter.is_triple:
            assert sorted(meter.non_odd_lengths) == [
                meter.scale_factor * dur
                for dur in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 6)
            ]
        else:
            if meter.n_beats == 4:
                # test ot implemented for 2
                assert sorted(meter.non_odd_lengths) == [
                    meter.scale_factor * dur
                    for dur in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8)
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
            odd=(4, 7, 8, 10, 12, 13),
            not_odd=(3 / 2, 3, 6, 9, 18, 27, 2),
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


def test_split_at_metric_strong_points(slow):
    @dataclass
    class TestCase:
        onset: TIME_TYPE
        release: TIME_TYPE

        @property
        def dur(self):
            return self.release - self.onset

        def __repr__(self):
            return f"{float(self.onset)}_to_{float(self.release)}"

    # Temp
    four_four = Meter("4/4")
    result = four_four.split_at_metric_strong_points(
        [TestCase(0, 9), TestCase(9, 14), TestCase(14, 18)]
    )
    if not slow:
        random.seed(42)
        p = 0.05
    n_bars = 2
    min_split_dur = 1.0  # TODO
    for ts in Meter.available_time_signatures:
        grid = TIME_TYPE(0.25)
        meter = Meter(ts, min_weight=grid)
        grid_n = int(meter.bar_dur / grid)
        for offset_i in range(grid_n):
            offset = grid * offset_i
            for segment_i in range(2, grid_n * n_bars):
                if not slow and random.random() > p:
                    continue
                dur = segment_i * grid
                segment = [TestCase(offset, offset + dur)]
                result = meter.split_at_metric_strong_points(
                    segment, min_split_dur=min_split_dur
                )
                if dur <= min_split_dur:
                    assert len(result) == 1
                else:
                    if len(result) == 1:
                        assert (
                            len(
                                meter.split_at_metric_strong_points(
                                    segment,
                                    min_split_dur=min_split_dur,
                                    force_split=True,
                                )
                            )
                            > 1
                        )
                    for item in result:
                        onsets = meter.weights_between(
                            grid, item.onset, item.release
                        )
                        assert item.dur <= min_split_dur or all(
                            onsets[0]["weight"] > onset["weight"]
                            for onset in onsets[1:]
                        )


def test_split_odd_duration(slow):
    @dataclass
    class TestCase:
        onset: TIME_TYPE
        release: TIME_TYPE

        @property
        def dur(self):
            return self.release - self.onset

        def __repr__(self):
            return f"{float(self.onset)}_to_{float(self.release)}"

    # Temp
    four_four = Meter("4/4")
    result = four_four.split_at_metric_strong_points(
        [TestCase(0, 9), TestCase(9, 14), TestCase(14, 18)]
    )
    if not slow:
        random.seed(42)
        p = 0.05
    n_bars = 2
    min_split_dur = 1.0
    for ts in Meter.available_time_signatures:

        meter = Meter(ts)
        grid = meter.weight_to_grid[-2]
        grid_n = int(meter.bar_dur / grid)
        for offset_i in range(grid_n):
            offset = grid * offset_i
            for segment_i in range(2, grid_n * n_bars):
                if not slow and random.random() > p:
                    continue
                dur = segment_i * grid
                segment = TestCase(offset, offset + dur)
                result = meter.split_odd_duration(segment, min_split_dur)
                assert (
                    not meter.duration_is_odd(result[0].dur)
                    or result[0].dur <= min_split_dur
                )


def weight_to_grid_doctest():
    # TODO this doctest doesn't run because (apparently) it's under a
    #   cached_property (rather than a property). Figure out a way around that.
    assert sorted(
        (weight, float(grid))
        for (weight, grid) in Meter("4/4").weight_to_grid.items()
    ) == [(-3, 0.125), (-2, 0.25), (-1, 0.5), (0, 1.0), (1, 2.0), (2, 4.0)]

    assert sorted(
        (weight, float(grid))
        for (weight, grid) in Meter("3/4").weight_to_grid.items()
    ) == [(-3, 0.125), (-2, 0.25), (-1, 0.5), (0, 1.0), (1, 3.0)]

    assert sorted(
        (weight, float(grid))
        for (weight, grid) in Meter("12/8").weight_to_grid.items()
    ) == [(-3, 0.125), (-2, 0.25), (-1, 0.5), (0, 1.5), (1, 3.0), (2, 6.0)]
