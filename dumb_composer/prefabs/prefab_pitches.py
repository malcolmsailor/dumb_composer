import re
import textwrap
import typing as t
from abc import abstractmethod
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property
from numbers import Number
from pathlib import Path

import yaml

from dumb_composer.chords.intervals import (
    ascending_chord_intervals,
    ascending_chord_intervals_within_range,
    descending_chord_intervals_within_range,
)
from dumb_composer.pitch_utils.types import BASS, TIME_TYPE, ScalarInterval, Voice
from dumb_composer.prefabs.prefab_rhythms import (
    MissingPrefabError,
    PrefabBase,
    match_metric_strength_strs,
)
from dumb_composer.shared_classes import Allow

RELATIVE_DEGREE_REGEX = re.compile(
    r"""
        ^(?P<alteration>[#b]?)
        (?P<lower_auxiliary>L?)
        (?P<relative_degree>-?\d+)
        (?P<ioi_constraints>(?:[Mm](?:(?:\d+/\d+)|(?:\d+(?:\.\d+)?))){0,2})
        (?P<tie_to_next>t?)$
    """,
    flags=re.VERBOSE,
)

MAX_IOI_REGEX = re.compile(r"M(?:(?P<frac>\d/\d)|(?P<float>\d+(?:\.\d+)?))")
MIN_IOI_REGEX = re.compile(r"m(?:(?P<frac>\d/\d)|(?P<float>\d+(?:\.\d+)?))")


# TODO: (Malcolm 2023-08-02) write code to find all prefabs that can be placed in
#   rhythmic unison parallel motion against one another


class PrefabPitchBase(PrefabBase):
    relative_degrees: t.Sequence
    tie_to_next: bool
    alterations: tuple | dict
    lower_auxiliaries: tuple | set

    def stepwise_internally(self) -> bool:
        raise NotImplementedError

    def returns_to_main_pitch(self) -> bool:
        raise NotImplementedError

    def matches_criteria(self, *args, **kwargs) -> bool:
        raise NotImplementedError

    @cached_property
    def interval_sum(self) -> int:
        raise NotImplementedError

    def approaches_arrival_pitch_by_step(self, interval_to_next) -> bool:
        return abs(self.interval_sum - interval_to_next) == 1

    def approaches_arrival_pitch_obliquely(
        self, interval_to_next: ScalarInterval
    ) -> bool:
        return abs(self.interval_sum - interval_to_next) == 0


@dataclass(frozen=True, init=False)
class SingletonPitch(PrefabPitchBase):
    relative_degrees: t.Tuple[int] = (0,)
    tie_to_next: bool = False
    alterations: t.Tuple = ()
    lower_auxiliaries: t.Tuple = ()

    @cached_property
    def interval_sum(self) -> int:
        return 0

    def matches_criteria(self, *args, **kwargs):
        return True

    def stepwise_internally(self) -> bool:
        return True

    def returns_to_main_pitch(self) -> bool:
        return True


@dataclass
class PrefabPitches(PrefabPitchBase):
    """
    >>> pp = PrefabPitches(
    ...     interval_to_next=[1, 3],
    ...     metric_strength_str="__",
    ...     relative_degrees=[0, 2],
    ...     constraints=[2],
    ... )
    >>> pp.matches_criteria(1, "sw", relative_chord_factors=[2, 4])
    True
    >>> pp.matches_criteria(1, "sw", relative_chord_factors=[3, 5])
    False
    >>> pp.stepwise_internally()
    False

    If interval_to_next is None, matches anything:

    >>> pp = PrefabPitches(
    ...     interval_to_next=None,
    ...     metric_strength_str="_",
    ...     relative_degrees=[0],
    ...     constraints=[2],
    ... )
    >>> pp.matches_criteria(5, "s", relative_chord_factors=[2, 4])
    True
    >>> pp.matches_criteria(173, "s", relative_chord_factors=[2, 4])
    True

    For more specificity, relative degrees can be strings.

    Alterations to relative degrees are specified with "#" and "b":
    >>> pp = PrefabPitches(
    ...     interval_to_next=None,
    ...     metric_strength_str="___",
    ...     relative_degrees=[0, "#-1", 0],
    ... )
    >>> pp.alterations
    {1: '#'}
    >>> pp.stepwise_internally()
    True

    Minimum and maximum durations for relative degrees can be specified with 'm'
    and 'M' respectively following the relative degree. These can be indicated
    as ints (will be cast to floats), floats, or Fractions.
    >>> pp = PrefabPitches(
    ...     interval_to_next=None,
    ...     metric_strength_str="___",
    ...     relative_degrees=[0, "#-1M2m0.25", "0m2/3M4/1"],
    ... )
    >>> pp.min_iois
    {1: 0.25, 2: Fraction(2, 3)}
    >>> pp.max_iois
    {1: 2.0, 2: Fraction(4, 1)}

    If the last relative degree should be tied to the first note of the next
    pattern, this can be indicated by appending 't' to the end of the string.
    >>> pp = PrefabPitches(
    ...     interval_to_next=[2],
    ...     metric_strength_str="___",
    ...     relative_degrees=[0, 1, "2t"],
    ... )
    >>> pp.tie_to_next
    True
    """

    interval_to_next: int | t.Sequence[int] | None
    metric_strength_str: str
    relative_degrees: t.Sequence[t.Union[int, str]]
    # constraints: specifies intervals relative to the initial pitch that
    #   must be contained in the chord.
    #   Thus if constraints=[-2], the prefab could occur starting on the 3rd or
    #   fifth of a triad, but not on the root. If constraints = [-2, -4], it
    #   could occur on only the fifth of a triad. Etc.
    constraints: t.Sequence[int] = ()
    # negative_constraints: specifies intervals relative to the initial pitch that
    #   must NOT be contained in the chord.
    negative_constraints: t.Sequence[int] = ()
    allow_suspension: Allow = Allow.YES
    allow_preparation: Allow = Allow.NO
    # allow_resolution: Allow = Allow.YES # Not yet implemented
    avoid_interval_to_next: t.Sequence[int] = ()
    avoid_voices: t.Container[Voice] = frozenset()

    # __post_init__ defines the following attributes:
    alterations: t.Dict[int, str] = field(default_factory=dict, init=False)
    lower_auxiliaries: set[int] = field(default_factory=set, init=False)
    tie_to_next: bool = field(default=False, init=False)
    intervals: t.Optional[t.List[int]] = field(default=None, init=False)
    min_iois: t.Dict[int, TIME_TYPE] = field(default_factory=dict, init=False)
    max_iois: t.Dict[int, TIME_TYPE] = field(default_factory=dict, init=False)

    def __post_init__(self):
        assert len(self.relative_degrees) == len(self.metric_strength_str)
        if isinstance(self.interval_to_next, int):
            self.interval_to_next = (self.interval_to_next,)
        temp_degrees = []
        for i, relative_degree in enumerate(self.relative_degrees):
            m = re.match(RELATIVE_DEGREE_REGEX, str(relative_degree))
            if not m:
                raise ValueError(f"Illegal relative degree {relative_degree}")
            temp_degrees.append(int(m.group("relative_degree")))
            if m.group("alteration"):
                self.alterations[i] = m.group("alteration")
            if m.group("lower_auxiliary"):
                self.lower_auxiliaries.add(i)
            if m.group("ioi_constraints"):
                max_ioi_m = re.search(MAX_IOI_REGEX, m.group("ioi_constraints"))
                if max_ioi_m:
                    if max_ioi_m.group("float"):
                        self.max_iois[i] = float(  # type:ignore
                            max_ioi_m.group("float")
                        )
                    else:
                        self.max_iois[i] = Fraction(max_ioi_m.group("frac"))
                min_ioi_m = re.search(MIN_IOI_REGEX, m.group("ioi_constraints"))
                if min_ioi_m:
                    if min_ioi_m.group("float"):
                        self.min_iois[i] = float(  # type:ignore
                            min_ioi_m.group("float")
                        )
                    else:
                        self.min_iois[i] = Fraction(min_ioi_m.group("frac"))
            if m.group("tie_to_next"):
                if i != len(self.relative_degrees) - 1:
                    raise ValueError("only last relative degree can be tied to next")
                if self.interval_to_next is not None:
                    if len(self.interval_to_next) != 1:
                        raise ValueError(
                            "if last relative degree is tied, interval_to_next "
                            "can only have length 1, but interval_to_next is "
                            f"{self.interval_to_next}"
                        )
                    if not int(m.group("relative_degree")) == self.interval_to_next[0]:
                        raise ValueError(
                            f"tied final relative degree {m.group('relative_degree')}"
                            f" != interval_to_next {self.interval_to_next}"
                        )
                self.tie_to_next = True
        self.relative_degrees = temp_degrees
        self.intervals = [
            b - a  # type:ignore
            for a, b in zip(self.relative_degrees[:-1], self.relative_degrees[1:])
        ]

    def __len__(self):
        return len(self.metric_strength_str)

    @cached_property
    def min_relative_degree(self) -> int:
        out = min(self.relative_degrees)
        if isinstance(out, str):
            breakpoint()
        else:
            return out
        raise NotImplementedError

    def matches_criteria(
        self,
        interval_to_next: t.Optional[int] = None,
        metric_strength_str: t.Optional[str] = None,
        relative_chord_factors: t.Container[int] | None = None,
        is_suspension: bool = False,
        is_preparation: bool = False,
        interval_is_diatonic: bool = True,
        voice: Voice | None = None,
    ) -> bool:
        # TODO: (Malcolm 2023-08-01) other criteria we would like to consider:
        #   - whether the note is chromatically raised/lowered
        if interval_to_next is not None:
            if (
                self.interval_to_next is not None
                and interval_to_next not in self.interval_to_next  # type:ignore
            ):
                return False
            if interval_to_next in self.avoid_interval_to_next:
                return False
        if metric_strength_str is not None and not match_metric_strength_strs(
            metric_strength_str, self.metric_strength_str
        ):
            return False
        if (
            self.constraints
            and relative_chord_factors is not None
            and any(
                constraint not in relative_chord_factors
                for constraint in self.constraints
            )
        ):
            return False
        if (
            self.negative_constraints
            and relative_chord_factors is not None
            and any(
                constraint in relative_chord_factors
                for constraint in self.negative_constraints
            )
        ):
            return False
        if is_suspension and self.allow_suspension == Allow.NO:
            return False
        if not is_suspension and self.allow_suspension == Allow.ONLY:
            return False
        if is_preparation and self.allow_preparation == Allow.NO:
            return False
        if not is_preparation and self.allow_preparation == Allow.ONLY:
            return False
        if (not interval_is_diatonic) and self.tie_to_next:
            # we only want to tie diatonic notes ("chromatic" notes will
            #   change in the next harmony)
            return False
        if voice is not None and voice in self.avoid_voices:
            return False
        if voice is BASS:
            if self.min_relative_degree < -1:
                return False
        return True

    def __hash__(self):
        """We want each PrefabPitches instance to be considered unique so we
        just return the id of the instance."""
        return id(self)

    @cached_property
    def interval_sum(self) -> int:
        return sum(self.intervals)  # type:ignore

    def stepwise_internally(self) -> bool:
        for i in self.intervals:  # type:ignore
            if abs(i) > 1:
                return False
        return True

    def returns_to_main_pitch(self) -> bool:
        return 0 == self.relative_degrees[-1] == self.relative_degrees[0]


# PP = PrefabPitches

# TWO_PREFABS = (
#     PP([0], "__", relative_degrees=[0, -1], negative_constraints=[-1]),
#     PP([0], "__", relative_degrees=[0, 1]),
#     PP([-1, 2, 3], "__", [0, -2], [-2]),
#     PP(
#         [-1, -2, 2],
#         "__",
#         [0, -3],
#         [-3],
#         allow_suspension=Allow.NO,
#         negative_constraints=[-5],
#     ),
#     PP([1, 3, 4, 5, -2, -3], "__", [0, 2], [2]),
#     PP(
#         [1, 2, 5],
#         "__",
#         [0, 3],
#         [3],
#         negative_constraints=[1],
#         allow_suspension=Allow.NO,
#     ),
# )

# THREE_PREFABS = (
#     PP([0, -2, -3, 1, 2], "___", [0, -3, 0], [-3], negative_constraints=[-2]),
#     PP([-2], "__w", [0, -3, -1], [-3]),
#     PP(
#         None,
#         "___",
#         [0, -1, 0],
#         allow_preparation=Allow.YES,
#         avoid_interval_to_next=(-1,),
#     ),
#     # TODO: (Malcolm 2023-08-13) I'm wondering if rather than explicitly specifying
#     #   sharps in cases like this there should be a flag to indicate "lower auxiliary"
#     #   and a probability of chromatically raising lower auxiliaries in cases where
#     #   they are less than a certain length.
#     #   It would also be useful to set a certain maximum length for lower auxiliaries
#     #   regardless.
#     PP(
#         None,
#         "___",
#         [0, "#-1", 0],
#         allow_preparation=Allow.YES,
#         allow_suspension=Allow.NO,
#         avoid_interval_to_next=(-1,),
#     ),
#     PP(
#         [4],
#         "___",
#         [0, 2, "4t"],
#         constraints=[2, 4],
#         allow_preparation=Allow.YES,
#     ),
#     PP([2], "___", [0, 1, "2t"], constraints=[2], allow_preparation=Allow.YES),
#     PP(
#         [2],
#         "___",
#         [0, -2, "2t"],
#         constraints=[-2, 2],
#         allow_preparation=Allow.YES,
#     ),
#     PP(
#         [2],
#         "___",
#         [0, -3, "2t"],
#         constraints=[-3, 2],
#         allow_preparation=Allow.YES,
#     ),
# )

# FOUR_PREFABS = (
#     PP([0], "s___", [0, 1, 0, "#-1"]),
#     PP([-2], "s___", [0, 1, 0, -1]),
#     PP([0, 2], "s___", [0, "#-1", 0, 1]),
#     PP([0, -2], "s___", [0, -3, -2, -1], [-3], negative_constraints=[-5]),
#     PP([1, -2, -3], "_w__", [0, 1, "#-1", 0], allow_suspension=Allow.NO),
#     PP([-1, 1, 3, 4, 5, 7], "_w__", [0, 1, 2, 0], [2]),
#     PP([-1, 1, 3, 4], "____", [0, 2, 1, 0], constraints=[2]),
#     PP([-1], "____", [0, 2, 1, 0], constraints=[1, 3]),
#     PP([-1, -3, 1], "____", [0, -2, -4, 0], constraints=[-2, -4]),
#     PP([-1, -3, 0], "____", [0, -2, -4, -2], constraints=[-2, -4]),
#     # PP([-6], )
#     # triad arpeggiations up
#     PP([2], "___w", [0, 2, 4, 3], constraints=[2, 4]),
#     PP([3], "___w", [0, 2, 5, 4], constraints=[2, 5]),
#     PP([3], "___w", [0, 3, 5, 4], constraints=[3, 5]),
#     PP(
#         [7],
#         "___w",
#         [0, 2, 4, "#6"],
#         constraints=[2, 4],
#         negative_constraints=[6],
#     ),
#     PP([7], "___w", [0, 2, 5, "#6"], constraints=[2, 5]),
#     PP([7], "___w", [0, 3, 5, "#6"], constraints=[3, 5]),
#     PP([6, 9], "____", [0, 2, 5, 7], constraints=[2, 5]),
#     PP([6], "____", [0, 3, 5, 7], constraints=[3, 5]),
#     PP([4, 6, 9], "____", [0, 2, 4, 7], constraints=[2, 4]),
#     # triad arpeggiations down
#     PP([-6, -3, -5], "____", [0, -3, -5, -7], constraints=[-3, -5]),
#     PP([-7, -4], "____", [0, -2, -4, -2], constraints=[-2, -4]),
# )

# ASC_SCALE_FRAGMENTS = tuple(PP([i], "_" * i, list(range(i))) for i in range(1, 12))
# DESC_SCALE_FRAGMENTS = tuple(
#     PP([i], "_" * abs(i), list(range(0, i, -1))) for i in range(-1, -12, -1)
# )
# PREFABS = (
#     TWO_PREFABS
#     + THREE_PREFABS
#     + FOUR_PREFABS
#     + ASC_SCALE_FRAGMENTS
#     + DESC_SCALE_FRAGMENTS
# )


def get_scale_prefabs(max_scale=12):
    def _get_asc_scales_of_extent(
        scale_extent: int,
        chords: tuple[tuple[int, ...], ...] = ((0, 2, 4), (0, 2, 5), (0, 3, 5)),
    ):
        out = []
        steps = list(range(scale_extent))
        for chord in chords:
            # We assume that the first item in chord is 0
            assert chord[0] == 0
            chord_steps = set(
                ascending_chord_intervals_within_range(chord, scale_extent)
            )
            metric_accumulator = []
            for i in range(scale_extent):
                # If the step is a non-chord tone with chord tones on either side,
                #   it should be weak. (In the case of scalar fourths, we are agnostic
                #   what the metric strength of the intervening notes should be)
                if (
                    i not in chord_steps
                    and i - 1 in chord_steps
                    and i + 1 in chord_steps
                ):
                    metric_accumulator.append("w")
                else:
                    metric_accumulator.append("_")
            metric_strength_str = "".join(metric_accumulator)
            out.append(
                PrefabPitches(
                    scale_extent,
                    metric_strength_str,
                    relative_degrees=steps,
                    constraints=chord[1:],
                )
            )
        return out

    def _get_desc_scales_of_extent(
        scale_extent: int,
        chords: tuple[tuple[int, ...], ...] = ((0, 2, 4), (0, 2, 5), (0, 3, 5)),
    ):
        assert scale_extent < 0
        out = []
        steps = list(range(0, scale_extent, -1))
        for chord in chords:
            # We assume that the first item in chord is 0
            assert chord[0] == 0
            chord_steps = set(
                descending_chord_intervals_within_range(chord, scale_extent)
            )
            metric_accumulator = []
            for i in range(0, scale_extent, -1):
                # If the step is a non-chord tone with chord tones on either side,
                #   it should be weak. (In the case of scalar fourths, we are agnostic
                #   what the metric strength of the intervening notes should be)
                if (
                    i not in chord_steps
                    and i - 1 in chord_steps
                    and i + 1 in chord_steps
                ):
                    metric_accumulator.append("w")
                else:
                    metric_accumulator.append("_")
            metric_strength_str = "".join(metric_accumulator)
            out.append(
                PrefabPitches(
                    scale_extent,
                    metric_strength_str,
                    relative_degrees=steps,
                    constraints=chord[1:],
                )
            )
        return out

    out = []
    for scale_extent in range(2, max_scale + 1):
        out += _get_asc_scales_of_extent(scale_extent)
        out += _get_desc_scales_of_extent(-scale_extent)

    return out


class PrefabPitchDirectory:
    def __init__(
        self,
        config_path: str | Path,
        auto_add_scales: bool = True,
        allow_singleton_pitch: bool = True,
    ):
        self._memo = {}
        self._singleton = SingletonPitch() if allow_singleton_pitch else None
        with open(config_path, "r") as yaml_file:
            config_list = yaml.safe_load(yaml_file)
        self._prefabs: list[PrefabPitchBase] = []
        for prefab_kwargs in config_list:
            metric_stength_arg = prefab_kwargs["metric_strength_str"]
            if (not isinstance(metric_stength_arg, str)) and len(
                metric_stength_arg
            ) > 1:
                for metric_strength_str in metric_stength_arg:
                    self._prefabs.append(
                        PrefabPitches(
                            **(
                                prefab_kwargs
                                | {"metric_strength_str": metric_strength_str}
                            )
                        )
                    )
            else:
                self._prefabs.append(PrefabPitches(**prefab_kwargs))
        if auto_add_scales:
            self._prefabs += get_scale_prefabs()

    def __call__(
        self,
        interval_to_next: int,
        metric_strength_str: str,
        voice: Voice,
        relative_chord_factors: t.Sequence[int],
        is_suspension: bool = False,
        is_preparation: bool = False,
        interval_is_diatonic: bool = True,
    ) -> t.List[PrefabPitchBase]:
        tup = (
            interval_to_next,
            metric_strength_str,
            relative_chord_factors,
            is_suspension,
            is_preparation,
            interval_is_diatonic,
            voice,
        )
        if tup in self._memo:
            return self._memo[tup].copy()
        out: list[PrefabPitchBase] = [
            prefab for prefab in self._prefabs if prefab.matches_criteria(*tup)
        ]
        if not out:
            if len(metric_strength_str) == 1 and self._singleton is not None:
                out = [self._singleton]
            else:
                raise MissingPrefabError(
                    textwrap.dedent(
                        f"""No PrefabPitches instance matching criteria
                        \tinterval_to_next: {interval_to_next}
                        \tmetric_strength_str: {metric_strength_str}
                        \tvoice: {voice}
                        \trelative_chord_factors: {relative_chord_factors}
                        \tis_suspension: {is_suspension}
                        \tis_preparation: {is_preparation}
                        \tinterval_is_diatonic: {interval_is_diatonic}"""
                    )
                )
        self._memo[tup] = out
        return out.copy()
