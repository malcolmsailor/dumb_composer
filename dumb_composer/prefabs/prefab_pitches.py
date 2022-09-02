from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Number
import re
import textwrap
import typing as t

from dumb_composer.shared_classes import Allow
from dumb_composer.prefabs.prefab_rhythms import (
    match_metric_strength_strs,
    MissingPrefabError,
)

RELATIVE_DEGREE_REGEX = re.compile(
    r"""
        ^(?P<alteration>[#b]?)
        (?P<relative_degree>-?\d+)
        (?P<ioi_constraints>(?:[Mm](?:(?:\d+/\d+)|(?:\d+(?:\.\d+)?))){0,2})
        (?P<tie_to_next>t?)$
    """,
    flags=re.VERBOSE,
)

MAX_IOI_REGEX = re.compile(r"M(?:(?P<frac>\d/\d)|(?P<float>\d+(?:\.\d+)?))")
MIN_IOI_REGEX = re.compile(r"m(?:(?P<frac>\d/\d)|(?P<float>\d+(?:\.\d+)?))")


@dataclass(frozen=True, init=False)
class SingletonPitch:
    relative_degrees: t.Tuple[int] = (0,)
    tie_to_next: bool = False
    alterations: t.Tuple = ()

    def matches_criteria(self, *args, **kwargs):
        return True


@dataclass
class PrefabPitches:
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

    interval_to_next: t.Optional[t.Union[int, t.Sequence[int]]]
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

    # __post_init__ defines the following attributes:
    alterations: t.Dict[int, str] = field(default_factory=dict, init=False)
    tie_to_next: bool = field(default=False, init=False)
    intervals: t.Optional[t.List[int]] = field(default=None, init=False)
    min_iois: t.Dict[int, Number] = field(default_factory=dict, init=False)
    max_iois: t.Dict[int, Number] = field(default_factory=dict, init=False)

    def __post_init__(self):
        assert len(self.relative_degrees) == len(self.metric_strength_str)
        temp_degrees = []
        for i, relative_degree in enumerate(self.relative_degrees):
            m = re.match(RELATIVE_DEGREE_REGEX, str(relative_degree))
            if not m:
                raise ValueError(f"Illegal relative degree {relative_degree}")
            temp_degrees.append(int(m.group("relative_degree")))
            if m.group("alteration"):
                self.alterations[i] = m.group("alteration")
            if m.group("ioi_constraints"):
                max_ioi_m = re.search(MAX_IOI_REGEX, m.group("ioi_constraints"))
                if max_ioi_m:
                    if max_ioi_m.group("float"):
                        self.max_iois[i] = float(max_ioi_m.group("float"))
                    else:
                        self.max_iois[i] = Fraction(max_ioi_m.group("frac"))
                min_ioi_m = re.search(MIN_IOI_REGEX, m.group("ioi_constraints"))
                if min_ioi_m:
                    if min_ioi_m.group("float"):
                        self.min_iois[i] = float(min_ioi_m.group("float"))
                    else:
                        self.min_iois[i] = Fraction(min_ioi_m.group("frac"))
            if m.group("tie_to_next"):
                if i != len(self.relative_degrees) - 1:
                    raise ValueError(
                        "only last relative degree can be tied to next"
                    )
                if len(self.interval_to_next) != 1:
                    raise ValueError(
                        "if last relative degree is tied, interval_to_next "
                        "can only have length 1, but interval_to_next is "
                        f"{self.interval_to_next}"
                    )
                if (
                    not int(m.group("relative_degree"))
                    == self.interval_to_next[0]
                ):
                    raise ValueError(
                        f"tied final relative degree {m.group('relative_degree')}"
                        f" != interval_to_next {self.interval_to_next}"
                    )
                self.tie_to_next = True

        self.relative_degrees = temp_degrees
        if isinstance(self.interval_to_next, int):
            self.interval_to_next = (self.interval_to_next,)
        self.intervals = [
            b - a
            for a, b in zip(
                self.relative_degrees[:-1], self.relative_degrees[1:]
            )
        ]

    def __len__(self):
        return len(self.metric_strength_str)

    def matches_criteria(
        self,
        interval_to_next: t.Optional[int] = None,
        metric_strength_str: t.Optional[str] = None,
        relative_chord_factors: t.Optional[int] = None,
        is_suspension: bool = False,
        is_preparation: bool = False,
        interval_is_diatonic: bool = True,
    ) -> bool:
        if (
            None not in (interval_to_next, self.interval_to_next)
            and interval_to_next not in self.interval_to_next
        ):
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
        if not interval_is_diatonic and self.tie_to_next:
            # we only want to tie diatonic notes ("chromatic" notes will
            #   change in the next harmony)
            return False
        return True

    def __hash__(self):
        """We want each PrefabPitches instance to be considered unique so we
        just return the id of the instance."""
        return id(self)


PP = PrefabPitches

TWO_PREFABS = (
    PP([-1, 2, 3], "__", [0, -2], [-2]),
    PP([-1, -2, 2], "__", [0, -3], [-3], allow_suspension=Allow.NO),
    PP([1, 3, 4, 5, -2, -3], "__", [0, 2], [2]),
    PP(
        [1, 2, 5],
        "__",
        [0, 3],
        [3],
        negative_constraints=[1],
        allow_suspension=Allow.NO,
    ),
)

THREE_PREFABS = (
    PP([0, -2, -3, 1, 2], "___", [0, -3, 0], [-3]),
    PP([-2], "__w", [0, -3, -1], [-3]),
    PP(None, "___", [0, -1, 0], allow_preparation=Allow.YES),
    PP(
        None,
        "___",
        [0, "#-1", 0],
        allow_preparation=Allow.YES,
        allow_suspension=Allow.NO,
    ),
    PP(
        [4],
        "___",
        [0, 2, "4t"],
        constraints=[2, 4],
        allow_preparation=Allow.YES,
    ),
    PP([2], "___", [0, 1, "2t"], constraints=[2], allow_preparation=Allow.YES),
    PP(
        [2],
        "___",
        [0, -2, "2t"],
        constraints=[-2, 2],
        allow_preparation=Allow.YES,
    ),
    PP(
        [2],
        "___",
        [0, -3, "2t"],
        constraints=[-3, 2],
        allow_preparation=Allow.YES,
    ),
)

FOUR_PREFABS = (
    PP([0], "s___", [0, 1, 0, "#-1"]),
    PP([-2], "s___", [0, 1, 0, -1]),
    PP([0, 2], "s___", [0, "#-1", 0, 1]),
    PP([0, -2], "s___", [0, -3, -2, -1], [-3]),
    PP([1, -2, -3], "_w__", [0, 1, "#-1", 0], allow_suspension=Allow.NO),
    PP([-1, 1, 3, 4, 5, 7], "_w__", [0, 1, 2, 0], [2]),
    PP([-1, 1, 3, 4], "____", [0, 2, 1, 0], constraints=[2]),
    PP([-1], "____", [0, 2, 1, 0], constraints=[1, 3]),
    PP([-1, -3, 1], "____", [0, -2, -4, 0], constraints=[-2, -4]),
    PP([-1, -3, 0], "____", [0, -2, -4, -2], constraints=[-2, -4]),
    # PP([-6], )
    # triad arpeggiations up
    PP([2], "___w", [0, 2, 4, 3], constraints=[2, 4]),
    PP([3], "___w", [0, 2, 5, 4], constraints=[2, 5]),
    PP([3], "___w", [0, 3, 5, 4], constraints=[3, 5]),
    PP(
        [7],
        "___w",
        [0, 2, 4, "#6"],
        constraints=[2, 4],
        negative_constraints=[6],
    ),
    PP([7], "___w", [0, 2, 5, "#6"], constraints=[2, 5]),
    PP([7], "___w", [0, 3, 5, "#6"], constraints=[3, 5]),
    PP([6, 9], "____", [0, 2, 5, 7], constraints=[2, 5]),
    PP([6], "____", [0, 3, 5, 7], constraints=[3, 5]),
    PP([4, 6, 9], "____", [0, 2, 4, 7], constraints=[2, 4]),
    # triad arpeggiations down
    PP([-6, -3, -5], "____", [0, -3, -5, -7], constraints=[-3, -5]),
    PP([-7, -4], "____", [0, -2, -4, -2], constraints=[-2, -4]),
)

ASC_SCALE_FRAGMENTS = tuple(
    PP([i], "_" * i, list(range(i))) for i in range(1, 12)
)
DESC_SCALE_FRAGMENTS = tuple(
    PP([i], "_" * abs(i), list(range(0, i, -1))) for i in range(-1, -12, -1)
)
PREFABS = (
    TWO_PREFABS
    + THREE_PREFABS
    + FOUR_PREFABS
    + ASC_SCALE_FRAGMENTS
    + DESC_SCALE_FRAGMENTS
)


class PrefabPitchDirectory:
    def __init__(self, allow_singleton_pitch: bool = True):
        self._memo = {}
        self._singleton = SingletonPitch() if allow_singleton_pitch else None

    def __call__(
        self,
        interval_to_next: int,
        metric_strength_str: str,
        relative_chord_factors: t.Sequence[int],
        is_suspension: bool = False,
        is_preparation: bool = False,
        interval_is_diatonic: bool = True,
    ) -> t.List[PrefabPitches]:
        tup = (
            interval_to_next,
            metric_strength_str,
            relative_chord_factors,
            is_suspension,
            is_preparation,
            interval_is_diatonic,
        )
        if tup in self._memo:
            return self._memo[tup].copy()
        out = [prefab for prefab in PREFABS if prefab.matches_criteria(*tup)]
        if not out:
            if len(metric_strength_str) == 1 and self._singleton is not None:
                out = [self._singleton]
            else:
                raise MissingPrefabError(
                    textwrap.dedent(
                        f"""No PrefabPitches instance matching criteria
                        \tinterval_to_next: {interval_to_next}
                        \tmetric_strength_str: {metric_strength_str}
                        \trelative_chord_factors: {relative_chord_factors}
                        \tis_suspension: {is_suspension}
                        \tis_preparation: {is_preparation}
                        \tinterval_is_diatonic: {interval_is_diatonic}
                    """
                    )
                )
        self._memo[tup] = out
        return out.copy()
