from dataclasses import dataclass
import textwrap
import typing as t

from dumb_composer.prefabs.prefab_rhythms import match_metric_strength_strs


class MissingPrefabError(Exception):
    pass


@dataclass
class PrefabPitches:
    interval_to_next: t.Union[int, t.Sequence[int]]
    metric_strength_str: str
    relative_degrees: t.Sequence[int]
    # constraints: specifies intervals relative to the initial pitch that
    #   must be contained in the chord.
    #   Thus if constraints=[-2], the prefab could occur starting on the 3rd or
    #   fifth of a triad, but not on the root. If constraints = [-2, -4], it
    #   could occur on only the fifth of a triad. Etc.
    constraints: t.Sequence[int] = ()

    def __post_init__(self):
        if isinstance(self.interval_to_next, int):
            self.interval_to_next = (self.interval_to_next,)
        self.intervals = [
            b - a
            for a, b in zip(
                self.relative_degrees[:-1], self.relative_degrees[1:]
            )
        ]
        assert len(self.relative_degrees) == len(self.metric_strength_str)

    def __len__(self):
        return len(self.metric_strength_str)

    def matches_criteria(
        self,
        interval_to_next: t.Optional[int] = None,
        metric_strength_str: t.Optional[str] = None,
        relative_chord_factors: t.Optional[int] = None,
        # harmonic_interval_above_bass: t.Optional[int] = None,
        # chord_intervals: t.Optional[t.Tuple[int]] = None,
    ) -> bool:
        if (
            interval_to_next is not None
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
        return True


PP = PrefabPitches

PREFABS = (
    PP([0, -2], "s___", [0, 1, 0, -1]),
    PP([0, 2], "s___", [0, -1, 0, 1]),
    PP([0, -2], "s___", [0, -3, -2, -1], [-3]),
    PP([1, -2, -3], "_w__", [0, 1, -1, 0]),
    PP([-1, 1, 3, 4, 5, 7], "_w__", [0, 1, 2, 0], [2]),
    PP([-1, 1, 3, 4], "____", [0, 2, 1, 0], constraints=[2]),
    # PP([-6], )
    # triad arpeggiations up
    PP([2], "swsw", [0, 2, 4, 3], constraints=[2, 4]),
    PP([3], "swsw", [0, 2, 5, 4], constraints=[2, 5]),
    PP([3], "swsw", [0, 3, 5, 4], constraints=[3, 5]),
    PP([7], "swsw", [0, 2, 4, 6], constraints=[2, 4]),
    PP([7], "swsw", [0, 2, 5, 6], constraints=[2, 5]),
    PP([7], "swsw", [0, 3, 5, 6], constraints=[3, 5]),
    PP([6, 9], "____", [0, 2, 5, 7], constraints=[2, 5]),
    PP([6], "____", [0, 3, 5, 7], constraints=[3, 5]),
    PP([4, 6, 9], "____", [0, 2, 4, 7], constraints=[2, 4]),
    # triad arpeggiations down
    PP([-6, -3, -5], "____", [0, -3, -5, -7], constraints=[-3, -5]),
)


class PrefabPitchDirectory:
    def __init__(self):
        self._memo = {}

    def __call__(
        self,
        melodic_interval: int,
        metric_strength_str: str,
        relative_chord_factors: t.Sequence[int],
    ) -> t.List[PrefabPitches]:
        tup = (
            melodic_interval,
            metric_strength_str,
            relative_chord_factors,
        )
        if tup in self._memo:
            return self._memo[tup].copy()
        out = [prefab for prefab in PREFABS if prefab.matches_criteria(*tup)]
        if not out:
            raise MissingPrefabError(
                textwrap.dedent(
                    f"""No PrefabPitches instance matching criteria
                    \tinterval_to_next: {melodic_interval}
                    \tmetric_strength_str: {metric_strength_str}
                    \trelative_chord_factors: {relative_chord_factors}
                    """
                )
            )
        self._memo[tup] = out
        return out.copy()
