import itertools as it
import logging
import random
import typing as t
from dataclasses import asdict, dataclass

from voice_leader import (
    CardinalityDiffersTooMuch,
    NoMoreVoiceLeadingsError,
    voice_lead_pitches,
)

from dumb_composer.constants import (
    CLOSE_REGISTERS,
    DEFAULT_ACCOMP_RANGE,
    DEFAULT_BASS_RANGE,
    HI_PITCH,
    KEYBOARD_STYLE_REGISTERS,
    LOW_PITCH,
    OPEN_REGISTERS,
    TET,
)
from dumb_composer.pitch_utils.put_in_range import get_all_in_range, put_in_range
from dumb_composer.pitch_utils.spacings import SpacingConstraints, validate_spacing
from dumb_composer.shared_classes import Allow
from dumb_composer.utils.attr_compiler import attr_compiler

from .utils.recursion import UndoRecursiveStep


class NoSpacings(UndoRecursiveStep):
    pass


def spacing_method(f):
    setattr(f, "spacing_method", True)
    return f


# Older version of spacing_method had an exterior wrapper that could take
# kwargs... not sure why.
# def spacing_method(**kwargs):
#     def wrap(f):
#         setattr(f, "spacing_method", True)
#         return f
#     return wrap


# @attr_compiler("_all_spacings", "spacing_method")
# class ChordSpacer:
#     """
#     >>> cs = ChordSpacer()
#     >>> cs([0, 4, 7], spacing="keyboard_style", register=0)
#     [48, 64, 67]
#     >>> cs([0, 4, 7], spacing="open_position", register=1)
#     [48, 55, 64]
#     >>> cs([2, 5, 9], spacing="open_position", register=1) # NB parallel 5ths
#     [50, 57, 65]
#     """

#     def __init__(self):
#         self._prev_pitches = None
#         self._prev_spacing = None

#     def __call__(
#         self,
#         pcs: t.Iterable[int],
#         init: bool = False,
#         spacing: t.Optional[str] = None,
#         register: t.Optional[int] = None,
#         l_bound: t.Optional[int] = None,
#         u_bound: t.Optional[int] = None,
#     ):
#         """
#         Creates a new spacing if:
#             - it is the first time the instance is called
#             - `spacing` is not None and differs from the last time the instance
#                  was called
#             - `init` is True
#         (If spacing is None and creating a new spacing, then randomly chooses
#         from close_position, open_position, or keyboard_style.)

#         Otherwise, voice-leads from previous chord.
#         """
#         if (
#             self._prev_pitches is None
#             or (spacing is not None and spacing != self._prev_spacing)
#             or init
#         ):
#             pitches, spacing = self._init_pitches(
#                 pcs,
#                 spacing=spacing,
#                 register=register,
#                 l_bound=l_bound,
#                 u_bound=u_bound,
#             )
#             self._prev_spacing = spacing
#         else:
#             pitches = voice_lead_pitches(
#                 self._prev_pitches,
#                 pcs,
#                 preserve_bass=True,
#                 min_pitch=l_bound,
#                 max_pitch=u_bound,
#             )
#         self._prev_pitches = pitches
#         return pitches

#     @spacing_method
#     @staticmethod
#     def keyboard_style(
#         pcs,
#         register: t.Optional[int] = None,
#         n: t.Optional[int] = None,
#         l_bound: t.Optional[int] = None,
#         u_bound: t.Optional[int] = None,
#     ):
#         """
#         >>> ChordSpacer.keyboard_style([0, 4, 7], register=0)
#         [48, 64, 67]
#         >>> ChordSpacer.keyboard_style([0, 4, 7], register=0, n=4)
#         [48, 64, 67, 72]
#         """
#         if register is None:
#             register = random.randrange(len(KEYBOARD_STYLE_REGISTERS))
#         (bass_l_bound, _), (chord_l_bound, _) = KEYBOARD_STYLE_REGISTERS[
#             register
#         ]
#         bass_l_bound = (
#             max(l_bound, bass_l_bound) if l_bound is not None else bass_l_bound
#         )
#         if n is None:
#             n = len(pcs)
#         out = [
#             put_in_range(pcs[0], low=bass_l_bound),
#             put_in_range(pcs[1 % len(pcs)], low=chord_l_bound),
#         ]
#         for i in range(2, n):
#             pc = pcs[i % len(pcs)]
#             prev_pc = out[-1]
#             interval = (pc - prev_pc) % TET
#             out.append(prev_pc + interval)
#         return out

#     @spacing_method
#     @staticmethod
#     def close_position(
#         pcs,
#         register: t.Optional[int] = None,
#         n: t.Optional[int] = None,
#         l_bound: t.Optional[int] = None,
#         u_bound: t.Optional[int] = None,
#     ):
#         """
#         >>> ChordSpacer.close_position([2, 5, 9], register=1)
#         [74, 77, 81]
#         """
#         if register is None:
#             register = random.randrange(len(CLOSE_REGISTERS))
#         rl_bound, _ = CLOSE_REGISTERS[register]
#         l_bound = max(l_bound, rl_bound) if l_bound is not None else rl_bound
#         if n is None:
#             n = len(pcs)
#         out = [put_in_range(pcs[0], low=l_bound)]
#         for i in range(1, n):
#             pc = pcs[i % len(pcs)]
#             prev_pc = out[-1]
#             interval = (pc - prev_pc) % TET
#             out.append(prev_pc + interval)
#         return out

#     @spacing_method
#     @staticmethod
#     def open_position(
#         pcs,
#         register: t.Optional[int] = None,
#         n: t.Optional[int] = None,
#         l_bound: t.Optional[int] = None,
#         u_bound: t.Optional[int] = None,
#     ):
#         """
#         >>> ChordSpacer.open_position([2, 5, 9], register=1)
#         [50, 57, 65]
#         """
#         if register is None:
#             register = random.randrange(len(OPEN_REGISTERS))
#         # TODO actually use register upper bounds?
#         rl_bound, _ = OPEN_REGISTERS[register]
#         l_bound = max(rl_bound, l_bound) if l_bound is not None else rl_bound
#         if n is None:
#             n = len(pcs)
#         out = [put_in_range(pcs[0], low=l_bound)]
#         sign = 1
#         i = 2
#         # we want i to take the values 0, 2, 1, 3, 2, 4, etc.
#         while len(out) < n:
#             pc = pcs[i % len(pcs)]
#             prev_pc = out[-1]
#             interval = (pc - prev_pc) % TET
#             out.append(prev_pc + interval)
#             sign *= -1
#             if sign < 0:
#                 i -= 1
#             else:
#                 i += 2
#         return out

#     def _init_pitches(
#         self,
#         pcs,
#         spacing: t.Optional[str] = None,
#         register: t.Optional[int] = None,
#         n: t.Optional[int] = None,
#         l_bound: t.Optional[int] = None,
#         u_bound: t.Optional[int] = None,
#     ):
#         if spacing is None:
#             spacing = random.choice(self._all_spacings)
#         return (
#             getattr(self, spacing)(
#                 pcs, register=register, n=n, l_bound=l_bound, u_bound=u_bound
#             ),
#             spacing,
#         )

#     # The definition of _all_spacings should be the last line in the class
#     # definition
#     _all_spacings = tuple(
#         name
#         for name, f in locals().items()
#         if getattr(f, "spacing_method", False)
#     )


# @attr_compiler("_all_spacings", "spacing_method")
# class ChordSpacer2:
#     def __init__(self):
#         self._prev_pitches = None

#     def _get_spacings(
#         self, spacings: t.Optional[t.Union[t.Iterable[str], str]]
#     ):
#         if spacings is None:
#             spacings = self._all_spacings
#         elif isinstance(spacings, str):
#             return [spacings]
#         return random.sample(spacings, len(spacings))

#     def __call__(
#         self,
#         pcs: t.Iterable[int],
#         spacings: t.Optional[t.Union[t.Iterable[str], str]] = None,
#         l_bound: int = LOW_PITCH,
#         u_bound: int = HI_PITCH,
#     ):
#         if self._prev_pitches is not None:
#             try:
#                 pitches = voice_lead_pitches(
#                     self._prev_pitches,
#                     pcs,
#                     preserve_bass=True,
#                     min_pitch=l_bound,
#                     max_pitch=u_bound,
#                 )
#             except NoMoreVoiceLeadingsError:
#                 pass
#             else:
#                 self._prev_pitches = pitches
#                 yield pitches
#         spacings = self._get_spacings(spacings)
#         for spacing in spacings:
#             spacing_method = getattr(self, spacing)
#             for pitches in spacing_method(
#                 pcs, l_bound=l_bound, u_bound=u_bound
#             ):
#                 self._prev_pitches = pitches
#                 yield pitches
#         raise NoSpacings()

#     @spacing_method
#     @staticmethod
#     def open_position(
#         pcs,
#         n: t.Optional[int] = None,
#         l_bound: int = LOW_PITCH,
#         u_bound: int = HI_PITCH,
#     ):
#         """
#         >>> sorted(ChordSpacer2.open_position([2, 5, 9], l_bound=48, u_bound=84))
#         [[50, 57, 65], [62, 69, 77]]
#         """
#         # if register is None:
#         #     register = random.randrange(len(OPEN_REGISTERS))
#         # rl_bound, _ = OPEN_REGISTERS[register]
#         if n is None:
#             n = len(pcs)
#         out = [put_in_range(pcs[0], low=l_bound, high=u_bound)]
#         sign = 1
#         i = 2
#         # we want i to take the values 0, 2, 1, 3, 2, 4, etc.
#         while len(out) < n:
#             pc = pcs[i % len(pcs)]
#             prev_pc = out[-1]
#             interval = (pc - prev_pc) % TET
#             out.append(prev_pc + interval)
#             sign *= -1
#             if sign < 0:
#                 i -= 1
#             else:
#                 i += 2
#         n_octaves = (u_bound - out[-1]) // 12 + 1
#         for octave in random.sample(range(0, n_octaves * 12, 12), k=n_octaves):
#             yield [p + octave for p in out]

#     @spacing_method
#     @staticmethod
#     def close_position(
#         pcs,
#         n: t.Optional[int] = None,
#         l_bound: int = LOW_PITCH,
#         u_bound: int = HI_PITCH,
#     ):
#         """
#         >>> sorted(ChordSpacer2.close_position([2, 5, 9], l_bound=48, u_bound=84))
#         [[50, 53, 57], [62, 65, 69], [74, 77, 81]]
#         """
#         # if register is None:
#         #     register = random.randrange(len(CLOSE_REGISTERS))
#         # rl_bound, _ = CLOSE_REGISTERS[register]
#         # l_bound = max(l_bound, rl_bound) if l_bound is not None else rl_bound
#         if n is None:
#             n = len(pcs)
#         out = [put_in_range(pcs[0], low=l_bound)]
#         for i in range(1, n):
#             pc = pcs[i % len(pcs)]
#             prev_pc = out[-1]
#             interval = (pc - prev_pc) % TET
#             out.append(prev_pc + interval)
#         n_octaves = (u_bound - out[-1]) // 12 + 1
#         for octave in random.sample(range(0, n_octaves * 12, 12), k=n_octaves):
#             yield [p + octave for p in out]

#     # The definition of _all_spacings should be the last line in the class
#     # definition
#     _all_spacings = tuple(
#         name
#         for name, f in locals().items()
#         if getattr(f, "spacing_method", False)
#     )


@dataclass
class SimpleSpacerSettings:
    bass_range: t.Tuple[int, int] = (33, 53)
    accomp_range: t.Tuple[int, int] = None  # type:ignore

    def __post_init__(self):
        logging.debug(f"running SimpleSpacerSettings __post_init__()")
        if self.bass_range is None:
            self.bass_range = DEFAULT_BASS_RANGE
        if self.accomp_range is None:
            self.accomp_range = DEFAULT_ACCOMP_RANGE
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # type:ignore


class SimpleSpacer:
    """
    >>> simple_spacer = SimpleSpacer()  # default settings
    >>> pcs = (0, 4, 7)
    >>> omissions = (Allow.NO,) * 3
    >>> spacings = simple_spacer(pcs, omissions)  # a generator

    The order of spacings is random
    >>> next(spacings)  # doctest: +SKIP
    [36, 55, 64]
    >>> next(spacings)  # doctest: +SKIP
    [48, 52, 67]
    """

    def __init__(self, settings: t.Optional[SimpleSpacerSettings] = None):
        if settings is None:
            settings = SimpleSpacerSettings()
        self._prev_pitches = None
        self.settings = settings

    @staticmethod
    def _apply_omissions(
        pcs: t.Sequence[int],
        omissions: t.Sequence[Allow],
        include_bass: bool = True,
    ) -> t.Tuple[t.List[int], t.List[int]]:
        """
        `omissions` should be the same length as `pcs`

        The return value consists of a list of pcs together with a list of indices

        Args:
            pcs: chord members
            omissions: Allow enums of same length as pcs indicating whether item
                should be omitted
            include_bass: whether bass should be included irrespective of omissions

        >>> allow_all = (Allow.YES,) * 3
        >>> SimpleSpacer._apply_omissions((0, 4, 7), allow_all)
        ([0, 4, 7], [1, 2])
        >>> SimpleSpacer._apply_omissions((0, 4, 7), allow_all, include_bass=False)
        ([0, 4, 7], [0, 1, 2])

        >>> allow_none = (Allow.NO,) * 3
        >>> SimpleSpacer._apply_omissions((0, 4, 7), allow_none, include_bass=True)
        ([0, 4, 7], [])
        >>> SimpleSpacer._apply_omissions((0, 4, 7), allow_none, include_bass=False)
        ([0, 4, 7], [])

        Pcs corresponding to Allow.ONLY are omitted from the return value since
        they can never be used anyway.
        >>> SimpleSpacer._apply_omissions((0, 4, 7), (Allow.ONLY,) * 3)
        ([0], [])
        >>> SimpleSpacer._apply_omissions(
        ...     (0, 4, 7), (Allow.ONLY,) * 3, include_bass=False
        ... )
        ([], [])

        """
        assert len(pcs) == len(omissions)
        retained_pcs, possible_omissions = [], []
        i = 0
        for j, (pc, omit) in enumerate(zip(pcs, omissions)):
            if omit is Allow.ONLY and ((not include_bass) or j != 0):
                continue
            retained_pcs.append(pc)
            if (not include_bass or j != 0) and (omit is not Allow.NO):
                possible_omissions.append(i)
            i += 1
        return retained_pcs, possible_omissions

    def _sub(
        self,
        pcs: t.Sequence[int],
        possible_omissions: t.Sequence[int],
        min_accomp_pitch: int,
        max_accomp_pitch: int,
        include_bass: bool,
        min_bass_pitch: t.Optional[int],
        max_bass_pitch: t.Optional[int],
        spacing_constraints: t.Optional[SpacingConstraints] = None,
    ) -> t.Iterator[t.Tuple[int, ...]]:
        if self._prev_pitches is not None:
            try:
                pitches = voice_lead_pitches(
                    self._prev_pitches,
                    pcs,
                    preserve_bass=include_bass,
                    min_pitch=min_accomp_pitch,
                    max_pitch=max_accomp_pitch,
                    min_bass_pitch=min_bass_pitch,
                    max_bass_pitch=max_bass_pitch,
                )
            except (NoMoreVoiceLeadingsError, CardinalityDiffersTooMuch):
                pass
            else:
                self._prev_pitches = pitches
                logging.debug(f"{self.__class__.__name__} yielding spacing {pitches}")
                yield pitches
        if include_bass:
            assert min_bass_pitch is not None and max_bass_pitch is not None
            bass_options = get_all_in_range(pcs[0], min_bass_pitch, max_bass_pitch)
            accomp_options = [
                get_all_in_range(pc, min_accomp_pitch, max_accomp_pitch)
                for pc in pcs[1:]
            ]
            # It would be nice to shuffle the Cartesian product without
            #   having to calculate the whole thing/place it in memory.
            # TODO: (Malcolm) why do we use product and not combinations here?
            spacings = list(it.product(bass_options, *accomp_options))
        else:
            accomp_options = [
                get_all_in_range(pc, min_accomp_pitch, max_accomp_pitch) for pc in pcs
            ]
            spacings = list(it.product(*accomp_options))
        random.shuffle(spacings)
        for spacing in spacings:
            # put pitches in ascending order
            spacing = tuple(sorted(spacing))
            # we skip spacings that do not validate. It would be more efficient not to
            # generate them in the first place though.
            if (spacing_constraints is not None) and (
                not validate_spacing(spacing, spacing_constraints)
            ):
                continue
            self._prev_pitches = spacing
            logging.debug(f"{self.__class__.__name__} yielding spacing {spacing}")
            yield spacing
        for i in range(len(possible_omissions)):
            for indices in it.combinations(possible_omissions, i + 1):
                logging.debug(f"{self.__class__.__name__} omitting indices {indices}")
                try:
                    yield from self._sub(
                        [pc for i, pc in enumerate(pcs) if i not in indices],
                        (),
                        min_accomp_pitch,
                        max_accomp_pitch,
                        include_bass,
                        min_bass_pitch,
                        max_bass_pitch,
                    )
                except NoSpacings:
                    pass
        if include_bass:
            raise NoSpacings(
                f"no more spacings for pcs {pcs} with bass between "
                f"{min_bass_pitch} and {max_bass_pitch} and other parts "
                f"between {min_accomp_pitch} and {max_accomp_pitch}"
            )
        else:
            raise NoSpacings(
                f"no more spacings for pcs {pcs} between "
                f"{min_accomp_pitch} and {max_accomp_pitch}"
            )

    def __call__(
        self,
        pcs: t.Sequence[int],
        omissions: t.Sequence[Allow],
        min_accomp_pitch: t.Optional[int] = None,
        max_accomp_pitch: t.Optional[int] = None,
        include_bass: bool = True,
        min_bass_pitch: t.Optional[int] = None,
        max_bass_pitch: t.Optional[int] = None,
        spacing_constraints: t.Optional[SpacingConstraints] = None,
    ) -> t.Iterator[t.Tuple[int, ...]]:
        pcs, possible_omissions = self._apply_omissions(pcs, omissions, include_bass)
        min_accomp_pitch = (
            self.settings.accomp_range[0]
            if min_accomp_pitch is None
            else min_accomp_pitch
        )
        max_accomp_pitch = (
            self.settings.accomp_range[1]
            if max_accomp_pitch is None
            else max_accomp_pitch
        )
        if include_bass:
            min_bass_pitch = (
                self.settings.bass_range[0]
                if min_bass_pitch is None
                else min_bass_pitch
            )
            max_bass_pitch = (
                self.settings.bass_range[1]
                if max_bass_pitch is None
                else max_bass_pitch
            )
        yield from self._sub(
            pcs,
            possible_omissions,
            min_accomp_pitch,
            max_accomp_pitch,
            include_bass,
            min_bass_pitch,
            max_bass_pitch,
            spacing_constraints,
        )
