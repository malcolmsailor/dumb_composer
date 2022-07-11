import math
import random
from types import MappingProxyType
from typing import Iterable, Optional, Sequence, Tuple

from voice_leader import voice_lead_pitches

from dumb_composer.constants import (
    CLOSE_REGISTERS,
    KEYBOARD_STYLE_REGISTERS,
    OPEN_REGISTERS,
    TET,
)

from .utils import attr_compiler
from .pitch_utils import put_in_range


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


@attr_compiler("_all_spacings", "spacing_method")
class ChordSpacer:
    def __init__(self):
        self._prev_pitches = None

    def __call__(
        self,
        pcs: Iterable[int],
        init: bool = False,
        spacing: Optional[str] = None,
    ):
        """If the first time the instance is called, or init is True, creates
        a new spacing. (If spacing is None, then randomly choosing from
        close_position, open_position, or keyboard_style.)

        Otherwise, voice-leads from previous chord.
        """
        print(pcs)
        if self._prev_pitches is None or init:
            pitches = self._init_pitches(pcs, spacing=spacing)
        else:
            print(self._prev_pitches, pcs)
            pitches = voice_lead_pitches(
                self._prev_pitches, pcs, preserve_root=True
            )
        self._prev_pitches = pitches
        return pitches

    @spacing_method
    def keyboard_style(self, pcs, register: int = None, n: int = None):
        if register is None:
            register = random.randrange(len(KEYBOARD_STYLE_REGISTERS))
        (bass_l_bound, _), (chord_l_bound, _) = KEYBOARD_STYLE_REGISTERS[
            register
        ]
        if n is None:
            n = len(pcs)
        out = [
            put_in_range(pcs[0], low=bass_l_bound),
            put_in_range(pcs[1 % len(pcs)], low=chord_l_bound),
        ]
        for i in range(2, n):
            pc = pcs[i % len(pcs)]
            prev_pc = out[-1]
            interval = (pc - prev_pc) % TET
            out.append(prev_pc + interval)
        return out

    @spacing_method
    def close_position(self, pcs, register: int = None, n: int = None):
        if register is None:
            register = random.randrange(len(CLOSE_REGISTERS))
        l_bound, _ = CLOSE_REGISTERS[register]
        if n is None:
            n = len(pcs)
        out = [put_in_range(pcs[0], low=l_bound)]
        for i in range(1, n):
            pc = pcs[i % len(pcs)]
            prev_pc = out[-1]
            interval = (pc - prev_pc) % TET
            out.append(prev_pc + interval)
        return out

    @spacing_method
    def open_position(self, pcs, register: int = None, n: int = None):
        if register is None:
            register = random.randrange(len(OPEN_REGISTERS))
        l_bound, _ = OPEN_REGISTERS[register]
        if n is None:
            n = len(pcs)
        out = [put_in_range(pcs[0], low=l_bound)]
        sign = 1
        i = 2
        # we want i to take the values 0, 2, 1, 3, 2, 4, etc.
        while len(out) < n:
            pc = pcs[i % len(pcs)]
            prev_pc = out[-1]
            interval = (pc - prev_pc) % TET
            out.append(prev_pc + interval)
            sign *= -1
            if sign < 0:
                i -= 1
            else:
                i += 2
        return out

    def _init_pitches(self, pcs, *args, spacing=None, **kwargs):
        if spacing is None:
            spacing = random.choice(self._all_spacings)
            print(spacing)
        return getattr(self, spacing)(pcs, *args, **kwargs)

    # The definition of _all_spacings should be the last line in the class
    # definition
    _all_spacings = tuple(
        name
        for name, f in locals().items()
        if getattr(f, "spacing_method", False)
    )
