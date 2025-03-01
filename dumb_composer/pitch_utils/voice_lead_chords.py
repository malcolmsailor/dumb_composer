from __future__ import annotations

import logging
import typing as t
from collections import Counter
from itertools import chain
from types import MappingProxyType

from voice_leader import (
    CardinalityDiffersTooMuch,
    voice_lead_pitches_multiple_options_iter,
)

from dumb_composer.chords.chords import (
    Chord,
    Tendency,
    get_chords_from_rntxt,
    is_same_harmony,  # Used in doctests
)
from dumb_composer.pitch_utils.parts import (
    outer_voices_have_forbidden_antiparallels,
    succession_has_forbidden_parallels,
)
from dumb_composer.pitch_utils.pcs import PitchClass
from dumb_composer.pitch_utils.spacings import SpacingConstraints, validate_spacing
from dumb_composer.pitch_utils.types import Pitch, Suspension
from dumb_composer.utils.recursion import UndoRecursiveStep

LOGGER = logging.getLogger(__name__)


def unresolved_tendency_tone(
    pitch: Pitch,
    pitch_tendency: Tendency,
    resolution_pc: PitchClass,
    chord2: Chord,
    chord2_suspension_resolution_pcs: t.Container[PitchClass],
    chord2_melody_pitch: Pitch | None,
) -> bool:
    if (chord2_melody_pitch is not None) and (
        (
            # Avoid resolving tendency tones to tendency tones already
            # in the melody
            tendency_tone_already_in_melody := (
                (chord2_melody_pitch % 12 == resolution_pc)
                and (
                    (
                        chord2_melody_tendency := chord2.get_pitch_tendency(
                            chord2_melody_pitch
                        )
                    )
                    is not Tendency.NONE
                )
            )
        )
        or (
            # Avoid moving to pitch-classes that are presently delayed
            # by suspensions
            pitch_class_has_suspension := (
                resolution_pc in chord2_suspension_resolution_pcs
            )
        )
    ):
        if tendency_tone_already_in_melody:
            LOGGER.debug(
                f"Can't resolve {pitch=} with {pitch_tendency=} because "
                f"{chord2_melody_pitch=} w/ {chord2_melody_tendency=}"  # type:ignore
            )
        elif pitch_class_has_suspension:  # type:ignore
            LOGGER.debug(
                f"Can't resolve {pitch=} with {pitch_tendency=} because "
                f"of suspension that will resolve to this pitch-class"
            )
        return True
    return False


def voice_lead_chords(
    chord1: Chord,
    chord2: Chord,
    chord1_pitches: t.Sequence[Pitch],
    chord1_suspensions: t.Mapping[Pitch, Suspension] = MappingProxyType({}),
    chord2_bass_pitch: Pitch | None = None,
    chord2_melody_pitch: Pitch | None = None,
    chord2_included_pitches: t.Sequence[Pitch] | None = None,
    chord2_suspensions: t.Dict[Pitch, Suspension] | None = None,
    min_pitch: t.Optional[int] = None,
    max_pitch: t.Optional[int] = None,
    min_bass_pitch: t.Optional[int] = None,
    max_bass_pitch: t.Optional[int] = None,
    raise_error_on_failure_to_resolve_tendencies: bool = False,
    max_diff_number_of_voices: int = 0,
    # I'm setting spacing_constraints.max_adjacent_interval because that was how
    #   it was when I wrote all the doctests and I don't want to rewrite them
    spacing_constraints: SpacingConstraints = SpacingConstraints(
        max_adjacent_interval=12
    ),
    # TODO: (Malcolm 2023-07-22) document
    normalize_voice_assignments: bool = True,
    allow_unresolved_tendencies: bool = False,
    allow_doubled_tendencies: bool = True,
) -> t.Iterator[t.Tuple[Pitch]]:
    """Voice-lead, taking account of tendency tones, etc.

    # TODO: (Malcolm 2023-08-25) move these specific cases to a test file or something
    >>> rntxt = "m1 d: i b3 iiø65"
    >>> i, ii65 = get_chords_from_rntxt(rntxt)
    >>> chord2_suspensions = {
    ...     69: Suspension(
    ...         pitch=69,
    ...         resolves_by=-2,
    ...         dissonant=True,
    ...         interval_above_bass=2,
    ...         score=3.0,
    ...         begins_on_prev=False,
    ...         resolves_on_next=True,
    ...     ),
    ...     62: Suspension(
    ...         pitch=62,
    ...         resolves_by=-1,
    ...         dissonant=True,
    ...         interval_above_bass=7,
    ...         score=1.0,
    ...         begins_on_prev=False,
    ...         resolves_on_next=False,
    ...     ),
    ... }
    >>> vl_iter = voice_lead_chords(
    ...     i,
    ...     ii65,
    ...     chord1_pitches=(38, 62, 65, 69),
    ...     chord1_suspensions={},
    ...     chord2_melody_pitch=69,
    ...     chord2_bass_pitch=43,
    ...     chord2_suspensions=chord2_suspensions,
    ... )
    >>> next(vl_iter), next(vl_iter)
    ((43, 62, 64, 69), (43, 58, 62, 69))

    >>> rntxt = "m1 ii65 b3 ii7"
    >>> ii65, ii7 = get_chords_from_rntxt(rntxt)
    >>> suspension1a = Suspension(
    ...     pitch=60,
    ...     resolves_by=-1,
    ...     dissonant=True,
    ...     interval_above_bass=7,
    ...     resolves_on_next=False,
    ... )
    >>> suspension1b = Suspension(
    ...     pitch=76, resolves_by=-2, dissonant=True, interval_above_bass=11
    ... )
    >>> suspension2 = Suspension(
    ...     pitch=60, resolves_by=-1, dissonant=True, interval_above_bass=10
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     ii65,
    ...     ii7,
    ...     (53, 60, 69, 76),
    ...     chord1_suspensions={60: suspension1a, 76: suspension1b},
    ...     chord2_bass_pitch=50,
    ...     chord2_melody_pitch=74,
    ...     chord2_suspensions={60: suspension2},
    ... )
    >>> next(vl_iter), next(vl_iter)
    ((50, 60, 69, 74), (50, 60, 65, 74))

    # >>> rntxt = "m1 F: viio64 b3 viio6/ii"
    # >>> viio64, viio6_of_ii = get_chords_from_rntxt(rntxt)
    # >>> suspension = Suspension(
    # ...     pitch=46, resolves_by=-1, dissonant=True, interval_above_bass=0, score=1.25
    # ... )
    # >>> vl_iter = voice_lead_chords(
    # ...     viio64,
    # ...     viio6_of_ii,
    # ...     (46, 55, 55, 64),
    # ...     chord2_bass_pitch=46,
    # ...     chord2_melody_pitch=66,
    # ...     chord2_suspensions={46: suspension},
    # ... )
    # >>> next(vl_iter)

    # TODO: (Malcolm 2023-07-22) is this example working as intended?
    # >>> rntxt = '''m1 F: vi b3 viio7/V'''
    # >>> vi, viio7_of_V = get_chords_from_rntxt(rntxt)
    # >>> suspension1 = Suspension(
    # ...     pitch=64, resolves_by=-2, dissonant=True, interval_above_bass=2
    # ... )
    # >>> suspension2 = Suspension(
    # ...     pitch=58, resolves_by=-2, dissonant=False, interval_above_bass=8
    # ... )
    # >>> vl_iter = voice_lead_chords(
    # ...     vi,
    # ...     viio7_of_V,
    # ...     (50, 53, 58, 64),
    # ...     chord1_suspensions={64: suspension1, 58: suspension2},
    # ...     chord2_bass_pitch=47,
    # ...     chord2_included_pitches=(56,),
    # ...     chord2_melody_pitch=62,
    # ... )
    # >>> next(vl_iter)

    >>> rntxt = "m1 a: iv6 b3 C: V2"
    >>> iv6, V2 = get_chords_from_rntxt(rntxt)
    >>> suspension1 = Suspension(
    ...     pitch=69, resolves_by=-2, dissonant=True, interval_above_bass=4
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     iv6,
    ...     V2,
    ...     (41, 62, 69, 69),
    ...     chord2_bass_pitch=41,
    ...     chord2_melody_pitch=71,
    ...     chord2_suspensions={69: suspension1},
    ... )
    >>> next(vl_iter)
    (41, 62, 69, 71)



    >>> rntxt = '''m1 C: I b2 I6 b3 V6 b4 ii
    ... m2 V43 b2 V/IV b3 IV b4 V'''
    >>> I, I6, V6, ii, V43, V_of_IV, IV, V = get_chords_from_rntxt(rntxt)

    Note: we don't down-weight doubling thirds at all. When it comes to close-position
    chords this corresponds to baroque figured-bass practice, but it corresponds less
    well to other styles of harmonization. (On the other hand, see Huron's comments
    on doubling.)
    >>> vl_iter = voice_lead_chords(I, I6, (60, 64, 67, 72))
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((64, 64, 67, 72), (64, 67, 67, 72), (52, 64, 67, 72))

    >>> vl_iter = voice_lead_chords(I, I6, (60, 64, 67, 72), max_bass_pitch=60)
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((52, 64, 67, 72), (52, 67, 67, 72), (52, 60, 67, 72))

    If `raise_error_on_failure_to_resolve_tendencies` is True and a tendency tone
    can't be resolved, there is a ValueError
    >>> next(
    ...     voice_lead_chords(
    ...         V6,
    ...         ii,
    ...         (59, 62, 67),  # doctest: +IGNORE_EXCEPTION_DETAIL
    ...         raise_error_on_failure_to_resolve_tendencies=True,
    ...     )
    ... )
    Traceback (most recent call last):
    ValueError:

    >>> vl_iter = voice_lead_chords(V6, I, (47, 55, 62, 67))
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((48, 55, 64, 67), (48, 52, 60, 67), (48, 55, 60, 64))

    >>> vl_iter = voice_lead_chords(V43, I, (50, 59, 65, 67))
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((48, 60, 64, 67), (60, 60, 64, 67), (48, 55, 60, 64))

    Although the pitch-class content of I and V_of_IV is the same, the results aren't
    necessarily the same because we will avoid doubling tendency tones such as the
    leading-tone of V_of_IV. However, they are usually the same.
    >>> vl_iter = voice_lead_chords(V43, V_of_IV, (50, 59, 65, 67))
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((48, 60, 64, 67), (60, 60, 64, 67), (48, 55, 60, 64))

    ------------------------------------------------------------------------------------
    Suspensions in chord 1
    ------------------------------------------------------------------------------------

    Suspension w/ repetition of same harmony:
    >>> suspension = Suspension(
    ...     pitch=65, resolves_by=-1, dissonant=True, interval_above_bass=5
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     I, I, (48, 55, 65, 72), chord1_suspensions={65: suspension}
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((48, 55, 64, 72), (48, 55, 64, 76), (48, 55, 64, 67))

    Suspension w/ change of harmony:
    >>> suspension = Suspension(
    ...     pitch=62, resolves_by=-2, dissonant=True, interval_above_bass=10
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     I6, IV, (52, 62, 67, 72), chord1_suspensions={62: suspension}
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((53, 60, 69, 72), (53, 60, 69, 69), (53, 60, 65, 69))

    Suspension in bass w/ repetition of same harmony:
    >>> suspension = Suspension(
    ...     pitch=50, resolves_by=-2, dissonant=True, interval_above_bass=0
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     I, I, (50, 55, 64, 67), chord1_suspensions={50: suspension}
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((48, 55, 64, 67), (48, 55, 64, 64), (48, 52, 64, 67))

    Suspension in bass w/ change of harmony:
    >>> suspension = Suspension(
    ...     pitch=52, resolves_by=-2, dissonant=True, interval_above_bass=0
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     ii, V43, (52, 53, 62, 69), chord1_suspensions={52: suspension}
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((50, 53, 59, 67), (50, 55, 65, 71), (50, 55, 59, 65))

    # TODO: (Malcolm 2023-07-22) chord1 suspensions in inner voices

    ------------------------------------------------------------------------------------
    Providing chord 2 melody pitch
    ------------------------------------------------------------------------------------

    Melody "overlaps" with inner voice
    >>> vl_iter = voice_lead_chords(I, V6, (48, 55, 64, 67), chord2_melody_pitch=62)
    >>> next(vl_iter), next(vl_iter)
    ((47, 55, 62, 62), (47, 55, 55, 62))

    >>> next(vl_iter)  # There are no more voice-leadings that work
    Traceback (most recent call last):
    StopIteration

    Melody with suspension in bass
    >>> suspension = Suspension(
    ...     pitch=48, resolves_by=-1, dissonant=True, interval_above_bass=0
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     V6,
    ...     V6,
    ...     (48, 50, 55, 62),
    ...     chord2_melody_pitch=67,
    ...     chord1_suspensions={48: suspension},
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((47, 50, 55, 67), (47, 50, 62, 67), (47, 62, 62, 67))

    Melody with suspension in inner part
    >>> suspension = Suspension(
    ...     pitch=53, resolves_by=-1, dissonant=True, interval_above_bass=5
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     I,
    ...     I,
    ...     (48, 53, 60, 67),
    ...     chord2_melody_pitch=72,
    ...     chord1_suspensions={53: suspension},
    ... )
    >>> next(vl_iter), next(vl_iter)
    ((48, 52, 60, 72), (48, 52, 64, 72))

    Any further possibilities would require the bass to cross over the suspension, which
    we don't allow:
    >>> next(vl_iter)
    Traceback (most recent call last):
    StopIteration

    Melody with suspension in melody
    >>> suspension = Suspension(
    ...     pitch=69, resolves_by=-2, dissonant=False, interval_above_bass=9
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     I,
    ...     V6,
    ...     (48, 52, 60, 69),
    ...     chord2_melody_pitch=67,
    ...     chord1_suspensions={69: suspension},
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((47, 50, 62, 67), (47, 55, 62, 67), (47, 50, 55, 67))

    ------------------------------------------------------------------------------------
    Suspensions in chord 2
    ------------------------------------------------------------------------------------

    # TODO: (Malcolm 2023-07-21) the melody crosses "under" the suspension voice here.
    # I should see about preventing that (although I believe I'm always providing
    # the melody pitch)
    Suspension in inner voice:
    >>> suspension = Suspension(
    ...     pitch=60, resolves_by=-1, dissonant=True, interval_above_bass=10
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     I, V43, (48, 55, 60, 64), chord2_suspensions={60: suspension}
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter), next(vl_iter)
    ((50, 55, 60, 65), (50, 53, 60, 67), (38, 55, 60, 65), (50, 53, 55, 60))

    Unprepared suspension in inner voice:
    >>> suspension = Suspension(
    ...     pitch=71, resolves_by=-2, dissonant=True, interval_above_bass=6
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     I, IV, (60, 67, 72, 76), chord2_suspensions={71: suspension}
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((65, 71, 72, 77), (65, 65, 71, 72), (53, 71, 72, 77))

    # TODO: (Malcolm 2023-07-19) these results suggest we're favoring different
    #   not doubling pitches *too* much
    Suspension in melody voice:
    >>> suspension = Suspension(
    ...     pitch=72, resolves_by=-1, dissonant=True, interval_above_bass=10
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     I, V43, (48, 55, 64, 72), chord2_suspensions={72: suspension}
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((50, 55, 65, 72), (38, 55, 65, 72), (50, 65, 67, 72))

    Unprepared suspension in melody voice:
    >>> suspension = Suspension(
    ...     pitch=64, resolves_by=-2, dissonant=False, interval_above_bass=9
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     I,
    ...     V,
    ...     (48, 52, 60, 67),
    ...     chord2_melody_pitch=64,
    ...     chord2_suspensions={64: suspension},
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((55, 55, 59, 64), (43, 55, 59, 64), (55, 55, 55, 64))

    # TODO: (Malcolm 2023-08-11)
    # Unprepared suspension which is tendency tone in another voice:
    # >>> suspension = Suspension(
    # ...     pitch=65, resolves_by=-1, dissonant=True, interval_above_bass=5
    # ... )
    # >>> vl_iter = voice_lead_chords(
    # ...     V43,
    # ...     I,
    # ...     (50, 65, 67),
    # ...     chord2_melody_pitch=65,
    # ...     chord2_suspensions={65: suspension},
    # ... )
    # >>> next(vl_iter), next(vl_iter), next(vl_iter)


    Suspension overlapping with melody voice:
    >>> suspension = Suspension(
    ...     pitch=72, resolves_by=-1, dissonant=True, interval_above_bass=10
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     I,
    ...     V,
    ...     (48, 64, 72, 72),
    ...     chord2_melody_pitch=74,
    ...     chord2_suspensions={72: suspension},
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((43, 62, 72, 74), (43, 67, 72, 74), (55, 62, 72, 74))

    Suspension in melody voice whose preparation is unison w/ another voice:
    >>> rntxt = "m1 F: ii6 b3 viio6/ii"
    >>> ii6, viio6_of_ii = get_chords_from_rntxt(rntxt)
    >>> suspension = Suspension(
    ...     pitch=67, resolves_by=-1, dissonant=True, interval_above_bass=10
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     ii6,
    ...     viio6_of_ii,
    ...     (46, 62, 67, 67),
    ...     chord2_melody_pitch=67,
    ...     chord2_suspensions={67: suspension},
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((45, 60, 60, 67), (45, 57, 60, 67), (57, 60, 60, 67))

    Unprepared suspension overlapping with melody voice:
    >>> suspension = Suspension(
    ...     pitch=72, resolves_by=-1, dissonant=True, interval_above_bass=10
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     I,
    ...     V,
    ...     (48, 64, 67, 72),
    ...     chord2_melody_pitch=74,
    ...     chord2_suspensions={72: suspension},
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((43, 67, 72, 74), (55, 67, 72, 74), (43, 62, 72, 74))

    Suspension in bass voice:
    >>> suspension = Suspension(
    ...     pitch=48, resolves_by=-1, dissonant=True, interval_above_bass=0
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     I, V6, (48, 55, 64, 72), chord2_suspensions={48: suspension}
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((48, 55, 62, 74), (48, 55, 67, 74), (48, 55, 62, 67))

    If we actually specify the bass, it shouldn't make any difference in this case:
    >>> suspension = Suspension(
    ...     pitch=48, resolves_by=-1, dissonant=True, interval_above_bass=0
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     I,
    ...     V6,
    ...     (48, 55, 64, 72),
    ...     chord2_bass_pitch=48,
    ...     chord2_suspensions={48: suspension},
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((48, 55, 62, 74), (48, 55, 67, 74), (48, 55, 62, 67))

    Multiple inner suspensions:
    >>> suspension1 = Suspension(
    ...     pitch=62, resolves_by=-2, dissonant=True, interval_above_bass=10
    ... )
    >>> suspension2 = Suspension(
    ...     pitch=65, resolves_by=-1, dissonant=True, interval_above_bass=1
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     V43,
    ...     I6,
    ...     (50, 62, 65, 71),
    ...     chord2_suspensions={62: suspension1, 65: suspension2},
    ... )

    # TODO: (Malcolm 2023-08-25) why is this failing now?
    # >>> next(vl_iter)
    # ((52, 62, 65, 72), (40, 62, 65, 72))
    >>> next(
    ...     vl_iter
    ... )  # Because all 3 upper voices are fixed, this exhausts the iterator
    Traceback (most recent call last):
    StopIteration

    Melody suspension w/ inner suspension:
    >>> suspension1 = Suspension(
    ...     pitch=62, resolves_by=-2, dissonant=True, interval_above_bass=10
    ... )
    >>> suspension2 = Suspension(
    ...     pitch=65, resolves_by=-1, dissonant=True, interval_above_bass=1
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     V43,
    ...     I6,
    ...     (50, 55, 62, 65),
    ...     chord2_suspensions={62: suspension1, 65: suspension2},
    ... )
    >>> next(vl_iter), next(vl_iter)
    ((52, 55, 62, 65), (40, 55, 62, 65))

    Bass suspension w/ melody suspension:
    >>> suspension1 = Suspension(
    ...     pitch=62, resolves_by=-2, dissonant=True, interval_above_bass=0
    ... )
    >>> suspension2 = Suspension(
    ...     pitch=77, resolves_by=-1, dissonant=True, interval_above_bass=3
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     V43,
    ...     I,
    ...     (62, 67, 71, 77),
    ...     chord2_suspensions={62: suspension1, 77: suspension2},
    ... )
    >>> next(vl_iter), next(vl_iter)
    ((62, 67, 72, 77), (62, 72, 72, 77))

    Bass suspension w/ inner suspension:
    >>> suspension1 = Suspension(
    ...     pitch=62, resolves_by=-2, dissonant=True, interval_above_bass=0
    ... )
    >>> suspension2 = Suspension(
    ...     pitch=65, resolves_by=-1, dissonant=True, interval_above_bass=3
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     V43,
    ...     I,
    ...     (62, 65, 67, 71),
    ...     chord2_suspensions={62: suspension1, 65: suspension2},
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((62, 65, 67, 72), (62, 65, 72, 79), (62, 65, 72, 72))

    Suspension whose preparation is doubled pitch in chord1:
    >>> suspension = Suspension(
    ...     pitch=60, resolves_by=-1, dissonant=True, interval_above_bass=5
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     I6,
    ...     V,
    ...     (52, 60, 60, 67),
    ...     chord2_suspensions={60: suspension},
    ...     chord2_melody_pitch=67,
    ... )
    >>> next(vl_iter), next(vl_iter)
    ((55, 60, 62, 67), (43, 60, 62, 67))

    ------------------------------------------------------------------------------------
    Chord progression likely to lead to parallels
    ------------------------------------------------------------------------------------

    >>> vl_iter = voice_lead_chords(I, ii, (60, 64, 67, 72))
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((62, 65, 65, 69), (62, 62, 65, 69), (50, 65, 69, 69))

    ------------------------------------------------------------------------------------
    Spacing constraints
    ------------------------------------------------------------------------------------

    >>> vl_iter = voice_lead_chords(
    ...     I,
    ...     V6,
    ...     (60, 64, 67, 72),
    ...     spacing_constraints=SpacingConstraints(max_adjacent_interval=5),
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((59, 62, 67, 67), (59, 62, 62, 67), (59, 74, 74, 79))

    Starting from a very widely spaced chord
    >>> vl_iter = voice_lead_chords(
    ...     I,
    ...     V6,
    ...     (36, 48, 64, 79),
    ...     spacing_constraints=SpacingConstraints(max_adjacent_interval=9),
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((35, 55, 62, 67), (35, 62, 67, 74), (35, 55, 62, 62))

    ------------------------------------------------------------------------------------
    Different numbers of voices
    ------------------------------------------------------------------------------------

    We normalize the voice-leading displacement by the number of voices. (Otherwise,
    voice-leadings with fewer voices would always be ranked higher since the dropped
    voices have 0 displacement.) This doesn't necessarily give ideal results.

    >>> rntxt = '''m1 C: I b2 I6 b3 V6 b4 ii
    ... m2 V43 b2 V/IV b3 IV b4 V'''
    >>> I, I6, V6, ii, V43, V_of_IV, IV, V = get_chords_from_rntxt(rntxt)
    >>> vl_iter = voice_lead_chords(I, V6, (60, 64, 67), max_diff_number_of_voices=1)
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((59, 62, 67, 67), (59, 62, 67), (59, 62, 67, 74))

    ------------------------------------------------------------------------------------
    Prespecifying included pitches
    ------------------------------------------------------------------------------------

    Specifying melody + one inner pitch
    >>> vl_iter = voice_lead_chords(
    ...     I6,
    ...     V,
    ...     (52, 55, 60, 67),
    ...     chord2_melody_pitch=71,
    ...     chord2_included_pitches=(62,),
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((55, 55, 62, 71), (55, 62, 62, 71), (43, 55, 62, 71))

    If we don't specify a melody but specify an included pitch, the included pitch
    may be in the melody or not:
    >>> vl_iter = voice_lead_chords(
    ...     I6, V, (52, 55, 60, 67), chord2_included_pitches=(67,)
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((55, 62, 67, 71), (55, 59, 62, 67), (43, 62, 67, 71))

    # TODO: (Malcolm 2023-07-21) test this w/ specifying bass suspensions
    Specifying bass pitch:
    >>> vl_iter = voice_lead_chords(I6, V, (52, 55, 60, 67), chord2_bass_pitch=43)
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((43, 59, 62, 67), (43, 55, 62, 71), (43, 55, 59, 62))

    Fully specifying the output chord (there's no reason to do this):
    >>> vl_iter = voice_lead_chords(
    ...     I6,
    ...     V,
    ...     (52, 55, 60, 67),
    ...     chord2_melody_pitch=71,
    ...     chord2_bass_pitch=43,
    ...     chord2_included_pitches=(55, 62),
    ... )

    # TODO: (Malcolm 2023-08-25) why is this failing?
    # >>> next(vl_iter)
    # (43, 55, 62, 71)
    >>> next(vl_iter)
    Traceback (most recent call last):
    StopIteration

    ------------------------------------------------------------------------------------
    Avoid resolving tendency tone when resolution pitch-class already present in melody
    ------------------------------------------------------------------------------------

    # TODO: (Malcolm 2023-08-03) fix
    >>> rntxt = '''m1 F: viio7/V b3 V'''
    >>> viio7_of_V, V = get_chords_from_rntxt(rntxt)

    With default settings, this will not return anything:
    >>> vl_iter = voice_lead_chords(
    ...     viio7_of_V,
    ...     V,
    ...     (35, 53, 62, 74),  # B2 F4 D5 D6
    ...     chord2_melody_pitch=64,  # E5 in melody
    ... )  # chord2 must not resolve 53 to 52
    >>> next(vl_iter)
    Traceback (most recent call last):
    StopIteration

    With `allow_unresolved_tendencies`, the F4 will proceed to a different pitch:
    >>> vl_iter = voice_lead_chords(
    ...     viio7_of_V,
    ...     V,
    ...     (35, 53, 62, 74),  # B2 F4 D5 D6
    ...     chord2_melody_pitch=64,  # E5 in melody
    ...     allow_unresolved_tendencies=True,
    ... )  # chord2 must not resolve 53 to 52
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((36, 55, 60, 64), (36, 48, 60, 64), (36, 48, 55, 64))

    ------------------------------------------------------------------------------------
    Other miscellany
    ------------------------------------------------------------------------------------

    Note that even with a tendency tone in the top voice, the top voice can appear not
    to resolve it if another voice moves to a higher pitch (because the output result is
    sorted by pitch.)

    Compare the following (note the last result with last pitch 79) with the next example
    >>> suspension1 = Suspension(
    ...     pitch=62, resolves_by=-2, dissonant=True, interval_above_bass=0
    ... )
    >>> suspension2 = Suspension(
    ...     pitch=65, resolves_by=-1, dissonant=True, interval_above_bass=3
    ... )
    >>> vl_iter = voice_lead_chords(
    ...     V43,
    ...     I,
    ...     (62, 65, 67, 71),
    ...     chord2_suspensions={62: suspension1, 65: suspension2},
    ... )
    >>> next(vl_iter), next(vl_iter)
    ((62, 65, 67, 72), (62, 65, 72, 79))

    Compare the following with the previous example
    >>> vl_iter = voice_lead_chords(
    ...     V43,
    ...     I,
    ...     (62, 65, 67, 71),
    ...     chord2_suspensions={62: suspension1, 65: suspension2},
    ...     chord2_melody_pitch=72,
    ... )
    >>> next(vl_iter), next(vl_iter)
    ((62, 65, 67, 72), (62, 65, 72, 72))

    >>> rntxt = '''m1 F: V b2 Cad64 b3 V54'''
    >>> V, Cad64, V54 = get_chords_from_rntxt(rntxt)

    """

    if normalize_voice_assignments is False:
        raise NotImplementedError("# TODO: (Malcolm 2023-07-22) ")

    chord2_max_notes = len(chord1_pitches) + max_diff_number_of_voices
    chord2_min_notes = len(chord1_pitches) - max_diff_number_of_voices

    chord1_melody_pitch = max(chord1_pitches)

    bass_suspension = None
    soprano_suspension = False
    if chord2_suspensions:
        # if enforce_preparations:
        #   assert all(p in chord1_pitches for p in chord2_suspensions)

        for pitch, suspension in chord2_suspensions.items():
            if suspension.interval_above_bass == 0:
                bass_suspension = pitch
            elif pitch == chord2_melody_pitch:
                soprano_suspension = True
            elif pitch == chord1_melody_pitch:
                if chord2_melody_pitch is None:
                    chord2_melody_pitch = pitch
                    soprano_suspension = True
        if bass_suspension is not None:
            chord2_suspensions.pop(bass_suspension)
    elif chord2_suspensions is None:
        chord2_suspensions = {}

    chord2_suspension_resolution_pcs = {
        (p + s.resolves_by) % 12 for (p, s) in chord2_suspensions.items()
    }

    # within_harmony = is_same_harmony(chord1, chord2)

    resolution_pitches = []
    unresolved_suspensions = []
    unresolved_tendencies = []
    pitches_without_tendencies = []

    chord2_suspension_pitches_doubled_in_chord1 = Counter(
        p for p in chord1_pitches if p in chord2_suspensions
    )
    for p in chord2_suspension_pitches_doubled_in_chord1:
        chord2_suspension_pitches_doubled_in_chord1[p] -= 1

    # tendency_pcs = set()  # for keeping track of doubled tendency-tones
    tendency_pcs: dict[PitchClass, Pitch] = {}

    for i, pitch in enumerate(
        chord1_pitches[: (None if chord2_melody_pitch is None else -1)]
    ):
        # ------------------------------------------------------------------------------
        # Case 1 pitch is suspension in chord 1
        # ------------------------------------------------------------------------------
        if pitch in chord1_suspensions:
            suspension = chord1_suspensions[pitch]
            if suspension.resolves_on_next:
                resolution_pitch = pitch + suspension.resolves_by
                if resolution_pitch % 12 in chord2.pcs:
                    if i != 0 and (
                        pitch not in chord2_suspensions
                        or chord2_suspension_pitches_doubled_in_chord1[pitch]
                    ):
                        # we don't include the bass among the resolution pitches because it
                        # is always already included
                        resolution_pitches.append(resolution_pitch)
                else:
                    unresolved_suspensions.append(pitch)
            else:
                assert (
                    pitch in chord2_suspensions
                ), f"{suspension=} has `resolves_on_next=False`, but {pitch=} is not in `chord2_suspensions`"
        # ------------------------------------------------------------------------------
        # Case 2 pitch has tendency *and* pitch is not in next chord (NB this covers
        # the case where there is no change of harmony too)
        # We also ignore the tendency where it is in the bass and the next bass pitch
        #   is provided
        # ------------------------------------------------------------------------------
        elif (
            ((pitch_tendency := chord1.get_pitch_tendency(pitch)) is not Tendency.NONE)
            and (pitch not in chord2)
            and not (chord2_bass_pitch is not None and pitch == chord1_pitches[0])
        ):
            resolution = chord2.get_tendency_resolutions(pitch, pitch_tendency)

            pc = pitch % 12

            # If the pitch needs to resolve, find the resolution
            if resolution is not None:
                # If pitch is not in bass and pitch is not suspended in chord2
                if (i != 0) and (
                    pitch not in chord2_suspensions
                    or chord2_suspension_pitches_doubled_in_chord1[pitch]
                ):
                    resolution_pc = resolution.to % 12

                    # Check for unresolved tendency tones
                    if unresolved_tendency_tone(
                        pitch,
                        pitch_tendency,
                        resolution_pc,
                        chord2,
                        chord2_suspension_resolution_pcs,
                        chord2_melody_pitch,
                    ):
                        if not allow_unresolved_tendencies:
                            return
                        unresolved_tendencies.append(pitch)
                    else:
                        if pc in tendency_pcs:
                            if not allow_doubled_tendencies:
                                return

                            # doubled tendency tone handling. We hope not to see these,
                            # but we need to handle them if we do. For now, arbitrarily,
                            # we only include the last doubling of each tendency tone,
                            # by moving any prior instances to `unresolved_tendencies`
                            unresolved_tendencies.append(tendency_pcs[pc])

                        # we don't include the bass among the resolution pitches because
                        # it is always already included
                        resolution_pitches.append(resolution.to)
                else:
                    # I'm a little bit confused by the control flow here but it seems to give the
                    # correct results.
                    # TODO: (Malcolm 2023-07-20) review
                    pass

            else:
                if pc in tendency_pcs:
                    # doubled tendency tone handling. We hope not to see these, but we need to handle
                    # them if we do. For now, arbitrarily, we only include the last doubling
                    # of each tendency tone.
                    unresolved_tendencies = [
                        p for p in unresolved_tendencies if p % 12 != pc
                    ]
                unresolved_tendencies.append(pitch)

            tendency_pcs[pc] = pitch

        # ------------------------------------------------------------------------------
        # Case 3 pitch is not suspension and either has no tendency, or has a tendency
        #   that can't be resolved on the next chord
        # ------------------------------------------------------------------------------
        else:
            if i != 0 and (
                (
                    pitch not in chord2_suspensions
                    or chord2_suspension_pitches_doubled_in_chord1[pitch]
                )
                or
                # If the pitch *is* in chord2_suspensions but it is a doubling a suspension
                #   in the melody, we need to add it to pitches_without_tendencies
                (pitch == chord2_melody_pitch)
            ):
                pitches_without_tendencies.append(pitch)
        if chord2_suspension_pitches_doubled_in_chord1[pitch]:
            chord2_suspension_pitches_doubled_in_chord1[pitch] -= 1

    if unresolved_suspensions:
        breakpoint()
        raise ValueError("Suspensions cannot resolve")
    if raise_error_on_failure_to_resolve_tendencies and unresolved_tendencies:
        raise ValueError("Tendencies cannot resolve")

    prespecified_pitches = (
        tuple(resolution_pitches)
        + (
            (chord2_melody_pitch,)
            if (chord2_melody_pitch is not None and not soprano_suspension)
            else ()
        )
        + tuple(chord2_included_pitches if chord2_included_pitches is not None else ())
    )

    # semitone suspension resolutions pitches are never doubled in chord2
    # whole-tone suspension resolutions *can* be doubled, if the doubling is
    # in a different octave. To avoid doubling in the same octave,
    # we calculate the suspension resolutions and supply them in
    # `exclude_motions` below.
    # We also include `whole_tone_suspension_resolution_pitches` as
    #   `prefer_to_omit` argument to get_voicing_option_weights()
    whole_tone_suspension_resolution_pitches = [
        p - 2 for (p, s) in chord2_suspensions.items() if s.resolves_by == -2
    ]

    chord2_options = chord2.get_pcs_needed_to_complete_voicing(
        other_chord_factors=prespecified_pitches,
        suspensions=chord2_suspensions,
        bass_suspension=bass_suspension,
        min_notes=chord2_min_notes,
        max_notes=chord2_max_notes,
    )

    chord2_option_weights = chord2.get_voicing_option_weights(
        chord2_options,
        prespecified_pitches,
        prefer_to_omit_pcs=[p % 12 for p in whole_tone_suspension_resolution_pitches],
    )

    pitches_to_voice_lead_from = (
        ([chord1_pitches[0]] if bass_suspension is None else [])
        + pitches_without_tendencies
        + unresolved_tendencies
    )

    chord2_suspension_pitches = (
        () if bass_suspension is None else (bass_suspension,)
    ) + tuple(chord2_suspensions)

    if chord2_melody_pitch is not None:
        if max_pitch is None:
            max_pitch = chord2_melody_pitch
        else:
            max_pitch = min(chord2_melody_pitch, max_pitch)

    if bass_suspension is not None:
        if min_pitch is None:
            min_pitch = bass_suspension + 1
        else:
            min_pitch = max(bass_suspension + 1, min_pitch)

    exclude_motions = {
        i: [r - p for r in whole_tone_suspension_resolution_pitches]
        for i, p in enumerate(pitches_to_voice_lead_from)
    }

    # Hack to set chord2 bass pitch:
    if chord2_bass_pitch is not None:
        # TODO: (Malcolm 2023-07-21) assertion that takes account of suspensions
        # assert chord2_bass_pitch % 12
        min_bass_pitch = chord2_bass_pitch
        max_bass_pitch = chord2_bass_pitch

    # To avoid having the bass cross above any suspensions or their resolutions, we set
    # max_bass_pitch here
    elif resolution_pitches:
        max_bass_pitch = min(
            chain(
                resolution_pitches,
                chord2_suspension_pitches,
                () if max_bass_pitch is None else (max_bass_pitch,),
            )
        )
    pcs_to_voice_lead_to = [
        ([chord2.pcs[0]] if bass_suspension is None else []) + option
        for option in chord2_options
    ]

    # if not len(pitches_to_voice_lead_from) == len(pcs_to_voice_lead_to[0]):
    #     breakpoint()

    try:
        for (
            candidate_pitches,
            voice_assignments,
        ) in voice_lead_pitches_multiple_options_iter(
            pitches_to_voice_lead_from,
            pcs_to_voice_lead_to,
            chord2_option_weights=chord2_option_weights,
            # We handle the bass separately if there is a bass suspension
            preserve_bass=bass_suspension is None,
            avoid_bass_crossing=bass_suspension is None,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            min_bass_pitch=min_bass_pitch,
            max_bass_pitch=max_bass_pitch,
            exclude_motions=exclude_motions,
            ignore_voice_assignments=normalize_voice_assignments,
        ):
            output = tuple(
                sorted(
                    candidate_pitches + prespecified_pitches + chord2_suspension_pitches
                )
            )
            # if chord1_pitches == (45, 66, 72):
            #     breakpoint()
            if not validate_spacing(output, spacing_constraints):
                LOGGER.debug(f"spacing {output} did not validate")
                continue
            if succession_has_forbidden_parallels(chord1_pitches, output):
                LOGGER.debug(f"forbidden parallels in {output=}")
                continue
            if outer_voices_have_forbidden_antiparallels(chord1_pitches, output):
                LOGGER.debug(f"forbidden antiparallels in {output=}")
                continue
            yield output
    except CardinalityDiffersTooMuch as exc:
        raise ValueError from exc
