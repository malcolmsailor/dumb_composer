from __future__ import annotations

import typing as t
from bisect import bisect
from functools import cached_property

from music21.key import Key
from music21.roman import Minor67Default, RomanNumeral

from dumb_composer.pitch_utils.intervals import reduce_compound_interval
from dumb_composer.pitch_utils.put_in_range import get_all_in_range, put_in_range
from dumb_composer.pitch_utils.types import (
    ChromaticInterval,
    Interval,
    Pitch,
    PitchClass,
    ScalarInterval,
    ScaleDegree,
    SpelledPitchClass,
)


class ScaleDict2:
    _scales: dict[tuple[SpelledPitchClass, t.Literal["major", "minor"]], Scale2] = {}

    @classmethod
    def __getitem__(
        cls, tonic_and_mode: tuple[SpelledPitchClass, t.Literal["major", "minor"]]
    ) -> Scale2:
        if tonic_and_mode in cls._scales:
            return cls._scales[tonic_and_mode]

        scale = Scale2(*tonic_and_mode)
        cls._scales[tonic_and_mode] = scale
        return scale


SCALE_CACHE = ScaleDict2()


def strictly_increasing(l: t.Sequence):
    return all(x < y for x, y in zip(l, l[1:]))


class ScaleDict:
    """This class exists simply to cache scales.

    When a scale is retrieved, if it does not exist, it is created.

    We can index the contents with a tuple of pcs, in which case the other arguments to
    `Scale` are assumed to have their default values:

    >>> scale_dict = ScaleDict()
    >>> scale_dict[(0, 2, 4, 5, 7, 9, 11)]
    Scale(pcs=(0, 2, 4, 5, 7, 9, 11), zero_pitch=0, tet=12)

    Or we can index the contents with a tuple of arguments to `Scale`:

    >>> scale_dict[(0, 2, 4, 5, 7, 9, 10), 3, 24]
    Scale(pcs=(0, 2, 4, 5, 7, 9, 10), zero_pitch=3, tet=24)
    """

    def __init__(self):
        self._scales = {}

    def __getitem__(self, args: t.Tuple[PitchClass, ...] | t.Tuple[t.Any, ...]):
        """ """
        if not isinstance(args, tuple):
            raise ValueError
        if isinstance(args[0], int):
            # in this case we guess that only pcs have been passed
            args = (args,)
        try:
            return self._scales[args]
        except KeyError:
            new_scale = Scale(*args)  # type:ignore
            self._scales[args] = new_scale
            return new_scale


# (Malcolm 2023-10-15) I do not know why I defined a second scale class `Scale2`.
class Scale2:
    def __init__(self, tonic: SpelledPitchClass, mode: t.Literal["major", "minor"]):
        if mode == "major":
            tonic = tonic[0].upper() + tonic[1:]
        elif mode == "minor":
            tonic = tonic[0].lower() + tonic[1:]
        else:
            raise ValueError
        self._music21_key = Key(tonic)
        self._tonic = tonic
        self._mode = mode
        self._cache: dict[str, tuple[PitchClass, ...]] = {}

    @cached_property
    def pcs(self) -> tuple[PitchClass, ...]:
        # music21 returns the scale *including the upper octave*, which we do
        #   not want
        return tuple(p.pitchClass for p in self._music21_key.pitches[:-1])

    def pcs_for_rn(self, rn_token: str) -> t.Tuple[PitchClass, ...]:
        if rn_token in self._cache:
            return self._cache[rn_token]
        pcs = self.fit_to_rn(rn_token)
        self._cache[rn_token] = pcs
        return pcs

    def fit_to_rn(
        self,
        rn_token: str,
        sixth_minor: Minor67Default = Minor67Default.CAUTIONARY,
        seventh_minor: Minor67Default = Minor67Default.CAUTIONARY,
    ) -> t.Tuple[PitchClass, ...]:
        """
        >>> C_major = Scale2("C", "major")
        >>> C_minor = Scale2("C", "minor")
        >>> C_major.fit_to_rn("viio7")  # Note A-flat
        (0, 2, 4, 5, 7, 8, 11)
        >>> C_major.fit_to_rn("Ger6")  # Note A-flat, E-flat, F-sharp
        (0, 2, 3, 6, 7, 8, 11)



        Sometimes flats are indicated for chord factors that are already flatted
        in the relevant scale. We handle those with a bit of a hack:
        >>> C_minor.fit_to_rn("Vb9")
        (0, 2, 3, 5, 7, 8, 11)

        # TODO: (Malcolm 2023-08-25) move elsewhere
        # >>> C_major.fit_to_rn(RN("Vb9/vi", keyOrScale="C"))
        # (9, 11, 0, 2, 4, 5, 8)

        There can be a similar issue with raised degrees. If the would-be raised
        degree is already in the scale, we leave it unaltered:
        >>> C_minor.fit_to_rn("V+")
        (0, 2, 3, 5, 7, 8, 11)
        """

        def _add_inflection(degree: ScaleDegree, inflection: ChromaticInterval):
            inflected_pitch = (scale_pcs[degree] + inflection) % 12
            if (
                inflection < 0
                and inflected_pitch == scale_pcs[(degree - 1) % len(scale_pcs)]
            ):
                # hack to handle "b9" etc. when already in scale
                return
            if (
                inflection > 0
                and inflected_pitch == scale_pcs[(degree + 1) % len(scale_pcs)]
            ):
                return
            scale_pcs[degree] = inflected_pitch

        rn = RomanNumeral(
            rn_token,
            self._music21_key,
            sixthMinor=sixth_minor,
            seventhMinor=seventh_minor,
        )
        assert (
            rn.secondaryRomanNumeralKey is None
        ), "Scale2.pcs_for_rn should be called on secondary key"

        # # music21 returns the scale *including the upper octave*, which we do
        # #   not want
        # scale_pcs = [p.pitchClass for p in key.pitches[:-1]]  # type:ignore
        scale_pcs = list(self.pcs)
        for pitch in rn.pitches:
            # NB degrees are 1-indexed so we must subtract 1 below
            (
                degree,
                accidental,
            ) = self._music21_key.getScaleDegreeAndAccidentalFromPitch(  # type:ignore
                pitch
            )
            if accidental is not None:
                inflection = int(accidental.alter)
                _add_inflection(degree - 1, inflection)  # type:ignore
        try:
            assert len(scale_pcs) == len(set(scale_pcs))
        except AssertionError:
            # these special cases are hacks until music21's RomanNumeral
            #   handling is repaired or I figure out another solution
            if rn.figure == "bII7":
                scale_pcs = list(self.pcs)
                # scale_pcs = [p.pitchClass for p in key.pitches[:-1]]  # type:ignore
                _add_inflection(1, -1)
                _add_inflection(5, -1)
            else:
                raise
        return tuple(scale_pcs)


class Scale:
    def __init__(
        self,
        pcs: t.Sequence[int],
        zero_pitch: int = 0,
        tet: int = 12,
    ):
        """
        Args:
            pcs: must
                - have at least 2 items.
                - tonic is understood to be the first item.
                - strictly ascending except for one jump.
                - in range [0, tet)
        (Malcolm 2023-10-15) As far as I can tell today, `zero_pitch` is just there
        for the unusual case where we want a pc other than C to be 0.
        """
        assert len(pcs) > 1
        for i in range(len(pcs), 0, -1):
            pc2 = pcs[i % len(pcs)]
            pc1 = pcs[(i - 1) % len(pcs)]
            if pc2 < pc1:
                break

        pcs = pcs[i:] + pcs[:i]  # type:ignore
        assert strictly_increasing(pcs)
        assert min(pcs) >= 0 and max(pcs) < tet
        self._pcs = pcs
        self._pcs_set = set(pcs)
        self._tonic_idx = (-i) % len(pcs)  # type:ignore
        self._len = len(pcs)
        self._zero_pitch = zero_pitch
        self._tet = tet
        self._lookup_pc = {pc: i for i, pc in enumerate(self._pcs)}

    def __repr__(self):
        return f"{self.__class__.__name__}(pcs={self._pcs}, zero_pitch={self._zero_pitch}, tet={self._tet})"

    @property
    def tonic_pc(self):  # pylint: disable=missing-docstring
        """
        >>> scale = Scale([7, 9, 11, 0, 2, 4, 6])  # G major
        >>> scale.tonic_pc
        7
        """
        return self[0]

    def __len__(self):
        return self._len

    def __contains__(self, pitch: int):
        return pitch % self._tet in self._pcs

    def __getitem__(self, key: int) -> int:
        """
        >>> scale = Scale([7, 9, 11, 0, 2, 4, 6])  # G major
        >>> scale[35]  # tonic in 5th octave is 5 * 7
        67
        """
        octave, scale_degree = divmod(key + self._tonic_idx, len(self))
        return self._pcs[scale_degree] + self._tet * octave + self._zero_pitch

    # def get_degree(
    #     self,
    #     degree: ScaleDegree,
    #     prev_degree: ScaleDegree | None = None,
    #     next_degree: ScaleDegree | None = None,
    # ) -> PitchClass:
    #     pass

    def pitch_has_upper_step(self, pitch: Pitch) -> bool:
        """
        >>> d_minor = Scale([2, 4, 5, 7, 9, 10, 1])  # D harmonic minor
        >>> all(d_minor.pitch_has_upper_step(pc) for pc in [2, 4, 5, 7, 9, 1])
        True
        >>> d_minor.pitch_has_upper_step(10)
        False
        """
        if len(self) == 7:
            next_pitch = self[self.index(pitch) + 1]
            return next_pitch - pitch in (1, 2)
        raise ValueError(
            "pitch_has_upper_step() only implemented for heptatonic scales"
        )

    def pitch_has_lower_step(self, pitch: Pitch) -> bool:
        """
        >>> d_minor = Scale([2, 4, 5, 7, 9, 10, 1])  # D harmonic minor
        >>> all(d_minor.pitch_has_lower_step(pc) for pc in [2, 4, 5, 7, 9, 10])
        True
        >>> d_minor.pitch_has_lower_step(1)
        False
        """
        if len(self) == 7:
            next_pitch = self[self.index(pitch) - 1]
            return pitch - next_pitch in (1, 2)
        raise ValueError(
            "pitch_has_lower_step() only implemented for heptatonic scales"
        )

    def adjusted_lower_bound(self, pitch: Pitch) -> Pitch:
        """
        Note: this function assumes that scale pitches have no more than a small number
        of intervening chromatic steps. If that assumption doesn't hold the
        implementation will be quite inefficient.

        >>> d_minor = Scale([2, 4, 5, 7, 9, 10, 1])  # D harmonic minor
        >>> d_minor.adjusted_lower_bound(50)  # D4
        50
        >>> d_minor.adjusted_lower_bound(48)  # C4 -> C#
        49
        >>> d_minor.adjusted_lower_bound(47)  # B3 -> C#
        49
        """

        while pitch not in self:
            pitch += 1
        return pitch

    def adjusted_upper_bound(self, pitch: Pitch) -> Pitch:
        """
        Note: this function assumes that scale pitches have no more than a small number
        of intervening chromatic steps. If that assumption doesn't hold the
        implementation will be quite inefficient.

        >>> d_minor = Scale([2, 4, 5, 7, 9, 10, 1])  # D harmonic minor
        >>> d_minor.adjusted_upper_bound(50)  # D4
        50
        >>> d_minor.adjusted_upper_bound(48)  # C4 -> Bb
        46
        >>> d_minor.adjusted_upper_bound(47)  # B3 -> Bb
        46
        """

        while pitch not in self:
            pitch -= 1
        return pitch

    def get_auxiliary(
        self,
        scale_degree: int,
        alteration: str,
        lowered_degrees: str = "natural",
    ) -> int:
        """

        Args:
            alteration: either "+" or "#" for raised, or "-" or "b" for lowered.
            lowered_degrees: either "natural" or "chromatic".

        If alteration is "+", the pitch one semitone below the next higher
        scale pitch is returned.

        E.g., in D harmonic minor
        - D will be raised to D#
        - Bb will be raised to B#

        >>> d_minor = Scale([2, 4, 5, 7, 9, 10, 1])  # D harmonic minor
        >>> d_minor.get_auxiliary(35, "+")  # raised ^1 = D#
        63

        >>> d_minor.get_auxiliary(33, "+")  # raised ^6 = B#
        60


        If alteration is '-', the behavior is influenced by `lower_degrees`.

        If `lower_degrees` is "natural", then if there is a semitone below
        the scale degree, it is left unaltered. Otherwise, it is lowered by
        a semitone.

        E.g., in D harmonic minor
        - E will be lowered to Eb
        - D will be left unaltered
        - C# will be lowered to C (not Cb)

        >>> d_minor.get_auxiliary(36, "-")  # ^2 has whole-tone below, lowered to Eb
        63
        >>> d_minor.get_auxiliary(35, "-")  # ^1 has semitone below, left as D
        62
        >>> d_minor.get_auxiliary(34, "-")  # ^7 has augmented 2nd below, lowered to C
        60

        Otherwise if `lower_degrees` is "chromatic", then the pitch one semitone
        above the next lower scale pitch is returned. This is analogous to the
        behavior of "+", but is less musically common.

        E.g., in D harmonic minor
        - C# will be lowered to Cb

        >>> d_minor.get_auxiliary(
        ...     34, "-", lowered_degrees="chromatic"
        ... )  # ^7 has augmented 2nd below, lowered to Cb
        59
        """
        if alteration in ("+", "#"):
            return self[scale_degree + 1] - 1
        elif lowered_degrees == "chromatic":
            return self[scale_degree - 1] + 1
        unaltered = self[scale_degree]
        if unaltered - self[scale_degree - 1] == 1:
            return unaltered
        return unaltered - 1

    @property
    def pcs(self):  # pylint: disable=missing-docstring
        return self._pcs

    def index(self, pitch):
        """
        >>> scale = Scale([7, 9, 11, 0, 2, 4, 6])  # G major
        >>> scale.index(67)  # tonic in 5th octave is 5 * 7
        35
        >>> scale.index(11)
        2
        >>> scale.index(4)
        -2
        """
        octave, pc = divmod(pitch - self._zero_pitch, self._tet)
        try:
            return self._index_sub(octave, pc)
        except KeyError:
            raise IndexError(
                f"pitch {pitch} with pitch-class {pc} is not in scale {self._pcs}"
            )

    def _index_sub(self, octave: int, pc: int):
        idx = self._lookup_pc[pc] - self._tonic_idx
        return octave * len(self) + idx

    def nearest_index(self, pitch, scale2: t.Optional[Scale] = None):
        """The use of this function is to find best approximation to generic
        interval within the scale to a pitch which may or may not be in the
        scale (see get_interval below).

        Behavior of this function depends on whether scale2 (i.e., the scale
        to which the pitch belongs) is passed or not.

        Case 1: scale 2 is not passed.
        ==============================

        If 'pitch' is not in the scale but one pitch of the scale is closer
        than others, index to that pitch is returned. (N.B., this may not always
        be the best solution musically. E.g., in F pentatonic, we might prefer
        to think of E as "a step below F" but this function will return the
        index to F (as though E were in fact F-flat).)
        >>> pentatonic_scale = Scale([5, 7, 9, 0, 2])  # F pentatonic
        >>> pentatonic_scale.nearest_index(65)  # F5, returns 5 * 5 for F5
        25
        >>> pentatonic_scale.nearest_index(64)  # E5, returns 5 * 5 for F5
        25
        >>> pentatonic_scale.nearest_index(63)  # Eb5, returns 5 * 5 - 1 for D5
        24

        However we mostly use heptatonic scales, where there is never an
        interval wider than a whole tone between pitches, so for any pitch not
        in the scale, there will be two pitches equidistant from it. In those
        cases, if scale2 is not provided, by default we return the lower of the
        possible indices (which corresponds to "sharpening" the lower pitch).
        >>> scale = Scale([7, 9, 11, 0, 2, 4, 6])  # G major
        >>> scale.nearest_index(61)  # C#5/Db5, returns 7 * 5 - 4 for C5
        31

        Case 2: scale2 is passed.
        ==========================

        If 'pitch' is not in this scale, then

            - if the pitch *above* 'pitch' in this scale is present in scale2,
                then index to the pitch below is returned (i.e., the new pitch
                is interpreted as a sharpened version of a pitch in this scale).

        >>> scale1 = Scale([7, 9, 11, 0, 2, 4, 6])  # G major
        >>> scale2 = Scale([4, 6, 8, 9, 11, 1, 3])  # E major
        >>> scale1.nearest_index(68, scale2)  # G#5/Ab5, returns 7 * 5 for G5
        35

        - if the pitch *below* 'pitch' in this scale is present in scale2,
                then index to the pitch above is returned (i.e., the new pitch
                is interpreted as a flattened version of a pitch in this scale).

        >>> scale3 = Scale([8, 10, 0, 1, 3, 5, 7])  # A-flat major
        >>> scale1.nearest_index(68, scale3)  # G#5/Ab5, returns 7 * 5 + 1 for A5
        36

        In the case where both the pitches above and below are present in
        scale2 (e.g., in a chromatic scale), the behavior as the same as in
        case 1.

        >>> chromatic = Scale([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        >>> scale1.nearest_index(68, chromatic)  # G#5/Ab5, returns 7 * 5 for G5
        35
        """
        octave, pc = divmod(pitch - self._zero_pitch, self._tet)
        try:
            return self._index_sub(octave, pc)
        except KeyError:
            pass
        upper_i = bisect(self._pcs, pc) - self._tonic_idx
        upper_pc = self[upper_i]
        lower_pc = self[upper_i - 1]
        if scale2 is not None:
            if upper_pc in scale2:
                return octave * len(self) + upper_i - 1
            if lower_pc in scale2:
                return octave * len(self) + upper_i
        above = upper_pc - pc
        below = pc - lower_pc
        if above < below:
            i = upper_i
        else:
            i = upper_i - 1
        return octave * len(self) + i

    def get_interval(
        self,
        pitch1: int,
        pitch2: int,
        scale2: t.Optional[Scale] = None,
        reduce_compounds: bool = False,
    ) -> ScalarInterval:
        """Gets generic interval between two pitches.

        >>> s = Scale([0, 2, 4, 5, 7, 9, 11])  # C major
        >>> s.get_interval(60, 65)
        3
        >>> s.get_interval(60, 86)
        15
        >>> s.get_interval(60, 86, reduce_compounds=True)
        1
        >>> s.get_interval(64, 50)
        -8

        With second pitch present only in second scale:

        >>> s2 = Scale([4, 6, 8, 9, 11, 1, 3])  # E major
        >>> s.get_interval(60, 68, scale2=s2)  # an augmented fifth
        4

        >>> s3 = Scale([8, 10, 0, 1, 3, 5, 7])  # A-flat major
        >>> s.get_interval(60, 68, scale2=s3)  # a minor 6th
        5

        With first pitch present only in second scale:

        >>> s2 = Scale([4, 6, 8, 9, 11, 1, 3])  # E major
        >>> s.get_interval(68, 60, scale2=s2)  # an augmented fifth
        -4

        >>> s3 = Scale([8, 10, 0, 1, 3, 5, 7])  # A-flat major
        >>> s.get_interval(68, 72, scale2=s3)  # a minor 3rd
        2

        Both pitches present only in second scale:

        >>> s2 = Scale([4, 6, 8, 9, 11, 1, 3])  # E major
        >>> s.get_interval(61, 68, scale2=s2)  # a perfect fifth
        4
        """
        out = self.nearest_index(pitch2, scale2) - self.nearest_index(pitch1, scale2)
        if reduce_compounds:
            out = out % len(self)
        return out

    def get_reduced_scalar_interval(
        self, pitch1: int, pitch2: int, scale2: t.Optional[Scale] = None
    ) -> Interval:
        """
        "Reduced" intervals are intervals <= octave (no compound intervals), with
        unisons distinguished from octaves.

        >>> s = Scale([0, 2, 4, 5, 7, 9, 11])  # C major
        >>> s.get_reduced_scalar_interval(60, 60)  # C5, C5 -> Unison
        0
        >>> s.get_reduced_scalar_interval(60, 48)  # C5, C4 -> Descending octave
        -7
        >>> s.get_reduced_scalar_interval(48, 60)  # C4, C5 -> Ascending octave
        7
        >>> s.get_reduced_scalar_interval(48, 72)  # C4, C6 -> Compound ascending octave
        7
        >>> s.get_reduced_scalar_interval(72, 57)  # C6, A4 -> Compound descending third
        -2
        """
        return reduce_compound_interval(
            self.get_interval(pitch1, pitch2, scale2), n_steps_per_octave=len(self)
        )

    def get_scalar_forward_interval(
        self, pitch1: int, pitch2: int, scale2: t.Optional[Scale] = None
    ):
        """
        By "forward interval" I mean the interval measured "upwards" from the
        pitch-class of the first pitch to the pitch-class of the second pitch.

        E.g., a descending 7th counts as a 2nd, etc.

        This differs from "interval class" as it occurs in music theory because
        we treat ascending 2nds/descending 7ths as different from
        ascending 7ths/descending 2nds.

        # TODO: (Malcolm 2023-07-12) come up with a better name than "forward interval"?

        >>> s = Scale([0, 2, 4, 5, 7, 9, 11])  # C major
        >>> s.get_scalar_forward_interval(60, 65)  # C5, F5 -> 4th
        3
        >>> s.get_scalar_forward_interval(60, 86)  # C5, D7 -> 2nd
        1
        >>> s.get_scalar_forward_interval(64, 50)  # E5, D4 -> 7th
        6
        >>> s.get_scalar_forward_interval(60, 60)  # C5, C5 -> Unison
        0
        >>> s.get_scalar_forward_interval(48, 60)  # C4, C5 -> Unison
        0
        """
        return self.get_interval(pitch1, pitch2, scale2) % len(self)

    def pitch_is_diatonic(self, pitch: int) -> bool:
        """
        Indicates whether the pitch is diatonic to this scale.
        >>> s = Scale([0, 2, 4, 5, 7, 9, 11])  # C major
        >>> s.pitch_is_diatonic(60)
        True
        >>> s.pitch_is_diatonic(61)
        False
        """
        return (pitch % 12) in self._pcs_set

    def find_intervals(
        self,
        starting_pitch: Pitch,
        eligible_pcs: t.Sequence[PitchClass],
        min_pitch: Pitch,
        max_pitch: Pitch,
        max_interval: ScalarInterval | None = None,
        forbidden_intervals: t.Iterable[int] | None = None,
        allow_steps_outside_of_range: bool = False,
    ) -> t.List[ScalarInterval]:
        """
        >>> d_minor = Scale([2, 4, 5, 7, 9, 10, 1])  # D harmonic minor

        >>> d_minor.find_intervals(  # From tonic to other tonic triad members
        ...     62, eligible_pcs=[2, 5, 9], min_pitch=50, max_pitch=74
        ... )
        [-7, 0, 7, -5, 2, -3, 4]

        >>> d_minor.find_intervals(  # From #^6 (not in scale) to dominant triad
        ...     71, eligible_pcs=[9, 1, 4], min_pitch=50, max_pitch=74
        ... )
        [-8, -1, -6, 1, -11, -4]

        If any pc in `eligible_pcs` is not in the scale, an IndexError is raised.
        E.g., pc 0 is outside of scale:
        >>> d_minor.find_intervals(  # From #^6 (not in scale) to dominant *minor*
        ...     71, eligible_pcs=[9, 0, 4], min_pitch=50, max_pitch=74
        ... )
        Traceback (most recent call last):
        IndexError: pitch 0 with pitch-class 0 is not in scale [1, 2, 4, 5, 7, 9, 10]

        """

        if forbidden_intervals is not None:
            raise NotImplementedError("# TODO: (Malcolm 2023-07-15) ?")
        min_pitch = self.adjusted_lower_bound(min_pitch)
        max_pitch = self.adjusted_upper_bound(max_pitch)
        _min_interval = self.get_interval(starting_pitch, min_pitch)
        _max_interval = self.get_interval(starting_pitch, max_pitch)

        if max_interval is not None:
            _min_interval = max(_min_interval, -max_interval)
            _max_interval = min(_max_interval, max_interval)

        if allow_steps_outside_of_range:
            if not _min_interval and self.pitch_has_lower_step(starting_pitch):
                _min_interval = -1
            if not _max_interval and self.pitch_has_upper_step(starting_pitch):
                _max_interval = 1

        starting_index = self.nearest_index(starting_pitch)

        eligible_scalar_pcs = (self.index(eligible_pc) for eligible_pc in eligible_pcs)
        scalar_pcs_in_range = get_all_in_range(
            eligible_scalar_pcs,
            low=starting_index + _min_interval,
            high=starting_index + _max_interval,
            steps_per_octave=len(self),
        )
        intervals = [scalar_pc - starting_index for scalar_pc in scalar_pcs_in_range]
        return intervals
