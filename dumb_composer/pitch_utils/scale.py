from bisect import bisect
import typing as t


def strictly_increasing(l: t.Iterable):
    return all(x < y for x, y in zip(l, l[1:]))


class ScaleDict:
    def __init__(self):
        self._scales = {}

    def __getitem__(self, args: t.Tuple):
        """Can be indexed with a tuple of pcs only.

        >>> scale_dict = ScaleDict()
        >>> scale_dict[(0, 2, 4, 5, 7, 9, 11)].pcs
        (0, 2, 4, 5, 7, 9, 11)

        Can also be indexed with more args to scale (as a tuple).

        >>> scale_dict[(0, 2, 4, 5, 7, 9, 10), 3].pcs
        (0, 2, 4, 5, 7, 9, 10)
        """
        if not isinstance(args, tuple):
            raise ValueError
        if isinstance(args[0], int):
            # in this case we guess that only pcs have been passed
            args = (args,)
        try:
            return self._scales[args]
        except KeyError:
            new_scale = Scale(*args)
            self._scales[args] = new_scale
            return new_scale


class Scale:
    """Dummy class so that when I type hint Scale in the methods for Scale
    below it will be defined. TODO when I get back to the internet see how to
    type hint a method with the class itself."""


class Scale:
    def __init__(
        self,
        pcs: t.Iterable[list],
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
        """
        assert len(pcs) > 1
        for i in range(len(pcs), 0, -1):
            pc2 = pcs[i % len(pcs)]
            pc1 = pcs[(i - 1) % len(pcs)]
            if pc2 < pc1:
                break
        # print(pcs, i, pcs[i:] + pcs[:i])
        pcs = pcs[i:] + pcs[:i]
        assert strictly_increasing(pcs)
        assert min(pcs) >= 0 and max(pcs) < tet
        self._pcs = pcs
        self._tonic_idx = (-i) % len(pcs)
        self._len = len(pcs)
        self._zero_pitch = zero_pitch
        self._tet = tet
        self._lookup_pc = {pc: i for i, pc in enumerate(self._pcs)}

    @property
    def tonic_pc(self):  # pylint: disable=missing-docstring
        """
        >>> scale = Scale([7,9,11,0,2,4,6]) # G major
        >>> scale.tonic_pc
        7
        """
        return self[0]

    def __len__(self):
        return self._len

    def __contains__(self, pitch):
        return pitch % self._tet in self._pcs

    def __getitem__(self, key: int) -> int:
        """
        >>> scale = Scale([7,9,11,0,2,4,6]) # G major
        >>> scale[35] # tonic in 5th octave is 5 * 7
        67
        """
        octave, scale_degree = divmod(key + self._tonic_idx, len(self))
        return self._pcs[scale_degree] + self._tet * octave + self._zero_pitch

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
        >>> g_major = Scale([7,9,11,0,2,4,6])
        >>> g_major.get_auxiliary(35, "+") # raised ^1 = G#
        68

        >>> d_minor = Scale([2,4,5,7,9,10,1]) # D harmonic minor
        >>> d_minor.get_auxiliary(33, "+") # raised ^6 = B#
        60


        If alteration is '-', the behavior is influenced by `lower_degrees`.

        If `lower_degrees` is "natural", then if there is a semitone below
        the scale degree, it is left unaltered. Otherwise, it is lowered by
        a semitone.

        >>> d_minor.get_auxiliary(36, "-") # ^2 has whole-tone below, lowered to Eb
        63
        >>> d_minor.get_auxiliary(35, "-") # ^1 has semitone below, left as D
        62
        >>> d_minor.get_auxiliary(34, "-") # ^7 has augmented 2nd below, lowered to C
        60

        Otherwise if `lower_degrees` is "chromatic", then the pitch one semitone
        above the next lower scale pitch is returned. This is analogous to the
        behavior of "+", but is less musically common.
        >>> d_minor.get_auxiliary(34, "-",
        ...     lowered_degrees="chromatic") # ^7 has augmented 2nd below, lowered to Cb
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
        >>> scale = Scale([7,9,11,0,2,4,6]) # G major
        >>> scale.index(67) # tonic in 5th octave is 5 * 7
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
        """Use of this function is to find best approximation to generic
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
        >>> pentatonic_scale = Scale([5,7,9,0,2]) # F pentatonic
        >>> pentatonic_scale.nearest_index(65) # F5, returns 5 * 5 for F5
        25
        >>> pentatonic_scale.nearest_index(64) # E5, returns 5 * 5 for F5
        25
        >>> pentatonic_scale.nearest_index(63) # Eb5, returns 5 * 5 - 1 for D5
        24

        However we mostly use heptatonic scales, where there is never an
        interval wider than a semitone between pitches, so for any pitch not
        in the scale, there will be two pitches equidistant from it. In those
        cases, if scale2 is not provided, by default we return the lower of the
        possible indices (which corresponds to "sharpening" the lower pitch).
        >>> scale = Scale([7,9,11,0,2,4,6]) # G major
        >>> scale.nearest_index(61) # C#5/Db5, returns 7 * 5 - 4 for C5
        31

        Case 2: scale2 is passed.
        ==========================

        If 'pitch' is not in this scale, then

            - if the pitch *above* 'pitch' in this scale is present in scale2,
                then index to the pitch below is returned (i.e., the new pitch
                is interpreted as a sharpened version of a pitch in this scale).

        >>> scale1 = Scale([7,9,11,0,2,4,6]) # G major
        >>> scale2 = Scale([4,6,8,9,11,1,3]) # E major
        >>> scale1.nearest_index(68, scale2) # G#5/Ab5, returns 7 * 5 for G5
        35

        - if the pitch *below* 'pitch' in this scale is present in scale2,
                then index to the pitch above is returned (i.e., the new pitch
                is interpreted as a flattened version of a pitch in this scale).

        >>> scale3 = Scale([8,10,0,1,3,5,7]) # A-flat major
        >>> scale1.nearest_index(68, scale3) # G#5/Ab5, returns 7 * 5 + 1 for A5
        36

        In the case where both the pitches above and below are present in
        scale2 (e.g., in a chromatic scale), the behavior as the same as in
        case 1.

        >>> chromatic = Scale([0,1,2,3,4,5,6,7,8,9,10,11])
        >>> scale1.nearest_index(68, chromatic) # G#5/Ab5, returns 7 * 5 for G5
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
        self, pitch1: int, pitch2: int, scale2: t.Optional[Scale] = None
    ):
        """Gets generic interval between two pitches.

        >>> s = Scale([0,2,4,5,7,9,11]) # C major
        >>> s.get_interval(60, 65)
        3
        >>> s.get_interval(60, 86)
        15
        >>> s.get_interval(64, 50)
        -8

        With second pitch present only in second scale:

        >>> s2 = Scale([4,6,8,9,11,1,3]) # E major
        >>> s.get_interval(60, 68, scale2=s2) # an augmented fifth
        4

        >>> s3 = Scale([8,10,0,1,3,5,7]) # A-flat major
        >>> s.get_interval(60, 68, scale2=s3) # a minor 6th
        5

        With first pitch present only in second scale:

        >>> s2 = Scale([4,6,8,9,11,1,3]) # E major
        >>> s.get_interval(68, 60, scale2=s2) # an augmented fifth
        -4

        >>> s3 = Scale([8,10,0,1,3,5,7]) # A-flat major
        >>> s.get_interval(68, 72, scale2=s3) # a minor 3rd
        2

        Both pitches present only in second scale:

        >>> s2 = Scale([4,6,8,9,11,1,3]) # E major
        >>> s.get_interval(61, 68, scale2=s2) # a perfect fifth
        4
        """
        return self.nearest_index(pitch2, scale2) - self.nearest_index(
            pitch1, scale2
        )

    def get_interval_class(
        self, pitch1: int, pitch2: int, scale2: t.Optional[Scale] = None
    ):
        """
        >>> s = Scale([0,2,4,5,7,9,11]) # C major
        >>> s.get_interval_class(60, 65)
        3
        >>> s.get_interval_class(60, 86)
        1
        >>> s.get_interval_class(64, 50)
        6
        """
        return self.get_interval(pitch1, pitch2, scale2) % len(self)
