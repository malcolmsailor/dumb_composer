"""Because of the way @cacher works it is advantageous to put music21's
(slow) reading of romanText files into a separate module.
"""
import os
from io import StringIO

import music21
from cache_lib import cacher

################################################################################
# In order for transposition to work for RomanNumerals we need to patch
#   music21 (at least for now). (See
#   https://github.com/cuthbertLab/music21/issues/1413)
################################################################################


def rn_transpose(self, value, *, inPlace=False):
    """
    Overrides :meth:`~music21.harmony.Harmony.transpose` so that `key`
    attribute is transposed as well.

    >>> rn = music21.roman.RomanNumeral("I", "C")
    >>> rn
    <music21.roman.RomanNumeral I in C major>
    >>> rn.transpose(4)
    <music21.roman.RomanNumeral I in E major>
    >>> rn.transpose(-4, inPlace=True)
    >>> rn
    <music21.roman.RomanNumeral I in A- major>
    """
    post = super(music21.roman.RomanNumeral, self).transpose(value, inPlace=inPlace)
    if not inPlace:
        post.key = self.key.transpose(value, inPlace=False)  # type:ignore
        return post
    else:
        self.key = self.key.transpose(value, inPlace=False)
        return None


music21.roman.RomanNumeral.transpose = rn_transpose  # type:ignore

import copy
import typing as t

from music21 import key, roman, stream
from music21.romanText import rtObjects
from music21.romanText.translate import RomanTextTranslateException


def _copySingleMeasure(rtTagged, p, kCurrent):
    """
    Given a RomanText token, a Part used as the current container,
    and the current Key, return a Measure copied from the past of the Part.

    This is used in cases of definitions such as:
    m23=m21
    """
    m = None
    # copy from a past location; need to change key
    # environLocal.printDebug(['calling _copySingleMeasure()'])
    targetNumber, unused_targetRepeat = rtTagged.getCopyTarget()
    if len(targetNumber) > 1:  # pragma: no cover
        # this is an encoding error
        raise RomanTextTranslateException(
            "a single measure cannot define a copy operation for multiple measures"
        )
    # TODO: ignoring repeat letters
    target = targetNumber[0]
    for mPast in p.getElementsByClass(stream.Measure):  # type:ignore
        if mPast.number == target:
            try:
                m = copy.deepcopy(mPast)
            except TypeError:  # pragma: no cover
                raise RomanTextTranslateException(
                    f"Failed to copy measure {mPast.number}:"
                    + " did you perhaps parse an RTOpus object with romanTextToStreamScore "
                    + "instead of romanTextToStreamOpus?"
                )
            m.number = rtTagged.number[0]
            # update all keys
            for rnPast in m.getElementsByClass(roman.RomanNumeral):
                if kCurrent is None:  # pragma: no cover
                    # should not happen
                    raise RomanTextTranslateException(
                        "attempting to copy a measure but no past key definitions are found"
                    )
                if rnPast.editorial.get("followsKeyChange"):
                    kCurrent = rnPast.key
                elif rnPast.pivotChord is not None:
                    kCurrent = rnPast.pivotChord.key
                else:
                    rnPast.key = kCurrent
                if rnPast.secondaryRomanNumeral is not None:
                    newRN = roman.RomanNumeral(rnPast.figure, copy.deepcopy(kCurrent))
                    newRN.duration = copy.deepcopy(rnPast.duration)
                    newRN.lyrics = copy.deepcopy(rnPast.lyrics)
                    m.replace(rnPast, newRN)

            break
    return m, kCurrent


def _copyMultipleMeasures(
    rtMeasure: rtObjects.RTMeasure,
    p: stream.Part,  # type:ignore
    kCurrent: t.Optional[key.Key],
):
    """
    Given a RomanText token for a RTMeasure, a
    Part used as the current container, and the current Key,
    return a Measure range copied from the past of the Part.

    This is used for cases such as:
    m23-25 = m20-22
    """
    # the key provided needs to be the current key
    # environLocal.printDebug(['calling _copyMultipleMeasures()'])

    targetNumbers, unused_targetRepeat = rtMeasure.getCopyTarget()
    if len(targetNumbers) == 1:  # pragma: no cover
        # this is an encoding error
        raise RomanTextTranslateException(
            "a multiple measure range cannot copy a single measure"
        )
    # TODO: ignoring repeat letters
    targetStart = targetNumbers[0]
    targetEnd = targetNumbers[1]

    if (
        rtMeasure.number[1] - rtMeasure.number[0] != targetEnd - targetStart
    ):  # pragma: no cover
        raise RomanTextTranslateException(
            "both the source and destination sections need to have the same number of measures"
        )
    if rtMeasure.number[0] < targetEnd:  # pragma: no cover
        raise RomanTextTranslateException(
            "the source section cannot overlap with the destination section"
        )

    measures = []
    for mPast in p.getElementsByClass(stream.Measure):  # type:ignore
        if mPast.number in range(targetStart, targetEnd + 1):
            try:
                m = copy.deepcopy(mPast)
            except TypeError:  # pragma: no cover
                raise RomanTextTranslateException(
                    "Failed to copy measure {0} to measure range {1}-{2}: ".format(
                        mPast.number, targetStart, targetEnd
                    )
                    + "did you perhaps parse an RTOpus object with romanTextToStreamScore "
                    + "instead of romanTextToStreamOpus?"
                )

            m.number = rtMeasure.number[0] + mPast.number - targetStart
            measures.append(m)
            # update all keys
            allRNs = list(m.getElementsByClass(roman.RomanNumeral))
            for rnPast in allRNs:
                if kCurrent is None:  # pragma: no cover
                    # should not happen
                    raise RomanTextTranslateException(
                        "attempting to copy a measure but no past key definitions are found"
                    )
                if rnPast.editorial.get("followsKeyChange"):
                    kCurrent = rnPast.key  # type:ignore
                elif rnPast.pivotChord is not None:
                    kCurrent = rnPast.pivotChord.key  # type:ignore
                else:
                    rnPast.key = kCurrent
                if rnPast.secondaryRomanNumeral is not None:
                    newRN = roman.RomanNumeral(rnPast.figure, copy.deepcopy(kCurrent))
                    newRN.duration = copy.deepcopy(rnPast.duration)
                    newRN.lyrics = copy.deepcopy(rnPast.lyrics)
                    m.replace(rnPast, newRN)

        if mPast.number == targetEnd:
            break
    return measures, kCurrent


music21.romanText.translate._copySingleMeasure = _copySingleMeasure
music21.romanText.translate._copyMultipleMeasures = _copyMultipleMeasures


################################################################################
# End patch
################################################################################


def write_music21_cache(return_value, cache_path):
    # fp must be full path
    abs_path = os.path.abspath(cache_path)
    music21.freezeThaw.StreamFreezer(return_value).write(fp=abs_path)


def read_music21_cache(cache_path):
    # fp must be full path
    abs_path = os.path.abspath(cache_path)
    thawer = music21.freezeThaw.StreamThawer()
    thawer.open(abs_path)
    return thawer.stream


@cacher(write_cache_f=write_music21_cache, read_cache_f=read_music21_cache)
def parse_rntxt(rn_data: str) -> music21.stream.Score:  # type:ignore
    # forceSource=True disables music21's caching, which we want to do because
    #   music21 doesn't update cache after music21 itself is changed.
    return music21.converter.parse(
        rn_data, format="romanText", forceSource=True  # type:ignore
    )


@cacher()
def transpose_and_write_rntxt(rntxt: str, transpose: int):
    score = parse_rntxt(rntxt)
    text_stream = StringIO()
    score.transpose(transpose).write("romanText", text_stream)  # type:ignore
    return text_stream.getvalue()


def get_ts_from_rntxt(rn_data: str) -> str:
    score = parse_rntxt(rn_data)
    m21_ts = score[music21.meter.TimeSignature].first()  # type:ignore
    ts_str = f"{m21_ts.numerator}/{m21_ts.denominator}"  # type:ignore
    return ts_str
