"""Because of the way @cacher works it is advantageous to put music21's
(slow) reading of romanText files into a separate module.
"""
from io import StringIO
import os

import music21

from dumb_composer.utils.cache_lib import cacher

################################################################################
# In order for transposition to work for RomanNumerals we need to patch
#   music21 (at least for now). (See
#   https://github.com/cuthbertLab/music21/issues/1413)
################################################################################


def rn_transpose(self, value, *, inPlace=False):
    """
    Overrides :meth:`~music21.harmony.Harmony.transpose` so that `key`
    attribute is transposed as well.

    >>> rn = music21.roman.RomanNumeral('I', 'C')
    >>> rn
    <music21.roman.RomanNumeral I in C major>
    >>> rn.transpose(4)
    <music21.roman.RomanNumeral I in E major>
    >>> rn.transpose(-4, inPlace=True)
    >>> rn
    <music21.roman.RomanNumeral I in A- major>
    """
    post = super(music21.roman.RomanNumeral, self).transpose(
        value, inPlace=inPlace
    )
    if not inPlace:
        post.key = self.key.transpose(value, inPlace=False)
        return post
    else:
        self.key = self.key.transpose(value, inPlace=False)
        return None


music21.roman.RomanNumeral.transpose = rn_transpose

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
def parse_rntxt(rn_data: str) -> music21.stream.Score:
    return music21.converter.parse(rn_data, format="romanText")


@cacher()
def transpose_and_write_rntxt(rntxt: str, transpose: int):
    score = parse_rntxt(rntxt)
    text_stream = StringIO()
    score.transpose(transpose).write("romanText", text_stream)
    return text_stream.getvalue()
