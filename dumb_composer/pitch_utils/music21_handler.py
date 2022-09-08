"""Because of the way @cacher works it is advantageous to put music21's
(slow) reading of romanText files into a separate module.
"""
import os

import music21

from dumb_composer.utils.cache_lib import cacher


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
