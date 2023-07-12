import typing as t
from dumb_composer.constants import TIME_TYPE

Pitch = int
PitchClass = int
PitchOrPitchClass = int
ChordFactor = int
TimeStamp = TIME_TYPE
ChromaticInterval = int
ScaleDegree = int
VoiceCount = int

# 'BassFactor' gives the index of each pc in close position starting from the bass.
BassFactor = int

VoiceAssignments = t.Tuple[int, ...]

MelodicAtom = t.Tuple[Pitch | None, Pitch | None]
RNToken = str
