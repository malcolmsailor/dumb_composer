import typing as t
from abc import ABCMeta, abstractmethod
from copy import copy
from dataclasses import asdict, dataclass
from enum import Enum
from fractions import Fraction
from types import MappingProxyType

TIME_TYPE = Fraction
Pitch = int
PitchClass = int
SpelledPitchClass = str
PitchOrPitchClass = int
ChordFactor = int
TimeStamp = TIME_TYPE
Interval = int
ChromaticInterval = int
ScalarInterval = int
ScaleDegree = int
VoiceCount = int

# 'BassFactor' gives the index of each pc in close position starting from the bass.
BassFactor = int

VoiceAssignments = t.Tuple[int, ...]

MelodicAtom = t.Tuple[Pitch | None, Pitch | None]
RNToken = str

Weight = float

AllowLiteral = t.Literal["YES", "NO", "ONLY"]


@dataclass
class SettingsBase:
    pass


class Voice(Enum):
    pass


class AbstractVoice(Voice):
    GLOBAL = 0


GLOBAL = AbstractVoice.GLOBAL


class OuterVoice(Voice):
    BASS = 0
    MELODY = 1


BASS = OuterVoice.BASS
MELODY = OuterVoice.MELODY


class InnerVoice(Voice):
    TENOR = 2
    ALTO = 3


TENOR = InnerVoice.TENOR
ALTO = InnerVoice.ALTO


class VoicePair(Voice):
    TENOR_AND_ALTO = 4


TENOR_AND_ALTO = VoicePair.TENOR_AND_ALTO


class VoiceGroup(Voice):
    ACCOMPANIMENTS = 5


ACCOMPANIMENTS = VoiceGroup.ACCOMPANIMENTS


class TwoPartResult(t.TypedDict):
    bass: Pitch
    melody: Pitch


class FourPartResult(t.TypedDict):
    bass: Pitch
    tenor: Pitch
    alto: Pitch
    melody: Pitch


voice_string_to_enum = MappingProxyType(
    {
        "bass": OuterVoice.BASS,
        "tenor": InnerVoice.TENOR,
        "alto": InnerVoice.ALTO,
        "soprano": OuterVoice.MELODY,
        "melody": OuterVoice.MELODY,
        "tenor_and_alto": VoicePair.TENOR_AND_ALTO,
        "alto_and_tenor": VoicePair.TENOR_AND_ALTO,
    }
)

voice_enum_to_string: dict[Voice, str] = {
    OuterVoice.BASS: "bass",
    InnerVoice.TENOR: "tenor",
    InnerVoice.ALTO: "alto",
    OuterVoice.MELODY: "melody",
    VoicePair.TENOR_AND_ALTO: "tenor_and_alto",
}


@dataclass
class DFItem:
    def copy(self):
        return copy(self)


@dataclass
class Annotation(DFItem):
    onset: TimeStamp
    text: str
    track: int = 0
    # Not to be changed
    type: t.Literal["text"] = "text"

    def __post_init__(self):
        # TODO remove the next lines when I've figured out how to get
        #   text annotations to display correctly
        self.text = self.text.replace("_", "").replace(" ", "")


@dataclass
class Note(DFItem):
    pitch: Pitch
    onset: TimeStamp
    release: TimeStamp
    tie_to_next: bool = False
    tie_to_prev: bool = False
    grace: bool = False
    # We can use voice, when available, to differentiate ties
    voice: t.Optional[str] = None
    part: t.Optional[str] = None
    spelling: t.Optional[str] = None
    track: int = 1

    # Not to be changed
    type: t.Literal["note"] = "note"
    # _type = "note"

    def is_valid(self) -> bool:
        if self.onset is None:
            return False
        if self.pitch is None:
            return False
        return self.release is not None or self.grace

    @property
    def dur(self) -> TIME_TYPE:
        return self.release - self.onset

    @dur.setter
    def dur(self, val: TimeStamp):
        self.release = self.onset + val

    def copy(self, remove_ties: bool = False) -> "Note":
        out = copy(self)
        if remove_ties:
            out.tie_to_next = False
            out.tie_to_prev = False
        return out

    def partial_copy(self, remove_ties: bool = False, **kwargs) -> "Note":
        out = self.copy(remove_ties=remove_ties)
        for attr_name, attr_val in kwargs.items():
            setattr(out, attr_name, attr_val)
        return out

    def set_spelling(self, step: str, alter: int):
        if alter < 0:
            acc = -alter * "-"
        else:
            acc = alter * "#"
        self.spelling = step + acc

    def __repr__(self):
        return str(self)

    def __str__(self):
        init_tie = "⌒" if self.tie_to_prev else ""
        end_tie = "⌒" if self.tie_to_next else ""
        return f"{init_tie}{self.pitch}:{self.onset}-{self.release}{end_tie}"


Simultaneity = t.Sequence[Note | None]
NoteChange = tuple[Note | None, Note | None]
SimultaneousNoteChange = tuple[NoteChange, ...]


class RecursiveWorker(t.Protocol):
    def step(self) -> t.Iterator[t.Any]:
        raise NotImplementedError

    def append_attempt(self, step_result: t.Any):
        raise NotImplementedError

    @property
    def step_i(self) -> int:
        raise NotImplemented

    @property
    def finished(self) -> bool:
        raise NotImplementedError

    @property
    def ready(self) -> bool:
        raise NotImplementedError


@dataclass
class Suspension:
    pitch: Pitch
    resolves_by: int
    dissonant: bool
    interval_above_bass: int
    # "score" is meant to be used to weight how likely we are to use
    #   each suspension.
    score: float = 1.0
    begins_on_prev: bool = False
    resolves_on_next: bool = True


@dataclass
class AbstractSuspension:
    pc: PitchClass
    resolves_by: ChromaticInterval
    dissonant: bool
    begins_on_prev: bool = False
    resolves_on_next: bool = True
