import logging
import typing as t

from dumb_composer.constants import TRACKS
from dumb_composer.pitch_utils.chords import Chord, is_same_harmony
from dumb_composer.pitch_utils.scale import Scale
from dumb_composer.pitch_utils.types import (
    GLOBAL,
    Annotation,
    OuterVoice,
    Pitch,
    PitchClass,
    ScalarInterval,
    TimeStamp,
    Voice,
)
from dumb_composer.suspensions import Suspension, SuspensionCombo
from dumb_composer.time import Meter

LOGGER = logging.getLogger(__name__)

from dumb_composer.classes.scores import _ScoreBase


class ScoreInterface:
    def __init__(
        self,
        score: _ScoreBase,
        get_i: t.Callable[[_ScoreBase], int],
        validate: t.Callable[[_ScoreBase], bool],
    ):
        self._score = score
        self._get_i = get_i
        # TODO: (Malcolm 2023-08-04) perhaps validation can be implemented more for
        #   user
        self._validate = validate

    @property
    def ts(self) -> Meter:
        return self._score.ts

    @property
    def score(self):  # pylint: disable=missing-docstring
        return self._score

    # ----------------------------------------------------------------------------------
    # State and validation
    # ----------------------------------------------------------------------------------

    def validate_state(self) -> bool:
        return self._validate(self._score)

    @property
    def i(self):
        return self._get_i(self._score)

    @property
    def empty(self) -> bool:
        # TODO: (Malcolm 2023-08-09) think about validation here.
        # assert self.validate_state()
        return self.i == 0

    @property
    def complete(self) -> bool:
        # assert self.validate_state()
        assert self.i <= len(self._score.chords)
        return self.i == len(self._score.chords)

    @property
    def structural_lengths(self) -> list[int]:
        return [len(v) for v in self._score._structural.values()]

    # ----------------------------------------------------------------------------------
    # Chords, etc.
    # ----------------------------------------------------------------------------------

    def annotate_chords(self):
        global_annots = self._score.annotations[GLOBAL]
        if "chords" in global_annots:
            LOGGER.warning("chords already appear to be annotated, skipping")
            return
        chord_track = TRACKS["abstract"][GLOBAL]
        for chord in self._score.chords:
            global_annots["chords"][chord.onset] = Annotation(
                chord.onset, chord.token, track=chord_track
            )

    @property
    def current_chord(self) -> Chord:
        return self._score.chords[self.i]

    @property
    def prev_chord(self) -> Chord:
        assert self.i > 0
        return self._score.chords[self.i - 1]

    @property
    def next_chord(self) -> Chord | None:
        if self.i + 1 >= len(self._score.chords):
            return None
        return self._score.chords[self.i + 1]

    @property
    def current_scale(self) -> Scale:
        return self._score.scales[self.i]

    @property
    def prev_scale(self) -> Scale:
        assert self.i > 0
        return self._score.scales[self.i - 1]

    @property
    def next_scale(self) -> Scale | None:
        if self.i + 1 >= len(self._score.scales):
            return None
        return self._score.scales[self.i + 1]

    @property
    def prev_foot_pc(self) -> PitchClass:
        """The "notional" bass pitch-class.

        It is "notional" because the pitch-class of the *actual* bass may be a
        suspension, etc. The "foot" differs from the "root" in that it isn't necessarily
        the root. E.g., in a V6 chord in C major, with a bass suspension C--B, on the C,
        the foot is the third B.
        """
        if self.i < 1:
            raise ValueError()
        return self._score.pc_bass[self.i - 1]

    @property
    def current_foot_pc(self) -> PitchClass:
        """The "notional" bass pitch-class.

        It is "notional" because the pitch-class of the *actual* bass may be a
        suspension, etc. The "foot" differs from the "root" in that it isn't necessarily
        the root. E.g., in a V6 chord in C major, with a bass suspension C--B, on the C,
        the foot is the third B.
        """
        return self._score.pc_bass[self.i]

    @property
    def next_foot_pc(self) -> PitchClass | None:
        """The "notional" bass pitch-class.

        It is "notional" because the pitch-class of the *actual* bass may be a
        suspension, etc. The "foot" differs from the "root" in that it isn't necessarily
        the root. E.g., in a V6 chord in C major, with a bass suspension C--B, on the C,
        the foot is the third B.
        """
        if self.i + 1 >= len(self._score.chords):
            return None
        return self._score.pc_bass[self.i + 1]

    # ----------------------------------------------------------------------------------
    # Pitches and intervals
    # ----------------------------------------------------------------------------------

    def prev_pitch(self, voice: Voice, offset_from_current: int = 1) -> Pitch:
        if self.i < offset_from_current:
            raise ValueError
        return self._score._structural[voice][self.i - offset_from_current]

    def current_pitch(self, voice: Voice) -> Pitch | None:
        try:
            return self._score._structural[voice][self.i]
        except IndexError:
            return None

    def prev_pitches(self) -> tuple[Pitch]:
        assert self.i > 0
        out = (pitches[self.i - 1] for pitches in self._score._structural.values())
        return tuple(sorted(out))

    def prev_structural_interval_above_bass(self, voice: Voice) -> ScalarInterval:
        if voice is OuterVoice.BASS:
            return 0
        return self.prev_scale.get_reduced_scalar_interval(
            self.prev_pitch(OuterVoice.BASS), self.prev_pitch(voice)
        )

    # ----------------------------------------------------------------------------------
    # Inspect suspensions
    # ----------------------------------------------------------------------------------

    def prev_suspension(
        self, voice: Voice  # , offset_from_current: int = 1
    ) -> Suspension | None:
        # if self.i < offset_from_current:
        #     raise ValueError
        if self.i < 1:
            raise ValueError()
        return self._score._suspensions[voice].get(self.prev_chord.onset, None)
        # return self._score._suspensions[voice].get(self.i - offset_from_current, None)

    def prev_is_suspension(self, voice: Voice) -> bool:
        return self.prev_suspension(voice) is not None

    def current_suspension(self, voice: Voice) -> Suspension | None:
        # return self._score._suspensions[voice].get(self.i, None)
        return self._score._suspensions[voice].get(self.current_chord.onset, None)

    def current_is_suspension(self, voice: Voice) -> bool:
        return self.current_suspension(voice) is not None

    def suspension_in_any_voice(self) -> bool:
        for suspensions in self._score._suspensions.values():
            if self.current_chord.onset in suspensions:
                return True
        return False

    # ----------------------------------------------------------------------------------
    # Suspension resolutions
    # ----------------------------------------------------------------------------------

    def prev_resolution(self, voice: Voice) -> Pitch | None:
        assert self.i > 0
        return self._score._suspension_resolutions[voice].get(
            self.prev_chord.onset, None
        )
        # return self._score._suspension_resolutions[voice].get(self.i - 1, None)

    def prev_is_resolution(self, voice: Voice) -> bool:
        return self.prev_resolution(voice) is not None

    def current_resolution(self, voice: Voice) -> Pitch | None:
        return self._score._suspension_resolutions[voice].get(
            self.current_chord.onset, None
        )
        # return self._score._suspension_resolutions[voice].get(self.i, None)

    def all_current_resolutions(self) -> dict[Voice, Pitch | None]:
        out = {}
        for voice, suspensions in self._score._suspension_resolutions.items():
            out[voice] = suspensions.get(self.current_chord.onset, None)
        return out

    def current_is_resolution(self, voice: Voice) -> bool:
        return self.current_resolution(voice) is not None

    def current_resolution_pcs(self):
        out = set()
        for voice in self._score._suspension_resolutions:
            if self.i in self._score._suspension_resolutions[voice]:
                out.add(self._score._suspension_resolutions[voice][self.i] % 12)
        return out

    def resolution_in_any_voice(self, time: TimeStamp) -> bool:
        for resolutions in self._score._suspension_resolutions.values():
            if time in resolutions:
                return True
        return False

    # ----------------------------------------------------------------------------------
    # Suspension preparations
    # ----------------------------------------------------------------------------------

    def prev_is_preparation(self, voice: Voice) -> bool:
        suspension = self.current_suspension
        if suspension is None:
            return False
        return self._score._structural[voice] == suspension.pitch

    # ----------------------------------------------------------------------------------
    # Alter chords
    # ----------------------------------------------------------------------------------
    def split_current_chord_at(self, time: TimeStamp):
        LOGGER.debug(
            f"Splitting chord at {time=} w/ duration "
            f"{self.current_chord.release - self.current_chord.onset}"
        )
        self._score.split_ith_chord_at(self.i, time)
        LOGGER.debug(
            f"After split chord has duration "
            f"{self.current_chord.release - self.current_chord.onset}"
        )

    def merge_current_chords_if_they_were_previously_split(self):
        if self._score._split_chords[self.i]:
            assert self.next_chord is not None
            assert is_same_harmony(self.current_chord, self.next_chord)
            self._score.merge_ith_chords(self.i)

    # ----------------------------------------------------------------------------------
    # Add and remove suspensions
    # ----------------------------------------------------------------------------------

    def _add_suspension(self, voice: Voice, suspension: Suspension, onset: TimeStamp):
        suspensions = self._score._suspensions[voice]
        assert onset not in suspensions
        suspensions[onset] = suspension

    def _remove_suspension(self, voice: Voice, onset: TimeStamp):
        suspensions = self._score._suspensions[voice]
        suspensions.pop(onset)  # raises KeyError if missing

    def _annotate_suspension(self, voice: Voice, onset: TimeStamp):
        annots = self._score.annotations[voice]["suspensions"]
        annots[onset] = Annotation(onset, "S", track=TRACKS["structural"][voice])

    def _remove_suspension_annotation(self, voice: Voice, onset: TimeStamp):
        annots = self._score.annotations[voice]["suspensions"]
        annots.pop(onset)

    def _add_suspension_resolution(
        self, voice: Voice, pitch: Pitch, release: TimeStamp
    ):
        assert not release in self._score._suspension_resolutions[voice]
        self._score._suspension_resolutions[voice][release] = pitch

    def _remove_suspension_resolution(self, voice: Voice, release: TimeStamp):
        assert release in self._score._suspension_resolutions[voice]
        del self._score._suspension_resolutions[voice][release]

    def apply_suspension(
        self,
        suspension: Suspension,
        suspension_release: TimeStamp,
        voice: Voice,
        annotate: bool = True,
    ) -> Pitch:
        LOGGER.debug(f"applying {suspension=} with {suspension_release=} at {self.i=}")
        # if self.i >= 10:
        #     breakpoint()
        if suspension_release < self.current_chord.release:
            # If the suspension resolves during the current chord, we need to split
            #   the current chord to permit that
            self.split_current_chord_at(suspension_release)
        else:
            # Otherwise, make sure the suspension resolves at the onset of the
            #   next chord
            assert (
                self.next_chord is not None
                and suspension_release == self.next_chord.onset
            )
        # TODO: (Malcolm 2023-08-10) I don't know why we were calculating the suspended
        #   pitch like this rather than retrieving it directly from the pitch attribute
        #   of the suspension
        # suspended_pitch = self.prev_pitch(voice)
        suspended_pitch = suspension.pitch
        self._add_suspension_resolution(
            voice, suspended_pitch + suspension.resolves_by, suspension_release
        )
        self._add_suspension(voice, suspension, self.current_chord.onset)
        if annotate:
            self._annotate_suspension(voice, self.current_chord.onset)
        return suspended_pitch

    def apply_suspensions(
        self,
        suspension_combo: SuspensionCombo,
        suspension_release: TimeStamp,
        annotate: bool = True,
    ):
        LOGGER.debug(f"applying {suspension_combo=} at {self.i=}")
        for voice, suspension in suspension_combo.items():
            self.apply_suspension(
                suspension,
                suspension_release=suspension_release,
                voice=voice,
                annotate=annotate,
            )

    def undo_suspension(
        self, voice: Voice, suspension_release: TimeStamp, annotate: bool = True
    ) -> None:
        LOGGER.debug(f"undoing suspension in {voice=} at {self.i=}")
        self._remove_suspension_resolution(voice, suspension_release)
        # TODO: (Malcolm 2023-08-10) double check this is the right onset
        self._remove_suspension(voice, self.current_chord.onset)
        if annotate:
            self._remove_suspension_annotation(voice, self.current_chord.onset)
        # if not self.suspension_in_any_voice():
        if not self.resolution_in_any_voice(self.current_chord.release):
            self.merge_current_chords_if_they_were_previously_split()

    def undo_suspensions(
        self,
        suspension_release: TimeStamp,
        suspension_combo: SuspensionCombo,
        annotate: bool = True,
    ):
        LOGGER.debug(f"undoing {suspension_combo=} at {self.i=}")
        for voice in suspension_combo:
            self.undo_suspension(voice, suspension_release, annotate)

    # TODO: (Malcolm 2023-08-04) prev_structural_interval_above_bass?

    def at_chord_change(
        self,
        compare_scales: bool = True,
        compare_inversions: bool = True,
        allow_subsets: bool = False,
    ) -> bool:
        return self.empty or not is_same_harmony(
            self.prev_chord,
            self.current_chord,
            compare_scales,
            compare_inversions,
            allow_subsets,
        )
