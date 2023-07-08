from collections import defaultdict
from dataclasses import dataclass
import logging
from numbers import Number
import typing as t
import random
import warnings
import pandas as pd
from .pitch_utils.chords import Chord, Inflection

from dumb_composer.pitch_utils.intervals import (
    get_relative_chord_factors,
)
from .utils.math_ import weighted_sample_wo_replacement
from .utils.recursion import DeadEnd
from .pitch_utils.scale import Scale, ScaleDict
from dumb_composer.shared_classes import Note, Score

from dumb_composer.prefabs.prefab_pitches import (
    PrefabPitchDirectory,
    PrefabPitches,
)
from dumb_composer.prefabs.prefab_rhythms import (
    PrefabRhythmDirectory,
    PrefabRhythms,
    SingletonRhythm,
)

# TODO prefab "inertia": more likely to select immediately preceding prefab
#    if possible (or choose from previous N prefabs)


@dataclass
class PrefabApplierSettings:
    # either "soprano", "tenor", or "bass"
    prefab_voice: str = "soprano"

    # prefab_inertia is approximately "the probability of choosing the Nth
    #   last prefab applied to a segment of the same length". E.g., if the
    #   segment is length 2, and prefab_inertia is (0.5, 0.2), then there is
    #   about a 50% chance of choosing the last-chosen length-2 prefab, and
    #   a 20% chance of choosing the second-last-chosen length-2 prefab. The
    #   remaining probability mass is distributed to the other prefabs.
    #   The items in prefab_inertia should be in (0.0, 1.0) and should sum to
    #   at most 1.
    prefab_inertia: t.Optional[t.Tuple[float, ...]] = (0.5, 0.2)
    forbidden_parallels: t.Sequence[int] = (7, 0)


class PrefabApplier:
    prefab_rhythm_dir = PrefabRhythmDirectory()
    prefab_pitch_dir = PrefabPitchDirectory()

    def __init__(self, settings: t.Optional[PrefabApplierSettings] = None):
        if settings is None:
            settings = PrefabApplierSettings()
        self.settings = settings
        self._prefab_rhythm_stacks: t.DefaultDict[
            Number, t.List[PrefabRhythms]
        ] = defaultdict(list)
        self._prefab_pitch_stacks: t.DefaultDict[t.List[PrefabPitches]] = defaultdict(
            list
        )

    def _append_from_prefabs(
        self,
        initial_pitch: int,
        current_chord: Chord,
        prefab_pitches: PrefabPitches,
        prefab_rhythms: PrefabRhythms,
        scale: Scale,
        next_mel_pitch: int,
        track: int = 1,
    ) -> t.List[Note]:
        notes = []
        # TODO if and when we allow "chromatic" suspensions (i.e.,
        #   suspensions that belong to the chord/scale of the preparation
        #   but not to the chord/scale of the suspension) scale.index can
        #   fail.
        orig_scale_degree = scale.index(initial_pitch)
        if isinstance(prefab_rhythms, SingletonRhythm):
            releases = [current_chord.release - current_chord.onset]  # type:ignore
        else:
            releases = prefab_rhythms.releases
        offset = current_chord.onset

        for i, (rel_scale_degree, onset, release) in enumerate(
            zip(prefab_pitches.relative_degrees, prefab_rhythms.onsets, releases)
        ):
            if i in prefab_pitches.alterations:
                new_pitch = scale.get_auxiliary(
                    orig_scale_degree + rel_scale_degree,
                    prefab_pitches.alterations[i],
                )
            else:
                new_pitch = scale[orig_scale_degree + rel_scale_degree]
            if (
                i > 0
                and abs(
                    (
                        current_mod_degree := (orig_scale_degree + rel_scale_degree)
                        % len(scale)
                    )
                    - (
                        prev_mod_degree := (
                            orig_scale_degree + prefab_pitches.relative_degrees[i - 1]
                        )
                        % len(scale)
                    )
                )
                in (1, len(scale) - 1)
                and abs(new_pitch - notes[-1].pitch) > 2
            ):
                # The abs(new_pitch - notes[-1].pitch) > 2 condition is
                #   necessary because we only want to run this logic if there
                #   actually *is* an augmented second because of the case where
                #   an augmented second is immediately repeated. E.g., given
                #   pitches (G#, F, G#), if we raise the F after processing the
                #   first pair of pitches, we would raise it *again* after
                #   processing the second pair.
                aug2nds = current_chord.augmented_second_adjustments
                if all(
                    degree in aug2nds
                    for degree in (prev_mod_degree, current_mod_degree)
                ):
                    if (
                        aug2nds[current_mod_degree] is Inflection.NONE
                        and aug2nds[prev_mod_degree] is Inflection.NONE
                    ):
                        pass
                    elif aug2nds[current_mod_degree] is Inflection.UP:
                        new_pitch += 1
                    elif aug2nds[current_mod_degree] is Inflection.DOWN:
                        new_pitch -= 1
                    elif aug2nds[prev_mod_degree] is Inflection.UP:
                        notes[-1].pitch += 1
                    elif aug2nds[prev_mod_degree] is Inflection.DOWN:
                        notes[-1].pitch -= 1
                    else:
                        # Inflection of both pitches must be EITHER; we adjust
                        # the previous pitch.
                        if new_pitch > notes[-1].pitch:
                            notes[-1].pitch += 1
                        else:
                            notes[-1].pitch -= 1
            notes.append(Note(new_pitch, onset + offset, release + offset, track=track))
        if prefab_pitches.tie_to_next:
            notes[-1].tie_to_next = True
        # TODO I need to finish implementing augmented 2nds here
        # elif abs(notes[-1].pitch - next_mel_pitch) > 2 and :
        #     pass
        return notes

    def get_decorated_voice(self, score: Score):
        if self.settings.prefab_voice == "bass":
            return score.structural_bass
        return score.structural_melody

    def get_nondecorated_structural_voice(self, score: Score):
        if self.settings.prefab_voice == "bass":
            return score.structural_melody
        return score.structural_bass

    def _get_prefab_weights(
        self,
        prefab_options: t.List[t.Union[PrefabPitches, PrefabRhythms]],
        prefab_stack_key: t.Any,
        prefab_type: t.Type,
    ) -> t.List[float]:
        """Weights will not sum to 1 in those cases where all prefab_options
        are in the first n items of the stack.

        >>> prefab_applier = PrefabApplier()
        >>> segment_dur = 3.0
        >>> rhythm_options = prefab_applier.prefab_rhythm_dir(segment_dur)
        >>> prefab_applier._get_prefab_weights(  # doctest: +SKIP
        ...     prefab_options=rhythm_options,
        ...     prefab_stack_key=segment_dur,
        ...     prefab_type=PrefabRhythms)
        [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
        """
        if prefab_type is PrefabRhythms:
            stacks = self._prefab_rhythm_stacks
        else:
            stacks = self._prefab_pitch_stacks
        stack = stacks[prefab_stack_key]
        prefab_inertia = self.settings.prefab_inertia
        temp = {}
        remaining_mass = 1.0
        for prefab, prob in zip(reversed(stack), prefab_inertia):  # type:ignore
            if prefab in prefab_options:
                temp[prefab] = prob
                remaining_mass -= prob
        if len(prefab_options) == len(temp):
            return [temp.get(prefab) for prefab in prefab_options]  # type:ignore
        epsilon = remaining_mass / (len(prefab_options) - len(temp))
        return [temp.get(prefab, epsilon) for prefab in prefab_options]

    def _step(self, score: Score):
        current_i = len(score.prefabs)
        current_scale = score.scales[current_i]
        next_scale = score.scales[current_i + 1]
        current_chord = score.chords[current_i]
        decorated_voice = self.get_decorated_voice(score)

        current_mel_pitch = decorated_voice[current_i]
        next_mel_pitch = decorated_voice[current_i + 1]
        is_suspension = current_i in score.suspension_indices
        is_preparation = current_i + 1 in score.suspension_indices
        is_resolution = current_i - 1 in score.suspension_indices
        is_after_tie = current_i - 1 in score.tied_prefab_indices
        start_with_rest = (
            None
            if current_i not in score.allow_prefab_start_with_rest
            else score.allow_prefab_start_with_rest[current_i]
        )
        # TODO I use segment_dur as key to the prefab_stacks. But I think
        #   maybe I should use a string that combines the segment_dur with
        #   the metric weight of the start and end point.
        segment_dur = current_chord.release - current_chord.onset  # type:ignore
        rhythm_options = self.prefab_rhythm_dir(
            segment_dur,  # type:ignore
            # TODO metric strength
            is_suspension=is_suspension,
            is_preparation=is_preparation,
            is_resolution=is_resolution,
            is_after_tie=is_after_tie,
            start_with_rest=start_with_rest,
        )
        rhythm_weights = self._get_prefab_weights(
            rhythm_options, segment_dur, PrefabRhythms  # type:ignore
        )
        for rhythm in weighted_sample_wo_replacement(rhythm_options, rhythm_weights):
            self._prefab_rhythm_stacks[segment_dur].append(rhythm)
            score.allow_prefab_start_with_rest[
                current_i + 1
            ] = rhythm.allow_next_to_start_with_rest
            generic_melody_pitch_interval = current_scale.get_interval(
                current_mel_pitch, next_mel_pitch, scale2=next_scale
            )
            interval_is_diatonic = current_scale.pitch_is_diatonic(next_mel_pitch)
            relative_chord_factors = get_relative_chord_factors(
                0
                if self.settings.prefab_voice == "bass"
                else score.structural_melody_intervals[current_i],
                current_chord.intervals_above_bass,
                len(current_scale),
            )
            pitch_options = self.prefab_pitch_dir(
                generic_melody_pitch_interval,
                rhythm.metric_strength_str,
                relative_chord_factors,
                is_suspension=is_suspension,
                is_preparation=is_preparation,
                interval_is_diatonic=interval_is_diatonic,
            )
            pitch_weights = self._get_prefab_weights(
                # I'm not completely sure what the appropriate value for
                #   "prefab_stack_key" is; trying "segment_dur". If I change
                #   it I also need to change it when appending/popping from
                #   self._prefab_pitch_stacks below
                pitch_options,
                segment_dur,
                PrefabPitches,
            )
            for pitches in weighted_sample_wo_replacement(pitch_options, pitch_weights):
                self._prefab_pitch_stacks[segment_dur].append(pitches)
                if pitches.tie_to_next:
                    assert current_i not in score.tied_prefab_indices
                    score.tied_prefab_indices.add(current_i)
                out = self._append_from_prefabs(
                    current_mel_pitch,
                    current_chord,
                    pitches,
                    rhythm,
                    current_scale,
                    next_mel_pitch,
                    track=score.prefab_track,
                )
                if not self._has_forbidden_parallels(current_i, score, out):
                    logging.debug(
                        f"{self.__class__.__name__} yielding notes "
                        + " ".join(str(note) for note in out)
                    )
                    yield out
                if pitches.tie_to_next:
                    score.tied_prefab_indices.remove(current_i)
                self._prefab_pitch_stacks[segment_dur].pop()
            del score.allow_prefab_start_with_rest[current_i + 1]
            self._prefab_rhythm_stacks[segment_dur].pop()
        raise DeadEnd()

    def _has_forbidden_parallels(
        self, current_i: int, score: Score, realized_notes: t.List[Note]
    ) -> bool:
        decorated_voice = self.get_decorated_voice(score)
        this_next_pitch = decorated_voice[current_i + 1]
        nondecorated_voice = self.get_nondecorated_structural_voice(score)
        other_next_pitch = nondecorated_voice[current_i + 1]
        harmonic_interval = abs(this_next_pitch - other_next_pitch) % 12
        logging.debug(f"harmonic_interval={harmonic_interval}")
        if harmonic_interval not in self.settings.forbidden_parallels:
            return False
        other_current_pitch = nondecorated_voice[current_i]
        other_mel_interval = other_next_pitch - other_current_pitch
        this_mel_interval = this_next_pitch - realized_notes[-1].pitch
        return other_mel_interval == this_mel_interval

    def _final_step(self, score: Score):
        # TODO eventually it would be nice to be able to decorate the last note
        #   etc.
        current_i = len(score.prefabs)
        current_chord = score.chords[current_i]
        out = Note(
            self.get_decorated_voice(score)[current_i],
            current_chord.onset,  # type:ignore
            current_chord.release,  # type:ignore
        )
        logging.debug(f"{self.__class__.__name__} yielding note {str(out)}")
        yield [out]
