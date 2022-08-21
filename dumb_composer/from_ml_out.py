from typing import Sequence

import numpy as np
from dumb_composer.dumb_accompanist import DumbAccompanist

# from rn_sequences import decode_repr2, repr2_decoded_to_pcs

# TEMP remove TODO
decode_repr2 = repr2_decoded_to_pcs = lambda x: x

DEFAULT_TIMESIGS = "4/4 4/4 4/4 3/4 3/4 6/8 9/8"


def get_chord_df(
    ml_out: Sequence[str],
    ts: str,
    tonic_pc: int,
    *args,
    relative_key_annotations=True,
    **kwargs,
):
    decoded = decode_repr2(
        ml_out,
        ts,
        *args,
        relative_key_annotations=relative_key_annotations,
        **kwargs,
    )
    chord_df = repr2_decoded_to_pcs(
        decoded, tonic_pc, relative_key_annotations=relative_key_annotations
    )
    # TODO replace with a more permamanent solution?
    if np.isnan(chord_df.release.iloc[-1]):
        chord_df.release.iloc[-1] = chord_df.onset.iloc[-1] + 4
    return chord_df


def ml_out_handler(
    ml_out: Sequence[str],
    ts: str,
    tonic_pc: int,
    *args,
    repr_: int = 2,
    text_annotations: bool = False,
    relative_key_annotations=True,
    **kwargs,
):
    if repr_ != 2:
        raise NotImplementedError
    chord_df = get_chord_df(
        ml_out,
        ts,
        tonic_pc,
        *args,
        relative_key_annotations=relative_key_annotations,
        **kwargs,
    )
    dc = DumbAccompanist(text_annotations=text_annotations)
    out_df = dc(chord_df, ts)
    return out_df
