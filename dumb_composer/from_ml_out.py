from typing import Sequence
from .dumb_composer import DumbComposer

from rn_sequences import decode_repr2, repr2_decoded_to_pcs

DEFAULT_TIMESIGS = "4/4 4/4 4/4 3/4 3/4 6/8 9/8"


def ml_out_handler(
    ml_out: Sequence[str],
    ts: str,
    tonic_pc: int,
    *args,
    repr_: int = 2,
    text_annotations: bool = False,
    relative_key_annotations=True,
    **kwargs
):
    if repr_ != 2:
        raise NotImplementedError
    decoded = decode_repr2(
        ml_out,
        ts,
        *args,
        relative_key_annotations=relative_key_annotations,
        **kwargs
    )
    chord_df = repr2_decoded_to_pcs(
        decoded, tonic_pc, relative_key_annotations=relative_key_annotations
    )

    dc = DumbComposer(text_annotations=text_annotations)
    out_df = dc(chord_df, ts)
    return out_df
