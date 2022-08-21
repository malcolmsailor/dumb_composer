"""Takes a "homophonic" dataframe and converts it to a "midi" dataframe.

"Homophonic" dataframe has columns "onset" and "release"; subsequent columns are
understood to be voices.

"Midi_df" has columns "onset", "type", "pitch", "release", "track", and "other"
"""

import pandas as pd


def homodf_to_mididf(homodf: pd.DataFrame) -> pd.DataFrame:
    out = []
    voice_cols = list(
        reversed(
            [col for col in homodf.columns if col not in ("onset", "release")]
        )
    )
    for i, row in homodf.iterrows():
        for i, voice_col in enumerate(voice_cols):
            out.append(
                {
                    "onset": row.onset,
                    "type": "note",
                    "pitch": row[voice_col],
                    "release": row.release,
                    "track": i + 1,
                }
            )
    return pd.DataFrame(out)
