import re
from collections import Counter
import sys
from typing import Iterable, Optional, Callable, Sequence, Union

import numpy as np
import pandas as pd

################################################################################

VERTICAL_CHARS = (
    " ",
    "\u2581",
    "\u2582",
    "\u2583",
    "\u2584",
    "\u2585",
    "\u2586",
    "\u2587",
    "\u2588",
)

HORIZONTAL_CHARS = (
    " ",
    "\u258f",
    "\u258e",
    "\u258d",
    "\u258c",
    "\u258b",
    "\u258a",
    "\u2589",
    "\u2588",
)


def _get_bar(bar_prop, char_height, bar_width, horizontal=False):
    ref_chars = HORIZONTAL_CHARS if horizontal else VERTICAL_CHARS
    assert 0 <= bar_prop <= 1
    height = bar_prop * char_height
    full_chars, frac = divmod(height, 1)
    full_chars = int(full_chars)
    chars = [ref_chars[-1] * bar_width] * full_chars
    frac_char = ref_chars[round(frac * (len(ref_chars) - 1))] * bar_width
    chars.append(frac_char)
    chars.extend([" " * bar_width] * (char_height - len(chars)))
    return chars


def _concat_bars(bars):
    return reversed(list(map(list, zip(*bars))))


def print_histogram(
    array_like,
    char_height=5,
    char_width=20,
    bins=None,
    range=None,
    name="",
    file=sys.stdout,
):
    if bins is None:
        bins = char_width
    counts, bin_edges = np.histogram(
        array_like, bins=np.array(bins), range=range
    )
    counter = Counter({edge: count for edge, count in zip(bin_edges, counts)})
    print(counter)
    print_bar(name, counter, char_height, file=file)


def print_bar(
    name,
    counter,
    char_height=5,
    bar_width=None,
    min_width=20,
    max_bar_width=None,
    horizontal=False,
    file=sys.stdout,
    sort=False,
):
    if sort:
        counter = Counter(
            {name: count for name, count in counter.most_common()}
        )

    most_common_name, most_common_count = counter.most_common(1)[0]
    most_common_freq = most_common_count / sum(counter.values())
    if bar_width is None:
        if len(counter) < min_width:
            bar_width = int(round(min_width / len(counter)))
            if max_bar_width is not None:
                bar_width = min(bar_width, max_bar_width)
        else:
            bar_width = 1
    counts = np.array(list(counter.values()))
    props = counts / counts.max()
    bars = [_get_bar(p, char_height, bar_width, horizontal) for p in props]
    lines = bars if horizontal else _concat_bars(bars)
    print(
        f"{name} barplot (most common value is "
        f"'{most_common_name}': {most_common_freq}):",
        file=file,
    )
    if not horizontal:
        for line in lines:
            print("".join(line), file=file)
        return
    labels = [str(item) for item in counter]
    max_label_len = max(len(label) for label in labels)
    for label, line in zip(labels, lines):
        print(f"{label:>{max_label_len}} " + "".join(line), file=file)


def df_to_bar(
    name: str,
    df: pd.DataFrame,
    cols: Optional[Union[str, Iterable[str]]] = None,
    keys: Optional[str] = None,
    top_n: Optional[int] = None,
    **kwargs,
):
    if cols is None:
        cols = df.columns
    elif isinstance(cols, str):
        cols = (cols,)
    if keys is None:
        idx_col = df.index
    else:
        idx_col = df[keys]
    for col in cols:
        counter = Counter(
            {
                k: v
                for k, v, _ in zip(
                    idx_col,
                    df[col],
                    range(top_n if top_n is not None else len(df)),
                )
            }
        )
        print_bar(name + " " + col, counter=counter, **kwargs)


################################################################################


if __name__ == "__main__":
    data = np.random.poisson(10, 100)
    counter = Counter(data)
    print_bar("Test vertical plot", counter)
    print_bar("Test horizontal plot", counter, char_height=20, horizontal=True)
