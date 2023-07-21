import re
import sys
from collections import Counter
from typing import Callable, Iterable, Optional, Sequence, Union

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
    """
    >>> _get_bar(0.017857142857142856, char_height=5, bar_width=1, horizontal=True)
    ['▏', ' ', ' ', ' ', ' ']
    """
    ref_chars = HORIZONTAL_CHARS if horizontal else VERTICAL_CHARS

    # Check the bar_prop is valid
    assert 0.0 <= bar_prop <= 1.0

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
    horizontal=False,
    file=sys.stdout,
):
    if bins is None:
        bins = char_width
    counts, bin_edges = np.histogram(array_like, bins=np.array(bins), range=range)
    counter = Counter({edge: count for edge, count in zip(bin_edges, counts)})
    print(counter)
    print_bar(name, counter, char_height, file=file, horizontal=horizontal)


LOREM_IPSUM = (  # for doc-tests
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
    "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex "
    "ea commodo consequat. Duis aute irure dolor in reprehenderit in "
    "voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur "
    "sint occaecat cupidatat non proident, sunt in culpa qui officia "
    "deserunt mollit anim id est laborum."
)


def print_bar(
    name,
    counter,
    char_height=5,
    bar_width=None,
    min_width=20,
    max_bar_width=None,
    horizontal=False,
    file=sys.stdout,
    sort_by_count=False,
    sort_by_key=False,
):
    """
    >>> data = Counter(LOREM_IPSUM)
    >>> print_bar(  # doctest: +NORMALIZE_WHITESPACE
    ...     "Demo 1", data, file=None
    ... )  # the default file `sys.stdout` makes the output not captured by the doctest
    Demo 1 barplot (most common value is ' ': 0.153):
         █
         █▁
     ▁ ▆ ██     ▃▁
     █▅█▂██ ▃█▃▄██ ▁▆
    ▁██████▆██████▂██▂▂▃▂▁▂▂▁▁▂▁

    >>> print_bar(
    ...     "Demo 2", data, bar_width=2, file=None
    ... )  # doctest: +NORMALIZE_WHITESPACE
    Demo 2 barplot (most common value is ' ': 0.153):
              ██
              ██▁▁
      ▁▁  ▆▆  ████          ▃▃▁▁
      ██▅▅██▂▂████  ▃▃██▃▃▄▄████  ▁▁▆▆
    ▁▁████████████▆▆████████████▂▂████▂▂▂▂▃▃▂▂▁▁▂▂▂▂▁▁▁▁▂▂▁▁

    >>> print_bar(
    ...     "Demo 3", data, sort_by_count=True, file=None
    ... )  # doctest: +NORMALIZE_WHITESPACE
    Demo 3 barplot (most common value is ' ': 0.153):
    █
    █▁
    ██▆▃▁▁
    ███████▆▅▄▃▃▂▁
    ██████████████▆▃▂▂▂▂▂▂▂▁▁▁▁▁
    >>> print_bar(
    ...     "Demo 4", data, horizontal=True, char_height=65, sort_by_key=True, file=None
    ... )  # doctest: +NORMALIZE_WHITESPACE
    Demo 4 barplot (most common value is ' ': 0.153):
      █████████████████████████████████████████████████████████████████
    , ███▉
    . ███▉
    D █
    E █
    L █
    U █
    a ███████████████████████████▊
    b ██▉
    c ███████████████▎
    d █████████████████▎
    e ███████████████████████████████████▍
    f ██▉
    g ██▉
    h █
    i ████████████████████████████████████████▏
    l ████████████████████▏
    m ████████████████▎
    n ███████████████████████
    o ███████████████████████████▊
    p ██████████▌
    q ████▊
    r █████████████████████
    s █████████████████▎
    t ██████████████████████████████▋
    u ██████████████████████████▊
    v ██▉
    x ██▉

    """
    if sort_by_count:
        counter = Counter({name: count for name, count in counter.most_common()})
    if sort_by_key:
        counter = Counter({name: counter[name] for name in sorted(counter.keys())})

    most_common_name, most_common_count = counter.most_common(1)[0]
    most_common_freq = most_common_count / sum(counter.values())
    if horizontal:
        assert bar_width in (1, None)
        bar_width = 1
    elif bar_width is None:
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
        f"'{most_common_name}': {most_common_freq:.3f}):",
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


# if __name__ == "__main__":
#     data = np.random.poisson(10, 100)
#     counter = Counter(data)
#     print_bar("Test vertical plot", counter)
#     print_bar("Test horizontal plot", counter, char_height=20, horizontal=True)
