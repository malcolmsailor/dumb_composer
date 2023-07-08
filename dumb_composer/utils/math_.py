from collections import Counter  # used by doctest
import random
import typing as t
import itertools as it
import numpy as np


def set_seeds(seed: int):
    random.seed(seed)


def softmax(x, temperature=1.0):
    exp = np.exp(temperature * np.array(x))
    return exp / exp.sum()


def softmax_from_slope(n: int, slope: float) -> np.ndarray:
    x = np.arange(n + 1)
    return softmax(x * slope)


def quadratic_arc(
    min_x: float, max_x: float, min_y: float = 0.0, max_y: float = 1.0
) -> t.Callable[[float], float]:
    """
    Returns a function that scales the quadratic from (0, 0) to (1, 1) to from
    (min_x,  min_y) to (max_x, max_y).

    >>> f = quadratic_arc(0, 2)
    >>> [f(x * 0.25) for x in range(9)]
    [0.0, 0.015625, 0.0625, 0.140625, 0.25, 0.390625, 0.5625, 0.765625, 1.0]

    If the argument to the returned function < min_x, it returns min_y.

    >>> [f(x * 0.25) for x in range(-4, 0)]
    [0.0, 0.0, 0.0, 0.0]
    """
    delta_x = max_x - min_x
    delta_y = max_y - min_y

    def f(x):
        if x < min_x:
            return min_y
        return ((x - min_x) / delta_x) ** 2 * delta_y + min_y

    return f


def linear_arc(
    min_x: float, max_x: float, min_y: float = 0.0, max_y: float = 1.0
) -> t.Callable[[float], float]:
    """
    Return the line from (min_x, min_y) to (max_x, max_y).

    >>> f = linear_arc(1, 5)
    >>> [f(x) for x in range(7)]
    [-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    """
    slope = (max_y - min_y) / (max_x - min_x)
    return lambda x: slope * (x - min_x)


def _swap(seq: t.List[t.Any], i: int, j: int) -> None:
    """Swaps items in-place."""
    temp = seq[i]
    seq[i] = seq[j]
    seq[j] = temp


def weighted_sample_wo_replacement(
    choices: t.Sequence[t.Any], weights: t.Sequence[float]
) -> t.Iterator[t.Any]:
    """
    There is probably a faster algorithm for this...

    >>> choices = ["a", "b", "c"]
    >>> weights = [0.6, 0.3, 0.1]
    >>> [x for x in weighted_sample_wo_replacement(choices, weights)]  # doctest: +SKIP
    ['a', 'b', 'c']
    >>> [x for x in weighted_sample_wo_replacement(choices, weights)]  # doctest: +SKIP
    ['b', 'a', 'c']
    """
    # we need to copy weights to avoid modifying them in-place.
    if not choices:
        return
    weights = list(weights)
    choice_indices = list(range(len(choices)))
    index_pointers = list(range(len(choices)))
    out = []
    for i in range(len(choices) - 1):
        choice_i = random.choices(index_pointers[i:], weights=weights[i:])[0]
        yield choices[choice_indices[choice_i]]
        out.append(choices[choice_indices[choice_i]])
        _swap(choice_indices, i, choice_i)
        _swap(weights, i, choice_i)
    yield choices[choice_indices[-1]]
