import itertools
import random
from typing import Any, Iterable, Sequence


def _flatten_iterables_sub(x: Any | Iterable[Any]) -> list[Any]:
    if isinstance(x, Iterable):
        out = []
        for y in x:
            out += _flatten_iterables_sub(y)
        return out
    else:
        return [x]


def flatten_iterables(l: Iterable[Any]) -> list[Any]:
    """
    >>> flatten_iterables([1, 2, 3])
    [1, 2, 3]
    >>> flatten_iterables(())
    []
    >>> flatten_iterables([1, (2, 3), [[4], [], [(i for i in range(5, 7))]]])
    [1, 2, 3, 4, 5, 6]
    """
    return _flatten_iterables_sub(l)


def shuffled_cartesian_product(*args, repeat=1) -> Iterable[Any]:
    """
    # TODO: (Malcolm 2023-07-12) it would be nice/fun to make a lazy version that
    doesn't store the entire product in memory in order to shuffle it.

    >>> list(shuffled_cartesian_product("ABCD", "xy"))  # doctest: +SKIP
    [('C', 'y'), ('B', 'x'), ('B', 'y'), ('A', 'y'), ('A', 'x'), ('C', 'x'),
    ('D', 'y'), ('D', 'x')]

    """
    out = list(itertools.product(*args, repeat=repeat))
    random.shuffle(out)
    yield from out


def yield_from_sequence_of_iters(
    iter_seq: Sequence[Iterable[Any]], shuffle: bool = False
):
    """
    >>> iter1 = range(3)
    >>> iter2 = range(3, 7)
    >>> list(yield_from_sequence_of_iters([iter1, iter2]))
    [0, 3, 1, 4, 2, 5, 6]

    >>> iter3 = []
    >>> list(yield_from_sequence_of_iters([iter2, iter1, iter3]))
    [3, 0, 4, 1, 5, 2, 6]

    >>> list(
    ...     yield_from_sequence_of_iters([iter1, iter2], shuffle=True)
    ... )  # doctest: +SKIP
    [3, 4, 0, 1, 2, 5, 6]
    """
    iter_list = list(iter(i) for i in iter_seq)

    if shuffle:
        while iter_list:
            i = random.randrange(len(iter_list))
            try:
                yield next(iter_list[i])
            except StopIteration:
                iter_list.pop(i)
    else:
        while iter_list:
            to_pop = []
            for i, iter_ in enumerate(iter_list):
                try:
                    yield next(iter_)
                except StopIteration:
                    to_pop.append(i)
            for i in reversed(to_pop):
                iter_list.pop(i)
