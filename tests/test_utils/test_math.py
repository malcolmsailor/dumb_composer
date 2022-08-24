from collections import Counter
import random
import typing as t
from dataclasses import dataclass

import numpy as np


from dumb_composer.utils.math_ import weighted_sample_wo_replacement


def test_weighted_sample_wo_replacement():
    @dataclass
    class TestCase:
        choices: list[t.Any]
        weights: list[float]
        n_trials: int = 1000

    testcases = [
        TestCase(["a", "b", "c"], [0.6, 0.4, 0.2]),
        TestCase(["a"], [1.0]),
        TestCase([], []),
        TestCase(["a", "b", "c"], [0.9, 0.08, 0.02]),
    ]

    for testcase in testcases:
        random.seed(42)
        results = [
            list(
                weighted_sample_wo_replacement(
                    testcase.choices, testcase.weights
                )
            )
            for _ in range(testcase.n_trials)
        ]
        sorted_choices = sorted(testcase.choices)
        indices = Counter()
        for result in results:
            assert sorted(result) == sorted_choices
            for i, item in enumerate(result):
                indices[item] += i
        # we take neg_sums so sort will be reversed (np.argsort doesn't have a
        #   "reverse" or similar flag)
        neg_sums = [-indices[choice] for choice in testcase.choices]
        assert (np.argsort(neg_sums) == np.argsort(testcase.weights)).all()
