from unittest import TestCase

import numpy as np

import rulelearn.algorithms.r2n.utilities as util


class TestUtilities(TestCase):

    def test_simplify_ruleset(self):
        ruleset = [np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                   np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                   np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
                   np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
                   np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])]

        expected = [np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                    np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])]

        actual = util.simplify_ruleset(ruleset)
        np.testing.assert_array_equal(actual, expected)

    def test_simplify_ruleset1(self):
        ruleset = [np.array([0, 1, 1, 1]),
                   np.array([0, 1, 0, 0]),
                   np.array([1, 1, 0, 1]),
                   np.array([1, 1, 0, 0]),
                   np.array([1, 0, 1, 0]),
                   np.array([0, 1, 1, 0]),
                   np.array([1, 1, 0, 0]),
                   np.array([0, 1, 0, 0]),
                   np.array([1, 0, 1, 0])]

        expected = [np.array([0.0, 1.0, 0.0, 0.0]),
                    np.array([1.0, 0.0, 1.0, 0.0])]

        actual = util.simplify_ruleset(ruleset)
        np.testing.assert_array_equal(actual, expected)

    def test_filter_trivial_predicates(self):
        ruleset = np.array(
            [[1, 1, 0, 0, 0],
             [0, 0, 0, 1, 1]])
        evaluated_predicates = np.array([[1, 1, 0, 1, 0],
                                         [1, 0, 1, 1, 0],
                                         [1, 0, 0, 1, 0]]).T
        expected_ruleset = np.array([[0, 1, 0, 0, 0]])
        actual = util.filter_trivial_predicates(ruleset, evaluated_predicates)
        np.testing.assert_array_equal(actual, expected_ruleset)

