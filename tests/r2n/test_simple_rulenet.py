import unittest.mock
from unittest import TestCase

import numpy as np
import torch

import rulelearn.algorithms.r2n.mask as wm
import rulelearn.algorithms.r2n.simple_rulenet as rn

EPSILON = 0.5


class TestSimpleRuleNet(TestCase):
    def test_conjunction_layer_true(self):
        mock_mask = wm.SimpleWeightMask(3, 2, temp=1, cooling=True)
        mock_mask.forward = unittest.mock.MagicMock(return_value=torch.tensor([
            [1, 1, 0],
            [0, 0, 1]
        ]))
        and_layer = rn.ConjunctionLayer(mock_mask)
        x = torch.tensor([1., 1., 1.])
        actual = and_layer(x)
        expected = torch.tensor([1., 1.])
        self.assertTrue(torch.allclose(actual, expected))

    def test_conjunction_layer_false(self):
        mock_mask = wm.SimpleWeightMask(3, 2, temp=1, cooling=True)
        mock_mask.forward = unittest.mock.MagicMock(return_value=torch.tensor([
            [1, 1, 0],
            [0, 0, 1]
        ]))
        and_layer = rn.ConjunctionLayer(mock_mask)
        x = torch.tensor([0., 1., 1.])
        actual = and_layer(x)
        expected = torch.tensor([0., 1.])
        self.assertTrue(torch.allclose(actual, expected))

    def test_disjunction_layer_false(self):
        x = torch.tensor([[0., 1., 0.]])
        actual = _evaluate_disjunction(x)
        expected = torch.zeros_like(actual)
        np.testing.assert_equal(expected.numpy(), actual.data.numpy())

    def test_disjunction_layer_true(self):
        x = torch.tensor([[1., 0., 0.]])
        actual = _evaluate_disjunction(x)
        expected = torch.ones_like(actual)
        self.assertEqual(expected.numpy(), actual.data.numpy())


def _evaluate_disjunction(x):
    mock_mask = wm.SimpleWeightMask(1, 3, temp=1, cooling=True)
    mock_mask.forward = unittest.mock.MagicMock(return_value=torch.tensor([[1., 0., 1.]]))
    or_layer = rn.DisjunctionLayer(mock_mask)
    return or_layer(x)
