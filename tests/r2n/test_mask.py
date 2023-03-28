from unittest import TestCase

import torch

from rulelearn.algorithms.r2n.mask import SimpleWeightMask


class TestMask(TestCase):
    def test_simple_weight_mask_shape(self):
        mask = SimpleWeightMask(3, 2, temp = 1, cooling= False)
        actual = mask.forward()
        expected_shape = torch.Size([3, 2])
        self.assertEqual(actual.shape, expected_shape)
