from unittest import TestCase

import torch

import symlearn.algorithms.r2n.preprocessing_layers as preproc


class TestPreprocessingLayer(TestCase):

    def test_negation_layer(self):
        layer = preproc.NegationLayer()
        xs = torch.tensor([[1, 0, 1], [0, 0, 1]])
        actual = layer(xs)
        expected = torch.tensor([[1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 1, 0]])
        self.assertTrue(torch.allclose(expected, actual))

    def test_linear_feature_layer_get_predicates(self):
        layer = preproc.LinearFeatureLayer(3, 2)
        expected = ['-0.004322529*x1 + 0.30971587*x2 - 0.47518533*x3 - 0.011439145 >= 0',
                    '-0.42489457*x1 - 0.22236899*x2 + 0.15482074*x3 + 0.457775 >= 0']
        for i, pred in enumerate(layer.get_predicates(['x1', 'x2', 'x3'], normalizer=None, thr=0)):
            self.assertEqual(expected[i], str(pred))

    def test_basis_expansion_layer(self):
        xs = torch.tensor([[1, 0, 1], [0, 0, 1]])
        functions = [lambda x, y: x + y,
                     lambda x: -x]
        layer = preproc.BasisExpansionLayer(functions)
        actual = layer(xs)
        expected = torch.tensor([[2., 1., 2., 1., 0., 1., 2., 1., 2., -1., 0., -1., 1., 0., 1.],
                                 [0., 0., 1., 0., 0., 1., 1., 1., 2., 0., 0., -1., 0., 0., 1.]])
        torch.testing.assert_close(actual, expected)

    def test_expanded_linear_feature_layer(self):
        xs = torch.tensor([[1, 0, 1], [0, 0, 1]])
        functions = [lambda x, y: x + y,
                     lambda x: -x]
        layer = preproc.ExpandedLinearFeatureLayer(functions, 3, 2)
        actual = layer(xs).data
        expected = torch.tensor([[0.3243585527, 0.6980250478],
                                 [0.4636095166, 0.6726788282]])
        torch.testing.assert_close(actual, expected)

        expected = ['-0.0019330978*f0_x1_x1 + 0.13850912*f0_x1_x2 - 0.21250933*f0_x1_x3 - 0.19001864*f0_x2_x1 - '
                    '0.09944643*f0_x2_x2 + 0.06923795*f0_x2_x3 - 0.0051157475*f0_x3_x1 + 0.20472318*f0_x3_x2 - '
                    '0.02291362*f0_x3_x3 + 0.06832266*f1_x1 - 0.07803108*f1_x2 - 0.050752968*f1_x3 - 0.24666992*x1 - '
                    '0.17100051*x2 - 0.10643555*x3 - 0.100645915 >= 0',
                    '0.009564608*f0_x1_x1 + 0.1020751*f0_x1_x2 + 0.15492523*f0_x1_x3 - 0.17504364*f0_x2_x1 - '
                    '0.11243601*f0_x2_x2 + 0.093782246*f0_x2_x3 + 0.21440524*f0_x3_x1 - 0.053137377*f0_x3_x2 + '
                    '0.19321325*f0_x3_x3 - 0.041617364*f1_x1 + 0.02732107*f1_x2 + 0.23379296*f1_x3 - 0.23952346*x1 - '
                    '0.162546*x2 - 0.06536698*x3 + 0.22308403 >= 0'
                    ]
        for i, p in enumerate(layer.get_predicates(['x1', 'x2', 'x3'], None, thr=0)):
            self.assertEqual(expected[i], str(p))


