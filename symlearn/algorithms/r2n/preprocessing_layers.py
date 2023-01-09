import inspect
import itertools
import operator
from copy import copy
from functools import reduce

import numpy as np
import torch
from torch import nn

import symlearn.algorithms.r2n.mask as wm
from symlearn.algorithms.r2n.base import PredicateLearningLayer
from symlearn.trxf.core import Feature, Predicate, Relation


class NegationLayer(nn.Module):
    """
    Augment the input layer with its negation to explicitly provide negated conditions to the downstream rule net.
    """

    def __init__(self):
        super(NegationLayer, self).__init__()

    def forward(self, x):
        return torch.cat((x, torch.ones_like(x) - x), 1)


def _normalize_weights(bias, weights):
    max_coef = np.max(np.abs(weights), axis=1)
    normbias = np.divide(bias, max_coef)
    max_coef = np.tile(max_coef, (weights.shape[1], 1)).T
    normweights = np.divide(weights, max_coef)
    return normbias, normweights


class LinearFeatureLayer(PredicateLearningLayer):
    """
    This layer takes as input a set of input featues and creates a linear combinations thereof to constuct a predicate in the form \sum w x - b > 0
    where w and b are the weights and biases of the neural network layer.
    """

    def __init__(self, dim_in, dim_out, init_temp=1):
        super(LinearFeatureLayer, self).__init__()
        torch.manual_seed(0)
        self.layer = nn.Linear(dim_in, dim_out)
        self.mask = wm.DeterministicWeightMask(torch.ones(dim_out, dim_in))
        self.temp = init_temp
        self.dim_out = dim_out

    def forward(self, x):
        z = self.mask()
        regularized_weights = z * self.layer.weight
        self.layer.weight.data = regularized_weights
        return torch.sigmoid((1 / self.temp) * self.layer(x))

    def get_penalty(self):
        # return self.mask().mean()
        return self.layer.weight.norm(1)

    def update_temp(self, temp):
        self.temp = temp

    def get_temp(self):
        return self.temp

    def get_predicates(self, column_names, normalizer, thr, name_format='generic'):
        weight = copy(self.layer.weight.detach().numpy())
        bias = copy(self.layer.bias.detach().numpy())
        assert len(column_names) == weight.shape[1], 'column_names: {} but weight dim_in: {}'.format(column_names,
                                                                                                     weight.shape[1])
        norm_bias, norm_weights = _normalize_weights(bias, weight)
        weight[abs(norm_weights) < thr] = 0
        bias[abs(norm_bias) < thr] = 0
        if normalizer:
            weight = normalizer.transform(weight)

        features = []
        for w, b in zip(weight, bias):
            terms = [x for x in zip(w, column_names) if abs(x[0]) > 0]
            feature = ' + '.join(list(map(lambda term: str(term[0]) + '*' + str(term[1]), terms)))
            feature += ' + ' + str(b)
            features.append(feature.replace('+ -', '- '))

        predicates = list(map(lambda expression: Predicate(Feature(expression), Relation.GE, 0), features))
        return predicates


class BasisExpansionLayer(nn.Module):
    def __init__(self, basis_functions, application_list=None):
        super().__init__()
        self.functions = basis_functions + [lambda x: x]
        self.application_list = application_list

    def forward(self, x):
        result = []
        batch_size = x.size()[0]
        for i in range(batch_size):
            batch_result = []
            for f in self.functions:
                arity = len(inspect.getfullargspec(f).args)
                args = [x[i]] * arity
                cartesian = torch.cartesian_prod(*args).detach().numpy()
                if arity == 1:
                    cartesian = np.expand_dims(cartesian, axis=1)
                batch_result += [f(*args) for args in cartesian]
            result.append(batch_result)

        return torch.FloatTensor(result)


class ExpandedLinearFeatureLayer(PredicateLearningLayer):
    def __init__(self, basis_functions, dim_in, dim_out):
        super().__init__()
        torch.manual_seed(0)

        self.functions = basis_functions
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.expansion_layer = BasisExpansionLayer(self.functions)
        arities = [len(inspect.getfullargspec(f).args) for f in self.functions] + [1]
        dim_mid = sum([dim_in ** n for n in arities])
        self.linear_feature_layer = LinearFeatureLayer(dim_mid, self.dim_out)

    def forward(self, x):
        return nn.Sequential(self.expansion_layer, self.linear_feature_layer)(x)

    def get_predicates(self, column_names, normalizer, thr, name_format='generic'):
        norm_obj = copy(normalizer)
        if normalizer is not None:
            scale = normalizer.scale_
            expanded_scale = _expand_normalizer(scale, self.functions)
            norm_obj.scale_ = expanded_scale
            norm_obj.n_features_in_ = len(expanded_scale)
        expanded_col_names = _expand_columns(column_names, self.functions, name_format)
        return self.linear_feature_layer.get_predicates(expanded_col_names, norm_obj, thr, name_format)

    def get_penalty(self):
        return self.linear_feature_layer.get_penalty()

    def update_temp(self, temp):
        self.linear_feature_layer.update_temp(temp)


def _expand_columns(column_names, functions, name_format):
    names = []
    for i, f in enumerate(functions):
        arity = len(inspect.getfullargspec(f).args)
        repeat = [column_names] * arity
        cols = itertools.product(*repeat)
        names += ['f{}_'.format(i) + '_'.join(tup) for tup in cols] if name_format == 'generic' \
            else ['*'.join(tup) for tup in cols]

    return names + column_names


def _expand_normalizer(scale, functions):
    expanded_scale = []
    for i, f in enumerate(functions):
        arity = len(inspect.getfullargspec(f).args)
        repeat = [scale] * arity
        scales = itertools.product(*repeat)
        expanded_scale += [_prod(tup) for tup in scales]

    return np.concatenate((np.array(expanded_scale), scale))


def _prod(iterable):
    return reduce(operator.mul, iterable, 1)


class PredicateLayer(nn.Module):
    """
    This predicate layer combines categorical and numerical input features and, depending of the NegLayer flag, negates the categorical features.
    """

    def __init__(self, learning_layer, negation):
        super(PredicateLayer, self).__init__()
        self.LearningLayer = learning_layer
        self.negation = negation

    def forward(self, x):
        x_cat_in = x[0]
        x_cat_out = x_cat_in
        x_num_in = x[1]
        x_num_out = x_num_in
        if x_num_in.nelement() != 0:
            x_num_out = self.LearningLayer(x_num_in)
        if x_cat_in.nelement() != 0 and self.negation:
            neglayer = NegationLayer()
            x_cat_out = neglayer(x_cat_out)
        result = torch.cat((x_num_out, x_cat_out), -1)
        return result
