from functools import partial

import numpy as np
import sklearn.metrics as metrics
import torch

basic_metrics_binary = dict(
    acc=metrics.accuracy_score,
    bal_acc=metrics.balanced_accuracy_score,
    f1=lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average="weighted"),
    precision=lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, average="weighted"),
    recall=lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, average="weighted"),
)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def compute_metrics(y_true, y_pred):
    res = dict()
    for metric in basic_metrics_binary:
        res[metric] = basic_metrics_binary[metric](y_true, y_pred)
    return res


def apply_ruleset(ruleset, x):
    """ Apply a ruleset to an input vector

    :param ruleset: A 2D list conforming to the format of the output of `extract_ruleset()`
    :param x: A list-like object of 0/1 values representing a single input vector
    :return: The truth value of applying ruleset to x
    """
    transform = {1: lambda z: z == 1,
                 0: lambda z: True,
                 -1: lambda z: z == 0}
    or_result = False
    for rule in ruleset:
        and_result = True
        for i, t in enumerate([transform[r] for r in rule]):
            and_result &= t(x[i])
        or_result |= and_result
    return or_result


def batch_apply_ruleset(ruleset, xs):
    """ Batch-apply a ruleset to a list of input vectors

    :param ruleset: A 2D list conforming to the format of the output of `extract_ruleset()`
    :param xs: A (2D) list-like object of input vectors
    :return: A list of truth values corresponding to each input vector in xs
    """
    return list(map(partial(apply_ruleset, ruleset), xs))


def _rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s


def filter_trivial_predicates(set_of_rules, pred_evaluation):
    """
    This function filters the ruleset by evaluating the rules and predicates on a dataset X.
    It returns a ruleset that filters all the always true predicates and removes the rules that contain an always false predicate.

    @param set_of_rules: 2D binary list representing a ruleset.
    @param pred_evaluation: 2D binary `np.array` representing the evaluation of each predicate on the input data. Equal to the transpose of the binarized data.
    @return: 2D binary list representing a ruleset.
    """
    # Evaluate which of the predicates evaluate to Always true or always False

    always_true = np.count_nonzero(pred_evaluation, axis=1) / pred_evaluation.shape[1] == 1
    always_false = np.count_nonzero(pred_evaluation, axis=1) / pred_evaluation.shape[1] == 0
    # Remove the predicates that are always True
    set_of_rules = [np.array(i, dtype=int) for i in set_of_rules]
    for i in set_of_rules:
        i[always_true] = 0
        # Select and remove the rules have at least one always False predicate
    always_false_rules = [np.count_nonzero(np.logical_and(rule, always_false)) for rule in set_of_rules]
    set_of_rules = [i for i, j in zip(set_of_rules, always_false_rules) if j == 0]

    return set_of_rules


def simplify_ruleset(ruleset):
    ruleset = [rule for rule in ruleset if
               np.count_nonzero(rule) != 0]  # Remove the ALWAYS FALSE rules in the disjunction.
    simplified_rules = []
    for rule1 in ruleset:
        is_redundant = False
        for rule2 in ruleset:
            if not np.array_equal(rule2, rule1) and _subsumes(rule2, rule1):
                is_redundant = True
                break
        if not is_redundant:
            simplified_rules.append(rule1)
    simplified_rules = _deduplicate(simplified_rules)

    return simplified_rules


def _deduplicate(simplified_rules):
    tupled_lst = set(map(tuple, simplified_rules))
    simplified_rules = list(map(list, tupled_lst))
    simplified_rules = [np.array(i, dtype=int) for i in simplified_rules]
    return simplified_rules


def _subsumes(rule1, rule2):
    return np.array_equal(np.logical_and(rule1, rule2), rule1)
