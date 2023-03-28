import os
from unittest import TestCase

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from rulelearn.algorithms.r2n import r2n_algo, utilities
from rulelearn.algorithms.r2n.preprocessing_layers import LinearFeatureLayer
from rulelearn.algorithms.r2n.r2n_algo import _binarize_input
from rulelearn.trxf.core.utils import batch_evaluate


class TestR2Nalgo(TestCase):
    def test_data_preprocessing(self):
        input_data, label, cat_col_names, num_col_names = generate_data()
        R2run = r2n_algo.R2Nalgo()
        R2run.cat_column_names = cat_col_names 
        R2run.num_column_names = num_col_names
        sol = R2run._data_preprocessing(input_data, label, True)
        # Below we check that the number of numerical and categorical features is correctly calculated.
        self.assertEqual(sol[1].shape[1], 4)
        self.assertEqual(sol[2].shape[1], 3)

    def test_preprocessing_normalization(self):
        input_data, label, cat_col_names, num_col_names = generate_data()
        R2run = r2n_algo.R2Nalgo()
        R2run.cat_column_names = cat_col_names 
        R2run.num_column_names = num_col_names
        y, xc_train, xn_train  = R2run._data_preprocessing(input_data, label, True)
        _, xc_test, xn_test  = R2run._data_preprocessing(input_data, label, False)
        np.testing.assert_array_equal(xn_test, xn_train)

    def test_preprocessing_only_categorical(self):
        X_cat = np.random.randint(2, size=(100, 2))
        rule = lambda z: 1.0 if z[1] != 1 else 0.0
        y = np.array([rule(z) for z in X_cat])

        data = pd.DataFrame(np.concatenate((X_cat, y.reshape(-1, 1)), axis=1))
        data.columns = [['z0', 'z1', 'y']]
        label = data[['y']]
        input_data = data.drop(columns=['y'], axis=1)
        input_data.loc[:, ['z0', 'z1']] = input_data.loc[:, ['z0', 'z1']].astype(int)
        R2run = r2n_algo.R2Nalgo()
        R2run.cat_column_names = ['z0','z1'] 
        R2run.num_column_names = []
        sol =  R2run._data_preprocessing(input_data, label, True)
        # Below we check that the number of categorical features is correctly calculated.
        self.assertEqual(sol[1].shape[1], 4)

    def test_predicate_evaluation(self):
        X_tot = np.array([[0.5, 0.1, 0.2, 0.5, 0.1]])

        weights = np.array([
            [1, -2.4, 1.0, -6.9, 1],
            [2, 8.3, -3.2, 6.5, 1]
        ]).astype('float32')
        biases = np.array([1.0, -1.0]).astype('float32')
        mock_layer = LinearFeatureLayer(5, 2)
        mock_layer.layer.weight = torch.nn.Parameter(torch.tensor(weights))
        mock_layer.layer.bias = torch.nn.Parameter(torch.tensor(biases))

        expected = np.array([[False], [True]])
        actual = _binarize_input(X_tot, predicate_learning_layer=mock_layer)
        np.testing.assert_array_equal(actual, expected)

    def test_e2e_num_cat(self):
        torch.manual_seed(2)
        np.random.seed(2)
        # Control variables
        data_loc = '../../examples/r2n/data/toy_example.csv'

        # Prepare the data
        data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), data_loc))
        input_data = data.drop(columns=['y', 'Unnamed: 0'])
        input_data[['Z_0', 'Z_1']] = input_data[['Z_0', 'Z_1']].replace(0, 'No')
        input_data[['Z_0', 'Z_1']] = input_data[['Z_0', 'Z_1']].replace(1, 'Yes')
        label = data[['y']]
        cat_col_names = ['Z_0','Z_1']
        num_col_names = ['X_0','X_1','X_2','X_3','X_4']
        
        input_train, input_test, label_train, label_test = train_test_split(input_data, label, train_size=0.75)
        R2run = r2n_algo.R2Nalgo(cat_columns=cat_col_names, num_columns=num_col_names, n_seeds=1, decay_rate=0.9, coef=10 ** -3, normalize_num=False)
        R2run.fit(input_train, label_train)
        y_pred_internal = R2run.predict(input_test)
        acc_internal = utilities.compute_metrics(label_test, y_pred_internal)['acc']
        actual = R2run.export_rules_to_trxf_dnf_ruleset(thr=0.0)
        y_pred_trxf = batch_evaluate(actual, input_test)
        acc_trxf = utilities.compute_metrics(label_test, y_pred_trxf)['acc']
        print(str(R2run.export_rules_to_trxf_dnf_ruleset(thr=0.0)))
        self.assertAlmostEqual(acc_trxf, acc_internal, places=2)
        self.assertGreater(acc_internal, 0.99)

    def test_e2e_num(self):
        torch.manual_seed(2)
        np.random.seed(2)
        input_test, input_train, label_test, label_train = prepare_num_data()
        num_col_names = ['X_0','X_1','X_2','X_3','X_4']
        R2run = r2n_algo.R2Nalgo(num_columns=num_col_names, n_seeds=1, decay_rate=0.9, coef=10 ** -3, normalize_num=True)
        R2run.fit(input_train, label_train)
        y_pred_internal = R2run.predict(input_test)
        actual = R2run.export_rules_to_trxf_dnf_ruleset(thr=0.1)
        y_pred_trxf = batch_evaluate(actual, input_test)
        print(actual)
        self.assertGreater(utilities.compute_metrics(label_test, y_pred_trxf)['acc'], 0.95)

    def test_e2e_num_basis_expansion(self):
        torch.manual_seed(2)
        np.random.seed(2)
        input_test, input_train, label_test, label_train = prepare_num_data()
        functions = [lambda x, y: x * y]
        R2run = r2n_algo.R2Nalgo(
            n_seeds=1, decay_rate=0.7, coef=10 ** -3, normalize_num=False, basis_functions=functions, name_format='quad')
        R2run.fit(input_train, label_train)
        y_pred_internal = R2run.predict(input_test)
        actual = R2run.export_rules_to_trxf_dnf_ruleset(thr=0.1)
        y_pred_trxf = batch_evaluate(actual, input_test)
        print(actual)
        print('trxf acc:', utilities.compute_metrics(label_test, y_pred_trxf))
        self.assertGreater(utilities.compute_metrics(label_test, y_pred_trxf)['acc'], 0.95)

    def test_e2e_cat(self):
        torch.manual_seed(2)
        np.random.seed(2)
        # Control variables
        data_loc = '../../examples/r2n/data/toy_example.csv'

        # Prepare the data
        data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), data_loc))
        input_data = data.drop(columns=['y', 'Unnamed: 0'])
        input_data[['Z_0', 'Z_1']] = input_data[['Z_0', 'Z_1']].replace(0, 'No')
        input_data[['Z_0', 'Z_1']] = input_data[['Z_0', 'Z_1']].replace(1, 'Yes')
        label = data[['y']]
        input_data = input_data[['Z_0', 'Z_1']]
        input_train, input_test, label_train, label_test = train_test_split(input_data, label, train_size=0.75)
        R2run = r2n_algo.R2Nalgo(n_seeds=1, decay_rate=0.5, coef=10 ** -3, normalize_num=False, negation=True)
        R2run.fit(input_train, label_train)
        y_pred_internal = R2run.predict(input_test)
        acc_internal = utilities.compute_metrics(label_test, y_pred_internal)['acc']
        actual = R2run.export_rules_to_trxf_dnf_ruleset(thr=0.0)
        y_pred_trxf = batch_evaluate(actual, input_test)
        acc_trxf = utilities.compute_metrics(label_test, y_pred_trxf)['acc']
        print(str(R2run.export_rules_to_trxf_dnf_ruleset(thr=0.1)))
        # self.assertAlmostEqual(acc_trxf, acc_internal, places=2)
        self.assertGreater(acc_trxf, 0.89)

    def test_e2e_churn(self):
        torch.manual_seed(2)
        np.random.seed(2)
        data_loc = '../../examples/r2n/data/churn_prob_out_35.csv'
        label_col = 'CHURN'
        to_drop = ['Id', 'CHURN', '3_Class', '5_Class', 'is_test_set', 'pChurn']
        data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), data_loc))
        data.columns = data.columns.str.replace(' ', '')
        input_data = data.drop(columns=to_drop)
        label = data[[label_col]]
        label = label.replace('F', 0)
        label = label.replace('T', 1)
        input_train, input_test, label_train, label_test = train_test_split(input_data, label, train_size=0.75)
        R2run = r2n_algo.R2Nalgo(n_seeds=1, min_temp=10 ** -4, decay_rate=0.98, coef=5 * 10 ** -4, normalize_num=True)
        R2run.fit(input_train, label_train)
        actual = R2run.export_rules_to_trxf_dnf_ruleset()
        y_pred = batch_evaluate(actual, input_test)
        y_pred_internal = R2run.predict(input_test)
        print(utilities.compute_metrics(label_test, y_pred)['acc'])
        self.assertGreater(utilities.compute_metrics(label_test, y_pred)['acc'], 0.78)


def generate_data():
    X = np.random.rand(100, 3)
    X_cat = np.random.randint(2, size=(X.shape[0], 2))
    rule = lambda x, z: 1.0 if (x[1] > 0.5) or (x[0] > 0.5 and z[1] != 1) else 0.0
    y = np.array([rule(x, z) for x, z in zip(X, X_cat)])
    data = pd.DataFrame(np.concatenate((X, X_cat, y.reshape(-1, 1)), axis=1))
    data.columns = [['x0', 'x1', 'x2', 'z0', 'z1', 'y']]
    label = data[['y']]
    input_data = data.drop(columns=['y'], axis=1)
    input_data.loc[:, ['z0', 'z1']] = input_data.loc[:, ['z0', 'z1']].astype(int)
    cat_col_names = ['z0','z1']
    num_col_names = ['x0','x1','x2']
    return input_data, label, cat_col_names, num_col_names


def prepare_num_data():
    # Control variables
    data_loc = '../../examples/r2n/data/toy_example.csv'
    label_col = 'y'
    # Prepare the data
    data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), data_loc))
    data = data.select_dtypes(include=['float'])
    data[label_col] = data[label_col]
    input_data = data.drop(columns=['y'])
    label = data[['y']]
    input_train, input_test, label_train, label_test = train_test_split(input_data, label, train_size=0.75)
    return input_test, input_train, label_test, label_train



