import math

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset

import rule_induction.r2n.preprocessing_layers as pp
import rule_induction.r2n.simple_rulenet as rn
import rule_induction.r2n.utilities as util
from rule_induction.r2n.training import train as train_r2n
from rule_induction.trxf.core.conjunction import Conjunction
from rule_induction.trxf.core.dnf_ruleset import DnfRuleSet
from rule_induction.trxf.core.feature import Feature
from rule_induction.trxf.core.predicate import Predicate, Relation


def _normalize_weights(bias, weights):
    max_coef = np.max(np.abs(weights), axis=1)
    normbias = np.divide(bias, max_coef)
    max_coef = np.tile(max_coef, (weights.shape[1], 1)).T
    normweights = np.divide(weights, max_coef)
    return normbias, normweights


clsdfass R2Nalgo:
    """
    Wrapper for the R2N algorithm which is described in the following paper: arxiv.org/abs/2201.06515. 
    """

    def __init__(self,
                 batch_size: int = 100,
                 learning_rate: float = 1e-3,
                 coef: float = 1e-3,
                 n_rules: int = 25,
                 n_conditions: int = 10,
                 n_seeds: int = 2,
                 init_temp: float = 1,
                 min_temp: float = 10 ** -4,
                 decay_rate: float = 0.999,
                 negation: bool = False,
                 normalize_num: bool = False
                 ):
        """
        batch_size : int
            The batch size used during the training of the NN
        learning rate : float
            Learning rate of the neural network
        coef: float
            Sparsity promoting coefficient 
        n_rules: int
            Numbers of rules in the and layer
        n_conditions: int
            Numbers of output nodes of the multivariate predicate layer 
        n_seeds: int
            Number of times the optimizer and network weights are reinitialized (number of runs)
        max_epochs: int
            Number of epochs ran per seed 
        init_temp: float
            Initial temperature of the cooling schedule
        min_temp: float
            Minimal temperature of the cooling schedule
        decay_rate: float
            Decay rate of the cooling schedule
        negation: bool
            Whether negations of categorical features are considered or not
         """
        super(R2Nalgo, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.coef = coef
        self.n_rules = n_rules
        self.n_conditions = n_conditions
        self.n_seeds = n_seeds
        self.decay_rate = decay_rate
        self.init_temp = init_temp
        self.min_temp = min_temp
        self.max_epochs = math.ceil(
            1.2 * (math.log(self.min_temp) - math.log(self.init_temp)) / math.log(self.decay_rate))
        self.negation = negation
        self.normalize_num = normalize_num

        self.encoder_object = None
        self.norm_object = None
        self.n_cat_features, self.n_num_features = None, None
        self.num_column_names, self.cat_column_names = None, None
        self.model = None
        self.opt_rule = None
        self.predicate_learning_layer = None

    def fit(self, train, train_labels):
        """
        The fit function for R2N algorithm. All float and int dtypes are considered numerical features. 
        All others are treated as categorical.
        """
        X_cat, X_num, train_dataloader = self._preprocess(train, train_labels)

        # Training loop 
        loss_fn = nn.MSELoss()
        best_loss = math.inf
        best_acc = -1
        for _ in np.arange(0, self.n_seeds):

            self.model, mlpnet, rulenet = self._construct_r2n()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, amsgrad=True)
            temp = self.init_temp
            for t in range(self.max_epochs):
                temp, loss, success_rate_train = train_r2n(train_dataloader, self.model, loss_fn, optimizer, temp,
                                                           self.min_temp, self.decay_rate, self.coef, mv_reg=self.coef,
                                                           multivariate=(self.n_num_features != 0))
                if t % 10 == 0:
                    print("Epoch:", t, "Success rate:", '%.1f' % success_rate_train, "Loss:", '%.5f' % loss,
                          "Temperature:", '%.1e' % temp)
                if temp < self.min_temp and loss < best_loss:
                    if self.n_num_features > 0:
                        self.predicate_learning_layer = mlpnet
                    self.opt_rule = rulenet.extract_ruleset()
                    best_loss = loss
                    best_acc = success_rate_train
            print('---------')

        print("Max performance of the network:", best_acc)

        # Post-processing
        binarized_input = _binarize_input(X_num, X_cat, self.predicate_learning_layer, self.negation)
        self.opt_rule = util.filter_trivial_predicates(util.simplify_ruleset(self.opt_rule), binarized_input)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        :param features: A pandas dataframe containing feature values
        :return predictions: A pandas dataframe containing a single column called '__predictions__'
        """
        _, X_cat_t, X_num_t, _, _ = _data_preprocessing(features, None,
                                                        self.normalize_num,
                                                        self.encoder_object,
                                                        self.norm_object,
                                                        training=False)
        pred_evaluation = _binarize_input(X_num_t, X_cat_t, self.predicate_learning_layer, self.negation)
        eval_sol = util.batch_apply_ruleset(self.opt_rule, pred_evaluation.T)
        predictions = pd.DataFrame(data=eval_sol, columns=['__predictions__'])

        return predictions

    def export_rules_to_trxf_dnf_ruleset(self, thr=0):
        learned_predicates = []
        predetermined_predicates = []
        if self.n_num_features > 0:
            learned_predicates = self.predicate_learning_layer.get_predicates(self.num_column_names, self.norm_object, thr)
        if self.n_cat_features > 0:
            a = list(zip(self.cat_column_names, self.encoder_object.categories_))
            for item in a:
                col_name = item[0]
                categories = item[1]
                predetermined_predicates += [Predicate(Feature(col_name), Relation.EQ, v) for v in categories]
        predicates = learned_predicates + predetermined_predicates

        return DnfRuleSet([Conjunction([pred for pred, w in zip(predicates, rule) if w > 0]) for rule in self.opt_rule],
                          then_part='Churn')

    def _preprocess(self, train, train_labels):
        self.num_column_names = list(train.select_dtypes(include=['float']).columns.str.replace(' ', ''))
        self.cat_column_names = list(train.select_dtypes(exclude=['float']).columns.str.replace(' ', ''))
        y, X_cat, X_num, self.encoder_object, self.norm_object = _data_preprocessing(
            train, train_labels, self.normalize_num, None, None, training=True)
        self.n_num_features = X_num.shape[1] if X_num is not None else 0
        self.n_cat_features = X_cat.shape[1] if X_cat is not None else 0
        dataset = _NumCatDataset(y, X_num, X_cat)
        train_dataloader = DataLoader(dataset, self.batch_size)
        return X_cat, X_num, train_dataloader

    def _construct_r2n(self):
        pred_learning = None
        dim_num = 0
        if self.n_num_features > 0:
            functions = [lambda x, y: x * y]
            pred_learning = pp.ExpandedLinearFeatureLayer(functions, self.n_num_features, self.n_conditions)
            # pred_learning = pp.LinearFeatureLayer(self.n_num_features, self.n_conditions)
            dim_num = pred_learning.dim_out
        pred_layer = pp.PredicateLayer(pred_learning, self.negation)
        rulenet = rn.SimpleRuleNet(dim_num + self.n_cat_features, self.n_rules)
        model = nn.Sequential(pred_layer, rulenet)
        return model, pred_learning, rulenet


# Preprocessing function R2N ####

def _categorical_encoding(X_cat):
    """
    Encodes all categorical variables with OneHot encoding
    """
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit_transform(X_cat)
    X = ohe.transform(X_cat).toarray()
    return ohe, X


def _data_preprocessing(train, train_labels, normalize, enc_obj, norm_obj, training):
    """
    Imports a pandas data frame and selects the numerical ('float') and categorical (NOT 'float') input features and
    returns a pytorch dataframe and the number of categorical and numerical features.
    """
    X_cat, X_num = None, None
    table_X_num, table_X_cat = train.select_dtypes(include=['float']), train.select_dtypes(exclude=['float'])
    cat_features, num_features = table_X_cat.shape[1], table_X_num.shape[1]

    if num_features > 0:
        if normalize:
            if training:
                norm_obj = StandardScaler(with_mean=False)
                table_X_num = norm_obj.fit_transform(np.array(table_X_num))
            else:
                assert norm_obj is not None
                table_X_num = norm_obj.transform(np.array(table_X_num))
        X_num = np.array(table_X_num, dtype=float)

    if cat_features > 0:
        if training:
            print(table_X_num.shape[1], "numerical features and ", table_X_cat.shape[1], "categorical features")
            print('---------')
            enc_obj, X_cat = _categorical_encoding(table_X_cat.astype(str))
            print(X_cat.shape[1], "categorical dummies")
            print('---------')
        else:
            assert enc_obj is not None
            X_cat = enc_obj.transform(table_X_cat.astype(str)).toarray()

    if training:
        y = np.array(train_labels, dtype=int)
    else:
        y = None

    return y, X_cat, X_num, enc_obj, norm_obj


class _NumCatDataset(Dataset):
    def __init__(self, y, X_num=None, X_cat=None):
        self.X_num = X_num
        self.X_cat = X_cat
        self.num_tensor = torch.tensor(X_num, dtype=torch.float64) if X_num is not None else None
        self.cat_tensor = torch.tensor(X_cat, dtype=torch.float64) if X_cat is not None else None
        self.y_train = torch.tensor(y)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        xcat = self.cat_tensor[idx] if self.X_cat is not None else torch.empty(0)
        xnum = self.num_tensor[idx] if self.X_num is not None else torch.empty(0)
        return (xcat, xnum), self.y_train[idx]


def _binarize_input(X_num=None, X_cat=None, predicate_learning_layer=None, negation=False):
    """
    Evaluates the predicates and concatenates if necessary the numerical and categorical ones.
    """
    pred_evaluation = None
    if X_num is not None:
        pred_evaluation = (predicate_learning_layer(torch.tensor(X_num).float()) > 0.5).T.detach().numpy()
    if X_cat is not None:
        if X_num is not None:
            pred_evaluation = np.concatenate((pred_evaluation, X_cat.T), axis=0)
        else:
            pred_evaluation = X_cat.T
        if negation is not False:
            pred_evaluation = np.concatenate((pred_evaluation, np.logical_not(X_cat.T)), axis=0)
    if pred_evaluation is None:
        raise ValueError('The input X_num and X_cat should not be both None')
    return pred_evaluation
