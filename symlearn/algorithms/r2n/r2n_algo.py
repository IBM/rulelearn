import math

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset

import symlearn.algorithms.r2n.preprocessing_layers as pp
import symlearn.algorithms.r2n.simple_rulenet as rn
import symlearn.algorithms.r2n.utilities as util
from symlearn.algorithms.r2n.training import train as train_r2n
from symlearn.trxf.core.conjunction import Conjunction
from symlearn.trxf.core.dnf_ruleset import DnfRuleSet
from symlearn.trxf.core.feature import Feature
from symlearn.trxf.core.predicate import Predicate, Relation


class R2Nalgo:
    """
    Wrapper for the R2N algorithm which is described in the following paper: arxiv.org/abs/2201.06515. 
    """

    def __init__(self,
                 cat_columns: list = None,
                 num_columns: list = None,
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
                 normalize_num: bool = False,
                 basis_functions: list = None,
                 name_format='generic'
                 ):
        """
        batch_size : int
            The batch size used during the training of the NN
        learning rate : float
            Learning rate of the neural network
        coef: float
            Sparsity promoting coefficient. Larger is sparser.
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
        normalize_num: bool
            Scale the input features before training. Only works for polynomial basis functions.
        basis_functions: list of lambdas
            List of basis expansion functions to be applied to the original input
        name_format: bool
            An option to specify a more informative feature names for some basis expansions (currently only quadratic).
            E.g., x1*x0 instead of f1_x1_x0 as the feature name.
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
        self.basis_functions = basis_functions
        self.name_format = name_format

        self.encoder_object = None
        self.norm_object = None
        self.n_cat_features, self.n_num_features = None, None
        self.num_column_names, self.cat_column_names =  num_columns, cat_columns
        self.model = None
        self.opt_rule = None
        self.predicate_learning_layer = None

    def fit(self, train, train_labels):
        """
        The fit function for R2N algorithm. All float and int dtypes are considered numerical features. 
        All others are treated as categorical.
        """

        # Partition the input features through their dtype if cat and num features are not provided. 
        train.columns = train.columns.str.replace(' ', '')
        if self.cat_column_names == None: 
            self.cat_column_names = list(train.select_dtypes(exclude=['float']).columns.str.replace(' ', ''))         
        if self.num_column_names == None: 
            self.num_column_names = list(train.select_dtypes(include=['float']).columns.str.replace(' ', ''))   
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
        self.opt_rule = util.simplify_ruleset(util.filter_trivial_predicates(self.opt_rule, binarized_input))

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        :param features: A pandas dataframe containing feature values
        :return predictions: A pandas dataframe containing a single column called '__predictions__'
        """
        features.columns = features.columns.str.replace(' ', '')
        _, X_cat_t, X_num_t = self._data_preprocessing(features, None, training=False)
        pred_evaluation = _binarize_input(X_num_t, X_cat_t, self.predicate_learning_layer, self.negation)
        eval_sol = util.batch_apply_ruleset(self.opt_rule, pred_evaluation.T)
        predictions = pd.DataFrame(data=eval_sol, columns=['__predictions__'])

        return predictions

    def export_rules_to_trxf_dnf_ruleset(self, thr=0):
        """
        Args:
            thr: Threshold to remove negligible coefficients. Any coefficient that has a smaller ratio to the dominating
            coefficient than thr will be eliminated in the extracted ruleset.


        Returns:
            trxf.DnfRuleSet
        """
        learned_predicates = []
        predetermined_predicates = []
        if self.n_num_features > 0:
            learned_predicates = self.predicate_learning_layer.get_predicates(
                self.num_column_names, self.norm_object, thr, self.name_format)
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
        y, X_cat, X_num = self._data_preprocessing(train, train_labels, training=True)
        self.n_num_features = X_num.shape[1] if X_num is not None else 0
        self.n_cat_features = X_cat.shape[1] if X_cat is not None else 0
        dataset = _NumCatDataset(y, X_num, X_cat)
        train_dataloader = DataLoader(dataset, self.batch_size)
        return X_cat, X_num, train_dataloader

    def _construct_r2n(self):
        pred_learning = None
        dim_num = 0
        if self.n_num_features > 0:
            if self.basis_functions is not None:
                pred_learning = pp.ExpandedLinearFeatureLayer(self.basis_functions, self.n_num_features, self.n_conditions)
            else:
                pred_learning = pp.LinearFeatureLayer(self.n_num_features, self.n_conditions)
            dim_num = pred_learning.dim_out
        pred_layer = pp.PredicateLayer(pred_learning, self.negation)
        dim_cat = 2 * self.n_cat_features if self.negation else self.n_cat_features
        rulenet = rn.SimpleRuleNet(dim_num + dim_cat, self.n_rules)
        model = nn.Sequential(pred_layer, rulenet)
        return model, pred_learning, rulenet
    
    def _data_preprocessing(self, train, train_labels, training):
        """
        Imports a pandas data frame and selects the numerical ('float') and categorical (NOT 'float') input features and
        returns a pytorch dataframe and the number of categorical and numerical features.
        """
        X_cat, X_num = None, None
        table_X_num, table_X_cat = train[self.num_column_names], train[self.cat_column_names] 

        if len(self.num_column_names)>0:
            if self.normalize_num:
                if training:
                    self.norm_object = StandardScaler(with_mean=False)
                    table_X_num = self.norm_object.fit_transform(np.array(table_X_num))
                else:
                    assert self.norm_object is not None
                    table_X_num = self.norm_object.transform(np.array(table_X_num))
            X_num = np.array(table_X_num, dtype=float)

        if len(self.cat_column_names)>0:
            if training:
                print(table_X_num.shape[1], "numerical features and ", table_X_cat.shape[1], "categorical features")
                print('---------')
                self.encoder_object, X_cat = _categorical_encoding(table_X_cat.astype(str))
                print(X_cat.shape[1], "categorical dummies")
                print('---------')
            else:
                assert self.encoder_object is not None
                X_cat = self.encoder_object.transform(table_X_cat.astype(str)).toarray()

        if training:
            y = np.array(train_labels, dtype=int)
        else:
            y = None

        return y, X_cat, X_num



# Preprocessing function R2N ####

def _categorical_encoding(X_cat):
    """
    Encodes all categorical variables with OneHot encoding
    """
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit_transform(X_cat)
    X = ohe.transform(X_cat).toarray()
    return ohe, X


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
