import os
from itertools import combinations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from algorithms.Aggregations import A1_aggr, A2_aggr, A3_aggr, A5_aggr, A6_aggr, A7_aggr, A8_aggr, \
    A9_aggr, A10_aggr
from algorithms.MinMaxArrays import find_min_max_pairs
from algorithms.Orders import xu_yager_order, lex1_order, lex2_order, get_decisions


class PomyslAlgorithm:
    def __init__(self, aggregation_type="A1", random_state=42,
                 order="xuyager", dataset="", iteration=""):
        self.order = order
        self.aggregation_type = aggregation_type
        self.random_state = random_state
        self.subtable_col_indexes = None
        self.fit_models = None
        self.y_labels = None
        self.dataset = dataset
        self.iteration = iteration

    def get_params(self):
        return {
            'aggregation_type': self.aggregation_type,
            'random_state': self.random_state,
            'order': self.order,
            'dataset': self.dataset,
            'iteration': self.iteration
        }

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

    def __repr__(self):
        return f"aggregation_type='{self.aggregation_type}', random_state={self.random_state}, " \
               f"order='{self.order}'"

    def fit(self, X_train, y_train):
        self.y_labels = np.unique(y_train)
        row_fit_models = []
        models = [
            RandomForestClassifier(n_jobs=-1, random_state=self.random_state),
            SVC(kernel="linear", probability=True, random_state=self.random_state),
            DecisionTreeClassifier(random_state=self.random_state),
            MLPClassifier(max_iter=1000, random_state=self.random_state),
            KNeighborsClassifier(n_neighbors=1, p=1),
            KNeighborsClassifier(n_neighbors=3, p=1),
            KNeighborsClassifier(n_neighbors=5, p=1)
        ]
        for model in models:
            model.fit(X_train, y_train)
            row_fit_models.append(model)
        self.fit_models = row_fit_models

    def predict(self, X_test):
        aggregation_functions = {
            'A1': A1_aggr, 'A2': A2_aggr, 'A3': A3_aggr, 'A4': A3_aggr, 'A5': A5_aggr, 'A6': A6_aggr, 'A7': A7_aggr,
            'A8': A8_aggr, 'A9': A9_aggr, 'A10': A10_aggr}
        orders = {'xuyager': xu_yager_order, 'lex1': lex1_order, 'lex2': lex2_order}
        subtables = []
        row_predicions = []
        for model in self.fit_models:
            row_predicions.append(model.predict_proba(X_test))
        subtables.append(row_predicions)
        min_max_array = [find_min_max_pairs(subtable) for subtable in subtables]
        aggregated = aggregation_functions.get(self.aggregation_type)(min_max_array)
        ordered = orders.get(self.order)(aggregated)
        decisions = get_decisions(ordered)
        multiclass_labels = [self.y_labels[d] for d in decisions]
        return multiclass_labels


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    alg = PomyslAlgorithm()
    alg.fit(X_train, y_train)
    y_pred = alg.predict(X_test)
    print(accuracy_score(y_pred, y_test))
