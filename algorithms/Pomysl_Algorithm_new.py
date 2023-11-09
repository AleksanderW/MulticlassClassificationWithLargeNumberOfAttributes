import os
from itertools import combinations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


from algorithms.Aggregations import A1_aggr, A2_aggr, A3_aggr, A5_aggr, A6_aggr, A7_aggr, A8_aggr, \
    A9_aggr, A10_aggr
from algorithms.MinMaxArrays import find_min_max_pairs
from algorithms.Orders import xu_yager_order, lex1_order, lex2_order, get_decisions


class PomyslAlgorithm:
    def __init__(self, aggregation_type="A1", random_state=42,
                 order="xuyager", dataset="", iteration="", models=None):

        if models is None:
            models = [
                [
                    RandomForestClassifier(n_jobs=-1),
                    ExtraTreesClassifier(n_jobs=-1),
                ],
                [
                    KNeighborsClassifier(n_neighbors=1, p=1),
                    KNeighborsClassifier(n_neighbors=3, p=1),
                    KNeighborsClassifier(n_neighbors=5, p=1)
                ]
            ]
        self.models = models
        self.order = order
        self.aggregation_type = aggregation_type
        self.random_state = random_state
        self.subtable_col_indexes = None
        self.fit_group_of_models = None
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

        for model_list in self.models:
            models_for_row = []
            for model in model_list:
                model.fit(X_train, y_train)
                models_for_row.append(model)
            row_fit_models.append(models_for_row)
        self.fit_group_of_models = row_fit_models

    def predict(self, X_test):
        aggregation_functions = {
            'A1': A1_aggr, 'A2': A2_aggr, 'A3': A3_aggr, 'A4': A3_aggr, 'A5': A5_aggr, 'A6': A6_aggr, 'A7': A7_aggr,
            'A8': A8_aggr, 'A9': A9_aggr, 'A10': A10_aggr}
        orders = {'xuyager': xu_yager_order, 'lex1': lex1_order, 'lex2': lex2_order}
        grouped_predictions = []
        for grouped_models in self.fit_group_of_models:
            row_preds = []
            for model in grouped_models:
                row_preds.append(model.predict_proba(X_test))
            grouped_predictions.append(row_preds)
        min_max_array = [find_min_max_pairs(gpredict) for gpredict in grouped_predictions]
        aggregated = aggregation_functions.get(self.aggregation_type)(min_max_array)
        ordered = orders.get(self.order)(aggregated)
        decisions = get_decisions(ordered)
        multiclass_labels = [self.y_labels[d] for d in decisions]
        return multiclass_labels

    def predict_proba(self, X_test):
        row_predictions_proba = []

        for grouped_models in self.fit_group_of_models:
            row_preds = []
            for model in grouped_models:
                row_preds.append(model.predict_proba(X_test))
            row_predictions_proba.append(row_preds)

        min_max_array = [find_min_max_pairs(gpredict) for gpredict in row_predictions_proba]
        aggregation_function = {
            'A1': A1_aggr, 'A2': A2_aggr, 'A3': A3_aggr, 'A4': A3_aggr,
            'A5': A5_aggr, 'A6': A6_aggr, 'A7': A7_aggr, 'A8': A8_aggr,
            'A9': A9_aggr, 'A10': A10_aggr
        }
        aggregated = aggregation_function.get(self.aggregation_type)(min_max_array)

        for i in range(len(aggregated)):
            for j in range(len(aggregated[i])):
                aggregated[i][j] = np.mean(aggregated[i][j])
            if np.sum(aggregated[i]) != 0:
                aggregated[i] = aggregated[i] / np.sum(aggregated[i])
            else:
                aggregated[i] = np.ones_like(aggregated[i]) / len(aggregated[i])

        return aggregated


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    alg = PomyslAlgorithm()
    alg.fit(X_train, y_train)
    y_pred = alg.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    y_pred_proba = alg.predict_proba(X_test)
    print(roc_auc_score(y_test, y_pred_proba, multi_class="ovo", labels=np.unique(y)))
