import os
from itertools import combinations

import numpy as np
from scipy.special import softmax
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from algorithms.Aggregations import (
    A1_aggr,
    A2_aggr,
    A3_aggr,
    A4_aggr,
    A5_aggr,
    A6_aggr,
    A7_aggr,
    A8_aggr,
    A9_aggr,
    A10_aggr,
)
from algorithms.MinMaxArrays import find_min_max_pairs
from algorithms.Orders import xu_yager_order, lex1_order, lex2_order, get_decisions


class ProposedAlgorithm:
    def __init__(
        self,
        amount_of_subtables=2,
        aggregation_type="A1",
        random_state=42,
        order="xuyager",
        dataset="",
        iteration="",
        table_split=2,
        models=None,
    ):

        self.models = models
        self.table_split = table_split
        self.order = order
        self.amount_of_subtables = amount_of_subtables
        self.aggregation_type = aggregation_type
        self.random_state = random_state
        self.subtable_col_indexes = None
        self.fit_models = None
        self.y_labels = None
        self.dataset = dataset
        self.iteration = iteration

        if models is None:
            self.models = [
                [
                    RandomForestClassifier(n_jobs=-1, random_state=self.random_state),
                    ExtraTreesClassifier(n_jobs=-1, random_state=self.random_state),
                    DecisionTreeClassifier(random_state=self.random_state),
                ],
                [
                    KNeighborsClassifier(n_neighbors=1, metric="manhattan", n_jobs=-1),
                    KNeighborsClassifier(n_neighbors=3, metric="manhattan", n_jobs=-1),
                    KNeighborsClassifier(n_neighbors=5, metric="manhattan", n_jobs=-1),
                ],
            ]

    def get_params(self):
        return {
            "amount_of_subtables": self.amount_of_subtables,
            "aggregation_type": self.aggregation_type,
            "random_state": self.random_state,
            "order": self.order,
            "dataset": self.dataset,
            "iteration": self.iteration,
        }

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

    def __repr__(self):
        return (
            f"ProposedAlgorithm(amount_of_subtables={self.amount_of_subtables}, "
            f"aggregation_type='{self.aggregation_type}', random_state={self.random_state}, "
            f"knn_neighbours={self.kNN_neighbours}, order='{self.order}')"
        )

    def fit(self, X_train, y_train):
        self.y_labels = np.unique(y_train)

        def split_into_subtables1():
            # The process begins by evaluating the available features in the input data and determining whether there
            # are enough features to create subtables with a certain number of attributes. Depending on this
            # evaluation, the code decides on the subtable sizes and whether feature replacement is allowed.
            # Subsequently, random column indices are chosen for these subtables, ensuring compatibility with the
            # determined subtable sizes or maximum neighbor counts

            # Handle scenarios where there are insufficient features
            # in the input data to create subtables with a sufficient number of features.
            if X_train.shape[1] / self.amount_of_subtables > max(self.kNN_neighbours):
                self.subtable_columns = int(X_train.shape[1] / self.amount_of_subtables)
                replace = False
            else:
                self.subtable_columns = max(self.kNN_neighbours)
                replace = True
                shuffle_with_return_log(
                    self.dataset,
                    self.iteration,
                    self.amount_of_subtables,
                    self.random_state,
                    self.aggregation_type,
                    self.kNN_neighbours,
                    self.order,
                    self.subtable_columns,
                )
            # end

            col_indexes = []
            np.random.seed(self.random_state)
            for i in range(self.amount_of_subtables):
                col_indexes.append(
                    (
                        np.random.choice(
                            X_train.shape[1], self.subtable_columns, replace=replace
                        )
                    )
                )

            return col_indexes

        def split_into_subtables2():
            # # Process begins by shuffling attribute indices and dividing them into subtables. Any remaining
            # # attributes are distributed among these subtables. For each subtable, the code extends it with
            # # additional attributes if the size criterion is met.
            attributes = np.arange(X_train.shape[1])
            np.random.seed(self.random_state)
            np.random.shuffle(attributes)

            col_indexes = []
            attributes_per_subtable = len(attributes) // self.amount_of_subtables

            for i in range(self.amount_of_subtables):
                start_idx = i * attributes_per_subtable
                end_idx = (i + 1) * attributes_per_subtable
                col_indexes.append(tuple(attributes[start_idx:end_idx]))

            if len(attributes) % self.amount_of_subtables != 0:
                remaining_attributes = attributes[
                    attributes_per_subtable * self.amount_of_subtables :
                ]
                for i, attr in enumerate(remaining_attributes):
                    col_indexes[i] += (attr,)

            col_indexes_extended = []

            for subtable in col_indexes:
                if int(X_train.shape[1] / 2) + 1 - len(subtable) >= 0:
                    num_remaining = int(X_train.shape[1] / 2) + 1 - len(subtable)
                    remaining_attributes = [
                        attr for attr in attributes if attr not in subtable
                    ]
                    selected_attributes = np.random.choice(
                        remaining_attributes, num_remaining, replace=False
                    )
                    col_indexes_extended.append(subtable + tuple(selected_attributes))
                else:
                    col_indexes_extended.append(subtable)
            return col_indexes_extended

        def split_into_subtables3():
            svc = SVC(kernel="linear").fit(X_train, y_train)

            importance = permutation_importance(
                svc, X_train, y_train, scoring="accuracy"
            ).importances_mean
            feature_importance = [(i, importance[i]) for i in range(len(importance))]
            sorted_feature_importance = sorted(
                feature_importance, key=lambda x: x[1], reverse=True
            )
            sorted_feature_indexes = [index for index, _ in sorted_feature_importance]

            if len(importance) < self.amount_of_subtables:
                split3_features_log(
                    self.dataset,
                    self.iteration,
                    self.amount_of_subtables,
                    self.random_state,
                    self.aggregation_type,
                    len(importance),
                )
                self.amount_of_subtables = len(importance)

            tabs = [[] for _ in range(self.amount_of_subtables)]

            for index, _ in enumerate(sorted_feature_indexes):
                tabs[index % self.amount_of_subtables].append(
                    sorted_feature_indexes[index]
                )

            max_length = max(len(tab) for tab in tabs)

            for tab in tabs:
                tab.extend([None] * (max_length - len(tab)))

            tabs_transposed = [list(column) for column in zip(*tabs)]

            for i, column in enumerate(tabs_transposed):
                if i % 2 == 1:
                    column.reverse()

            tabs_modified = [
                [column for column in row if column is not None]
                for row in zip(*tabs_transposed)
            ]
            cleaned_tabs = [
                [item for item in inner_list if item is not None]
                for inner_list in tabs_modified
            ]

            return cleaned_tabs

        split_options = {
            1: split_into_subtables1,
            2: split_into_subtables2,
            3: split_into_subtables3,
        }

        split_into_subtables = split_options[self.table_split]
        self.subtable_col_indexes = split_into_subtables()
        row_fit_models = []

        for i in range(self.amount_of_subtables):
            X_train_subset_split = X_train[:, self.subtable_col_indexes[i]]
            models_for_row = []

            for model_list in self.models:
                models_for_subtable = []

                for model in model_list:
                    model.fit(X_train_subset_split, y_train)
                    models_for_subtable.append(model)

                models_for_row.append(models_for_subtable)
            row_fit_models.append(models_for_row)
        self.fit_models = row_fit_models

    def predict(self, X_test):
        aggregation_functions = {
            "A1": A1_aggr,
            "A2": A2_aggr,
            "A3": A3_aggr,
            "A4": A3_aggr,
            "A5": A5_aggr,
            "A6": A6_aggr,
            "A7": A7_aggr,
            "A8": A8_aggr,
            "A9": A9_aggr,
            "A10": A10_aggr,
        }
        orders = {"xuyager": xu_yager_order, "lex1": lex1_order, "lex2": lex2_order}
        predictions = []

        for i in range(self.amount_of_subtables):
            X_test_subset_split = X_test[:, self.subtable_col_indexes[i]]
            row_predictions = []

            for model_list in self.fit_models[i]:
                model_predictions = []

                for model in model_list:
                    model_attributes = model.get_params()
                    if "random_state" in model_attributes:
                        setattr(self, "random_state", model_attributes["random_state"])

                    if hasattr(model, "predict_proba"):
                        model_predictions.append(
                            model.predict_proba(X_test_subset_split)
                        )
                    elif hasattr(model, "decision_function"):
                        decision_scores = model.decision_function(X_test_subset_split)
                        probabilities = softmax(decision_scores, axis=1)
                        model_predictions.append(probabilities)

                row_predictions.append(model_predictions)

            predictions.append(row_predictions)

        min_max_array = []

        for predictions in predictions:
            for subtable in predictions:
                min_max_array.append(find_min_max_pairs(subtable))

        aggregated = aggregation_functions.get(self.aggregation_type)(min_max_array)
        ordered = orders.get(self.order)(aggregated)

        warning_log(
            self.dataset,
            self.iteration,
            self.amount_of_subtables,
            self.random_state,
            self.aggregation_type,
            ordered,
            self.order,
        )

        decisions = get_decisions(ordered)
        multiclass_labels = [self.y_labels[d] for d in decisions]

        return multiclass_labels

    def predict_proba(self, X_test):
        aggregation_functions = {
            "A1": A1_aggr,
            "A2": A2_aggr,
            "A3": A3_aggr,
            "A4": A4_aggr,
            "A5": A5_aggr,
            "A6": A6_aggr,
            "A7": A7_aggr,
            "A8": A8_aggr,
            "A9": A9_aggr,
            "A10": A10_aggr,
        }
        orders = {"xuyager": xu_yager_order, "lex1": lex1_order, "lex2": lex2_order}
        predictions = []

        for i in range(self.amount_of_subtables):
            X_test_subset_split = X_test[:, self.subtable_col_indexes[i]]
            row_predictions = []

            for model_list in self.fit_models[i]:
                model_predictions = []

                for model in model_list:
                    if hasattr(model, "predict_proba"):
                        model_predictions.append(
                            model.predict_proba(X_test_subset_split)
                        )
                    elif hasattr(model, "decision_function"):
                        decision_scores = model.decision_function(X_test_subset_split)
                        probabilities = softmax(decision_scores, axis=1)
                        model_predictions.append(probabilities)

                row_predictions.append(model_predictions)

            predictions.append(row_predictions)

        min_max_array = []

        for predictions in predictions:
            for subtable in predictions:
                min_max_array.append(find_min_max_pairs(subtable))

        aggregated = aggregation_functions.get(self.aggregation_type)(min_max_array)
        for i in range(len(aggregated)):
            for j in range(len(aggregated[i])):
                aggregated[i][j] = np.mean(aggregated[i][j])
            if np.sum(aggregated[i]) != 0:
                aggregated[i] = aggregated[i] / np.sum(aggregated[i])
            else:
                aggregated[i] = np.ones_like(aggregated[i]) / len(aggregated[i])
        return aggregated


def warning_log(
    dataset, iteration, subtables, random_state, aggregation, tuples, order
):
    all_tuples = []
    for tup in tuples:
        equal_tuples = []
        highest_tuple = (
            tup[0][0],
            tup[0][1],
        )  # istotne są tylko z wybieranych decyzji, a nie wszystkie
        for tup1, tup2 in combinations(tup, 2):
            if tup1[0] == tup2[0] and tup1[1] == tup2[1]:
                if highest_tuple > tup2:
                    break
                if not tup1 in equal_tuples:
                    equal_tuples.append(tup1)
                if not tup2 in equal_tuples:
                    equal_tuples.append(tup2)
            if equal_tuples:
                all_tuples.append(equal_tuples)
    all_tuples = [
        [(round(val[0], 4), round(val[1], 4), val[2]) for val in subtuple]
        for subtuple in all_tuples
    ]  # przejrzystość wyników
    if all_tuples:
        message = (
            f"{dataset} {aggregation} tab{subtables} i{iteration} rs{random_state} {order}, "
            + "equal tuples: "
            + f"{all_tuples}"
        )
        log_file_path = "log_file.txt"
        if os.path.exists(log_file_path):
            with open(log_file_path, "r") as f:
                lines = f.readlines()
                existing_messages = [line.strip() for line in lines]

            if message not in existing_messages:
                with open(log_file_path, "a") as f:
                    f.write(f"{message}\n")
        else:
            with open(log_file_path, "w") as f:
                f.write(f"{message}\n")
    return all_tuples


def shuffle_with_return_log(
    dataset, iteration, subtables, random_state, aggregation, order, columns
):
    message = f"{dataset} {aggregation} tab{subtables} i{iteration} rs{random_state} {order}: column amount{columns}"
    shuffle_path = "shuffle_log.txt"
    if os.path.exists(shuffle_path):
        with open(shuffle_path, "r") as f:
            lines = f.readlines()
            existing_messages = [line.strip() for line in lines]

        if message not in existing_messages:
            with open(shuffle_path, "a") as f:
                f.write(f"{message}\n")
    else:
        with open(shuffle_path, "w") as f:
            f.write(f"{message}\n")


def split3_features_log(
    dataset, iteration, subtables, random_state, aggregation, less_tables
):
    message = (f"{dataset} {aggregation} tab{subtables} i{iteration} rs{random_state} : "
               f"subtables amount for {subtables} subtables set to the max amount of features ({less_tables})")
    shuffle_path = "table_split3_log.txt"
    if os.path.exists(shuffle_path):
        with open(shuffle_path, "r") as f:
            lines = f.readlines()
            existing_messages = [line.strip() for line in lines]

        if message not in existing_messages:
            with open(shuffle_path, "a") as f:
                f.write(f"{message}\n")
    else:
        with open(shuffle_path, "w") as f:
            f.write(f"{message}\n")


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    alg = ProposedAlgorithm(amount_of_subtables=7, table_split=2)
    alg.fit(X_train, y_train)
    y_pred = alg.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    y_pred_proba = alg.predict_proba(X_test)
    print(roc_auc_score(y_test, y_pred_proba, multi_class="ovo", labels=np.unique(y)))
