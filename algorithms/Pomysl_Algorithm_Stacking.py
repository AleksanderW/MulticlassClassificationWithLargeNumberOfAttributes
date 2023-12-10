import os
from itertools import combinations
from scipy.special import softmax
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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


class PomyslStackingAlgorithm:
    """
    Parameters:
    - aggregation_type (str): The type of aggregation to use. Defaults to "A1".
    - random_state (int): The random seed for reproducibility. Defaults to 42.
    - order (str): The order of the models. Defaults to "xuyager".
    - dataset (str): The dataset name. Defaults to an empty string.
    - iteration (str): The iteration number. Defaults to an empty string.
    - models (list): The list of models to use. Defaults to None.
    """

    def __init__(
        self,
        aggregation_type="A1",
        random_state=42,
        order="xuyager",
        dataset="",
        iteration="",
        models=None,
    ):

        self.models = models
        self.order = order
        self.aggregation_type = aggregation_type
        self.random_state = random_state
        self.subtable_col_indexes = None
        self.fit_group_of_models = None
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
            f"aggregation_type='{self.aggregation_type}', random_state={self.random_state}, "
            f"order='{self.order}'"
        )

    def fit(self, X_train, y_train):
        """
        Fits the stacking classifier to the training data.

        Args:
            X_train (array-like): The input samples.
            y_train (array-like): The target values.

        """
        # Get unique labels in y_train
        self.y_labels = np.unique(y_train)

        row_fit_models = []

        # Iterate over each list of models
        for model_list in self.models:
            for model in model_list:
                model_attributes = model.get_params()
                if "random_state" in model_attributes:
                    setattr(self, "random_state", model_attributes["random_state"])
            # Create a list of tuples containing the model name and the model object
            models_for_row = [
                (
                    f"{model.__class__.__name__}_{'_'.join([f'{k}={v}' for k, v in model.get_params().items()])}".replace(
                        "__", "_"
                    ),
                    model,
                )
                for model in model_list
            ]

            # Create a stacking classifier using the models and fit it to the training data
            stacking_classifier = StackingClassifier(
                estimators=models_for_row,
                final_estimator=LogisticRegression(
                    n_jobs=-1, random_state=self.random_state
                ),
                n_jobs=-1,
            )
            stacking_classifier.fit(X_train, y_train)

            # Append the fitted stacking classifier to the list of fitted models
            row_fit_models.append(stacking_classifier)

        # Update the fit_group_of_models attribute
        self.fit_group_of_models = row_fit_models

    def predict(self, X_test):
        # Define aggregation functions
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

        # Define orders
        orders = {"xuyager": xu_yager_order, "lex1": lex1_order, "lex2": lex2_order}

        # List to store grouped predictions
        grouped_predictions = []

        # Iterate over each stacked model
        for stacked_model in self.fit_group_of_models:
            # Get predictions for each model in the stacked model
            row_preds = [
                model.predict_proba(X_test)
                if hasattr(model, "predict_proba")
                else softmax(model.decision_function(X_test), axis=1)
                for model in stacked_model.estimators_
            ]
            # Append row predictions to the grouped predictions list
            grouped_predictions.append(row_preds)

        # Calculate the min-max array for each group of predictions
        min_max_array = [
            find_min_max_pairs(gpredict) for gpredict in grouped_predictions
        ]

        # Perform aggregation based on the selected aggregation type
        aggregated = aggregation_functions.get(self.aggregation_type)(min_max_array)

        # Order the aggregated values based on the selected order
        ordered = orders.get(self.order)(aggregated)

        # Get decisions based on the ordered values
        decisions = get_decisions(ordered)

        # Get multiclass labels based on the decisions
        multiclass_labels = [self.y_labels[d] for d in decisions]

        return multiclass_labels

    def predict_proba(self, X_test):
        """
        Predicts the class probabilities for the given test data.

        Parameters:
            X_test (array-like): The test data.

        Returns:
            array-like: The predicted class probabilities.
        """
        # Initialize an empty list to store the predicted probabilities for each row
        row_predictions_proba = []

        # Iterate over each group of models
        for grouped_models in self.fit_group_of_models:
            # Generate predictions for each model in the group
            row_preds = [
                model.predict_proba(X_test)
                if hasattr(model, "predict_proba")
                else softmax(model.decision_function(X_test), axis=1)
                for model in grouped_models.estimators_
            ]
            # Append the predictions for the current group of models to the list
            row_predictions_proba.append(row_preds)

        # Find the minimum and maximum pairs for each group of predictions
        min_max_array = [
            find_min_max_pairs(gpredict) for gpredict in row_predictions_proba
        ]

        # Define the aggregation function based on the aggregation type
        aggregation_function = {
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

        # Aggregate the predictions using the specified aggregation function
        aggregated = aggregation_function.get(self.aggregation_type)(min_max_array)

        # Normalize the aggregated predictions
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    alg = PomyslStackingAlgorithm()
    alg.fit(X_train, y_train)
    y_pred = alg.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    y_pred_proba = alg.predict_proba(X_test)
    print(roc_auc_score(y_test, y_pred_proba, multi_class="ovo", labels=np.unique(y)))
