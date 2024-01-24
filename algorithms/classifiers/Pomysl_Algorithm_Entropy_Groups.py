import copy

import numpy as np
from scipy.special import softmax
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

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


class PomyslEntropyGroupsAlgorithm:
    """
    Parameters:
        aggregation_type (str): The type of aggregation to use. Defaults to "A1".
        random_state (int): The seed value for random number generation. Defaults to 42.
        order (str): The order of the models to be used. Defaults to "xuyager".
        dataset (str): The dataset to be used. Defaults to an empty string.
        iteration (str): The iteration of the model. Defaults to an empty string.
        models (list): The list of models to be used. Defaults to None.
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
                    RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1),
                    RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                ],
                [
                    MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42),
                    MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=2000, random_state=42),
                    MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=2000, random_state=42),
                ],
                [
                    SVC(kernel='linear', C=1, random_state=42),
                    SVC(kernel='rbf', C=1, random_state=42),
                    SVC(kernel='poly', C=1, random_state=42),
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
        self.y_labels = np.unique(y_train)
        self.log_losses = {}
        skf = RepeatedStratifiedKFold(n_splits=2, random_state=self.random_state)
        chosen_models = []

        for model_list in self.models:
            model_mean_losses = []

            for model in model_list:
                model_attributes = model.get_params()

                if "random_state" in model_attributes:
                    setattr(self, "random_state", model_attributes["random_state"])

                fold_losses = []

                for train_index, val_index in skf.split(X_train, y_train):
                    X_train_split, X_val = X_train[train_index], X_train[val_index]
                    y_train_split, y_val = y_train[train_index], y_train[val_index]

                    model.fit(X_train_split, y_train_split)

                    if hasattr(model, "predict_proba"):
                        y_val_pred_proba = model.predict_proba(X_val)
                    elif hasattr(model, "decision_function"):
                        decision_values = model.decision_function(X_val)
                        y_val_pred_proba = softmax(decision_values, axis=1)

                    loss = log_loss(y_val, y_val_pred_proba, labels=self.y_labels)

                    fold_losses.append(loss)
                self.log_losses[model.__str__()] = copy.copy(fold_losses)
                mean_loss = np.mean(fold_losses)
                model_mean_losses.append((model, mean_loss))

            average_loss = sum(loss for _, loss in model_mean_losses) / len(model_mean_losses)
            sorted_losses = sorted(model_mean_losses, key=lambda x: x[1])
            filtered_models = [(model, loss) for model, loss in sorted_losses[:2]]
            filtered_models += [(model, loss) for model, loss in sorted_losses[2:] if loss <= average_loss]

            chosen_models.append(filtered_models)

        # Calculate the mean loss for each group of models within chosen_models
        group_mean_losses = [np.mean([loss for _, loss in group]) for group in chosen_models]

        # Calculate the average of group_mean_losses
        average_group_mean_loss = np.mean(group_mean_losses)

        filtered_chosen_models = [group for group, mean_loss in zip(chosen_models, group_mean_losses) if
                                  mean_loss <= average_group_mean_loss]

        if len(filtered_chosen_models) < 2:
            # Keep the two groups with the lowest mean losses
            sorted_chosen_models = sorted(chosen_models, key=lambda group: np.mean([loss for _, loss in group]))
            filtered_chosen_models = sorted_chosen_models[:2]

        row_fit_models = []

        for model_list in filtered_chosen_models:
            models_for_row = []

            for model, _ in model_list:
                model.fit(X_train, y_train)
                models_for_row.append(model)

            row_fit_models.append(models_for_row)
        self.log_losses["removed_models"] = row_fit_models.__str__()
        self.fit_group_of_models = row_fit_models

        # for i, model_list in enumerate(self.models):
        #     for j, model in enumerate(model_list):
        #         acc_mean = np.mean([fold['acc'] for fold in metrics[i][j]])
        #         log_loss_mean = np.mean([fold['log_loss'] for fold in metrics[i][j]])
        #         roc_auc_mean = np.mean([fold['roc_auc'] for fold in metrics[i][j]])
        #         model_str = str(model).ljust(70)
        #         print(
        #             f"{model_str} - \t Mean Accuracy: {acc_mean:.4f},\t Mean Roc Auc: {roc_auc_mean:.4f},\t Mean Log Loss: {log_loss_mean:.4f}")
        #     print()

    def predict(self, X_test):
        """
        Predict the class labels for the given test data.

        Args:
            X_test (array-like): The test data to predict on.

        Returns:
            list: The predicted class labels.
        """
        # Define the aggregation functions and orders
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

        grouped_predictions = []
        for grouped_models in self.fit_group_of_models:
            row_preds = []
            for model in grouped_models:
                if hasattr(model, "predict_proba"):
                    row_preds.append(model.predict_proba(X_test))
                elif hasattr(model, "decision_function"):
                    decision_scores = model.decision_function(X_test)
                    probabilities = softmax(decision_scores, axis=1)
                    row_preds.append(probabilities)
            grouped_predictions.append(row_preds)

        # Find the minimum and maximum pairs for each group of predictions
        min_max_array = [
            find_min_max_pairs(gpredict) for gpredict in grouped_predictions
        ]

        # Aggregate the predictions using the specified aggregation function
        aggregated = aggregation_functions.get(self.aggregation_type)(min_max_array)

        # Order the aggregated predictions using the specified order
        ordered = orders.get(self.order)(aggregated)

        # Get the decisions based on the ordered predictions
        decisions = get_decisions(ordered)

        # Convert the decisions to multiclass labels
        multiclass_labels = [self.y_labels[d] for d in decisions]

        return multiclass_labels

    def predict_proba(self, X_test):
        """
        Compute the predicted probabilities for each class using a combination of
        multiple models.

        Parameters:
        - X_test: The input data to make predictions on.

        Returns:
        - aggregated: The aggregated predicted probabilities for each class.
        """
        # Store the predicted probabilities for each model
        row_predictions_proba = []

        # Iterate over each group of models
        for grouped_models in self.fit_group_of_models:
            row_preds = []
            # Iterate over each individual model in the group
            for model in grouped_models:
                # Check if the model has a predict_proba method
                if hasattr(model, "predict_proba"):
                    # Use the predict_proba method to get the predicted probabilities
                    row_preds.append(model.predict_proba(X_test))
                # Check if the model has a decision_function method
                elif hasattr(model, "decision_function"):
                    # Use the decision_function method to get the decision scores
                    decision_scores = model.decision_function(X_test)
                    # Compute the probabilities using softmax function
                    probabilities = softmax(decision_scores, axis=1)
                    row_preds.append(probabilities)
            # Append the predicted probabilities for each model to the row_predictions_proba list
            row_predictions_proba.append(row_preds)

        # Find the minimum and maximum values for each group of predictions
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
        # Apply the aggregation function to the min_max_array
        aggregated = aggregation_function.get(self.aggregation_type)(min_max_array)

        # Normalize the aggregated predictions and compute the mean
        for i in range(len(aggregated)):
            for j in range(len(aggregated[i])):
                aggregated[i][j] = np.mean(aggregated[i][j])
            if np.sum(aggregated[i]) != 0:
                aggregated[i] = aggregated[i] / np.sum(aggregated[i])
            else:
                aggregated[i] = np.ones_like(aggregated[i]) / len(aggregated[i])

        # Return the aggregated predicted probabilities
        return aggregated


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    alg = PomyslEntropyGroupsAlgorithm()
    alg.fit(X_train, y_train)
    y_pred = alg.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    y_pred_proba = alg.predict_proba(X_test)
    print(roc_auc_score(y_test, y_pred_proba, multi_class="ovo", labels=np.unique(y)))
