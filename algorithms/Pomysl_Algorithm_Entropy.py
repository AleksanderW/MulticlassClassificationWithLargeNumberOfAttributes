from scipy.special import softmax
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
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


class PomyslEntropyAlgorithm:
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
        Fit the stacking classifiers to the training data.

        Parameters:
        - X_train: array-like, shape (n_samples, n_features)
            The training input samples.
        - y_train: array-like, shape (n_samples,)
            The target values.
        """

        # Calculate unique labels in the training data
        self.y_labels = np.unique(y_train)

        # Create a stratified k-fold object
        skf = StratifiedKFold(shuffle=True, random_state=self.random_state)
        chosen_models = []

        # Iterate over each list of models
        for model_list in self.models:
            # List to store mean losses for each model
            model_mean_losses = []

            # Iterate over each model in the list
            for model in model_list:
                # Get the attributes of the model
                model_attributes = model.get_params()

                # Set the random state attribute if it exists
                if "random_state" in model_attributes:
                    setattr(self, "random_state", model_attributes["random_state"])

                # List to store losses for each fold
                fold_losses = []

                # Perform k-fold cross-validation
                for train_index, val_index in skf.split(X_train, y_train):
                    # Split the data into training and validation sets
                    X_train_split, X_val = X_train[train_index], X_train[val_index]
                    y_train_split, y_val = y_train[train_index], y_train[val_index]

                    # Fit the model to the training data
                    model.fit(X_train_split, y_train_split)

                    # Predict probabilities for the validation data
                    if hasattr(model, "predict_proba"):
                        y_val_pred_proba = model.predict_proba(X_val)
                    elif hasattr(model, "decision_function"):
                        decision_values = model.decision_function(X_val)
                        y_val_pred_proba = softmax(decision_values, axis=1)

                    # Calculate the loss using log loss
                    loss = log_loss(y_val, y_val_pred_proba, labels=self.y_labels)

                    # Append the loss to the fold_losses list
                    fold_losses.append(loss)

                # Calculate the mean loss for the model
                mean_loss = np.mean(fold_losses)

                # Append the model name and mean loss to the model_mean_losses list
                model_mean_losses.append((model, mean_loss))

            # Calculate the average mean loss
            average_loss = sum(loss for _, loss in model_mean_losses) / len(model_mean_losses)

            # Sort models based on mean loss in ascending order
            sorted_losses = sorted(model_mean_losses, key=lambda x: x[1])
            # Keep at least the top two models
            filtered_models = [model for model, loss in sorted_losses[:2]]

            # Include additional models whose mean loss is below or equal to the average
            filtered_models += [model for model, loss in sorted_losses[2:] if loss <= average_loss]

            chosen_models.append(filtered_models)

        row_fit_models = []

        for model_list in chosen_models:
            models_for_row = []

            for model in model_list:
                # Fit the model to the training data
                model.fit(X_train, y_train)

                models_for_row.append(model)

            row_fit_models.append(models_for_row)

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
    alg = PomyslEntropyAlgorithm()
    alg.fit(X_train, y_train)
    y_pred = alg.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    y_pred_proba = alg.predict_proba(X_test)
    print(roc_auc_score(y_test, y_pred_proba, multi_class="ovo", labels=np.unique(y)))
