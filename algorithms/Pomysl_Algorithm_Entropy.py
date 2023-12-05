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

        if models is None:
            models = [
                [
                    RandomForestClassifier(n_jobs=-1),
                    ExtraTreesClassifier(n_jobs=-1),
                    DecisionTreeClassifier(),
                ],
                [
                    KNeighborsClassifier(n_neighbors=1, metric="manhattan", n_jobs=-1),
                    KNeighborsClassifier(n_neighbors=3, metric="manhattan", n_jobs=-1),
                    KNeighborsClassifier(n_neighbors=5, metric="manhattan", n_jobs=-1),
                ],
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
        # Store unique labels from the training data
        self.y_labels = np.unique(y_train)

        # Initialize StratifiedKFold with 5 splits
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # Initialize an empty list to store metrics for each model
        metrics = []

        # Iterate over each model list
        for model_list in self.models:
            # Initialize an empty list to store metrics for each fold of the current model
            model_metrics = []

            # Perform stratified k-fold cross-validation
            for train_index, test_index in skf.split(X_train, y_train):
                # Split the data into training and testing folds
                X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
                y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

                # Initialize an empty list to store metrics for each model in the current fold
                fold_metrics = []

                # Iterate over each model in the current model list
                for model in model_list:
                    # Fit the model on the training fold
                    model.fit(X_train_fold, y_train_fold)

                    # Make predictions on the testing fold
                    y_pred_fold = model.predict(X_test_fold)
                    y_pred_proba_fold = model.predict_proba(X_test_fold)

                    # Calculate evaluation metrics for the current fold and model
                    acc_fold = accuracy_score(y_test_fold, y_pred_fold)
                    roc_auc_fold = roc_auc_score(
                        y_test_fold,
                        y_pred_proba_fold,
                        multi_class="ovo",
                        labels=self.y_labels,
                    )
                    loss = log_loss(
                        y_test_fold, y_pred_proba_fold, labels=self.y_labels
                    )

                    # Append the metrics for the current model to the fold metrics list
                    fold_metrics.append(
                        {
                            "model": model,
                            "acc": acc_fold,
                            "roc_auc": roc_auc_fold,
                            "log_loss": loss,
                        }
                    )

                # Append the fold metrics to the model metrics list
                model_metrics.append(fold_metrics)

            # Append the model metrics to the overall metrics list
            metrics.append(model_metrics)

        # Select the best-performing models based on mean ROC AUC score
        for i, model_list in enumerate(self.models):
            worst_model_index = np.argmax(
                [
                    np.mean([fold["log_loss"] for fold in metrics[i][j]])
                    for j in range(len(model_list))
                ]
            )
            self.models[i].pop(worst_model_index)

        # for i, model_list in enumerate(self.models):
        #     for j, model in enumerate(model_list):
        #         acc_mean = np.mean([fold['acc'] for fold in metrics[i][j]])
        #         log_loss_mean = np.mean([fold['log_loss'] for fold in metrics[i][j]])
        #         roc_auc_mean = np.mean([fold['roc_auc'] for fold in metrics[i][j]])
        #         model_str = str(model).ljust(70)
        #         print(
        #             f"{model_str} - \t Mean Accuracy: {acc_mean:.4f},\t Mean Roc Auc: {roc_auc_mean:.4f},\t Mean Log Loss: {log_loss_mean:.4f}")
        #     print()

        # Fit the remaining models on the entire training set
        row_fit_models = []
        for model_list in self.models:
            models_for_row = []
            for model in model_list:
                model.fit(X_train, y_train)
                models_for_row.append(model)
            row_fit_models.append(models_for_row)
        self.fit_group_of_models = row_fit_models

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
