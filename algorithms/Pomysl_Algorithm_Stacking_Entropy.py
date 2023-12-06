import numpy as np
from scipy.special import softmax
from sklearn.datasets import load_iris
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
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


class PomyslStackingEntropyAlgorithm:
    """
    Parameters:
        aggregation_type (str): The type of aggregation to be used. Defaults to "A1".
        random_state (int): The random state for reproducibility. Defaults to 42.
        order (str): The order in which the models are aggregated. Defaults to "xuyager".
        dataset (str): The dataset to be used. Defaults to an empty string.
        iteration (str): The iteration to be used. Defaults to an empty string.
        models (list): A list of models to be used. Defaults to None.
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
        # List to store the fitted stacking classifiers
        row_fit_models = []

        # Create a stratified k-fold object
        skf = StratifiedKFold(shuffle=True, random_state=self.random_state)

        # Iterate over each list of models
        for model_list in self.models:
            # List to store mean losses for each model
            model_mean_losses = []

            # Iterate over each model in the list
            for model in model_list:
                # List to store losses for each fold
                fold_losses = []

                # Perform k-fold cross-validation
                for train_index, val_index in skf.split(X_train, y_train):
                    # Split the data into training and validation sets
                    X_train_split, X_val = X_train[train_index], X_train[val_index]
                    y_train_split, y_val = y_train[train_index], y_train[val_index]

                    # Generate a unique model name based on its parameters
                    model_name = f"{model.__class__.__name__}_{'_'.join([f'{k}={v}' for k, v in model.get_params().items()])}"

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
                model_mean_losses.append((model_name, mean_loss))

            # Sort the model_mean_losses list based on mean loss
            model_mean_losses.sort(key=lambda x: x[1])
            # Get the name and index of the worst performing model
            worst_model_name = model_mean_losses[-1][0]
            worst_model_index = next(
                i
                for i, model in enumerate(model_list)
                if model.__class__.__name__ in worst_model_name
            )
            # Remove the worst performing model from the list
            del model_list[worst_model_index]

            # Create a list of (model name, model) tuples for the remaining models
            models_for_row = [
                (
                    f"{model.__class__.__name__}_{'_'.join([f'{k}={v}' for k, v in model.get_params().items()])}",
                    model,
                )
                for model in model_list
            ]

            # Create a stacking classifier using the remaining models
            stacking_classifier = StackingClassifier(
                estimators=models_for_row,
                final_estimator=LogisticRegression(n_jobs=-1, random_state=self.random_state),
                n_jobs=-1,
            )
            # Fit the stacking classifier to the entire training dataset
            stacking_classifier.fit(X_train, y_train)

            # Append the fitted stacking classifier to the row_fit_models list
            row_fit_models.append(stacking_classifier)

        # Store the fitted stacking classifiers in the self.fit_group_of_models attribute
        self.fit_group_of_models = row_fit_models

    def predict(self, X_test):
        """
        Predict the multiclass labels for the given test data.

        Args:
            X_test (array-like): The test data to make predictions on.

        Returns:
            list: The predicted multiclass labels.
        """

        # Dictionary mapping aggregation function names to corresponding functions
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

        # Dictionary mapping order names to corresponding functions
        orders = {"xuyager": xu_yager_order, "lex1": lex1_order, "lex2": lex2_order}

        # List to store predictions from each model
        grouped_predictions = []

        # Iterate over each model in the list of fit_group_of_models
        for stacked_model in self.fit_group_of_models:
            # Predict probabilities or decision function values for X_test using each model
            row_preds = [
                model.predict_proba(X_test) if hasattr(model, "predict_proba")
                else softmax(model.decision_function(X_test), axis=1)
                # Otherwise, use softmax on decision function values
                for model in stacked_model.estimators_
            ]

            # Append the predictions for each model to the grouped_predictions list
            grouped_predictions.append(row_preds)

        # List to store the minimum and maximum pairs for each prediction in grouped_predictions
        min_max_array = [
            find_min_max_pairs(gpredict) for gpredict in grouped_predictions
        ]

        # Apply the aggregation function specified by self.aggregation_type to min_max_array
        aggregated = aggregation_functions.get(self.aggregation_type)(min_max_array)

        # Apply the order function specified by self.order to the aggregated result
        ordered = orders.get(self.order)(aggregated)

        # Get the decision indices based on the ordered result
        decisions = get_decisions(ordered)

        # Retrieve the corresponding labels for the decision indices from the y_labels list
        multiclass_labels = [self.y_labels[d] for d in decisions]

        # Return the final multiclass labels
        return multiclass_labels

    def predict_proba(self, X_test):
        row_predictions_proba = []  # List to store predictions for each model

        # Iterate over each group of models
        for grouped_models in self.fit_group_of_models:
            row_preds = [
                model.predict_proba(X_test)  # Use predict_proba if available
                if hasattr(model, "predict_proba")
                else softmax(
                    model.decision_function(X_test), axis=1
                )  # Otherwise use softmax
                for model in grouped_models.estimators_
            ]
            row_predictions_proba.append(row_preds)  # Append predictions to the list

        # Calculate the minimum and maximum values for each prediction
        min_max_array = [
            find_min_max_pairs(gpredict) for gpredict in row_predictions_proba
        ]

        # Define a dictionary of aggregation functions
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

        # Apply the selected aggregation function to the list of minimum and maximum values
        aggregated = aggregation_function.get(self.aggregation_type)(min_max_array)

        # Normalize the aggregated values
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
    alg = PomyslStackingEntropyAlgorithm()
    alg.fit(X_train, y_train)
    y_pred = alg.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    y_pred_proba = alg.predict_proba(X_test)
    print(roc_auc_score(y_test, y_pred_proba, multi_class="ovo", labels=np.unique(y)))
