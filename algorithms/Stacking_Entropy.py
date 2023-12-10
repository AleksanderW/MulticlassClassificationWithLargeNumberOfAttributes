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


class StackingEntropy:
    def __init__(
            self,
            random_state=42,
            n_jobs=-1,
            models=None,
    ):
        self.models = models
        self.random_state = random_state
        self.subtable_col_indexes = None
        self.fit_classifier = None
        self.y_labels = None
        self.n_jobs = n_jobs
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
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

    def fit(self, X_train, y_train):
        """
        Trains the classifier on the given training data.

        Args:
            X_train (array-like): The input features of the training data.
            y_train (array-like): The target labels of the training data.

        Returns:
            None
        """
        self.y_labels = np.unique(y_train)
        skf = StratifiedKFold(shuffle=True, random_state=self.random_state)
        selected_models = []

        # Iterate over each list of models
        for model_list in self.models:
            # List to store mean losses for each model
            model_mean_losses = []

            # Iterate over each model in the list
            for model in model_list:
                model_attributes = model.get_params()
                if "random_state" in model_attributes:
                    setattr(self, "random_state", model_attributes["random_state"])

                # List to store losses for each fold
                fold_losses = []

                # Perform k-fold cross-validation
                for train_index, val_index in skf.split(X_train, y_train):
                    # Split the data into training and validation sets
                    X_train_split, X_val = X_train[train_index], X_train[val_index]
                    y_train_split, y_val = y_train[train_index], y_train[val_index]


                    # Fit the model on the training split and predict probabilities on the validation split
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

            # Add the remaining models to the selected models list
            for model in filtered_models:
                model_name = f"{model.__class__.__name__}_{'_'.join([f'{k}={v}' for k, v in model.get_params().items()])}"
                selected_models.append((model_name, model))

        # Create a stacking classifier with the selected models as estimators
        stacking_classifier = StackingClassifier(
            estimators=selected_models,
            final_estimator=LogisticRegression(
                n_jobs=self.n_jobs, random_state=self.random_state
            ),
            n_jobs=self.n_jobs,
        )
        stacking_classifier.fit(X_train, y_train)

        self.fit_classifier = stacking_classifier

    def predict_proba(self, X_test):
        if self.fit_classifier is None:
            raise ValueError("The classifier is not fitted. Call fit() first.")

        predictions_proba = self.fit_classifier.predict_proba(X_test)
        return predictions_proba

    def predict(self, X_test):
        if self.fit_classifier is None:
            raise ValueError("The classifier is not fitted. Call fit() first.")

        predictions = self.fit_classifier.predict(X_test)
        return predictions


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    alg = StackingEntropy()
    alg.fit(X_train, y_train)
    y_pred = alg.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    y_pred_proba = alg.predict_proba(X_test)
    print(roc_auc_score(y_test, y_pred_proba, multi_class="ovo", labels=np.unique(y)))
