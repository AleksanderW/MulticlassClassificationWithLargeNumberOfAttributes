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
        self.y_labels = np.unique(y_train)
        skf = StratifiedKFold(shuffle=True, random_state=self.random_state)
        selected_models = []

        for model_list in self.models:
            model_mean_losses = []

            for model in model_list:
                fold_losses = []

                for train_index, val_index in skf.split(X_train, y_train):
                    X_train_split, X_val = X_train[train_index], X_train[val_index]
                    y_train_split, y_val = y_train[train_index], y_train[val_index]

                    model_name = f"{model.__class__.__name__}_{'_'.join([f'{k}={v}' for k, v in model.get_params().items()])}"

                    model.fit(X_train_split, y_train_split)
                    y_val_pred_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else softmax(
                        model.decision_function(X_val), axis=1)
                    loss = log_loss(y_val, y_val_pred_proba, labels=self.y_labels)
                    fold_losses.append(loss)

                mean_loss = np.mean(fold_losses)
                model_mean_losses.append((model_name, mean_loss))

            worst_model_name = max(model_mean_losses, key=lambda x: x[1])[0]
            worst_model_index = next(
                i for i, model in enumerate(model_list) if model.__class__.__name__ in worst_model_name)
            del model_list[worst_model_index]

            for model in model_list:
                model_name = f"{model.__class__.__name__}_{'_'.join([f'{k}={v}' for k, v in model.get_params().items()])}"
                selected_models.append((model_name, model))

        stacking_classifier = StackingClassifier(
            estimators=selected_models,
            final_estimator=LogisticRegression(n_jobs=self.n_jobs, random_state=self.random_state),
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
