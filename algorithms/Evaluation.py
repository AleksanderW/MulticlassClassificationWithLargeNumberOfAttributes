import os

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import softmax
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def evaluate_classifiers(
    clf, name, dataset, n_splits, step, random_state, selector, parallel=True, feature_selection = False
):
    """Evaluate classifiers using k-fold cross-validation.
     Args:
         clf (object): Classifier object.
         name (str): Name of the classifier.
         dataset (str): Name of the dataset.
         n_splits (int): Number of splits for cross-validation.
         step (int): Step value.
         random_state (int): Random state value.
         selector (str): Feature selection method.
         parallel (bool, optional): Whether to use parallel computing. Defaults to True.
     Returns:
         dict: Dictionary of evaluation metrics.
     """
    # Read the dataset
    data = pd.read_csv(
        os.path.join("..", "dataset", "csv", dataset + "_FINAL.csv"), delimiter=";"
    )
    X = data.iloc[:, :-1].set_index("sample_id")
    y = data.iloc[:, -1].values

    def evaluate_fold(iteration, train_index, test_index, feature_selection):
        """Evaluate a single fold of cross-validation.
        Args:
            iteration (int): Iteration number.
            train_index (array): Indices of training samples.
            test_index (array): Indices of test samples.
        Returns:
            dict: Dictionary of evaluation metrics.
        """
        # Preprocess the data

        scaler = MinMaxScaler()
        label_encoder = LabelEncoder()
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
        y_encoded = label_encoder.transform(y)
        scaler.fit(X_train, y_train)
        X_train = scaler.transform(X_train)

        if feature_selection:
            # Perform feature selection
            if selector == "RFECV" or selector == "RFE":
                feature_selection = joblib.load(os.path.join("..", ".cache", f"{selector}_models",
                                                             f"{dataset}_{random_state}_{step}_{iteration}.sav")).support_
            elif selector == "StabilitySelection":
                feature_selection = joblib.load(os.path.join("..", ".cache", f"{selector}_models",
                                                             f"{dataset}_{random_state}_{iteration}.sav")).get_support(
                    indices=False)
            # Preprocess the test data
            X_train = X_train[:, feature_selection]
            X_test_scaled = scaler.fit_transform(X_test, y_test)
            X_test = X_test_scaled[:, feature_selection]
        # Set classifier parameters
        if "dataset" in clf.get_params():
            clf.dataset = dataset
        if "iteration" in clf.get_params():
            clf.iteration = iteration
        if "random_state" in clf.get_params():
            clf.random_state = random_state
        # Train the classifier
        clf.fit(X_train, y_train)
        # Make predictions
        y_pred = clf.predict(X_test)


        try:
            y_pred_proba = clf.predict_proba(X_test)
        except AttributeError:
            y_pred_scores = clf.decision_function(X_test)
            y_pred_proba = softmax(y_pred_scores, axis=1)

        # Calculate evaluation metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(
                y_test, y_pred_proba, multi_class="ovo", labels=np.unique(y_encoded)
            ),
        }

        averages = [
            # 'micro',
            "macro",
            # 'weighted'
        ]
        for average in averages:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average=average, zero_division=0
            )
            metrics[f"precision_{average}"] = precision
            metrics[f"recall_{average}"] = recall
            metrics[f"f1_score_{average}"] = f1

        return metrics

    if parallel:
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_fold)(iteration, train_index, test_index, feature_selection)
            for iteration, (train_index, test_index) in enumerate(
                StratifiedKFold(
                    n_splits=n_splits, random_state=random_state, shuffle=True
                ).split(X, y)
            )
        )
    else:
        results = [
            evaluate_fold(iteration, train_index, test_index, feature_selection)
            for iteration, (train_index, test_index) in enumerate(
                StratifiedKFold(
                    n_splits=n_splits, random_state=random_state, shuffle=True
                ).split(X, y)
            )
        ]

    accuracy = [result["accuracy"] for result in results]
    balanced_accuracy = [result["balanced_accuracy"] for result in results]
    # precision_mic = [result['precision_micro'] for result in results]
    precision_mac = [result["precision_macro"] for result in results]
    # precision_wgt = [result['precision_weighted'] for result in results]
    # recall_mic = [result['recall_micro'] for result in results]
    recall_mac = [result["recall_macro"] for result in results]
    # recall_wgt = [result['recall_weighted'] for result in results]
    # f1_score_mic = [result['f1_score_micro'] for result in results]
    f1_score_mac = [result["f1_score_macro"] for result in results]
    # f1_score_wgt = [result['f1_score_weighted'] for result in results]
    auc = [result["auc"] for result in results]

    return {
        "name": name,
        "dataset": dataset,
        "accuracy": (np.mean(accuracy), np.std(accuracy)),
        "balanced_accuracy": (np.mean(balanced_accuracy), np.std(balanced_accuracy)),
        # 'precision_mic': (np.mean(precision_mic), np.std(precision_mic)),
        "precision_mac": (np.mean(precision_mac), np.std(precision_mac)),
        # 'precision_wgt': (np.mean(precision_wgt), np.std(precision_wgt)),
        # 'recall_mic': (np.mean(recall_mic), np.std(recall_mic)),
        "recall_mac": (np.mean(recall_mac), np.std(recall_mac)),
        # 'recall_wgt': (np.mean(recall_wgt), np.std(recall_wgt)),
        # 'f1_score_mic': (np.mean(f1_score_mic), np.std(f1_score_mic)),
        "f1_score_mac": (np.mean(f1_score_mac), np.std(f1_score_mac)),
        # 'f1_score_wgt': (np.mean(f1_score_wgt), np.std(f1_score_wgt)),
        "auc": (np.mean(auc), np.std(auc)),
    }
