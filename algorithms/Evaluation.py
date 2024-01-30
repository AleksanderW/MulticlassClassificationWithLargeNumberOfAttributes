import copy
import os
import random
import warnings
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import shuffle

from algorithms.Settings import external_shuffles, random_states, train_size

# warnings.filterwarnings("ignore",
#                         message="The least populated class in y has only 4 members, which is less than n_splits=5.",
#                         append=True)

# Funkcja która zwraca indexy wierszy zamiast tablic tak jak train_test_split
def custom_train_test_split(X, y, train_size=0.8, shuffle=True, random_state=None, stratify=None):
    train_indices, test_indices = train_test_split(range(len(X)), train_size=train_size, shuffle=shuffle, random_state=random_state, stratify=stratify)
    return train_indices, test_indices

# TODO ADD graphs showing log_loss in crossvalidation (SSH DATASET)
# W wersji sklearn 1.4.0 ClassifierMixin wymusza posiadanie metody fit? https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html#sklearn-base-classifiermixin
def evaluate_classifiers(
        clf: ClassifierMixin, name, dataset, n_splits, step, random_state,
        # selector,
        parallel=True, feature_selection=False
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
    y = LabelEncoder().fit_transform(y)

    def evaluate_fold(iteration, X_train, X_test, y_train, y_test, feature_selection):
        # if set(y_train).difference(set(y_test)) is not {}:
        #     print(clf, y_train, y_test, set(y_train).difference(set(y_test)))
        #     raise Exception("Zbiór testowy nie zawiera klasy ze zbioru treningowego")
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if "dataset" in clf.get_params():
            clf.dataset = dataset
        if "iteration" in clf.get_params():
            clf.iteration = iteration
        if "random_state" in clf.get_params():
            clf.random_state = random_state
        # Train the classifier
        # make local copy of clf (tak dla pewności że nie nadpisuje obiektu)
        clf_ = copy.deepcopy(clf)
        clf_.fit(X_train, y_train)
        # Make predictions
        y_pred = clf_.predict(X_test)

        # warnings.simplefilter("error")

        try:
            y_pred_proba = clf_.predict_proba(X_test)
        except AttributeError:
            y_pred_scores = clf_.decision_function(X_test)
            y_pred_proba = softmax(y_pred_scores, axis=1)
        # try:
        # Calculate evaluation metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(
                y_test, y_pred_proba, multi_class="ovo", labels=np.unique(y)
            ),
        }
        # except:
        #     print(clf_, y_pred, y_test)
        #     print(set(y_train).difference(set(y_test)))
        try:
            metrics["losses"] = clf_.log_losses.__str__()
        except AttributeError:
            metrics["losses"] = clf_.__str__() + "WITHOUT log_loss_field"
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

        # warnings.simplefilter("ignore")
        return metrics

    ###################
    np.random.seed(random_state)
    random.seed(random_state)
    random_seeds = [random.randint(0, 2 ** 32 - 1) for _ in range(5)]

    # Check Test sets diferences
    evaluate_fold_parameters = [(seed, *custom_train_test_split(X, y, train_size=train_size, shuffle=True,
                                                        random_state=seed, stratify=y)) for seed in random_seeds]
    number_of_test_samples = 0
    unique_test_data = set()
    for _, train_index, test_index in evaluate_fold_parameters:
        number_of_test_samples+= len(test_index)
        unique_test_data = unique_test_data.union(set(test_index))
    print(dataset,clf, "unikatowość:" ,len(unique_test_data)/number_of_test_samples, "próbek:", number_of_test_samples)

    if parallel:
        results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(lambda seed: evaluate_fold(seed, *train_test_split(X, y, train_size=train_size, shuffle=True,
                                                                              random_state=seed, stratify=y),
                                                      feature_selection))(seed)
            for seed in random_seeds
        )
    else:
        results = [evaluate_fold(seed, *train_test_split(X, y, train_size=train_size, shuffle=True,
                                                         random_state=seed, stratify=y),
                                 feature_selection)
                   for seed in random_seeds]


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
    log_loss = {index: result["losses"] for index, result in enumerate(results)}

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
        "log_loss": log_loss
    }
