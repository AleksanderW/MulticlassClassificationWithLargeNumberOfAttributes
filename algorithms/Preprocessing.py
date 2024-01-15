import itertools
import os
import random
import joblib
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
# from stability_selection.stability_selection import StabilitySelection
from Settings import datasets, steps, n_splits, random_states, feature_selectors


def process_iteration(
        iteration, train_index, test_index, selector, random_state, dataset, step, X, y
):
    """
    Process each iteration of the feature selection algorithm.

    Args:
        iteration (int): The current iteration number.
        train_index (array-like): The indices of the training data.
        test_index (array-like): The indices of the test data.
        selector (str): The type of feature selection algorithm to use.
        random_state (int): The random state for reproducibility.
        dataset (str): The name of the dataset.
        step (int): The step size for feature selection.
        X (array-like): The input features.
        y (array-like): The target variable.
    """
    print(iteration, selector, random_state, dataset, step)

    if selector == "RFECV":
        # Recursive Feature Elimination with Cross-Validation
        feature_selection = RFECV(
            estimator=SVC(kernel="linear", random_state=random_state),
            step=step,
            cv=StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True),
            scoring="accuracy",
            n_jobs=-1,
        )
    # elif selector == "StabilitySelection":
    #     # Stability Selection
    #     feature_selection = StabilitySelection(
    #         base_estimator=SVC(kernel="linear", random_state=random_state), n_jobs=-1
    #     )
    elif selector == "RFE":
        # Recursive Feature Elimination
        feature_selection = RFE(
            estimator=SVC(kernel="linear", random_state=random_state),
            step=step,
            n_features_to_select=100,
        )
    folder_path = os.path.join("..", ".cache", f"{selector}_models")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if selector == "RFECV" or selector == "RFE":
        # nazwabazy_randomstate_step_iteracja
        # File name format: {dataset}_{randomstate}_{step}_{iteration}.sav
        file_name = f"{dataset}_{random_state}_{step}_{iteration}.sav"
    elif selector == "StabilitySelection":
        # nazwabazy_randomstate_iteracja
        # File name format: {dataset}_{randomstate}_{iteration}.sav
        file_name = f"{dataset}_{random_state}_{iteration}.sav"
    model_path = os.path.join(folder_path, file_name)
    if os.path.exists(model_path):
        print(f"Model already exists: {model_path}")
        return None
    else:
        scaler = MinMaxScaler()
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        scaler.fit(X_train, y_train)
        X_train_scaled = scaler.transform(X_train)
        feature_selection.fit(X_train_scaled, y_train)
        joblib.dump(feature_selection, model_path)
        print(sum(feature_selection.get_support()), model_path)


# Main loop
for selector in feature_selectors:
    for random_state, dataset, n_split, step in itertools.product(
        random_states, datasets, n_splits, steps
    ):
        skf = StratifiedKFold(n_splits=n_split, random_state=random_state, shuffle=True)
        data = pd.read_csv(
            os.path.join("..", "dataset", "csv", dataset + "_FINAL.csv"), delimiter=";"
        )
        X = data.iloc[:, :-1].set_index("sample_id")
        y = data.iloc[:, -1].values

        Parallel(n_jobs=-1)(
            delayed(process_iteration)(
                iteration,
                train_index,
                test_index,
                selector,
                random_state,
                dataset,
                step,
                X,
                y,
            )
            for iteration, (train_index, test_index) in enumerate(skf.split(X, y))
        )
