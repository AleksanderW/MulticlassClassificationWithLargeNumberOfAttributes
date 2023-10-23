import os
import os.path as op
from itertools import product

import joblib
import numpy as np
import pandas as pd

from algorithms.Constants import extract_variables, set_params, check_if_models_exist
from algorithms.Evaluation import evaluate_classifiers
from algorithms.Proposed_Algorithm import ProposedAlgorithm
from algorithms.Settings import datasets, steps, ALG_amount_of_subtables, ALG_aggregations, n_splits, random_states, \
    classifiers, ALG_orders, ALG_kNN_neighbours, parallel_evaluation, ALG_split_variant, feature_selectors

path = op.join("..", "results")
if not os.path.exists(path):
    os.makedirs(path)

for selector, step, n_split, random_state in product(feature_selectors, steps, n_splits, random_states):
    for dataset in datasets:
        file_path = op.join(path,
                            "{}_RandomState{}_step{}_nsplit{}_{}.xlsx".format(dataset, random_state, step, n_split,
                                                                              selector))
        missing_models = check_if_models_exist(dataset, random_state, step, selector)
        if missing_models:
            continue

        selected_features = []
        for iteration in range(n_split):
            feature_selection = joblib.load(os.path.join("..", ".cache", f"{selector}_models",
                                                         f"{dataset}_{random_state}_{step}_{iteration}.sav"))
            selected_features.append(set(np.where(feature_selection.get_support())[0]))

        results = {"unique features": len(selected_features[0].union(*selected_features[1:])),
                   "intersection": len(selected_features[0].intersection(*selected_features[1:])),
                   "difference features": len(selected_features[0].union(*selected_features[1:])) - len(
                       selected_features[0].intersection(*selected_features[1:])),
                   "mean number of features": np.mean([len(x) for x in selected_features]),
                   "std": np.std([len(x) for x in selected_features]), }

        print(results)
