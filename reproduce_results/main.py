import os
import os.path as op
from itertools import product

import pandas as pd

from algorithms.Constants import extract_variables, set_params, check_if_models_exist, create_logs, empty_logs
from algorithms.Evaluation import evaluate_classifiers
from algorithms.Proposed_Algorithm import ProposedAlgorithm
from algorithms.Proposed_Algorithm_Stacking import ProposedStackingAlgorithm
from algorithms.Settings import datasets, steps, ALG_amount_of_subtables, ALG_aggregations, n_splits, random_states, \
    classifiers, ALG_orders, ALG_kNN_neighbours, parallel_evaluation, ALG_split_variant, feature_selectors

path = op.join("..", "results")
if not os.path.exists(path):
    os.makedirs(path)

empty_logs()
for selector in feature_selectors:
    create_logs(selector)
    for step, n_split, random_state in product(steps, n_splits, random_states):
        clfs_loop = classifiers.copy()
        clfs_loop += [
            (ProposedStackingAlgorithm(amount_of_subtables=subtab, aggregation_type=agg, knn_neighbours=kNN, order=order,
                               table_split=table_split, random_state=random_state),
             f"PROP_ALG_s{subtab}_{agg}_k{kNN}_{order}_{table_split}") for agg, subtab, kNN, order, table_split in
            product(ALG_aggregations, ALG_amount_of_subtables, ALG_kNN_neighbours, ALG_orders, ALG_split_variant)]
        clfs_loop = set_params(clfs_loop, random_state)

        for dataset in datasets:
            if selector == "RFECV":
                file_path = op.join(path,
                                    "{}_RandomState{}_step{}_nsplit{}_{}.xlsx".format(dataset, random_state, step,
                                                                                      n_split,
                                                                                      selector))
            elif selector == "StabilitySelection":
                file_path = op.join(path,
                                    "{}_RandomState{}_nsplit{}_{}.xlsx".format(dataset, random_state, n_split,
                                                                               selector))
            missing_models = check_if_models_exist(dataset, random_state, step, selector)
            if missing_models:
                continue
            results = []
            for clf, name in clfs_loop:
                result = evaluate_classifiers(clf, name, dataset, n_split, step, random_state, selector,
                                              parallel=parallel_evaluation)
                name, subtab, agg, kNN, ordr, split = extract_variables(name)
                results.append({'classifier': name,
                                'subtables': subtab,
                                'aggr': agg,
                                'kNN': kNN,
                                'order': ordr,
                                'split': split,
                                'acc': result['accuracy'][0],
                                'acc_std': result['accuracy'][1],
                                'bal_acc': result['balanced_accuracy'][0],
                                'bal_acc_std': result['balanced_accuracy'][1],
                                'mic_prec': result['precision_mic'][0],
                                'mic_prec_std': result['precision_mic'][1],
                                'mac_prec': result['precision_mac'][0],
                                'mac_prec_std': result['precision_mac'][1],
                                'wgt_prec': result['precision_wgt'][0],
                                'wgt_prec_std': result['precision_wgt'][1],
                                'mic_recall': result['recall_mic'][0],
                                'mic_recall_std': result['recall_mic'][1],
                                'mac_recall': result['recall_mac'][0],
                                'mac_recall_std': result['recall_mac'][1],
                                'wgt_recall': result['recall_wgt'][0],
                                'wgt_recall_std': result['recall_wgt'][1],
                                'mic_f1': result['f1_score_mic'][0],
                                'mic_f1_std': result['f1_score_mic'][1],
                                'mac_f1': result['f1_score_mac'][0],
                                'mac_f1_std': result['f1_score_mac'][1],
                                'wgt_f1': result['f1_score_wgt'][0],
                                'wgt_f1_std': result['f1_score_wgt'][1]
                                })

            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(by=['acc', 'bal_acc'], ascending=False)
            writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
            results_df.to_excel(writer, sheet_name='Results', index=False)
            writer.close()
