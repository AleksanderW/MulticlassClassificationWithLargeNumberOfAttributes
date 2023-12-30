import os
import os.path as op
from itertools import product

import pandas as pd

from algorithms.Constants import (
    extract_variables,
    set_params,
    check_if_models_exist,
    create_logs,
    empty_logs,
)
from algorithms.Evaluation import evaluate_classifiers
from algorithms.Settings import (
    datasets,
    steps,
    ALG_aggregations,
    ALG_models,
    n_splits,
    random_states,
    classifiers,
    ALG_orders,
    parallel_evaluation,
    feature_selectors,
    ALG_amount_of_subtables,
    ALG_split_variant,
)
from algorithms.classifiers.Pomysl_Algorithm import PomyslAlgorithm
from algorithms.classifiers.Pomysl_Algorithm_Entropy import PomyslEntropyAlgorithm
from algorithms.classifiers.Pomysl_Algorithm_Entropy_Groups import PomyslEntropyGroupsAlgorithm
from algorithms.classifiers.Proposed_Algorithm import ProposedAlgorithm
from algorithms.classifiers.Stacking_Entropy import StackingEntropy

path = op.join("..", "results")
if not os.path.exists(path):
    os.makedirs(path)

empty_logs()

selector_file_paths = {
    "RFECV": "{}_RandomState{}_selector{}_step{}_nsplit{}_{}.xlsx",
    "RFE": "{}_RandomState{}_selector{}_step{}_nsplit{}_{}.xlsx",
    "StabilitySelection": "{}_RandomState{}_nsplit{}_{}.xlsx",
}

for selector in feature_selectors:
    create_logs(selector)
    for step, n_split, random_state in product(steps, n_splits, random_states):
        clfs_loop = classifiers.copy()
        # clfs_loop += [
        #     (
        #         ProposedAlgorithm(
        #             amount_of_subtables=subtab,
        #             aggregation_type=agg,
        #             order=order,
        #             models=ALG_models,
        #             table_split=table_split,
        #             random_state=random_state,
        #         ),
        #         f"PROP_ALG_s{subtab}_{agg}_{order}_{table_split}",
        #     )
        #     for agg, subtab, order, table_split in product(
        #         ALG_aggregations, ALG_amount_of_subtables, ALG_orders, ALG_split_variant
        #     )
        # ]

        clfs_loop += [
            (
                PomyslAlgorithm(
                    aggregation_type=agg,
                    order=order,
                    random_state=random_state,
                    models=ALG_models,
                ),
                f"POM_ALG_{agg}_{order}",
            )
            for agg, order in product(ALG_aggregations, ALG_orders)
        ]

        clfs_loop += [
            (
                PomyslEntropyAlgorithm(
                    aggregation_type=agg,
                    order=order,
                    random_state=random_state,
                    models=ALG_models,
                ),
                f"POM_ENTROPY_{agg}_{order}",
            )
            for agg, order in product(ALG_aggregations, ALG_orders)
        ]
        clfs_loop += [
            (
                PomyslEntropyGroupsAlgorithm(
                    aggregation_type=agg,
                    order=order,
                    random_state=random_state,
                    models=ALG_models,
                ),
                f"POM_enGROUPS_{agg}_{order}",
            )
            for agg, order in product(ALG_aggregations, ALG_orders)
        ]
        # clfs_loop += [
        #     (
        #         PomyslStackingAlgorithm(
        #             aggregation_type=agg,
        #             order=order,
        #             random_state=random_state,
        #             models=ALG_models,
        #         ),
        #         f"POM_STACKING_{agg}_{order}",
        #     )
        #     for agg, order in product(ALG_aggregations, ALG_orders)
        # ]
        #
        # clfs_loop += [
        #     (
        #         PomyslStackingAlgorithm(
        #             aggregation_type=agg,
        #             order=order,
        #             random_state=random_state,
        #             models=ALG_models,
        #         ),
        #         f"POM_STACKING_{agg}_{order}",
        #     )
        #     for agg, order in product(ALG_aggregations, ALG_orders)
        # ]
        #
        # clfs_loop += [
        #     (
        #         PomyslStackingEntropyAlgorithm(
        #             aggregation_type=agg,
        #             order=order,
        #             random_state=random_state,
        #             models=ALG_models,
        #         ),
        #         f"POM_StackingEntropy_{agg}_{order}",
        #     )
        #     for agg, order in product(ALG_aggregations, ALG_orders)
        # ]
        # clfs_loop += [
        #     (
        #         StackingEntropy(
        #             random_state=random_state,
        #             models=ALG_models,
        #         ),
        #         f"STACKING_Entropy",
        #     )
        # ]
        clfs_loop = set_params(clfs_loop, random_state)

        for dataset in datasets:
            file_path = op.join(
                path,
                selector_file_paths[selector].format(
                    dataset, random_state, selector, step, n_split, selector
                ),
            )
            missing_models = check_if_models_exist(
                dataset, random_state, step, selector
            )

            if missing_models:
                continue
            results = []
            for clf, name in clfs_loop:
                result = evaluate_classifiers(
                    clf,
                    name,
                    dataset,
                    n_split,
                    step,
                    random_state,
                    selector,
                    parallel=parallel_evaluation,
                )
                name, subtab, agg, ordr, split = extract_variables(name)
                results.append(
                    {
                        "classifier": name,
                        "subtables": subtab,
                        "aggr": agg,
                        "order": ordr,
                        "split": split,
                        "auc": result["auc"][0],
                        "auc_std": result["auc"][1],
                        "acc": result["accuracy"][0],
                        "acc_std": result["accuracy"][1],
                        "bal_acc": result["balanced_accuracy"][0],
                        "bal_acc_std": result["balanced_accuracy"][1],
                        # 'mic_prec': result['precision_mic'][0],
                        # 'mic_prec_std': result['precision_mic'][1],
                        "mac_prec": result["precision_mac"][0],
                        "mac_prec_std": result["precision_mac"][1],
                        # 'wgt_prec': result['precision_wgt'][0],
                        # 'wgt_prec_std': result['precision_wgt'][1],
                        # 'mic_recall': result['recall_mic'][0],
                        # 'mic_recall_std': result['recall_mic'][1],
                        "mac_recall": result["recall_mac"][0],
                        "mac_recall_std": result["recall_mac"][1],
                        # 'wgt_recall': result['recall_wgt'][0],
                        # 'wgt_recall_std': result['recall_wgt'][1],
                        # 'mic_f1': result['f1_score_mic'][0],
                        # 'mic_f1_std': result['f1_score_mic'][1],
                        "mac_f1": result["f1_score_mac"][0],
                        "mac_f1_std": result["f1_score_mac"][1],
                        # 'wgt_f1': result['f1_score_wgt'][0],
                        # 'wgt_f1_std': result['f1_score_wgt'][1]
                    }
                )

            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(
                by=["auc", "acc", "bal_acc"], ascending=False
            )
            writer = pd.ExcelWriter(file_path, engine="xlsxwriter")
            results_df.to_excel(writer, sheet_name="Results", index=False)
            writer.close()
