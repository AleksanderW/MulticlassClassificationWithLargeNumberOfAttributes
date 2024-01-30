import os

import numpy as np
import pandas as pd

from algorithms.Settings import datasets

with open('class_distribution.txt', 'w') as f:
    for dataset in datasets:
        data = pd.read_csv(
            os.path.join("..", "dataset", "csv", dataset + "_FINAL.csv"), delimiter=";"
        )
        X = data.iloc[:, :-1].set_index("sample_id")
        y = data.iloc[:, -1].values

        class_distribution = pd.Series(y).value_counts()
        total_samples = len(y)
        percentage_distribution = class_distribution / total_samples
        distribution_tuples = [np.round(percentage_distribution[class_name], 3) for class_name, count in
                               class_distribution.items()]

        f.write(
            f'Dataset name: {dataset}, Objects {X.shape[0]}, Features {X.shape[1]} Amount Of Classes: {len(class_distribution), "Distribution: ", distribution_tuples}\n')

        for class_name, count in class_distribution.items():
            f.write(
                f'Class: {class_name}, Count: {count}, Distribution: {np.round(percentage_distribution[class_name], 3)}\n')
        f.write('\n')


col_names = ["Dataset", "Samples", "Genes", "Classes (& Class distribution)"]
stats = pd.DataFrame(columns=col_names)
for dataset in datasets:
    data = pd.read_csv(
        os.path.join("..", "dataset", "csv", dataset + "_FINAL.csv"), delimiter=";"
    )
    X = data.iloc[:, :-1].set_index("sample_id")
    y = data.iloc[:, -1].values

    class_distribution = pd.Series(y).value_counts()
    total_samples = len(y)
    percentage_distribution = class_distribution / total_samples
    distribution_tuples = [np.round(percentage_distribution[class_name], 3) for class_name, count in
                           class_distribution.items()]

    row = {'Dataset': [dataset[:3]], 'Samples':[X.shape[0]], 'Genes':[X.shape[1]]}


    class_info = f"{len(set(y))}("
    for class_name, count in class_distribution.items():
        class_info += f'{np.round(percentage_distribution[class_name], 2)} '
    class_info += ")"
    row["Classes (& Class distribution)"] = [class_info]
    # print(row)
    stats = pd.concat([stats, pd.DataFrame(row)], ignore_index=True)
print(stats)
stats.to_latex(open("class_distribution_latex.txt","w"))
