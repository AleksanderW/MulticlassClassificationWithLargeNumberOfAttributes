import os
import re

ALL_name = 'ALL_GSE13425'
BTu_name = 'BTu_GSE4290'
HeC_name = 'HeC_GSE14323'
HFF_name = 'HFF_GSE5406'
SPs_name = 'SPs_GSE13355'
SSh_name = 'SSh_GSE13904'


def extract_variables(text):
    pattern = r'PROP_ALG_s(\w+)_(\w+)_k\[(.+?)\]_(\w+)_(\w+)'
    match = re.search(pattern, text)
    if match:
        return "PROP_ALG", match.group(1), match.group(2), match.group(3), match.group(4), match.group(5)
    else:
        return text, '', '', '', '', ''


def set_params(classifiers, random_state):
    table = []
    for clf in classifiers:
        try:
            clf[0].set_params(random_state=random_state)
        except ValueError:
            table.append(clf)
            continue
        table.append(clf)
    return table


def check_if_models_exist(dataset, random_state, step, selector):
    folder_path = os.path.join("..", ".cache", f"{selector}_models")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    missing_models = []

    for iteration in range(5):
        model_name = f"{dataset}_{random_state}"
        if selector == "RFECV":
            model_name += f"_{step}_{iteration}.sav"
        elif selector == "StabilitySelection":
            model_name += f"_{iteration}.sav"
        model_path = os.path.join(folder_path, model_name)
        if not os.path.exists(model_path):
            missing_models.append(iteration)

    if missing_models:
        log_message = ""
        for iteration in missing_models:
            log_message += f"Dataset: {dataset}, random_state: {random_state},"
            if selector == "RFECV":
                log_message += f" step: {step},"
            log_message += f" iteration: {iteration}\n"

        log_file_path = "missing_models_log.txt"
        if os.path.exists(log_file_path):
            with open(log_file_path, "a") as log_file:
                log_file.write("\n" + log_message)
        else:
            with open(log_file_path, "w") as log_file:
                log_file.write(f"Missing {selector} models:\n" + log_message)

    return missing_models


def create_logs(selector):
    log_files = ["shuffle_log.txt", "table_split3_log.txt", "log_file.txt"]

    for log_file in log_files:
        try:
            with open(log_file, "a") as file:
                file.write("## " + selector + "\n")
        except FileNotFoundError:
            with open(log_file, "w") as file:
                file.write("## " + selector + "\n")


def empty_logs():
    log_files = ["shuffle_log.txt", "table_split3_log.txt", "log_file.txt"]
    for log_file in log_files:
        with open(log_file, "w") as file:
            file.truncate(0)
