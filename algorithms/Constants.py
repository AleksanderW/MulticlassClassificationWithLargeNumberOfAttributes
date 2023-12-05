import os
import re

ALL_name = 'ALL_GSE13425'
BTu_name = 'BTu_GSE4290'
HeC_name = 'HeC_GSE14323'
HFF_name = 'HFF_GSE5406'
SPs_name = 'SPs_GSE13355'
SSh_name = 'SSh_GSE13904'


def extract_variables(text):
    """
    Extracts variables from the given text using regular expressions.

    Args:
        text (str): The input text.

    Returns:
        tuple: A tuple containing the extracted variables.
    """

    # Define the regular expression patterns
    pattern = r'(\w+[_A-Za-z]+)_s(\w+)_(\w+)_(\w+)_(\w+)'
    pattern2 = r'(\w+[_A-Za-z]+)_(\w+)_(\w+)'

    # Search for matches using the first pattern
    match = re.search(pattern, text)

    # If a match is found, return the matched groups
    if match:
        return match.groups(default='')

    # If no match is found using the first pattern, search using the second pattern
    match2 = re.search(pattern2, text)

    # If a match is found using the second pattern, return the specific groups
    if match2:
        return match2.group(1), "", match2.group(2), match2.group(3), ""

    # If no match is found using either pattern, return default values
    return text, '', '', '', ''


if __name__ == "__main__":
    # Example usage:
    input_text = "'PROP_ALG_s5_A10_xuyager_2'"
    result = extract_variables(input_text)
    print(result)

    input_text2 = "'POM_ALG_A6_xuyager"
    result2 = extract_variables(input_text2)
    print(result2)

    input_text3 = "Invalid_Text"
    result3 = extract_variables(input_text3)
    print(result3)


def set_params(classifiers, random_state):
    """
    Set the random_state parameter for each classifier and model in the list.

    Args:
        classifiers (list): A list of tuples containing the classifiers and their names.
        random_state (int): The random state parameter to set.

    Returns:
        list: A list of tuples that were not able to set the random_state parameter.

    """
    table = []
    for clf_tuple in classifiers:
        clf, clf_name = clf_tuple
        try:
            clf.set_params(random_state=random_state)
        except ValueError:
            table.append(clf_tuple)
            continue
        if hasattr(clf, 'models') and isinstance(clf.models, list):
            for model_list in clf.models:
                for model in model_list:
                    try:
                        model.set_params(random_state=random_state)
                    except ValueError:
                        pass
        table.append(clf_tuple)

    return table


def check_if_models_exist(dataset, random_state, step, selector):
    """
    Check if the models exist for a given dataset, random state, step, and selector.

    Args:
        dataset (str): The name of the dataset.
        random_state (int): The random state.
        step (int): The step value (used for RFECV and RFE selectors).
        selector (str): The selector type.

    Returns:
        list: A list of missing model iterations.
    """
    # Create the folder path for the models
    folder_path = os.path.join("..", ".cache", f"{selector}_models")
    os.makedirs(folder_path, exist_ok=True)

    # Initialize list to store missing models
    missing_models = []

    # Check for missing models
    for iteration in range(5):
        model_name = f"{dataset}_{random_state}"
        if selector in ["RFECV", "RFE"]:
            model_name += f"_{step}_{iteration}.sav"
        elif selector == "StabilitySelection":
            model_name += f"_{iteration}.sav"
        model_path = os.path.join(folder_path, model_name)
        if not os.path.exists(model_path):
            missing_models.append(iteration)

    # Log missing models if any
    if missing_models:
        log_message = "\n".join([
            f"Dataset: {dataset}, random_state: {random_state},{' step: ' + str(step) if selector in ['RFECV', 'RFE'] else ''} iteration: {iteration}"
            for iteration in missing_models
        ])

        log_file_path = "missing_models_log.txt"
        with open(log_file_path, "a" if os.path.exists(log_file_path) else "w") as log_file:
            log_file.write(f"Missing {selector} models:\n{log_message}\n")

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
