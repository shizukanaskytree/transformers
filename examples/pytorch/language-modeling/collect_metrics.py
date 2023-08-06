import argparse
import os
import re

from tabulate import tabulate


def get_cola_metric(filename):
    with open(filename, "r") as f:
        content = f.read()

    match = re.search(r'eval_matthews_correlation\s*=\s*([-\d.]+)', content)
    if match:
        matthews_correlation = float(match.group(1)) * 100
        # print(f"eval_matthews_correlation: {matthews_correlation}")

    return {"Matthews_corr": matthews_correlation}


def get_sst2_metric(filename):
    with open(filename, "r") as f:
        content = f.read()

    # Define the patterns
    accuracy_pattern = r"eval_accuracy\s*=\s*(\d+\.\d+)"

    # Search for the patterns
    accuracy_match = re.search(accuracy_pattern, content)

    # Extract the values
    accuracy = float(accuracy_match.group(1)) * 100 if accuracy_match else None

    # print(f"Accuracy: {accuracy}")

    return {"Accuracy": accuracy}


def get_mrpc_metric(filename):
    with open(filename, "r") as f:
        content = f.read()

    # Define the patterns
    accuracy_pattern = r"eval_accuracy\s*=\s*(\d+\.\d+)"
    f1_pattern = r"eval_f1\s*=\s*(\d+\.\d+)"

    # Search for the patterns
    accuracy_match = re.search(accuracy_pattern, content)
    f1_match = re.search(f1_pattern, content)

    # Extract the values
    accuracy = float(accuracy_match.group(1)) * 100 if accuracy_match else None
    f1 = float(f1_match.group(1)) * 100 if f1_match else None

    # print(f"Accuracy: {accuracy}")
    # print(f"F1: {f1}")

    return {"Accuracy": accuracy, "F1": f1}


def get_stsb_metric(filename):
    with open(filename, "r") as f:
        content = f.read()

    pearson_pattern = r"eval_pearson\s*=\s*([0-9.]+)"
    spearman_pattern = r"eval_spearmanr\s*=\s*([0-9.]+)"

    pearson_match = re.search(pearson_pattern, content)
    spearman_match = re.search(spearman_pattern, content)

    if pearson_match:
        pearson = float(pearson_match.group(1)) * 100
        # print(f"pearson: {pearson}")

    if spearman_match:
        spearman = float(spearman_match.group(1)) * 100
        # print(f"spearman: {spearman}")

    return {"Pearson_corr": pearson, "Spearman_corr": spearman}


def get_qqp_metric(filename):
    with open(filename, "r") as f:
        content = f.read()

    # Define the patterns
    accuracy_pattern = r"eval_accuracy\s*=\s*(\d+\.\d+)"
    f1_pattern = r"eval_f1\s*=\s*(\d+\.\d+)"

    # Search for the patterns
    accuracy_match = re.search(accuracy_pattern, content)
    f1_match = re.search(f1_pattern, content)

    # Extract the values
    accuracy = float(accuracy_match.group(1)) * 100 if accuracy_match else None
    f1 = float(f1_match.group(1)) * 100 if f1_match else None

    # print(f"Accuracy: {accuracy}")
    # print(f"F1 Score: {f1}")
    return {"Accuracy": accuracy, "F1": f1}


def get_mnli_metric(filename):
    with open(filename, "r") as f:
        content = f.read()

    # regex pattern for eval_accuracy
    pattern_accuracy = r"eval_accuracy\s*=\s*(\d+\.\d+)"
    # regex pattern for eval_accuracy_mm
    pattern_accuracy_mm = r"eval_accuracy_mm\s*=\s*(\d+\.\d+)"

    # find matches
    match_accuracy = re.search(pattern_accuracy, content)
    match_accuracy_mm = re.search(pattern_accuracy_mm, content)

    if match_accuracy:
        # print("eval_accuracy:", match_accuracy.group(1))
        match_accuracy = float(match_accuracy.group(1)) * 100
        # print(f"match_accuracy: {match_accuracy}")

    if match_accuracy_mm:
        # print("eval_accuracy_mm:", match_accuracy_mm.group(1))
        match_accuracy_mm = float(match_accuracy_mm.group(1)) * 100
        # print(f"match_accuracy_mm: {match_accuracy_mm}")

    return {"Matched_acc": match_accuracy, "Mismatched_acc": match_accuracy_mm}


def get_qnli_metric(filename):
    with open(filename, "r") as f:
        content = f.read()

    # Define the patterns
    accuracy_pattern = r"eval_accuracy\s*=\s*(\d+\.\d+)"

    # Search for the patterns
    accuracy_match = re.search(accuracy_pattern, content)

    # Extract the values
    accuracy = float(accuracy_match.group(1)) * 100 if accuracy_match else None

    # print(f"Accuracy: {accuracy}")
    return {"Accuracy": accuracy}


def get_rte_metric(filename):
    with open(filename, "r") as f:
        content = f.read()

    # Define the patterns
    accuracy_pattern = r"eval_accuracy\s*=\s*(\d+\.\d+)"

    # Search for the patterns
    accuracy_match = re.search(accuracy_pattern, content)

    # Extract the values
    accuracy = float(accuracy_match.group(1)) * 100 if accuracy_match else None

    # print(f"Accuracy: {accuracy}")
    return {"Accuracy": accuracy}


def get_wnli_metric(filename):
    with open(filename, "r") as f:
        content = f.read()

    # Define the patterns
    accuracy_pattern = r"eval_accuracy\s*=\s*(\d+\.\d+)"

    # Search for the patterns
    accuracy_match = re.search(accuracy_pattern, content)

    # Extract the values
    accuracy = float(accuracy_match.group(1)) * 100 if accuracy_match else None

    # print(f"Accuracy: {accuracy}")
    return {"Accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Get list of log files in a directory.")
    parser.add_argument("--log_dir", type=str, help="Path to the directory containing log files. e.g., '/home/xiaofeng.wu/prjs/transformers/dev/eval_bert/logs/20230728_080403'")
    args = parser.parse_args()
    log_dir = args.log_dir
    log_files = os.listdir(log_dir)
    # print("List of log files in the directory:", log_files)

    glue_tasks = {
        "CoLA": "cola_glue_stacked_bert",
        "SST-2": "sst2_glue_stacked_bert",
        "MRPC": "mrpc_glue_stacked_bert",
        "STS-B": "stsb_glue_stacked_bert",
        "QQP": "qqp_glue_stacked_bert",
        "MNLI": "mnli_glue_stacked_bert",
        "QNLI": "qnli_glue_stacked_bert",
        "RTE": "rte_glue_stacked_bert",
        "WNLI": "wnli_glue_stacked_bert"
    }

    ### Evaluation results
    ### https://huggingface.co/bert-base-uncased
    ###
    ### | Task | MNLI-(m/mm) | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | Average |
    ### |------|-------------|-----|------|-------|------|-------|------|-----|---------|
    ### |      | 84.6/83.4   | 71.2| 90.5 | 93.5  | 52.1 | 85.8  | 88.9 | 66.4| 79.6    |

    ### Define the metrics table
    baseline_metrics_table = {
        'MNLI': {'Matched_acc': 84.6, "Mismatched_acc": 83.4},
        'QQP': {'Accuracy': 71.2},
        'QNLI': {'Accuracy': 90.5},
        'SST-2': {'Accuracy': 93.5},
        'CoLA': {'Matthews_corr': 52.1},
        'STS-B': {'Pearson_corr': 85.8},
        'MRPC': {'F1': 88.9},
        'RTE': {'Accuracy': 66.4},
        'WNLI': {'Accuracy': 56.34},  # No new data for WNLI, so keeping the old value
    }

    funcs = {
        "CoLA": get_cola_metric,
        "SST-2": get_sst2_metric,
        "MRPC": get_mrpc_metric,
        "STS-B": get_stsb_metric,
        "QQP": get_qqp_metric,
        "MNLI": get_mnli_metric,
        "QNLI": get_qnli_metric,
        "RTE": get_rte_metric,
        "WNLI": get_wnli_metric
    }

    # Initialize an empty list to store the rows of the table
    table = []

    for task_name, log_prefix in glue_tasks.items():
        # print(f"Task: {task_name}")
        log_file = f"{log_dir}{log_prefix}_*.log"
        matching_logs = [f for f in log_files if f.startswith(log_prefix)]
        if len(matching_logs) > 0:
            filename = os.path.join(log_dir, matching_logs[-1])
            if task_name in funcs:
                metrics = funcs[task_name](filename)
                # print(metrics)
                # Add the task name and metrics to the table
                for metric, value in metrics.items():
                    baseline = baseline_metrics_table[task_name].get(metric, None)
                    if baseline is not None and value is not None:
                        diff = round(value - baseline, 2)
                    else:
                        diff = "N/A"
                    table.append([task_name, metric, value, baseline, diff])

    # Print the table
    print(tabulate(table, headers=["Task", "Metric", "Value (%)", "Baseline (%)", "Difference (%)"]))

if __name__ == "__main__":
    main()
