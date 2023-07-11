import os

model_path = "pretrained-bert"

# Get the list of directories in the model_path/runs
runs_path = os.path.join(model_path, "runs")
dirs = [d for d in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, d))]

# Sort the directories by their creation time
sorted_dirs = sorted(dirs, key=lambda x: os.path.getctime(os.path.join(runs_path, x)))

# Get the latest folder path
latest_folder = sorted_dirs[-1]

print(f"latest_folder: {latest_folder}")
# Use the latest folder path to load the model or tokenizer
# model = BertForMaskedLM.from_pretrained(os.path.join(runs_path, latest_folder))
# tokenizer = BertTokenizerFast.from_pretrained(os.path.join(runs_path, latest_folder))

# Example usage
print(f"Latest folder: {latest_folder}")
print(f"Model path: {os.path.join(runs_path, latest_folder)}")
