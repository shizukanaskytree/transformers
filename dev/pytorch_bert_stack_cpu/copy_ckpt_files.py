import os
import shutil

source_folder = "pretrained-bert-1-layer/checkpoint-1"
destination_folder = "pretrained-bert-2-layers/checkpoint-1-stack"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    print(f"Created destination folder: {destination_folder}")

# List all files in the source folder
files = os.listdir(source_folder)
print(f"Len of files: {len(files)}")

cnt = 0
# Iterate over the files and copy them to the destination folder
for file_name in files:
    # Exclude optimizer.pt and pytorch_model.bin
    if file_name not in ["optimizer.pt", "pytorch_model.bin"]:
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)
        shutil.copyfile(source_file, destination_file)
        print(f"Copied {file_name} to destination folder.")
        cnt += 1

print(f"{cnt} files copying completed.")
