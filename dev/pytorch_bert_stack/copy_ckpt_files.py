# dev/pytorch_bert_stack/copy_ckpt_files.py
import os
import shutil
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Copy checkpoint files from source to destination folder.')
parser.add_argument('--source_folder', required=True, help='Path to the source folder')
parser.add_argument('--destination_folder', required=True, help='Path to the destination folder')
args = parser.parse_args()

source_folder = args.source_folder
destination_folder = args.destination_folder

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
        print(f"Copied {file_name} to destination folder {destination_folder}.")
        cnt += 1

print(f"{cnt} files copying completed.")


# import os
# import shutil

# source_folder = "pretrained-bert-1-layer/checkpoint-68" ### XXX
# destination_folder = "pretrained-bert-2-layers/checkpoint-68-stack" ### XXX

# # Create the destination folder if it doesn't exist
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)
#     print(f"Created destination folder: {destination_folder}")

# # List all files in the source folder
# files = os.listdir(source_folder)
# print(f"Len of files: {len(files)}")

# cnt = 0
# # Iterate over the files and copy them to the destination folder
# for file_name in files:
#     # Exclude optimizer.pt and pytorch_model.bin
#     if file_name not in ["optimizer.pt", "pytorch_model.bin"]:
#         source_file = os.path.join(source_folder, file_name)
#         destination_file = os.path.join(destination_folder, file_name)
#         shutil.copyfile(source_file, destination_file)
#         print(f"Copied {file_name} to destination folder.")
#         cnt += 1

# print(f"{cnt} files copying completed.")
