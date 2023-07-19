import os
import shutil
import argparse

from utils import find_second_largest_checkpoint

def main():
    parser = argparse.ArgumentParser(description='Copy checkpoint files from source to destination folder. e.g., pretrained-bert-1-layer')
    parser.add_argument('--src_ckpts_folder', required=True, help='Path to the source folder')
    parser.add_argument('--dest_ckpt_folder', required=True, help='Path to the destination folder')
    args = parser.parse_args()

    num_second_largest_checkpoint, second_largest_checkpoint_path = find_second_largest_checkpoint(args.src_ckpts_folder)
    # print(f"num_second_largest_checkpoint: {num_second_largest_checkpoint}") ### 4
    # print(f"second_largest_checkpoint_path: {second_largest_checkpoint_path}") ### pretrained-bert-1-layer/checkpoint-4

    # infer the destination folder
    dest_checkpoint_path = os.path.join(args.dest_ckpt_folder, f"checkpoint-{num_second_largest_checkpoint}-stacked")
    # print(f"dest_checkpoint_path: {dest_checkpoint_path}")

    ### Create the destination folder if it doesn't exist
    if not os.path.exists(dest_checkpoint_path):
        os.makedirs(dest_checkpoint_path)
        print(f"Created destination folder: {dest_checkpoint_path}")

    # List all files in the source folder
    files = os.listdir(second_largest_checkpoint_path)
    print(f"Len of files: {len(files)}")

    cnt = 0
    # Iterate over the files and copy them to the destination folder
    for file_name in files:
        # Exclude optimizer.pt and pytorch_model.bin
        if file_name not in ["optimizer.pt", "pytorch_model.bin"]:
            source_file = os.path.join(second_largest_checkpoint_path, file_name)
            destination_file = os.path.join(dest_checkpoint_path, file_name)
            shutil.copyfile(source_file, destination_file)
            print(f"Copied {file_name} to destination folder {dest_checkpoint_path}.")
            cnt += 1

    print(f"{cnt} files copying completed.")

if __name__ == "__main__":
    main()
