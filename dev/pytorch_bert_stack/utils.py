import os

def find_second_largest_checkpoint(src_ckpts_folder):
    """
    Input:
        src_ckpts_folder:
            pretrained-bert-1-layer

    example:
        num_second_largest_checkpoint, second_largest_checkpoint_path = \
            find_second_largest_checkpoint(args.src_ckpts_folder)

        num_second_largest_checkpoint: 4
        second_largest_checkpoint_path: pretrained-bert-1-layer/checkpoint-4
    """
    checkpoints = [filename for filename in os.listdir(src_ckpts_folder) if filename.startswith("checkpoint-")]
    if len(checkpoints) < 2:
        raise ValueError("At least two checkpoints are required in the source folder.")

    checkpoints = [int(checkpoint.split("-")[1]) for checkpoint in checkpoints]
    checkpoints.sort(reverse=True)
    num_second_largest_checkpoint = checkpoints[1]
    # the folder path to the second largest checkpoint
    second_largest_checkpoint_path = os.path.join(src_ckpts_folder, f"checkpoint-{num_second_largest_checkpoint}")
    return num_second_largest_checkpoint, second_largest_checkpoint_path
