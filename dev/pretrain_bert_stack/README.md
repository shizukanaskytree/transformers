# PyTorch BERT Stack CPU

This repository contains Python scripts for training and evaluating a BERT-based stack model using PyTorch.

## Files (in order of execution)

- `run-steps.sh`: Main entry script to call and run other Python scripts.
- `pretrain_1_layer.py`: Script for pretraining a single-layer stack model.
- `copy_ckpt_files.py`: Script to copy checkpoint files.
- `stack_model.py`: Stack model implementation.
- `stack_optim.py`: Stack model optimizer implementation.
- `load_training_args.py`: Script to load training arguments.
- `pretrain_2_layers.py`: Script for pretraining a 2-layer stack model.
- `run-stack.sh`: Shell script to run the stack model.
- `compare_stack_original_2_layers.py`: Script to compare the original stack model with a 2-layer stack model.
- `tests/`: Directory containing test scripts.
- `logs/`: Directory for storing log files.
- `training.log`: Log file for training process.

## Running the Scripts

To run the Python scripts, execute the `run-steps.sh` script, which will call and run the necessary scripts in the correct order.
