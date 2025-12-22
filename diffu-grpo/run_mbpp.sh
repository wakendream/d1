#!/bin/bash

# Training script for MBPP dataset
# This script demonstrates how to run GRPO training on MBPP
export HF_ENDPOINT=https://hf-mirror.com
export LOGDIR=checkpoints
mkdir -p $LOGDIR

export WANDB_INIT_TIMEOUT=300  # 5 minutes (default is 60 seconds)
export WANDB_CONSOLE=wrap
export WANDB_START_METHOD=thread

DATASET="mbpp"
RUN_NAME=${DATASET}_base_bs12
MODEL_PATH=/shared/LLaDA-8B-Instruct  # Update this to your model path
NUM_ITER=12  # number of policy gradient inner updates iterations

# Note: MBPP has a larger training set than HumanEval, so you can use larger batch sizes
# MBPP train split has ~374 examples, test split has ~90 examples

accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port 12347 diffu_grpo_train.py \
    --config slurm_scripts/train.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME \
    --max_completion_length 256 \
    --block_length 64 \
    --diffusion_steps 128

