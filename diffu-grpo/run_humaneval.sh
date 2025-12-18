#!/bin/bash

# Training script for HumanEval dataset
# This script demonstrates how to run GRPO training on HumanEval

export LOGDIR=checkpoints
mkdir -p $LOGDIR

DATASET="humaneval"
RUN_NAME=${DATASET}_base_bs12
MODEL_PATH=/shared/LLaDA-8B-Instruct  # Update this to your model path
NUM_ITER=12  # number of policy gradient inner updates iterations

# Note: HumanEval is a small dataset (164 examples), so adjust batch size accordingly
# You may want to use a smaller batch size or more iterations

accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port 12346 diffu_grpo_train.py \
    --config slurm_scripts/train.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME \
    --max_completion_length 512 \
    --block_length 64 \
    --diffusion_steps 128

