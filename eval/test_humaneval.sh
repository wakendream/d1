#!/bin/bash

# Test script for HumanEval evaluation
# This script runs a simple evaluation on the HumanEval dataset using the base model

MODEL_PATH=${1:-"/shared/LLaDA-8B-Instruct"}  # Default model path, can be overridden
OUTPUT_DIR="eval_results/humaneval_test"
GEN_LENGTH=512  # Code generation typically needs longer sequences
BLOCK_LENGTH=64
DIFFUSION_STEPS=128
BATCH_SIZE=2  # Smaller batch size for code generation

echo "Running HumanEval evaluation..."
echo "Model path: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Generation length: $GEN_LENGTH"
echo "Batch size: $BATCH_SIZE"

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node 1 \
    --master_port 29411 \
    eval.py \
    --dataset humaneval \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --gen_length $GEN_LENGTH \
    --block_length $BLOCK_LENGTH \
    --diffusion_steps $DIFFUSION_STEPS \
    --output_dir "$OUTPUT_DIR" \
    --add_reasoning

echo "Evaluation completed! Results saved to $OUTPUT_DIR"

