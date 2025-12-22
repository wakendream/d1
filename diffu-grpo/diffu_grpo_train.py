import torch
import wandb
import os
import glob
import re
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl import TrlParser, ModelConfig
from peft import LoraConfig

# Custom imports
from diffu_grpo_trainer import DiffuGRPOTrainer
from diffu_grpo_config import DiffuGRPOConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
    reward_len,
    code_format_reward_func,
    code_extraction_reward_func,
    humaneval_correctness_reward_func,
    mbpp_correctness_reward_func,
)
from data_utils import (
    get_gsm8k_questions,
    get_countdown_questions,
    get_sudoku_questions,
    set_random_seed,
    get_math_questions,
    get_humaneval_questions,
    get_mbpp_questions,
)


def find_wandb_run_id_from_checkpoint(checkpoint_path, wandb_dir="wandb"):
    """
    Try to find the wandb run ID associated with a checkpoint.
    This searches for wandb runs that match the checkpoint's run_name AND step.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"  ✗ Checkpoint path does not exist: {checkpoint_path}")
        return None
    
    # Extract run_name and step from checkpoint path
    # Format: checkpoints/{run_name}/checkpoint-{step}
    checkpoint_parts = Path(checkpoint_path).parts
    run_name = None
    checkpoint_step = None
    
    for i, part in enumerate(checkpoint_parts):
        if part == "checkpoints" and i + 1 < len(checkpoint_parts):
            run_name = checkpoint_parts[i + 1]
            # Look for checkpoint-{step} in the path
            for j in range(i + 2, len(checkpoint_parts)):
                if checkpoint_parts[j].startswith("checkpoint-"):
                    try:
                        checkpoint_step = int(checkpoint_parts[j].split("-")[-1])
                        break
                    except ValueError:
                        pass
            break
    
    if not run_name:
        print(f"  ✗ Could not extract run_name from checkpoint path: {checkpoint_path}")
        print(f"    Path parts: {checkpoint_parts}")
        return None
    
    print(f"  Extracted run_name: {run_name}, checkpoint_step: {checkpoint_step}")
    
    # Search for wandb runs with matching run_name
    wandb_path = Path(wandb_dir)
    if not wandb_path.exists():
        print(f"  ✗ Wandb directory does not exist: {wandb_dir}")
        return None
    
    run_dirs = sorted(wandb_path.glob("run-*"), key=os.path.getmtime, reverse=True)
    print(f"  Found {len(run_dirs)} wandb run directories")
    
    # Collect all matching runs with their global_step
    matching_runs = []
    
    for run_dir in run_dirs:
        config_file = run_dir / "files" / "config.yaml"
        summary_file = run_dir / "files" / "wandb-summary.json"
        
        if not config_file.exists():
            continue
            
        try:
            # Check if run_name matches
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if f'value: {run_name}' not in content:
                    continue
            
            # Extract run ID
            run_id = run_dir.name.split('-')[-1]
            
            # Try to get global_step from summary file
            global_step = None
            if summary_file.exists():
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                        global_step = summary.get('train/global_step')
                except Exception as e:
                    print(f"  ⚠ Could not read summary from {run_dir.name}: {e}")
            
            matching_runs.append({
                'run_id': run_id,
                'run_dir': run_dir,
                'global_step': global_step,
                'mtime': os.path.getmtime(run_dir)
            })
            
        except Exception as e:
            continue
    
    if not matching_runs:
        print(f"  ✗ No matching wandb run found for run_name: {run_name}")
        return None
    
    # If we have a checkpoint_step, find the run with the closest global_step
    if checkpoint_step is not None:
        best_run = None
        best_diff = float('inf')
        
        for run_info in matching_runs:
            if run_info['global_step'] is not None:
                diff = abs(run_info['global_step'] - checkpoint_step)
                if diff < best_diff:
                    best_diff = diff
                    best_run = run_info
        
        if best_run and best_diff <= 10:  # Allow 10 steps difference
            print(f"  ✓ Found matching run: {best_run['run_dir'].name} (run_id: {best_run['run_id']}, global_step: {best_run['global_step']})")
            return best_run['run_id']
        elif best_run:
            print(f"  ⚠ Best match has global_step {best_run['global_step']}, but checkpoint is step {checkpoint_step} (diff: {best_diff})")
            print(f"    Using: {best_run['run_dir'].name} (run_id: {best_run['run_id']})")
            return best_run['run_id']
    
    # Fallback: return the most recent matching run
    best_run = max(matching_runs, key=lambda x: x['mtime'])
    print(f"  ✓ Using most recent matching run: {best_run['run_dir'].name} (run_id: {best_run['run_id']}, global_step: {best_run['global_step']})")
    return best_run['run_id']


def main(grpo_config, model_config):

    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)
    
    # Configure wandb resume if resuming from checkpoint
    if grpo_config.resume_from_checkpoint:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        wandb_dir = os.path.join(script_dir, "wandb")
        
        # Also try relative path from checkpoint
        checkpoint_dir = os.path.dirname(os.path.dirname(grpo_config.resume_from_checkpoint))
        wandb_dir_alt = os.path.join(checkpoint_dir, "..", "wandb")
        wandb_dir_alt = os.path.abspath(wandb_dir_alt)
        
        print(f"Looking for wandb run ID from checkpoint: {grpo_config.resume_from_checkpoint}")
        print(f"Searching in wandb directory: {wandb_dir}")
        print(f"Alternative wandb directory: {wandb_dir_alt}")
        
        # Try primary directory first
        wandb_run_id = find_wandb_run_id_from_checkpoint(
            grpo_config.resume_from_checkpoint,
            wandb_dir=wandb_dir
        )
        
        # If not found, try alternative directory
        if not wandb_run_id and os.path.exists(wandb_dir_alt):
            print(f"Trying alternative wandb directory: {wandb_dir_alt}")
            wandb_run_id = find_wandb_run_id_from_checkpoint(
                grpo_config.resume_from_checkpoint,
                wandb_dir=wandb_dir_alt
            )
        
        if wandb_run_id:
            os.environ["WANDB_RESUME"] = "allow"
            os.environ["WANDB_RUN_ID"] = wandb_run_id
            print(f"✓ Found wandb run ID: {wandb_run_id}")
            print(f"✓ Resuming wandb run: {wandb_run_id}")
        else:
            print("⚠ Warning: Could not find wandb run ID for checkpoint. Starting new run.")
            print(f"  Checkpoint path: {grpo_config.resume_from_checkpoint}")
            print(f"  Primary wandb directory exists: {os.path.exists(wandb_dir)}")
            print(f"  Alternative wandb directory exists: {os.path.exists(wandb_dir_alt)}")

    # Load dataset based on configuration
    if grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif grpo_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    elif grpo_config.dataset == "sudoku":
        dataset = get_sudoku_questions()
        reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "math":
        dataset = get_math_questions("train")
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]
    elif grpo_config.dataset == "humaneval":
        dataset = get_humaneval_questions("test")  # HumanEval only has test split
        reward_functions = [
            code_format_reward_func,
            code_extraction_reward_func,
            humaneval_correctness_reward_func,
        ]
    elif grpo_config.dataset == "mbpp":
        dataset = get_mbpp_questions("train")  # MBPP has train and test splits
        reward_functions = [
            code_format_reward_func,
            code_extraction_reward_func,
            mbpp_correctness_reward_func,
        ]
    else:
        raise ValueError(f"Unknown dataset: {grpo_config.dataset}")

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=grpo_config.seed)

    # Split dataset if needed
    if grpo_config.dataset in ["countdown", "sudoku"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    elif grpo_config.dataset == "humaneval":
        # For HumanEval, we can use all data for training since it's a small dataset
        # or split if needed (e.g., use first 100 for training, rest for eval)
        train_set = dataset  # Use all data for training
    elif grpo_config.dataset == "mbpp":
        # For MBPP, use train split for training (already loaded as train split)
        train_set = dataset  # Use all training data
    else:
        train_set = dataset

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        grpo_config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )
    # Initialize and run trainer
    trainer = DiffuGRPOTrainer(
        args=grpo_config,
        model=model,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_set,
    )

    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
