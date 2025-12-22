#!/usr/bin/env python3
"""
Test script to verify checkpoint resume and wandb run ID lookup.
This script does not require GPU and can be run quickly.
"""

import os
import sys
from pathlib import Path

# Add the current directory to path to import the function
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffu_grpo_train import find_wandb_run_id_from_checkpoint


def test_checkpoint_resume(checkpoint_path=None):
    """
    Test checkpoint resume functionality.
    
    Args:
        checkpoint_path: Path to checkpoint. If None, uses default from train.yaml
    """
    # Default checkpoint path
    if checkpoint_path is None:
        checkpoint_path = "/root/d1/diffu-grpo/checkpoints/mbpp_base_bs12/checkpoint-2200"
    
    print("=" * 80)
    print("Testing Checkpoint Resume and WandB Run ID Lookup")
    print("=" * 80)
    print()
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint path does not exist: {checkpoint_path}")
        return False
    
    print(f"✓ Checkpoint path exists: {checkpoint_path}")
    print()
    
    # Get wandb directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wandb_dir = os.path.join(script_dir, "wandb")
    
    print(f"WandB directory: {wandb_dir}")
    print(f"WandB directory exists: {os.path.exists(wandb_dir)}")
    print()
    
    # Test the function
    print("Calling find_wandb_run_id_from_checkpoint...")
    print("-" * 80)
    
    wandb_run_id = find_wandb_run_id_from_checkpoint(
        checkpoint_path,
        wandb_dir=wandb_dir
    )
    
    print("-" * 80)
    print()
    
    if wandb_run_id:
        print("=" * 80)
        print(f"✅ SUCCESS: Found wandb run ID: {wandb_run_id}")
        print("=" * 80)
        print()
        print("Environment variables that would be set:")
        print(f"  WANDB_RESUME=allow")
        print(f"  WANDB_RUN_ID={wandb_run_id}")
        return True
    else:
        print("=" * 80)
        print("❌ FAILED: Could not find wandb run ID")
        print("=" * 80)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test checkpoint resume functionality")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (default: /root/d1/diffu-grpo/checkpoints/mbpp_base_bs12/checkpoint-2200)"
    )
    
    args = parser.parse_args()
    
    success = test_checkpoint_resume(args.checkpoint)
    sys.exit(0 if success else 1)

