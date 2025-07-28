#!/usr/bin/env python
"""Example script for training Lucky_ACT with task conditioning on multiple datasets.

This script demonstrates:
1. Loading multiple datasets (sim + real)
2. Adding task descriptions
3. Training with the recommended order (pretrain, multitask, fine-tune)
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.policies.lucky_act import LuckyACTConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.task_conditioned_dataset import (
    TaskConditionedDataset,
    create_task_conditioned_datasets,
)
from lerobot.policies.lucky_act import LuckyACTPolicy
# The train_policy utility is not available in the library, so we define a dummy here
# from lerobot.utils.train import train_policy
def train_policy(policy, dataloader, optimizer, steps):
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("outputs/lucky_act_multitask")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------
    # Step 1: Define datasets with task descriptions
    # ------------------------------------------------------------------
    dataset_configs = [
        # Simulated dataset (main sim2real target)
        {
            "repo_id": "your_username/sim_pick_cube",
            "task_description": "Pick up the red cube and place it in the blue bin (simulation)",
        },
        # Real-world datasets
        {
            "repo_id": "your_username/real_pick_cube_kitchen",
            "task_description": "Pick up the red cube and place it in the blue bin on kitchen counter",
        },
        {
            "repo_id": "your_username/real_pick_cube_lab",
            "task_description": "Pick up the red cube and place it in the blue bin on lab bench",
        },
        {
            "repo_id": "your_username/real_grasp_varied",
            "task_description": "Grasp various objects and move them to target locations",
        },
        {
            "repo_id": "your_username/real_stack_blocks",
            "task_description": "Stack colored blocks in the specified order",
        },
        {
            "repo_id": "your_username/real_pour_liquid",
            "task_description": "Pour liquid from bottle into cup without spilling",
        },
    ]
    
    # ------------------------------------------------------------------
    # Step 2: Configure Lucky_ACT with task conditioning
    # ------------------------------------------------------------------
    policy_config = LuckyACTConfig(
        # Task conditioning settings
        use_task_conditioning=True,
        task_encoder_type="sentence-transformer",  # or "clip" or "learned"
        task_encoder_model="sentence-transformers/all-MiniLM-L6-v2",
        use_adaln_task_context=True,
        use_task_token=True,
        task_token_position="prepend",
        
        # Standard ACT settings
        chunk_size=10,
        n_action_steps=10,
        
        # Training settings
        device=device,
    )
    
    # ------------------------------------------------------------------
    # Phase 1: Pretrain on simulation data
    # ------------------------------------------------------------------
    logger.info("Phase 1: Pretraining on simulation data...")
    
    # Load only sim dataset
    sim_dataset = TaskConditionedDataset(
        LeRobotDataset(dataset_configs[0]["repo_id"]),
        dataset_configs[0]["task_description"],
    )
    
    # Initialize policy
    policy = LuckyACTPolicy(
        config=policy_config,
        dataset_stats=sim_dataset.meta.stats,
    )
    
    # Create dataloader
    sim_loader = DataLoader(
        sim_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Pretrain
    pretrain_steps = 50000
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    
    policy.train()
    for step in range(pretrain_steps):
        batch = next(iter(sim_loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Add action_is_pad if not present (required by ACT)
        if "action_is_pad" not in batch:
            batch["action_is_pad"] = torch.zeros_like(batch["action"][..., 0], dtype=torch.bool)
        
        # Forward pass - returns loss and loss_dict
        loss, loss_dict = policy(batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 1000 == 0:
            logger.info(f"Pretrain step {step}/{pretrain_steps}, loss: {loss.item():.4f}")
    
    # Save pretrained checkpoint
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": policy_config,
    }, output_dir / "pretrained_sim.pt")
    
    # ------------------------------------------------------------------
    # Phase 2: Multitask training on all datasets
    # ------------------------------------------------------------------
    logger.info("Phase 2: Multitask training on all datasets...")
    
    # Create multi-dataset with task descriptions
    multitask_dataset = create_task_conditioned_datasets(dataset_configs)
    
    # Create dataloader with balanced sampling
    multitask_loader = DataLoader(
        multitask_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Continue training
    multitask_steps = 100000
    
    for step in range(multitask_steps):
        batch = next(iter(multitask_loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Add action_is_pad if not present (required by ACT)
        if "action_is_pad" not in batch:
            batch["action_is_pad"] = torch.zeros_like(batch["action"][..., 0], dtype=torch.bool)
        
        # Forward pass - returns loss and loss_dict
        loss, loss_dict = policy(batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 1000 == 0:
            logger.info(f"Multitask step {step}/{multitask_steps}, loss: {loss.item():.4f}")
    
    # Save multitask checkpoint
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": policy_config,
    }, output_dir / "multitask.pt")
    
    # ------------------------------------------------------------------
    # Phase 3: Fine-tune on main real dataset
    # ------------------------------------------------------------------
    logger.info("Phase 3: Fine-tuning on main real dataset...")
    
    # Choose main real dataset (e.g., kitchen environment)
    main_real_dataset = TaskConditionedDataset(
        LeRobotDataset(dataset_configs[1]["repo_id"]),
        dataset_configs[1]["task_description"],
    )
    
    main_real_loader = DataLoader(
        main_real_dataset,
        batch_size=16,  # Smaller batch for fine-tuning
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Lower learning rate for fine-tuning
    finetune_optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    finetune_steps = 10000
    
    for step in range(finetune_steps):
        batch = next(iter(main_real_loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Add action_is_pad if not present (required by ACT)
        if "action_is_pad" not in batch:
            batch["action_is_pad"] = torch.zeros_like(batch["action"][..., 0], dtype=torch.bool)
        
        # Forward pass - returns loss and loss_dict
        loss, loss_dict = policy(batch)
        
        # Backward pass
        finetune_optimizer.zero_grad()
        loss.backward()
        finetune_optimizer.step()
        
        if step % 500 == 0:
            logger.info(f"Fine-tune step {step}/{finetune_steps}, loss: {loss.item():.4f}")
    
    # Save final model
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": finetune_optimizer.state_dict(),
        "config": policy_config,
    }, output_dir / "final_finetuned.pt")
    
    logger.info("Training complete! Model saved to:", output_dir)
    
    # ------------------------------------------------------------------
    # Evaluation example
    # ------------------------------------------------------------------
    logger.info("Example evaluation with different task descriptions...")
    
    policy.eval()
    with torch.no_grad():
        # Create a dummy observation
        dummy_obs = {
            "observation.images.top": torch.randn(1, 3, 224, 224).to(device),
            "observation.state": torch.randn(1, 7).to(device),
        }
        
        # Test with different task descriptions
        test_tasks = [
            "Pick up the red cube and place it in the blue bin",
            "Stack the blocks in order: red, green, blue",
            "Pour water from the bottle into the cup",
        ]
        
        for task in test_tasks:
            dummy_obs["task_description"] = task
            action = policy.select_action(dummy_obs)
            logger.info(f"Task: '{task}' -> Action shape: {action.shape}")


if __name__ == "__main__":
    main() 