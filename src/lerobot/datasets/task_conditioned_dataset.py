"""Task-conditioned dataset wrapper for Lucky_ACT.

This module provides utilities to add task descriptions to dataset samples,
enabling task-conditioned training with multiple datasets.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


class TaskConditionedDataset(Dataset):
    """Wrapper that adds task descriptions to dataset samples.
    
    This wrapper can handle:
    1. Single dataset with one task description
    2. Multiple datasets with different task descriptions
    3. Custom task descriptions per episode or sample
    """
    
    def __init__(
        self,
        dataset: Union[LeRobotDataset, List[LeRobotDataset]],
        task_descriptions: Union[str, List[str], Dict[int, str]],
        default_description: str = "Perform the task",
    ):
        """
        Args:
            dataset: Single dataset or list of datasets
            task_descriptions: Task descriptions in various formats:
                - str: Single description for all samples
                - List[str]: One description per dataset (for multi-dataset)
                - Dict[int, str]: Mapping from episode_index to description
            default_description: Fallback description if none provided
        """
        self.datasets = [dataset] if isinstance(dataset, LeRobotDataset) else dataset
        self.default_description = default_description
        
        # Process task descriptions
        if isinstance(task_descriptions, str):
            # Single description for all
            self._task_map = lambda dataset_idx, episode_idx: task_descriptions
        
        elif isinstance(task_descriptions, list):
            if len(self.datasets) == 1:
                # List of descriptions for single dataset - assume one per episode
                episode_descriptions = {i: desc for i, desc in enumerate(task_descriptions)}
                self._task_map = lambda dataset_idx, episode_idx: episode_descriptions.get(
                    episode_idx, self.default_description
                )
            else:
                # One description per dataset
                if len(task_descriptions) != len(self.datasets):
                    raise ValueError(
                        f"Number of task descriptions ({len(task_descriptions)}) "
                        f"must match number of datasets ({len(self.datasets)})"
                    )
                dataset_descriptions = task_descriptions
                self._task_map = lambda dataset_idx, episode_idx: dataset_descriptions[dataset_idx]
        
        elif isinstance(task_descriptions, dict):
            # Direct episode mapping
            self._task_map = lambda dataset_idx, episode_idx: task_descriptions.get(
                episode_idx, self.default_description
            )
        
        else:
            raise ValueError(
                f"task_descriptions must be str, list, or dict, got {type(task_descriptions)}"
            )
        
        # Calculate cumulative dataset sizes for multi-dataset indexing
        self._cumulative_sizes = []
        cumsum = 0
        for ds in self.datasets:
            cumsum += len(ds)
            self._cumulative_sizes.append(cumsum)
        
        logger.info(
            f"Created TaskConditionedDataset with {len(self.datasets)} dataset(s), "
            f"total size: {self._cumulative_sizes[-1]}"
        )
    
    def __len__(self) -> int:
        """Total number of samples across all datasets."""
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0
    
    def __getitem__(self, idx: int) -> Dict:
        """Get sample with added task_description field."""
        # Find which dataset this index belongs to
        dataset_idx = 0
        local_idx = idx
        
        for i, cumsize in enumerate(self._cumulative_sizes):
            if idx < cumsize:
                dataset_idx = i
                if i > 0:
                    local_idx = idx - self._cumulative_sizes[i - 1]
                break
        
        # Get the sample from the appropriate dataset
        sample = self.datasets[dataset_idx][local_idx]
        
        # Extract episode index from sample
        episode_idx = sample.get("episode_index", 0)
        if isinstance(episode_idx, torch.Tensor):
            episode_idx = episode_idx.item()
        
        # Add task description
        sample["task_description"] = self._task_map(dataset_idx, episode_idx)
        
        return sample
    
    @property
    def features(self):
        """Return features from the first dataset."""
        return self.datasets[0].features if self.datasets else {}
    
    @property
    def meta(self):
        """Return metadata from the first dataset."""
        return self.datasets[0].meta if self.datasets else None


def create_task_conditioned_datasets(
    dataset_configs: List[Dict],
    shared_transforms: Optional[callable] = None,
) -> TaskConditionedDataset:
    """Create a multi-dataset task-conditioned dataset from configurations.
    
    Args:
        dataset_configs: List of dataset configurations, each containing:
            - repo_id: Dataset repository ID
            - task_description: Task description string
            - root: Optional root directory
            - transform: Optional dataset-specific transform
        shared_transforms: Transform to apply to all datasets
    
    Returns:
        TaskConditionedDataset wrapping all specified datasets
    
    Example:
        configs = [
            {
                "repo_id": "user/sim_dataset",
                "task_description": "Pick up the red cube in simulation",
            },
            {
                "repo_id": "user/real_dataset_1", 
                "task_description": "Pick up the red cube on the table",
            },
            {
                "repo_id": "user/real_dataset_2",
                "task_description": "Place the red cube in the basket",
            },
        ]
        dataset = create_task_conditioned_datasets(configs)
    """
    datasets = []
    task_descriptions = []
    
    for config in dataset_configs:
        # Load dataset
        dataset = LeRobotDataset(
            repo_id=config["repo_id"],
            root=config.get("root"),
            transform=config.get("transform", shared_transforms),
        )
        datasets.append(dataset)
        
        # Extract task description
        task_desc = config.get("task_description", f"Task from {config['repo_id']}")
        task_descriptions.append(task_desc)
        
        logger.info(f"Loaded dataset {config['repo_id']} with task: {task_desc}")
    
    return TaskConditionedDataset(datasets, task_descriptions)


# Convenience function for single dataset
def add_task_descriptions_to_dataset(
    dataset: LeRobotDataset,
    task_description: Union[str, Dict[int, str]] = "Perform the task",
) -> TaskConditionedDataset:
    """Add task descriptions to a single dataset.
    
    Args:
        dataset: LeRobotDataset to wrap
        task_description: Either a single string or episode->description mapping
    
    Returns:
        TaskConditionedDataset with task descriptions added
    """
    return TaskConditionedDataset(dataset, task_description) 