# Task Conditioning in Lucky_ACT

This document describes the task conditioning features added to Lucky_ACT for multitask learning and sim2real transfer.

## Overview

Lucky_ACT now supports task conditioning through natural language descriptions, enabling:
- **Multitask learning** across different tasks and domains
- **Sim2real transfer** with better generalization
- **Zero-shot task switching** at inference time

The implementation combines two powerful techniques:
1. **Task Tokens**: Task descriptions are encoded as embeddings and prepended to the transformer input sequence
2. **AdaLayerNorm**: Task embeddings modulate layer normalization throughout the network

## Features

### Task Encoders

Three types of task encoders are supported:

1. **Sentence-BERT** (default)
   - Pre-trained language model for rich semantic understanding
   - Best for natural language task descriptions
   - Model: `sentence-transformers/all-MiniLM-L6-v2`

2. **CLIP Text Encoder**
   - Vision-language aligned embeddings
   - Good for tasks with visual descriptions
   - Model: `openai/clip-vit-base-patch32`

3. **Learned Embeddings**
   - Trainable task embeddings
   - Best for fixed task vocabularies
   - Configurable vocabulary size

### Architecture Changes

1. **Task Token Integration**
   - Task embeddings are projected to model dimension
   - Added as special tokens to transformer input
   - Configurable position (prepend/append)

2. **Adaptive Layer Normalization**
   - Standard LayerNorm replaced with AdaLayerNorm
   - Task embedding modulates scale and shift
   - Applied throughout encoder/decoder layers

## Configuration

Enable task conditioning in your config:

```python
from lerobot.policies.lucky_act import LuckyACTConfig

config = LuckyACTConfig(
    # Enable task conditioning
    use_task_conditioning=True,
    
    # Task encoder settings
    task_encoder_type="sentence-transformer",  # or "clip", "learned"
    task_encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    
    # Architecture options
    use_task_token=True,           # Add task as transformer token
    use_adaln_task_context=True,   # Use AdaLayerNorm
    task_token_position="prepend", # or "append"
    
    # Other standard ACT settings...
)
```

## Dataset Setup

### Single Dataset with Task Description

```python
from lerobot.datasets.task_conditioned_dataset import add_task_descriptions_to_dataset

# Simple: one description for all samples
dataset = add_task_descriptions_to_dataset(
    your_dataset,
    "Pick up the red cube and place it in the bin"
)

# Advanced: different descriptions per episode
episode_tasks = {
    0: "Pick up the red cube",
    1: "Pick up the blue cube",
    2: "Stack the cubes",
}
dataset = add_task_descriptions_to_dataset(your_dataset, episode_tasks)
```

### Multiple Datasets (Multitask/Sim2Real)

```python
from lerobot.datasets.task_conditioned_dataset import create_task_conditioned_datasets

configs = [
    {
        "repo_id": "user/sim_dataset",
        "task_description": "Pick up cube in simulation",
    },
    {
        "repo_id": "user/real_dataset_1",
        "task_description": "Pick up cube on kitchen table",
    },
    {
        "repo_id": "user/real_dataset_2",
        "task_description": "Pick up cube on lab bench",
    },
]

dataset = create_task_conditioned_datasets(configs)
```

## Training Strategy

For sim2real transfer with multiple datasets:

### 1. Pretrain on Simulation
```python
# Train on sim data first to learn general skills
sim_dataset = TaskConditionedDataset(sim_data, "Task in simulation")
# Train for N epochs...
```

### 2. Multitask Training
```python
# Mix all datasets with their task descriptions
all_datasets = create_task_conditioned_datasets(dataset_configs)
# Continue training on mixed data...
```

### 3. Fine-tune on Target Domain
```python
# Optional: specialize on main deployment environment
target_dataset = TaskConditionedDataset(real_data, "Task in target domain")
# Fine-tune with lower learning rate...
```

## Inference

At inference time, simply provide the task description:

```python
# The policy will encode the description automatically
observation["task_description"] = "Pick up the red cube"
action = policy.select_action(observation)

# Switch tasks on the fly
observation["task_description"] = "Stack the blue blocks"
action = policy.select_action(observation)
```

## Implementation Details

### Task Encoder Module (`task_encoder.py`)
- Abstract `TaskEncoder` base class
- Concrete implementations with caching
- Factory function `create_task_encoder()`

### Lucky_ACT Core (`lucky_act_core.py`)
- `TaskConditionedTransformerEncoderLayer`: Custom layer with AdaLayerNorm
- `TaskConditionedTransformerEncoder`: Encoder that passes task context
- Task token projection and positional embedding

### Policy Updates (`modeling_lucky_act.py`)
- `_maybe_compute_task_embedding()`: Converts descriptions to embeddings
- Automatic injection before forward passes
- Handles various input formats (string, list, tensor)

### Dataset Wrapper (`task_conditioned_dataset.py`)
- `TaskConditionedDataset`: Adds task descriptions to samples
- Supports single/multiple datasets
- Flexible task assignment strategies

## Best Practices

1. **Task Description Guidelines**
   - Be consistent across similar tasks
   - Include relevant context (e.g., "in simulation", "on table")
   - Keep descriptions concise but informative

2. **Dataset Balancing**
   - Consider upsampling smaller datasets
   - Monitor per-dataset performance during training
   - Use appropriate batch sizes for each phase

3. **Hyperparameter Tuning**
   - Adjust learning rates between phases
   - Consider different optimizers for fine-tuning
   - Monitor task embedding space with visualization

4. **Evaluation**
   - Test on held-out tasks to measure generalization
   - Visualize task embeddings with t-SNE/UMAP
   - Track per-task metrics during training

## Example: Complete Sim2Real Pipeline

See `examples/train_lucky_act_multitask.py` for a complete example including:
- Dataset setup with 6 datasets (1 sim, 5 real)
- Three-phase training strategy
- Evaluation with different task descriptions
- Checkpointing and model saving

## Troubleshooting

### "Task conditioning enabled but no task_embedding found"
- Ensure your dataset provides `task_description` field
- Check that `use_task_conditioning=True` in config

### High memory usage
- Reduce `task_encoder_cache_size` if using many unique descriptions
- Consider using learned embeddings for fixed task sets
- Use gradient checkpointing for large models

### Poor task discrimination
- Visualize task embeddings to check separation
- Try different task encoder types
- Ensure task descriptions are sufficiently distinct

## Citation

If you use task-conditioned Lucky_ACT in your research, please cite:
```bibtex
@software{lucky_act_task_2024,
  title={Task-Conditioned Lucky_ACT for Multitask and Sim2Real Robot Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/lerobot}
}
``` 