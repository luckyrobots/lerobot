"""Unit tests for Lucky_ACT task conditioning functionality."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.policies.lucky_act import LuckyACTConfig, LuckyACTPolicy
from lerobot.policies.lucky_act.task_encoder import create_task_encoder
from lerobot.policies.adaptive_layer_norm import AdaLayerNorm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.task_conditioned_dataset import (
    TaskConditionedDataset,
    add_task_descriptions_to_dataset,
)


class TestTaskEncoder:
    """Test task encoder functionality."""
    
    def test_sentence_transformer_encoder(self):
        """Test sentence transformer encoder."""
        encoder = create_task_encoder("sentence-transformer")
        
        # Single string
        embedding = encoder.encode("Pick up the red cube")
        assert embedding.shape == (1, encoder.get_embedding_dim())
        
        # List of strings
        embeddings = encoder.encode(["Task 1", "Task 2", "Task 3"])
        assert embeddings.shape == (3, encoder.get_embedding_dim())
        
        # Test caching
        embedding1 = encoder.encode("Same task")
        embedding2 = encoder.encode("Same task")
        assert torch.allclose(embedding1, embedding2)
    
    def test_clip_encoder(self):
        """Test CLIP text encoder."""
        encoder = create_task_encoder("clip")
        
        embedding = encoder.encode("Grasp the object")
        assert embedding.shape == (1, encoder.get_embedding_dim())
        
        # Test normalization
        norm = torch.norm(embedding, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)
    
    def test_learned_encoder(self):
        """Test learned encoder with vocabulary."""
        encoder = create_task_encoder("learned", vocab_size=10, embedding_dim=64)
        
        # First task gets ID 0
        embedding1 = encoder.encode("Task A")
        assert embedding1.shape == (1, 64)
        
        # Same task gets same embedding
        embedding2 = encoder.encode("Task A")
        assert torch.allclose(embedding1, embedding2)
        
        # Different task gets different ID
        embedding3 = encoder.encode("Task B")
        assert not torch.allclose(embedding1, embedding3)
        
        # Test vocabulary limit
        for i in range(8):
            encoder.encode(f"Task {i}")
        
        # This should raise since vocab is full
        with pytest.raises(ValueError, match="vocabulary full"):
            encoder.encode("One more task")


class TestAdaLayerNorm:
    """Test AdaLayerNorm functionality."""
    
    def test_adaln_modulation(self):
        """Test that AdaLayerNorm properly modulates based on context."""
        hidden_size = 256
        context_dim = 128
        batch_size = 4
        seq_len = 10
        
        adaln = AdaLayerNorm(hidden_size, context_dim)
        
        # Create inputs
        x = torch.randn(batch_size, seq_len, hidden_size)
        context = torch.randn(batch_size, context_dim)
        
        # Forward pass
        output = adaln(x, context)
        assert output.shape == x.shape
        
        # Test that different contexts produce different outputs
        context2 = torch.randn(batch_size, context_dim)
        output2 = adaln(x, context2)
        assert not torch.allclose(output, output2)
        
        # Test initialization (should be close to vanilla LayerNorm at start)
        adaln_fresh = AdaLayerNorm(hidden_size, context_dim)
        vanilla_ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        with torch.no_grad():
            adaln_out = adaln_fresh(x, torch.zeros(batch_size, context_dim))
            vanilla_out = vanilla_ln(x)
            assert torch.allclose(adaln_out, vanilla_out, atol=1e-5)


class TestTaskConditionedDataset:
    """Test task-conditioned dataset wrapper."""
    
    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a mock LeRobotDataset."""
        # This is a simplified mock - in real tests you'd use actual dataset
        class MockDataset:
            def __init__(self, size=100):
                self.size = size
                self.features = {"observation.image": {"shape": [3, 224, 224]}}
                self.meta = type('obj', (object,), {'stats': {}})
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return {
                    "observation.image": torch.randn(3, 224, 224),
                    "action": torch.randn(7),
                    "episode_index": idx // 10,  # 10 frames per episode
                }
        
        return MockDataset()
    
    def test_single_dataset_single_description(self, mock_dataset):
        """Test wrapper with single dataset and description."""
        dataset = TaskConditionedDataset(mock_dataset, "Pick up the cube")
        
        assert len(dataset) == len(mock_dataset)
        
        sample = dataset[0]
        assert "task_description" in sample
        assert sample["task_description"] == "Pick up the cube"
    
    def test_multiple_datasets(self, mock_dataset):
        """Test wrapper with multiple datasets."""
        datasets = [mock_dataset, mock_dataset, mock_dataset]
        descriptions = ["Task A", "Task B", "Task C"]
        
        dataset = TaskConditionedDataset(datasets, descriptions)
        
        assert len(dataset) == 300  # 3 datasets of 100 samples each
        
        # Check that descriptions are assigned correctly
        sample1 = dataset[50]  # From first dataset
        assert sample1["task_description"] == "Task A"
        
        sample2 = dataset[150]  # From second dataset
        assert sample2["task_description"] == "Task B"
        
        sample3 = dataset[250]  # From third dataset
        assert sample3["task_description"] == "Task C"
    
    def test_episode_mapping(self, mock_dataset):
        """Test episode-specific task descriptions."""
        episode_descriptions = {
            0: "Episode 0 task",
            1: "Episode 1 task",
            2: "Episode 2 task",
        }
        
        dataset = TaskConditionedDataset(
            mock_dataset,
            episode_descriptions,
            default_description="Default task"
        )
        
        # Episode 0
        sample = dataset[5]
        assert sample["task_description"] == "Episode 0 task"
        
        # Episode 1
        sample = dataset[15]
        assert sample["task_description"] == "Episode 1 task"
        
        # Episode without description uses default
        sample = dataset[35]  # Episode 3
        assert sample["task_description"] == "Default task"


class TestLuckyACTTaskConditioning:
    """Test Lucky_ACT policy with task conditioning."""
    
    @pytest.fixture
    def config(self):
        """Create a test config."""
        return LuckyACTConfig(
            # Small model for testing
            dim_model=128,
            n_heads=4,
            dim_feedforward=256,
            n_encoder_layers=2,
            n_decoder_layers=2,
            
            # Task conditioning
            use_task_conditioning=True,
            task_encoder_type="learned",
            task_embedding_dim=64,
            use_adaln_task_context=True,
            use_task_token=True,
            
            # Required features
            input_features={
                "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=[3, 224, 224]),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=[7]),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=[7]),
            },
            
            # Disable flow for simplicity
            enable_flow_fusion=False,
            auto_infer_features=False,
        )
    
    def test_policy_initialization(self, config):
        """Test that policy initializes correctly with task conditioning."""
        policy = LuckyACTPolicy(config)
        
        # Check task encoder
        assert policy._task_encoder is not None
        assert policy._task_encoder.get_embedding_dim() == 64
        
        # Check model has task token projection
        assert hasattr(policy.model, 'task_token_proj')
        assert hasattr(policy.model, 'task_token_pos_embed')
    
    def test_forward_with_task_embedding(self, config):
        """Test forward pass with task embedding."""
        policy = LuckyACTPolicy(config)
        policy.eval()
        
        batch_size = 2
        batch = {
            "observation.images.top": torch.randn(batch_size, 3, 224, 224),
            "observation.state": torch.randn(batch_size, 7),
            "action": torch.randn(batch_size, config.chunk_size, 7),
            "action_is_pad": torch.zeros(batch_size, config.chunk_size, dtype=torch.bool),
            "task_embedding": torch.randn(batch_size, 64),
        }
        
        with torch.no_grad():
            actions_pred, aux = policy(batch)
        
        assert actions_pred.shape == (batch_size, config.chunk_size, 7)
    
    def test_forward_with_task_description(self, config):
        """Test forward pass with task description string."""
        policy = LuckyACTPolicy(config)
        policy.eval()
        
        batch = {
            "observation.images.top": torch.randn(1, 3, 224, 224),
            "observation.state": torch.randn(1, 7),
            "action": torch.randn(1, config.chunk_size, 7),
            "action_is_pad": torch.zeros(1, config.chunk_size, dtype=torch.bool),
            "task_description": "Pick up the red cube",
        }
        
        with torch.no_grad():
            actions_pred, aux = policy(batch)
        
        assert actions_pred.shape == (1, config.chunk_size, 7)
        assert "task_embedding" in batch  # Should be added by policy
    
    def test_forward_without_task_conditioning(self):
        """Test that policy works without task conditioning."""
        config = LuckyACTConfig(
            use_task_conditioning=False,
            dim_model=128,
            n_heads=4,
            input_features={
                "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=[3, 224, 224]),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=[7]),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=[7]),
            },
            enable_flow_fusion=False,
            auto_infer_features=False,
        )
        
        policy = LuckyACTPolicy(config)
        policy.eval()
        
        batch = {
            "observation.images.top": torch.randn(1, 3, 224, 224),
            "observation.state": torch.randn(1, 7),
            "action": torch.randn(1, config.chunk_size, 7),
            "action_is_pad": torch.zeros(1, config.chunk_size, dtype=torch.bool),
        }
        
        with torch.no_grad():
            actions_pred, aux = policy(batch)
        
        assert actions_pred.shape == (1, config.chunk_size, 7)
    
    def test_task_conditioning_validation(self):
        """Test config validation for task conditioning."""
        # Should fail: task conditioning enabled but no method specified
        with pytest.raises(ValueError, match="requires at least one of"):
            LuckyACTConfig(
                use_task_conditioning=True,
                use_task_token=False,
                use_adaln_task_context=False,
            )
        
        # Should fail: invalid task token position
        with pytest.raises(ValueError, match="task_token_position"):
            LuckyACTConfig(
                use_task_conditioning=True,
                task_token_position="invalid",
            )
        
        # Should fail: invalid encoder type
        with pytest.raises(ValueError, match="Unknown task_encoder_type"):
            LuckyACTConfig(
                use_task_conditioning=True,
                task_encoder_type="invalid",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 