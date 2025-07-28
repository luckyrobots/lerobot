"""Task encoder module for Lucky_ACT policy.

Provides flexible task description encoding with support for multiple backends
and efficient caching for production use.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class TaskEncoder(ABC):
    """Abstract base class for task encoders."""
    
    @abstractmethod
    def encode(self, descriptions: Union[str, List[str]]) -> torch.Tensor:
        """Encode task descriptions into fixed-size embeddings."""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the task embeddings."""
        pass


class SentenceTransformerEncoder(TaskEncoder):
    """Task encoder using Sentence-BERT models."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_size: int = 1024,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEncoder. "
                "Install with: pip install sentence-transformers"
            ) from e
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name).to(self.device)
        self.model.eval()
        
        # Enable caching for repeated task descriptions
        self._encode_cached = lru_cache(maxsize=cache_size)(self._encode_single)
        
        logger.info(f"Initialized SentenceTransformerEncoder with {model_name} on {self.device}")
    
    def _encode_single(self, description: str) -> torch.Tensor:
        """Encode a single description (cached)."""
        with torch.no_grad():
            embedding = self.model.encode(
                description,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False,
            )
        return embedding
    
    def encode(self, descriptions: Union[str, List[str]]) -> torch.Tensor:
        """Encode task descriptions with caching support."""
        if isinstance(descriptions, str):
            return self._encode_cached(descriptions).unsqueeze(0)
        
        # Batch encoding for lists
        embeddings = []
        for desc in descriptions:
            embeddings.append(self._encode_cached(desc))
        
        return torch.stack(embeddings)
    
    def get_embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


class CLIPTextEncoder(TaskEncoder):
    """Task encoder using CLIP text encoder."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        cache_size: int = 1024,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).text_model.to(self.device)
        self.model.eval()
        
        # Get text projection if available
        try:
            full_model = AutoModel.from_pretrained(model_name)
            self.text_projection = full_model.text_projection.to(self.device)
            self.text_projection.eval()
        except AttributeError:
            self.text_projection = None
            logger.warning("CLIP text projection not found, using raw encoder output")
        
        self._encode_cached = lru_cache(maxsize=cache_size)(self._encode_single)
        
        logger.info(f"Initialized CLIPTextEncoder with {model_name} on {self.device}")
    
    def _encode_single(self, description: str) -> torch.Tensor:
        """Encode a single description (cached)."""
        with torch.no_grad():
            inputs = self.tokenizer(
                description,
                padding=True,
                truncation=True,
                max_length=77,  # CLIP's max length
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            # Use pooled output (CLS token)
            text_embeds = outputs.pooler_output
            
            # Apply projection if available
            if self.text_projection is not None:
                text_embeds = self.text_projection(text_embeds)
            
            # Normalize as CLIP does
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        return text_embeds.squeeze(0)
    
    def encode(self, descriptions: Union[str, List[str]]) -> torch.Tensor:
        """Encode task descriptions."""
        if isinstance(descriptions, str):
            return self._encode_cached(descriptions).unsqueeze(0)
        
        embeddings = []
        for desc in descriptions:
            embeddings.append(self._encode_cached(desc))
        
        return torch.stack(embeddings)
    
    def get_embedding_dim(self) -> int:
        """Return embedding dimension."""
        if self.text_projection is not None:
            return self.text_projection.out_features
        else:
            return self.model.config.hidden_size


class LearnedTaskEncoder(TaskEncoder, nn.Module):
    """Learnable task encoder with vocabulary."""
    
    def __init__(
        self,
        vocab_size: int = 1000,
        embedding_dim: int = 384,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Task string to ID mapping
        self.task_to_id: Dict[str, int] = {}
        self.next_id = 0
        
        # Learnable embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim).to(self.device)
        nn.init.normal_(self.embeddings.weight, std=0.02)
        
        logger.info(f"Initialized LearnedTaskEncoder with vocab_size={vocab_size}, dim={embedding_dim}")
    
    def _get_task_id(self, description: str) -> int:
        """Get or assign task ID for description."""
        # Use hash for consistent mapping
        task_hash = hashlib.md5(description.encode()).hexdigest()[:8]
        
        if task_hash not in self.task_to_id:
            if self.next_id >= self.vocab_size:
                raise ValueError(f"Task vocabulary full ({self.vocab_size}). Consider increasing vocab_size.")
            self.task_to_id[task_hash] = self.next_id
            self.next_id += 1
            logger.debug(f"Assigned ID {self.task_to_id[task_hash]} to task: {description[:50]}...")
        
        return self.task_to_id[task_hash]
    
    def encode(self, descriptions: Union[str, List[str]]) -> torch.Tensor:
        """Encode task descriptions using learned embeddings."""
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        
        task_ids = [self._get_task_id(desc) for desc in descriptions]
        task_ids_tensor = torch.tensor(task_ids, device=self.device)
        
        return self.embeddings(task_ids_tensor)
    
    def get_embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim


def create_task_encoder(
    encoder_type: str = "sentence-transformer",
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs
) -> TaskEncoder:
    """Factory function to create task encoders.
    
    Args:
        encoder_type: Type of encoder ("sentence-transformer", "clip", "learned")
        model_name: Model name/path for pretrained encoders
        device: Device to use
        **kwargs: Additional arguments for specific encoder types
    
    Returns:
        TaskEncoder instance
    """
    if encoder_type == "sentence-transformer":
        model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return SentenceTransformerEncoder(model_name, device, **kwargs)
    
    elif encoder_type == "clip":
        model_name = model_name or "openai/clip-vit-base-patch32"
        return CLIPTextEncoder(model_name, device, **kwargs)
    
    elif encoder_type == "learned":
        return LearnedTaskEncoder(device=device, **kwargs)
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}") 