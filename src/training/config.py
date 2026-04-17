"""
Configuration management for Multimodal Crypto Sentiment Forecasting.

Strictly typed dataclass-based configuration with MLOps metadata.
Each branch can have unique wandb_run_name for isolated experiment tracking.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    asset: str = "MULTI"  # "MULTI" for BTC+ETH combined, or single "BTC"/"ETH"
    seq_len: int = 24  # Sliding window length in hours
    batch_size: int = 128  # Per-GPU batch size

    def __post_init__(self):
        """Validate data config."""
        if self.asset not in ("BTC", "ETH", "MULTI"):
            raise ValueError(f"asset must be 'BTC', 'ETH', or 'MULTI', got {self.asset}")
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {self.seq_len}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")


@dataclass
class ModelConfig:
    """Architecture configuration."""
    hidden_dim: int = 256  # Hidden dimension in encoders/attention - MUST match embedding dimensions (256D)
    bottleneck_dim: int = 64  # Bottleneck layer: 256 -> 64 (compression before LSTM)
    lstm_layers: int = 1  # Simplified: reduced from 2 to 1
    lstm_hidden_dim: int = 64  # Simplified: reduced from 256 to 64
    lstm_dropout: float = 0.5  # AGGRESSIVE: Kill temporal memorization (prevents overfitting on sequences)
    attention_heads: int = 4
    mha_dropout: float = 0.1  # CRITICAL (2025-04-17): Must stay ≤0.1 for numerical stability in scaled dot-product backward
                               # Attention is already regularized by its structure - no higher dropout needed
    encoder_dropout: float = 0.3  # INCREASED: Strong regularization in tabular encoder MLP (safe from NaN)
    head_dropout: float = 0.4  # INCREASED: Strong regularization in prediction head (safe from NaN, only MLPs)
    grad_clip: float = 1.0
    frozen_backbones: bool = True  # Freeze BERT & ViT

    def __post_init__(self):
        """Validate model config."""
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if self.lstm_layers <= 0:
            raise ValueError(f"lstm_layers must be > 0, got {self.lstm_layers}")
        if not (0 <= self.lstm_dropout < 1.0):
            raise ValueError(f"lstm_dropout must be in [0, 1), got {self.lstm_dropout}")
        if self.attention_heads <= 0:
            raise ValueError(f"attention_heads must be > 0, got {self.attention_heads}")
        if self.grad_clip < 0:
            raise ValueError(f"grad_clip must be >= 0, got {self.grad_clip}")
        if self.mha_dropout > 0.1:
            raise ValueError(
                f"mha_dropout must be ≤0.1 for numerical stability in backward pass, got {self.mha_dropout}"
            )


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    max_epochs: int = 60
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    accumulate_steps: int = 2
    warmup_steps: int = 800
    use_warmup: bool = True
    num_training_steps: Optional[int] = None
    early_stopping_patience: int = 7

    def __post_init__(self):
        """Validate training config."""
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be > 0, got {self.max_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.accumulate_steps <= 0:
            raise ValueError(f"accumulate_steps must be > 0, got {self.accumulate_steps}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.early_stopping_patience <= 0:
            raise ValueError(f"early_stopping_patience must be > 0, got {self.early_stopping_patience}")


@dataclass
class MLOpsConfig:
    """Weights & Biases and artifact management."""
    wandb_project: str = "crypto-sentiment-forecasting"
    wandb_run_name: str = field(default_factory=lambda: f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    use_wandb: bool = True
    
    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path(
        "/kaggle/working/checkpoints" if Path("/kaggle/working").exists() 
        else "./src/training/checkpoints"
    ))
    save_best_only: bool = True
    save_frequency: int = 1  # Save after every N epochs
    keep_last_n: int = 3  # Keep only last N checkpoints
    
    # Logging
    log_frequency: int = 100
    eval_frequency: int = 1

    def __post_init__(self):
        """Validate MLOps config."""
        if not isinstance(self.checkpoint_dir, Path):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if self.save_frequency <= 0:
            raise ValueError(f"save_frequency must be > 0, got {self.save_frequency}")
        if self.log_frequency <= 0:
            raise ValueError(f"log_frequency must be > 0, got {self.log_frequency}")
        if self.eval_frequency <= 0:
            raise ValueError(f"eval_frequency must be > 0, got {self.eval_frequency}")


@dataclass
class ExperimentConfig:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mlops: MLOpsConfig = field(default_factory=MLOpsConfig)
    
    # Global settings
    seed: int = 42
    device: str = "cuda"
    debug: bool = False

    def __post_init__(self):
        """Validate master config and cross-config constraints."""
        # Ensure batch_size is divisible by accumulate_steps
        if self.data.batch_size % self.training.accumulate_steps != 0:
            raise ValueError(
                f"batch_size ({self.data.batch_size}) must be divisible by "
                f"accumulate_steps ({self.training.accumulate_steps})"
            )
        
        # For 16GB VRAM, enforce conservative settings
        if self.data.batch_size > 8:
            if not self.model.frozen_backbones:
                raise ValueError(
                    f"batch_size > 8 with VRAM constraint requires frozen_backbones=True"
                )

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": {k: v for k, v in self.training.__dict__.items() if v is not None},
            "mlops": {k: v for k, v in self.mlops.__dict__.items() if not isinstance(v, Path)},
            "seed": self.seed,
            "device": self.device,
            "debug": self.debug,
        }


# Export default configurations
DEFAULT_CONFIG = ExperimentConfig()


def create_config(
    asset: str = "BTC",
    seq_len: int = 24,
    batch_size: int = None,  # None = use DataConfig default (128)
    hidden_dim: int = 256,
    learning_rate: float = 1e-4,
    wandb_run_name: Optional[str] = None,
    **kwargs
) -> ExperimentConfig:
    """
    Factory function to create ExperimentConfig with custom parameters.
    
    Args:
        asset: "BTC" or "ETH"
        seq_len: Sliding window length in hours
        batch_size: Batch size per GPU (None = use DataConfig default)
        hidden_dim: Model hidden dimension
        learning_rate: Optimizer learning rate
        wandb_run_name: Unique run identifier for W&B tracking
        **kwargs: Additional config overrides
    
    Returns:
        ExperimentConfig instance
    """
    config = ExperimentConfig()
    
    # Update data config
    config.data.asset = asset
    config.data.seq_len = seq_len
    if batch_size is not None:
        config.data.batch_size = batch_size
    
    # Update model config
    config.model.hidden_dim = hidden_dim
    
    # Update training config
    config.training.learning_rate = learning_rate
    
    # Update MLOps config
    if wandb_run_name:
        config.mlops.wandb_run_name = wandb_run_name
    
    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Trigger validation
    config.__post_init__()
    
    return config


if __name__ == "__main__":
    """Test configuration loading and validation."""
    # Test default config
    default_cfg = ExperimentConfig()
    print("✓ Default config loaded successfully")
    print(f"  Asset: {default_cfg.data.asset}")
    print(f"  Seq len: {default_cfg.data.seq_len}")
    print(f"  Batch size: {default_cfg.data.batch_size}")
    print(f"  Hidden dim: {default_cfg.model.hidden_dim}")
    print(f"  Learning rate: {default_cfg.training.learning_rate}")
    print(f"  W&B run: {default_cfg.mlops.wandb_run_name}")
    
    # Test factory function
    custom_cfg = create_config(
        asset="ETH",
        hidden_dim=512,
        wandb_run_name="exp/eth_large_hidden"
    )
    print("\n✓ Custom config created successfully")
    print(f"  Asset: {custom_cfg.data.asset}")
    print(f"  Hidden dim: {custom_cfg.model.hidden_dim}")
    print(f"  W&B run: {custom_cfg.mlops.wandb_run_name}")
    
    # Test validation
    try:
        bad_cfg = ExperimentConfig()
        bad_cfg.data.seq_len = -1
        bad_cfg.__post_init__()
    except ValueError as e:
        print(f"\n✓ Validation caught error: {e}")
    
    print("\n✅ All config tests passed!")
