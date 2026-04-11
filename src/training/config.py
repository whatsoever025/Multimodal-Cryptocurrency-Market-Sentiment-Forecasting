"""
Configuration management for Multimodal Crypto Sentiment Forecasting.

Strictly typed dataclass-based configuration with MLOps metadata.
Each branch can have unique wandb_run_name for isolated experiment tracking.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import datetime


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    asset: str = "MULTI"  # "MULTI" for BTC+ETH combined, or single "BTC"/"ETH"
    seq_len: int = 24  # Sliding window length in hours
    batch_size: int = 8  # Per-GPU batch size
    max_text_length: int = 512  # Token sequence length for BERT
    image_size: int = 224  # ResNet50 input size (224x224)
    shuffle_train: bool = True
    num_workers: int = 4  # Data loading workers
    pin_memory: bool = True
    prefetch_factor: int = 2

    def __post_init__(self):
        """Validate data config."""
        if self.asset not in ("BTC", "ETH", "MULTI"):
            raise ValueError(f"asset must be 'BTC', 'ETH', or 'MULTI', got {self.asset}")
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {self.seq_len}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.max_text_length <= 0:
            raise ValueError(f"max_text_length must be > 0, got {self.max_text_length}")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {self.image_size}")


@dataclass
class ModelConfig:
    """Architecture configuration."""
    hidden_dim: int = 512  # Hidden dimension in encoders/LSTM (increased from 256)
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    attention_heads: int = 4
    mha_dropout: float = 0.1
    encoder_dropout: float = 0.2
    head_dropout: float = 0.2
    grad_clip: float = 1.0
    frozen_backbones: bool = True  # Freeze BERT & ResNet50
    use_gradient_checkpointing: bool = True  # Mandatory for 16GB VRAM
    init_weights: bool = True

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


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    max_epochs: int = 60  # Increased from 50 (more capacity needs more training)
    learning_rate: float = 5e-5  # Reduced from 1e-4 for stability with larger model
    weight_decay: float = 1e-5
    accumulate_steps: int = 2  # Gradient accumulation for 16GB VRAM
    warmup_steps: int = 800  # Increased warmup for larger model (was 500)
    warmup_proportion: float = 0.1  # Alternate: warmup as % of total steps
    use_warmup: bool = True
    scheduler_type: str = "cosine"  # "cosine" or "constant"
    num_training_steps: Optional[int] = None  # Set during training loop

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


@dataclass
class OptimizationConfig:
    """Mixed precision and optimization settings."""
    mixed_precision: bool = True  # Use torch.cuda.amp.autocast
    dtype: str = "float16"  # "float16" or "bfloat16"
    use_scaler: bool = True  # Use GradScaler with mixed precision
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000

    def __post_init__(self):
        """Validate optimization config."""
        if self.dtype not in ("float16", "bfloat16"):
            raise ValueError(f"dtype must be 'float16' or 'bfloat16', got {self.dtype}")


@dataclass
class MLOpsConfig:
    """Weights & Biases and artifact management."""
    wandb_project: str = "crypto-sentiment-forecasting"
    wandb_entity: Optional[str] = None
    wandb_run_name: str = field(default_factory=lambda: f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    wandb_tags: list = field(default_factory=list)
    wandb_notes: str = ""
    use_wandb: bool = True
    
    # HuggingFace Hub configuration
    hf_repo_name: str = "crypto-sentiment-model"
    hf_repo_type: str = "model"
    hf_push_to_hub: bool = False
    
    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path("/kaggle/working/checkpoints"))
    save_best_only: bool = True
    save_frequency: int = 1  # Save after every N epochs
    keep_last_n: int = 3  # Keep only last N checkpoints
    
    # Logging
    log_frequency: int = 100  # Log metrics every N steps
    eval_frequency: int = 1  # Evaluate every N epochs

    def __post_init__(self):
        """Validate MLOps config."""
        if not isinstance(self.checkpoint_dir, Path):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if self.save_frequency <= 0:
            raise ValueError(f"save_frequency must be > 0, got {self.save_frequency}")
        if self.log_frequency <= 0:
            raise ValueError(f"log_frequency must be > 0, got {self.log_frequency}")


@dataclass
class InferenceConfig:
    """Inference and evaluation settings."""
    device: str = "cuda"
    inference_batch_size: int = 32
    use_amp: bool = True
    num_inference_workers: int = 4

    def __post_init__(self):
        """Validate inference config."""
        if self.inference_batch_size <= 0:
            raise ValueError(f"inference_batch_size must be > 0, got {self.inference_batch_size}")


@dataclass
class ExperimentConfig:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    mlops: MLOpsConfig = field(default_factory=MLOpsConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
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
            if not self.model.frozen_backbones or not self.model.use_gradient_checkpointing:
                raise ValueError(
                    f"batch_size > 8 with VRAM constraint requires "
                    f"frozen_backbones=True AND use_gradient_checkpointing=True"
                )

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": {k: v for k, v in self.training.__dict__.items() if v is not None},
            "optimization": self.optimization.__dict__,
            "mlops": {k: v for k, v in self.mlops.__dict__.items() if not isinstance(v, Path)},
            "inference": self.inference.__dict__,
            "seed": self.seed,
            "device": self.device,
            "debug": self.debug,
        }


# Export default configurations
DEFAULT_CONFIG = ExperimentConfig()


def create_config(
    asset: str = "BTC",
    seq_len: int = 24,
    batch_size: int = 8,
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
        batch_size: Batch size per GPU
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
