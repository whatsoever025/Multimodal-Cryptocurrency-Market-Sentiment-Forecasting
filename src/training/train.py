"""
Production training loop with VRAM management, W&B integration, and best model checkpointing.

Key Features:
- Trainer class with full state management (save/load/train/validate)
- AMP (Automatic Mixed Precision) with float16 for VRAM reduction
- Gradient accumulation (batch_size=8, accumulate_steps=2 → effective BS=16)
- W&B integration per-branch via wandb_run_name
- Best model checkpointing with experiment naming
"""

import torch

torch.autograd.set_detect_anomaly(True) # Công cụ truy tìm nguồn gốc NaN

import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import json
import argparse
from datetime import datetime
import sys

# Suppress transformers warnings
import warnings
warnings.filterwarnings("ignore")

try:
    import wandb
except ImportError:
    wandb = None

from .config import ExperimentConfig, create_config
from .dataset import CryptoMultimodalDataset, multimodal_collate_fn, create_dataloaders
from .model import MultimodalFusionNet


logger = logging.getLogger(__name__)


class Trainer:
    """
    Training orchestrator with state management, checkpointing, and MLOps integration.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        device: str = "cuda",
    ):
        """
        Initialize trainer.
        
        Args:
            config: ExperimentConfig instance
            model: MultimodalFusionNet or similar
            device: "cuda" or "cpu"
        """
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # Optimizer & scheduler
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # State tracking
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"Trainer initialized (device={device})")
    
    def setup_optimizer(self) -> None:
        """
        Initialize optimizer with conservative hyperparameters for multimodal fine-tuning.
        
        CRITICAL: Learning Rate Strategy
        - For multimodal architectures (BERT + ResNet + LSTM + Attention):
          - Safe range: 5e-5 to 1e-4 (fine-tuning frozen backbones)
          - Default: 5e-5 (balanced between stability and convergence speed)
          - If encountering gradient explosion: reduce to 1e-5 to 5e-6
        
        - AMP (Automatic Mixed Precision) sensitivity:
          - float16 amplifies gradient magnitude by ~100x
          - Conservative LR mitigates explosion risk in mixed precision
        """
        # AdamW optimizer with conservative defaults for multimodal training
        self.optimizer = optim.AdamW(
            self.model.get_trainable_params(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),  # Default AdamW betas
            eps=1e-8,             # Numerical stability for float16
        )
        logger.info(
            f"✓ Optimizer: AdamW\n"
            f"  Learning Rate: {self.config.training.learning_rate:.2e}\n"
            f"  Weight Decay: {self.config.training.weight_decay:.2e}\n"
            f"  Strategy: Conservative (AMP-safe for multimodal fine-tuning)"
        )
        
        # Learning rate scheduler
        if self.config.training.use_warmup:
            # Warmup + cosine anneal
            from transformers import get_cosine_schedule_with_warmup
            
            total_steps = self.config.training.num_training_steps
            if total_steps is None:
                total_steps = 1000  # Fallback
            
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.training.warmup_steps,
                num_training_steps=total_steps,
            )
            logger.info(f"Scheduler: Cosine with warmup ({self.config.training.warmup_steps} steps)")
        else:
            self.scheduler = None
            logger.info("Scheduler: None (constant LR)")
        
        # AMP GradScaler
        if self.config.optimization.mixed_precision:
            self.scaler = GradScaler(
                init_scale=self.config.optimization.init_scale,
                growth_factor=self.config.optimization.growth_factor,
                backoff_factor=self.config.optimization.backoff_factor,
                growth_interval=self.config.optimization.growth_interval,
            )
            logger.info("AMP GradScaler initialized")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Run one training epoch with gradient accumulation, AMP, and explicit gradient clipping.
        
        ============================================================================
        CRITICAL: AMP + Gradient Clipping Pattern (PyTorch Best Practice)
        ============================================================================
        
        When using GradScaler (AMP), gradient clipping must follow this order:
        
        1. Forward pass (autocast context)           → loss in float16
        2. Backward pass (scaler.scale)              → gradients scaled by scale factor
        3. UNSCALE gradients (scaler.unscale_)       ← CRITICAL: Must come first!
        4. Clip gradients (clip_grad_norm_)          @ normal float32 scale
        5. Optimizer step (scaler.step)              @ scaled learning rate
        6. Update scale (scaler.update)              → dynamic loss scaling
        
        Why this order?
        - GradScaler multiplies loss by scale (default 65536) before backward
        - Gradients are scaled proportionally
        - Clipping at scaled magnitude → actual gradient norm too small → training stalls
        - unscale_() restores true gradient magnitude before clipping
        - After clipping, scaler.step() applies learning rate
        
        ============================================================================
        
        VRAM Management:
        - Batch size 8 (seq_len 24 = 192 effective samples)
        - Gradient accumulation: accumulate_steps=2 → effective batch=16
        - Mixed precision: forward/backward in float16, updates in float32
        - Gradient clipping: max_norm=1.0 (prevents explosion in LSTM/Attention)
        
        Args:
            train_loader: Training DataLoader
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_steps = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # ========== FORWARD PASS (AMP) ==========
            # Use autocast to reduce VRAM (float16 for forward/backward)
            with autocast(enabled=self.config.optimization.mixed_precision):
                predictions = self.model(batch)  # (batch, 1) - float16 in autocast
                targets = batch["target"].unsqueeze(1)  # (batch, 1)
                
                # MSE loss - inherit dtype from predictions (float16 in autocast)
                loss = nn.MSELoss()(predictions, targets)
                
                # CRITICAL: Scale loss for gradient accumulation
                # This prevents individual batch gradients from being too small
                loss = loss / self.config.training.accumulate_steps
            
            # ========== BACKWARD PASS (SCALED) ==========
            # GradScaler multiplies loss by scale factor (e.g., 65536) before backward
            # This prevents underflow in float16 gradients
            if self.scaler is not None:
                # AMP path: scale loss, backward, then unscale before clip
                self.scaler.scale(loss).backward()
            else:
                # Non-AMP path: direct backward (fallback for CPU/debugging)
                loss.backward()
            
            # ========== GRADIENT ACCUMULATION CHECK ==========
            # Only update weights every accumulate_steps batches
            if (batch_idx + 1) % self.config.training.accumulate_steps == 0:
                
                # ========== UNSCALE GRADIENTS (CRITICAL FOR AMP) ==========
                # MUST happen before clip_grad_norm_!
                # Restores true gradient scale for proper clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    # After unscale_, gradients are in float32 at true scale
                
                # ========== GRADIENT CLIPPING ==========
                # Prevents explosion in LSTM/Attention layers
                # max_norm=1.0: clip total gradient norm to ≤ 1.0
                max_grad_norm = self.config.model.grad_clip
                if max_grad_norm > 0:
                    # Returns total gradient norm (useful for debugging)
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_grad_norm,
                        norm_type=2.0,  # L2 norm (standard for gradient clipping)
                    )
                    
                    # Log clipping if gradients were large
                    if total_norm > max_grad_norm:
                        logger.debug(
                            f"Gradient clipped: norm={total_norm:.4f} "
                            f"(clipped to {max_grad_norm})"
                        )
                
                # ========== OPTIMIZER STEP (with scaling) ==========
                # Apply learning rate update to scaled gradients
                if self.scaler is not None:
                    # AMP path: step with scaler (applies learning rate)
                    self.scaler.step(self.optimizer)
                    # Update scale factor for next iteration
                    # (increases if no NaNs, decreases if overflow detected)
                    self.scaler.update()
                else:
                    # Non-AMP path: standard optimizer step
                    self.optimizer.step()
                
                # ========== LEARNING RATE SCHEDULE ==========
                # Update learning rate (warmup → cosine decay)
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # ========== RESET GRADIENTS ==========
                # Zero out accumulated gradients for next accumulation cycle
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # ========== LOGGING ==========
                if self.global_step % self.config.mlops.log_frequency == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Epoch {self.epoch+1} | Step {self.global_step} | "
                        f"Loss {loss.item():.6f} | LR {current_lr:.2e}"
                    )
                    
                    if wandb is not None and wandb.run is not None:
                        wandb.log({
                            "train_loss": loss.item(),
                            "learning_rate": current_lr,
                            "global_step": self.global_step,
                        })
            
            # Accumulate loss for epoch average
            total_loss += loss.item() * self.config.training.accumulate_steps
            num_steps += 1
        
        avg_loss = total_loss / num_steps
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Run validation.
        
        Args:
            val_loader: Validation DataLoader
        
        Returns:
            (mse_loss, mae_loss)
        """
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        num_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward with AMP
                with autocast(
                    enabled=self.config.optimization.mixed_precision,
                    dtype=torch.float16 if self.config.optimization.dtype == "float16" else torch.float32,
                ):
                    predictions = self.model(batch)  # (batch, 1)
                    targets = batch["target"].unsqueeze(1)  # (batch, 1)
                    
                    mse = nn.MSELoss()(predictions, targets)
                    mae = nn.L1Loss()(predictions, targets)
                
                total_mse += mse.item()
                total_mae += mae.item()
                num_steps += 1
        
        avg_mse = total_mse / num_steps
        avg_mae = total_mae / num_steps
        self.val_losses.append(avg_mse)
        
        return avg_mse, avg_mae
    
    def save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
            is_best: If True, mark as best model
        """
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": self.config.to_dict(),
        }
        
        torch.save(checkpoint, path)
        
        status = "BEST" if is_best else "checkpoint"
        logger.info(f"✓ Saved {status} model to {path}")
        
        # Log to W&B
        if wandb is not None and wandb.run is not None:
            wandb.save(str(path))
    
    def load_checkpoint(self, path: Path) -> None:
        """
        Load training checkpoint and restore state.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_epoch = checkpoint["best_epoch"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        
        logger.info(f"✓ Loaded checkpoint from {path}")
    
    def cleanup_old_checkpoints(self) -> None:
        """
        Delete old checkpoints, keeping only the last N periodic checkpoints.
        Preserves best model checkpoint always.
        """
        checkpoint_dir = self.config.mlops.checkpoint_dir
        keep_last_n = self.config.mlops.keep_last_n
        
        # Find all periodic checkpoints (not best)
        periodic_checkpoints = sorted([
            p for p in checkpoint_dir.glob(f"{self.config.mlops.wandb_run_name}_epoch_*.pt")
        ])
        
        # Delete old ones
        if len(periodic_checkpoints) > keep_last_n:
            for old_ckpt in periodic_checkpoints[:-keep_last_n]:
                old_ckpt.unlink()
                logger.info(f"  Deleted old checkpoint: {old_ckpt.name}")


def setup_logging(level=logging.INFO) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(args):
    """
    Main training script.
    
    Args:
        args: Parsed command-line arguments
    """
    # Setup
    setup_logging()
    logger.info("=" * 80)
    logger.info("Training Multimodal Crypto Sentiment Model")
    logger.info("=" * 80)
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    
    device = "cuda" if torch.cuda.is_available() and not args.debug else "cpu"
    logger.info(f"Device: {device}")
    
    # Load or create config
    if args.config:
        logger.info(f"Loading config from {args.config}...")
        # For now, use default config (could load from YAML in future)
        config = ExperimentConfig()
    else:
        config = create_config(
            asset=args.asset,
            wandb_run_name=args.run_name,
        )
        config.debug = args.debug
    
    logger.info(f"Config: asset={config.data.asset}, seq_len={config.data.seq_len}, batch_size={config.data.batch_size}")
    logger.info(f"Model: hidden_dim={config.model.hidden_dim}, frozen_backbones={config.model.frozen_backbones}")
    logger.info(f"Training: lr={config.training.learning_rate:.2e}, epochs={config.training.max_epochs}")
    logger.info(f"MLOps: wandb_run={config.mlops.wandb_run_name}")
    
    # Create dataloaders
    logger.info("\n" + "-" * 80)
    logger.info("Loading datasets...")
    print("[PROGRESS] Starting to load datasets (this may take 1-5 minutes)...")
    sys.stdout.flush()
    # NOTE: num_workers=0 always (Kaggle multi-worker deadlock fix)
    # Multi-worker DataLoader spawns intermittently deadlock on Kaggle
    try:
        dataloaders = create_dataloaders(config, num_workers=0)
        print("[PROGRESS] ✓ All datasets loaded successfully!")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}", exc_info=True)
        print(f"[ERROR] Failed to load datasets: {e}")
        sys.stdout.flush()
        raise
    train_loader = dataloaders["train"]
    val_loader = dataloaders["validation"]
    
    # Set total steps for scheduler
    config.training.num_training_steps = len(train_loader) * config.training.max_epochs // config.training.accumulate_steps
    
    # Initialize model
    logger.info("\n" + "-" * 80)
    logger.info("Initializing model...")
    model = MultimodalFusionNet(config)
    logger.info(f"Model device: {next(model.parameters()).device}")
    
    # Initialize trainer
    trainer = Trainer(config, model, device=device)
    trainer.setup_optimizer()
    
    # W&B initialization
    if config.mlops.use_wandb and wandb is not None:
        logger.info("\n" + "-" * 80)
        logger.info("Initializing W&B...")
        wandb.init(
            project=config.mlops.wandb_project,
            name=config.mlops.wandb_run_name,
            entity=config.mlops.wandb_entity,
            config=config.to_dict(),
            tags=config.mlops.wandb_tags,
            notes=config.mlops.wandb_notes,
        )
        logger.info(f"✓ W&B initialized: {config.mlops.wandb_project}/{config.mlops.wandb_run_name}")
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume:
        logger.info("\n" + "-" * 80)
        logger.info("Resuming from checkpoint...")
        
        # Find latest checkpoint
        checkpoint_files = sorted([
            p for p in config.mlops.checkpoint_dir.glob(f"{config.mlops.wandb_run_name}_epoch_*.pt")
        ])
        
        if checkpoint_files:
            latest_ckpt = checkpoint_files[-1]
            trainer.load_checkpoint(latest_ckpt)
            start_epoch = trainer.epoch + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            logger.warning("No checkpoint found, starting from scratch")
    
    # Training loop
    logger.info("\n" + "-" * 80)
    logger.info("Starting training...")
    logger.info("-" * 80 + "\n")
    print("[PROGRESS] ✓ Setup complete, training begins now...")
    sys.stdout.flush()
    
    for epoch in range(start_epoch, config.training.max_epochs):
        trainer.epoch = epoch
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        logger.info(f"Epoch {epoch+1:3d} | Train Loss {train_loss:.6f}")
        
        # Validate
        if (epoch + 1) % config.mlops.eval_frequency == 0:
            val_mse, val_mae = trainer.validate(val_loader)
            logger.info(f"Epoch {epoch+1:3d} | Val MSE {val_mse:.6f} | Val MAE {val_mae:.6f}")
            
            # Log to W&B
            if wandb is not None and wandb.run is not None:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_mse": val_mse,
                    "val_mae": val_mae,
                })
            
            # Save best model
            if val_mse < trainer.best_val_loss:
                trainer.best_val_loss = val_mse
                trainer.best_epoch = epoch + 1
                
                checkpoint_path = config.mlops.checkpoint_dir / f"{config.mlops.wandb_run_name}_best.pt"
                trainer.save_checkpoint(checkpoint_path, is_best=True)
        
        # Save periodic checkpoint
        if (epoch + 1) % config.mlops.save_frequency == 0:
            periodic_ckpt_path = config.mlops.checkpoint_dir / f"{config.mlops.wandb_run_name}_epoch_{epoch+1:03d}.pt"
            trainer.save_checkpoint(periodic_ckpt_path, is_best=False)
            trainer.cleanup_old_checkpoints()
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Best Val Loss: {trainer.best_val_loss:.6f} (Epoch {trainer.best_epoch})")
    logger.info(f"Best Model Checkpoint: {config.mlops.checkpoint_dir / f'{config.mlops.wandb_run_name}_best.pt'}")
    logger.info("=" * 80)
    
    # Finish W&B
    if wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multimodal crypto sentiment model")
    parser.add_argument("--asset", choices=["MULTI"], default="MULTI", help="Cryptocurrency asset (multi-asset: BTC+ETH combined)")
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--config", type=str, default=None, help="Config file path (YAML)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--debug", action="store_true", help="Debug mode (small dataset)")
    
    args = parser.parse_args()
    
    # Auto-generate run name if not provided
    if args.run_name is None:
        args.run_name = f"{args.asset.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    main(args)
