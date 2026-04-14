"""
Production training loop with VRAM management, W&B integration, and best model checkpointing.

Key Features:
- Trainer class with full state management (save/load/train/validate)
- Pure float32 training (no AMP) for numerical stability
- Gradient accumulation configured via config.training.accumulate_steps
- Gradient clipping via config.model.grad_clip
- W&B integration per-branch via wandb_run_name
- Best model checkpointing with experiment naming
"""

import torch

torch.autograd.set_detect_anomaly(True) # Công cụ truy tìm nguồn gốc NaN

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import logging
import json
import argparse
from datetime import datetime
import sys
from tqdm import tqdm

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
from .utils import setup_logging, format_duration


logger = logging.getLogger(__name__)


def safe_wandb_log(log_dict: Dict, commit: bool = True) -> bool:
    """
    Safely log to W&B with comprehensive error handling.
    
    Args:
        log_dict: Dictionary of metrics to log
        commit: Whether to commit the log
    
    Returns:
        True if successful, False otherwise
    """
    if wandb is None or wandb.run is None:
        return False
    
    try:
        wandb.log(log_dict, commit=commit)
        return True
    except Exception as e:
        logger.warning(f"Failed to log to W&B: {e}")
        return False


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
        
        # State tracking
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = []  # Store all validation metrics for comparison
        
        logger.info(f"Trainer initialized (device={device})")
    
    def setup_optimizer(self) -> None:
        """
        Initialize optimizer with conservative hyperparameters for offline feature extraction.
        
        CRITICAL: Learning Rate Strategy (Pure Float32)
        - Off-line features: only TabularEncoder, CrossModalAttention, LSTM, and PredictionHead are trainable
        - Safe range: 1e-4 to 5e-4 (learnable components only, no backbones)
        - Default: 1e-4 (balanced between stability and convergence speed)
        - If encountering NaN or gradient explosion: reduce to 5e-5 to 1e-5
        
        - Pure float32 advantages:
          - No gradient scaling complexity (no underflow/overflow)
          - Direct gradient magnitudes (easier debugging)
          - Sufficient VRAM on Kaggle 16GB (BS=8, seq_len=24)
          - Gradient clipping handles LSTM/Attention instability
        """
        # AdamW optimizer with conservative defaults for multimodal training
        self.optimizer = optim.AdamW(
            self.model.get_trainable_params(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),  # Default AdamW betas
            eps=1e-8,             # Numerical stability (float32 safe)
        )
        logger.info(
            f"✓ Optimizer: AdamW\n"
            f"  Learning Rate: {self.config.training.learning_rate:.2e}\n"
            f"  Weight Decay: {self.config.training.weight_decay:.2e}\n"
            f"  Strategy: Conservative (Pure float32, no AMP)"
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
        logger.info("Pure float32 training (no AMP)")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Run one training epoch with gradient accumulation and explicit gradient clipping.
        Pure float32 implementation (no AMP) for numerical stability.
        Collects predictions for comprehensive metric computation.
        
        ============================================================================
        TRAINING LOOP STRUCTURE (Pure Float32 + tqdm)
        ============================================================================
        
        1. Forward pass:         model(batch) → predictions (float32)
        2. Compute loss:         MSE(predictions, targets)
        3. Scale loss:           loss / accumulate_steps (for accumulation)
        4. Backward pass:        loss.backward() (standard PyTorch)
        5. Accumulation check:   if (batch_idx + 1) % accumulate_steps == 0
        6. Gradient clipping:    clip_grad_norm_(model.parameters(), max_norm=1.0)
        7. Optimizer step:       optimizer.step() (standard update)
        8. Schedule step:        scheduler.step() (learning rate decay)
        9. Zero gradients:       optimizer.zero_grad() (reset for next accumulation)
        10. tqdm update:         Update progress bar with moving loss average
        
        Why pure float32?
        - Eliminates float16 underflow/overflow in backward pass
        - Eliminates GradScaler scaling complexity
        - Direct gradient magnitudes → easier debugging
        - Kaggle 16GB VRAM sufficient for BS=8, seq_len=24
        
        ============================================================================
        
        VRAM Management:
        - Batch size 8 (seq_len 24 = 192 effective samples)
        - Gradient accumulation: accumulate_steps=2 → effective batch=16
        - Precision: Native float32 (no mixed precision)
        - Gradient clipping: max_norm=1.0 (prevents explosion in LSTM/Attention)
        
        Args:
            train_loader: Training DataLoader
        
        Returns:
            Dict with keys: 'loss', 'mse', 'mae', 'rmse', 'r2', 'correlation',
                          'prediction_error_mean', 'prediction_error_std',
                          'predictions', 'targets'
        """
        self.model.train()
        total_loss = 0.0
        num_steps = 0
        
        # Collect predictions for comprehensive metrics
        all_predictions = []
        all_targets = []
        
        # Wrap DataLoader with tqdm for batch-level progress visibility
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.epoch+1}",
            total=len(train_loader),
            leave=True,  # Keep progress bar after epoch completes
            unit="batch",
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device (float32 by default)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # ========== FORWARD PASS (FLOAT32) ==========
            # Standard PyTorch forward pass - all tensors remain float32
            predictions = self.model(batch)  # (batch,) shape
            targets = batch["target"]  # (batch,) shape
            
            # Collect for epoch-level metrics
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())
            
            # Compute MSE loss
            loss = nn.MSELoss()(predictions, targets)
            
            # Scale loss for gradient accumulation
            # Prevents accumulated gradients from growing too large
            loss = loss / self.config.training.accumulate_steps
            
            # ========== BACKWARD PASS (STANDARD) ==========
            # Standard PyTorch backward - gradients computed in float32
            loss.backward()
            
            # ========== GRADIENT ACCUMULATION CHECK ==========
            # Only update weights every accumulate_steps batches
            if (batch_idx + 1) % self.config.training.accumulate_steps == 0:
                
                # ========== GRADIENT CLIPPING ==========
                # Prevents gradient explosion in LSTM/Attention layers
                # Clips total gradient norm to ≤ max_norm from config
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.model.grad_clip,  # From config (default: 1.0)
                    norm_type=2.0,  # L2 norm (standard)
                )
                
                # Log if clipping occurred (indicator of training instability)
                if total_norm > self.config.model.grad_clip:
                    logger.debug(
                        f"Gradient clipped: norm={total_norm:.4f} → {self.config.model.grad_clip}"
                    )
                
                # ========== OPTIMIZER STEP ==========
                # Standard optimizer update (float32)
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
                        log_dict = {
                            "train_step_loss": loss.item(),
                            "learning_rate": current_lr,
                            "global_step": self.global_step,
                        }
                        
                        # Add GPU memory stats if available
                        if torch.cuda.is_available():
                            log_dict["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
                            log_dict["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
                            log_dict["gpu_memory_percent"] = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
                        
                        safe_wandb_log(log_dict)
            
            # Accumulate loss for epoch average
            total_loss += loss.item() * self.config.training.accumulate_steps
            num_steps += 1
            
            # ========== UPDATE PROGRESS BAR ==========
            # Show moving average of loss (smoothed with exponential moving average)
            # Formula: EMA = (current_loss + (n-1) * prev_EMA) / n
            moving_avg_loss = total_loss / num_steps
            pbar.set_postfix({
                "loss": moving_avg_loss,
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
        
        pbar.close()
        
        avg_loss = total_loss / num_steps
        self.train_losses.append(avg_loss)
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)  # (total_samples,) - already 1D
        all_targets = torch.cat(all_targets, dim=0)  # (total_samples,) - already 1D
        
        # MSE, MAE, RMSE
        mse = torch.mean((all_predictions - all_targets) ** 2).item()
        mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
        rmse = np.sqrt(mse)
        
        # R² Score
        ss_res = torch.sum((all_predictions - all_targets) ** 2).item()
        ss_tot = torch.sum((all_targets - all_targets.mean()) ** 2).item()
        r2_score = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Correlation
        pred_mean = all_predictions.mean()
        target_mean = all_targets.mean()
        numerator = torch.sum((all_predictions - pred_mean) * (all_targets - target_mean))
        denom = torch.sqrt(
            torch.sum((all_predictions - pred_mean) ** 2) * torch.sum((all_targets - target_mean) ** 2)
        )
        correlation = (numerator / denom).item() if denom > 0 else 0.0
        
        # Prediction error analysis
        prediction_errors = all_predictions - all_targets
        prediction_error_mean = prediction_errors.mean().item()
        prediction_error_std = prediction_errors.std().item()
        
        # Min/Max ranges
        pred_min, pred_max = all_predictions.min().item(), all_predictions.max().item()
        target_min, target_max = all_targets.min().item(), all_targets.max().item()
        
        logger.info(
            f"Epoch {self.epoch+1} completed | Avg Loss: {avg_loss:.6f} | MSE: {mse:.6f} | "
            f"MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2_score:.6f}"
        )
        
        return {
            "loss": avg_loss,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2_score,
            "correlation": correlation,
            "prediction_error_mean": prediction_error_mean,
            "prediction_error_std": prediction_error_std,
            "pred_min": pred_min,
            "pred_max": pred_max,
            "target_min": target_min,
            "target_max": target_max,
            "predictions": all_predictions,
            "targets": all_targets,
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Run validation (pure float32, no AMP) with comprehensive metrics collection.
        
        Collects predictions and targets for:
        - Standard metrics: MSE, MAE, RMSE
        - Statistical metrics: R², correlation, prediction bias, error std
        - Ground truth vs prediction logging
        
        Args:
            val_loader: Validation DataLoader
        
        Returns:
            Dict with keys: 'mse', 'mae', 'rmse', 'r2', 'correlation', 
                          'prediction_error_mean', 'prediction_error_std',
                          'predictions', 'targets'
        """
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        num_steps = 0
        
        # Collect all predictions and targets for post-hoc analysis
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device (float32)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass (pure float32, no AMP)
                predictions = self.model(batch)  # (batch,)
                targets = batch["target"]  # (batch,)
                
                # Compute metrics
                mse = nn.MSELoss()(predictions, targets)
                mae = nn.L1Loss()(predictions, targets)
                
                total_mse += mse.item()
                total_mae += mae.item()
                num_steps += 1
                
                # Collect for global metrics
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Compute per-batch averages
        avg_mse = total_mse / num_steps
        avg_mae = total_mae / num_steps
        self.val_losses.append(avg_mse)
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)  # (total_samples,) - already 1D
        all_targets = torch.cat(all_targets, dim=0)  # (total_samples,) - already 1D
        
        # Compute additional metrics
        rmse = torch.sqrt(torch.tensor(avg_mse)).item()
        
        # R² Score: 1 - (SS_res / SS_tot)
        ss_res = torch.sum((all_predictions - all_targets) ** 2).item()
        ss_tot = torch.sum((all_targets - all_targets.mean()) ** 2).item()
        r2_score = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Correlation coefficient between predictions and targets
        pred_mean = all_predictions.mean()
        target_mean = all_targets.mean()
        numerator = torch.sum((all_predictions - pred_mean) * (all_targets - target_mean))
        denom = torch.sqrt(
            torch.sum((all_predictions - pred_mean) ** 2) * torch.sum((all_targets - target_mean) ** 2)
        )
        correlation = (numerator / denom).item() if denom > 0 else 0.0
        
        # Prediction error analysis (bias and std)
        prediction_errors = all_predictions - all_targets
        prediction_error_mean = prediction_errors.mean().item()  # Bias
        prediction_error_std = prediction_errors.std().item()  # Prediction spread
        
        # Min/Max of predictions and targets (for range checking)
        pred_min, pred_max = all_predictions.min().item(), all_predictions.max().item()
        target_min, target_max = all_targets.min().item(), all_targets.max().item()
        
        return {
            "mse": avg_mse,
            "mae": avg_mae,
            "rmse": rmse,
            "r2": r2_score,
            "correlation": correlation,
            "prediction_error_mean": prediction_error_mean,
            "prediction_error_std": prediction_error_std,
            "pred_min": pred_min,
            "pred_max": pred_max,
            "target_min": target_min,
            "target_max": target_max,
            "predictions": all_predictions,
            "targets": all_targets,
        }
    
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
    # CRITICAL: Enforce num_workers=0 regardless of config
    logger.info(f"DataLoader settings: batch_size={config.data.batch_size}, num_workers=0 (forced), pin_memory=True")
    try:
        dataloaders = create_dataloaders(
            config,
            features_dir=args.features_dir,
            num_workers=0  # CRITICAL: Always 0 on Kaggle
        )
        print("[PROGRESS] ✓ All datasets loaded successfully!")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}", exc_info=True)
        print(f"[ERROR] Failed to load datasets: {e}")
        sys.stdout.flush()
        raise
    train_loader = dataloaders["train"]
    val_loader = dataloaders["validation"]
    test_loader = dataloaders["test_in_domain"]  # Load test dataloader for final evaluation
    
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
            settings=wandb.Settings(
                # Disable service ping wait to avoid timeout issues
                _service_wait=0,
                _disable_stats=False,
            ),
        )
        logger.info(f"✓ W&B initialized: {config.mlops.wandb_project}/{config.mlops.wandb_run_name}")
        
        # Log GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory_total = gpu_props.total_memory / (1024**3)  # GB
            logger.info(f"✓ GPU: {gpu_name} ({gpu_memory_total:.1f}GB)")
            wandb.config.update({
                "gpu_name": gpu_name,
                "gpu_memory_gb": gpu_memory_total,
            })
    
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
        train_metrics = trainer.train_epoch(train_loader)
        train_loss = train_metrics["loss"]
        logger.info(
            f"Epoch {epoch+1:3d} | "
            f"Train Loss {train_loss:.6f} | "
            f"Train MSE {train_metrics['mse']:.6f} | "
            f"Train RMSE {train_metrics['rmse']:.6f} | "
            f"Train MAE {train_metrics['mae']:.6f} | "
            f"Train R² {train_metrics['r2']:.6f}"
        )
        
        # Log training metrics to W&B
        if wandb is not None and wandb.run is not None:
            # Core training metrics - log simple scalars first
            train_log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_mse": train_metrics["mse"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "train_correlation": train_metrics["correlation"],
                "train_prediction_bias": train_metrics["prediction_error_mean"],
                "train_prediction_error_std": train_metrics["prediction_error_std"],
                "train_pred_min": train_metrics["pred_min"],
                "train_pred_max": train_metrics["pred_max"],
                "train_target_min": train_metrics["target_min"],
                "train_target_max": train_metrics["target_max"],
            }
            
            # Add GPU memory stats if available
            if torch.cuda.is_available():
                train_log_dict["train_gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
                train_log_dict["train_gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
                train_log_dict["train_gpu_memory_percent"] = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
            
            wandb.log(train_log_dict, commit=False)  # Don't commit yet, add visualizations
            
            # Create visualizations for training data
            predictions = train_metrics["predictions"].numpy()
            targets = train_metrics["targets"].numpy()
            
            # Scatter plot (first 500 samples for efficiency)
            plot_limit = min(500, len(predictions))
            try:
                wandb_plot = wandb.plot.scatter(
                    wandb.Table(data=[
                        [x, y] for x, y in zip(targets[:plot_limit].tolist(), predictions[:plot_limit].tolist())
                    ], columns=["Ground Truth", "Prediction"]),
                    "Ground Truth", "Prediction", title="[TRAIN] Predictions vs Ground Truth"
                )
                wandb.log({"train_predictions_scatter": wandb_plot}, commit=False)
            except Exception as e:
                logger.warning(f"Failed to log training scatter plot: {e}")
            
            # Error histogram
            errors = predictions - targets
            try:
                wandb.log({"train_prediction_error_histogram": wandb.Histogram(errors)}, commit=False)
            except Exception as e:
                logger.warning(f"Failed to log training error histogram: {e}")
            
            # Sample predictions table (first 100)
            sample_limit = min(100, len(predictions))
            table_data = [
                [i, targets[i], predictions[i], errors[i], errors[i] / max(abs(targets[i]), 1e-6)]
                for i in range(sample_limit)
            ]
            try:
                wandb.log({
                    "train_predictions_table": wandb.Table(
                        data=table_data,
                        columns=["Sample", "Ground Truth", "Prediction", "Error", "Relative Error"]
                    )
                }, commit=True)  # Commit here - this is the final log for this epoch
            except Exception as e:
                logger.warning(f"Failed to log training predictions table: {e}")
        
        # Validate
        if (epoch + 1) % config.mlops.eval_frequency == 0:
            val_metrics = trainer.validate(val_loader)
            val_mse = val_metrics["mse"]
            logger.info(
                f"Epoch {epoch+1:3d} | "
                f"Val MSE {val_mse:.6f} | "
                f"Val RMSE {val_metrics['rmse']:.6f} | "
                f"Val MAE {val_metrics['mae']:.6f} | "
                f"Val R² {val_metrics['r2']:.6f}"
            )
            
            # Log comprehensive metrics to W&B
            if wandb is not None and wandb.run is not None:
                # Log core metrics first (don't commit yet)
                val_log_dict = {
                    "epoch": epoch + 1,
                    "val_mse": val_metrics["mse"],
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "val_r2": val_metrics["r2"],
                    "val_correlation": val_metrics["correlation"],
                    "val_prediction_bias": val_metrics["prediction_error_mean"],
                    "val_prediction_error_std": val_metrics["prediction_error_std"],
                    "val_pred_min": val_metrics["pred_min"],
                    "val_pred_max": val_metrics["pred_max"],
                    "val_target_min": val_metrics["target_min"],
                    "val_target_max": val_metrics["target_max"],
                }
                
                # Add GPU memory stats
                if torch.cuda.is_available():
                    val_log_dict["val_gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
                    val_log_dict["val_gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
                    val_log_dict["val_gpu_memory_percent"] = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
                
                wandb.log(val_log_dict, commit=False)
                
                # Create prediction error scatter plot (ground truth vs predictions)
                predictions = val_metrics["predictions"].numpy()
                targets = val_metrics["targets"].numpy()
                
                # Create scatter plot for first 500 samples (memory efficiency)
                plot_limit = min(500, len(predictions))
                try:
                    wandb_plot = wandb.plot.scatter(
                        wandb.Table(data=[
                            [x, y] for x, y in zip(targets[:plot_limit].tolist(), predictions[:plot_limit].tolist())
                        ], columns=["Ground Truth", "Prediction"]),
                        "Ground Truth", "Prediction", title="[VAL] Predictions vs Ground Truth"
                    )
                    wandb.log({"val_predictions_scatter": wandb_plot}, commit=False)
                except Exception as e:
                    logger.warning(f"Failed to log validation scatter plot: {e}")
                
                # Create histogram of prediction errors
                errors = predictions - targets
                try:
                    wandb.log({"val_prediction_error_histogram": wandb.Histogram(errors)}, commit=False)
                except Exception as e:
                    logger.warning(f"Failed to log validation error histogram: {e}")
                
                # Log actual values as table for samples (first 100 samples for inspection)
                sample_limit = min(100, len(predictions))
                table_data = [
                    [i, targets[i], predictions[i], errors[i], errors[i] / max(abs(targets[i]), 1e-6)]
                    for i in range(sample_limit)
                ]
                try:
                    wandb.log({
                        "val_predictions_table": wandb.Table(
                            data=table_data,
                            columns=["Sample", "Ground Truth", "Prediction", "Error", "Relative Error"]
                        )
                    }, commit=True)  # Commit after validation logging
                except Exception as e:
                    logger.warning(f"Failed to log validation predictions table: {e}")
            
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
    
    # ==================== TEST EVALUATION ====================
    logger.info("\n" + "=" * 80)
    logger.info("Evaluating on Test Set...")
    logger.info("=" * 80)
    
    test_metrics = trainer.validate(test_loader)
    test_mse = test_metrics["mse"]
    
    logger.info(
        f"Test Results:\n"
        f"  MSE:  {test_metrics['mse']:.6f}\n"
        f"  RMSE: {test_metrics['rmse']:.6f}\n"
        f"  MAE:  {test_metrics['mae']:.6f}\n"
        f"  R²:   {test_metrics['r2']:.6f}\n"
        f"  Correlation: {test_metrics['correlation']:.6f}\n"
        f"  Prediction Bias: {test_metrics['prediction_error_mean']:.6f}\n"
        f"  Prediction Error Std: {test_metrics['prediction_error_std']:.6f}"
    )
    
    # Log comprehensive test metrics to W&B
    if wandb is not None and wandb.run is not None:
        # Log core metrics first
        test_log_dict = {
            "test_mse": test_metrics["mse"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "test_correlation": test_metrics["correlation"],
            "test_prediction_bias": test_metrics["prediction_error_mean"],
            "test_prediction_error_std": test_metrics["prediction_error_std"],
            "test_pred_min": test_metrics["pred_min"],
            "test_pred_max": test_metrics["pred_max"],
            "test_target_min": test_metrics["target_min"],
            "test_target_max": test_metrics["target_max"],
        }
        
        # Add GPU memory stats
        if torch.cuda.is_available():
            test_log_dict["test_gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
            test_log_dict["test_gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
            test_log_dict["test_gpu_memory_percent"] = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
        
        wandb.log(test_log_dict, commit=False)
        
        # Create prediction error scatter plot (ground truth vs predictions)
        predictions = test_metrics["predictions"].numpy()
        targets = test_metrics["targets"].numpy()
        
        # Create scatter plot for first 500 samples (memory efficiency)
        plot_limit = min(500, len(predictions))
        try:
            wandb_plot = wandb.plot.scatter(
                wandb.Table(data=[
                    [x, y] for x, y in zip(targets[:plot_limit].tolist(), predictions[:plot_limit].tolist())
                ], columns=["Ground Truth", "Prediction"]),
                "Ground Truth", "Prediction", title="[TEST] Predictions vs Ground Truth"
            )
            wandb.log({"test_predictions_scatter": wandb_plot}, commit=False)
        except Exception as e:
            logger.warning(f"Failed to log test scatter plot: {e}")
        
        # Create histogram of prediction errors
        errors = predictions - targets
        try:
            wandb.log({"test_prediction_error_histogram": wandb.Histogram(errors)}, commit=False)
        except Exception as e:
            logger.warning(f"Failed to log test error histogram: {e}")
        
        # Log actual values as table for samples (first 100 samples for inspection)
        sample_limit = min(100, len(predictions))
        table_data = [
            [i, targets[i], predictions[i], errors[i], errors[i] / max(abs(targets[i]), 1e-6)]
            for i in range(sample_limit)
        ]
        try:
            wandb.log({
                "test_predictions_table": wandb.Table(
                    data=table_data,
                    columns=["Sample", "Ground Truth", "Prediction", "Error", "Relative Error"]
                )
            }, commit=True)  # Final commit after test evaluation
        except Exception as e:
            logger.warning(f"Failed to log test predictions table: {e}")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Best Val Loss: {trainer.best_val_loss:.6f} (Epoch {trainer.best_epoch})")
    logger.info(f"Test MSE: {test_metrics['mse']:.6f}")
    logger.info(f"Test R²: {test_metrics['r2']:.6f}")
    logger.info(f"Best Model Checkpoint: {config.mlops.checkpoint_dir / f'{config.mlops.wandb_run_name}_best.pt'}")
    logger.info("=" * 80)
    
    # Finish W&B
    if wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multimodal crypto sentiment model")
    parser.add_argument("--asset", choices=["MULTI"], default="MULTI", help="Cryptocurrency asset (multi-asset: BTC+ETH combined)")
    parser.add_argument("--features-dir", type=str, default="./data/features", help="Local path to pre-extracted Kaggle features (contains text_embeddings_*.pt, image_embeddings_*.pt, tabular_features_scaled_*.pt, target_scores_scaled_*.pt)")
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
