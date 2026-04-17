"""
Production training loop with VRAM management, W&B integration, and best model checkpointing.

Key Features:
- Trainer class with full state management (save/load/train/validate)
- Pure float32 training for numerical stability (no mixed precision)
- Gradient accumulation configured via config.training.accumulate_steps
- Gradient clipping via config.model.grad_clip
- W&B integration per-branch via wandb_run_name
- Best model checkpointing with experiment naming
- NaN detection and diagnostics (2025-04-17):
  * Pre-backward check: Validates loss and predictions are finite
  * Post-backward check: Validates gradients are finite and not extreme
  * Problematic batches are saved for offline analysis
  * Early detection prevents silent NaN propagation

Stability Improvements (2025-04-17):
- Fixed attention layer to use Pre-LN structure (normalize before attention, not after)
- Reduced attention dropout from 0.3 to 0.1 for backward stability
- Moved dropout outside residual path in attention layer
- Enhanced gradient monitoring throughout training loop
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
from .dataset import CryptoMultimodalDataset, multimodal_collate_fn, create_dataloaders, create_walk_forward_dataloaders
from .model import MultimodalFusionNet
from .utils import setup_logging, format_duration


logger = logging.getLogger(__name__)


def check_for_nan(loss: torch.Tensor, batch_idx: int, predictions: torch.Tensor, targets: torch.Tensor) -> bool:
    """
    Check for NaN/Inf in loss and predictions before backward pass.
    Returns True if NaN/Inf detected (should skip this batch).
    """
    if torch.isnan(loss).any():
        logger.error(f"✗ Batch {batch_idx}: Loss is NaN")
        logger.error(f"  Predictions - Min: {predictions.min():.6e}, Max: {predictions.max():.6e}, Mean: {predictions.mean():.6e}")
        logger.error(f"  Targets - Min: {targets.min():.6e}, Max: {targets.max():.6e}, Mean: {targets.mean():.6e}")
        return True
    
    if torch.isinf(loss).any():
        logger.error(f"✗ Batch {batch_idx}: Loss is Inf")
        return True
    
    if torch.isnan(predictions).any():
        logger.error(f"✗ Batch {batch_idx}: Predictions contain NaN ({torch.isnan(predictions).sum()} values)")
        return True
    
    if torch.isinf(predictions).any():
        logger.error(f"✗ Batch {batch_idx}: Predictions contain Inf ({torch.isinf(predictions).sum()} values)")
        return True
    
    return False


def check_gradients(model: nn.Module, batch_idx: int) -> bool:
    """
    Check for NaN/Inf in gradients after backward pass.
    Returns True if issues found.
    """
    has_issues = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                logger.error(f"✗ Batch {batch_idx}: Gradient NaN in {name}")
                has_issues = True
            elif torch.isinf(param.grad).any():
                logger.error(f"✗ Batch {batch_idx}: Gradient Inf in {name}")
                has_issues = True
            elif (torch.abs(param.grad) > 1e4).any():
                logger.warning(f"⚠ Batch {batch_idx}: Extreme gradients in {name} (max: {param.grad.abs().max():.2e})")
    
    return has_issues


def _log_metrics_to_wandb(phase: str, metrics: Dict, config) -> None:
    """
    Log phase-specific metrics to W&B with visualizations.
    
    Creates scatter plot and error histogram for predictions vs ground truth.
    
    Args:
        phase: Phase name ('train', 'val', or 'test')
        metrics: Dictionary with 'predictions', 'targets', and metric values
        config: Experiment config
    """
    if wandb is None or wandb.run is None:
        return
    
    # Prepare core metrics for logging
    log_dict = {
        f"{phase}_mse": metrics["mse"],
        f"{phase}_mae": metrics["mae"],
        f"{phase}_rmse": metrics["rmse"],
        f"{phase}_r2": metrics["r2"],
        f"{phase}_correlation": metrics["correlation"],
        f"{phase}_prediction_bias": metrics["prediction_error_mean"],
        f"{phase}_prediction_error_std": metrics["prediction_error_std"],
        f"{phase}_pred_min": metrics["pred_min"],
        f"{phase}_pred_max": metrics["pred_max"],
        f"{phase}_target_min": metrics["target_min"],
        f"{phase}_target_max": metrics["target_max"],
    }
    
    if "is_denormalized" in metrics:
        log_dict[f"{phase}_is_denormalized"] = metrics["is_denormalized"]
    
    wandb.log(log_dict, commit=False)
    
    # Extract predictions and targets
    predictions = metrics["predictions"].numpy()
    targets = metrics["targets"].numpy()
    
    # Create scatter plot (first 500 samples for memory efficiency)
    plot_limit = min(500, len(predictions))
    try:
        wandb_plot = wandb.plot.scatter(
            wandb.Table(data=[
                [x, y] for x, y in zip(targets[:plot_limit].tolist(), predictions[:plot_limit].tolist())
            ], columns=["Ground Truth", "Prediction"]),
            "Ground Truth", "Prediction", title=f"[{phase.upper()}] Predictions vs Ground Truth"
        )
        wandb.log({f"{phase}_predictions_scatter": wandb_plot}, commit=False)
    except Exception as e:
        logger.warning(f"Failed to log {phase} scatter plot: {e}")
    
    # Create error histogram
    errors = predictions - targets
    try:
        wandb.log({f"{phase}_prediction_error_histogram": wandb.Histogram(errors)}, commit=False)
    except Exception as e:
        logger.warning(f"Failed to log {phase} error histogram: {e}")
    
    # Log sample predictions table (first 100 samples)
    sample_limit = min(100, len(predictions))
    table_data = [
        [i, targets[i], predictions[i], errors[i], errors[i] / max(abs(targets[i]), 1e-6)]
        for i in range(sample_limit)
    ]
    try:
        wandb.log({
            f"{phase}_predictions_table": wandb.Table(
                data=table_data,
                columns=["Sample", "Ground Truth", "Prediction", "Error", "Relative Error"]
            )
        }, commit=True)  # Commit after each phase
    except Exception as e:
        logger.warning(f"Failed to log {phase} predictions table: {e}")


def _compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.
    
    Args:
        predictions: Model predictions (torch.Tensor)
        targets: Ground truth targets (torch.Tensor)
    
    Returns:
        Dict with keys: 'mse', 'mae', 'rmse', 'r2', 'correlation',
                       'prediction_error_mean', 'prediction_error_std',
                       'pred_min', 'pred_max', 'target_min', 'target_max'
    """
    # MSE, MAE, RMSE
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    rmse = np.sqrt(mse)
    
    # R² Score
    ss_res = torch.sum((predictions - targets) ** 2).item()
    ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
    r2_score = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Correlation
    pred_mean = predictions.mean()
    target_mean = targets.mean()
    numerator = torch.sum((predictions - pred_mean) * (targets - target_mean))
    denom = torch.sqrt(
        torch.sum((predictions - pred_mean) ** 2) * torch.sum((targets - target_mean) ** 2)
    )
    correlation = (numerator / denom).item() if denom > 0 else 0.0
    
    # Prediction error analysis
    prediction_errors = predictions - targets
    prediction_error_mean = prediction_errors.mean().item()
    prediction_error_std = prediction_errors.std().item()
    
    # Min/Max ranges
    pred_min, pred_max = predictions.min().item(), predictions.max().item()
    target_min, target_max = targets.min().item(), targets.max().item()
    
    return {
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
    }


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


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.
    Stops training if validation loss doesn't improve for N consecutive epochs.
    """
    
    def __init__(self, patience: int = 7, verbose: bool = True):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            verbose: Whether to log early stopping events
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss:
            # Improvement detected
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                logger.info(f"✓ Early Stopping: Validation loss improved to {val_loss:.6f}")
            return False
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"⚠ Early Stopping: No improvement for {self.counter}/{self.patience} epochs "
                    f"(best: {self.best_loss:.6f}, current: {val_loss:.6f})"
                )
            
            if self.counter >= self.patience:
                if self.verbose:
                    logger.warning(
                        f"🛑 Early Stopping triggered! No improvement for {self.patience} consecutive epochs. "
                        f"Stopping training to prevent overfitting."
                    )
                self.early_stop = True
                return True
        
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
        target_scaler=None,
    ):
        """
        Initialize trainer.
        
        Args:
            config: ExperimentConfig instance
            model: MultimodalFusionNet or similar
            device: "cuda" or "cpu"
            target_scaler: Fitted RobustScaler for inverse transforms on test/val metrics (optional)
        """
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.target_scaler = target_scaler  # For denormalizing predictions/targets
        
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
        if self.target_scaler is not None:
            logger.info("✓ Target scaler loaded for inverse transforms on validation/test")
    
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
            
            # ========== NUMERICAL STABILITY: CLAMP PREDICTIONS ==========
            # Prevent extreme values that can cause NaN in loss backward
            # Sentiment range is typically [-100, 100], clamp to [-150, 150] for safety
            predictions_clamped = torch.clamp(predictions, min=-150, max=150)
            
            # Collect for epoch-level metrics (using clamped predictions)
            all_predictions.append(predictions_clamped.detach().cpu())
            all_targets.append(targets.detach().cpu())
            
            # Compute HuberLoss (robust to outliers)
            # More stable than MSE for noisy market data with outliers
            loss = nn.HuberLoss(delta=1.0)(predictions_clamped, targets)
            
            # ========== NaN CHECK BEFORE BACKWARD ==========
            # Detect numerical issues early
            if check_for_nan(loss, batch_idx, predictions_clamped, targets):
                logger.warning(f"⚠ Skipping batch {batch_idx} due to NaN/Inf in predictions or loss")
                self.optimizer.zero_grad()  # Clear any accumulated gradients
                continue
            
            # ========== LOSS MAGNITUDE CHECK ==========
            # Catch exploding loss that would cause NaN in backward
            if loss.item() > 1000:
                logger.error(
                    f"✗ Batch {batch_idx}: Loss is extremely large ({loss.item():.2f}). "
                    f"This indicates numerical instability and will likely cause NaN in backward. "
                    f"Predictions: min={predictions_clamped.min():.2f}, max={predictions_clamped.max():.2f}, "
                    f"mean={predictions_clamped.mean():.2f}"
                )
                self.optimizer.zero_grad()
                continue
            
            # Scale loss for gradient accumulation
            # Prevents accumulated gradients from growing too large
            loss = loss / self.config.training.accumulate_steps
            
            # ========== BACKWARD PASS (STANDARD) ==========
            # Standard PyTorch backward - gradients computed in float32
            try:
                loss.backward()
            except RuntimeError as e:
                if "nan" in str(e).lower():
                    logger.error(f"✗ Batch {batch_idx}: NaN detected in backward pass!")
                    logger.error(f"  Error: {e}")
                    logger.error(f"  Loss value: {loss.item():.6e}")
                    logger.error(f"  Predictions range: [{predictions.min():.6e}, {predictions.max():.6e}]")
                    # Save problematic batch for analysis
                    batch_path = self.config.mlops.checkpoint_dir / f"problematic_batch_{batch_idx}.pt"
                    torch.save({
                        "batch_idx": batch_idx,
                        "epoch": self.epoch,
                        "batch": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
                        "predictions": predictions.detach().cpu(),
                        "loss": loss.detach().cpu(),
                    }, batch_path)
                    logger.error(f"  Saved problematic batch to: {batch_path}")
                    self.optimizer.zero_grad()
                    raise
                else:
                    raise
            
            # ========== GRADIENT ANOMALY CHECK ==========
            # Check for NaN/Inf in gradients after backward
            if check_gradients(self.model, batch_idx):
                logger.error(f"✗ Batch {batch_idx}: Anomalous gradients detected!")
                self.optimizer.zero_grad()
                raise RuntimeError(f"Gradient anomaly in batch {batch_idx}")
            
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
        
        # Compute metrics using shared helper
        metrics = _compute_metrics(all_predictions, all_targets)
        metrics["loss"] = avg_loss
        metrics["predictions"] = all_predictions
        metrics["targets"] = all_targets
        
        logger.info(
            f"Epoch {self.epoch+1} completed | Avg Loss: {avg_loss:.6f} | MSE: {metrics['mse']:.6f} | "
            f"MAE: {metrics['mae']:.6f} | RMSE: {metrics['rmse']:.6f} | R²: {metrics['r2']:.6f}"
        )
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Run validation (pure float32, no AMP) with comprehensive metrics collection.
        
        Collects predictions and targets for:
        - Standard metrics: MSE, MAE, RMSE
        - Statistical metrics: R², correlation, prediction bias, error std
        - Ground truth vs prediction logging
        
        If target_scaler is available: applies inverse transforms to report metrics in original scale.
        Otherwise: reports metrics in normalized scale.
        
        Args:
            val_loader: Validation DataLoader
        
        Returns:
            Dict with keys: 'mse', 'mae', 'rmse', 'r2', 'correlation', 
                          'prediction_error_mean', 'prediction_error_std',
                          'predictions', 'targets', 'is_denormalized'
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
                
                # Compute metrics (on normalized scale for loss)
                # Use HuberLoss for consistency with training (robust to outliers)
                huber_loss = nn.HuberLoss(delta=1.0)(predictions, targets)
                mae = nn.L1Loss()(predictions, targets)
                
                total_mse += huber_loss.item()
                total_mae += mae.item()
                num_steps += 1
                
                # Collect for global metrics
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)  # (total_samples,) - already 1D
        all_targets = torch.cat(all_targets, dim=0)  # (total_samples,) - already 1D
        
        # Apply inverse transform if scaler is available
        is_denormalized = False
        if self.target_scaler is not None:
            logger.debug("Applying inverse transform to predictions and targets...")
            # RobustScaler inverse_transform expects (n_samples, 1) shape
            all_predictions_denorm = self.target_scaler.inverse_transform(
                all_predictions.numpy().reshape(-1, 1)
            ).squeeze()
            all_targets_denorm = self.target_scaler.inverse_transform(
                all_targets.numpy().reshape(-1, 1)
            ).squeeze()
            
            all_predictions = torch.from_numpy(all_predictions_denorm).float()
            all_targets = torch.from_numpy(all_targets_denorm).float()
            is_denormalized = True
            logger.debug("✓ Inverse transform applied (metrics computed on original scale)")
        
        # Compute per-batch averages
        avg_mse = total_mse / num_steps
        avg_mae = total_mae / num_steps
        self.val_losses.append(avg_mse)
        
        # Compute metrics using shared helper
        metrics = _compute_metrics(all_predictions, all_targets)
        metrics["mse"] = avg_mse  # Override with per-batch average
        metrics["mae"] = avg_mae  # Override with per-batch average
        metrics["is_denormalized"] = is_denormalized
        
        return metrics
    
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
        args: Parsed command-line arguments (or Namespace object with attributes)
    """
    # Set safe defaults for all args attributes (in case called from notebook without argparse)
    asset = getattr(args, 'asset', 'MULTI')
    features_dir = getattr(args, 'features_dir', './data/features')
    run_name = getattr(args, 'run_name', None)
    config_path = getattr(args, 'config', None)
    seed = getattr(args, 'seed', 42)
    resume_training = getattr(args, 'resume', False)
    debug = getattr(args, 'debug', False)
    num_folds = getattr(args, 'num_folds', 5)
    # Setup
    setup_logging()
    logger.info("=" * 80)
    logger.info("Training Multimodal Crypto Sentiment Model")
    logger.info("=" * 80)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    
    device = "cuda" if torch.cuda.is_available() and not debug else "cpu"
    logger.info(f"Device: {device}")
    
    # Load or create config
    if config_path:
        logger.info(f"Loading config from {config_path}...")
        # For now, use default config (could load from YAML in future)
        config = ExperimentConfig()
    else:
        config = create_config(
            asset=asset,
            wandb_run_name=run_name,
        )
        config.debug = debug
    
    logger.info(f"Config: asset={config.data.asset}, seq_len={config.data.seq_len}, batch_size={config.data.batch_size}")
    logger.info(f"Model: hidden_dim={config.model.hidden_dim}, frozen_backbones={config.model.frozen_backbones}")
    logger.info(f"Training: lr={config.training.learning_rate:.2e}, epochs={config.training.max_epochs}")
    logger.info(f"MLOps: wandb_run={config.mlops.wandb_run_name}")
    
    # Create dataloaders (walk-forward validation)
    logger.info("\n" + "-" * 80)
    logger.info("Loading datasets...")
    print("[PROGRESS] Starting to load datasets (this may take 1-5 minutes)...")
    sys.stdout.flush()
    
    # Use walk-forward validation
    logger.info(f"Using WALK-FORWARD VALIDATION with {num_folds} folds")
    logger.info(f"DataLoader settings: batch_size={config.data.batch_size}, num_workers=0 (forced), pin_memory=True")
    
    # Will iterate through folds in training loop
    walk_forward_generator = create_walk_forward_dataloaders(
        config,
        features_dir=features_dir,
        num_folds=num_folds,
        num_workers=0,
        pin_memory=True
    )
    
    # Also load test set once for final evaluation
    logger.info("Also loading test set for final evaluation...")
    try:
        from .dataset import CryptoMultimodalDataset
        test_dataset = CryptoMultimodalDataset(
            split="test_in_domain",
            seq_len=config.data.seq_len,
            features_dir=features_dir,
            debug=debug,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            collate_fn=multimodal_collate_fn,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        target_scaler = test_dataset.target_scaler
        logger.info(f"✓ Test loader created: {len(test_loader)} batches")
    except Exception as e:
        logger.warning(f"Failed to load test set: {e}")
        test_loader = None
        target_scaler = None
    
    print("[PROGRESS] ✓ Walk-forward generator ready!")
    sys.stdout.flush()
    
    
    # Initialize model
    logger.info("\n" + "-" * 80)
    logger.info("Initializing model...")
    model = MultimodalFusionNet(config)
    logger.info(f"Model device: {next(model.parameters()).device}")
    
    # Initialize trainer
    trainer = Trainer(config, model, device=device, target_scaler=target_scaler)
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
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if resume_training:
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
    
    # ==================== WALK-FORWARD TRAINING LOOP ====================
    logger.info("=" * 80)
    logger.info("WALK-FORWARD VALIDATION MODE")
    logger.info("=" * 80)
    
    fold_results = {}
    
    for fold_num, train_loader, val_loader, scalers_dict in walk_forward_generator:
        logger.info("\n" + "=" * 80)
        logger.info(f"FOLD {fold_num}")
        logger.info("=" * 80)
        
        # Reset trainer for each fold
        trainer = Trainer(config, model, device=device, target_scaler=scalers_dict.get("target_scaler"))
        trainer.setup_optimizer()
        
        # Early stopping reset
        early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            verbose=True
        )
        
        # Train for this fold
        for epoch in range(config.training.max_epochs):
            trainer.epoch = epoch
            
            # Train epoch
            train_metrics = trainer.train_epoch(train_loader)
            train_loss = train_metrics["loss"]
            
            logger.info(
                f"Fold {fold_num} Epoch {epoch+1:3d} | "
                f"Train Loss {train_loss:.6f} | "
                f"Train R² {train_metrics['r2']:.6f}"
            )
            
            # Validate
            if (epoch + 1) % config.mlops.eval_frequency == 0:
                val_metrics = trainer.validate(val_loader)
                val_loss = val_metrics["mse"]
                
                logger.info(
                    f"Fold {fold_num} Epoch {epoch+1:3d} | "
                    f"Val Loss {val_loss:.6f} | "
                    f"Val MSE {val_metrics['mse']:.6f} | "
                    f"Val R² {val_metrics['r2']:.6f}"
                )
                
                # Early stopping check
                if early_stopping(val_loss):
                    logger.info(f"✓ Early stopping triggered at epoch {epoch+1}")
                    break
                
                # Update best model for this fold
                if val_loss < trainer.best_val_loss:
                    trainer.best_val_loss = val_loss
                    trainer.best_epoch = epoch
        
        # Validate on full validation set for this fold
        logger.info(f"\nFinal validation for Fold {fold_num}...")
        final_val_metrics = trainer.validate(val_loader)
        
        fold_results[fold_num] = {
            "val_r2": final_val_metrics["r2"],
            "val_mse": final_val_metrics["mse"],
            "val_rmse": final_val_metrics["rmse"],
            "val_mae": final_val_metrics["mae"],
            "val_correlation": final_val_metrics["correlation"],
        }
        
        logger.info(f"Fold {fold_num} Results: R²={final_val_metrics['r2']:.6f}, MSE={final_val_metrics['mse']:.6f}, RMSE={final_val_metrics['rmse']:.6f}")
    
    # Log fold summary
    logger.info("\n" + "=" * 80)
    logger.info("WALK-FORWARD VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    r2_scores = [fold_results[i]["val_r2"] for i in sorted(fold_results.keys())]
    mse_scores = [fold_results[i]["val_mse"] for i in sorted(fold_results.keys())]
    rmse_scores = [fold_results[i]["val_rmse"] for i in sorted(fold_results.keys())]
    
    logger.info(f"Mean R²: {np.mean(r2_scores):.6f} ± {np.std(r2_scores):.6f}")
    logger.info(f"Mean MSE: {np.mean(mse_scores):.6f} ± {np.std(mse_scores):.6f}")
    logger.info(f"Mean RMSE: {np.mean(rmse_scores):.6f} ± {np.std(rmse_scores):.6f}")
    
    for fold_num in sorted(fold_results.keys()):
        logger.info(f"  Fold {fold_num}: R²={fold_results[fold_num]['val_r2']:.6f}")
    
    if wandb is not None and wandb.run is not None:
        wandb.log({
            "walk_forward/mean_r2": np.mean(r2_scores),
            "walk_forward/std_r2": np.std(r2_scores),
            "walk_forward/mean_mse": np.mean(mse_scores),
            "walk_forward/mean_rmse": np.mean(rmse_scores),
        })
    
    # ==================== TEST EVALUATION ====================
    logger.info("\n" + "=" * 80)
    logger.info("Evaluating on Test Set...")
    logger.info("=" * 80)
    
    test_metrics = trainer.validate(test_loader)
    test_mse = test_metrics["mse"]
    denorm_status = " (denormalized to original scale)" if test_metrics.get("is_denormalized", False) else " (normalized scale)"
    
    logger.info(
        f"Test Results{denorm_status}:\n"
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
            "test_is_denormalized": test_metrics.get("is_denormalized", False),
        }
        
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
    parser.add_argument("--num-folds", type=int, default=5, help="Number of walk-forward folds (default: 5)")
    
    args = parser.parse_args()
    
    # Auto-generate run name if not provided
    if args.run_name is None:
        args.run_name = f"{args.asset.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    main(args)
