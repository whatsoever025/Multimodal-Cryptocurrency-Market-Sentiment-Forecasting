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

# NOTE: set_detect_anomaly is enabled only in debug mode (gated in main()).
# Leaving it on globally slows training by 2-5x due to per-op NaN checking.

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
from .dataset import multimodal_collate_fn, create_walk_forward_dataloaders
from .model import MultimodalFusionNet
from .utils import setup_logging, format_duration


logger = logging.getLogger(__name__)

# v5 target columns — one independent model is trained per target
TARGET_NAMES = ["y_baseline", "y_heuristic", "y_vol_adj_return"]


def _reset_weights(module: nn.Module) -> None:
    """
    Reinitialize the weights of a module to their default distributions.
    
    Called via `model.apply(_reset_weights)` at the start of each walk-forward fold
    to ensure each fold trains from a fresh initialization.
    
    WHY THIS MATTERS: Without per-fold reset, fold N's model starts from fold N-1's
    trained weights. By the final fold the model has implicitly seen ALL prior folds'
    train+val data, making validation R² optimistically biased relative to test R².
    
    Handles:
    - nn.Linear: kaiming_uniform (matches PyTorch default)
    - nn.LSTM: orthogonal for weight_hh, kaiming for weight_ih, zeros for bias
    - nn.LayerNorm: ones for weight, zeros for bias
    - nn.Parameter (fusion_token): normal(0, 0.02) matching model.py initialization
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)  # Orthogonal init for recurrent weights (stable gradients)
            elif 'bias' in name:
                nn.init.zeros_(param)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    
    # Reset learnable [FUSION] token if present on the module
    if hasattr(module, "fusion_token") and isinstance(module.fusion_token, nn.Parameter):
        nn.init.normal_(module.fusion_token, mean=0.0, std=0.02)

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
            # Use separate `if` (not `elif`) so Inf is independently checked
            # even when NaN is also present in the same gradient tensor.
            if torch.isnan(param.grad).any():
                logger.error(f"✗ Batch {batch_idx}: Gradient NaN in {name}")
                has_issues = True
            if torch.isinf(param.grad).any():
                logger.error(f"✗ Batch {batch_idx}: Gradient Inf in {name}")
                has_issues = True
            if not has_issues and (torch.abs(param.grad) > 1e4).any():
                logger.warning(f"⚠ Batch {batch_idx}: Extreme gradients in {name} (max: {param.grad.abs().max():.2e})")
    
    return has_issues


def compute_train_targets_mean(train_loader: DataLoader, target_idx: int = 0) -> float:
    """
    Compute mean of training targets for a specific target variable.
    Used as benchmark for R² OOS historical mean formula.
    
    Args:
        train_loader: Training DataLoader
        target_idx: Target index (0=y_baseline, 1=y_heuristic, 2=y_vol_adj_return)
    
    Returns:
        Mean of training targets (scalar float)
    """
    all_targets = []
    for batch in train_loader:
        targets = batch["target"][:, target_idx]  # (batch_size,)
        all_targets.append(targets)
    
    all_targets = torch.cat(all_targets, dim=0)
    train_mean = all_targets.mean().item()
    return train_mean


def _compute_metrics(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    target_name: Optional[str] = None,
    train_targets_mean: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics with target-specific R² OOS formula.
    
    Args:
        predictions: Model predictions (torch.Tensor)
        targets: Ground truth targets (torch.Tensor)
        target_name: One of "y_baseline", "y_heuristic", "y_vol_adj_return"
                     Determines which R² OOS benchmark to use.
        train_targets_mean: Mean of training targets (for historical mean benchmark).
                           Required if target_name is "y_baseline".
    
    Returns:
        Dict with keys: 'mse', 'mae', 'rmse',
                       'r2'     - Standard R² (benchmarks against test-set mean ȳ)
                       'r2_oos' - OOS R² with target-specific benchmark:
                                  * y_vol_adj_return: Zero-predictor (Gu, Kelly & Xiu 2020)
                                    Benchmark: ŷ=0 (not test mean, since true mean ≈ 0)
                                  * y_baseline: Historical mean (structural funding rate positive bias)
                                    Benchmark: ŷ=mean(y_train)
                                  * y_heuristic: Zero-predictor (Z-score normalized, mean ≈ 0)
                                    Benchmark: ŷ=0
                       'r2_oos_benchmark_zero' - Always GKX 2020 (debug)
                       'r2_oos_benchmark_historical_mean' - Always historical mean (debug)
                       'correlation', 'prediction_error_mean', 'prediction_error_std',
                       'pred_min', 'pred_max', 'target_min', 'target_max'
    """
    # MSE, MAE, RMSE
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    rmse = np.sqrt(mse)
    
    # Residual sum of squares (shared by all R² variants)
    ss_res = torch.sum((predictions - targets) ** 2).item()

    # --- Standard R² ---
    # Denominator: sum((y - ȳ)²) — benchmarks against predicting the test-set mean.
    ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
    r2_score = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # --- R² OOS Variant 1: GKX 2020 (Zero-predictor) ---
    # Benchmark: ŷ_i = 0 (appropriate for returns with mean ≈ 0)
    ss_tot_oos_gkx = torch.sum(targets ** 2).item()
    r2_oos_gkx = 1.0 - (ss_res / ss_tot_oos_gkx) if ss_tot_oos_gkx > 0 else 0.0

    # --- R² OOS Variant 2: Historical Mean Benchmark ---
    # Benchmark: ŷ_i = mean(y_train) (appropriate for series with structural mean ≠ 0)
    # Formula: 1 - SS_res / sum((y_test - mean(y_train))²)
    if train_targets_mean is not None:
        ss_tot_oos_hist = torch.sum((targets - train_targets_mean) ** 2).item()
        r2_oos_hist = 1.0 - (ss_res / ss_tot_oos_hist) if ss_tot_oos_hist > 0 else 0.0
    else:
        r2_oos_hist = 0.0
    
    # --- Select appropriate R² OOS formula based on target ---
    # (See docstring for rationale)
    if target_name == "y_baseline":
        # Funding rate: has structural positive bias, not mean=0
        # Benchmark: mean(y_train)
        r2_oos_primary = r2_oos_hist
    elif target_name == "y_heuristic":
        # Z-score normalized heuristic: mean ≈ 0
        # Both formulas equivalent; use GKX for consistency
        r2_oos_primary = r2_oos_gkx
    elif target_name == "y_vol_adj_return":
        # 1h return: extremely noisy, mean ≈ 0.0001
        # GKX 2020 is academically correct for financial returns
        r2_oos_primary = r2_oos_gkx
    else:
        # Fallback: use GKX
        r2_oos_primary = r2_oos_gkx
    
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
        "r2_oos": r2_oos_primary,
        "r2_oos_benchmark_zero": r2_oos_gkx,  # Debug: always GKX 2020
        "r2_oos_benchmark_historical_mean": r2_oos_hist,  # Debug: always historical mean
        "correlation": correlation,
        "prediction_error_mean": prediction_error_mean,
        "prediction_error_std": prediction_error_std,
        "pred_min": pred_min,
        "pred_max": pred_max,
        "target_min": target_min,
        "target_max": target_max,
    }

class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.
    Stops training if validation loss doesn't improve by at least min_delta
    for N consecutive epochs.
    
    min_delta is critical for financial time-series: without it, a noise improvement
    of 1e-6 resets the patience counter, causing the model to train far longer than
    necessary and memorize validation-period patterns.
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, verbose: bool = True):
        """
        Args:
            patience:  Number of epochs with no meaningful improvement before stopping
            min_delta: Minimum absolute improvement in val loss to count as 'improved'.
                       Prevents noise-level decreases from resetting the patience counter.
                       Default 1e-4 = loss must drop by at least 0.01% of its magnitude.
            verbose:   Whether to log early stopping events
        """
        self.patience = patience
        self.min_delta = min_delta
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
        
        if val_loss < self.best_loss - self.min_delta:
            # Meaningful improvement (exceeds min_delta threshold)
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                logger.info(f"✓ Early Stopping: Validation loss improved to {val_loss:.6f} (Δ > {self.min_delta})")
            return False
        else:
            # No meaningful improvement
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"⚠ Early Stopping: No improvement for {self.counter}/{self.patience} epochs "
                    f"(best: {self.best_loss:.6f}, current: {val_loss:.6f}, "
                    f"Δ={self.best_loss - val_loss:.2e} < min_delta={self.min_delta})"
                )
            
            if self.counter >= self.patience:
                if self.verbose:
                    logger.warning(
                        f"🛑 Early Stopping triggered! No meaningful improvement for {self.patience} "
                        f"consecutive epochs. Stopping to prevent overfitting."
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
        self.gradient_norms = []  # Track gradient norms for monitoring
        
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
          - Sufficient VRAM on Kaggle 16GB (BS=128, MULTI asset, offline features)
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
    
    def train_epoch(self, train_loader: DataLoader, target_idx: int = 0, train_targets_mean: Optional[float] = None) -> Dict[str, float]:
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
        - Kaggle 16GB VRAM sufficient for BS=128 with offline features
        
        ============================================================================
        
        VRAM Management:
        - Batch size 128 (MULTI asset: BTC+ETH combined, seq_len=24)
        - Frozen backbones + offline feature extraction → backbones NOT in VRAM
        - Only trainable components in VRAM: TabularEncoder, CrossModalAttn,
          Bottleneck, TemporalLSTM, PredictionHead (~few MB)
        - Gradient accumulation: accumulate_steps=2 → effective batch=256
        - Precision: Native float32 (no mixed precision)
        - Gradient clipping: max_norm=1.0 (prevents explosion in LSTM/Attention)
        
        Args:
            train_loader: Training DataLoader
        
        Returns:
            Dict with keys: 'loss', 'mse', 'mae', 'rmse', 'r2', 'r2_oos', 'correlation',
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
        
        epoch_gradient_norms = []  # Collect gradient norms for this epoch
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device (float32 by default)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # ========== EMBEDDING NOISE REGULARIZATION ==========
            # Add small Gaussian noise to pre-extracted embeddings during training only.
            # Prevents the model from memorizing specific embedding fingerprints from
            # training folds (a key overfitting mechanism when using frozen features).
            # std=0.01 is small relative to embedding scale (typically std≈0.1-0.5),
            # enough to act as regularization without distorting the semantic signal.
            if self.model.training and self.config.training.embedding_noise_std > 0:
                noise_std = self.config.training.embedding_noise_std
                batch = dict(batch)  # Shallow copy to avoid modifying the original batch dict
                batch["text_embedding"] = batch["text_embedding"] + torch.randn_like(batch["text_embedding"]) * noise_std
                batch["image_embedding"] = batch["image_embedding"] + torch.randn_like(batch["image_embedding"]) * noise_std

            # ========== FORWARD PASS (FLOAT32) ==========
            # Standard PyTorch forward pass - all tensors remain float32
            predictions = self.model(batch)  # (batch,) single-target | (batch, num_targets) multi-target
            num_targets = getattr(self.model, 'num_targets', 1)

            # ========== NUMERICAL STABILITY: CLAMP PREDICTIONS ==========
            predictions_clamped = torch.clamp(predictions, min=-150, max=150)

            if num_targets > 1:
                # Multi-target: compute mean Huber loss across all heads.
                # Collect metrics for the primary target (target_idx) for progress logging.
                targets_all = batch["target"][:, :num_targets]  # (batch, num_targets)
                head_losses = [
                    nn.HuberLoss(delta=self.config.training.huber_delta)(
                        predictions_clamped[:, i], targets_all[:, i]
                    )
                    for i in range(num_targets)
                ]
                loss = sum(head_losses) / num_targets
                # Track primary-target predictions for epoch-level metrics
                all_predictions.append(predictions_clamped[:, target_idx].detach().cpu())
                all_targets.append(targets_all[:, target_idx].detach().cpu())
            else:
                targets = batch["target"][:, target_idx]  # (batch,) — slice one target column
                all_predictions.append(predictions_clamped.detach().cpu())
                all_targets.append(targets.detach().cpu())
                loss = nn.HuberLoss(delta=self.config.training.huber_delta)(predictions_clamped, targets)
            
            # ========== NaN CHECK BEFORE BACKWARD ==========
            # For multi-target, check against the flat clamped tensor (same NaN surface).
            _check_preds = predictions_clamped.reshape(-1)
            _check_tgts  = (targets_all if num_targets > 1 else targets).reshape(-1)
            if check_for_nan(loss, batch_idx, _check_preds, _check_tgts):
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
                
                # Track gradient norm for monitoring (convert to scalar immediately)
                epoch_gradient_norms.append(total_norm.item() if isinstance(total_norm, torch.Tensor) else float(total_norm))
                
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
        
        # Guard against all-skipped epoch (every batch had NaN/Inf)
        if num_steps == 0 or len(all_predictions) == 0:
            logger.error(f"Epoch {self.epoch+1}: ALL batches were skipped (NaN/Inf). Returning zero metrics.")
            return {
                "loss": float("nan"), "mse": float("nan"), "mae": float("nan"),
                "rmse": float("nan"), "r2": float("nan"), "r2_oos": float("nan"),
                "correlation": float("nan"), "prediction_error_mean": float("nan"),
                "prediction_error_std": float("nan"), "pred_min": float("nan"),
                "pred_max": float("nan"), "target_min": float("nan"),
                "target_max": float("nan"), "predictions": torch.tensor([]),
                "targets": torch.tensor([]),
            }
        
        avg_loss = total_loss / num_steps
        self.train_losses.append(avg_loss)
        
        # Compute epoch-level gradient norm statistics
        # (All values in epoch_gradient_norms are already scalars, converted at append time)
        if epoch_gradient_norms:
            avg_gradient_norm = float(np.mean(epoch_gradient_norms))
            max_gradient_norm = float(np.max(epoch_gradient_norms))
        else:
            avg_gradient_norm = 0.0
            max_gradient_norm = 0.0
        self.gradient_norms.append(avg_gradient_norm)
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)  # (total_samples,) - already 1D
        all_targets = torch.cat(all_targets, dim=0)  # (total_samples,) - already 1D
        
        # Compute metrics using shared helper
        target_name = TARGET_NAMES[target_idx] if target_idx < len(TARGET_NAMES) else None
        metrics = _compute_metrics(all_predictions, all_targets, target_name=target_name, train_targets_mean=train_targets_mean)
        metrics["loss"] = avg_loss
        metrics["predictions"] = all_predictions
        metrics["targets"] = all_targets
        metrics["avg_gradient_norm"] = avg_gradient_norm
        metrics["max_gradient_norm"] = max_gradient_norm
        
        logger.info(
            f"Epoch {self.epoch+1} completed | Avg Loss: {avg_loss:.6f} | MSE: {metrics['mse']:.6f} | "
            f"MAE: {metrics['mae']:.6f} | RMSE: {metrics['rmse']:.6f} | "
            f"R²: {metrics['r2']:.6f} | R²_OOS: {metrics['r2_oos']:.6f} | "
            f"Avg Grad Norm: {avg_gradient_norm:.6f}"
        )
        
        return metrics
    
    def validate(self, val_loader: DataLoader, target_idx: int = 0, train_targets_mean: Optional[float] = None) -> Dict[str, float]:
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
            Dict with keys: 'mse', 'mae', 'rmse', 'r2', 'r2_oos', 'correlation', 
                          'prediction_error_mean', 'prediction_error_std',
                          'predictions', 'targets', 'is_denormalized',
                          'normalized_huber', 'normalized_mae'
        """
        self.model.eval()
        total_huber = 0.0  # Accumulates HuberLoss (NOT MSE) — used for val_losses tracking
        total_mae = 0.0
        num_steps = 0
        
        # Collect all predictions and targets for post-hoc analysis
        all_predictions = []
        all_targets = []
        all_is_post_etf = []
        
        num_targets = getattr(self.model, 'num_targets', 1)

        with torch.no_grad():
            for batch in val_loader:
                # Move to device (float32)
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass (pure float32, no AMP)
                predictions = self.model(batch)  # (batch,) or (batch, num_targets)

                if num_targets > 1:
                    targets_all = batch["target"][:, :num_targets]  # (batch, num_targets)
                    # Mean Huber loss across heads (mirrors training)
                    head_losses = [
                        nn.HuberLoss(delta=self.config.training.huber_delta)(
                            predictions[:, i], targets_all[:, i]
                        )
                        for i in range(num_targets)
                    ]
                    huber_loss = sum(head_losses) / num_targets
                    mae = sum(
                        nn.L1Loss()(predictions[:, i], targets_all[:, i])
                        for i in range(num_targets)
                    ) / num_targets
                    # Collect primary-target predictions for main metric path
                    all_predictions.append(predictions[:, target_idx].cpu())
                    all_targets.append(targets_all[:, target_idx].cpu())
                else:
                    targets = batch["target"][:, target_idx]  # (batch,) — single column
                    huber_loss = nn.HuberLoss(delta=self.config.training.huber_delta)(predictions, targets)
                    mae = nn.L1Loss()(predictions, targets)
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())

                total_huber += huber_loss.item()
                total_mae += mae.item()
                num_steps += 1

                # Extract is_post_ETF flag from tabular data (last timestep, col 6)
                # Since StandardScaler is used, the binary 0/1 becomes negative/positive
                is_post_etf_scaled = batch["tabular"][:, -1, 6].cpu()
                is_post_etf_binary = (is_post_etf_scaled > 0.0).float()
                all_is_post_etf.append(is_post_etf_binary)
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)  # (total_samples,) - already 1D
        all_targets = torch.cat(all_targets, dim=0)  # (total_samples,) - already 1D
        all_is_post_etf = torch.cat(all_is_post_etf, dim=0)
        
        # Per-batch averages on normalized scale — used for consistent early stopping / val_losses
        # These are HuberLoss-based, not MSE, and always on normalized scale.
        avg_huber = total_huber / num_steps
        avg_mae_normalized = total_mae / num_steps
        self.val_losses.append(avg_huber)  # val_losses always tracks normalized HuberLoss
        
        # Apply inverse transform if scaler is available
        is_denormalized = False
        train_targets_mean_for_metrics = train_targets_mean  # Default: use as-is
        
        if self.target_scaler is not None:
            logger.debug("Applying inverse transform to predictions and targets...")
            # RobustScaler inverse_transform expects (n_samples, 1) shape.
            # Use squeeze(axis=1) instead of squeeze() to safely handle single-sample batches
            # (squeeze() would collapse (1,) → scalar which breaks downstream tensor ops).
            # RobustScaler was fit on (N,3). To invert a single column,
            # manually apply: x_orig = x_scaled * scale[col] + center[col]
            col = getattr(self, "_current_target_idx", 0)
            center = self.target_scaler.center_[col]
            scale  = self.target_scaler.scale_[col]
            all_predictions_denorm = all_predictions.numpy() * scale + center
            all_targets_denorm     = all_targets.numpy()     * scale + center
            
            all_predictions = torch.from_numpy(all_predictions_denorm).float()
            all_targets = torch.from_numpy(all_targets_denorm).float()
            is_denormalized = True
            
            # CRITICAL: Also denormalize train_targets_mean to match test targets scale
            # train_targets_mean was calculated from normalized training targets
            # When test targets are denormalized, train_targets_mean must also be denormalized
            # for R² OOS historical mean benchmark to be correct
            if train_targets_mean is not None:
                train_targets_mean_for_metrics = train_targets_mean * scale + center
                logger.debug(f"✓ Denormalized train_targets_mean: {train_targets_mean:.6f} → {train_targets_mean_for_metrics:.6f}")
            
            logger.debug("✓ Inverse transform applied (metrics computed on original scale)")
        
        # Compute all metrics via shared helper.
        # When is_denormalized=True, mse/mae/rmse/r2 are in original (denormalized) scale.
        # When is_denormalized=False, they are in normalized scale.
        # 'normalized_huber' is always in normalized scale and is what drives early stopping.
        target_name = TARGET_NAMES[target_idx] if target_idx < len(TARGET_NAMES) else None
        metrics = _compute_metrics(all_predictions, all_targets, target_name=target_name, train_targets_mean=train_targets_mean_for_metrics)
        metrics["normalized_huber"] = avg_huber       # Always normalized — use for early stopping
        metrics["normalized_mae"] = avg_mae_normalized  # Always normalized
        metrics["is_denormalized"] = is_denormalized
        metrics["predictions"] = all_predictions
        metrics["targets"] = all_targets
        metrics["is_post_etf"] = all_is_post_etf
        
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
    tabular_filename = getattr(args, 'tabular_file', 'tabular_features.pt')
    ablation_mode = getattr(args, 'ablation', 'full')
    # Derive tabular_input_size from filename: extended (11 feat) → 27, base (7 feat) → 23
    _n_tab_features = 11 if tabular_filename == "tabular_features_extended.pt" else 7
    config.model.tabular_input_size = _n_tab_features + 16  # +16 for asset embedding
    targets_filter = getattr(args, 'targets', None)  # e.g. ["y_baseline"] or None for all
    num_targets = getattr(args, 'num_targets', 1)    # 1 = single-target (default), 3 = multi-target joint loss
    learning_rate            = getattr(args, 'learning_rate', None)
    weight_decay             = getattr(args, 'weight_decay', None)
    early_stopping_patience  = getattr(args, 'early_stopping_patience', None)
    early_stopping_min_delta = getattr(args, 'early_stopping_min_delta', None)
    ema_alpha                = getattr(args, 'ema_alpha', None)
    embedding_noise_std      = getattr(args, 'embedding_noise_std', None)
    huber_delta              = getattr(args, 'huber_delta', None)
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
    
    # Enable anomaly detection only in debug mode.
    # Leaving it on in production slows training by 2-5x.
    if debug:
        torch.autograd.set_detect_anomaly(True)
        logger.info("⚠ Anomaly detection ENABLED (debug mode — expect slower training)")
    else:
        torch.autograd.set_detect_anomaly(False)
    
    # Load or create config
    if config_path:
        logger.info(f"Loading config from {config_path}...")
        # For now, use default config (could load from YAML in future)
        config = ExperimentConfig()
        config.data.asset = asset
        if wandb_run_name := run_name:
            config.mlops.wandb_run_name = wandb_run_name
    else:
        config = create_config(
            asset=asset,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            ema_alpha=ema_alpha,
            embedding_noise_std=embedding_noise_std,
            huber_delta=huber_delta,
            wandb_run_name=run_name,
        )
    config.debug = debug

    # ── CLI overrides: apply ALL explicitly-passed args after config creation ──
    # This ensures overrides work for BOTH the --config branch and the default branch.
    _overrides = {
        "learning_rate":            learning_rate,
        "weight_decay":             weight_decay,
        "early_stopping_patience":  early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "ema_alpha":                ema_alpha,
        "embedding_noise_std":      embedding_noise_std,
        "huber_delta":              huber_delta,
    }
    applied = []
    for field, value in _overrides.items():
        if value is not None:
            setattr(config.training, field, value)
            applied.append(f"{field}={value}")
    if applied:
        logger.info(f"✓ CLI overrides applied: {', '.join(applied)}")
    
    logger.info(f"Config: asset={config.data.asset}, seq_len={config.data.seq_len}, batch_size={config.data.batch_size}")
    logger.info(f"Model: hidden_dim={config.model.hidden_dim}, frozen_backbones={config.model.frozen_backbones}")
    logger.info(f"Training: lr={config.training.learning_rate:.2e}, epochs={config.training.max_epochs}")
    logger.info(f"MLOps: wandb_run={config.mlops.wandb_run_name}")
    if ablation_mode != "full":
        logger.info(f"⚠ ABLATION MODE: {ablation_mode} — some modalities zeroed out")
    if num_targets > 1:
        logger.info(f"⚠ MULTI-TARGET MODE: one model with {num_targets} heads, joint mean Huber loss")
    
    # Create dataloaders (walk-forward validation)
    logger.info("\n" + "-" * 80)
    logger.info("Loading datasets...")
    print("[PROGRESS] Starting to load datasets (this may take 1-5 minutes)...")
    sys.stdout.flush()
    
    # Walk-forward generators are created fresh per-target inside the target loop.
    logger.info(f"Using WALK-FORWARD VALIDATION with {num_folds} folds")
    logger.info(f"DataLoader settings: batch_size={config.data.batch_size}, num_workers=0 (forced), pin_memory=True")
    
    # Also load test set once for final evaluation.
    # Load directly from BTC/ETH subdirs, mirroring the logic in
    # create_walk_forward_dataloaders (test slice = last 15% of each asset).
    logger.info("Also loading test set for final evaluation...")
    test_loader = None
    target_scaler = None
    try:
        from .dataset import WalkForwardDataset
        import json

        _features_dir = Path(features_dir)
        btc_subdir = _features_dir / "BTC"
        eth_subdir = _features_dir / "ETH"

        if btc_subdir.exists() and eth_subdir.exists():
            # ---- load per-asset tensors and concatenate ----
            _btc_text  = torch.load(btc_subdir / "text_embeddings.pt",  map_location="cpu")
            _btc_image = torch.load(btc_subdir / "image_embeddings.pt", map_location="cpu")
            _btc_tab   = torch.load(btc_subdir / tabular_filename,       map_location="cpu")
            _btc_tgt   = torch.load(btc_subdir / "target_scores.pt",    map_location="cpu")

            _eth_text  = torch.load(eth_subdir / "text_embeddings.pt",  map_location="cpu")
            _eth_image = torch.load(eth_subdir / "image_embeddings.pt", map_location="cpu")
            _eth_tab   = torch.load(eth_subdir / tabular_filename,       map_location="cpu")
            _eth_tgt   = torch.load(eth_subdir / "target_scores.pt",    map_location="cpu")

            _btc_len = _btc_text.shape[0]
            _eth_len = _eth_text.shape[0]

            _test_text  = torch.cat([_btc_text,  _eth_text],  dim=0)
            _test_image = torch.cat([_btc_image, _eth_image], dim=0)
            _test_tab   = torch.cat([_btc_tab,   _eth_tab],   dim=0)
            _test_tgt   = torch.cat([_btc_tgt,   _eth_tgt],   dim=0)
        else:
            # Fallback: single consolidated files + split_metadata.json
            _test_text  = torch.load(_features_dir / "text_embeddings.pt",  map_location="cpu")
            _test_image = torch.load(_features_dir / "image_embeddings.pt", map_location="cpu")
            _test_tab   = torch.load(_features_dir / tabular_filename,        map_location="cpu")
            _test_tgt   = torch.load(_features_dir / "target_scores.pt",    map_location="cpu")
            _total = _test_text.shape[0]
            _btc_len = _total // 2
            _eth_len = _total // 2

        # ---- target engineering (must match create_walk_forward_dataloaders) ----
        _test_tgt[:, 0] = _test_tgt[:, 0] * 1000.0
        _test_tgt[:, 1] = torch.clamp(_test_tgt[:, 1], min=-5.0, max=5.0)

        _total_samples = _test_text.shape[0]
        _timestamps = torch.arange(_total_samples, dtype=torch.long)

        # ---- compute test slice boundaries (last 15%) per asset ----
        _test_pct = 0.15
        _train_end_per_asset = int(_btc_len * (1.0 - _test_pct))
        _test_slice = slice(_train_end_per_asset, _btc_len)

        # ---- fit scalers on train portion only (no data leakage) ----
        from sklearn.preprocessing import StandardScaler, RobustScaler

        _btc_train_sl = slice(0, _train_end_per_asset)
        _btc_scaler_tab = StandardScaler().fit(_test_tab[_btc_train_sl].numpy())
        _btc_scaler_tgt = RobustScaler().fit(_test_tgt[_btc_train_sl].numpy())

        _eth_train_sl = slice(_btc_len, _btc_len + _train_end_per_asset)
        _eth_scaler_tab = StandardScaler().fit(_test_tab[_eth_train_sl].numpy())
        _eth_scaler_tgt = RobustScaler().fit(_test_tgt[_eth_train_sl].numpy())

        # Apply scalers per-asset
        _tab_scaled = _test_tab.clone()
        _tgt_scaled = _test_tgt.clone()

        _tab_scaled[:_btc_len] = torch.from_numpy(
            _btc_scaler_tab.transform(_test_tab[:_btc_len].numpy())
        ).float()
        _tgt_scaled[:_btc_len] = torch.from_numpy(
            _btc_scaler_tgt.transform(_test_tgt[:_btc_len].numpy())
        ).float()

        _tab_scaled[_btc_len:_btc_len + _eth_len] = torch.from_numpy(
            _eth_scaler_tab.transform(_test_tab[_btc_len:_btc_len + _eth_len].numpy())
        ).float()
        _tgt_scaled[_btc_len:_btc_len + _eth_len] = torch.from_numpy(
            _eth_scaler_tgt.transform(_test_tgt[_btc_len:_btc_len + _eth_len].numpy())
        ).float()

        # ---- create WalkForwardDataset for the test slice ----
        _test_ds = WalkForwardDataset(
            text_embeddings=_test_text,
            image_embeddings=_test_image,
            tabular_data=_tab_scaled,
            target_scores=_tgt_scaled,
            timestamps=_timestamps,
            data_slice=_test_slice,
            seq_len=config.data.seq_len,
            btc_len=_btc_len,
            eth_len=_eth_len,
        )

        test_loader = torch.utils.data.DataLoader(
            _test_ds,
            batch_size=config.data.batch_size,
            shuffle=False,
            collate_fn=multimodal_collate_fn,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        target_scaler = _btc_scaler_tgt  # BTC scaler used as default for denorm
        logger.info(f"✓ Test loader created: {len(test_loader)} batches "
                    f"(BTC test=[{_train_end_per_asset}:{_btc_len}], ETH test=[{_btc_len + _train_end_per_asset}:{_btc_len + _eth_len}])")

        # Clean up large temporaries
        del _test_text, _test_image, _test_tab, _test_tgt, _tab_scaled, _tgt_scaled
    except Exception as e:
        logger.warning(f"Failed to load test set: {e}", exc_info=True)
        test_loader = None
        target_scaler = None
    
    print("[PROGRESS] ✓ Walk-forward generator ready!")
    sys.stdout.flush()
    
    
    # Model and Trainer are initialised fresh per-target inside the target loop below
    
    # W&B initialisation is done per-target inside the target loop below
    
    # Resume: start_epoch is applied to the first fold of the first target only.
    # Checkpoint loading happens inside the target loop after the trainer is created.
    start_epoch = 0
    resume_ckpt_path = None
    if resume_training:
        logger.info("\n" + "-" * 80)
        logger.info("Resuming from checkpoint...")
        checkpoint_files = sorted([
            p for p in config.mlops.checkpoint_dir.glob(f"{config.mlops.wandb_run_name}_epoch_*.pt")
        ])
        if checkpoint_files:
            resume_ckpt_path = checkpoint_files[-1]
            logger.info(f"Will resume from: {resume_ckpt_path}")
        else:
            logger.warning("No checkpoint found, starting from scratch")
    
    # Training loop
    logger.info("\n" + "-" * 80)
    logger.info("Starting training...")
    logger.info("-" * 80 + "\n")
    print("[PROGRESS] ✓ Setup complete, training begins now...")
    sys.stdout.flush()
    
    # ==================== MULTI-TARGET TRAINING (num_targets > 1) ====================
    # One shared model with N independent prediction heads, trained with mean Huber loss
    # across all heads simultaneously. Used to produce the multi-target baseline for RQ3.
    if num_targets > 1:
        # Ensure num_targets does not exceed available targets
        num_targets = min(num_targets, len(TARGET_NAMES))

        logger.info("=" * 80)
        logger.info(f"MULTI-TARGET MODE: one model, {num_targets} heads, joint loss")
        logger.info("=" * 80)

        model = MultimodalFusionNet(config, ablation_mode=ablation_mode, num_targets=num_targets)
        logger.info(f"  Multi-target model initialised ({num_targets} heads)")

        if config.mlops.use_wandb and wandb is not None:
            if wandb.run is not None:
                wandb.finish()
            mt_run_name = f"{config.mlops.wandb_run_name}_multi_target_{num_targets}h"
            wandb.init(
                project=config.mlops.wandb_project,
                name=mt_run_name,
                config={**config.to_dict(), "num_targets": num_targets, "mode": "multi_target"},
                settings=wandb.Settings(_service_wait=0, _disable_stats=False),
                reinit=True,
            )

        walk_forward_generator = create_walk_forward_dataloaders(
            config, features_dir=features_dir, num_folds=num_folds, num_workers=0, pin_memory=True,
            tabular_filename=tabular_filename,
        )

        fold_results_mt = {}
        last_train_targets_mean = {i: 0.0 for i in range(num_targets)}

        for fold_num, train_loader, val_loader, scalers_dict in walk_forward_generator:
            logger.info(f"\n{'='*80}\nFOLD {fold_num} [multi-target]\n{'='*80}")

            # Compute per-target training means for R² OOS benchmarks
            train_means = {i: compute_train_targets_mean(train_loader, target_idx=i) for i in range(num_targets)}
            last_train_targets_mean = train_means

            model.apply(_reset_weights)
            trainer = Trainer(config, model, device=device, target_scaler=scalers_dict.get("target_scaler"))
            trainer._current_target_idx = 0  # primary target for scaler/denorm
            trainer.setup_optimizer()

            early_stopping = EarlyStopping(
                patience=config.training.early_stopping_patience,
                min_delta=config.training.early_stopping_min_delta,
                verbose=True,
            )
            ema_val_loss: Optional[float] = None

            for epoch in range(config.training.max_epochs):
                trainer.epoch = epoch
                # train_epoch uses target_idx=0 for metric logging; loss is joint across all heads
                train_metrics = trainer.train_epoch(train_loader, target_idx=0, train_targets_mean=train_means[0])

                if (epoch + 1) % config.mlops.eval_frequency == 0:
                    val_metrics = trainer.validate(val_loader, target_idx=0, train_targets_mean=train_means[0])
                    val_loss = val_metrics["normalized_huber"]
                    ema_alpha = config.training.ema_alpha
                    ema_val_loss = val_loss if ema_val_loss is None else ema_alpha * val_loss + (1 - ema_alpha) * ema_val_loss
                    if early_stopping(ema_val_loss):
                        logger.info(f"✓ Early stopping at epoch {epoch+1}")
                        break

            # Final validation: collect per-target metrics by running validate once per head
            per_target_val = {}
            for tidx in range(num_targets):
                tname = TARGET_NAMES[tidx]
                vm = trainer.validate(val_loader, target_idx=tidx, train_targets_mean=train_means[tidx])
                per_target_val[tname] = {"val_r2_oos": vm["r2_oos"], "val_mae": vm["mae"], "val_r2": vm["r2"]}
                logger.info(f"  Fold {fold_num} [{tname}]: R²_OOS={vm['r2_oos']:.4f} | MAE={vm['mae']:.6f}")

            fold_results_mt[fold_num] = per_target_val

        # ---- Test evaluation (multi-target) ----
        if test_loader is not None:
            if target_scaler is not None:
                trainer.target_scaler = target_scaler
            trainer._current_target_idx = 0
            mt_test_results = {}
            for tidx in range(num_targets):
                tname = TARGET_NAMES[tidx]
                trainer._current_target_idx = tidx
                tm = trainer.validate(test_loader, target_idx=tidx, train_targets_mean=last_train_targets_mean[tidx])
                mt_test_results[tname] = {"r2_oos": tm["r2_oos"], "mae": tm["mae"], "r2": tm["r2"]}
                logger.info(f"  [TEST] {tname}: R²_OOS={tm['r2_oos']:.4f} | MAE={tm['mae']:.6f}")

        logger.info("\n[MULTI-TARGET] Summary:")
        for tidx in range(num_targets):
            tname = TARGET_NAMES[tidx]
            fold_r2s = [fold_results_mt[f][tname]["val_r2_oos"] for f in fold_results_mt]
            logger.info(f"  {tname}: CV R²_OOS {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")

        # Save predictions
        out_dir = config.mlops.checkpoint_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / f"mfn_test_predictions_multi_target_{num_targets}h.npz",
                 **{f"{tname}_MFN_multi": mt_test_results[tname] for tname in mt_test_results})

        if wandb is not None and wandb.run is not None:
            wandb.finish()

        logger.info("=" * 80)
        logger.info("MULTI-TARGET TRAINING COMPLETE")
        logger.info("=" * 80)
        return  # Skip single-target loop below

    # ==================== PER-TARGET TRAINING LOOP ====================
    # v5: 3 independent models, one per target (y_baseline, y_heuristic, y_vol_adj_return).
    # Each model is re-initialised from scratch so there is zero cross-target leakage.
    logger.info("=" * 80)
    logger.info(f"OUTER LOOP: Training {len(TARGET_NAMES)} independent models")
    logger.info("=" * 80)

    # Filter targets if --targets specified
    active_targets = TARGET_NAMES
    if targets_filter:
        active_targets = [t for t in TARGET_NAMES if t in targets_filter]
        if not active_targets:
            raise ValueError(f"No valid targets in {targets_filter}. Available: {TARGET_NAMES}")
        logger.info(f"⚠ Training only selected targets: {active_targets}")

    all_target_results = {}   # {target_name: {fold_results, test_metrics}}

    for target_idx, target_name in [(TARGET_NAMES.index(t), t) for t in active_targets]:
        logger.info("\n" + "#" * 80)
        logger.info(f"TARGET {target_idx+1}/{len(TARGET_NAMES)}: {target_name}")
        logger.info("#" * 80)

        # Lists to accumulate Out-of-Sample (OOS) predictions across all folds + test set
        # This is required because Pre-ETF data is in the validation folds, not the test set.
        oos_preds = []
        oos_tgts = []
        oos_is_post_etf = []
        oos_benchmarks = []

        # Re-create walk-forward generator (it is a generator — exhausted after one pass)
        walk_forward_generator = create_walk_forward_dataloaders(
            config,
            features_dir=features_dir,
            num_folds=num_folds,
            num_workers=0,
            pin_memory=True,
            tabular_filename=tabular_filename,
        )

        # Fresh model for this target (single-target mode: num_targets=1)
        model = MultimodalFusionNet(config, ablation_mode=ablation_mode, num_targets=1)
        logger.info(f"  Fresh model initialised for {target_name}")

        # W&B run per target
        if config.mlops.use_wandb and wandb is not None:
            if wandb.run is not None:
                wandb.finish()
            target_run_name = f"{config.mlops.wandb_run_name}_{target_name}"
            wandb.init(
                project=config.mlops.wandb_project,
                name=target_run_name,
                config={**config.to_dict(), "target_name": target_name, "target_idx": target_idx},
                settings=wandb.Settings(_service_wait=0, _disable_stats=False),
                reinit=True,
            )
            logger.info(f"  W&B run: {target_run_name}")

            # --- Define custom x-axes ---
            # 'fold_num' drives fold_summary/* (final metrics per fold, X=fold).
            # Per-epoch train/val metrics use step=epoch_in_fold and are registered
            # per fold inside the fold loop via define_metric calls below.
            wandb.define_metric("fold_num")  # custom x-axis for fold-level charts
            wandb.define_metric("fold_summary/*", step_metric="fold_num")

        # ==================== WALK-FORWARD TRAINING LOOP ====================
        logger.info("=" * 80)
        logger.info(f"WALK-FORWARD VALIDATION — {target_name}")
        logger.info("=" * 80)

        fold_results = {}

        for fold_num, train_loader, val_loader, scalers_dict in walk_forward_generator:
            logger.info("\n" + "=" * 80)
            logger.info(f"FOLD {fold_num}")
            logger.info("=" * 80)

            # Compute training set targets mean for R² OOS benchmark
            train_targets_mean = compute_train_targets_mean(train_loader, target_idx=target_idx)
            logger.info(f"  Train targets mean (for R² OOS benchmark): {train_targets_mean:.6f}")

            # Reset model weights for each fold.
            # CRITICAL for generalization: without this, fold N's model is fine-tuned from
            # fold N-1's weights — meaning by the final fold the model has implicitly seen
            # ALL prior folds' training+val data, causing optimistic validation R² that
            # doesn't transfer to the held-out test set.
            # Fresh initialization per fold ensures each fold's val metric is honest.
            model.apply(_reset_weights)
            logger.info(f"✓ Model weights reset for Fold {fold_num}")

            # Reset trainer for each fold
            trainer = Trainer(config, model, device=device, target_scaler=scalers_dict.get("target_scaler"))
            trainer._current_target_idx = target_idx
            trainer.setup_optimizer()
            
            # Global step offset: each fold's epochs are offset by (fold_num-1)*max_epochs
            # so fold1 uses steps 1..max_epochs, fold2 uses max_epochs+1..2*max_epochs, etc.
            # This makes all folds visible as SEPARATE curves in WandB (no overlap).
            fold_step_offset = (fold_num - 1) * config.training.max_epochs
            logger.info(f"✓ Fold {fold_num} initialized | WandB step offset: {fold_step_offset}")

            # Register per-fold metrics so WandB uses epoch_in_fold as x-axis.
            # Each fold gets its own set of curves (fold_1/train_loss, fold_2/train_loss, ...).
            # X-axis = epoch_in_fold (1 → max_epochs), same scale across all folds
            # → easy visual comparison of training convergence per fold.
            if wandb is not None and wandb.run is not None:
                epoch_axis = f"fold_{fold_num}/epoch"
                wandb.define_metric(epoch_axis)
                for metric_name in [
                    f"fold_{fold_num}/train_loss", f"fold_{fold_num}/train_r2",
                    f"fold_{fold_num}/train_r2_oos", f"fold_{fold_num}/train_rmse",
                    f"fold_{fold_num}/train_mae", f"fold_{fold_num}/train_correlation",
                    f"fold_{fold_num}/train_avg_grad_norm", f"fold_{fold_num}/train_max_grad_norm",
                    f"fold_{fold_num}/val_loss_normalized", f"fold_{fold_num}/val_loss_ema",
                    f"fold_{fold_num}/val_r2", f"fold_{fold_num}/val_r2_oos",
                    f"fold_{fold_num}/val_rmse", f"fold_{fold_num}/val_mae",
                    f"fold_{fold_num}/val_correlation",
                ]:
                    wandb.define_metric(metric_name, step_metric=epoch_axis)

            # Resume: load checkpoint into the first fold of the first target only.
            if resume_ckpt_path and target_idx == 0 and fold_num == 1:
                trainer.load_checkpoint(resume_ckpt_path)
                start_epoch = trainer.epoch + 1
                logger.info(f"Resumed from checkpoint — starting at epoch {start_epoch}")

            # Early stopping reset.
            # min_delta=1e-5 (reduced from 1e-4): financial time-series improvements are
            # often < 1e-4 per epoch in the slow-learning phase (epoch 2-9 plateau observed).
            # 1e-4 was cutting training at epoch ~16 when loss was still decreasing.
            # 1e-5 responds to genuine stagnation without triggering on normal noise.
            early_stopping = EarlyStopping(
                patience=config.training.early_stopping_patience,
                min_delta=config.training.early_stopping_min_delta,
                verbose=True
            )

            # Train for this fold.
            # On resume, start_epoch > 0 for the FIRST fold of FIRST target only.
            fold_start = start_epoch if (target_idx == 0 and fold_num == 1) else 0
            start_epoch = 0  # Reset so subsequent folds/targets always start at 0

            # EMA of val loss for early stopping.
            # Raw val loss on small val windows (~500 samples) is very noisy —
            # a single noisy batch can make val loss spike or dip by 0.003+,
            # which triggers early stopping prematurely.
            # EMA with α=0.3 smooths this noise while still responding to real trends.
            # Formula: ema = α * current + (1-α) * previous
            ema_val_loss: Optional[float] = None
            
            final_train_metrics = None  # Track final train metrics for fold summary
            logger.info(f"\n{'='*80}")
            logger.info(f"Starting Target '{TARGET_NAMES[target_idx]}' - Fold {fold_num}/5")
            logger.info(f"Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}")
            logger.info(f"{'='*80}")

            for epoch in range(fold_start, config.training.max_epochs):
                trainer.epoch = epoch
                epoch_in_fold = epoch - fold_start + 1  # Epoch per fold (1, 2, 3, ...)

                # Train epoch
                train_metrics = trainer.train_epoch(train_loader, target_idx=target_idx, train_targets_mean=train_targets_mean)
                final_train_metrics = train_metrics  # Store for fold summary
                train_loss = train_metrics["loss"]

                logger.info(
                    f"Fold {fold_num}/5 Epoch {epoch_in_fold:3d}/{config.training.max_epochs} | "
                    f"Train Loss {train_loss:.6f} | "
                    f"Train R² {train_metrics['r2']:.6f}"
                )

                # Log train metrics per fold.
                # Key: fold_N/train_* — each fold has its own named curve.
                # Step: epoch_in_fold (1 → max_epochs) — same x-axis scale across all folds.
                # WandB will render fold_1/train_loss, fold_2/train_loss, ... as
                # separate charts that can be pinned side-by-side for comparison.
                if wandb is not None and wandb.run is not None:
                    train_log_dict = {
                        f"fold_{fold_num}/epoch":              epoch_in_fold,
                        f"fold_{fold_num}/train_loss":         train_loss,
                        f"fold_{fold_num}/train_r2":           train_metrics["r2"],
                        f"fold_{fold_num}/train_r2_oos":       train_metrics["r2_oos"],
                        f"fold_{fold_num}/train_rmse":         train_metrics["rmse"],
                        f"fold_{fold_num}/train_mae":          train_metrics["mae"],
                        f"fold_{fold_num}/train_correlation":  train_metrics["correlation"],
                        f"fold_{fold_num}/train_avg_grad_norm": train_metrics["avg_gradient_norm"],
                        f"fold_{fold_num}/train_max_grad_norm": train_metrics["max_gradient_norm"],
                    }
                    wandb.log(train_log_dict, commit=False)

                # Validate every eval_frequency epochs
                run_val = (epoch + 1) % config.mlops.eval_frequency == 0
                if run_val:
                    val_metrics = trainer.validate(val_loader, target_idx=target_idx, train_targets_mean=train_targets_mean)
                    # Use normalized_huber for early stopping & best-model comparison:
                    # always on normalized scale regardless of whether target_scaler is set,
                    # ensuring consistent comparisons across folds.
                    val_loss = val_metrics["normalized_huber"]

                    logger.info(
                        f"Fold {fold_num}/5 Epoch {epoch_in_fold:3d}/{config.training.max_epochs} | "
                        f"Val HuberLoss (normalized) {val_loss:.6f} | "
                        f"Val MSE {val_metrics['mse']:.6f} | "
                        f"Val R² {val_metrics['r2']:.6f}"
                    )

                    # Log val metrics under the same fold_N/ namespace.
                    # epoch axis = fold_N/epoch (same as train), so train and val for
                    # this fold share the x-axis and can be overlaid in WandB.
                    if wandb is not None and wandb.run is not None:
                        val_log_dict = {
                            f"fold_{fold_num}/epoch":              epoch_in_fold,
                            f"fold_{fold_num}/val_loss_normalized": val_loss,
                            f"fold_{fold_num}/val_r2":             val_metrics["r2"],
                            f"fold_{fold_num}/val_r2_oos":         val_metrics["r2_oos"],
                            f"fold_{fold_num}/val_rmse":           val_metrics["rmse"],
                            f"fold_{fold_num}/val_mae":            val_metrics["mae"],
                            f"fold_{fold_num}/val_correlation":    val_metrics["correlation"],
                        }
                        wandb.log(val_log_dict, commit=True)
                        logger.debug(f"✓ W&B val logged: fold={fold_num}, epoch_in_fold={epoch_in_fold}")
                    else:
                        if wandb is not None and wandb.run is not None:
                            wandb.log({}, commit=True)

                    # Early stopping: use EMA-smoothed val loss instead of raw val loss.
                    # Raw val loss on small val windows is too noisy for direct stopping decisions.
                    ema_alpha = config.training.ema_alpha
                    if ema_val_loss is None:
                        ema_val_loss = val_loss  # Bootstrap with first observation
                    else:
                        ema_val_loss = ema_alpha * val_loss + (1 - ema_alpha) * ema_val_loss

                    if wandb is not None and wandb.run is not None:
                        wandb.log({f"fold_{fold_num}/val_loss_ema": ema_val_loss}, commit=False)

                    if early_stopping(ema_val_loss):
                        logger.info(f"✓ Early stopping triggered at epoch {epoch+1} (EMA val loss: {ema_val_loss:.6f})")
                        break

                    # Update best model for this fold (based on raw val loss, not EMA)
                    if val_loss < trainer.best_val_loss:
                        trainer.best_val_loss = val_loss
                        trainer.best_epoch = epoch

                else:
                    # No validation this epoch — commit the train log
                    if wandb is not None and wandb.run is not None:
                        wandb.log({}, commit=True)

            # Validate on full validation set for this fold.
            # Use a fresh validate call, but mark it so val_losses doesn't get a duplicate append.
            # (val_losses was already appended during the per-epoch training loop above.)
            logger.info(f"\nFinal validation for Fold {fold_num}...")
            final_val_metrics = trainer.validate(val_loader, target_idx=target_idx, train_targets_mean=train_targets_mean)
            trainer.val_losses.pop()  # Remove the extra append from this summary-only validate() call

            fold_results[fold_num] = {
                "val_r2": final_val_metrics["r2"],
                "val_r2_oos": final_val_metrics["r2_oos"],
                "val_mse": final_val_metrics["mse"],
                "val_rmse": final_val_metrics["rmse"],
                "val_mae": final_val_metrics["mae"],
                "val_correlation": final_val_metrics["correlation"],
            }
            
            # Accumulate OOS data from this validation fold
            if "predictions" in final_val_metrics:
                oos_preds.append(final_val_metrics["predictions"])
                oos_tgts.append(final_val_metrics["targets"])
                oos_is_post_etf.append(final_val_metrics.get("is_post_etf"))
                
                # Get the denormalized train_targets_mean for this fold
                _fold_benchmark = train_targets_mean
                if trainer.target_scaler is not None and train_targets_mean is not None:
                    col = getattr(trainer, "_current_target_idx", 0)
                    center = trainer.target_scaler.center_[col]
                    scale = trainer.target_scaler.scale_[col]
                    _fold_benchmark = train_targets_mean * scale + center
                
                if _fold_benchmark is not None:
                    # Create a tensor of the benchmark mean, same shape as predictions
                    bm_tensor = torch.full_like(final_val_metrics["predictions"], _fold_benchmark)
                    oos_benchmarks.append(bm_tensor)
                else:
                    oos_benchmarks.append(torch.zeros_like(final_val_metrics["predictions"]))

            logger.info(f"Fold {fold_num} Results: R²={final_val_metrics['r2']:.6f}, MSE={final_val_metrics['mse']:.6f}, RMSE={final_val_metrics['rmse']:.6f}, MAE={final_val_metrics['mae']:.6f}")

            # Sanity check: Display first 5 predictions vs actual values
            logger.info("\nSanity Check - First 5 Predictions vs Actual:")
            predictions = final_val_metrics.get("predictions", None)
            targets = final_val_metrics.get("targets", None)
            if predictions is not None and targets is not None:
                for i in range(min(5, len(predictions))):
                    logger.info(f"  [{i+1}] Predicted: {predictions[i].item():.4f} | Actual: {targets[i].item():.4f} | Error: {abs(predictions[i].item() - targets[i].item()):.4f}")

            # Log per-fold summary metrics to W&B.
            # We use a CUSTOM x-axis 'fold_num' (defined above via wandb.define_metric)
            # so these log as a separate line chart (X=fold, Y=metric) that does NOT
            # conflict with the per-epoch global step used by train/* and val/* curves.
            if wandb is not None and wandb.run is not None:
                fold_summary_dict = {
                    "fold_num":                        fold_num,
                    "fold_summary/train_r2":           final_train_metrics["r2"]     if final_train_metrics else 0.0,
                    "fold_summary/train_r2_oos":       final_train_metrics["r2_oos"] if final_train_metrics else 0.0,
                    "fold_summary/val_r2":             final_val_metrics["r2"],
                    "fold_summary/val_r2_oos":         final_val_metrics["r2_oos"],
                    "fold_summary/val_rmse":           final_val_metrics["rmse"],
                    "fold_summary/val_mae":            final_val_metrics["mae"],
                    "fold_summary/val_correlation":    final_val_metrics["correlation"],
                    "fold_summary/val_loss_normalized": final_val_metrics["normalized_huber"],
                }
                wandb.log(fold_summary_dict)  # no step= argument: WandB uses fold_num as x-axis
                logger.info(
                    f"✓ W&B fold chart logged: fold={fold_num} "
                    f"train_r2={final_train_metrics['r2'] if final_train_metrics else 0:.6f}, "
                    f"val_r2={final_val_metrics['r2']:.6f}"
                )

        # Store train_targets_mean from last fold for test evaluation
        last_fold_train_targets_mean = train_targets_mean

        # Fold results logged during per-fold training

        # ==================== TEST EVALUATION ====================
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating on Test Set...")
        logger.info("=" * 80)

        # Check if test_loader is available
        if test_loader is None:
            logger.warning("⚠ Test loader is None - skipping test evaluation (test dataset failed to load)")
            test_metrics = None
        else:
            # CRITICAL: Update trainer's target_scaler to test set's scaler before evaluation
            # The trainer currently has the last fold's scaler, which causes scale mismatch
            if target_scaler is not None:
                trainer.target_scaler = target_scaler
                logger.info("✓ Updated trainer scaler to test set scaler")
            else:
                trainer.target_scaler = None
                logger.info("⚠ No test set scaler available - will evaluate on normalized scale")

            trainer._current_target_idx = target_idx
            test_metrics = trainer.validate(test_loader, target_idx=target_idx, train_targets_mean=last_fold_train_targets_mean)

        # Only process test metrics if evaluation succeeded
        if test_metrics is not None:
            denorm_status = " (denormalized to original scale)" if test_metrics.get("is_denormalized", False) else " (normalized scale)"

            logger.info(
                f"Test Results{denorm_status}:\n"
                f"  MSE:    {test_metrics['mse']:.6f}\n"
                f"  RMSE:   {test_metrics['rmse']:.6f}\n"
                f"  MAE:    {test_metrics['mae']:.6f}\n"
                f"  R²:     {test_metrics['r2']:.6f}\n"
                f"  R²_OOS: {test_metrics['r2_oos']:.6f}  (Gu, Kelly & Xiu 2020)\n"
                f"  Correlation: {test_metrics['correlation']:.6f}\n"
                f"  Prediction Bias: {test_metrics['prediction_error_mean']:.6f}\n"
                f"  Prediction Error Std: {test_metrics['prediction_error_std']:.6f}"
            )

            # Accumulate OOS data from the test set
            if "predictions" in test_metrics:
                oos_preds.append(test_metrics["predictions"])
                oos_tgts.append(test_metrics["targets"])
                oos_is_post_etf.append(test_metrics.get("is_post_etf"))
                
                _test_benchmark = last_fold_train_targets_mean
                if trainer.target_scaler is not None and last_fold_train_targets_mean is not None:
                    col = getattr(trainer, "_current_target_idx", 0)
                    center = trainer.target_scaler.center_[col]
                    scale = trainer.target_scaler.scale_[col]
                    _test_benchmark = last_fold_train_targets_mean * scale + center
                
                if _test_benchmark is not None:
                    bm_tensor = torch.full_like(test_metrics["predictions"], _test_benchmark)
                    oos_benchmarks.append(bm_tensor)
                else:
                    oos_benchmarks.append(torch.zeros_like(test_metrics["predictions"]))

            # Log comprehensive test metrics to W&B
            if wandb is not None and wandb.run is not None:
                # Log core metrics first
                test_log_dict = {
                    "test_mse": test_metrics["mse"],
                    "test_mae": test_metrics["mae"],
                    "test_rmse": test_metrics["rmse"],
                    "test_r2": test_metrics["r2"],
                    "test_r2_oos": test_metrics["r2_oos"],
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

        # Store results for this target
        all_target_results[target_name] = {
            "fold_results": fold_results,
            "test_metrics": test_metrics,
        }

        # --- REGIME-LEVEL EVALUATION (AGGREGATED OOS) ---
        if len(oos_preds) > 0 and oos_is_post_etf[0] is not None:
            cat_preds = torch.cat(oos_preds)
            cat_tgts = torch.cat(oos_tgts)
            cat_is_post_etf = torch.cat(oos_is_post_etf)
            cat_benchmarks = torch.cat(oos_benchmarks)
            
            pre_etf_mask = cat_is_post_etf == 0
            post_etf_mask = cat_is_post_etf == 1
            
            logger.info("\n" + "-" * 40)
            logger.info(f"REGIME-LEVEL STABILITY ({target_name})")
            logger.info("-" * 40)
            
            # Helper to compute OOS metrics with array of benchmarks
            def _eval_regime(preds_subset, tgts_subset, bench_subset, name):
                mae = torch.mean(torch.abs(preds_subset - tgts_subset)).item()
                ss_res = torch.sum((preds_subset - tgts_subset) ** 2).item()
                
                # GKX 2020 (Zero Predictor)
                ss_tot_gkx = torch.sum(tgts_subset ** 2).item()
                r2_oos_gkx = 1.0 - (ss_res / ss_tot_gkx) if ss_tot_gkx > 0 else 0.0
                
                # Historical Mean
                ss_tot_hist = torch.sum((tgts_subset - bench_subset) ** 2).item()
                r2_oos_hist = 1.0 - (ss_res / ss_tot_hist) if ss_tot_hist > 0 else 0.0
                
                # Select based on target (mirrors _compute_metrics)
                if target_name == "y_baseline":
                    r2_oos_primary = r2_oos_hist
                else:
                    r2_oos_primary = r2_oos_gkx
                    
                logger.info(f"  [{name}] MAE: {mae:.6f} | R²_OOS: {r2_oos_primary:.6f} | n={len(tgts_subset)}")
            
            if pre_etf_mask.any():
                _eval_regime(cat_preds[pre_etf_mask], cat_tgts[pre_etf_mask], cat_benchmarks[pre_etf_mask], "Pre-ETF")
            if post_etf_mask.any():
                _eval_regime(cat_preds[post_etf_mask], cat_tgts[post_etf_mask], cat_benchmarks[post_etf_mask], "Post-ETF")
            logger.info("-" * 40 + "\n")

        # Finish W&B run for this target
        if wandb is not None and wandb.run is not None:
            wandb.finish()

    # END of per-target loop
    # END of per-target loop
    # ==================== GLOBAL SUMMARY ====================
    logger.info("\n" + "=" * 80)
    logger.info("ALL TARGETS — TRAINING COMPLETE")
    logger.info("=" * 80)
    
    # Save MFN predictions to disk for DM testing
    mfn_preds_dict = {}
    
    for tname, tresult in all_target_results.items():
        fold_r2 = [v["val_r2"] for v in tresult["fold_results"].values()]
        tm = tresult["test_metrics"]
        test_str = f"  Test R²={tm['r2']:.4f} | R²_OOS={tm['r2_oos']:.4f}" if tm else "  Test: N/A"
        logger.info(
            f"  {tname:15s}: CV R² {np.mean(fold_r2):.4f} ± {np.std(fold_r2):.4f} |{test_str}"
        )
        if tm and "predictions" in tm:
            mfn_preds_dict[f"{tname}_MFN"] = tm["predictions"].numpy() if hasattr(tm["predictions"], "numpy") else tm["predictions"]
            
    # Save to npz
    out_dir = config.mlops.checkpoint_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    mfn_preds_file = out_dir / f"mfn_test_predictions_{args.ablation}.npz"
    if mfn_preds_dict:
        np.savez(mfn_preds_file, **mfn_preds_dict)
        logger.info(f"✓ MFN Test predictions saved → {mfn_preds_file} (Use this for DM tests vs baselines)")

    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multimodal crypto sentiment model")
    parser.add_argument("--asset", choices=["MULTI"], default="MULTI",
                        help="Cryptocurrency asset (multi-asset: BTC+ETH combined)")
    parser.add_argument("--features-dir", type=str, default="./data/features",
                        help="Local path to pre-extracted features directory")
    parser.add_argument("--run-name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--config", type=str, default=None,
                        help="Config file path (YAML)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode (small dataset, CPU)")
    parser.add_argument("--num-folds", type=int, default=5,
                        help="Number of walk-forward folds (default: 5)")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "tabular_only", "no_text", "no_image"],
                        help="Ablation mode: full (all modalities), tabular_only, no_text, no_image")
    parser.add_argument("--targets", nargs="+", default=None,
                        choices=["y_baseline", "y_heuristic", "y_vol_adj_return"],
                        help="Which targets to train (default: all 3). Example: --targets y_baseline")
    parser.add_argument("--num-targets", type=int, default=1, dest="num_targets",
                        choices=[1, 2, 3],
                        help=(
                            "Number of prediction heads trained jointly. "
                            "1 (default): single-target mode — 3 independent models, one per target. "
                            "3: multi-target mode — one model with 3 heads, mean Huber loss across all heads. "
                            "Use --num-targets 3 to produce the multi-target baseline for RQ3 comparison."
                        ))
    # ── Training hyperparameter overrides (all optional; default = config.py values) ──
    parser.add_argument("--learning-rate", type=float, default=None, dest="learning_rate",
                        help="AdamW learning rate (default: TrainingConfig.learning_rate = 1e-5)")
    parser.add_argument("--weight-decay", type=float, default=None, dest="weight_decay",
                        help="AdamW weight decay (default: TrainingConfig.weight_decay = 3e-3)")
    parser.add_argument("--early-stopping-patience", type=int, default=None, dest="early_stopping_patience",
                        help="Early stopping patience in epochs (default: TrainingConfig.early_stopping_patience = 10)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=None, dest="early_stopping_min_delta",
                        help="Min val-loss improvement to reset patience (default: TrainingConfig.early_stopping_min_delta = 1e-5)")
    parser.add_argument("--ema-alpha", type=float, default=None, dest="ema_alpha",
                        help="EMA alpha for val loss smoothing in early stopping (default: TrainingConfig.ema_alpha = 0.3)")
    parser.add_argument("--embedding-noise-std", type=float, default=None, dest="embedding_noise_std",
                        help="Gaussian noise std for embedding regularization; 0 = disabled (default: 0.01)")
    parser.add_argument("--huber-delta", type=float, default=None, dest="huber_delta",
                        help="HuberLoss delta, shared between train & validate (default: 1.0)")
    parser.add_argument("--tabular-file", type=str, default="tabular_features.pt", dest="tabular_file",
                        help=(
                            "Filename of the tabular features tensor inside each asset subdir. "
                            "Use 'tabular_features.pt' (default, 7 base features) for the original experiment "
                            "or 'tabular_features_extended.pt' (11 features = 7 base + MA7/MA25/RSI/MACD) "
                            "for the technical-indicator ablation."
                        ))

    args = parser.parse_args()

    # Auto-generate run name if not provided
    if args.run_name is None:
        args.run_name = f"{args.asset.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    main(args)
