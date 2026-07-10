"""
Walk-forward multimodal dataset for BTC+ETH panel data.

Provides WalkForwardDataset (sliding-window) and create_walk_forward_dataloaders
(full pipeline: load, engineer targets, fit per-fold per-asset scalers, yield folds).
Scaling (StandardScaler for tabular, RobustScaler for targets) is fitted independently
per asset on each fold's training window to prevent temporal and cross-asset leakage.
"""

import sys
import torch
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
from .utils import format_duration

logger = logging.getLogger(__name__)


def multimodal_collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Collate function for multimodal batches.

    Returns:
        Dict with stacked tensors:
            tabular:         (batch, seq_len, n_features)
            text_embedding:  (batch, seq_len, 256)
            image_embedding: (batch, seq_len, 256)
            target:          (batch, 3)
            timestamp:       (batch,)
            asset_id:        (batch,)
    """
    stacked = {
        "tabular": torch.stack([sample["tabular"] for sample in batch]),
        "text_embedding": torch.stack([sample["text_embedding"] for sample in batch]),
        "image_embedding": torch.stack([sample["image_embedding"] for sample in batch]),
        "target": torch.stack([sample["target"] for sample in batch]),
        "timestamp": torch.stack([sample["timestamp"] for sample in batch]),
    }
    
    # Handle optional asset_id (fallback to 0 if not present)
    if "asset_id" in batch[0]:
        stacked["asset_id"] = torch.stack([sample["asset_id"] for sample in batch])
    else:
        stacked["asset_id"] = torch.zeros(len(batch), dtype=torch.long)
        
    return stacked


def walk_forward_split(data_len: int, window_size: int, step_size: int):
    """
    Generate expanding-window train/val splits in chronological order.

    Fold 1: train=[0:window_size],           val=[window_size:window_size+step_size]
    Fold 2: train=[0:window_size+step_size], val=[window_size+step_size:...]
    ...

    Args:
        data_len: Total data length (training pool only, test set excluded).
        window_size: Initial training window size.
        step_size: Validation fold size (and expansion step per fold).

    Yields:
        (train_slice, val_slice)
    """
    for i in range(0, data_len - window_size - step_size, step_size):
        train_end = i + window_size
        val_start = train_end
        val_end = val_start + step_size
        
        if val_end > data_len:
            val_end = data_len
        
        train_slice = slice(0, train_end)
        val_slice = slice(val_start, val_end)
        
        yield train_slice, val_slice


class WalkForwardDataset(torch.utils.data.Dataset):
    """
    Sliding-window dataset over a concatenated BTC+ETH panel.

    Ensures that 24-hour windows never cross the BTC/ETH boundary.
    y_baseline (col 0) is fetched at index + seq_len + (funding_horizon - 1), making the
    effective prediction horizon `funding_horizon` hours ahead of the last input step
    (default funding_horizon=8, matching the thesis's primary t+8h target).
    y_heuristic (col 1) and y_vol_adj_return (col 2) are fetched at index + seq_len (t+1h),
    unaffected by funding_horizon.

    funding_horizon is exposed purely as a diagnostic robustness knob (Section 4.x,
    "Funding-Rate Horizon Sensitivity") to test whether the funding target's apparent
    tractability decays as the forecast horizon spans additional 8-hour settlement
    cycles. It is NOT a proposed change to the thesis's primary t+8h scope.
    """
    def __init__(
        self,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        tabular_data: torch.Tensor,
        target_scores: torch.Tensor,
        timestamps: torch.Tensor,
        data_slice: slice,      # The slice of a SINGLE asset (e.g. 0 to 30000)
        seq_len: int = 24,
        total_samples_per_asset: int = 44500,
        btc_len: int = None,
        eth_len: int = None,
        funding_horizon: int = 8,
    ):
        self.seq_len = seq_len
        if funding_horizon < 1:
            raise ValueError(f"funding_horizon must be >= 1, got {funding_horizon}")
        self.funding_horizon = funding_horizon
        self.funding_offset = funding_horizon - 1  # index offset added to (real_idx + seq_len)
        # Fallback to total_samples_per_asset if dynamic lengths are not provided
        self.btc_len = btc_len if btc_len is not None else total_samples_per_asset
        self.eth_len = eth_len if eth_len is not None else total_samples_per_asset
        # Maintain total_samples_per_asset for asset ID mapping and compatibility
        self.total_samples_per_asset = self.btc_len
        
        # Save full data
        self.text_full = text_embeddings
        self.image_full = image_embeddings
        self.tabular_full = tabular_data
        self.target_full = target_scores
        self.timestamps_full = timestamps
        
        start = data_slice.start if data_slice.start is not None else 0
        stop = data_slice.stop if data_slice.stop is not None else self.btc_len
        
        # Buffer to prevent index overflow on the funding_horizon-ahead target.
        # __getitem__ accesses target_full[real_idx + seq_len + funding_offset], so we need:
        #   real_idx + seq_len + funding_offset  <=  N - 1   →   real_idx  <=  N - seq_len - funding_horizon
        # With range(..., N - seq_len - buffer + 1) the last value is N - seq_len - buffer,
        # so buffer must equal funding_horizon (not funding_offset) for the last target index to be N - 1.
        buffer = self.funding_horizon
        
        # Valid starts for BTC (within the slice)
        btc_valid_starts = list(range(start, min(stop - seq_len - buffer + 1, self.btc_len - seq_len - buffer + 1)))
        
        # Valid starts for ETH (shifted by btc_len)
        eth_valid_starts = list(range(
            start + self.btc_len, 
            min(stop - seq_len - buffer + 1 + self.btc_len, self.btc_len + self.eth_len - seq_len - buffer + 1)
        ))
        
        # Combine valid start indices for both assets
        self.valid_starts = btc_valid_starts + eth_valid_starts
        
        logger.info(
            f"  Created {self.funding_horizon}h-shifted WalkForwardDataset slice [{start}:{stop}] -> "
            f"BTC starts: {len(btc_valid_starts)}, ETH starts: {len(eth_valid_starts)} (Total: {len(self.valid_starts)})"
        )

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for length {len(self)}")
            
        real_idx = self.valid_starts[idx]
        
        # Determine asset ID: 0 = BTC (real_idx < total_samples_per_asset), 1 = ETH (otherwise)
        asset_id = 0 if real_idx < self.total_samples_per_asset else 1
        
        # Target splitting:
        # Col 0 (y_baseline) is target_raw_funding, predict at t+funding_horizon
        # (real_idx + seq_len + funding_offset); funding_offset = funding_horizon - 1.
        target_baseline = self.target_full[real_idx + self.seq_len + self.funding_offset, 0]
        
        # Col 1 (y_heuristic) and Col 2 (y_vol_adj_return) remain at t+1 (real_idx + seq_len)
        target_heuristic = self.target_full[real_idx + self.seq_len, 1]
        target_vol_adj_return = self.target_full[real_idx + self.seq_len, 2]
        
        target_vector = torch.stack([target_baseline, target_heuristic, target_vol_adj_return])
        
        return {
            "tabular": self.tabular_full[real_idx : real_idx + self.seq_len],
            "text_embedding": self.text_full[real_idx : real_idx + self.seq_len],
            "image_embedding": self.image_full[real_idx : real_idx + self.seq_len],
            "target": target_vector,
            "timestamp": self.timestamps_full[real_idx + self.seq_len],
            "asset_id": torch.tensor(asset_id, dtype=torch.long),
        }



def create_walk_forward_dataloaders(
    config,
    features_dir: str = None,
    num_folds: int = 5,
    num_workers: int = 0,
    pin_memory: bool = True,
    tabular_filename: str = "tabular_features.pt",
    tabular_dir: str = None,
    funding_horizon: int = 8,
    image_filename: str = "image_embeddings.pt",
):
    """
    Load pre-extracted feature tensors and yield walk-forward validation folds.

    Supports per-asset subdirectory layout (BTC/, ETH/) and legacy single-directory
    layout. Applies target engineering (y_baseline ×1000, y_heuristic clip ±5) and
    fits independent StandardScaler/RobustScaler per asset per fold on the training
    window only.

    Args:
        config: ExperimentConfig instance.
        features_dir: Directory containing BTC/ and ETH/ feature subdirs.
        num_folds: Number of walk-forward folds.
        num_workers: DataLoader workers (always 0 on Kaggle).
        pin_memory: Pin memory for GPU transfer.
        tabular_filename: Tabular tensor filename inside each asset subdir.
                          "tabular_features.pt" (7 features, default) or
                          "tabular_features_no_funding.pt" (6 features).
        tabular_dir: Override directory for tabular files (defaults to features_dir).
        funding_horizon: Prediction horizon in hours for the funding target (y_baseline,
                         col 0), default 8 (the thesis's primary target). Diagnostic-only
                         override for the horizon-sensitivity robustness check (e.g. 16, 24);
                         does not affect y_heuristic / y_vol_adj_return, which remain t+1h.
        image_filename: Image embedding tensor filename inside each asset subdir.
                        "image_embeddings.pt" (ViT, default) or "image_embeddings_clip.pt"
                        (diagnostic-only CLIP backbone swap, see extract_features.py).

    Yields:
        (fold_num, train_loader, val_loader, scalers_dict)
    """
    num_workers = 0  # Force num_workers=0 for Kaggle safety
    features_dir = Path(features_dir) if features_dir else Path("./data/features")
    tabular_dir  = Path(tabular_dir)  if tabular_dir  else features_dir
    
    logger.info("=" * 80)
    logger.info("WALK-FORWARD VALIDATION: Loading dynamic asset features")
    logger.info("=" * 80)
    
    # Check if subdirectory-based dynamic layout exists (BTC and ETH folders)
    btc_subdir = features_dir / "BTC"
    eth_subdir = features_dir / "ETH"
    
    if btc_subdir.exists() and eth_subdir.exists():
        logger.info("Detecting dynamic asset feature layout (separate BTC / ETH subdirectories)")
        print("[PROGRESS] Loading separate BTC and ETH embeddings...")
        sys.stdout.flush()
        
        # Load BTC
        btc_text = torch.load(btc_subdir / "text_embeddings.pt", map_location="cpu")
        btc_image = torch.load(btc_subdir / image_filename, map_location="cpu")
        btc_tab = torch.load(tabular_dir / "BTC" / tabular_filename, map_location="cpu")
        btc_tgt = torch.load(btc_subdir / "target_scores.pt", map_location="cpu")

        # Load ETH
        eth_text = torch.load(eth_subdir / "text_embeddings.pt", map_location="cpu")
        eth_image = torch.load(eth_subdir / image_filename, map_location="cpu")
        eth_tab = torch.load(tabular_dir / "ETH" / tabular_filename, map_location="cpu")
        eth_tgt = torch.load(eth_subdir / "target_scores.pt", map_location="cpu")
        
        btc_len = btc_text.shape[0]
        eth_len = eth_text.shape[0]
        logger.info(f"✓ Dynamic asset lengths read: BTC={btc_len} samples, ETH={eth_len} samples")
        
        # Concatenate on first dimension (sequence panel stack)
        text_embeddings = torch.cat([btc_text, eth_text], dim=0)
        image_embeddings = torch.cat([btc_image, eth_image], dim=0)
        tabular_data = torch.cat([btc_tab, eth_tab], dim=0)
        target_scores = torch.cat([btc_tgt, eth_tgt], dim=0)
        
        total_samples = text_embeddings.shape[0]
    else:
        # Backward compatibility mode
        logger.info("Using legacy single-directory layout")
        metadata_path = features_dir / "split_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Split metadata not found at {metadata_path}. "
                "Make sure to extract full-sequence embeddings first."
            )
        
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        logger.info(f"✓ Split metadata loaded: total_samples={metadata['total_samples']}")
        
        print("[PROGRESS] Loading embeddings...")
        sys.stdout.flush()
        
        text_embeddings = torch.load(features_dir / "text_embeddings.pt", map_location="cpu")
        image_embeddings = torch.load(features_dir / image_filename, map_location="cpu")
        tabular_data = torch.load(features_dir / tabular_filename, map_location="cpu")
        target_scores = torch.load(features_dir / "target_scores.pt", map_location="cpu")
        
        total_samples = text_embeddings.shape[0]
        btc_len = total_samples // 2
        eth_len = total_samples // 2

    # ========== TARGET ENGINEERING ADJUSTMENTS ==========
    logger.info("Applying Target Engineering adjustments...")
    
    # 1. Scale y_baseline (index 0) by 1000.0 to prevent underflow / vanishing gradients
    target_scores[:, 0] = target_scores[:, 0] * 1000.0
    logger.info("  ✓ y_baseline scaled up by 1000.0")
    
    # 2. Clip y_heuristic (index 1) to [-5.0, 5.0] to handle high kurtosis outliers
    target_scores[:, 1] = torch.clamp(target_scores[:, 1], min=-5.0, max=5.0)
    logger.info("  ✓ y_heuristic clipped to [-5.0, 5.0]")
    
    # 3. y_vol_adj_return (index 2) - Keep raw un-clipped values
    logger.info("  ✓ y_vol_adj_return loaded as raw un-clipped values")

    logger.info(f"✓ Embeddings loaded: {total_samples} samples")
    logger.info(f"  text_embeddings: {text_embeddings.shape}")
    logger.info(f"  image_embeddings: {image_embeddings.shape}")
    logger.info(f"  tabular_data: {tabular_data.shape}")
    
    # Split calculation is done per asset timeline (using BTC length)
    test_pct = 0.15  # holdout fraction
    train_end_idx_per_asset = int(btc_len * (1.0 - test_pct))
    
    logger.info(f"✓ Scalers fitted dynamically per asset (btc_len={btc_len}, eth_len={eth_len})")
    
    tabular_data_raw = tabular_data    # (N, 7)  raw float32
    target_scores_raw = target_scores  # (N, 3)  raw float32
    timestamps = torch.arange(total_samples, dtype=torch.long)
    
    # Calculate walk-forward splits on the timeline of a SINGLE asset (using BTC as anchor)
    data_len = train_end_idx_per_asset
    window_size = int(0.7 * data_len)   # 70%
    step_size = int(0.15 * data_len) // num_folds
    
    logger.info(f"\nWalk-Forward Configuration (Simultaneous Dual-Asset):")
    logger.info(f"  Asset Timeline Len: {data_len} (test={btc_len - data_len} isolated)")
    logger.info(f"  Initial train window: {window_size}")
    logger.info(f"  Validation fold size: {step_size}")
    logger.info(f"  Number of folds: {num_folds}")
    
    # Generate folds
    fold_num = 0
    for train_slice, val_slice in walk_forward_split(data_len, window_size, step_size):
        fold_num += 1
        if fold_num > num_folds:
            break
        
        logger.info(f"\n" + "-" * 80)
        logger.info(f"Creating Fold {fold_num}/{num_folds}")
        logger.info(f"  Train slice (relative): [{train_slice.start}:{train_slice.stop}]")
        logger.info(f"  Val slice (relative):   [{val_slice.start}:{val_slice.stop}]")
        
        # ========== PER-FOLD SCALER FITTING (INDEPENDENT PER ASSET) ==========
        # 1. Fit BTC scalers
        btc_train_idx = slice(train_slice.start, train_slice.stop)
        btc_scaler_tab = StandardScaler().fit(tabular_data_raw[btc_train_idx].numpy())
        btc_scaler_tgt = RobustScaler().fit(target_scores_raw[btc_train_idx].numpy())
        
        # 2. Fit ETH scalers (offset by btc_len)
        eth_train_idx = slice(train_slice.start + btc_len, train_slice.stop + btc_len)
        eth_scaler_tab = StandardScaler().fit(tabular_data_raw[eth_train_idx].numpy())
        eth_scaler_tgt = RobustScaler().fit(target_scores_raw[eth_train_idx].numpy())
        
        # ========== APPLY SCALERS SEPARATELY TO PREVENT CROSS-CONTAMINATION ==========
        # Prepare target arrays for transformation
        tabular_scaled = tabular_data_raw.clone()
        target_scaled = target_scores_raw.clone()
        
        # Transform BTC region
        btc_full_idx = slice(0, btc_len)
        tabular_scaled[btc_full_idx] = torch.from_numpy(btc_scaler_tab.transform(tabular_data_raw[btc_full_idx].numpy())).float()
        target_scaled[btc_full_idx] = torch.from_numpy(btc_scaler_tgt.transform(target_scores_raw[btc_full_idx].numpy())).float()
        
        # Transform ETH region
        eth_full_idx = slice(btc_len, btc_len + eth_len)
        tabular_scaled[eth_full_idx] = torch.from_numpy(eth_scaler_tab.transform(tabular_data_raw[eth_full_idx].numpy())).float()
        target_scaled[eth_full_idx] = torch.from_numpy(eth_scaler_tgt.transform(target_scores_raw[eth_full_idx].numpy())).float()
        
        # Create datasets for this fold (use scaled tensors)
        train_dataset = WalkForwardDataset(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            tabular_data=tabular_scaled,
            target_scores=target_scaled,
            timestamps=timestamps,
            data_slice=train_slice,
            seq_len=config.data.seq_len,
            btc_len=btc_len,
            eth_len=eth_len,
            funding_horizon=funding_horizon,
        )

        val_dataset = WalkForwardDataset(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            tabular_data=tabular_scaled,
            target_scores=target_scaled,
            timestamps=timestamps,
            data_slice=val_slice,
            seq_len=config.data.seq_len,
            btc_len=btc_len,
            eth_len=eth_len,
            funding_horizon=funding_horizon,
        )
        
        logger.info(f"  ✓ Val dataset: {len(val_dataset)} sequences")
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            collate_fn=multimodal_collate_fn,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=True,
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            collate_fn=multimodal_collate_fn,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=False,
        )
        
        # Keep BTC target scaler as the default for target inverse transformation in logs
        scalers_dict = {
            "tabular_scaler": btc_scaler_tab,
            "target_scaler": btc_scaler_tgt,
            "eth_target_scaler": eth_scaler_tgt,  # Keep both in case we want to use them
        }
        
        yield fold_num, train_loader, val_loader, scalers_dict
