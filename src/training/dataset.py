"""
Walk-Forward Multimodal Dataset

Provides WalkForwardDataset and create_walk_forward_dataloaders for
temporal walk-forward cross-validation over multi-asset (BTC + ETH) panel data.

Features are loaded from pre-extracted .pt files stored in per-asset
subdirectories (BTC/, ETH/) or a single consolidated directory.

Scaling (StandardScaler for tabular, RobustScaler for targets) is fitted
independently per asset on each fold's training window — no cross-asset or
temporal leakage.

Key classes / functions:
- WalkForwardDataset: Sliding-window dataset supporting dual-asset panels.
- walk_forward_split: Generator yielding (train_slice, val_slice) pairs.
- create_walk_forward_dataloaders: Full pipeline — load data, engineer
  targets, fit per-fold per-asset scalers, yield (fold, train_loader,
  val_loader, scalers_dict).
- multimodal_collate_fn: Collate function for DataLoader.
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
    
    Args:
        batch: List of dicts from WalkForwardDataset
    
    Returns:
        Dict with stacked tensors:
            - tabular: (batch_size, seq_len, 7)
            - text_embedding: (batch_size, seq_len, 256)
            - image_embedding: (batch_size, seq_len, 256)
             - target: (batch_size, 3)
            - timestamp: (batch_size,)
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
    Generate walk-forward train/val splits.
    
    CRITICAL: Walk-forward validation prevents look-ahead bias by training on
    [0, train_end], validating on [train_end, train_end+step_size], then
    progressively advancing the window forward.
    
    Args:
        data_len: Total data length
        window_size: Initial train window size
        step_size: Number of samples for each validation fold
    
    Yields:
        Tuple of (train_slice, val_slice) representing temporal folds
    
    Example:
        data_len = 100, window_size = 70, step_size = 15
        Fold 1: train=[0:70], val=[70:85]
        Fold 2: train=[0:85], val=[85:100]
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
    Walk-forward dataset supporting multi-asset panel data (BTC + ETH concatenated).
    Ensures sliding windows never cross the boundary between assets.
    Predicts baseline target 8 hours ahead, and other targets 1 hour ahead.
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
    ):
        self.seq_len = seq_len
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
        
        # Buffer to prevent index overflow on the 8h-ahead target.
        # __getitem__ accesses target_full[real_idx + seq_len + 7], so we need:
        #   real_idx + seq_len + 7  <=  N - 1   →   real_idx  <=  N - seq_len - 8
        # With range(..., N - seq_len - buffer + 1) the last value is N - seq_len - buffer,
        # so buffer must equal 8 (not 7) for the last target index to be N - 1.
        buffer = 8
        
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
            f"  Created 8h-shifted WalkForwardDataset slice [{start}:{stop}] -> "
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
        # Col 0 (y_baseline) is target_raw_funding, predict at t+8 (real_idx + seq_len + 7)
        target_baseline = self.target_full[real_idx + self.seq_len + 7, 0]
        
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
):
    """
    Create walk-forward validation folds.
    
    Loads concatenated train/validation/test embeddings, then applies
    temporal walk-forward splits for proper chronological validation.
    
    CRITICAL: Walk-forward respects temporal ordering:
    - Fold 1: Train on [0:70%], validate on [70%:85%]
    - Fold 2: Train on [0:85%], validate on [85%:100%]
    
    Args:
        config: ExperimentConfig instance
        features_dir: Local directory with extracted embeddings
        num_folds: Number of temporal validation folds
        num_workers: Data loading workers (always 0 on Kaggle)
        pin_memory: Pin memory for GPU transfer
    
    Yields:
        Tuple of (fold_num, train_loader, val_loader, scalers_dict)
    """
    num_workers = 0  # Force num_workers=0 for Kaggle safety
    features_dir = Path(features_dir) if features_dir else Path("./data/features")
    
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
        btc_image = torch.load(btc_subdir / "image_embeddings.pt", map_location="cpu")
        btc_tab = torch.load(btc_subdir / "tabular_features.pt", map_location="cpu")
        btc_tgt = torch.load(btc_subdir / "target_scores.pt", map_location="cpu")
        
        # Load ETH
        eth_text = torch.load(eth_subdir / "text_embeddings.pt", map_location="cpu")
        eth_image = torch.load(eth_subdir / "image_embeddings.pt", map_location="cpu")
        eth_tab = torch.load(eth_subdir / "tabular_features.pt", map_location="cpu")
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
        image_embeddings = torch.load(features_dir / "image_embeddings.pt", map_location="cpu")
        tabular_data = torch.load(features_dir / "tabular_features.pt", map_location="cpu")
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
