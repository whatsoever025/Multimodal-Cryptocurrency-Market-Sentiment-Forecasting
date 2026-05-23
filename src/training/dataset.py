"""
Simplified Offline Feature Dataset

Loads ALL data (embeddings + tabular + targets) from pre-extracted RAW Kaggle files.
No HuggingFace dataset loading. Scaling (StandardScaler for tabular, RobustScaler for targets)
are applied in-memory during dataset initialization based on training split.

CRITICAL: Safe sliding window logic prevents IndexError at dataset boundaries.
- __len__() returns total_samples - seq_len (to guarantee idx + seq_len exists)
- __getitem__(idx) fetches context [idx, idx+seq_len-1], target at idx+seq_len
"""

import os
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


class CryptoMultimodalDataset(torch.utils.data.Dataset):
    """
    Simplified offline multimodal dataset using pre-extracted RAW data.
    
    Loads:
    - Text embeddings (256-dim, pre-extracted by FinBERT)
    - Image embeddings (256-dim, pre-extracted by Vision Transformer ViT)
    - Tabular features (7 columns, RAW - will be scaled)
    - Target scores (RAW - will be scaled)
    
    All RAW data is loaded from Kaggle .pt files. NO HuggingFace dataset loading.
    Scaling is applied in-memory during initialization:
    - StandardScaler: Fitted on training split, applied to all splits
    - RobustScaler: Fitted on training split target scores, applied to all splits
    
    CRITICAL: Data structure:
    - text_embeddings: (total_samples, 256)
    - image_embeddings: (total_samples, 256)
    - tabular_features: (total_samples, 7) ← scaled by StandardScaler during init
    - target_scores: (total_samples,) ← scaled by RobustScaler during init
    
    Example:
        dataset = CryptoMultimodalDataset(
            split="train",
            seq_len=24,
            features_dir="/path/to/kaggle/features"
        )
        sample = dataset[0]
        # sample["tabular"]: (24, 7) ← scaled in-memory
        # sample["text_embedding"]: (24, 256)
        # sample["image_embedding"]: (24, 256)
        # sample["target"]: scalar ← scaled in-memory
    """
    
    def __init__(
        self,
        split: str = "train_val",
        seq_len: int = 24,
        features_dir: str = None,
        test_pct: float = 0.15,
        debug: bool = False,
    ):
        """
        Initialize simplified offline dataset with in-memory scaling.
        
        Args:
            split: "train", "validation", or "test_in_domain"
            seq_len: Sliding window length (hours)
            features_dir: Local directory containing pre-extracted Kaggle RAW features
            debug: If True, load only 100 samples for testing
        
        Scaling Strategy:
        - StandardScaler: Fit on training split tabular features, applied to all splits
        - RobustScaler: Fit on training split target scores, applied to all splits
        - If split != "train": Load scalers from training data first
        
        Note: features_dir should contain one of:
        
        Option 1 - Per-split files (legacy):
            - text_embeddings_{split}.pt
            - image_embeddings_{split}.pt
            - tabular_features_{split}.pt (RAW, no scaling)
            - target_scores_{split}.pt (RAW, no scaling)
        
        Option 2 - Consolidated files (new, from Kaggle dataset):
            - text_embeddings.pt (all samples)
            - image_embeddings.pt (all samples)
            - tabular_features.pt (all samples)
            - target_scores.pt (all samples)
            - split_metadata.json (defines train/val/test boundaries)
        """
        self.split = split
        self.seq_len = seq_len
        self.features_dir = Path(features_dir) if features_dir else None
        self.test_pct = test_pct
        self.debug = debug
        
        # Validate: must have features directory
        if not self.features_dir:
            raise ValueError("Must provide features_dir with pre-extracted Kaggle features")
        
        if not self.features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_dir}")
        
        logger.info(
            f"Initializing CryptoMultimodalDataset\n"
            f"  Split: {split}, Seq Len: {seq_len}"
        )
        print(f"[PROGRESS] Loading dataset ({split})...")
        sys.stdout.flush()
        
        # ==================== LOAD PRE-EXTRACTED EMBEDDINGS ====================
        logger.info("Loading pre-extracted embeddings...")
        print("[PROGRESS] Loading pre-extracted embeddings...")
        sys.stdout.flush()
        
        embed_start = time.time()
        self._load_embeddings_from_disk(split)
        embed_time = time.time() - embed_start
        logger.info(f"✓ Embeddings loaded in {format_duration(embed_time)}")
        print(f"[PROGRESS] ✓ Embeddings loaded ({format_duration(embed_time)})")
        sys.stdout.flush()
        
        # ==================== LOAD RAW TABULAR FEATURES & APPLY SCALING ====================
        logger.info("Loading RAW tabular features and targets...")
        print("[PROGRESS] Loading RAW tabular features and targets...")
        sys.stdout.flush()
        
        tabular_start = time.time()
        self._load_tabular_and_targets(split)
        tabular_time = time.time() - tabular_start
        logger.info(f"✓ Tabular features and targets loaded & scaled in {format_duration(tabular_time)}")
        print(f"[PROGRESS] ✓ Tabular features and targets loaded & scaled")
        sys.stdout.flush()
        
        # ==================== SAFE SLIDING WINDOW ====================
        self.max_valid_idx = self.total_samples - seq_len
        
        if self.max_valid_idx <= 0:
            raise ValueError(
                f"Dataset too small for seq_len={seq_len}. "
                f"Need at least {seq_len + 1} samples, got {self.total_samples}"
            )
        
        logger.info(
            f"Safe sliding window: __len__() will return {self.max_valid_idx} "
            f"(indices 0 to {self.max_valid_idx - 1})"
        )
        
        logger.info(f"✓ Dataset ready: {len(self)} valid sequences of length {seq_len}")
    
    def _load_embeddings_from_disk(self, split: str) -> None:
        """Load pre-extracted embeddings from disk as contiguous tensors.
        
        Supports two formats:
        1. Per-split files (legacy): text_embeddings_{split}.pt, image_embeddings_{split}.pt
        2. Consolidated files: text_embeddings.pt, image_embeddings.pt + split_metadata.json
        """
        import json
        
        # Try to load per-split files first (backward compatibility)
        text_embed_path = self.features_dir / f"text_embeddings_{split}.pt"
        image_embed_path = self.features_dir / f"image_embeddings_{split}.pt"
        
        if text_embed_path.exists() and image_embed_path.exists():
            # Per-split format
            logger.info(f"[Per-split format] Loading text embeddings from {text_embed_path}...")
            text_raw = torch.load(text_embed_path, map_location="cpu")
            self.text_embeddings = text_raw.contiguous()
            logger.info(f"✓ Text embeddings: {self.text_embeddings.shape}, contiguous={self.text_embeddings.is_contiguous()}")
            
            logger.info(f"[Per-split format] Loading image embeddings from {image_embed_path}...")
            image_raw = torch.load(image_embed_path, map_location="cpu")
            self.image_embeddings = image_raw.contiguous()
            logger.info(f"✓ Image embeddings: {self.image_embeddings.shape}, contiguous={self.image_embeddings.is_contiguous()}")
            
            # Store total samples from embeddings shape
            self.total_samples = self.text_embeddings.shape[0]
            
            # Validate shapes match
            assert self.image_embeddings.shape[0] == self.total_samples, \
                f"Image embeddings mismatch: {self.image_embeddings.shape[0]} vs {self.total_samples}"
        else:
            # Try consolidated format with split_metadata.json
            metadata_path = self.features_dir / "split_metadata.json"
            text_embed_path = self.features_dir / "text_embeddings.pt"
            image_embed_path = self.features_dir / "image_embeddings.pt"
            
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Split metadata not found: {metadata_path}. "
                    f"Expected either per-split files (text_embeddings_{split}.pt) or "
                    f"consolidated files (text_embeddings.pt + split_metadata.json)"
                )
            
            if not text_embed_path.exists():
                raise FileNotFoundError(f"Text embeddings not found: {text_embed_path}")
            if not image_embed_path.exists():
                raise FileNotFoundError(f"Image embeddings not found: {image_embed_path}")
            
            # Load split metadata
            logger.info(f"[Consolidated format] Loading split metadata from {metadata_path}...")
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # v5: metadata only has total_samples; boundaries computed from test_pct
            N = metadata["total_samples"]
            test_start = int(N * (1.0 - self.test_pct))
            if split == "train_val":
                start_idx, end_idx = 0, test_start
            elif split == "test":
                start_idx, end_idx = test_start, N
            elif split == "all":
                start_idx, end_idx = 0, N
            else:
                raise ValueError(f"split must be 'train_val', 'test', or 'all', got {split!r}")

            logger.info(f"✓ Split {split!r}: [{start_idx}:{end_idx}] (test_pct={self.test_pct})")
            
            # Load full consolidated embeddings
            logger.info(f"[Consolidated format] Loading text embeddings from {text_embed_path}...")
            text_full = torch.load(text_embed_path, map_location="cpu")
            logger.info(f"  Full shape: {text_full.shape}")
            self.text_embeddings = text_full[start_idx:end_idx].contiguous()
            logger.info(f"✓ Text embeddings ({split}): {self.text_embeddings.shape}, contiguous={self.text_embeddings.is_contiguous()}")
            
            logger.info(f"[Consolidated format] Loading image embeddings from {image_embed_path}...")
            image_full = torch.load(image_embed_path, map_location="cpu")
            logger.info(f"  Full shape: {image_full.shape}")
            self.image_embeddings = image_full[start_idx:end_idx].contiguous()
            logger.info(f"✓ Image embeddings ({split}): {self.image_embeddings.shape}, contiguous={self.image_embeddings.is_contiguous()}")
            
            # Store total samples from embeddings shape
            self.total_samples = self.text_embeddings.shape[0]
            
            # Validate shapes match
            assert self.image_embeddings.shape[0] == self.total_samples, \
                f"Image embeddings mismatch: {self.image_embeddings.shape[0]} vs {self.total_samples}"
    
    def _load_tabular_and_targets(self, split: str) -> None:
        """Load RAW tabular features and target scores, then apply scalers.
        
        Supports two formats:
        1. Per-split files (legacy): tabular_features_{split}.pt, target_scores_{split}.pt
        2. Consolidated files: tabular_features.pt, target_scores.pt + split_metadata.json
        
        Scaling Strategy:
        - For 'train' split: Fit StandardScaler and RobustScaler on raw data, apply in-place
        - For 'validation'/'test_in_domain': Load scalers from training data, apply scaled data
        
        This ensures proper data leakage prevention:
        - Training: Learn scaler statistics from train data
        - Validation/Test: Use train statistics to scale val/test data
        """
        import json
        
        # Try to load per-split files first (backward compatibility)
        tabular_path = self.features_dir / f"tabular_features_{split}.pt"
        target_path = self.features_dir / f"target_scores_{split}.pt"
        
        if tabular_path.exists() and target_path.exists():
            # Per-split format
            logger.info(f"[Per-split format] Loading RAW tabular features from {tabular_path}...")
            tabular_raw = torch.load(tabular_path, map_location="cpu")  # (total_samples, 7)
            logger.info(f"✓ Raw tabular features: {tabular_raw.shape}")
            
            logger.info(f"[Per-split format] Loading RAW target scores from {target_path}...")
            target_raw = torch.load(target_path, map_location="cpu")  # (total_samples,)
            logger.info(f"✓ Raw target scores: {target_raw.shape}")
        else:
            # Try consolidated format with split_metadata.json
            metadata_path = self.features_dir / "split_metadata.json"
            tabular_path = self.features_dir / "tabular_features.pt"
            target_path = self.features_dir / "target_scores.pt"
            
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Split metadata not found: {metadata_path}. "
                    f"Expected either per-split files (tabular_features_{split}.pt) or "
                    f"consolidated files (tabular_features.pt + split_metadata.json)"
                )
            
            if not tabular_path.exists():
                raise FileNotFoundError(f"Tabular features not found: {tabular_path}")
            if not target_path.exists():
                raise FileNotFoundError(f"Target scores not found: {target_path}")
            
            # Load split metadata
            logger.info(f"[Consolidated format] Loading split metadata from {metadata_path}...")
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # v5: metadata only has total_samples; boundaries computed from test_pct
            N = metadata["total_samples"]
            test_start = int(N * (1.0 - self.test_pct))
            if split == "train_val":
                start_idx, end_idx = 0, test_start
            elif split == "test":
                start_idx, end_idx = test_start, N
            elif split == "all":
                start_idx, end_idx = 0, N
            else:
                raise ValueError(f"split must be 'train_val', 'test', or 'all', got {split!r}")

            logger.info(f"✓ Split {split!r}: [{start_idx}:{end_idx}] (test_pct={self.test_pct})")
            
            # Load full consolidated files
            logger.info(f"[Consolidated format] Loading RAW tabular features from {tabular_path}...")
            tabular_full = torch.load(tabular_path, map_location="cpu")
            logger.info(f"  Full shape: {tabular_full.shape}")
            tabular_raw = tabular_full[start_idx:end_idx]
            logger.info(f"✓ Raw tabular features ({split}): {tabular_raw.shape}")
            
            logger.info(f"[Consolidated format] Loading RAW target scores from {target_path}...")
            target_full = torch.load(target_path, map_location="cpu")
            logger.info(f"  Full shape: {target_full.shape}")
            target_raw = target_full[start_idx:end_idx]
            logger.info(f"✓ Raw target scores ({split}): {target_raw.shape}")
        
        # ========== APPLY SCALERS ==========
        if split in ("train_val", "all"):
            # TRAIN: Fit scalers on raw data, then apply
            logger.info("Fitting StandardScaler on tabular features (training split)...")
            tabular_scaler = StandardScaler()
            tabular_np = tabular_raw.numpy()  # (total_samples, 7)
            tabular_scaled_np = tabular_scaler.fit_transform(tabular_np)
            self.tabular_data = torch.from_numpy(tabular_scaled_np).float()
            logger.info(f"✓ StandardScaler fitted and applied: {self.tabular_data.shape}")
            logger.info(f"  Scaler mean: {tabular_scaler.mean_}")
            logger.info(f"  Scaler scale: {tabular_scaler.scale_}")
            
            logger.info("Fitting RobustScaler on targets (N,3) [y_baseline,y_heuristic,y_pca]...")
            target_scaler = RobustScaler()
            target_np = target_raw.numpy()  # (N, 3)
            target_scaled_np = target_scaler.fit_transform(target_np)  # (N, 3)
            self.target_scores = torch.from_numpy(target_scaled_np).float()
            logger.info(f"✓ RobustScaler fitted: {self.target_scores.shape}")
            
            # Store scalers for validation/test (will be loaded by those splits)
            self.tabular_scaler = tabular_scaler
            self.target_scaler = target_scaler
        else:
            # VALIDATION/TEST: Load scalers from training split, apply to current split
            logger.info(f"Loading scalers from training split for {split}...")
            train_dataset = CryptoMultimodalDataset(
                split="train_val",
                seq_len=self.seq_len,
                features_dir=str(self.features_dir),
                test_pct=self.test_pct,
                debug=self.debug
            )
            tabular_scaler = train_dataset.tabular_scaler
            target_scaler = train_dataset.target_scaler
            
            # Apply training scalers to current split data
            logger.info(f"Applying StandardScaler from training to {split} tabular features...")
            tabular_np = tabular_raw.numpy()
            tabular_scaled_np = tabular_scaler.transform(tabular_np)
            self.tabular_data = torch.from_numpy(tabular_scaled_np).float()
            logger.info(f"✓ Tabular features scaled: {self.tabular_data.shape}")
            
            logger.info(f"Applying RobustScaler from training to {split} target scores...")
            target_np = target_raw.numpy()  # (N, 3)
            target_scaled_np = target_scaler.transform(target_np)  # (N, 3)
            self.target_scores = torch.from_numpy(target_scaled_np).float()
            logger.info(f"✓ Target scores scaled: {self.target_scores.shape}")
            
            self.tabular_scaler = tabular_scaler
            self.target_scaler = target_scaler
        
        # Make contiguous for efficient slicing
        self.tabular_data = self.tabular_data.contiguous()
        self.target_scores = self.target_scores.contiguous()
        
        # Create timestamps (indices for reference)
        self.timestamps = torch.arange(self.total_samples, dtype=torch.long)
        logger.info(f"✓ Timestamps tensor: {self.timestamps.shape}")
        
        # Validate shapes match total_samples
        assert self.tabular_data.shape[0] == self.total_samples, \
            f"Tabular mismatch: {self.tabular_data.shape[0]} vs {self.total_samples}"
        assert self.target_scores.shape[0] == self.total_samples, \
            f"Target mismatch: {self.target_scores.shape[0]} vs {self.total_samples}"
    
    def __len__(self) -> int:
        """
        CRITICAL: Return safe length to prevent IndexError.
        
        Safe formula: total_samples - seq_len
        Example: 62,266 samples - 24 seq_len = 62,242 valid indices
        For idx=62,241 (last valid): can fetch target at idx+seq_len=62,265 ✓
        """
        return self.max_valid_idx
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """OPTIMIZED: Zero-copy slicing - O(1) operation per index.
        
        CRITICAL: This method does NOTHING except slice pre-computed tensors.
        - NO dataset iteration
        - NO feature extraction  
        - NO tensor conversions
        - NO transformations/scalings (all pre-done)
        - Pure zero-copy view slicing from pre-allocated tensors
        
        Args:
            idx: Index of sequence start (must be in [0, __len__()))
        
        Returns:
            Dict with VIEWS (not copies) into pre-computed tensors:
                - tabular: (seq_len, 7) float32 view ← ALREADY SCALED
                - text_embedding: (seq_len, 256) float32 view
                - image_embedding: (seq_len, 256) float32 view
                - target: scalar float32 tensor ← ALREADY SCALED
                - timestamp: scalar int64 tensor
        
        Raises:
            IndexError: If idx >= __len__()
        """
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {len(self)}"
            )
        
        # Pure slicing - these are views into contiguous tensors (O(1) operation)
        return {
            "tabular": self.tabular_data[idx:idx + self.seq_len],
            "text_embedding": self.text_embeddings[idx:idx + self.seq_len],
            "image_embedding": self.image_embeddings[idx:idx + self.seq_len],
            "target": self.target_scores[idx + self.seq_len],
            "timestamp": self.timestamps[idx + self.seq_len],
        }


def multimodal_collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Collate function for multimodal batches.
    
    Args:
        batch: List of dicts from CryptoMultimodalDataset
    
    Returns:
        Dict with stacked tensors:
            - tabular: (batch_size, seq_len, 7)
            - text_embedding: (batch_size, seq_len, 256)
            - image_embedding: (batch_size, seq_len, 256)
             - target: (batch_size,)
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


def create_dataloaders(
    config,
    splits: Optional[Tuple[str, str, str]] = ("train", "validation", "test_in_domain"),
    features_dir: str = None,
    num_workers: int = 0,  # Always 0 on Kaggle (multi-worker deadlock fix)
    pin_memory: bool = True,
) -> Tuple[Dict[str, torch.utils.data.DataLoader], Dict[str, Any]]:
    """
    Create DataLoaders for all splits with progress tracking.
    
    All data is pre-scaled - NO scaling is applied during training.
    
    CRITICAL: DataLoader Optimization
    - num_workers=0: FORCED on Kaggle (multi-worker deadlock issues)
      Even if config.data.num_workers is set to 4, we override to 0
    - pin_memory=True: Transfer data to GPU via pinned memory (faster)
    
    Args:
        config: ExperimentConfig instance
        splits: Tuple of (train_split, val_split, test_split)
        features_dir: Local directory with pre-extracted Kaggle features
        num_workers: Number of data loading workers (IGNORED - always 0 for Kaggle safety)
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        Tuple of (dataloaders_dict, scalers_dict) where:
        - dataloaders_dict: Dict with keys "train", "validation", "test" and DataLoader values
        - scalers_dict: Dict with keys "tabular_scaler" and "target_scaler" for inverse transforms
    """
    # CRITICAL: Force num_workers=0 regardless of parameter or config
    num_workers = 0
    
    dataloaders = {}
    scalers_dict = {}  # Store scalers from datasets for inverse transforms
    
    overall_start = time.time()
    
    with tqdm(total=len(splits), desc="Creating DataLoaders", unit="split") as progress:
        for split_idx, split_name in enumerate(splits, 1):
            print(f"\n[PROGRESS] Creating DataLoader for {split_name} (num_workers=0, pin_memory={pin_memory})...")
            sys.stdout.flush()
            
            split_start = time.time()
            
            # Create dataset
            dataset = CryptoMultimodalDataset(
                split=split_name,
                seq_len=config.data.seq_len,
                features_dir=features_dir,
                debug=config.debug if hasattr(config, "debug") else False,
            )
            
            # Store scalers from train split for inverse transforms
            if split_name == "train":
                scalers_dict["tabular_scaler"] = dataset.tabular_scaler
                scalers_dict["target_scaler"] = dataset.target_scaler
                logger.info("✓ Scalers extracted from training dataset")
            
            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.data.batch_size,
                shuffle=(split_name == "train"),
                collate_fn=multimodal_collate_fn,
                num_workers=0,  # Always 0 on Kaggle
                pin_memory=pin_memory,
                drop_last=(split_name == "train"),  # Drop incomplete batches in training
            )
            
            dataloaders[split_name] = dataloader
            
            split_time = time.time() - split_start
            logger.info(f"✓ Created {split_name} DataLoader ({len(dataloader)} batches) in {format_duration(split_time)}")
            print(f"[PROGRESS] ✓ {split_name}: {len(dataloader)} batches")
            sys.stdout.flush()
            
            progress.update(1)
    
    total_time = time.time() - overall_start
    logger.info(f"All DataLoaders created in {format_duration(total_time)}")
    print(f"\n[PROGRESS] ✓ All dataloaders created! Total time: {format_duration(total_time)}")
    sys.stdout.flush()
    
    return dataloaders, scalers_dict


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
    Walk-forward dataset that slices pre-loaded embeddings/features for a given fold.
    
    Wraps the complete dataset and provides views for a specific train/val split.
    """
    
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
    ):
        self.seq_len = seq_len
        self.total_samples_per_asset = total_samples_per_asset
        
        # Save full data
        self.text_full = text_embeddings
        self.image_full = image_embeddings
        self.tabular_full = tabular_data
        self.target_full = target_scores
        self.timestamps_full = timestamps
        
        # We construct a list of valid starting indices for BTC and ETH separately.
        # Since we predict y_baseline at t+8, we must ensure the index real_idx + seq_len + 7
        # is always within the asset boundaries. Thus we subtract 7 from the max index.
        start = data_slice.start if data_slice.start is not None else 0
        stop = data_slice.stop if data_slice.stop is not None else total_samples_per_asset
        
        # Buffer of 7 steps to prevent index overflow on the 8h target
        buffer = 7
        
        # Valid starts for BTC (within the slice)
        btc_valid_starts = list(range(start, min(stop - seq_len - buffer + 1, total_samples_per_asset - seq_len - buffer + 1)))
        
        # Valid starts for ETH (shifted by total_samples_per_asset)
        eth_valid_starts = list(range(
            start + total_samples_per_asset, 
            min(stop - seq_len - buffer + 1 + total_samples_per_asset, 2 * total_samples_per_asset - seq_len - buffer + 1)
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
        
        # Col 1 (y_heuristic) and Col 2 (y_pca) remain at t+1 (real_idx + seq_len)
        target_heuristic = self.target_full[real_idx + self.seq_len, 1]
        target_pca = self.target_full[real_idx + self.seq_len, 2]
        
        target_vector = torch.stack([target_baseline, target_heuristic, target_pca])
        
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
    logger.info("WALK-FORWARD VALIDATION: Loading embeddings")
    logger.info("=" * 80)
    
    # Load split metadata
    metadata_path = features_dir / "split_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Split metadata not found at {metadata_path}. "
            "Make sure to extract full-sequence embeddings first."
        )
    
    import json
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    logger.info(f"✓ Split metadata loaded:")
    logger.info(f"  Total samples: {metadata['total_samples']}")
    
    # Load full-sequence embeddings
    print("[PROGRESS] Loading embeddings...")
    sys.stdout.flush()
    
    text_embeddings = torch.load(
        features_dir / "text_embeddings.pt",
        map_location="cpu"
    )
    image_embeddings = torch.load(
        features_dir / "image_embeddings.pt",
        map_location="cpu"
    )
    tabular_data = torch.load(
        features_dir / "tabular_features.pt",
        map_location="cpu"
    )
    target_scores = torch.load(
        features_dir / "target_scores.pt",
        map_location="cpu"
    )
    
    # ========== TARGET ENGINEERING ADJUSTMENTS ==========
    logger.info("Applying Target Engineering adjustments...")
    
    # 1. Scale y_baseline (index 0) by 1000.0 to prevent underflow / vanishing gradients
    target_scores[:, 0] = target_scores[:, 0] * 1000.0
    logger.info("  ✓ y_baseline scaled up by 1000.0")
    
    # 2. Clip y_heuristic (index 1) to [-5.0, 5.0] to handle high kurtosis outliers
    target_scores[:, 1] = torch.clamp(target_scores[:, 1], min=-5.0, max=5.0)
    logger.info("  ✓ y_heuristic clipped to [-5.0, 5.0]")
    
    # 3. y_pca (index 2) - Keep as is (analyzed as mostly noise)
    logger.info("  ✓ y_pca loaded as-is")

    
    total_samples = text_embeddings.shape[0]
    logger.info(f"\u2713 Embeddings loaded: {total_samples} samples")
    logger.info(f"  text_embeddings: {text_embeddings.shape}")
    logger.info(f"  image_embeddings: {image_embeddings.shape}")
    logger.info(f"  tabular_data: {tabular_data.shape}")
      # Split calculation is done per asset
    total_samples_per_asset = total_samples // 2  # 44500
    
    test_pct = 0.15  # holdout fraction
    train_end_idx_per_asset = int(total_samples_per_asset * (1.0 - test_pct))  # 37825
    
    logger.info(f"✓ Scalers fitted dynamically per asset (total_samples_per_asset={total_samples_per_asset})")
    
    tabular_data_raw = tabular_data    # (N, 7)  raw float32
    target_scores_raw = target_scores  # (N, 3)  raw float32
    timestamps = torch.arange(total_samples, dtype=torch.long)
    
    # Calculate walk-forward splits on the timeline of a SINGLE asset
    data_len = train_end_idx_per_asset  # 37825
    window_size = int(0.7 * data_len)   # 70%
    step_size = int(0.15 * data_len) // num_folds
    
    logger.info(f"\nWalk-Forward Configuration (Simultaneous Dual-Asset):")
    logger.info(f"  Asset Timeline Len: {data_len} (test={total_samples_per_asset - data_len} isolated)")
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
        
        # 2. Fit ETH scalers (offset by total_samples_per_asset)
        eth_train_idx = slice(train_slice.start + total_samples_per_asset, train_slice.stop + total_samples_per_asset)
        eth_scaler_tab = StandardScaler().fit(tabular_data_raw[eth_train_idx].numpy())
        eth_scaler_tgt = RobustScaler().fit(target_scores_raw[eth_train_idx].numpy())
        
        # ========== APPLY SCALERS SEPARATELY TO PREVENT CROSS-CONTAMINATION ==========
        # Prepare target arrays for transformation
        tabular_scaled = tabular_data_raw.clone()
        target_scaled = target_scores_raw.clone()
        
        # Transform BTC region
        btc_full_idx = slice(0, total_samples_per_asset)
        tabular_scaled[btc_full_idx] = torch.from_numpy(btc_scaler_tab.transform(tabular_data_raw[btc_full_idx].numpy())).float()
        target_scaled[btc_full_idx] = torch.from_numpy(btc_scaler_tgt.transform(target_scores_raw[btc_full_idx].numpy())).float()
        
        # Transform ETH region
        eth_full_idx = slice(total_samples_per_asset, 2 * total_samples_per_asset)
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
            total_samples_per_asset=total_samples_per_asset,
        )
        
        val_dataset = WalkForwardDataset(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            tabular_data=tabular_scaled,
            target_scores=target_scaled,
            timestamps=timestamps,
            data_slice=val_slice,
            seq_len=config.data.seq_len,
            total_samples_per_asset=total_samples_per_asset,
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



if __name__ == "__main__":
    """Test dataset loading and safe sliding window."""
    from training.config import ExperimentConfig
    
    config = ExperimentConfig(debug=True)
    
    print("=" * 80)
    print("Testing CryptoMultimodalDataset (Simplified)")
    print("=" * 80)
    
    # Test dataset initialization
    print("\n1. Loading dataset...")
    try:
        dataset = CryptoMultimodalDataset(
            split="train",
            seq_len=config.data.seq_len,
            features_dir="./data/features",
            debug=True,
        )
        print(f"   ✓ Dataset loaded")
        print(f"   Dataset length: {len(dataset)}")
        print(f"   Total samples in split: {dataset.total_samples}")
        print(f"   Max valid index: {dataset.max_valid_idx}")
    except FileNotFoundError as e:
        print(f"   ⚠ Skipping dataset test (features not found): {e}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✅ Dataset module ready!")
    print("=" * 80)
