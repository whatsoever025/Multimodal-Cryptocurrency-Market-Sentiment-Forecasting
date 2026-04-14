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
    - Image embeddings (256-dim, pre-extracted by ResNet50)
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
        split: str = "train",
        seq_len: int = 24,
        features_dir: str = None,
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
        
        Note: features_dir should contain:
            - text_embeddings_{split}.pt
            - image_embeddings_{split}.pt
            - tabular_features_{split}.pt (RAW, no scaling)
            - target_scores_{split}.pt (RAW, no scaling)
        """
        self.split = split
        self.seq_len = seq_len
        self.features_dir = Path(features_dir) if features_dir else None
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
        """Load pre-extracted embeddings from disk as contiguous tensors."""
        text_embed_path = self.features_dir / f"text_embeddings_{split}.pt"
        image_embed_path = self.features_dir / f"image_embeddings_{split}.pt"
        
        if not text_embed_path.exists():
            raise FileNotFoundError(f"Text embeddings not found: {text_embed_path}")
        if not image_embed_path.exists():
            raise FileNotFoundError(f"Image embeddings not found: {image_embed_path}")
        
        logger.info(f"Loading text embeddings from {text_embed_path}...")
        text_raw = torch.load(text_embed_path, map_location="cpu")
        self.text_embeddings = text_raw.contiguous()
        logger.info(f"✓ Text embeddings: {self.text_embeddings.shape}, contiguous={self.text_embeddings.is_contiguous()}")
        
        logger.info(f"Loading image embeddings from {image_embed_path}...")
        image_raw = torch.load(image_embed_path, map_location="cpu")
        self.image_embeddings = image_raw.contiguous()
        logger.info(f"✓ Image embeddings: {self.image_embeddings.shape}, contiguous={self.image_embeddings.is_contiguous()}")
        
        # Store total samples from embeddings shape
        self.total_samples = self.text_embeddings.shape[0]
        
        # Validate shapes match
        assert self.image_embeddings.shape[0] == self.total_samples, \
            f"Image embeddings mismatch: {self.image_embeddings.shape[0]} vs {self.total_samples}"
    
    def _load_tabular_and_targets(self, split: str) -> None:
        """Load RAW tabular features and target scores, then apply scalers.
        
        Scaling Strategy:
        - For 'train' split: Fit StandardScaler and RobustScaler on raw data, apply in-place
        - For 'validation'/'test_in_domain': Load scalers from training data, apply scaled data
        
        This ensures proper data leakage prevention:
        - Training: Learn scaler statistics from train data
        - Validation/Test: Use train statistics to scale val/test data
        """
        tabular_path = self.features_dir / f"tabular_features_{split}.pt"
        target_path = self.features_dir / f"target_scores_{split}.pt"
        
        if not tabular_path.exists():
            raise FileNotFoundError(f"Tabular features not found: {tabular_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Target scores not found: {target_path}")
        
        logger.info(f"Loading RAW tabular features from {tabular_path}...")
        tabular_raw = torch.load(tabular_path, map_location="cpu")  # (total_samples, 7)
        logger.info(f"✓ Raw tabular features: {tabular_raw.shape}")
        
        logger.info(f"Loading RAW target scores from {target_path}...")
        target_raw = torch.load(target_path, map_location="cpu")  # (total_samples,)
        logger.info(f"✓ Raw target scores: {target_raw.shape}")
        
        # ========== APPLY SCALERS ==========
        if split == "train":
            # TRAIN: Fit scalers on raw data, then apply
            logger.info("Fitting StandardScaler on tabular features (training split)...")
            tabular_scaler = StandardScaler()
            tabular_np = tabular_raw.numpy()  # (total_samples, 7)
            tabular_scaled_np = tabular_scaler.fit_transform(tabular_np)
            self.tabular_data = torch.from_numpy(tabular_scaled_np).float()
            logger.info(f"✓ StandardScaler fitted and applied: {self.tabular_data.shape}")
            logger.info(f"  Scaler mean: {tabular_scaler.mean_}")
            logger.info(f"  Scaler scale: {tabular_scaler.scale_}")
            
            logger.info("Fitting RobustScaler on target scores (training split)...")
            target_scaler = RobustScaler()
            target_np = target_raw.numpy().reshape(-1, 1)  # (total_samples, 1) for sklearn
            target_scaled_np = target_scaler.fit_transform(target_np).squeeze()  # Back to 1D
            self.target_scores = torch.from_numpy(target_scaled_np).float()
            logger.info(f"✓ RobustScaler fitted and applied: {self.target_scores.shape}")
            logger.info(f"  Scaler center (median): {target_scaler.center_}")
            logger.info(f"  Scaler scale (IQR): {target_scaler.scale_}")
            
            # Store scalers for validation/test (will be loaded by those splits)
            self.tabular_scaler = tabular_scaler
            self.target_scaler = target_scaler
        else:
            # VALIDATION/TEST: Load scalers from training split, apply to current split
            logger.info(f"Loading scalers from training split for {split}...")
            train_dataset = CryptoMultimodalDataset(
                split="train",
                seq_len=self.seq_len,
                features_dir=str(self.features_dir),
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
            target_np = target_raw.numpy().reshape(-1, 1)
            target_scaled_np = target_scaler.transform(target_np).squeeze()
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
    
    return stacked


def create_dataloaders(
    config,
    splits: Optional[Tuple[str, str, str]] = ("train", "validation", "test_in_domain"),
    features_dir: str = None,
    num_workers: int = 0,  # Always 0 on Kaggle (multi-worker deadlock fix)
    pin_memory: bool = True,
) -> Dict[str, torch.utils.data.DataLoader]:
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
        Dict with keys "train", "validation", "test" and DataLoader values
    """
    # CRITICAL: Force num_workers=0 regardless of parameter or config
    num_workers = 0
    
    dataloaders = {}
    
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
    
    return dataloaders


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
