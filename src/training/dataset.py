"""
Offline Feature Dataset

Loads pre-extracted text and image embeddings from disk, applies safe sliding window logic.

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

try:
    from datasets import load_dataset, concatenate_datasets
except ImportError:
    raise ImportError("'datasets' package required: pip install datasets")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError("'huggingface_hub' required: pip install huggingface_hub")

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError("'scikit-learn' package required: pip install scikit-learn")

from tqdm import tqdm

logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


class CryptoMultimodalDataset(torch.utils.data.Dataset):
    """
    Offline multimodal dataset using pre-extracted embeddings.
    
    Loads frozen FinBERT text embeddings and ResNet50 image embeddings from disk.
    Applies safe sliding window on tabular, text, and image sequences.
    
    Example:
        dataset = CryptoMultimodalDataset(
            asset="BTC",
            split="train",
            seq_len=24,
            features_dir="/path/to/features"
        )
        sample = dataset[0]
        # sample["tabular"]: (24, 7)
        # sample["text_embedding"]: (24, 256)
        # sample["image_embedding"]: (24, 256)
        # sample["target"]: scalar
    """
    
    def __init__(
        self,
        asset: str = "MULTI",
        split: str = "train",
        seq_len: int = 24,
        hf_features_repo_id: str = None,
        features_dir: str = None,
        debug: bool = False,
    ):
        """
        Initialize offline dataset.
        
        Args:
            asset: "BTC", "ETH", or "MULTI" (BTC+ETH)
            split: "train", "validation", or "test_in_domain"
            seq_len: Sliding window length (hours)
            hf_features_repo_id: HF repo ID for pre-extracted embeddings (e.g., username/crypto-features)
            features_dir: Local directory containing embeddings (used if hf_features_repo_id not provided)
            debug: If True, load only 100 samples for testing
        
        Note: Either hf_features_repo_id or features_dir must be provided (HF takes precedence)
        """
        self.asset = asset
        self.split = split
        self.seq_len = seq_len
        self.hf_features_repo_id = hf_features_repo_id
        self.features_dir = Path(features_dir) if features_dir else None
        self.debug = debug
        
        # Validate: must have either HF repo or local directory
        if not hf_features_repo_id and not features_dir:
            raise ValueError(
                "Must provide either hf_features_repo_id or features_dir. "
                "Example: hf_features_repo_id='username/crypto-features'"
            )
        
        logger.info(
            f"Initializing CryptoMultimodalDataset\n"
            f"  Asset: {asset}, Split: {split}, Seq Len: {seq_len}"
        )
        print(f"[PROGRESS] Loading dataset ({asset}/{split})...")
        sys.stdout.flush()
        
        # ==================== LOAD DATASET METADATA ====================
        logger.info("Loading dataset metadata from HuggingFace...")
        print("[PROGRESS] Loading dataset metadata...")
        sys.stdout.flush()
        
        start_time = time.time()
        
        # Load both BTC and ETH, then concatenate
        try:
            btc_dataset = load_dataset(
                "khanh252004/multimodal_crypto_sentiment_btc",
                split=split,
                cache_dir="/tmp/huggingface_cache",
            )
            eth_dataset = load_dataset(
                "khanh252004/multimodal_crypto_sentiment_eth",
                split=split,
                cache_dir="/tmp/huggingface_cache",
            )
            
            self.dataset = concatenate_datasets([btc_dataset, eth_dataset])
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        if debug:
            self.dataset = self.dataset.select(range(min(100, len(self.dataset))))
        
        self.total_samples = len(self.dataset)
        dataset_time = time.time() - start_time
        logger.info(f"✓ Loaded {self.total_samples} samples in {format_duration(dataset_time)}")
        print(f"[PROGRESS] ✓ Dataset loaded ({self.total_samples} samples)")
        sys.stdout.flush()
        
        # ==================== LOAD PRE-EXTRACTED EMBEDDINGS ====================
        logger.info("Loading pre-extracted embeddings...")
        print("[PROGRESS] Loading pre-extracted embeddings...")
        sys.stdout.flush()
        
        embed_start = time.time()
        
        if self.hf_features_repo_id:
            # Load from Hugging Face
            logger.info(f"Downloading embeddings from {self.hf_features_repo_id}...")
            self._load_embeddings_from_hf(split)
        else:
            # Load from local disk
            logger.info(f"Loading embeddings from {self.features_dir}...")
            self._load_embeddings_from_disk(split)
        
        embed_time = time.time() - embed_start
        logger.info(f"✓ Embeddings loaded in {format_duration(embed_time)}")
        print(f"[PROGRESS] ✓ Embeddings loaded ({format_duration(embed_time)})")
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
        
        # ==================== FIT SCALER ON TABULAR FEATURES ====================
        logger.info("Fitting StandardScaler on tabular features...")
        print("[PROGRESS] Fitting StandardScaler on tabular features...")
        sys.stdout.flush()
        
        scaler_start = time.time()
        self._fit_tabular_scaler()
        scaler_time = time.time() - scaler_start
        logger.info(f"✓ StandardScaler fitted in {format_duration(scaler_time)}")
        print(f"[PROGRESS] ✓ StandardScaler fitted")
        sys.stdout.flush()
        
        logger.info(f"✓ Dataset ready: {len(self)} valid sequences of length {seq_len}")
    
    def _load_embeddings_from_disk(self, split: str) -> None:
        """Load embeddings from local disk."""
        text_embed_path = self.features_dir / f"text_embeddings_{split}.pt"
        image_embed_path = self.features_dir / f"image_embeddings_{split}.pt"
        
        if not text_embed_path.exists():
            raise FileNotFoundError(f"Text embeddings not found: {text_embed_path}")
        if not image_embed_path.exists():
            raise FileNotFoundError(f"Image embeddings not found: {image_embed_path}")
        
        logger.info(f"Loading text embeddings from {text_embed_path}...")
        self.text_embeddings = torch.load(text_embed_path, map_location="cpu")
        logger.info(f"✓ Text embeddings: {self.text_embeddings.shape}")
        
        logger.info(f"Loading image embeddings from {image_embed_path}...")
        self.image_embeddings = torch.load(image_embed_path, map_location="cpu")
        logger.info(f"✓ Image embeddings: {self.image_embeddings.shape}")
        
        # Validate shapes
        assert self.text_embeddings.shape[0] == self.total_samples, \
            f"Text embeddings mismatch: {self.text_embeddings.shape[0]} vs {self.total_samples}"
        assert self.image_embeddings.shape[0] == self.total_samples, \
            f"Image embeddings mismatch: {self.image_embeddings.shape[0]} vs {self.total_samples}"
    
    def _load_embeddings_from_hf(self, split: str) -> None:
        """Download and load embeddings from Hugging Face."""
        logger.info(f"Downloading from {self.hf_features_repo_id}...")
        
        try:
            # Download text embeddings
            text_filename = f"text_embeddings_{split}.pt"
            logger.info(f"Downloading {text_filename}...")
            text_path = hf_hub_download(
                repo_id=self.hf_features_repo_id,
                filename=text_filename,
                repo_type="dataset",
                cache_dir="~/.cache/huggingface/datasets"
            )
            self.text_embeddings = torch.load(text_path, map_location="cpu")
            logger.info(f"✓ Text embeddings: {self.text_embeddings.shape}")
            
            # Download image embeddings
            image_filename = f"image_embeddings_{split}.pt"
            logger.info(f"Downloading {image_filename}...")
            image_path = hf_hub_download(
                repo_id=self.hf_features_repo_id,
                filename=image_filename,
                repo_type="dataset",
                cache_dir="~/.cache/huggingface/datasets"
            )
            self.image_embeddings = torch.load(image_path, map_location="cpu")
            logger.info(f"✓ Image embeddings: {self.image_embeddings.shape}")
            
            # Validate shapes
            assert self.text_embeddings.shape[0] == self.total_samples, \
                f"Text embeddings mismatch: {self.text_embeddings.shape[0]} vs {self.total_samples}"
            assert self.image_embeddings.shape[0] == self.total_samples, \
                f"Image embeddings mismatch: {self.image_embeddings.shape[0]} vs {self.total_samples}"
        
        except Exception as e:
            logger.error(f"Failed to download from HF: {e}", exc_info=True)
            raise
    
        """Fit StandardScaler on all tabular features (with log1p on volume).
        
        CRITICAL: Fit only once during initialization to prevent data leakage.
        - Extracts 7 tabular features from entire dataset split
        - Applies np.log1p ONLY to volume column (index 1)
        - Fits StandardScaler to normalize all 7 columns (mean=0, std=1)
        """
        tabular_features_list = []
        
        # Iterate through entire dataset split to collect tabular values
        for idx in tqdm(range(self.total_samples), desc="Fitting scaler", unit="samples"):
            sample = self.dataset[idx]
            
            # Extract 7 tabular features
            tabular_values = np.array([
                sample["return_1h"],
                sample["volume"],
                sample["funding_rate"],
                sample["fear_greed_value"],
                sample["gdelt_econ_volume"],
                sample["gdelt_econ_tone"],
                sample["gdelt_conflict_volume"],
            ], dtype=np.float32)
            
            # Apply log1p ONLY to volume (index 1)
            tabular_values[1] = np.log1p(tabular_values[1])
            
            tabular_features_list.append(tabular_values)
        
        # Stack into (total_samples, 7) array and fit scaler
        tabular_array = np.stack(tabular_features_list, axis=0)  # (total_samples, 7)
        self.scaler = StandardScaler()
        self.scaler.fit(tabular_array)
        
        logger.info(
            f"StandardScaler fitted on {self.total_samples} samples. "
            f"Means: {self.scaler.mean_}, Stds: {self.scaler.scale_}"
        )
    
    def _scale_tabular_features(self, tabular_values: np.ndarray) -> torch.Tensor:
        """Scale tabular features using fitted StandardScaler.
        
        Args:
            tabular_values: (seq_len, 7) numpy array of raw tabular features
        
        Returns:
            (seq_len, 7) torch.FloatTensor with normalized features
        """
        # Apply log1p ONLY to volume column (index 1) for all timesteps
        tabular_values = tabular_values.copy()  # Don't modify original
        tabular_values[:, 1] = np.log1p(tabular_values[:, 1])  # log1p to volume
        
        # Apply StandardScaler transformation
        scaled = self.scaler.transform(tabular_values)  # (seq_len, 7)
        
        # Convert to torch tensor
        return torch.tensor(scaled, dtype=torch.float32)
    
    def __len__(self) -> int:
        """
        CRITICAL: Return safe length to prevent IndexError.
        
        Safe formula: total_samples - seq_len
        Example: 31,133 samples - 24 seq_len = 31,109 valid indices
        For idx=31,108 (last valid): can fetch target at idx+seq_len=31,132 ✓
        """
        return self.max_valid_idx
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample (sequence of length seq_len + target).
        
        CRITICAL: idx must be in [0, __len__()), which guarantees idx+seq_len exists.
        
        Args:
            idx: Index of sequence start
        
        Returns:
            Dict with keys:
                - tabular: (seq_len, 7) tensor
                - text_embedding: (seq_len, 256) tensor
                - image_embedding: (seq_len, 256) tensor
                - target: scalar tensor (value at idx + seq_len)
                - timestamp: scalar (epoch time at idx + seq_len)
        
        Raises:
            IndexError: If idx >= __len__()
        """
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {len(self)}"
            )
        
        # Collect sequence [idx, idx+seq_len-1]
        tabular_list = []  # Will collect raw values, then scale all at once
        
        for t in range(self.seq_len):
            sample_idx = idx + t
            sample = self.dataset[sample_idx]
            
            # Extract 7 tabular features
            tabular_values = np.array([
                sample["return_1h"],
                sample["volume"],
                sample["funding_rate"],
                sample["fear_greed_value"],
                sample["gdelt_econ_volume"],
                sample["gdelt_econ_tone"],
                sample["gdelt_conflict_volume"],
            ], dtype=np.float32)
            
            tabular_list.append(tabular_values)
        
        # Get target at idx + seq_len (guaranteed to exist)
        target_sample = self.dataset[idx + self.seq_len]
        target_score = target_sample["target_score"]
        target = torch.tensor(target_score, dtype=torch.float32)
        timestamp = torch.tensor(idx + self.seq_len, dtype=torch.long)
        
        # Stack and scale tabular features
        tabular_raw = np.stack(tabular_list, axis=0)  # (seq_len, 7)
        tabular_stacked = self._scale_tabular_features(tabular_raw)  # (seq_len, 7)
        
        # Slice text and image embeddings
        text_embedding_stacked = self.text_embeddings[idx:idx + self.seq_len]  # (seq_len, 256)
        image_embedding_stacked = self.image_embeddings[idx:idx + self.seq_len]  # (seq_len, 256)
        
        return {
            "tabular": tabular_stacked,
            "text_embedding": text_embedding_stacked,
            "image_embedding": image_embedding_stacked,
            "target": target,
            "timestamp": timestamp,
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
    hf_features_repo_id: str = None,
    features_dir: str = None,
    num_workers: int = 0,  # Always 0 on Kaggle (multi-worker deadlock fix)
    pin_memory: bool = True,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create DataLoaders for all splits with progress tracking.
    
    CRITICAL: DataLoader Optimization
    - num_workers=0: Forced on Kaggle (multi-worker deadlock issues)
    - pin_memory=True: Transfer data to GPU via pinned memory (faster)
    
    Args:
        config: ExperimentConfig instance
        splits: Tuple of (train_split, val_split, test_split)
        hf_features_repo_id: HF repo ID for embeddings (takes precedence)
        features_dir: Local directory with embeddings (fallback)
        num_workers: Number of data loading workers (0 for Kaggle safety)
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        Dict with keys "train", "validation", "test" and DataLoader values
    """
    dataloaders = {}
    
    overall_start = time.time()
    
    with tqdm(total=len(splits), desc="Creating DataLoaders", unit="split") as progress:
        for split_idx, split_name in enumerate(splits, 1):
            print(f"\n[PROGRESS] Creating DataLoader for {split_name}...")
            sys.stdout.flush()
            
            split_start = time.time()
            
            # Create dataset
            dataset = CryptoMultimodalDataset(
                asset=config.data.asset,
                split=split_name,
                seq_len=config.data.seq_len,
                hf_features_repo_id=hf_features_repo_id,
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
    print("Testing CryptoMultimodalDataset")
    print("=" * 80)
    
    # Test dataset initialization
    print("\n1. Loading dataset...")
    try:
        dataset = CryptoMultimodalDataset(
            asset=config.data.asset,
            split="train",
            seq_len=config.data.seq_len,
            debug=True,
        )
        print(f"   ✓ Dataset loaded")
        print(f"   Dataset length: {len(dataset)}")
        print(f"   Total samples in split: {dataset.total_samples}")
        print(f"   Max valid index: {dataset.max_valid_idx}")
    except FileNotFoundError as e:
        print(f"   ⚠ Skipping dataset test (embeddings not found): {e}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✅ Dataset module ready!")
    print("=" * 80)
