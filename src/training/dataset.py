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
from .utils import format_duration

logger = logging.getLogger(__name__)


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
        scaler: Optional['StandardScaler'] = None,
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
            scaler: Pre-fitted StandardScaler (prevents data leakage). If provided, this dataset will NOT fit a new scaler.
                   CRITICAL: Only fit on training split, then pass to val/test!
        
        Note: Either hf_features_repo_id or features_dir must be provided (HF takes precedence)
        """
        self.asset = asset
        self.split = split
        self.seq_len = seq_len
        self.hf_features_repo_id = hf_features_repo_id
        self.features_dir = Path(features_dir) if features_dir else None
        self.debug = debug
        self.scaler = scaler  # Pre-fitted scaler (prevents data leakage)
        
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
        
        # ==================== EXTRACT & PRE-SCALE ALL TABULAR FEATURES ====================
        logger.info("Extracting and pre-scaling all tabular features...")
        print("[PROGRESS] Extracting and pre-scaling all tabular features...")
        sys.stdout.flush()
        
        tabular_start = time.time()
        if self.scaler is None:
            # TRAINING SPLIT: Fit a new scaler
            logger.info(f"[FIT MODE] Fitting scaler on {self.split} split (data leakage prevention)")
            self._extract_fit_and_scale_all_tabular(fit_scaler=True)
        else:
            # VALIDATION/TEST SPLITS: Use pre-fitted scaler
            logger.info(f"[TRANSFORM MODE] Using pre-fitted scaler on {self.split} split (no fitting)")
            self._extract_fit_and_scale_all_tabular(fit_scaler=False)
        tabular_time = time.time() - tabular_start
        logger.info(f"✓ All tabular features pre-scaled in {format_duration(tabular_time)}")
        print(f"[PROGRESS] ✓ All tabular features pre-scaled")
        sys.stdout.flush()
        
        logger.info(f"✓ Dataset ready: {len(self)} valid sequences of length {seq_len}")
    
    def _load_embeddings_from_disk(self, split: str) -> None:
        """Load embeddings from local disk as contiguous tensors."""
        text_embed_path = self.features_dir / f"text_embeddings_{split}.pt"
        image_embed_path = self.features_dir / f"image_embeddings_{split}.pt"
        
        if not text_embed_path.exists():
            raise FileNotFoundError(f"Text embeddings not found: {text_embed_path}")
        if not image_embed_path.exists():
            raise FileNotFoundError(f"Image embeddings not found: {image_embed_path}")
        
        logger.info(f"Loading text embeddings from {text_embed_path}...")
        text_raw = torch.load(text_embed_path, map_location="cpu")
        self.text_embeddings = text_raw.contiguous()  # Ensure contiguity for zero-copy slicing
        logger.info(f"✓ Text embeddings: {self.text_embeddings.shape}, contiguous={self.text_embeddings.is_contiguous()}")
        
        logger.info(f"Loading image embeddings from {image_embed_path}...")
        image_raw = torch.load(image_embed_path, map_location="cpu")
        self.image_embeddings = image_raw.contiguous()  # Ensure contiguity for zero-copy slicing
        logger.info(f"✓ Image embeddings: {self.image_embeddings.shape}, contiguous={self.image_embeddings.is_contiguous()}")
        
        # Validate shapes
        assert self.text_embeddings.shape[0] == self.total_samples, \
            f"Text embeddings mismatch: {self.text_embeddings.shape[0]} vs {self.total_samples}"
        assert self.image_embeddings.shape[0] == self.total_samples, \
            f"Image embeddings mismatch: {self.image_embeddings.shape[0]} vs {self.total_samples}"
    
    def _load_embeddings_from_hf(self, split: str) -> None:
        """Download and load embeddings from Hugging Face as contiguous tensors."""
        import os
        
        # Disable symlinks for Kaggle compatibility
        os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
        
        logger.info(f"Downloading from {self.hf_features_repo_id}...")
        
        try:
            # Download text embeddings
            text_filename = f"text_embeddings_{split}.pt"
            logger.info(f"Downloading {text_filename}...")
            text_path = hf_hub_download(
                repo_id=self.hf_features_repo_id,
                filename=text_filename,
                repo_type="dataset",
                cache_dir=os.path.expanduser("~/.cache/huggingface/datasets")
            )
            text_raw = torch.load(text_path, map_location="cpu")
            self.text_embeddings = text_raw.contiguous()  # Ensure contiguity
            logger.info(f"✓ Text embeddings: {self.text_embeddings.shape}, contiguous={self.text_embeddings.is_contiguous()}")
            
            # Download image embeddings
            image_filename = f"image_embeddings_{split}.pt"
            logger.info(f"Downloading {image_filename}...")
            image_path = hf_hub_download(
                repo_id=self.hf_features_repo_id,
                filename=image_filename,
                repo_type="dataset",
                cache_dir=os.path.expanduser("~/.cache/huggingface/datasets")
            )
            image_raw = torch.load(image_path, map_location="cpu")
            self.image_embeddings = image_raw.contiguous()  # Ensure contiguity
            logger.info(f"✓ Image embeddings: {self.image_embeddings.shape}, contiguous={self.image_embeddings.is_contiguous()}")
            
            # Validate shapes
            assert self.text_embeddings.shape[0] == self.total_samples, \
                f"Text embeddings mismatch: {self.text_embeddings.shape[0]} vs {self.total_samples}"
            assert self.image_embeddings.shape[0] == self.total_samples, \
                f"Image embeddings mismatch: {self.image_embeddings.shape[0]} vs {self.total_samples}"
        
        except Exception as e:
            logger.error(f"Failed to download from HF: {e}", exc_info=True)
            raise
    
    def _extract_fit_and_scale_all_tabular(self, fit_scaler: bool = True):
        """OPTIMIZED: Extract and scale ALL tabular features once in __init__.
        
        CRITICAL DATA LEAKAGE PREVENTION:
        - If fit_scaler=True (training split): Fit StandardScaler, then transform
        - If fit_scaler=False (val/test splits): Use pre-fitted scaler, only transform
        
        This prevents the validation/test sets from being scaled with different
        means/stds than the training set (which causes model to see "alien language").
        
        CRITICAL PERFORMANCE:
        - Called ONCE in __init__, never in __getitem__
        - Iterates through entire dataset split once to collect all 7 tabular features
        - Creates (total_samples, 7) numpy array
        - Applies np.log1p ONLY to volume column (index 1)
        - Fits OR uses pre-fitted StandardScaler (depending on fit_scaler flag)
        - Transforms all at once (vectorized)
        - Stores as contiguous torch.float32 tensor self.tabular_data
        
        Result: __getitem__ only slices self.tabular_data[idx:idx+seq_len] - O(1) per batch.
        """
        logger.info("Extracting all tabular features from dataset...")
        print("[PROGRESS]   Extracting tabular features...", end="", flush=True)
        
        tabular_features_list = []
        target_scores_list = []
        
        # SINGLE PASS: Collect all features once
        for idx in tqdm(range(self.total_samples), desc="Extracting", unit="samples", leave=False):
            sample = self.dataset[idx]
            
            # Extract 7 tabular features as float32
            tabular_values = np.array([
                sample["return_1h"],
                sample["volume"],
                sample["funding_rate"],
                sample["fear_greed_value"],
                sample["gdelt_econ_volume"],
                sample["gdelt_econ_tone"],
                sample["gdelt_conflict_volume"],
            ], dtype=np.float32)
            
            tabular_features_list.append(tabular_values)
            target_scores_list.append(sample["target_score"])
        
        print(" Done!", flush=True)
        
        # Stack into (total_samples, 7) numpy array
        tabular_array = np.stack(tabular_features_list, axis=0)  # Shape: (total_samples, 7)
        logger.info(f"Stacked tabular array shape: {tabular_array.shape}")
        
        # Apply log1p ONLY to volume column (index 1)
        print("[PROGRESS]   Applying log1p transformation...", flush=True)
        tabular_array[:, 1] = np.log1p(tabular_array[:, 1])
        
        # FIT or USE pre-fitted scaler
        if fit_scaler:
            # TRAINING SPLIT: Fit a new scaler
            print("[PROGRESS]   Fitting StandardScaler on training data...", flush=True)
            self.scaler = StandardScaler()
            self.scaler.fit(tabular_array)
            logger.info(
                f"StandardScaler FITTED on training data. Means: {self.scaler.mean_}, Stds: {self.scaler.scale_}"
            )
        else:
            # VALIDATION/TEST SPLIT: Use pre-fitted training scaler (NO FITTING)
            print(f"[PROGRESS]   Using pre-fitted scaler (from training split)...", flush=True)
            logger.info(
                f"StandardScaler already fitted. Using training means: {self.scaler.mean_}, stds: {self.scaler.scale_}"
            )
        
        # Transform ALL at once (vectorized operation: O(n) not O(n*seq_len))
        print("[PROGRESS]   Transforming features...", flush=True)
        scaled_array = self.scaler.transform(tabular_array)  # Shape: (total_samples, 7)
        
        # Convert to torch tensor (float32, contiguous for zero-copy slicing)
        self.tabular_data = torch.tensor(scaled_array, dtype=torch.float32).contiguous()
        logger.info(f"✓ Tabular tensor stored: {self.tabular_data.shape}, dtype={self.tabular_data.dtype}, contiguous={self.tabular_data.is_contiguous()}")
        
        # Store target scores and timestamps as tensors
        self.target_scores = torch.tensor(target_scores_list, dtype=torch.float32)
        self.timestamps = torch.arange(self.total_samples, dtype=torch.long)
        logger.info(f"✓ Target scores tensor: {self.target_scores.shape}")
        logger.info(f"✓ Timestamps tensor: {self.timestamps.shape}")

    
    def __len__(self) -> int:
        """
        CRITICAL: Return safe length to prevent IndexError.
        
        Safe formula: total_samples - seq_len
        Example: 31,133 samples - 24 seq_len = 31,109 valid indices
        For idx=31,108 (last valid): can fetch target at idx+seq_len=31,132 ✓
        """
        return self.max_valid_idx
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """OPTIMIZED: Zero-copy slicing - O(1) operation per index.
        
        CRITICAL: This method does NOTHING except slice pre-computed tensors.
        - NO dataset iteration
        - NO feature extraction  
        - NO tensor conversions (torch.tensor() removed)
        - NO transformations/scalings
        - Pure zero-copy view slicing from pre-allocated tensors
        
        Args:
            idx: Index of sequence start (must be in [0, __len__()))
        
        Returns:
            Dict with VIEWS (not copies) into pre-computed tensors:
                - tabular: (seq_len, 7) float32 view - ALREADY SCALED
                - text_embedding: (seq_len, 256) float32 view
                - image_embedding: (seq_len, 256) float32 view
                - target: scalar float32 tensor
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
    hf_features_repo_id: str = None,
    features_dir: str = None,
    num_workers: int = 0,  # Always 0 on Kaggle (multi-worker deadlock fix)
    pin_memory: bool = True,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create DataLoaders for all splits with progress tracking.
    
    CRITICAL DATA LEAKAGE PREVENTION:
    - Fit scaler ONLY on training split
    - Pass fitted scaler to validation/test splits
    - Ensures all splits use same scaling parameters (training's means/stds)
    
    CRITICAL: DataLoader Optimization
    - num_workers=0: FORCED on Kaggle (multi-worker deadlock issues)
      Even if config.data.num_workers is set to 4, we override to 0
    - pin_memory=True: Transfer data to GPU via pinned memory (faster)
    
    Args:
        config: ExperimentConfig instance
        splits: Tuple of (train_split, val_split, test_split)
        hf_features_repo_id: HF repo ID for embeddings (takes precedence)
        features_dir: Local directory with embeddings (fallback)
        num_workers: Number of data loading workers (IGNORED - always 0 for Kaggle safety)
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        Dict with keys "train", "validation", "test" and DataLoader values
    """
    # CRITICAL: Force num_workers=0 regardless of parameter or config
    num_workers = 0
    
    dataloaders = {}
    train_scaler = None  # Will be set after training dataset is created
    
    overall_start = time.time()
    
    with tqdm(total=len(splits), desc="Creating DataLoaders", unit="split") as progress:
        for split_idx, split_name in enumerate(splits, 1):
            print(f"\n[PROGRESS] Creating DataLoader for {split_name} (num_workers=0, pin_memory={pin_memory})...")
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
                scaler=train_scaler,  # Pass training scaler to val/test (prevents leakage)
            )
            
            # After training dataset is created, save its scaler for val/test
            if split_name == "train":
                train_scaler = dataset.scaler  # Extract fitted scaler from training dataset
                logger.info(f"✓ Training scaler extracted for use on validation/test splits")
            
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
