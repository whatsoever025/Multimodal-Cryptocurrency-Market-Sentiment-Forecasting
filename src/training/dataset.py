"""
Safe multimodal dataset with sliding window preprocessing.

CRITICAL: Safe sliding window logic prevents IndexError at dataset boundaries.
- __len__() returns total_samples - seq_len (to guarantee idx + seq_len exists)
- __getitem__(idx) fetches context [idx, idx+seq_len-1], target at idx+seq_len
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
from PIL import Image
import logging
import sys
import time
from tqdm import tqdm

try:
    from datasets import load_dataset, concatenate_datasets
except ImportError:
    raise ImportError("'datasets' package required: pip install datasets")

try:
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("'transformers' package required: pip install transformers")


logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


class CryptoMultimodalDataset(Dataset):
    """
    Multimodal dataset for cryptocurrency sentiment forecasting.
    
    Safe sliding window implementation:
    - Input: hourly samples with tabular, text, images, targets
    - Output: sequences of length seq_len with target at idx + seq_len
    
    Example:
        dataset = CryptoMultimodalDataset(asset="BTC", split="train", seq_len=24)
        tabular, text_ids, text_mask, images, target, timestamp = dataset[0]
        # tabular: (24, 7), text_ids: (24, 512), images: (24, 3, 224, 224), target: scalar
    """
    
    def __init__(
        self,
        asset: str = "BTC",
        split: str = "train",
        seq_len: int = 24,
        max_text_length: int = 512,
        image_size: int = 224,
        cache_images: bool = False,
        cache_text: bool = True,
        device: str = "cpu",
        debug: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            asset: "BTC" or "ETH"
            split: "train", "validation", or "test_in_domain"
            seq_len: Sliding window length (hours)
            max_text_length: Token sequence length for BERT
            image_size: ResNet50 input size (e.g., 224)
            cache_images: If True, load all images into memory (careful with VRAM)
            cache_text: If True, tokenize all text at init (recommended)
            device: "cpu" or "cuda"
            debug: If True, load only 100 samples for testing
        """
        self.asset = asset
        self.split = split
        self.seq_len = seq_len
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.cache_images = cache_images
        self.cache_text = cache_text
        self.device = device
        self.debug = debug
        
        logger.info(f"Loading multi-asset dataset ({split} split)...")
        
        # Load both BTC and ETH, then concatenate
        try:
            start_time = time.time()
            
            print(f"[PROGRESS] Downloading BTC dataset ({split})...")
            btc_start = time.time()
            with tqdm(desc=f"BTC {split}", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                btc_dataset = load_dataset("khanh252004/multimodal_crypto_sentiment_btc", split=split)
                pbar.update(1)  # Mark as complete
            btc_time = time.time() - btc_start
            
            print(f"[PROGRESS] Downloading ETH dataset ({split})...")
            eth_start = time.time()
            with tqdm(desc=f"ETH {split}", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                eth_dataset = load_dataset("khanh252004/multimodal_crypto_sentiment_eth", split=split)
                pbar.update(1)  # Mark as complete
            eth_time = time.time() - eth_start
            
            total_time = time.time() - start_time
            
            self.dataset = concatenate_datasets([btc_dataset, eth_dataset])
            
            btc_samples = len(btc_dataset)
            eth_samples = len(eth_dataset)
            total_samples = len(self.dataset)
            
            logger.info(f"✓ Dataset downloaded successfully!")
            logger.info(f"  BTC: {btc_samples} samples in {format_duration(btc_time)}")
            logger.info(f"  ETH: {eth_samples} samples in {format_duration(eth_time)}")
            logger.info(f"  Total: {total_samples} samples in {format_duration(total_time)}")
            
            print(f"[PROGRESS] ✓ BTC: {btc_samples} samples ({format_duration(btc_time)})")
            print(f"[PROGRESS] ✓ ETH: {eth_samples} samples ({format_duration(eth_time)})")
            print(f"[PROGRESS] ✓ Total: {total_samples} samples ({format_duration(total_time)})")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Failed to load multi-asset dataset: {e}")
            raise
        
        # For debugging, subsample
        if debug:
            self.dataset = self.dataset.select(range(min(100, len(self.dataset))))
        
        total_samples = len(self.dataset)
        logger.info(f"Loaded {total_samples} samples for {asset}/{split}")
        
        # Quick validation: test dataset accessibility
        try:
            print("[PROGRESS] Validating dataset accessibility...")
            sys.stdout.flush()
            _ = self.dataset[0]  # Try accessing first sample
            logger.info("✓ Dataset is accessible")
            print("[PROGRESS] ✓ Dataset accessible")
            sys.stdout.flush()
        except Exception as e:
            logger.warning(f"Dataset validation warning: {e}")
            print(f"[WARNING] Dataset validation issue: {e}")
            sys.stdout.flush()
        
        # CRITICAL: Safe sliding window math
        # max_idx = total_samples - seq_len
        # For idx in [0, max_idx), we can fetch target at idx + seq_len
        self.total_samples = total_samples
        self.max_valid_idx = total_samples - seq_len
        
        if self.max_valid_idx <= 0:
            raise ValueError(
                f"Dataset too small for seq_len={seq_len}. "
                f"Need at least {seq_len + 1} samples, got {total_samples}"
            )
        
        logger.info(
            f"Safe sliding window: dataset__len__() will return {self.max_valid_idx} "
            f"(indices 0 to {self.max_valid_idx - 1})"
        )
        
        # Initialize FinBERT tokenizer
        logger.info("Loading FinBERT tokenizer (first time may take 30-60s)...")
        print("[PROGRESS] Downloading FinBERT tokenizer from HuggingFace...")
        sys.stdout.flush()
        
        tokenizer_start = time.time()
        with tqdm(desc="FinBERT tokenizer", total=100, unit="%") as pbar:
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            pbar.update(100)
        tokenizer_time = time.time() - tokenizer_start
        
        logger.info(f"✓ FinBERT tokenizer loaded successfully in {format_duration(tokenizer_time)}")
        print(f"[PROGRESS] ✓ FinBERT tokenizer ready ({format_duration(tokenizer_time)})")
        sys.stdout.flush()
        
        # Pre-cache tokenized text if requested
        # NOTE: Disabled pre-tokenization (62k samples = 5-15min silent processing)
        # Instead, use lazy caching during data loading (much better UX)
        if cache_text:
            logger.info("Text caching enabled (lazy per-sample during data loading)")
        self._text_cache = {}  # Always initialize as empty dict
        
        # Pre-cache images if requested
        if cache_images:
            logger.warning("Caching images in memory (may use significant VRAM)")
            logger.info(f"Pre-loading all {total_samples} images...")
            self._image_cache = {}
            self._load_all_images()
        else:
            self._image_cache = {}
        
        logger.info(f"✓ Dataset ready: {len(self)} valid sequences of length {seq_len}")
    
    def _tokenize_all(self) -> None:
        """Pre-tokenize all text samples for faster training."""
        self._text_cache = {}
        for idx in range(self.total_samples):
            text = self.dataset[idx]["text_content"]
            encoded = self.tokenizer(
                text,
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self._text_cache[idx] = {
                "input_ids": encoded["input_ids"].squeeze(0),  # (max_text_length,)
                "attention_mask": encoded["attention_mask"].squeeze(0),  # (max_text_length,)
            }
    
    def _load_all_images(self) -> None:
        """Pre-load all images into memory - ALWAYS resized to (image_size, image_size)."""
        for idx in range(self.total_samples):
            image_path = self.dataset[idx]["image_path"]
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path.convert("RGB")  # Already PIL Image from HF dataset
            
            # Resize to fixed size
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
            
            # Convert to tensor and normalize
            image_tensor = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)
            image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
            self._image_cache[idx] = image_tensor
    
    def _get_text(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get tokenized text (input_ids, attention_mask)."""
        if idx in self._text_cache:
            item = self._text_cache[idx]
            return item["input_ids"], item["attention_mask"]
        
        # Tokenize on-the-fly
        text = self.dataset[idx]["text_content"]
        encoded = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Cache if enabled
        if self.cache_text:
            self._text_cache[idx] = {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }
        
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)
    
    def _get_image(self, idx: int) -> torch.Tensor:
        """Get image tensor - ALWAYS (3, image_size, image_size)."""
        if self.cache_images and idx in self._image_cache:
            return self._image_cache[idx]
        
        # Load on-the-fly
        image_path = self.dataset[idx]["image_path"]
        
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")
        
        # Resize to fixed size
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Convert to tensor
        np_arr = np.array(image)
        image_tensor = torch.tensor(np_arr, dtype=torch.float32).permute(2, 0, 1)
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
        
        if self.cache_images:
            self._image_cache[idx] = image_tensor
        
        return image_tensor
    
    def __len__(self) -> int:
        """
        CRITICAL: Return safe length to prevent IndexError.
        
        Safe formula: total_samples - seq_len
        Example: 31,133 samples - 24 seq_len = 31,109 valid indices
        For idx=31,108 (last valid): can fetch target at idx+seq_len=31,132 ✓
        """
        return self.max_valid_idx
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample (sequence of length seq_len + target).
        
        CRITICAL: idx must be in [0, __len__()), which guarantees idx+seq_len exists.
        
        Args:
            idx: Index of sequence start
        
        Returns:
            Dict with keys:
                - tabular: (seq_len, 7) tensor
                - text_ids: (seq_len, max_text_length) tensor
                - text_mask: (seq_len, max_text_length) tensor
                - images: (seq_len, 3, 224, 224) tensor
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
        tabular_list = []
        text_ids_list = []
        text_mask_list = []
        images_list = []
        
        for t in range(self.seq_len):
            sample_idx = idx + t
            sample = self.dataset[sample_idx]
            
            # Tabular: 7 features
            tabular_values = [
                sample["return_1h"],
                sample["volume"],
                sample["funding_rate"],
                sample["fear_greed_value"],
                sample["gdelt_econ_volume"],
                sample["gdelt_econ_tone"],
                sample["gdelt_conflict_volume"],
            ]
            
            tabular = torch.tensor(tabular_values, dtype=torch.float32)
            tabular_list.append(tabular)
            
            # Text
            text_ids, text_mask = self._get_text(sample_idx)
            text_ids_list.append(text_ids)
            text_mask_list.append(text_mask)
            
            # Images
            image = self._get_image(sample_idx)
            images_list.append(image)
        
        # Get target at idx + seq_len (guaranteed to exist)
        target_sample = self.dataset[idx + self.seq_len]
        target_score = target_sample["target_score"]
        target = torch.tensor(target_score, dtype=torch.float32)
        timestamp = torch.tensor(idx + self.seq_len, dtype=torch.long)
        
        # Stack all sequences
        tabular_stacked = torch.stack(tabular_list) 
        text_ids_stacked = torch.stack(text_ids_list) 
        text_mask_stacked = torch.stack(text_mask_list)
        images_stacked = torch.stack(images_list)
        
        return {
            "tabular": tabular_stacked,  # (seq_len, 7)
            "text_ids": text_ids_stacked,  # (seq_len, max_text_length)
            "text_mask": text_mask_stacked,  # (seq_len, max_text_length)
            "images": images_stacked,  # (seq_len, 3, 224, 224)
            "target": target,  # scalar
            "timestamp": timestamp,  # scalar (for debugging)
        }


def multimodal_collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Collate function for multimodal batches.
    
    Args:
        batch: List of dicts from CryptoMultimodalDataset
    
    Returns:
        Dict with stacked tensors:
            - tabular: (batch_size, seq_len, 7)
            - text_ids: (batch_size, seq_len, max_text_length)
            - text_mask: (batch_size, seq_len, max_text_length)
            - images: (batch_size, seq_len, 3, 224, 224)
            - targets: (batch_size,)
            - timestamps: (batch_size,)
    """
    stacked = {
        "tabular": torch.stack([sample["tabular"] for sample in batch]),  # (B, seq_len, 7)
        "text_ids": torch.stack([sample["text_ids"] for sample in batch]),  # (B, seq_len, max_len)
        "text_mask": torch.stack([sample["text_mask"] for sample in batch]),  # (B, seq_len, max_len)
        "images": torch.stack([sample["images"] for sample in batch]),  # (B, seq_len, 3, H, W)
        "target": torch.stack([sample["target"] for sample in batch]),  # (B,)
        "timestamp": torch.stack([sample["timestamp"] for sample in batch]),  # (B,)
    }
    
    return stacked


def create_dataloaders(
    config,
    splits: Optional[Tuple[str, str, str]] = ("train", "validation", "test_in_domain"),
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for all splits with progress tracking.
    
    Args:
        config: ExperimentConfig instance
        splits: Tuple of (train_split, val_split, test_split)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        Dict with keys "train", "validation", "test" and DataLoader values
    """
    dataloaders = {}
    
    overall_start = time.time()
    
    with tqdm(total=len(splits), desc="Creating DataLoaders", unit="split") as progress:
        for split_idx, split_name in enumerate(splits, 1):
            print(f"\n[PROGRESS] ({split_idx}/{len(splits)}) Loading {split_name} dataset...")
            sys.stdout.flush()
            
            split_start = time.time()
            dataset = CryptoMultimodalDataset(
                asset=config.data.asset,
                split=split_name,
                seq_len=config.data.seq_len,
                max_text_length=config.data.max_text_length,
                image_size=config.data.image_size,
                cache_images=False,
                cache_text=True,
                debug=config.debug,
            )
            split_time = time.time() - split_start
            
            print(f"[PROGRESS] ✓ {split_name} dataset loaded ({format_duration(split_time)}), creating DataLoader...")
            sys.stdout.flush()
            
            shuffle = (split_name == "train") and config.data.shuffle_train
            batch_size = config.data.batch_size if split_name == "train" else config.inference.inference_batch_size
            workers = num_workers if split_name == "train" else 0
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=multimodal_collate_fn,
                num_workers=workers,
                pin_memory=pin_memory,
                prefetch_factor=config.data.prefetch_factor if workers > 0 else None,
                persistent_workers=False if workers == 0 else True,
            )
            print(f"[PROGRESS] ✓ {split_name} DataLoader ready: {len(dataset)} sequences, {len(loader)} batches")
            sys.stdout.flush()
            
            dataloaders[split_name] = loader
            logger.info(f"Created DataLoader for {split_name}: {len(loader)} batches ({format_duration(split_time)})")
            
            # Update progress bar
            progress.update(1)
            progress.set_postfix({"last_split": split_name, "sequences": len(dataset)})
    
    total_time = time.time() - overall_start
    print(f"\n[PROGRESS] ✓ All dataloaders created! Total time: {format_duration(total_time)}")
    sys.stdout.flush()
    
    return dataloaders


if __name__ == "__main__":
    """Test dataset loading and safe sliding window."""
    import sys
    
    # Test configuration
    from config import ExperimentConfig
    
    config = ExperimentConfig(debug=True)
    
    print("=" * 80)
    print("Testing CryptoMultimodalDataset")
    print("=" * 80)
    
    # Test dataset initialization
    print("\n1. Loading dataset...")
    dataset = CryptoMultimodalDataset(
        asset=config.data.asset,
        split="train",
        seq_len=config.data.seq_len,
        debug=True,
    )
    print(f"   Dataset length: {len(dataset)}")
    print(f"   Total samples in split: {dataset.total_samples}")
    print(f"   Max valid index: {dataset.max_valid_idx}")
    
    # Test boundary conditions
    print("\n2. Testing boundary conditions...")
    try:
        # Valid index
        sample = dataset[0]
        print(f"   ✓ dataset[0] succeeded")
        print(f"     - tabular shape: {sample['tabular'].shape}")
        print(f"     - text_ids shape: {sample['text_ids'].shape}")
        print(f"     - images shape: {sample['images'].shape}")
        print(f"     - target shape: {sample['target'].shape}")
        
        # Last valid index
        last_idx = len(dataset) - 1
        sample = dataset[last_idx]
        print(f"   ✓ dataset[{last_idx}] (last valid) succeeded")
        
        # Out of bounds should raise IndexError
        try:
            sample = dataset[len(dataset)]
            print(f"   ✗ dataset[{len(dataset)}] should have raised IndexError")
            sys.exit(1)
        except IndexError:
            print(f"   ✓ dataset[{len(dataset)}] correctly raised IndexError")
    
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test DataLoader with collate function
    print("\n3. Testing DataLoader with collate function...")
    try:
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=multimodal_collate_fn,
            num_workers=0,
        )
        
        batch = next(iter(loader))
        print(f"   ✓ Batch created successfully")
        print(f"     - tabular shape: {batch['tabular'].shape} (expected: (4, 24, 7))")
        print(f"     - text_ids shape: {batch['text_ids'].shape} (expected: (4, 24, 512))")
        print(f"     - images shape: {batch['images'].shape} (expected: (4, 24, 3, 224, 224))")
        print(f"     - target shape: {batch['target'].shape} (expected: (4,))")
    
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✅ All dataset tests passed!")
    print("=" * 80)
