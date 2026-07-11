"""
Offline feature extraction pipeline for FinBERT text and ViT image embeddings.

Extracts and caches embeddings from frozen backbones once per asset, so the
training loop reads pre-computed tensors instead of running the encoders per batch.
Run once on Kaggle before training:
    python src/training/extract_features.py --asset MULTI --output_dir /kaggle/working/features
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import argparse
from pathlib import Path
from PIL import Image
from typing import Dict, Tuple
import time

try:
    import ta
except ImportError:
    raise ImportError("'ta' package required: pip install ta")

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    raise ImportError("'transformers' package required: pip install transformers")

try:
    import torchvision.models as models
except ImportError:
    raise ImportError("'torchvision' package required: pip install torchvision")

try:
    from datasets import load_dataset, concatenate_datasets
except ImportError:
    raise ImportError("'datasets' package required: pip install datasets")

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    raise ImportError("'huggingface_hub' required: pip install huggingface_hub")

# Scalers removed - will be applied during training in Kaggle, not during extraction

from tqdm import tqdm
from .utils import setup_logging, format_duration


logger = logging.getLogger(__name__)


class FrozenTextEncoder(nn.Module):
    """
    Frozen FinBERT encoder with a trainable 768→256 projection head.

    Input:  (batch, seq_len) token IDs
    Output: (batch, 256) projected [CLS] embeddings
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        logger.info("Loading FinBERT...")
        self.bert = AutoModel.from_pretrained("ProsusAI/finbert")
        
        # Freeze backbone completely (no gradients needed)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Project [CLS] token (768) to hidden_dim
        self.projection = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        nn.init.xavier_uniform_(self.projection.weight)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch, max_text_length)
            attention_mask: (batch, max_text_length)
        Returns:
            (batch, hidden_dim)
        """
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            cls_token = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        
        projected = self.projection(cls_token)  # (batch, hidden_dim)
        return self.dropout(projected)


IMAGE_BACKBONE_CHECKPOINTS = {
    "vit": "google/vit-base-patch16-224",
    "clip": "openai/clip-vit-base-patch16",
}


class FrozenImageEncoder(nn.Module):
    """
    Frozen image encoder with a trainable 768→256 projection head.

    backbone="vit"  (default): google/vit-base-patch16-224, ImageNet-pretrained, CLS token
                     taken from last_hidden_state (position 0).
    backbone="clip": openai/clip-vit-base-patch16 vision tower only (no text tower), using
                     CLIP's own pooler_output as the pooled embedding. Both backbones share
                     the same 768-dim output width, so the projection head is unchanged;
                     this is a diagnostic swap to test whether CLIP's image-text contrastive
                     pretraining transfers better to candlestick charts than ImageNet
                     classification pretraining (thesis Section 5.3, "Future Work").

    Input:  (batch, 3, 224, 224) normalised RGB images
    Output: (batch, 256) projected pooled embeddings
    """

    def __init__(self, hidden_dim: int = 256, backbone: str = "vit"):
        super().__init__()
        self.hidden_dim = hidden_dim
        if backbone not in IMAGE_BACKBONE_CHECKPOINTS:
            raise ValueError(
                f"Unknown image backbone {backbone!r}; choices: {list(IMAGE_BACKBONE_CHECKPOINTS)}"
            )
        self.backbone_name = backbone
        checkpoint = IMAGE_BACKBONE_CHECKPOINTS[backbone]

        if backbone == "clip":
            from transformers import CLIPVisionModel
            logger.info(f"Loading CLIP vision tower ({checkpoint})...")
            vision_model = CLIPVisionModel.from_pretrained(checkpoint)
        else:
            logger.info(f"Loading Vision Transformer ({checkpoint})...")
            vision_model = AutoModel.from_pretrained(checkpoint)

        # Freeze backbone
        for param in vision_model.parameters():
            param.requires_grad = False

        self.backbone = vision_model

        # Both checkpoints output a 768-dim pooled embedding
        vit_hidden_size = 768
        self.projection = nn.Linear(vit_hidden_size, hidden_dim)
        nn.init.xavier_uniform_(self.projection.weight)
        self.dropout = nn.Dropout(0.2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch, 3, 224, 224)
        Returns:
            (batch, hidden_dim)
        """
        with torch.no_grad():
            outputs = self.backbone(images, return_dict=True)
            if self.backbone_name == "clip":
                features = outputs.pooler_output  # (batch, 768), CLIP's own pooled output
            else:
                features = outputs.last_hidden_state[:, 0, :]  # (batch, 768), ViT [CLS]

        projected = self.projection(features)  # (batch, hidden_dim)
        return self.dropout(projected)


def load_asset_dataset(asset: str, debug: bool = False):
    """
    Download and return the full chronological HuggingFace dataset for one asset.

    Concatenates the train and test splits so that walk-forward splits can be
    applied downstream without any pre-existing chronological boundary.

    Args:
        asset: "BTC" or "ETH".
        debug: If True, truncate to 100 samples for quick testing.

    Returns:
        HuggingFace Dataset (full sequence, no split).
    """
    logger.info(f"Loading {asset} v5 dataset splits (train + test)...")

    # Use /kaggle/working for cache on Kaggle (/tmp is small), else /tmp
    _cache = "/kaggle/working/hf_cache" if Path("/kaggle/working").exists() else "/tmp/huggingface_cache"
    repo_name = f"khanh252004/multimodal_crypto_sentiment_{asset.lower()}"

    print(f"[PROGRESS] Downloading {asset} dataset from Hugging Face ({repo_name})...")
    ds_dict = load_dataset(repo_name, cache_dir=_cache)

    # Concatenate train and test splits to preserve full chronological sequence
    dataset = concatenate_datasets([ds_dict["train"], ds_dict["test"]])

    if debug:
        dataset = dataset.select(range(min(100, len(dataset))))

    logger.info(f"Loaded {len(dataset)} samples for {asset} (train={len(ds_dict['train'])}, test={len(ds_dict['test'])})")
    return dataset



def extract_text_embeddings(
    dataset,
    encoder: FrozenTextEncoder,
    output_path: Path,
    batch_size: int = 32,
    max_text_length: int = 512,
    device: str = "cuda",
) -> None:
    """
    Extract FinBERT [CLS] embeddings for all samples and save to disk.

    Args:
        dataset:         HuggingFace Dataset with a "text_content" column.
        encoder:         FrozenTextEncoder instance.
        output_path:     Path to save the (N, 256) float32 tensor.
        batch_size:      Processing batch size.
        max_text_length: FinBERT token sequence length.
        device:          "cuda" or "cpu".
    """
    logger.info(f"Extracting text embeddings ({len(dataset)} samples)...")
    print("[PROGRESS] Extracting text embeddings...")
    sys.stdout.flush()
    
    encoder = encoder.to(device)
    encoder.eval()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    
    all_embeddings = []
    
    # Process in batches
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    with tqdm(total=num_batches, desc="Text extraction", unit="batch") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(dataset))
            batch_samples = dataset[start_idx:end_idx]
            
            # Tokenize
            texts = batch_samples["text_content"]
            encoded = tokenizer(
                texts,
                max_length=max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            # Extract embeddings
            with torch.no_grad():
                batch_embeddings = encoder(input_ids, attention_mask)  # (batch, 256)
            
            all_embeddings.append(batch_embeddings.cpu())
            pbar.update(1)
    
    # Concatenate all embeddings
    text_embeddings = torch.cat(all_embeddings, dim=0)  # (total_samples, 256)
    logger.info(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Save to disk
    torch.save(text_embeddings, output_path)
    logger.info(f"✓ Saved text embeddings to {output_path}")
    print(f"[PROGRESS] ✓ Text embeddings saved ({text_embeddings.shape})")
    sys.stdout.flush()


def extract_image_embeddings(
    dataset,
    encoder: FrozenImageEncoder,
    output_path: Path,
    batch_size: int = 32,
    image_size: int = 224,
    device: str = "cuda",
) -> None:
    """
    Extract ViT [CLS] embeddings for all samples and save to disk.

    Args:
        dataset:     HuggingFace Dataset with an "image_path" column.
        encoder:     FrozenImageEncoder instance.
        output_path: Path to save the (N, 256) float32 tensor.
        batch_size:  Processing batch size.
        image_size:  ViT input resolution (default: 224).
        device:      "cuda" or "cpu".
    """
    logger.info(f"Extracting image embeddings ({len(dataset)} samples)...")
    print("[PROGRESS] Extracting image embeddings...")
    sys.stdout.flush()
    
    encoder = encoder.to(device)
    encoder.eval()
    
    all_embeddings = []
    
    # Process in batches
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    with tqdm(total=num_batches, desc="Image extraction", unit="batch") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(dataset))
            batch_samples = dataset[start_idx:end_idx]
            
            # Load and preprocess images
            images_list = []
            for image_path in batch_samples["image_path"]:
                if isinstance(image_path, str):
                    img = Image.open(image_path).convert("RGB")
                else:
                    # Assume PIL Image
                    img = image_path.convert("RGB") if hasattr(image_path, "convert") else image_path
                
                # Resize and normalize
                img = img.resize((image_size, image_size), Image.LANCZOS)
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_tensor = torch.tensor(img_array).permute(2, 0, 1)  # (3, H, W)
                images_list.append(img_tensor)
            
            images_batch = torch.stack(images_list).to(device)  # (batch, 3, H, W)
            
            # Extract embeddings
            with torch.no_grad():
                batch_embeddings = encoder(images_batch)  # (batch, 256)
            
            all_embeddings.append(batch_embeddings.cpu())
            pbar.update(1)
    
    # Concatenate all embeddings
    image_embeddings = torch.cat(all_embeddings, dim=0)  # (total_samples, 256)
    logger.info(f"Image embeddings shape: {image_embeddings.shape}")
    
    # Save to disk
    torch.save(image_embeddings, output_path)
    logger.info(f"✓ Saved image embeddings to {output_path}")
    print(f"[PROGRESS] ✓ Image embeddings saved ({image_embeddings.shape})")
    sys.stdout.flush()


def main(args):
    """Extract text, image, and tabular features for the specified asset(s) and save to disk."""
    setup_logging()
    logger.info("=" * 80)
    logger.info("Offline Feature Extraction Pipeline - v5 FULL SEQUENCE")
    logger.info("=" * 80)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Determine assets to process
    if args.asset == "MULTI":
        assets_to_process = ["BTC", "ETH"]
    else:
        assets_to_process = [args.asset]

    # Initialize encoders. Text encoder is created once regardless of how many image
    # backbones are requested. One image encoder is created per requested backbone so
    # the dataset (and text embeddings) only need to be loaded/extracted ONCE per asset,
    # even when extracting embeddings for multiple image backbones in the same run.
    text_encoder = FrozenTextEncoder(hidden_dim=256)
    image_backbones = args.image_backbones
    image_encoders = {b: FrozenImageEncoder(hidden_dim=256, backbone=b) for b in image_backbones}
    # Diagnostic-only backbone swap (thesis Section 5.3): "vit" (default) writes
    # image_embeddings.pt as before; any other backbone writes a separate file so it
    # does not overwrite the primary ViT embeddings used for the thesis's main results.
    image_output_filenames = {
        b: ("image_embeddings.pt" if b == "vit" else f"image_embeddings_{b}.pt") for b in image_backbones
    }

    base_output_dir = Path(args.output_dir)

    for asset in assets_to_process:
        logger.info("\n" + "=" * 80)
        logger.info(f"PROCESSING ASSET: {asset}")
        logger.info("=" * 80)

        # Create asset-specific output directory
        asset_output_dir = base_output_dir / asset
        asset_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Asset output directory: {asset_output_dir}")

        # Load full sequence for the asset (concatenating train + test splits)
        full_dataset = load_asset_dataset(asset, debug=args.debug)
        total_samples = len(full_dataset)
        logger.info(f"✓ {asset} sequence length: {total_samples} samples")

        # Save metadata containing total_samples for this asset
        import json
        metadata = {"total_samples": total_samples}
        metadata_path = asset_output_dir / "split_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Metadata saved: {metadata_path} (total_samples={total_samples})")

        # Text embeddings
        text_output_path = asset_output_dir / "text_embeddings.pt"
        if text_output_path.exists() and not args.force:
            logger.info(f"✓ Text embeddings already exist for {asset}: {text_output_path}")
            print(f"[PROGRESS] ({asset}) Skipping text extraction (file exists)")
        else:
            start_time = time.time()
            extract_text_embeddings(
                full_dataset,
                text_encoder,
                text_output_path,
                batch_size=32,
                device=device,
            )
            elapsed = time.time() - start_time
            logger.info(f"{asset} text extraction took {format_duration(elapsed)}")
            print(f"[PROGRESS] ({asset}) Text extraction complete ({format_duration(elapsed)})")

        # Image embeddings — one pass per requested backbone, reusing the same full_dataset
        for backbone in image_backbones:
            image_output_path = asset_output_dir / image_output_filenames[backbone]
            if image_output_path.exists() and not args.force:
                logger.info(f"✓ Image embeddings ({backbone}) already exist for {asset}: {image_output_path}")
                print(f"[PROGRESS] ({asset}) Skipping image extraction ({backbone}, file exists)")
                continue
            start_time = time.time()
            extract_image_embeddings(
                full_dataset,
                image_encoders[backbone],
                image_output_path,
                batch_size=32,
                device=device,
            )
            elapsed = time.time() - start_time
            logger.info(f"{asset} image extraction ({backbone}) took {format_duration(elapsed)}")
            print(f"[PROGRESS] ({asset}) Image extraction ({backbone}) complete ({format_duration(elapsed)})")

        # Tabular features (raw, no scaling)
        # Saves two files:
        #   tabular_features.pt          — 7 base features (used by default training)
        #   tabular_features_extended.pt — 11 features (7 base + MA7/MA25/RSI/MACD)
        tabular_output_path = asset_output_dir / "tabular_features_extended.pt"
        if tabular_output_path.exists() and not args.force:
            logger.info(f"✓ Tabular features already exist for {asset}: {tabular_output_path}")
            print(f"[PROGRESS] ({asset}) Skipping tabular extraction (file exists)")
        else:
            start_time = time.time()
            extract_tabular_features_sequence(
                full_dataset,
                tabular_output_path,
            )
            elapsed = time.time() - start_time
            logger.info(f"{asset} tabular extraction took {format_duration(elapsed)}")
            print(f"[PROGRESS] ({asset}) Tabular extraction complete ({format_duration(elapsed)})")

        # Also save the 7-base-feature subset as tabular_features.pt for default training
        base_tabular_path = asset_output_dir / "tabular_features.pt"
        if base_tabular_path.exists() and not args.force:
            logger.info(f"✓ Base tabular features already exist for {asset}: {base_tabular_path}")
        else:
            extended = torch.load(tabular_output_path, map_location="cpu")
            torch.save(extended[:, :7].contiguous(), base_tabular_path)
            logger.info(f"✓ Saved 7-feature base tensor → {base_tabular_path}")

        # Target scores (raw, no scaling)
        target_output_path = asset_output_dir / "target_scores.pt"
        if target_output_path.exists() and not args.force:
            logger.info(f"✓ Target scores already exist for {asset}: {target_output_path}")
            print(f"[PROGRESS] ({asset}) Skipping target extraction (file exists)")
        else:
            start_time = time.time()
            extract_target_scores_sequence(
                full_dataset,
                target_output_path,
            )
            elapsed = time.time() - start_time
            logger.info(f"{asset} target extraction took {format_duration(elapsed)}")
            print(f"[PROGRESS] ({asset}) Target extraction complete ({format_duration(elapsed)})")

        # Verify files for this asset
        text_path = asset_output_dir / "text_embeddings.pt"
        image_paths = {b: asset_output_dir / image_output_filenames[b] for b in image_backbones}
        tabular_path = asset_output_dir / "tabular_features_extended.pt"
        target_path = asset_output_dir / "target_scores.pt"

        if (text_path.exists() and all(p.exists() for p in image_paths.values()) and
            tabular_path.exists() and target_path.exists()):
            text_shape = torch.load(text_path, map_location="cpu").shape
            image_shapes = {b: torch.load(p, map_location="cpu").shape for b, p in image_paths.items()}
            tabular_shape = torch.load(tabular_path, map_location="cpu").shape
            target_shape = torch.load(target_path, map_location="cpu").shape
            logger.info(f"✓ {asset} Verification:")
            logger.info(f"  text_embeddings: {text_shape}")
            for b, shape in image_shapes.items():
                logger.info(f"  image_embeddings ({b}): {shape}")
            logger.info(f"  tabular_features: {tabular_shape}")
            logger.info(f"  target_scores: {target_shape}")
            print(f"[PROGRESS] ✓ {asset} features ready: text {text_shape}, image {image_shapes}, tabular {tabular_shape}, target {target_shape}")
        else:
            logger.warning(f"✗ {asset} features missing files!")
            print(f"[PROGRESS] ✗ {asset} features missing files!")

    print("[PROGRESS] ✓ Feature extraction pipeline complete!")
    sys.stdout.flush()




def extract_target_scores_sequence(
    dataset,
    output_path: Path,
) -> None:
    """
    Extract raw target scores (no scaling) and save as (N, 3) float32 tensor.

    Column order: [y_baseline, y_heuristic, y_vol_adj_return].
    Scaling and clipping are applied in-memory during training.

    Args:
        dataset:     HuggingFace Dataset with y_baseline, y_heuristic, y_vol_adj_return columns.
        output_path: Path to save the target tensor.
    """
    logger.info(f"Extracting v5 targets ({len(dataset)} samples)...")
    print("[PROGRESS] Extracting v5 targets (y_baseline, y_heuristic, y_vol_adj_return) RAW...")
    sys.stdout.flush()

    target_cols = ["y_baseline", "y_heuristic", "y_vol_adj_return"]
    targets = []
    for col in target_cols:
        arr = np.array(dataset[col], dtype=np.float32)
        targets.append(arr)
        logger.info(f"  {col}: [{arr.min():.5f}, {arr.max():.5f}]")

    # Stack → (N, 3)
    target_array = np.stack(targets, axis=1)
    target_tensor = torch.tensor(target_array, dtype=torch.float32)
    logger.info(f"Target tensor shape: {target_tensor.shape}  (N, 3)")

    torch.save(target_tensor, output_path)
    logger.info(f"✓ Saved v5 targets to {output_path}")
    print(f"[PROGRESS] ✓ Targets extracted and saved {target_tensor.shape}")
    sys.stdout.flush()


def _compute_technical_indicators(return_1h_arr: np.ndarray) -> dict:
    """
    Compute MA7_ratio, MA25_ratio, RSI(14), and MACD histogram from return_1h.

    Price is reconstructed as the cumulative product of (1 + return/100).
    MA ratios are expressed as percent deviations to stay scale-invariant across
    BTC and ETH. All computations are backward-looking (no look-ahead).
    """
    returns = pd.Series(return_1h_arr.astype(np.float64))
    price = (1.0 + returns.fillna(0.0) / 100.0).cumprod()

    ma7 = price.rolling(7, min_periods=1).mean()
    ma25 = price.rolling(25, min_periods=1).mean()
    ma7_ratio = ((price / ma7.clip(lower=1e-8)) - 1.0) * 100.0
    ma25_ratio = ((price / ma25.clip(lower=1e-8)) - 1.0) * 100.0

    rsi = ta.momentum.rsi(price, window=14).fillna(50.0)

    macd_obj = ta.trend.MACD(price, window_slow=26, window_fast=12, window_sign=9)
    macd_hist = macd_obj.macd_diff().fillna(0.0)

    return {
        "ma7_ratio":  ma7_ratio.fillna(0.0).astype(np.float32).values,
        "ma25_ratio": ma25_ratio.fillna(0.0).astype(np.float32).values,
        "rsi_14":     rsi.astype(np.float32).values,
        "macd_hist":  macd_hist.astype(np.float32).values,
    }


# v6 tabular feature column order (11 columns = 7 original + 4 technical indicators)
_BASE_FEATURE_NAMES = [
    "return_1h",              # 1-hour price return
    "volume",                 # trading volume
    "funding_rate",           # futures funding rate
    "gdelt_econ_volume",      # GDELT economic news volume
    "gdelt_econ_tone",        # GDELT economic news tone
    "gdelt_conflict_volume",  # GDELT conflict news volume
    "is_post_ETF",            # binary flag: 1 if >= 2024-01-01 (ETF approval)
]
_COMPUTED_FEATURE_NAMES = [
    "ma7_ratio",              # % deviation of price from 7-period MA (scale-invariant)
    "ma25_ratio",             # % deviation of price from 25-period MA (scale-invariant)
    "rsi_14",                 # RSI(14) momentum oscillator [0, 100]
    "macd_hist",              # MACD histogram (MACD line − signal line)
]
TABULAR_FEATURE_NAMES = _BASE_FEATURE_NAMES + _COMPUTED_FEATURE_NAMES




def extract_tabular_features_sequence(
    dataset,
    output_path: Path,
) -> None:
    """
    Extract 11 tabular features (7 base + 4 technical indicators) and save to disk.

    Feature order:
        [return_1h, volume, funding_rate,
         gdelt_econ_volume, gdelt_econ_tone, gdelt_conflict_volume,
         is_post_ETF,
         ma7_ratio, ma25_ratio, rsi_14, macd_hist]

    Technical indicators are derived from return_1h via price reconstruction.
    No scaling is applied here — scaling happens in-memory during training.

    Args:
        dataset:     HuggingFace Dataset.
        output_path: Path to save the (N, 11) float32 tensor.
    """
    logger.info(f"Extracting v6 tabular features ({len(dataset)} samples)...")
    print("[PROGRESS] Extracting tabular features (v6: 7 base + 4 technical indicators)...")
    sys.stdout.flush()

    # Load 7 base features from HF dataset
    tabular_features = []
    for feature_name in _BASE_FEATURE_NAMES:
        arr = np.array(dataset[feature_name], dtype=np.float32)
        tabular_features.append(arr)

    # Compute 4 technical indicators from return_1h (no HF column needed)
    return_1h_arr = np.array(dataset["return_1h"], dtype=np.float32)
    tech = _compute_technical_indicators(return_1h_arr)
    for feature_name in _COMPUTED_FEATURE_NAMES:
        tabular_features.append(tech[feature_name])

    # Stack → (N, 11)
    tabular_array = np.stack(tabular_features, axis=1).astype(np.float32)
    logger.info(f"Tabular array shape: {tabular_array.shape}")
    logger.info("Feature ranges (RAW):")
    for i, name in enumerate(TABULAR_FEATURE_NAMES):
        logger.info(f"  {name}: [{tabular_array[:, i].min():.4f}, {tabular_array[:, i].max():.4f}]")

    tabular_tensor = torch.tensor(tabular_array, dtype=torch.float32).contiguous()
    torch.save(tabular_tensor, output_path)
    logger.info(f"✓ Saved v6 tabular features to {output_path}")
    print(f"[PROGRESS] ✓ Tabular features extracted and saved {tabular_tensor.shape}")
    sys.stdout.flush()



def push_features_to_hf(
    output_dir: Path,
    repo_id: str,
    token: str = None,
    private: bool = False,
) -> None:
    """
    Upload extracted .pt feature files to a Hugging Face dataset repository.

    Args:
        output_dir: Directory containing .pt files to upload.
        repo_id:    HF repo ID (e.g., "username/crypto-features").
        token:      HF API token (uses cached token if None).
        private:    Whether to create a private repository.
    """
    logger.info(f"\nUploading features to HuggingFace: {repo_id}...")
    print(f"\n[PROGRESS] Uploading features to {repo_id}...")
    sys.stdout.flush()
    
    try:
        from huggingface_hub import HfFolder
        
        if token is None:
            token = HfFolder.get_token()
        
        if token is None:
            logger.error(
                "No HF token found. Please login first:\n"
                "  huggingface-cli login\n"
                "Or pass --token <your-token>"
            )
            raise ValueError("HF token required for upload")
        
        api = HfApi(token=token)
        
        # Create repo if doesn't exist
        logger.info(f"Creating/checking repo: {repo_id}")
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token
        )
        logger.info(f"✓ Repository ready")
        
        # Upload folder
        logger.info(f"Uploading {output_dir} to {repo_id}...")
        info = api.upload_folder(
            folder_path=str(output_dir),
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            allow_patterns=["*.pt"],
            ignore_patterns=[".git", "__pycache__"],
            commit_message="Upload pre-extracted crypto sentiment features"
        )
        logger.info(f"✓ Upload complete")
        print(f"[PROGRESS] ✓ Features uploaded to https://huggingface.co/datasets/{repo_id}")
        sys.stdout.flush()
        
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise


def push_features_to_kaggle(
    output_dir: Path,
    dataset_name: str,
    kaggle_username: str,
    kaggle_key: str,
    public: bool = False,
) -> None:
    """
    Create a dataset-metadata.json for Kaggle CLI upload.

    Writes the metadata file to output_dir. Upload with:
        cd <output_dir> && kaggle datasets version -m "message" -p .

    Args:
        output_dir:       Directory containing .pt files.
        dataset_name:     Kaggle dataset name slug.
        kaggle_username:  Kaggle account username (for metadata).
        kaggle_key:       Unused; kept for API compatibility.
        public:           Whether to make the dataset public.
    """
    logger.info(f"\nPreparing features for Kaggle upload: {dataset_name}...")
    print(f"\n[PROGRESS] Preparing features for Kaggle upload: {dataset_name}...")
    sys.stdout.flush()
    
    try:
        import json
        
        # Create dataset metadata
        metadata = {
            "title": dataset_name,
            "id": f"{kaggle_username}/{dataset_name}",
            "licenses": [{"name": "cc-by-nc-4"}],
            "resources": []
        }
        
        # Find all .pt files
        pt_files = sorted(list(output_dir.glob("*.pt")))
        logger.info(f"Found {len(pt_files)} .pt files")
        
        for pt_file in pt_files:
            metadata["resources"].append({
                "path": pt_file.name,
            })
        
        # Save metadata to dataset folder
        metadata_path = output_dir / "dataset-metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Metadata created: {metadata_path}")
        
        # Instructions for CLI upload
        logger.info(f"\nTo upload to Kaggle, use the CLI:")
        logger.info(f"  cd {output_dir}")
        logger.info(f"  kaggle datasets version -m 'Updated features' -p .")
        print(f"[PROGRESS] ✓ Metadata ready. Use CLI to upload: kaggle datasets version -p .")
        sys.stdout.flush()
        
    except Exception as e:
        logger.error(f"Kaggle upload failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and cache FinBERT text and Vision Transformer image embeddings"
    )
    parser.add_argument(
        "--asset",
        choices=["BTC", "ETH", "MULTI"],
        default="MULTI",
        help="Cryptocurrency asset (MULTI = BTC + ETH)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/features",
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if files exist",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (extract only 100 samples per split)",
    )
    parser.add_argument(
        "--image-backbones",
        nargs="+",
        choices=["vit", "clip"],
        default=["vit"],
        dest="image_backbones",
        help="Frozen image encoder backbone(s) to extract, e.g. '--image-backbones vit clip' "
             "extracts both in a single run (dataset/text embeddings are only loaded/extracted "
             "once, regardless of how many backbones are listed). 'vit' (default, "
             "google/vit-base-patch16-224, ImageNet-pretrained) is the thesis's primary "
             "configuration, saved as image_embeddings.pt. 'clip' (openai/clip-vit-base-patch16 "
             "vision tower) is a diagnostic-only alternative, saved separately as "
             "image_embeddings_clip.pt so it does not overwrite the primary ViT embeddings.",
    )
    parser.add_argument(
        "--push-to-hf",
        action="store_true",
        help="Upload extracted features to Hugging Face after extraction",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="Hugging Face repo ID for uploading (username/repo-name)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF API token (uses cached if not provided)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make HF repo private",
    )
    parser.add_argument(
        "--push-to-kaggle",
        action="store_true",
        help="Upload extracted features to Kaggle dataset after extraction",
    )
    parser.add_argument(
        "--kaggle-dataset-name",
        type=str,
        default=None,
        help="Kaggle dataset name (e.g., crypto-sentiment-embeddings)",
    )
    parser.add_argument(
        "--kaggle-username",
        type=str,
        default=None,
        help="Kaggle API username (uses ~/.kaggle/kaggle.json if not provided)",
    )
    parser.add_argument(
        "--kaggle-key",
        type=str,
        default=None,
        help="Kaggle API key (uses ~/.kaggle/kaggle.json if not provided)",
    )
    parser.add_argument(
        "--kaggle-public",
        action="store_true",
        help="Make Kaggle dataset public",
    )
    
    args = parser.parse_args()
    
    main(args)
    
    # Push to HF if requested
    if args.push_to_hf:
        if not args.hf_repo_id:
            print("[ERROR] --hf-repo-id required when --push-to-hf is set")
            sys.exit(1)
        
        # If single asset, upload from the asset subdirectory. If MULTI, upload parent directory.
        if args.asset == "MULTI":
            upload_dir = Path(args.output_dir)
        else:
            upload_dir = Path(args.output_dir) / args.asset
            
        push_features_to_hf(
            output_dir=upload_dir,
            repo_id=args.hf_repo_id,
            token=args.token,
            private=args.private,
        )
    
    # Push to Kaggle if requested
    if args.push_to_kaggle:
        if not args.kaggle_dataset_name:
            print("[ERROR] --kaggle-dataset-name required when --push-to-kaggle is set")
            sys.exit(1)
        
        # If single asset, prepare metadata in the asset subdirectory. If MULTI, in parent.
        if args.asset == "MULTI":
            upload_dir = Path(args.output_dir)
        else:
            upload_dir = Path(args.output_dir) / args.asset
        
        # Get Kaggle credentials
        kaggle_username = args.kaggle_username
        kaggle_key = args.kaggle_key
        
        if not kaggle_username or not kaggle_key:
            # Try to read from ~/.kaggle/kaggle.json
            import json
            kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
            if kaggle_json_path.exists():
                with open(kaggle_json_path) as f:
                    kaggle_config = json.load(f)
                    kaggle_username = kaggle_config.get("username")
                    kaggle_key = kaggle_config.get("key")
        
        if not kaggle_username or not kaggle_key:
            print("[ERROR] Kaggle credentials required. Provide via --kaggle-username and --kaggle-key or ~/.kaggle/kaggle.json")
            sys.exit(1)
        
        push_features_to_kaggle(
            output_dir=upload_dir,
            dataset_name=args.kaggle_dataset_name,
            kaggle_username=kaggle_username,
            kaggle_key=kaggle_key,
            public=args.kaggle_public,
        )
        
        print("\n" + "=" * 80)
        print("✅ Feature extraction and upload complete!")
        print("=" * 80)
        print(f"\nUse in training with:")
        print(f"  python src/training/train.py --hf-features-repo {args.hf_repo_id}")
