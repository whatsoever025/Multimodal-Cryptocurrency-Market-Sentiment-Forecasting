"""
Offline Feature Extraction Pipeline

Extracts embeddings from frozen FinBERT and ResNet50 once, saves to disk.
Eliminates per-batch backbone computations and I/O overhead during training.

Usage:
    python src/data/extract_features.py --asset MULTI --output_dir /path/to/features
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
from pathlib import Path
from PIL import Image
from typing import Dict, Tuple
import time

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

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    raise ImportError("'kaggle' required: pip install kaggle")

from tqdm import tqdm
from ..training.utils import setup_logging, format_duration


logger = logging.getLogger(__name__)


class FrozenTextEncoder(nn.Module):
    """
    Frozen FinBERT encoder for extracting text embeddings.
    
    Input: (batch, max_text_length) token IDs
    Output: (batch, 256) embeddings
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
            input_ids: (batch, max_text_length)
            attention_mask: (batch, max_text_length)
        
        Returns:
            (batch, hidden_dim)
        """
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            cls_token = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        
        projected = self.projection(cls_token)  # (batch, hidden_dim)
        return self.dropout(projected)


class FrozenImageEncoder(nn.Module):
    """
    Frozen ResNet50 encoder for extracting image embeddings.
    
    Input: (batch, 3, 224, 224) RGB images (normalized [0, 1])
    Output: (batch, 256) embeddings
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        logger.info("Loading ResNet50...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze backbone
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Remove classification head, keep backbone + avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Up to avgpool
        
        # Project avgpool output (2048) to hidden_dim
        self.projection = nn.Linear(2048, hidden_dim)
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
            features = self.backbone(images)  # (batch, 2048, 1, 1)
        
        features = features.squeeze(-1).squeeze(-1)  # (batch, 2048)
        projected = self.projection(features)  # (batch, hidden_dim)
        return self.dropout(projected)


def load_dataset_multi_asset(split: str = "train", debug: bool = False):
    """
    Load multi-asset dataset (BTC + ETH concatenated).
    
    Args:
        split: "train", "validation", or "test_in_domain"
        debug: If True, load only 100 samples for testing
    
    Returns:
        Concatenated HF Dataset
    """
    logger.info(f"Loading multi-asset dataset ({split} split)...")
    
    print(f"[PROGRESS] Downloading BTC dataset ({split})...")
    btc_dataset = load_dataset(
        "khanh252004/multimodal_crypto_sentiment_btc",
        split=split,
        cache_dir="/tmp/huggingface_cache",
    )
    
    print(f"[PROGRESS] Downloading ETH dataset ({split})...")
    eth_dataset = load_dataset(
        "khanh252004/multimodal_crypto_sentiment_eth",
        split=split,
        cache_dir="/tmp/huggingface_cache",
    )
    
    # Concatenate datasets
    dataset = concatenate_datasets([btc_dataset, eth_dataset])
    
    if debug:
        dataset = dataset.select(range(min(100, len(dataset))))
    
    logger.info(f"Loaded {len(dataset)} samples for {split}")
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
    Extract text embeddings for all samples and save to disk.
    
    Args:
        dataset: HuggingFace Dataset
        encoder: FrozenTextEncoder
        output_path: Path to save embeddings
        batch_size: Batch size for processing
        max_text_length: BERT token sequence length
        device: "cuda" or "cpu"
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
    Extract image embeddings for all samples and save to disk.
    
    Args:
        dataset: HuggingFace Dataset
        encoder: FrozenImageEncoder
        output_path: Path to save embeddings
        batch_size: Batch size for processing
        image_size: ResNet50 input size
        device: "cuda" or "cpu"
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
    """
    Main extraction script.
    
    Args:
        args: Parsed command-line arguments
    """
    setup_logging()
    logger.info("=" * 80)
    logger.info("Offline Feature Extraction Pipeline")
    logger.info("=" * 80)
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # Initialize encoders
    text_encoder = FrozenTextEncoder(hidden_dim=256)
    image_encoder = FrozenImageEncoder(hidden_dim=256)
    
    # Extract features for each split
    splits = ["train", "validation", "test_in_domain"]
    
    for split in splits:
        logger.info("\n" + "-" * 80)
        logger.info(f"Processing {split} split...")
        logger.info("-" * 80)
        
        # Load dataset
        dataset = load_dataset_multi_asset(split=split, debug=args.debug)
        
        # Text embeddings
        text_output_path = output_dir / f"text_embeddings_{split}.pt"
        if text_output_path.exists() and not args.force:
            logger.info(f"✓ Text embeddings already exist: {text_output_path}")
            print(f"[PROGRESS] Skipping text extraction (file exists)")
        else:
            start_time = time.time()
            extract_text_embeddings(
                dataset,
                text_encoder,
                text_output_path,
                batch_size=32,
                device=device,
            )
            elapsed = time.time() - start_time
            logger.info(f"Text extraction took {format_duration(elapsed)}")
            print(f"[PROGRESS] Text extraction complete ({format_duration(elapsed)})")
        
        # Image embeddings
        image_output_path = output_dir / f"image_embeddings_{split}.pt"
        if image_output_path.exists() and not args.force:
            logger.info(f"✓ Image embeddings already exist: {image_output_path}")
            print(f"[PROGRESS] Skipping image extraction (file exists)")
        else:
            start_time = time.time()
            extract_image_embeddings(
                dataset,
                image_encoder,
                image_output_path,
                batch_size=32,
                device=device,
            )
            elapsed = time.time() - start_time
            logger.info(f"Image extraction took {format_duration(elapsed)}")
            print(f"[PROGRESS] Image extraction complete ({format_duration(elapsed)})")
        
        # Tabular features (raw, no scaling)
        tabular_output_path = output_dir / f"tabular_features_{split}.pt"
        if tabular_output_path.exists() and not args.force:
            logger.info(f"✓ Tabular features already exist: {tabular_output_path}")
            print(f"[PROGRESS] Skipping tabular extraction (file exists)")
        else:
            start_time = time.time()
            extract_tabular_features(
                dataset,
                output_dir,
                split=split,
            )
            elapsed = time.time() - start_time
            logger.info(f"Tabular extraction took {format_duration(elapsed)}")
            print(f"[PROGRESS] Tabular extraction complete ({format_duration(elapsed)})")
        
        # Target scores (raw, no scaling)
        target_output_path = output_dir / f"target_scores_{split}.pt"
        if target_output_path.exists() and not args.force:
            logger.info(f"✓ Target scores already exist: {target_output_path}")
            print(f"[PROGRESS] Skipping target extraction (file exists)")
        else:
            start_time = time.time()
            extract_target_scores(
                dataset,
                target_output_path,
            )
            elapsed = time.time() - start_time
            logger.info(f"Target extraction took {format_duration(elapsed)}")
            print(f"[PROGRESS] Target extraction complete ({format_duration(elapsed)})")
        
        sys.stdout.flush()
    
    # Verify all files
    logger.info("\n" + "-" * 80)
    logger.info("Extraction Complete!")
    logger.info("-" * 80)
    
    for split in splits:
        text_path = output_dir / f"text_embeddings_{split}.pt"
        image_path = output_dir / f"image_embeddings_{split}.pt"
        tabular_path = output_dir / f"tabular_features_{split}.pt"
        target_path = output_dir / f"target_scores_{split}.pt"
        
        if (text_path.exists() and image_path.exists() and 
            tabular_path.exists() and target_path.exists()):
            text_shape = torch.load(text_path, map_location="cpu").shape
            image_shape = torch.load(image_path, map_location="cpu").shape
            tabular_shape = torch.load(tabular_path, map_location="cpu").shape
            target_shape = torch.load(target_path, map_location="cpu").shape
            logger.info(f"✓ {split}: text {text_shape}, image {image_shape}, tabular {tabular_shape}, target {target_shape}")
            print(f"[PROGRESS] ✓ {split}: text {text_shape}, image {image_shape}, tabular {tabular_shape}, target {target_shape}")
        else:
            logger.warning(f"✗ {split}: Missing files")
            print(f"[PROGRESS] ✗ {split}: Missing files")
    
    print("[PROGRESS] ✓ Feature extraction pipeline complete!")
    sys.stdout.flush()


def extract_target_scores(
    dataset,
    output_path: Path,
) -> None:
    """
    Extract RAW target_score (no scaling). Save to disk.
    Scaling will be done during training on Kaggle.
    
    Args:
        dataset: HuggingFace Dataset with target_score column
        output_path: Path to save raw target scores
    """
    logger.info(f"Extracting target_score ({len(dataset)} samples)...")
    print("[PROGRESS] Extracting target_score (RAW, no scaling)...")
    sys.stdout.flush()
    
    # Extract target scores (raw, no scaling)
    target_scores = np.array(dataset["target_score"], dtype=np.float32)
    logger.info(f"Target scores shape: {target_scores.shape}")
    logger.info(f"Target scores range (RAW): [{target_scores.min():.4f}, {target_scores.max():.4f}]")
    
    # Save raw targets
    target_tensor = torch.tensor(target_scores, dtype=torch.float32)
    torch.save(target_tensor, output_path)
    logger.info(f"✓ Saved raw target scores to {output_path}")
    print(f"[PROGRESS] ✓ Target scores extracted and saved ({target_tensor.shape})")
    sys.stdout.flush()


def extract_tabular_features(
    dataset,
    output_dir: Path,
    split: str = "train",
) -> None:
    """
    Extract RAW 7 tabular features (no scaling). Scaling will be done during training.
    
    Features:
    1. return_1h: Price return in 1 hour
    2. volume: Trading volume
    3. funding_rate: Futures funding rate
    4. fear_greed_value: Fear & Greed Index
    5. gdelt_econ_volume: News volume (economic)
    6. gdelt_econ_tone: News tone (economic)
    7. gdelt_conflict_volume: News volume (conflict)
    
    Args:
        dataset: HuggingFace Dataset with tabular columns
        output_dir: Directory to save raw features
        split: "train", "validation", or "test_in_domain"
    """
    logger.info(f"Extracting tabular features ({len(dataset)} samples)...")
    print("[PROGRESS] Extracting tabular features (RAW, no scaling)...")
    sys.stdout.flush()
    
    # Extract 7 tabular features
    tabular_features = []
    feature_names = [
        "return_1h",
        "volume",
        "funding_rate",
        "fear_greed_value",
        "gdelt_econ_volume",
        "gdelt_econ_tone",
        "gdelt_conflict_volume",
    ]
    
    for feature_name in feature_names:
        feature_values = np.array(dataset[feature_name], dtype=np.float32)
        tabular_features.append(feature_values)
    
    # Stack into (total_samples, 7) array
    tabular_array = np.stack(tabular_features, axis=1).astype(np.float32)
    logger.info(f"Tabular array shape: {tabular_array.shape}")
    logger.info(f"Feature ranges (RAW):")
    for i, name in enumerate(feature_names):
        logger.info(f"  {name}: [{tabular_array[:, i].min():.4f}, {tabular_array[:, i].max():.4f}]")
    
    # Save raw tabular features
    output_path = output_dir / f"tabular_features_{split}.pt"
    tabular_tensor = torch.tensor(tabular_array, dtype=torch.float32).contiguous()
    torch.save(tabular_tensor, output_path)
    logger.info(f"✓ Saved raw tabular features to {output_path}")
    print(f"[PROGRESS] ✓ Tabular features extracted and saved ({tabular_tensor.shape})")
    sys.stdout.flush()



def push_features_to_hf(
    output_dir: Path,
    repo_id: str,
    token: str = None,
    private: bool = False,
) -> None:
    """
    Upload extracted features to Hugging Face dataset.
    
    Args:
        output_dir: Directory containing .pt files
        repo_id: HF repo ID (e.g., username/crypto-features)
        token: HF API token (uses cached if not provided)
        private: Whether to make repo private
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
    Upload extracted features to Kaggle dataset.
    
    Args:
        output_dir: Directory containing .pt files
        dataset_name: Kaggle dataset name (e.g., crypto-sentiment-features)
        kaggle_username: Kaggle API username
        kaggle_key: Kaggle API key
        public: Whether to make dataset public
    """
    logger.info(f"\nUploading features to Kaggle: {dataset_name}...")
    print(f"\n[PROGRESS] Uploading features to Kaggle: {dataset_name}...")
    sys.stdout.flush()
    
    try:
        import json
        import os
        
        # Initialize Kaggle API with credentials
        # Set environment variables for Kaggle authentication
        os.environ['KAGGLE_USERNAME'] = kaggle_username
        os.environ['KAGGLE_KEY'] = kaggle_key
        
        api = KaggleApi()
        api.authenticate()
        logger.info(f"✓ Kaggle authentication successful")
        
        # Create dataset metadata
        metadata = {
            "title": dataset_name,
            "id": f"{kaggle_username}/{dataset_name}",
            "licenses": [{"name": "cc-by-nc-4"}],
            "resources": []
        }
        
        # Find all .pt files
        pt_files = sorted(list(output_dir.glob("*.pt")))
        logger.info(f"Found {len(pt_files)} .pt files to upload")
        
        for pt_file in pt_files:
            metadata["resources"].append({
                "path": pt_file.name,
            })
        
        # Save metadata to dataset folder
        metadata_path = output_dir / "dataset-metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Metadata created: {metadata_path}")
        
        # Create/update dataset on Kaggle
        logger.info(f"Creating/updating dataset on Kaggle...")
        api.dataset_create_new(
            folder=str(output_dir),
            public=public,
            quiet=False
        )
        logger.info(f"✓ Dataset uploaded successfully")
        print(f"[PROGRESS] ✓ Features uploaded to Kaggle: {kaggle_username}/{dataset_name}")
        sys.stdout.flush()
        
    except Exception as e:
        logger.error(f"Kaggle upload failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and cache FinBERT text and ResNet50 image embeddings"
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
        help="Kaggle dataset name (e.g., crypto-sentiment-features)",
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
        
        output_dir = Path(args.output_dir)
        push_features_to_hf(
            output_dir=output_dir,
            repo_id=args.hf_repo_id,
            token=args.token,
            private=args.private,
        )
    
    # Push to Kaggle if requested
    if args.push_to_kaggle:
        if not args.kaggle_dataset_name:
            print("[ERROR] --kaggle-dataset-name required when --push-to-kaggle is set")
            sys.exit(1)
        
        output_dir = Path(args.output_dir)
        
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
            output_dir=output_dir,
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
