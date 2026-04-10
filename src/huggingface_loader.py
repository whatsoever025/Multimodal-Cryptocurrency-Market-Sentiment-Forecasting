"""
Load datasets from HuggingFace Hub for Kaggle environment.
Optimized for multimodal crypto sentiment datasets.
"""

import os
from pathlib import Path
from typing import Optional, Dict
import logging

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("'datasets' package required: pip install datasets")


logger = logging.getLogger(__name__)


# Mapping of assets to HuggingFace dataset repos
HF_DATASET_REPOS = {
    "BTC": "khanh252004/multimodal_crypto_sentiment_btc",
    "ETH": "khanh252004/multimodal_crypto_sentiment_eth",
}


def get_dataset_repo(asset: str) -> str:
    """Get HuggingFace repo for given asset."""
    if asset not in HF_DATASET_REPOS:
        raise ValueError(f"Unknown asset: {asset}. Must be one of {list(HF_DATASET_REPOS.keys())}")
    return HF_DATASET_REPOS[asset]


def download_hf_dataset(
    asset: str = "BTC",
    output_dir: str = "data/raw",
    hf_token: Optional[str] = None,
) -> None:
    """
    Download multimodal crypto sentiment dataset from HuggingFace.
    
    Args:
        asset: Cryptocurrency asset ("BTC" or "ETH")
        output_dir: Where to save CSV files
        hf_token: HuggingFace API token (reads from env if None)
    
    Example:
        download_hf_dataset("BTC")
        # Downloads from khanh252004/multimodal_crypto_sentiment_btc
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get HF token from environment if not provided
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    
    # Get repo name for asset
    dataset_repo = get_dataset_repo(asset)
    
    logger.info(f"Loading dataset from HuggingFace: {dataset_repo}")
    
    try:
        # Load the dataset from HuggingFace
        logger.info("  → Downloading dataset...")
        dataset = load_dataset(
            dataset_repo,
            token=hf_token,
            trust_remote_code=True,
        )
        
        logger.info(f"  → Dataset loaded successfully")
        
        # Handle different dataset structures
        if isinstance(dataset, dict):  # Multi-split dataset
            logger.info(f"     Splits: {list(dataset.keys())}")
            df = dataset["train"].to_pandas() if "train" in dataset else next(iter(dataset.values())).to_pandas()
        else:  # Single split
            logger.info(f"     Columns: {dataset.column_names if hasattr(dataset, 'column_names') else 'N/A'}")
            df = dataset.to_pandas()
        
        logger.info(f"     Rows: {len(df)}")
        
        # Determine which files are in the dataset based on column names
        cols_lower = [col.lower() for col in df.columns]
        
        # Save all data to a single CSV (the dataset should have all required columns)
        output_file = output_dir / f"{asset}_combined_data.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"     ✓ Saved combined dataset to {asset}_combined_data.csv")
        
        # Also save individual CSVs if we can identify specific columns
        # This helps with compatibility with existing preprocessing code
        
        # Check for OHLCV/klines columns
        klines_cols = [col for col in df.columns if any(x in col.lower() for x in ['open', 'high', 'low', 'close', 'volume', 'time'])]
        if klines_cols:
            klines_df = df[klines_cols].copy()
            klines_path = output_dir / f"{asset}USDT_klines.csv"
            klines_df.to_csv(klines_path, index=False)
            logger.info(f"     ✓ Saved {asset}USDT_klines.csv")
        
        # Check for sentiment/fear-greed columns
        sentiment_cols = [col for col in df.columns if any(x in col.lower() for x in ['fear', 'greed', 'sentiment', 'score'])]
        if sentiment_cols:
            sentiment_df = df[sentiment_cols].copy()
            if 'timestamp' in cols_lower or 'date' in cols_lower or 'time' in cols_lower:
                time_cols = [col for col in df.columns if any(x in col.lower() for x in ['timestamp', 'date', 'time'])]
                sentiment_df = df[time_cols + sentiment_cols].copy()
            fgi_path = output_dir / "fear_greed_index.csv"
            sentiment_df.to_csv(fgi_path, index=False)
            logger.info(f"     ✓ Saved fear_greed_index.csv")
        
        # Check for text/news columns
        text_cols = [col for col in df.columns if any(x in col.lower() for x in ['news', 'text', 'title', 'content', 'article'])]
        if text_cols:
            text_df = df[text_cols].copy()
            news_path = output_dir / "huggingface_crypto_news.csv"
            text_df.to_csv(news_path, index=False)
            logger.info(f"     ✓ Saved huggingface_crypto_news.csv")
        
        # Check for funding rate columns
        funding_cols = [col for col in df.columns if any(x in col.lower() for x in ['funding', 'rate'])]
        if funding_cols:
            funding_df = df[funding_cols].copy()
            funding_path = output_dir / f"{asset}USDT_fundingRate.csv"
            funding_df.to_csv(funding_path, index=False)
            logger.info(f"     ✓ Saved {asset}USDT_fundingRate.csv")
        
        logger.info("\n✅ Dataset download completed successfully!")
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python huggingface_loader.py <ASSET> [OUTPUT_DIR]")
        print("Example: python huggingface_loader.py BTC data/raw")
        print(f"Available assets: {', '.join(HF_DATASET_REPOS.keys())}")
        sys.exit(1)
    
    asset = sys.argv[1].upper()
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/raw"
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    download_hf_dataset(asset, output_dir)
