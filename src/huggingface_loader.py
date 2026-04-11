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
    import pandas as pd
except ImportError:
    raise ImportError("'datasets' package required: pip install datasets")


logger = logging.getLogger(__name__)


# Mapping of assets to HuggingFace dataset repos
HF_DATASET_REPOS = {
    "BTC": "khanh252004/multimodal_crypto_sentiment_btc",
    "ETH": "khanh252004/multimodal_crypto_sentiment_eth",
}



def download_hf_dataset_multi_asset(
    assets: list = ["BTC", "ETH"],
    output_dir: str = "data/raw",
    hf_token: Optional[str] = None,
) -> None:
    """
    Download and merge multiple asset datasets from HuggingFace.
    Creates a combined dataset with 'asset' column indicating source.
    
    Args:
        assets: List of cryptocurrency assets to combine (e.g., ["BTC", "ETH"])
        output_dir: Where to save CSV files
        hf_token: HuggingFace API token (reads from env if None)
    
    Example:
        download_hf_dataset_multi_asset(["BTC", "ETH"])
        # Downloads and merges BTC and ETH datasets
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    
    logger.info(f"Loading multi-asset datasets: {assets}")
    
    dfs = []
    for asset in assets:
        asset_upper = asset.upper()
        logger.info(f"  Loading {asset_upper}...")
        
        try:
            dataset_repo = HF_DATASET_REPOS[asset_upper]
            dataset = load_dataset(
                dataset_repo,
                token=hf_token,
            )
            
            # Handle different dataset structures
            if isinstance(dataset, dict):
                df = dataset["train"].to_pandas() if "train" in dataset else next(iter(dataset.values())).to_pandas()
            else:
                df = dataset.to_pandas()
            
            df['asset'] = asset_upper  # Add asset indicator column
            dfs.append(df)
            logger.info(f"     ✓ {asset_upper}: {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Error loading {asset_upper}: {e}")
            raise
    
    # Merge all datasets
    logger.info(f"\n  Merging {len(dfs)} datasets...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    output_file = output_dir / "multiasset_combined_data.csv"
    combined_df.to_csv(output_file, index=False)
    logger.info(f"     ✓ Saved combined dataset: {len(combined_df)} total rows")
    
    logger.info("\n✅ Multi-asset dataset download completed successfully!")


if __name__ == "__main__":
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Always download multi-asset (BTC + ETH combined)
    download_hf_dataset_multi_asset(["BTC", "ETH"], output_dir)
