"""
DataAligner: Production-Ready Multimodal Crypto Sentiment Dataset Alignment

Orchestrates merging of 8 heterogeneous data sources (OHLCV, funding rates, liquidations,
on-chain metrics, macroeconomic news, fear/greed index, Reddit/crypto news sentiment)
into a single aligned hourly dataset with continuous sentiment regression targets.

Key Features:
- Master hourly time index from OHLCV base (BTC or ETH)
- Comprehensive merging: exact joins for hourly data, forward-fill for lower-frequency
- Continuous target: Volatility-Adjusted Tanh formula (-100 to +100)
- Image validation: Disk verification, automatic dropping of missing images
- Hugging Face integration: Cast images, push to Hub with full metadata
- Strict data leakage prevention: Forward-fill only (no bfill), drop future-unavailable targets
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
import warnings

import pandas as pd
import numpy as np

# Hugging Face imports
from datasets import Dataset, Image
from huggingface_hub import HfApi

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataAligner:
    """
    Aligns multimodal cryptocurrency market sentiment data for deep learning.
    
    Merges 8 data sources into a single hourly dataset with:
    - Market data (OHLCV, funding rates)
    - On-chain metrics (open interest, liquidations, long/short ratio)
    - Sentiment (macro news, fear/greed index)
    - Text data (aggregated Reddit/crypto news)
    - Continuous regression target (sentiment score -100 to +100)
    - Chart images (local path → embedded in HF Dataset)
    
    Attributes:
        asset (str): "BTC" or "ETH"
        data_dir (str): Path to data directory containing raw/ and processed/ subdirs
        image_dir (str): Path to processed images directory
        horizon_hours (int): Hours ahead for target calculation (default 24)
    """
    
    def __init__(
        self,
        asset: str = "BTC",
        data_dir: str = "data",
        image_dir: str = "data/processed/images",
        horizon_hours: int = 24
    ):
        """
        Initialize DataAligner.
        
        Args:
            asset: "BTC" or "ETH"
            data_dir: Root data directory (should contain raw/ and processed/ subdirs)
            image_dir: Directory containing generated candlestick images
            horizon_hours: Time horizon for target calculation (default 24)
        
        Raises:
            ValueError: If asset is not "BTC" or "ETH"
        """
        self.asset = asset.upper()
        if self.asset not in ("BTC", "ETH"):
            raise ValueError(f"Invalid asset {asset}. Must be 'BTC' or 'ETH'.")
        
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.image_dir = Path(image_dir)
        self.horizon_hours = horizon_hours
        
        # Validate directories exist
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir}")
        if not self.image_dir.exists():
            logger.warning(f"Image directory does not exist: {self.image_dir}")
        
        # Data storage
        self.df: Optional[pd.DataFrame] = None
        self.text_data: Optional[pd.DataFrame] = None
        
        logger.info(f"DataAligner initialized for {self.asset}")
        logger.info(f"Data directory: {self.data_dir.absolute()}")
        logger.info(f"Image directory: {self.image_dir.absolute()}")
        logger.info(f"Target horizon: {horizon_hours} hours")
    
    # ============================================================================
    # PHASE 1: DATA LOADING
    # ============================================================================
    
    def load_all_data(self) -> None:
        """Load all 8 data sources from raw directory."""
        logger.info("=" * 80)
        logger.info("PHASE 1: Loading all data sources")
        logger.info("=" * 80)
        
        self._load_ohlcv()
        self._load_hourly_data()
        self._load_lower_frequency_data()
        self._load_text_data()
        
        logger.info(f"All data sources loaded. Base dataset shape: {self.df.shape}")
    
    def _load_ohlcv(self) -> None:
        """Load OHLCV klines data and create master index."""
        csv_path = self.raw_dir / f"{self.asset}USDT_klines.csv"
        logger.info(f"Loading OHLCV from {csv_path}")
        
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.set_index('timestamp')
        
        logger.info(f"  ✓ Loaded {len(df)} OHLCV records from {df.index.min()} to {df.index.max()}")
        logger.info(f"  ✓ Columns: {list(df.columns)}")
        
        self.df = df
    
    def _load_hourly_data(self) -> None:
        """Load and merge all hourly-granularity datasets (exact joins on timestamp + asset)."""
        logger.info(f"\nLoading hourly datasets (exact join on timestamp, filtered for {self.asset})...")
        
        # Load coinalyze_open_interest.csv
        try:
            csv_path = self.raw_dir / "coinalyze_open_interest.csv"
            oi_df = pd.read_csv(csv_path)
            oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'], utc=True)
            oi_df = oi_df[oi_df['asset'] == self.asset].copy()
            oi_df = oi_df[['timestamp', 'open', 'high', 'low', 'close']].copy()
            oi_df.columns = ['timestamp', 'open_interest_open', 'open_interest_high', 
                           'open_interest_low', 'open_interest_close']
            oi_df = oi_df.set_index('timestamp')
            self.df = self.df.join(oi_df, how='left')
            logger.info(f"  ✓ Loaded open_interest ({len(oi_df)} records for {self.asset})")
        except FileNotFoundError:
            logger.warning(f"  ✗ File not found: coinalyze_open_interest.csv")
        
        # Load coinalyze_liquidations.csv
        try:
            csv_path = self.raw_dir / "coinalyze_liquidations.csv"
            liq_df = pd.read_csv(csv_path)
            liq_df['timestamp'] = pd.to_datetime(liq_df['timestamp'], utc=True)
            liq_df = liq_df[liq_df['asset'] == self.asset].copy()
            liq_df = liq_df[['timestamp', 'long_liquidations', 'short_liquidations']].copy()
            liq_df = liq_df.set_index('timestamp')
            # Fill missing liquidations with 0
            self.df = self.df.join(liq_df, how='left')
            self.df['long_liquidations'] = self.df['long_liquidations'].fillna(0)
            self.df['short_liquidations'] = self.df['short_liquidations'].fillna(0)
            logger.info(f"  ✓ Loaded liquidations ({len(liq_df)} records for {self.asset})")
        except FileNotFoundError:
            logger.warning(f"  ✗ File not found: coinalyze_liquidations.csv")
        
        # Load coinalyze_long_short_ratio.csv
        try:
            csv_path = self.raw_dir / "coinalyze_long_short_ratio.csv"
            lsr_df = pd.read_csv(csv_path)
            lsr_df['timestamp'] = pd.to_datetime(lsr_df['timestamp'], utc=True)
            lsr_df = lsr_df[lsr_df['asset'] == self.asset].copy()
            lsr_df = lsr_df[['timestamp', 'ratio', 'long_percentage', 'short_percentage']].copy()
            lsr_df.columns = ['timestamp', 'long_short_ratio', 'long_percentage', 'short_percentage']
            lsr_df = lsr_df.set_index('timestamp')
            self.df = self.df.join(lsr_df, how='left')
            logger.info(f"  ✓ Loaded long/short ratio ({len(lsr_df)} records for {self.asset})")
        except FileNotFoundError:
            logger.warning(f"  ✗ File not found: coinalyze_long_short_ratio.csv")
        
        # Load gdelt_macro.csv.csv (note: double .csv in filename)
        try:
            csv_path = self.raw_dir / "gdelt_macro.csv.csv"
            gdelt_df = pd.read_csv(csv_path)
            gdelt_df['timestamp'] = pd.to_datetime(gdelt_df['timestamp'], utc=True)
            gdelt_df = gdelt_df[['timestamp', 'news_volume', 'avg_sentiment_tone']].copy()
            gdelt_df = gdelt_df.set_index('timestamp')
            # Fill missing values with 0
            gdelt_df['news_volume'] = gdelt_df['news_volume'].fillna(0).astype(int)
            gdelt_df['avg_sentiment_tone'] = gdelt_df['avg_sentiment_tone'].fillna(0)
            self.df = self.df.join(gdelt_df, how='left')
            logger.info(f"  ✓ Loaded GDELT macro sentiment ({len(gdelt_df)} records)")
        except FileNotFoundError:
            logger.warning(f"  ✗ File not found: gdelt_macro.csv.csv")
        
        logger.info(f"  → After hourly merges: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
    
    def _load_lower_frequency_data(self) -> None:
        """Load and merge lower-frequency datasets using forward-fill (no bfill)."""
        logger.info(f"\nLoading lower-frequency datasets (forward-fill, no bfill)...")
        
        # Load funding rates (8-hour frequency)
        try:
            csv_path = self.raw_dir / f"{self.asset}USDT_fundingRate.csv"
            fr_df = pd.read_csv(csv_path)
            # Convert Unix milliseconds to datetime
            fr_df['timestamp'] = pd.to_datetime(fr_df['calc_time'], unit='ms', utc=True)
            fr_df = fr_df[['timestamp', 'last_funding_rate']].copy()
            fr_df.columns = ['timestamp', 'funding_rate']
            fr_df = fr_df.set_index('timestamp')
            fr_df = fr_df.sort_index()
            
            # Merge using merge_asof to forward-fill funding rates
            self.df = self.df.reset_index()
            fr_df = fr_df.reset_index()
            self.df = pd.merge_asof(
                self.df, fr_df,
                on='timestamp',
                direction='backward'  # Forward-fill: take last known rate
            )
            self.df = self.df.set_index('timestamp')
            
            logger.info(f"  ✓ Loaded funding rates ({len(fr_df)} records for {self.asset})")
        except FileNotFoundError:
            logger.warning(f"  ✗ File not found: {self.asset}USDT_fundingRate.csv")
        
        # Load Fear & Greed Index (daily frequency)
        try:
            csv_path = self.raw_dir / "fear_greed_index.csv"
            fg_df = pd.read_csv(csv_path)
            fg_df['timestamp'] = pd.to_datetime(fg_df['datetime'], utc=True)
            # Filter to data range
            data_start = self.df.index.min()
            data_end = self.df.index.max()
            fg_df = fg_df[(fg_df['timestamp'] >= data_start) & (fg_df['timestamp'] <= data_end)]
            fg_df = fg_df[['timestamp', 'value', 'value_classification']].copy()
            fg_df.columns = ['timestamp', 'fear_greed_value', 'fear_greed_classification']
            fg_df = fg_df.set_index('timestamp')
            fg_df = fg_df.sort_index()
            
            # Merge using merge_asof for daily data
            self.df = self.df.reset_index()
            fg_df = fg_df.reset_index()
            self.df = pd.merge_asof(
                self.df, fg_df,
                on='timestamp',
                direction='backward'  # Forward-fill: take last known fear/greed value
            )
            self.df = self.df.set_index('timestamp')
            
            logger.info(f"  ✓ Loaded fear/greed index ({len(fg_df)} records)")
        except FileNotFoundError:
            logger.warning(f"  ✗ File not found: fear_greed_index.csv")
        
        logger.info(f"  → After lower-frequency merges: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
    
    def _load_text_data(self) -> None:
        """Load and aggregate text data from reddit_posts.csv by hour."""
        logger.info(f"\nProcessing text data (aggregating reddit_posts by hour)...")
        
        try:
            csv_path = self.raw_dir / "reddit_posts.csv"
            # Note: reddit_posts.csv is large; read only necessary columns
            text_df = pd.read_csv(
                csv_path,
                usecols=['timestamp', 'combined_text', 'assets'],
                dtype={'combined_text': str, 'assets': str}
            )
            
            # Floor timestamps to hour
            text_df['timestamp'] = pd.to_datetime(text_df['timestamp'], utc=True)
            text_df['hour'] = text_df['timestamp'].dt.floor('H')
            
            # Group by hour and concatenate combined_text with [SEP] separator
            text_agg = text_df.groupby('hour')['combined_text'].apply(
                lambda x: ' [SEP] '.join(x.dropna().astype(str))
            ).reset_index()
            text_agg.columns = ['timestamp', 'text_content']
            text_agg = text_agg.set_index('timestamp')
            
            # Join to main dataframe
            self.df = self.df.join(text_agg, how='left')
            
            # Fill missing text with placeholder
            self.df['text_content'] = self.df['text_content'].fillna('[NO_EVENT] market is quiet')
            
            logger.info(f"  ✓ Loaded and aggregated text data ({len(text_agg)} unique hours)")
            logger.info(f"  ✓ Filled {(self.df['text_content'] == '[NO_EVENT] market is quiet').sum()} hours with placeholder")
        except FileNotFoundError:
            logger.warning(f"  ✗ File not found: reddit_posts.csv")
            self.df['text_content'] = '[NO_EVENT] market is quiet'
        
        logger.info(f"  → Final dataset after text merge: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
    
    # ============================================================================
    # PHASE 2: CONTINUOUS TARGET CALCULATION (Volatility-Adjusted Tanh)
    # ============================================================================
    
    def calculate_continuous_target(self) -> None:
        """
        Calculate continuous sentiment regression target using Volatility-Adjusted Tanh.
        
        Formula:
            R = (Close_future - Close_current) / Close_current
            sigma = rolling_std(hourly_returns, window=168)
            y = tanh(R / (k * sigma)) * 100, where k=1.5
        
        Safely handles edge cases:
        - When sigma ≈ 0: Uses epsilon (1e-5) to avoid division by zero
        - NaN targets: Drops rows at end of dataset where future close is unavailable
        """
        logger.info("=" * 80)
        logger.info("PHASE 2: Calculating continuous sentiment target")
        logger.info("=" * 80)
        
        initial_len = len(self.df)
        
        # Ensure 'close' column exists
        if 'close' not in self.df.columns:
            raise ValueError("'close' column not found in dataset")
        
        # Calculate hourly returns
        returns = self.df['close'].pct_change()
        logger.info(f"  ✓ Calculated hourly returns")
        
        # Calculate historical volatility (7-day rolling window = 168 hours)
        volatility = returns.rolling(window=168, min_periods=1).std()
        logger.info(f"  ✓ Calculated rolling volatility (window=168 hours)")
        
        # Calculate future returns (look ahead by horizon_hours)
        future_close = self.df['close'].shift(-self.horizon_hours)
        future_returns = (future_close - self.df['close']) / self.df['close']
        logger.info(f"  ✓ Calculated future returns (horizon={self.horizon_hours} hours)")
        
        # Calculate sentiment score using Volatility-Adjusted Tanh
        k = 1.5
        epsilon = 1e-5
        
        # Avoid division by zero: replace sigma near 0 with epsilon
        volatility_safe = volatility.copy()
        volatility_safe = volatility_safe.replace(0, epsilon)
        volatility_safe = volatility_safe.fillna(epsilon)
        
        # Calculate raw score
        raw_score = future_returns / (k * volatility_safe)
        sentiment_score = np.tanh(raw_score) * 100
        
        self.df['sentiment_score'] = sentiment_score
        
        logger.info(f"  ✓ Calculated sentiment scores: min={sentiment_score.min():.2f}, "
                   f"max={sentiment_score.max():.2f}, mean={sentiment_score.mean():.2f}")
        
        # Drop rows with NaN targets (at the end of the dataset)
        rows_before_drop = len(self.df)
        self.df = self.df.dropna(subset=['sentiment_score'])
        rows_dropped = rows_before_drop - len(self.df)
        
        logger.info(f"  ✓ Dropped {rows_dropped} rows with NaN targets (end of dataset)")
        logger.info(f"  → Dataset after target calculation: {len(self.df)} rows "
                   f"(dropped {rows_dropped} due to future unavailability)")
    
    # ============================================================================
    # PHASE 3: IMAGE MAPPING & DISK VALIDATION
    # ============================================================================
    
    def map_and_validate_images(self) -> None:
        """
        Map image paths and validate that physical files exist on disk.
        
        Image path format: data/processed/images/{asset_lower}/{int(unix_timestamp)}.png
        
        CRITICAL: Drops any rows where the image file does not exist on disk.
        This ensures 100% of final rows have corresponding images.
        """
        logger.info("=" * 80)
        logger.info("PHASE 3: Mapping and validating chart images")
        logger.info("=" * 80)
        
        asset_lower = self.asset.lower()
        asset_image_dir = self.image_dir / asset_lower
        
        logger.info(f"  Looking for images in: {asset_image_dir}")
        
        # Create image_path column with Unix timestamp format
        def get_image_path(timestamp):
            unix_ts = int(timestamp.timestamp())
            return str(asset_image_dir / f"{unix_ts}.png")
        
        self.df['image_path'] = self.df.index.map(get_image_path)
        logger.info(f"  ✓ Created image_path column for {len(self.df)} rows")
        
        # Validate image files exist on disk
        def validate_image(path):
            return os.path.exists(path)
        
        self.df['image_exists'] = self.df['image_path'].apply(validate_image)
        
        images_found = self.df['image_exists'].sum()
        images_missing = (~self.df['image_exists']).sum()
        
        logger.info(f"  ✓ Validation complete:")
        logger.info(f"    - Images found: {images_found}")
        logger.info(f"    - Images missing: {images_missing}")
        
        # Drop rows without images
        if images_missing > 0:
            logger.warning(f"  ⚠ Dropping {images_missing} rows due to missing image files")
            self.df = self.df[self.df['image_exists']].copy()
            self.df = self.df.drop(columns=['image_exists'])
            logger.info(f"  → Dataset after image validation: {len(self.df)} rows")
        else:
            self.df = self.df.drop(columns=['image_exists'])
            logger.info(f"  → All rows have corresponding images. Dataset: {len(self.df)} rows")
    
    # ============================================================================
    # PHASE 4: FINAL DATASET ASSEMBLY
    # ============================================================================
    
    def assemble_final_dataset(self) -> Tuple[Dataset, pd.DataFrame]:
        """
        Assemble final dataset with selected columns in proper order.
        
        Returns:
            Tuple of (HuggingFace Dataset, pandas DataFrame with all rows)
        """
        logger.info("=" * 80)
        logger.info("PHASE 4: Assembling final dataset")
        logger.info("=" * 80)
        
        # Define final columns in order
        final_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'funding_rate',
            'open_interest_open', 'open_interest_high', 'open_interest_low', 'open_interest_close',
            'long_liquidations', 'short_liquidations',
            'long_short_ratio', 'long_percentage', 'short_percentage',
            'news_volume', 'avg_sentiment_tone',
            'fear_greed_value', 'fear_greed_classification',
            'text_content',
            'sentiment_score',
            'image_path'
        ]
        
        # Check which columns exist
        existing_cols = [col for col in final_columns if col in self.df.columns]
        missing_cols = [col for col in final_columns if col not in self.df.columns]
        
        if missing_cols:
            logger.warning(f"  ⚠ Missing columns (will be skipped): {missing_cols}")
        
        # Add asset column
        df_final = self.df[existing_cols].copy()
        df_final['asset'] = self.asset
        
        # Reset index to make timestamp a column
        df_final = df_final.reset_index()
        
        # Reorder columns: timestamp and asset first
        cols = ['timestamp', 'asset'] + [col for col in df_final.columns 
                                         if col not in ['timestamp', 'asset']]
        df_final = df_final[cols]
        
        logger.info(f"  ✓ Final dataset shape: {df_final.shape}")
        logger.info(f"  ✓ Columns ({len(cols)}): {cols}")
        logger.info(f"\nSample row:")
        logger.info(f"  Timestamp: {df_final.iloc[0]['timestamp']}")
        logger.info(f"  Asset: {df_final.iloc[0]['asset']}")
        logger.info(f"  Close Price: ${df_final.iloc[0]['close']:.2f}")
        logger.info(f"  Sentiment Score: {df_final.iloc[0]['sentiment_score']:.2f}")
        logger.info(f"  Text Preview: {df_final.iloc[0]['text_content'][:80]}...")
        logger.info(f"  Image Path: {df_final.iloc[0]['image_path']}")
        
        # Convert to HF Dataset
        logger.info(f"\n  Converting to Hugging Face Dataset...")
        hf_dataset = Dataset.from_pandas(df_final)
        logger.info(f"  ✓ HF Dataset created with {len(hf_dataset)} rows")
        
        return hf_dataset, df_final
    
    # ============================================================================
    # PHASE 5: HUGGING FACE HUB INTEGRATION
    # ============================================================================
    
    def cast_images_and_push_to_hub(
        self,
        hf_dataset: Dataset,
        repo_id: str = "khanh252004/multimodal_crypto_sentiment",
        private: bool = False,
        dry_run: bool = False
    ) -> None:
        """
        Cast image_path column to Image type and push dataset to Hugging Face Hub.
        
        Args:
            hf_dataset: Hugging Face Dataset object
            repo_id: Target repository ID on HF Hub
            private: Whether to push as private dataset
            dry_run: If True, skip actual push (useful for testing)
        
        Raises:
            EnvironmentError: If HF_TOKEN not set in environment
        """
        logger.info("=" * 80)
        logger.info("PHASE 5: Hugging Face Hub Integration")
        logger.info("=" * 80)
        
        # Check for HF_TOKEN
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            logger.error("  ✗ HF_TOKEN environment variable not set")
            logger.error("  Please set HF_TOKEN before pushing to Hub:")
            logger.error("    export HF_TOKEN='your_token_here'")
            raise EnvironmentError(
                "HF_TOKEN not found in environment. "
                "Set it with: export HF_TOKEN='your_token_here'"
            )
        
        logger.info(f"  ✓ HF_TOKEN found in environment")
        
        # Cast image column
        logger.info(f"  Casting image_path column to Image type...")
        try:
            hf_dataset = hf_dataset.cast_column('image_path', Image())
            logger.info(f"  ✓ Image column cast successfully")
            
            # Verify by loading first image
            sample_image = hf_dataset[0]['image_path']
            logger.info(f"  ✓ Sample image loaded: shape={sample_image.size}")
        except Exception as e:
            logger.error(f"  ✗ Failed to cast image column: {str(e)}")
            raise
        
        # Push to Hub
        if dry_run:
            logger.info(f"  [DRY RUN] Would push to: {repo_id}")
            logger.info(f"  [DRY RUN] Private: {private}")
            logger.info(f"  [DRY RUN] Rows: {len(hf_dataset)}")
        else:
            logger.info(f"  Pushing to Hub: {repo_id}")
            try:
                hf_dataset.push_to_hub(
                    repo_id=repo_id,
                    private=private,
                    token=hf_token
                )
                logger.info(f"  ✓ Successfully pushed to {repo_id}")
                logger.info(f"  ✓ Dataset URL: https://huggingface.co/datasets/{repo_id}")
            except Exception as e:
                logger.error(f"  ✗ Failed to push to Hub: {str(e)}")
                raise
    
    # ============================================================================
    # MAIN ORCHESTRATION
    # ============================================================================
    
    def run(
        self,
        push_to_hub: bool = True,
        hub_repo_id: str = "khanh252004/multimodal_crypto_sentiment",
        hub_private: bool = False,
        hub_dry_run: bool = False
    ) -> pd.DataFrame:
        """
        Execute the complete data alignment pipeline.
        
        Orchestrates all phases:
        1. Load all 8 data sources
        2. Calculate continuous sentiment targets
        3. Map and validate chart images
        4. Assemble final dataset
        5. Push to Hugging Face Hub (optional)
        
        Args:
            push_to_hub: Whether to push final dataset to HF Hub
            hub_repo_id: Target repository on HF Hub
            hub_private: Whether dataset should be private
            hub_dry_run: If True, test push without actually uploading
        
        Returns:
            Final pandas DataFrame with all aligned data
        """
        logger.info("\n" + "=" * 80)
        logger.info("DataAligner: Starting complete pipeline")
        logger.info(f"Asset: {self.asset}")
        logger.info("=" * 80 + "\n")
        
        try:
            # Phase 1: Load all data
            self.load_all_data()
            
            # Phase 2: Calculate continuous targets
            self.calculate_continuous_target()
            
            # Phase 3: Map and validate images
            self.map_and_validate_images()
            
            # Phase 4: Assemble final dataset
            hf_dataset, df_final = self.assemble_final_dataset()
            
            # Phase 5: Push to Hub (optional)
            if push_to_hub:
                self.cast_images_and_push_to_hub(
                    hf_dataset,
                    repo_id=hub_repo_id,
                    private=hub_private,
                    dry_run=hub_dry_run
                )
            
            logger.info("\n" + "=" * 80)
            logger.info("✓ DataAligner pipeline completed successfully!")
            logger.info("=" * 80)
            
            return df_final
        
        except Exception as e:
            logger.error(f"\n✗ Pipeline failed with error: {str(e)}")
            raise


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Align multimodal cryptocurrency market sentiment data for deep learning"
    )
    parser.add_argument(
        "--asset",
        type=str,
        default="BTC",
        choices=["BTC", "ETH"],
        help="Cryptocurrency asset to process (default: BTC)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory containing raw/ and processed/ subdirs (default: data)"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/processed/images",
        help="Directory containing processed chart images (default: data/processed/images)"
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=24,
        help="Hours ahead for sentiment target calculation (default: 24)"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default="khanh252004/multimodal_crypto_sentiment",
        help="Target HF Hub repository ID (default: khanh252004/multimodal_crypto_sentiment)"
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Push dataset as private to HF Hub"
    )
    parser.add_argument(
        "--hub-dry-run",
        action="store_true",
        help="Test HF Hub push without actual upload"
    )
    
    args = parser.parse_args()
    
    # Initialize and run
    aligner = DataAligner(
        asset=args.asset,
        data_dir=args.data_dir,
        image_dir=args.image_dir,
        horizon_hours=args.horizon_hours
    )
    
    df_result = aligner.run(
        push_to_hub=not args.no_push,
        hub_repo_id=args.hub_repo_id,
        hub_private=args.hub_private,
        hub_dry_run=args.hub_dry_run
    )
    
    logger.info(f"\nFinal dataset shape: {df_result.shape}")
    logger.info(f"Columns: {list(df_result.columns)}")
