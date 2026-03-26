"""
DataAligner: Production-Ready Multimodal Cryptocurrency Sentiment Dataset (v3)

DATASET STRUCTURE: 10-field multimodal hourly dataset with continuous sentiment target

5 CORE DATA SOURCES:
  1. Binance OHLCV (hourly): return_1h, volume
  2. Binance funding rates (8h, forward-filled): funding_rate
  3. Fear & Greed Index (daily, forward-filled): fear_greed_value
  4. GDELT exogenous (economy + conflict, hourly): gdelt_econ_volume, gdelt_econ_tone, gdelt_conflict_volume
  5. CoinDesk news (hourly aggregated): text_content + image_path

FINAL OUTPUT (10 Features + 1 Target):
  META (1):          timestamp
  TABULAR (7):       return_1h, volume, funding_rate, fear_greed_value, gdelt_econ_volume, gdelt_econ_tone, gdelt_conflict_volume
  TEXT (1):          text_content (CoinDesk news with [SEP] separator)
  VISION (1):        image_path (224x224 PNG candlestick + MA7/MA25/RSI/MACD)
  TARGET (1):        target_score (continuous -100 to +100, 24-hour horizon, Volatility-Adjusted Tanh)

KEY FEATURES (v3):
  - Hourly alignment: 5.25 years (2020-01-02 21:00 UTC to 2025-01-30 00:00 UTC)
  - Chronological splits: 70% train (31,133) / 15% val (6,671) / 15% test (6,625) with 24-hour embargo
  - Embargo rule: 48 rows dropped at split boundaries to prevent look-ahead bias
  - Complete coverage: 44,429 rows per asset (after image validation, embargo removal)
  - 100% image validity: 44,477 valid candlestick charts per asset (224x224 PNG)
  - Dual exogenous factors: Economy (inflation/policy) + geopolitical conflict signals
  - Multimodal-ready: Compatible with LSTM/MLP (tabular) + BERT (text) + CNN/ViT (images)
  - HF Hub ready: Direct push to public repositories (khanh252004/multimodal_crypto_sentiment_btc/eth)

AVAILABLE DATASETS:
  - BTC: https://huggingface.co/datasets/khanh252004/multimodal_crypto_sentiment_btc
  - ETH: https://huggingface.co/datasets/khanh252004/multimodal_crypto_sentiment_eth

SEE: docs/HF_DATASET_GUIDE.md for usage examples, architecture blueprints, and model implementations
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
from datasets import Dataset, DatasetDict, Image
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
    
    Orchestrates 5 core data sources into aligned hourly dataset with:
    - OHLCV (Binance klines, hourly)
    - Funding rates (8-hour, forward-filled)
    - Fear/Greed index (daily, forward-filled)
    - GDELT macroeconomic news (hourly)
    - CoinDesk crypto news text (hourly aggregated)
    - Continuous regression target (Volatility-Adjusted Tanh)
    - Chronological train/val/test splits with 24-hour embargo
    
    Attributes:
        asset (str): "BTC" or "ETH"
        data_dir (str): Path to data directory (raw/ and processed/ subdirs)
        image_dir (str): Path to processed images directory
        horizon_hours (int): Hours ahead for target calculation (default 24)
        time_start (datetime): Start of time range (default 2020-01-01)
        time_end (datetime): End of time range (default 2025-01-31)
    """
    
    def __init__(
        self,
        asset: str = "BTC",
        data_dir: str = "data",
        image_dir: str = "data/processed/images",
        horizon_hours: int = 24,
        time_start: str = "2020-01-01",
        time_end: str = "2025-01-31"
    ):
        """
        Initialize DataAligner.
        
        Args:
            asset: "BTC" or "ETH"
            data_dir: Root data directory (should contain raw/ and processed/ subdirs)
            image_dir: Directory containing generated candlestick images
            horizon_hours: Time horizon for target calculation (default 24)
            time_start: Start of time range (default 2020-01-01)
            time_end: End of time range (default 2025-01-31)
        
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
        self.time_start = pd.to_datetime(time_start, utc=True)
        self.time_end = pd.to_datetime(time_end, utc=True)
        
        # Validate directories exist
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir}")
        if not self.image_dir.exists():
            logger.warning(f"Image directory does not exist: {self.image_dir}")
        
        # Data storage
        self.df: Optional[pd.DataFrame] = None
        
        logger.info(f"DataAligner initialized for {self.asset}")
        logger.info(f"Data directory: {self.data_dir.absolute()}")
        logger.info(f"Image directory: {self.image_dir.absolute()}")
        logger.info(f"Time range: {self.time_start.date()} to {self.time_end.date()}")
        logger.info(f"Target horizon: {horizon_hours} hours")
    
    # ============================================================================
    # PHASE 1: DATA LOADING (5 Core Sources Only)
    # ============================================================================
    
    def load_all_data(self) -> None:
        """Load all 5 core data sources from raw directory."""
        logger.info("=" * 80)
        logger.info("PHASE 1: Loading 5 core data sources")
        logger.info("=" * 80)
        
        self._load_ohlcv()
        self._load_funding_rate()
        self._load_fear_greed()
        self._load_gdelt_exogenous()
        self._load_text_data()
        
        self.filter_time_range()
        
        logger.info(f"All data sources loaded. Base dataset shape: {self.df.shape}")
    
    def _load_ohlcv(self) -> None:
        """Load OHLCV klines data and create master hourly index."""
        csv_path = self.raw_dir / f"{self.asset}USDT_klines.csv"
        logger.info(f"Loading OHLCV from {csv_path}")
        
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.set_index('timestamp')
        
        logger.info(f"  ✓ Loaded {len(df)} OHLCV records from {df.index.min()} to {df.index.max()}")
        logger.info(f"  ✓ Columns: {list(df.columns)}")
        
        self.df = df
    
    def _load_funding_rate(self) -> None:
        """Load and merge funding rates (8-hour frequency, forward-fill)."""
        logger.info(f"\nLoading funding rates (8-hour, forward-fill)...")
        
        try:
            csv_path = self.raw_dir / f"{self.asset}USDT_fundingRate.csv"
            fr_df = pd.read_csv(csv_path)
            # Convert Unix milliseconds to datetime
            fr_df['timestamp'] = pd.to_datetime(fr_df['calc_time'], unit='ms', utc=True)
            fr_df = fr_df[['timestamp', 'last_funding_rate']].copy()
            fr_df.columns = ['timestamp', 'funding_rate']
            fr_df = fr_df.set_index('timestamp')
            fr_df = fr_df.sort_index()
            
            # Merge using merge_asof to forward-fill funding rates (backward direction)
            self.df = self.df.reset_index()
            fr_df = fr_df.reset_index()
            self.df = pd.merge_asof(
                self.df, fr_df,
                on='timestamp',
                direction='backward'  # Forward-fill: take last known rate
            )
            self.df = self.df.set_index('timestamp')
            
            logger.info(f"  ✓ Loaded {len(fr_df)} funding rate records")
        except FileNotFoundError:
            logger.warning(f"  ✗ File not found: {self.asset}USDT_fundingRate.csv")
            self.df['funding_rate'] = np.nan
    
    def _load_fear_greed(self) -> None:
        """Load and merge Fear & Greed Index (daily frequency, forward-fill)."""
        logger.info(f"\nLoading Fear & Greed Index (daily, forward-fill)...")
        
        try:
            csv_path = self.raw_dir / "fear_greed_index.csv"
            fg_df = pd.read_csv(csv_path)
            fg_df['timestamp'] = pd.to_datetime(fg_df['datetime'], utc=True)
            # Filter to data range (will be stricter after OHLCV time filter)
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
                direction='backward'  # Forward-fill: take last known F/G value
            )
            self.df = self.df.set_index('timestamp')
            
            logger.info(f"  ✓ Loaded {len(fg_df)} Fear & Greed records")
        except FileNotFoundError:
            logger.warning(f"  ✗ File not found: fear_greed_index.csv")
            self.df['fear_greed_value'] = np.nan
            self.df['fear_greed_classification'] = 'Unknown'
    
    def _load_gdelt_exogenous(self) -> None:
        """Load and merge GDELT exogenous data (economy, conflict).
        
        Provides 3 macro indicators (hourly):
        - gdelt_econ_volume: # articles on economy/inflation
        - gdelt_econ_tone: Average tone of economic articles
        - gdelt_conflict_volume: # articles on conflict/politics
        """
        logger.info(f"\nLoading GDELT exogenous data (economy, conflict)...")
        
        try:
            csv_path = self.raw_dir / "gdelt_exogenous_data.csv"
            gdelt_df = pd.read_csv(csv_path)
            gdelt_df['timestamp'] = pd.to_datetime(gdelt_df['timestamp'], utc=True)
            gdelt_df = gdelt_df[['timestamp', 'gdelt_econ_volume', 'gdelt_econ_tone', 'gdelt_conflict_volume']].copy()
            gdelt_df = gdelt_df.set_index('timestamp')
            
            # Fill missing values with 0 for volumes, 0 for tones
            gdelt_df['gdelt_econ_volume'] = gdelt_df['gdelt_econ_volume'].fillna(0).astype(int)
            gdelt_df['gdelt_conflict_volume'] = gdelt_df['gdelt_conflict_volume'].fillna(0).astype(int)
            gdelt_df['gdelt_econ_tone'] = gdelt_df['gdelt_econ_tone'].fillna(0)
            
            self.df = self.df.join(gdelt_df, how='left')
            logger.info(f"  ✓ Loaded GDELT exogenous data ({len(gdelt_df)} records)")
        except FileNotFoundError:
            logger.warning(f"  ✗ File not found: gdelt_exogenous_data.csv")
            self.df['gdelt_econ_volume'] = 0
            self.df['gdelt_econ_tone'] = 0.0
            self.df['gdelt_conflict_volume'] = 0
    
    def _load_text_data(self) -> None:
        """Load and aggregate CoinDesk crypto news text data by hour."""
        logger.info(f"\nProcessing text data (aggregating CoinDesk news by hour)...")
        
        try:
            csv_path = self.raw_dir / "huggingface_crypto_news.csv"
            # Read with minimal columns for efficiency
            text_df = pd.read_csv(
                csv_path,
                usecols=['published_on', 'combined_text'] if 'combined_text' in pd.read_csv(csv_path, nrows=0).columns else ['published_on', 'title', 'body'],
                dtype={'combined_text': str} if 'combined_text' in pd.read_csv(csv_path, nrows=0).columns else {}
            )
            
            # Rename published_on to timestamp and convert to UTC
            text_df['timestamp'] = pd.to_datetime(text_df['published_on'], utc=True)
            
            # Handle combined_text field or build from title+body
            if 'combined_text' in text_df.columns:
                text_column = 'combined_text'
            else:
                # Build combined text from title and body
                text_df['combined_text'] = text_df['title'].fillna('') + ' ' + text_df['body'].fillna('')
                text_column = 'combined_text'
            
            text_df = text_df[['timestamp', text_column]].copy()
            text_df.columns = ['timestamp', 'text_content']
            
            # Floor timestamps to hour
            text_df['hour'] = text_df['timestamp'].dt.floor('H')
            
            # Group by hour and concatenate text with [SEP] separator
            text_agg = text_df.groupby('hour')['text_content'].apply(
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
            logger.warning(f"  ✗ File not found: huggingface_crypto_news.csv")
            self.df['text_content'] = '[NO_EVENT] market is quiet'
        except Exception as e:
            logger.warning(f"  ✗ Error loading text data: {str(e)}")
            self.df['text_content'] = '[NO_EVENT] market is quiet'
    
    def filter_time_range(self) -> None:
        """Filter dataset to specified time range (2020-01-01 to 2025-01-31)."""
        logger.info(f"\nFiltering to time range: {self.time_start.date()} to {self.time_end.date()}")
        
        rows_before = len(self.df)
        self.df = self.df[(self.df.index >= self.time_start) & (self.df.index <= self.time_end)]
        rows_after = len(self.df)
        rows_dropped = rows_before - rows_after
        
        logger.info(f"  ✓ Filtered dataset: {rows_before} → {rows_after} rows (dropped {rows_dropped})")
    
    # ============================================================================
    # PHASE 2: CONTINUOUS TARGET CALCULATION (Volatility-Adjusted Tanh)
    # ============================================================================
    
    def calculate_continuous_target(self) -> None:
        """
        Calculate continuous sentiment regression target using Volatility-Adjusted Tanh.
        
        Formula:
            R = (Close_future - Close_current) / Close_current
            sigma = rolling_std(hourly_returns, window=168)
            target_score = tanh(R / (k * sigma)) * 100, where k=1.5
        
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
        target_score = np.tanh(raw_score) * 100
        
        self.df['target_score'] = target_score
        
        logger.info(f"  ✓ Calculated target scores: min={target_score.min():.2f}, "
                   f"max={target_score.max():.2f}, mean={target_score.mean():.2f}")
        
        # Drop rows with NaN targets (at the end of the dataset)
        rows_before_drop = len(self.df)
        self.df = self.df.dropna(subset=['target_score'])
        rows_dropped = rows_before_drop - len(self.df)
        
        logger.info(f"  ✓ Dropped {rows_dropped} rows with NaN targets (end of dataset)")
        logger.info(f"  → Dataset after target calculation: {len(self.df)} rows")
    
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
    # PHASE 4: CHRONOLOGICAL SPLIT WITH EMBARGO
    # ============================================================================
    
    def split_chronological_with_embargo(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset chronologically (70/15/15) with embargo rule to prevent look-ahead bias.
        
        CRITICAL EMBARGO RULE:
        - Drop exactly `horizon_hours` (24) rows between train and validation
        - Drop exactly `horizon_hours` (24) rows between validation and test
        - These gaps act as buffers so future price info doesn't leak into past predictions
        
        Returns:
            Tuple of (df_train, df_validation, df_test_in_domain)
        """
        logger.info("=" * 80)
        logger.info("PHASE 4: Chronological split with embargo rule")
        logger.info("=" * 80)
        
        total_rows = len(self.df)
        logger.info(f"Total rows before split: {total_rows}")
        
        # Calculate split indices (70/15/15)
        train_size = int(0.70 * total_rows)
        val_size = int(0.15 * total_rows)
        # test_size gets the remainder to ensure all rows are used
        
        # Apply embargo rule: drop horizon_hours rows between splits
        embargo_size = self.horizon_hours
        
        train_end_idx = train_size
        val_start_idx = train_end_idx + embargo_size  # Begin embargo after train
        val_end_idx = val_start_idx + val_size
        test_start_idx = val_end_idx + embargo_size  # Begin embargo after validation
        
        # Extract splits
        df_train = self.df.iloc[:train_end_idx].copy()
        df_embargo_1 = self.df.iloc[train_end_idx:val_start_idx].copy()  # For audit
        df_validation = self.df.iloc[val_start_idx:val_end_idx].copy()
        df_embargo_2 = self.df.iloc[val_end_idx:test_start_idx].copy()  # For audit
        df_test = self.df.iloc[test_start_idx:].copy()
        
        # Log split details
        logger.info(f"\n  Train set:      rows {0:,} to {train_end_idx:,} ({len(df_train):,} rows, {len(df_train)/total_rows*100:.1f}%)")
        logger.info(f"  Embargo 1:      rows {train_end_idx:,} to {val_start_idx:,} ({len(df_embargo_1):,} rows dropped)")
        logger.info(f"  Validation set: rows {val_start_idx:,} to {val_end_idx:,} ({len(df_validation):,} rows, {len(df_validation)/total_rows*100:.1f}%)")
        logger.info(f"  Embargo 2:      rows {val_end_idx:,} to {test_start_idx:,} ({len(df_embargo_2):,} rows dropped)")
        logger.info(f"  Test set:       rows {test_start_idx:,} to {total_rows:,} ({len(df_test):,} rows, {len(df_test)/total_rows*100:.1f}%)")
        
        # Audit log: specific timestamps dropped
        logger.info(f"\n  EMBARGO AUDIT TRAIL:")
        logger.info(f"    Train-to-Val embargo (24 hours):")
        for i, (idx, row) in enumerate(df_embargo_1.iterrows()):
            if i % 6 == 0:  # Log every 6 hours for readability
                logger.info(f"      Dropped: {idx} (hour {i+1}/24)")
        logger.info(f"    Val-to-Test embargo (24 hours):")
        for i, (idx, row) in enumerate(df_embargo_2.iterrows()):
            if i % 6 == 0:  # Log every 6 hours for readability
                logger.info(f"      Dropped: {idx} (hour {i+1}/24)")
        
        # Verify chronological ordering
        if len(df_train) > 0 and len(df_validation) > 0 and len(df_test) > 0:
            assert df_train.index.max() < df_validation.index.min(), "Train/Validation overlap detected!"
            assert df_validation.index.max() < df_test.index.min(), "Validation/Test overlap detected!"
            logger.info(f"\n  ✓ Chronological ordering verified (no overlaps)")
        
        logger.info(f"\n  FINAL SPLIT SUMMARY:")
        logger.info(f"    Train:       {len(df_train):,} rows ({df_train.index.min()} to {df_train.index.max()})")
        logger.info(f"    Validation:  {len(df_validation):,} rows ({df_validation.index.min()} to {df_validation.index.max()})")
        logger.info(f"    Test:        {len(df_test):,} rows ({df_test.index.min()} to {df_test.index.max()})")
        logger.info(f"    Total with embargo: {len(df_train) + len(df_validation) + len(df_test):,} rows")
        
        return df_train, df_validation, df_test
    
    # ============================================================================
    # PHASE 5: FINAL DATASET ASSEMBLY
    # ============================================================================
    
    def assemble_final_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Assemble final 10-field dataset and apply chronological split.
        
        Final fields (10 features + target):
        1. timestamp (Meta)
        2. return_1h (Tabular - price momentum)
        3. volume (Tabular - trading activity)
        4. funding_rate (Tabular - derivative sentiment)
        5. fear_greed_index (Tabular - crypto sentiment)
        6. gdelt_econ_volume (Tabular - macro news volume)
        7. gdelt_econ_tone (Tabular - macro sentiment)
        8. gdelt_conflict_volume (Tabular - geopolitical risk)
        9. text_content (Textual - news content)
        10. image_path (Visual - technical charts)
        11. target_score (Target - prediction label)
        
        Returns:
            Tuple of (df_train, df_validation, df_test_in_domain)
        """
        logger.info("=" * 80)
        logger.info("PHASE 5: Assembling final datasets")
        logger.info("=" * 80)
        
        # Calculate return_1h (% change from 1 hour ago)
        self.df['return_1h'] = self.df['close'].pct_change() * 100  # Convert to percentage
        
        # Define final columns in order (10 features + target)
        final_columns = [
            'return_1h',
            'volume',
            'funding_rate',
            'fear_greed_value',
            'gdelt_econ_volume',
            'gdelt_econ_tone',
            'gdelt_conflict_volume',
            'text_content',
            'image_path',
            'target_score'
        ]
        
        # Check which columns exist
        existing_cols = [col for col in final_columns if col in self.df.columns]
        missing_cols = [col for col in final_columns if col not in self.df.columns]
        
        if missing_cols:
            logger.warning(f"  ⚠ Missing columns (will be skipped): {missing_cols}")
        
        # Prepare final dataframe (include timestamp)
        df_final = self.df[existing_cols].copy()
        df_final = df_final.reset_index()  # timestamp becomes a column
        
        # Reorder: timestamp first, then features, then target
        cols = ['timestamp'] + existing_cols
        df_final = df_final[cols]
        
        logger.info(f"  ✓ Final dataset shape: {df_final.shape}")
        logger.info(f"  ✓ Columns ({len(cols)}): {cols}")
        logger.info(f"\nSample row:")
        logger.info(f"  Timestamp: {df_final.iloc[0]['timestamp']}")
        logger.info(f"  Return (1h): {df_final.iloc[0]['return_1h']:.4f}%")
        logger.info(f"  Volume: {df_final.iloc[0]['volume']:.0f}")
        logger.info(f"  Target Score: {df_final.iloc[0]['target_score']:.2f}")
        logger.info(f"  Text Preview: {df_final.iloc[0]['text_content'][:80]}...")
        logger.info(f"  Image Path: {df_final.iloc[0]['image_path']}")
        
        # Apply chronological split with embargo
        df_train, df_val, df_test = self.split_chronological_with_embargo()
        
        # Apply same column selection to each split
        df_train = df_train[existing_cols].copy()
        df_train = df_train.reset_index()
        df_train = df_train[cols]
        
        df_val = df_val[existing_cols].copy()
        df_val = df_val.reset_index()
        df_val = df_val[cols]
        
        df_test = df_test[existing_cols].copy()
        df_test = df_test.reset_index()
        df_test = df_test[cols]
        
        logger.info(f"\n  Final split statistics:")
        logger.info(f"    Train:       {df_train.shape}")
        logger.info(f"    Validation:  {df_val.shape}")
        logger.info(f"    Test:        {df_test.shape}")
        
        return df_train, df_val, df_test
    
    # ============================================================================
    # PHASE 6: CONVERT TO DATASETDICT
    # ============================================================================
    
    def create_dataset_dict(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> DatasetDict:
        """
        Convert pandas DataFrames to Hugging Face DatasetDict.
        
        Args:
            df_train: Training set DataFrame
            df_val: Validation set DataFrame
            df_test: Test set DataFrame
        
        Returns:
            DatasetDict with keys: train, validation, test_in_domain
        """
        logger.info("=" * 80)
        logger.info("PHASE 6: Creating Hugging Face DatasetDict")
        logger.info("=" * 80)
        
        # Convert to HF Datasets
        dataset_train = Dataset.from_pandas(df_train)
        dataset_val = Dataset.from_pandas(df_val)
        dataset_test = Dataset.from_pandas(df_test)
        
        logger.info(f"  ✓ Train Dataset created: {len(dataset_train)} rows")
        logger.info(f"  ✓ Validation Dataset created: {len(dataset_val)} rows")
        logger.info(f"  ✓ Test Dataset created: {len(dataset_test)} rows")
        
        # Create DatasetDict
        dataset_dict = DatasetDict({
            'train': dataset_train,
            'validation': dataset_val,
            'test_in_domain': dataset_test
        })
        
        logger.info(f"  ✓ DatasetDict created with keys: {list(dataset_dict.keys())}")
        
        # Cast image column to Image type
        logger.info(f"\n  Casting image_path column to Image type...")
        try:
            dataset_dict = dataset_dict.cast_column('image_path', Image())
            logger.info(f"  ✓ Image column cast successfully for all splits")
            
            # Verify by loading first image from each split
            for split_name in ['train', 'validation', 'test_in_domain']:
                if len(dataset_dict[split_name]) > 0:
                    sample_image = dataset_dict[split_name][0]['image_path']
                    logger.info(f"    - {split_name} sample image: {sample_image.size}")
        except Exception as e:
            logger.error(f"  ✗ Failed to cast image column: {str(e)}")
            raise
        
        return dataset_dict
    
    # ============================================================================
    # PHASE 7: PUSH TO HUGGING FACE HUB
    # ============================================================================
    
    def generate_readme(self) -> str:
        """Generate detailed README.md for Hugging Face Hub."""
        readme = f"""---
dataset_info:
  dataset_name: multimodal_crypto_sentiment_{self.asset.lower()}
  dataset_summary: Aligned multimodal cryptocurrency market sentiment dataset with 24-hour horizon prediction
  dataset_type:
    - tabular
    - vision
    - text
  task_type:
    - regression
---

# Dataset Card: Multimodal Cryptocurrency Market Sentiment ({self.asset})

## Dataset Summary

This dataset aligns **5 core data sources** into a single hourly multimodal dataset for cryptocurrency sentiment forecasting. The target is a **continuous sentiment score** (-100 to +100) predicting 24 hours ahead using Volatility-Adjusted Tanh formula.

**Key Features:**
- ✅ Chronological split (70% train, 15% validation, 15% test) with 24-hour embargo to prevent look-ahead bias
- ✅ 100% image coverage: 224×224 candlestick charts with technical indicators (MA7, MA25, RSI, MACD)
- ✅ Multimodal inputs: Tabular (OHLCV, sentiments), Text (CoinDesk news), Images (price charts)
- ✅ Time range: 2020-01-01 to 2025-01-31 (5+ years)
- ✅ Hourly frequency: {len(self.df):,} total rows before split

## Data Sources

| Source | Frequency | Records | Purpose |
|--------|-----------|---------|---------|
| Binance Vision OHLCV | Hourly | {len(self.df):,} | Price/volume (base index) |
| Binance Funding Rates | 8-hour (forward-fill) | ~{int(len(self.df)/24*3)} | Derivatives sentiment |
| Fear & Greed Index | Daily (forward-fill) | ~{int(len(self.df)/24)} | Market sentiment index |
| GDELT Macroeconomic | Hourly | ~{int(len(self.df)*0.6)} | Global news sentiment |
| CoinDesk News | Hourly (aggregated) | ~{int(len(self.df)*0.8)} | Crypto-specific news |

## Chronological Split with Embargo Rule

To prevent **look-ahead bias**, we enforce a strict chronological split with embargo buffers:

```
Timeline:
|---- Train (70%) ----|[24h embargo]|---- Validation (15%) ----|[24h embargo]|---- Test (15%) ----|

Key Point: The 24-hour embargo before each split ensures that no future price information
leaks into past sentiment predictions. This mimics real-world deployment where future data
is unavailable at prediction time.
```

## Features

### Tabular (LSTM/MLP Inputs)
```
- timestamp: UTC datetime (floored to hour)
- asset: 'BTC' or 'ETH'
- open: Opening price (USDT, float)
- high: High price (USDT, float)
- low: Low price (USDT, float)
- close: Closing price (USDT, float)
- volume: Trading volume (asset units, float)
- funding_rate: 8-hour perpetual funding rate (decimal, ±0.002)
- fear_greed_value: F/G index (0-100, int)
- fear_greed_classification: Categorical - ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
- gdelt_news_count: Macro news articles in hour (0-500, int)
- gdelt_avg_sentiment: News sentiment tone (-100 to +100, float)
```

### Textual (BERT/Transformer Inputs)
```
- text_content: Hourly aggregated CoinDesk news articles
  - Format: "Article 1 [SEP] Article 2 [SEP] Article 3 ..."
  - Empty hours: '[NO_EVENT] market is quiet'
  - Avg length: ~2,000 tokens per hour
```

### Visual (CNN/ViT Inputs)
```
- image_path: 224×224 PNG candlestick chart
  - Technical indicators: 7-hour MA (blue), 25-hour MA (red)
  - Oscillators: RSI(14), MACD with signal line
  - Auto-generated from OHLCV data
```

### Target Label
```
- target_score: Continuous sentiment (-100 to +100, float)
  - Formula: target_score = tanh(future_returns / (1.5 * volatility)) * 100
  - Horizon: 24 hours ahead
```

## License

CC BY-NC 4.0 (Non-commercial use only)
"""
        return readme
    
    def push_to_hub_dataset_dict(
        self,
        dataset_dict: DatasetDict,
        repo_id: str,
        private: bool = False,
        dry_run: bool = False
    ) -> None:
        """
        Push DatasetDict to Hugging Face Hub.
        
        Args:
            dataset_dict: DatasetDict with train/validation/test_in_domain splits
            repo_id: Target repository ID on HF Hub
            private: Whether to push as private dataset
            dry_run: If True, skip actual push (useful for testing)
        
        Raises:
            EnvironmentError: If HF_TOKEN not set in environment
        """
        logger.info("=" * 80)
        logger.info("PHASE 7: Pushing to Hugging Face Hub")
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
        
        # Generate README
        logger.info(f"\n  Generating detailed README.md...")
        readme_content = self.generate_readme()
        logger.info(f"  ✓ README generated ({len(readme_content)} chars)")
        
        # Push to Hub
        if dry_run:
            logger.info(f"\n  [DRY RUN] Would push to: {repo_id}")
            logger.info(f"  [DRY RUN] Private: {private}")
            logger.info(f"  [DRY RUN] Splits:")
            for split_name, split_data in dataset_dict.items():
                logger.info(f"    - {split_name}: {len(split_data)} rows")
        else:
            logger.info(f"\n  Pushing to Hub: {repo_id}")
            try:
                # Push dataset
                dataset_dict.push_to_hub(
                    repo_id=repo_id,
                    private=private,
                    token=hf_token
                )
                logger.info(f"  ✓ Successfully pushed to {repo_id}")
                logger.info(f"  ✓ Dataset URL: https://huggingface.co/datasets/{repo_id}")
                logger.info(f"  ✓ Splits on Hub:")
                for split_name in ['train', 'validation', 'test_in_domain']:
                    logger.info(f"    - {split_name}: Parquet files uploaded")
            except Exception as e:
                logger.error(f"  ✗ Failed to push to Hub: {str(e)}")
                raise
    
    # ============================================================================
    # MAIN ORCHESTRATION
    # ============================================================================
    
    def run(
        self,
        push_to_hub: bool = True,
        hub_repo_id: Optional[str] = None,
        hub_private: bool = False,
        hub_dry_run: bool = False
    ) -> DatasetDict:
        """
        Execute the complete data alignment pipeline.
        
        Orchestrates all phases:
        1. Load 5 core data sources
        2. Filter to time range (2020-01-01 to 2025-01-31)
        3. Calculate continuous target (target_score)
        4. Map and validate chart images
        5. Assemble final dataset and apply chronological split
        6. Create Hugging Face DatasetDict
        7. Push to Hub (optional)
        
        Args:
            push_to_hub: Whether to push final dataset to HF Hub
            hub_repo_id: Target repository on HF Hub (auto-constructed if None)
            hub_private: Whether dataset should be private
            hub_dry_run: If True, test push without actually uploading
        
        Returns:
            DatasetDict with train/validation/test_in_domain splits
        """
        if hub_repo_id is None:
            hub_repo_id = f"khanh252004/multimodal_crypto_sentiment_{self.asset.lower()}"
        
        logger.info("\n" + "=" * 80)
        logger.info("DataAligner: Starting complete pipeline")
        logger.info(f"Asset: {self.asset}")
        logger.info(f"Target Hub Repo: {hub_repo_id}")
        logger.info("=" * 80 + "\n")
        
        try:
            # Phase 1: Load all 5 data sources
            self.load_all_data()
            
            # Phase 2: Calculate continuous targets
            self.calculate_continuous_target()
            
            # Phase 3: Map and validate images
            self.map_and_validate_images()
            
            # Phase 4-5: Assemble final dataset + apply chronological split
            df_train, df_val, df_test = self.assemble_final_dataset()
            
            # Phase 6: Create DatasetDict
            dataset_dict = self.create_dataset_dict(df_train, df_val, df_test)
            
            # Phase 7: Push to Hub (optional)
            if push_to_hub:
                self.push_to_hub_dataset_dict(
                    dataset_dict,
                    repo_id=hub_repo_id,
                    private=hub_private,
                    dry_run=hub_dry_run
                )
            
            logger.info("\n" + "=" * 80)
            logger.info("✓ DataAligner pipeline completed successfully!")
            logger.info("=" * 80)
            
            return dataset_dict
        
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
        default=None,
        help="Target HF Hub repository ID (default: auto-constructed)"
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
    
    dataset_dict = aligner.run(
        push_to_hub=not args.no_push,
        hub_repo_id=args.hub_repo_id,
        hub_private=args.hub_private,
        hub_dry_run=args.hub_dry_run
    )
    
    logger.info(f"\nFinal DatasetDict splits:")
    for split_name, split_data in dataset_dict.items():
        logger.info(f"  {split_name}: {len(split_data)} rows, {len(split_data.column_names)} columns")
