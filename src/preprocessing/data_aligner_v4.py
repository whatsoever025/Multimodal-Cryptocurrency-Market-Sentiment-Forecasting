"""
DataAligner v4: Multimodal Cryptocurrency Sentiment Dataset with Regime Features

Changes from v3:
  - NO pre-defined splits. The dataset is a single flat sequence (all rows in one
    split named "train"). Walk-forward splits are computed at training time by
    extract_features.py / create_walk_forward_dataloaders().
  - 2 new regime-indicator tabular features (9 total instead of 7):
      8. vol_30d  : 30-day realized volatility (rolling std of return_1h, 720h window)
      9. mom_30d  : 30-day price momentum     (rolling sum of return_1h, 720h window)
    Both are strictly causal (computed before the embargo, no look-ahead).

FINAL OUTPUT (12 Features + 1 Target per row):
  META (1):      timestamp
  TABULAR (9):   return_1h, volume, funding_rate, fear_greed_value,
                 gdelt_econ_volume, gdelt_econ_tone, gdelt_conflict_volume,
                 vol_30d, mom_30d
  TEXT (1):      text_content
  VISION (1):    image_path
  TARGET (1):    target_score

HF repos (v4):
  BTC: khanh252004/multimodal_crypto_sentiment_btc_v4
  ETH: khanh252004/multimodal_crypto_sentiment_eth_v4

Usage:
    python -m src.preprocessing.data_aligner_v4 --asset BTC
    python -m src.preprocessing.data_aligner_v4 --asset ETH
    python -m src.preprocessing.data_aligner_v4 --asset BTC --no-push  # local only
    python -m src.preprocessing.data_aligner_v4 --asset BTC --hub-dry-run
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from datasets import Dataset, Image
from huggingface_hub import HfApi

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataAlignerV4:
    """
    v4 DataAligner: flat single-split dataset with regime-indicator features.

    Key differences from v3:
    - No train/val/test split baked into the dataset.
      Walk-forward cross-validation splits are applied at training time.
    - 2 additional tabular features:
        vol_30d  = rolling 720h std(return_1h)     → realized volatility regime
        mom_30d  = rolling 720h sum(return_1h)     → momentum / trend regime
      Both are strictly causal (no look-ahead).
    """

    def __init__(
        self,
        asset: str = "BTC",
        data_dir: str = "data",
        image_dir: str = "data/processed/images",
        horizon_hours: int = 24,
        time_start: str = "2020-01-01",
        time_end: str = "2025-01-31",
    ):
        self.asset = asset.upper()
        if self.asset not in ("BTC", "ETH"):
            raise ValueError(f"asset must be 'BTC' or 'ETH', got {asset}")

        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.image_dir = Path(image_dir)
        self.horizon_hours = horizon_hours
        self.time_start = pd.to_datetime(time_start, utc=True)
        self.time_end = pd.to_datetime(time_end, utc=True)

        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir}")
        if not self.image_dir.exists():
            logger.warning(f"Image directory does not exist: {self.image_dir}")

        self.df: Optional[pd.DataFrame] = None
        logger.info(f"DataAlignerV4 initialized: asset={self.asset}, "
                    f"range={time_start}→{time_end}, horizon={horizon_hours}h")

    # =========================================================================
    # PHASE 1: DATA LOADING (identical to v3)
    # =========================================================================

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
        self._filter_time_range()
        logger.info(f"All data sources loaded. Base shape: {self.df.shape}")

    def _load_ohlcv(self) -> None:
        csv_path = self.raw_dir / f"{self.asset}USDT_klines.csv"
        logger.info(f"Loading OHLCV from {csv_path}")
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True).set_index('timestamp')
        logger.info(f"  ✓ {len(df)} OHLCV records: {df.index.min()} → {df.index.max()}")
        self.df = df

    def _load_funding_rate(self) -> None:
        logger.info("Loading funding rates (8-hour, forward-fill)...")
        try:
            csv_path = self.raw_dir / f"{self.asset}USDT_fundingRate.csv"
            fr = pd.read_csv(csv_path)
            fr['timestamp'] = pd.to_datetime(fr['calc_time'], unit='ms', utc=True)
            fr = fr[['timestamp', 'last_funding_rate']].rename(
                columns={'last_funding_rate': 'funding_rate'}
            ).set_index('timestamp').sort_index()
            self.df = self.df.reset_index()
            self.df = pd.merge_asof(self.df, fr.reset_index(), on='timestamp', direction='backward')
            self.df = self.df.set_index('timestamp')
            logger.info(f"  ✓ {len(fr)} funding rate records merged")
        except FileNotFoundError:
            logger.warning(f"  ✗ {self.asset}USDT_fundingRate.csv not found")
            self.df['funding_rate'] = np.nan

    def _load_fear_greed(self) -> None:
        logger.info("Loading Fear & Greed Index (daily, forward-fill)...")
        try:
            csv_path = self.raw_dir / "fear_greed_index.csv"
            fg = pd.read_csv(csv_path)
            fg['timestamp'] = pd.to_datetime(fg['datetime'], utc=True)
            fg = fg[['timestamp', 'value', 'value_classification']].rename(
                columns={'value': 'fear_greed_value', 'value_classification': 'fear_greed_classification'}
            ).set_index('timestamp').sort_index()
            self.df = self.df.reset_index()
            self.df = pd.merge_asof(self.df, fg.reset_index(), on='timestamp', direction='backward')
            self.df = self.df.set_index('timestamp')
            logger.info(f"  ✓ {len(fg)} Fear & Greed records merged")
        except FileNotFoundError:
            logger.warning("  ✗ fear_greed_index.csv not found")
            self.df['fear_greed_value'] = np.nan
            self.df['fear_greed_classification'] = 'Unknown'

    def _load_gdelt_exogenous(self) -> None:
        logger.info("Loading GDELT exogenous data...")
        try:
            csv_path = self.raw_dir / "gdelt_exogenous_data.csv"
            gdelt = pd.read_csv(csv_path)
            gdelt['timestamp'] = pd.to_datetime(gdelt['timestamp'], utc=True)
            gdelt = gdelt[['timestamp', 'gdelt_econ_volume', 'gdelt_econ_tone', 'gdelt_conflict_volume']].copy()
            gdelt = gdelt.set_index('timestamp')
            gdelt['gdelt_econ_volume'] = gdelt['gdelt_econ_volume'].fillna(0).astype(int)
            gdelt['gdelt_conflict_volume'] = gdelt['gdelt_conflict_volume'].fillna(0).astype(int)
            gdelt['gdelt_econ_tone'] = gdelt['gdelt_econ_tone'].fillna(0.0)
            self.df = self.df.join(gdelt, how='left')
            logger.info(f"  ✓ GDELT exogenous data merged")
        except FileNotFoundError:
            logger.warning("  ✗ gdelt_exogenous_data.csv not found")
            self.df['gdelt_econ_volume'] = 0
            self.df['gdelt_econ_tone'] = 0.0
            self.df['gdelt_conflict_volume'] = 0

    def _load_text_data(self) -> None:
        logger.info("Loading and aggregating CoinDesk news text...")
        try:
            csv_path = self.raw_dir / "huggingface_crypto_news.csv"
            cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
            use_cols = ['published_on', 'combined_text'] if 'combined_text' in cols else ['published_on', 'title', 'body']
            text_df = pd.read_csv(csv_path, usecols=use_cols)
            text_df['timestamp'] = pd.to_datetime(text_df['published_on'], utc=True)
            if 'combined_text' not in text_df.columns:
                text_df['combined_text'] = (
                    text_df.get('title', '').fillna('') + ' ' + text_df.get('body', '').fillna('')
                )
            text_df = text_df[['timestamp', 'combined_text']].copy()
            text_df.columns = ['timestamp', 'text_content']
            text_df['hour'] = text_df['timestamp'].dt.floor('h')
            agg = text_df.groupby('hour')['text_content'].apply(
                lambda x: ' [SEP] '.join(x.dropna().astype(str))
            ).reset_index()
            agg.columns = ['timestamp', 'text_content']
            agg = agg.set_index('timestamp')
            self.df = self.df.join(agg, how='left')
            self.df['text_content'] = self.df['text_content'].fillna('[NO_EVENT] market is quiet')
            logger.info(f"  ✓ {len(agg)} text hours aggregated")
        except FileNotFoundError:
            logger.warning("  ✗ huggingface_crypto_news.csv not found")
            self.df['text_content'] = '[NO_EVENT] market is quiet'
        except Exception as e:
            logger.warning(f"  ✗ Text load error: {e}")
            self.df['text_content'] = '[NO_EVENT] market is quiet'

    def _filter_time_range(self) -> None:
        before = len(self.df)
        self.df = self.df[(self.df.index >= self.time_start) & (self.df.index <= self.time_end)]
        logger.info(f"Time filter: {before} → {len(self.df)} rows")

    # =========================================================================
    # PHASE 2: CONTINUOUS TARGET CALCULATION (identical to v3)
    # =========================================================================

    def calculate_continuous_target(self) -> None:
        """
        Volatility-adjusted tanh target (same formula as v3).
        target_score = future_returns / (rolling_volatility + 1e-6)
        """
        logger.info("=" * 80)
        logger.info("PHASE 2: Calculating continuous target")
        logger.info("=" * 80)
        returns = self.df['close'].pct_change()
        volatility = returns.rolling(window=168, min_periods=1).std()
        future_close = self.df['close'].shift(-self.horizon_hours)
        future_returns = (future_close - self.df['close']) / self.df['close']
        self.df['target_score'] = future_returns / (volatility + 1e-6)
        before = len(self.df)
        self.df = self.df.dropna(subset=['target_score'])
        logger.info(f"  ✓ Target computed. Dropped {before - len(self.df)} NaN rows. "
                    f"Range: [{self.df['target_score'].min():.2f}, {self.df['target_score'].max():.2f}]")

    # =========================================================================
    # PHASE 3: IMAGE VALIDATION (identical to v3)
    # =========================================================================

    def map_and_validate_images(self) -> None:
        logger.info("=" * 80)
        logger.info("PHASE 3: Mapping and validating chart images")
        logger.info("=" * 80)
        asset_image_dir = self.image_dir / self.asset.lower()

        def get_image_path(ts):
            return str(asset_image_dir / f"{int(ts.timestamp())}.png")

        self.df['image_path'] = self.df.index.map(get_image_path)
        self.df['image_exists'] = self.df['image_path'].apply(os.path.exists)
        missing = (~self.df['image_exists']).sum()
        if missing > 0:
            logger.warning(f"  ⚠ Dropping {missing} rows with missing images")
            self.df = self.df[self.df['image_exists']].copy()
        self.df = self.df.drop(columns=['image_exists'])
        logger.info(f"  ✓ {len(self.df)} rows have valid images")

    # =========================================================================
    # PHASE 4 (NEW): REGIME FEATURE ENGINEERING
    # =========================================================================

    def compute_regime_features(self) -> None:
        """
        Compute 2 new strictly-causal regime indicator features:

        vol_30d  = rolling 720h std(return_1h)
            Realized volatility over the past 30 days.
            High vol_30d → uncertain / volatile regime (e.g. crash, halving)
            Low  vol_30d → stable trending regime
            The model can learn to down-weight sentiment signals when vol is extreme.

        mom_30d  = rolling 720h sum(return_1h)
            Cumulative 30-day price momentum.
            High mom_30d → institutional-driven bull run (e.g. ETF era)
            Negative     → bear market / capitulation
            Helps the model distinguish between retail and institutional regimes.

        Both are computed with min_periods=1 (expanding window at series start).
        Both are raw (NOT scaled) — rolling z-score normalization at training time
        will make them regime-agnostic.
        """
        logger.info("=" * 80)
        logger.info("PHASE 4 (NEW): Computing regime indicator features")
        logger.info("=" * 80)

        WINDOW = 720  # 30 days × 24 hours

        # Compute return_1h as % (needed for both final dataset and regime features)
        self.df['return_1h'] = self.df['close'].pct_change() * 100
        self.df['return_1h'] = self.df['return_1h'].fillna(0.0)
        
        returns = self.df['return_1h']

        # Strictly causal: .shift(1) so position t uses [t-720, t-1]
        self.df['vol_30d'] = (
            returns.rolling(window=WINDOW, min_periods=1)
            .std()
            .shift(1)
            .fillna(0.0)
        )

        self.df['mom_30d'] = (
            returns.rolling(window=WINDOW, min_periods=1)
            .sum()
            .shift(1)
            .fillna(0.0)
        )

        logger.info(f"  ✓ vol_30d: range [{self.df['vol_30d'].min():.4f}, {self.df['vol_30d'].max():.4f}]")
        logger.info(f"  ✓ mom_30d: range [{self.df['mom_30d'].min():.4f}, {self.df['mom_30d'].max():.4f}]")

    # =========================================================================
    # PHASE 5 (NEW): ASSEMBLE FLAT DATASET (no splits)
    # =========================================================================

    def assemble_flat_dataset(self) -> pd.DataFrame:
        """
        Assemble the final 12-column flat DataFrame (no train/val/test split).

        Columns (12 features + 1 target):
            timestamp, return_1h, volume, funding_rate, fear_greed_value,
            gdelt_econ_volume, gdelt_econ_tone, gdelt_conflict_volume,
            vol_30d, mom_30d, text_content, image_path, target_score

        Why no splits?
            The walk-forward validation splits are computed at training time
            by create_walk_forward_dataloaders() using split_metadata.json.
            Baking splits into HF datasets (v3 design) forced the extraction
            pipeline to re-concatenate them anyway (concatenate_datasets([train, val, test])).
            A single flat dataset is simpler and avoids that redundancy.
        """
        logger.info("=" * 80)
        logger.info("PHASE 5 (NEW): Assembling flat dataset (no splits)")
        logger.info("=" * 80)

        # Fill any remaining NaN in GDELT / funding
        for col in ['gdelt_econ_volume', 'gdelt_econ_tone', 'gdelt_conflict_volume']:
            self.df[col] = self.df[col].fillna(0)
        self.df['funding_rate'] = self.df['funding_rate'].fillna(0.0)
        self.df['fear_greed_value'] = self.df['fear_greed_value'].fillna(50)  # neutral default

        final_columns = [
            'return_1h',
            'volume',
            'funding_rate',
            'fear_greed_value',
            'gdelt_econ_volume',
            'gdelt_econ_tone',
            'gdelt_conflict_volume',
            'vol_30d',
            'mom_30d',
            'text_content',
            'image_path',
            'target_score',
        ]

        existing = [c for c in final_columns if c in self.df.columns]
        missing = [c for c in final_columns if c not in self.df.columns]
        if missing:
            logger.warning(f"  ⚠ Missing columns (skipped): {missing}")

        df_out = self.df[existing].copy().reset_index()  # timestamp becomes column
        df_out = df_out[['timestamp'] + existing]

        logger.info(f"  ✓ Flat dataset assembled: {df_out.shape}")
        logger.info(f"  ✓ Columns ({len(df_out.columns)}): {df_out.columns.tolist()}")
        logger.info(f"  ✓ Date range: {df_out['timestamp'].iloc[0]} → {df_out['timestamp'].iloc[-1]}")
        logger.info(f"  Sample row:")
        r = df_out.iloc[0]
        logger.info(f"    return_1h={r['return_1h']:.4f}  vol_30d={r['vol_30d']:.4f}  mom_30d={r['mom_30d']:.4f}")
        logger.info(f"    target_score={r['target_score']:.4f}")
        logger.info(f"    text_preview: {str(r['text_content'])[:80]}...")

        return df_out

    # =========================================================================
    # PHASE 6: CREATE HF DATASET (single split "train")
    # =========================================================================

    def create_hf_dataset(self, df: pd.DataFrame) -> Dataset:
        """
        Convert flat DataFrame to a single Hugging Face Dataset.

        The HF dataset uses split name "train" only (since there are no pre-defined splits).
        The extract_features.py pipeline loads this as:
            dataset = load_dataset(repo_id, split="train")
        and creates its own walk-forward boundaries at extraction time.
        """
        logger.info("=" * 80)
        logger.info("PHASE 6: Creating Hugging Face Dataset")
        logger.info("=" * 80)

        dataset = Dataset.from_pandas(df, preserve_index=False)
        logger.info(f"  ✓ Dataset created: {len(dataset)} rows, {len(dataset.column_names)} columns")

        # Cast image_path to Image type
        logger.info("  Casting image_path to Image type...")
        try:
            dataset = dataset.cast_column('image_path', Image())
            sample_img = dataset[0]['image_path']
            logger.info(f"  ✓ Image cast successful. Sample size: {sample_img.size}")
        except Exception as e:
            logger.error(f"  ✗ Failed to cast image column: {e}")
            raise

        return dataset

    # =========================================================================
    # PHASE 7: PUSH TO HF HUB
    # =========================================================================

    def push_to_hub(
        self,
        dataset: Dataset,
        repo_id: str,
        private: bool = False,
        dry_run: bool = False,
    ) -> None:
        logger.info("=" * 80)
        logger.info("PHASE 7: Pushing to Hugging Face Hub")
        logger.info("=" * 80)

        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            raise EnvironmentError(
                "HF_TOKEN not found in environment. "
                "Set it with: export HF_TOKEN='your_token_here'"
            )

        if dry_run:
            logger.info(f"  [DRY RUN] Would push {len(dataset)} rows to: {repo_id}")
            return

        logger.info(f"  Pushing {len(dataset)} rows to {repo_id}...")
        try:
            dataset.push_to_hub(
                repo_id=repo_id,
                split="train",          # single flat split
                private=private,
                token=hf_token,
                commit_message="v4: add vol_30d, mom_30d regime features; single flat split",
            )
            logger.info(f"  ✓ Pushed to https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            logger.error(f"  ✗ Push failed: {e}")
            raise

    # =========================================================================
    # MAIN ORCHESTRATION
    # =========================================================================

    def run(
        self,
        push_to_hub: bool = True,
        hub_repo_id: Optional[str] = None,
        hub_private: bool = False,
        hub_dry_run: bool = False,
    ) -> Dataset:
        """
        Execute full v4 pipeline:
          1. Load 5 data sources
          2. Calculate continuous target
          3. Validate images
          4. Compute regime features (vol_30d, mom_30d)  ← NEW
          5. Assemble flat dataset (no splits)           ← NEW
          6. Create HF Dataset
          7. Push to Hub (optional)
        """
        if hub_repo_id is None:
            hub_repo_id = f"khanh252004/multimodal_crypto_sentiment_{self.asset.lower()}_v4"

        logger.info("\n" + "=" * 80)
        logger.info("DataAlignerV4: Starting pipeline")
        logger.info(f"  Asset:    {self.asset}")
        logger.info(f"  Hub repo: {hub_repo_id}")
        logger.info("=" * 80 + "\n")

        try:
            self.load_all_data()                         # Phase 1
            self.calculate_continuous_target()           # Phase 2
            self.map_and_validate_images()               # Phase 3
            self.compute_regime_features()               # Phase 4 (new)
            df_flat = self.assemble_flat_dataset()       # Phase 5 (new)
            dataset = self.create_hf_dataset(df_flat)   # Phase 6
            if push_to_hub:
                self.push_to_hub(                        # Phase 7
                    dataset,
                    repo_id=hub_repo_id,
                    private=hub_private,
                    dry_run=hub_dry_run,
                )
            logger.info("\n" + "=" * 80)
            logger.info("✓ DataAlignerV4 pipeline complete!")
            logger.info("=" * 80)
            return dataset
        except Exception as e:
            logger.error(f"\n✗ Pipeline failed: {e}")
            raise


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DataAlignerV4: create v4 flat dataset with regime features"
    )
    parser.add_argument("--asset", choices=["BTC", "ETH"], default="BTC")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--image-dir", default="data/processed/images")
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--no-push", action="store_true", help="Skip HF Hub push")
    parser.add_argument("--hub-repo-id", default=None, help="Override default repo ID")
    parser.add_argument("--hub-private", action="store_true")
    parser.add_argument("--hub-dry-run", action="store_true",
                        help="Test without actual upload")
    parser.add_argument("--both-assets", action="store_true",
                        help="Run BTC then ETH sequentially")

    args = parser.parse_args()

    def run_asset(asset: str):
        aligner = DataAlignerV4(
            asset=asset,
            data_dir=args.data_dir,
            image_dir=args.image_dir,
            horizon_hours=args.horizon_hours,
        )
        dataset = aligner.run(
            push_to_hub=not args.no_push,
            hub_repo_id=args.hub_repo_id,
            hub_private=args.hub_private,
            hub_dry_run=args.hub_dry_run,
        )
        logger.info(f"\n{asset} dataset: {len(dataset)} rows, columns: {dataset.column_names}")
        return dataset

    if args.both_assets:
        logger.info("Running both BTC and ETH...")
        run_asset("BTC")
        run_asset("ETH")
    else:
        run_asset(args.asset)
