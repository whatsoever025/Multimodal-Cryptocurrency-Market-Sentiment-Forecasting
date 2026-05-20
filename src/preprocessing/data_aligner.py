"""
DataAligner v5: Multimodal Cryptocurrency Sentiment Dataset

ARCHITECTURAL CHANGES (v5):
  - DROPPED features: fear_greed_index, Open Interest, vol_30d, mom_30d
  - ADDED feature: is_post_ETF (binary, 1 if timestamp >= 2024-01-01)
  - KEPT tabular: return_1h, volume, funding_rate, gdelt_econ_volume,
                  gdelt_econ_tone, gdelt_conflict_volume
  - KEPT multimodal: text_content (FinBERT), image_path (ViT)

TARGET ENGINEERING (t vs t+1 — NO DATA LEAKAGE):
  Features at time t; targets are values observed at time t+1.

  Step-differences (computed at each row t):
    delta_funding   = funding_rate(t) - funding_rate(t-1)
    delta_tone      = gdelt_econ_tone(t) - gdelt_econ_tone(t-1)
    delta_conflict  = gdelt_conflict_volume(t) - gdelt_conflict_volume(t-1)

  Shift backward by 1 (.shift(-1)) so each row t carries what happens at t+1:
    target_delta_funding  -> delta_funding shifted to t
    target_return         -> return_1h shifted to t
    target_delta_tone     -> delta_tone shifted to t
    target_delta_conflict -> delta_conflict shifted to t

  Final targets:
    y_baseline  = target_delta_funding
    y_heuristic = weighted sum of Z-scored target variables
    y_pca       = PC1 of Z-scored target variables (sklearn PCA)

OUTPUT: Single DataFrame (no train/val/test split here).
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Hugging Face (optional — only needed for push_to_hub)
try:
    from datasets import Dataset, DatasetDict, Image as HFImage
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("huggingface_hub / datasets not installed — push_to_hub disabled.")


# ETF approval date (binary regime flag)
ETF_APPROVAL_DATE = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")


class DataAligner:
    """
    Aligns multimodal cryptocurrency market data into a single flat DataFrame.

    Data sources:
      1. Binance OHLCV (hourly): return_1h, volume
      2. Binance funding rates (8-hour, forward-filled): funding_rate
      3. GDELT exogenous (hourly): gdelt_econ_volume, gdelt_econ_tone, gdelt_conflict_volume
      4. CoinDesk news (hourly aggregated): text_content
      5. Candlestick chart images: image_path

    Features at time t (8 total):
      return_1h, volume, funding_rate,
      gdelt_econ_volume, gdelt_econ_tone, gdelt_conflict_volume,
      is_post_ETF,
      text_content, image_path

    Targets (3):
      y_baseline  — raw delta_funding at t+1
      y_heuristic — weighted Z-score composite at t+1
      y_pca       — PCA(1) of Z-scored target variables at t+1
    """

    def __init__(
        self,
        asset: str = "BTC",
        data_dir: str = "data",
        image_dir: str = "data/processed/images",
        time_start: str = "2020-01-01",
        time_end: str = "2025-01-31",
    ):
        self.asset = asset.upper()
        if self.asset not in ("BTC", "ETH"):
            raise ValueError(f"Invalid asset {asset}. Must be 'BTC' or 'ETH'.")

        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.image_dir = Path(image_dir)
        self.time_start = pd.to_datetime(time_start, utc=True)
        self.time_end = pd.to_datetime(time_end, utc=True)

        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir}")
        if not self.image_dir.exists():
            logger.warning(f"Image directory does not exist: {self.image_dir}")

        self.df: Optional[pd.DataFrame] = None

        logger.info(f"DataAligner v5 initialised for {self.asset}")
        logger.info(f"Data dir : {self.data_dir.absolute()}")
        logger.info(f"Image dir: {self.image_dir.absolute()}")
        logger.info(f"Time range: {self.time_start.date()} → {self.time_end.date()}")

    # =========================================================================
    # PHASE 1: DATA LOADING
    # =========================================================================

    def load_all_data(self) -> None:
        """Load all data sources and merge into a single hourly DataFrame."""
        logger.info("=" * 70)
        logger.info("PHASE 1: Loading data sources")
        logger.info("=" * 70)

        self._load_ohlcv()
        self._load_funding_rate()
        self._load_gdelt_exogenous()
        self._load_text_data()
        self._filter_time_range()

        logger.info(f"Base dataset shape after loading: {self.df.shape}")

    def _load_ohlcv(self) -> None:
        """Load Binance OHLCV klines → master hourly index."""
        csv_path = self.raw_dir / f"{self.asset}USDT_klines.csv"
        logger.info(f"Loading OHLCV from {csv_path}")

        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        self.df = df.set_index("timestamp")

        logger.info(
            f"  ✓ {len(self.df)} OHLCV rows  "
            f"({self.df.index.min()} → {self.df.index.max()})"
        )

    def _load_funding_rate(self) -> None:
        """Load Binance funding rates (8-hour cadence, forward-fill to hourly)."""
        logger.info("Loading funding rates (8-hour, forward-fill)...")
        try:
            csv_path = self.raw_dir / f"{self.asset}USDT_fundingRate.csv"
            fr = pd.read_csv(csv_path)
            fr["timestamp"] = pd.to_datetime(fr["calc_time"], unit="ms", utc=True)
            fr = fr[["timestamp", "last_funding_rate"]].rename(
                columns={"last_funding_rate": "funding_rate"}
            )
            fr = fr.sort_values("timestamp")

            self.df = pd.merge_asof(
                self.df.reset_index(),
                fr,
                on="timestamp",
                direction="backward",
            ).set_index("timestamp")

            logger.info(f"  ✓ {len(fr)} funding rate records merged")
        except FileNotFoundError:
            logger.warning(f"  ✗ {self.asset}USDT_fundingRate.csv not found")
            self.df["funding_rate"] = np.nan

    def _load_gdelt_exogenous(self) -> None:
        """Load GDELT macro indicators (economy + conflict, hourly)."""
        logger.info("Loading GDELT exogenous data...")
        try:
            csv_path = self.raw_dir / "gdelt_exogenous_data.csv"
            gd = pd.read_csv(csv_path)
            gd["timestamp"] = pd.to_datetime(gd["timestamp"], utc=True)
            gd = gd[
                ["timestamp", "gdelt_econ_volume", "gdelt_econ_tone", "gdelt_conflict_volume"]
            ].set_index("timestamp")

            gd["gdelt_econ_volume"] = gd["gdelt_econ_volume"].fillna(0).astype(int)
            gd["gdelt_conflict_volume"] = gd["gdelt_conflict_volume"].fillna(0).astype(int)
            gd["gdelt_econ_tone"] = gd["gdelt_econ_tone"].fillna(0.0)

            self.df = self.df.join(gd, how="left")
            logger.info(f"  ✓ {len(gd)} GDELT records joined")
        except FileNotFoundError:
            logger.warning("  ✗ gdelt_exogenous_data.csv not found")
            self.df["gdelt_econ_volume"] = 0
            self.df["gdelt_econ_tone"] = 0.0
            self.df["gdelt_conflict_volume"] = 0

    def _load_text_data(self) -> None:
        """Load CoinDesk news, aggregate by hour with [SEP] separator."""
        logger.info("Processing text data (hourly aggregation)...")
        try:
            csv_path = self.raw_dir / "huggingface_crypto_news.csv"
            cols_available = pd.read_csv(csv_path, nrows=0).columns.tolist()

            if "combined_text" in cols_available:
                text_df = pd.read_csv(csv_path, usecols=["published_on", "combined_text"])
            else:
                text_df = pd.read_csv(csv_path, usecols=["published_on", "title", "body"])
                text_df["combined_text"] = (
                    text_df["title"].fillna("") + " " + text_df["body"].fillna("")
                )

            text_df["timestamp"] = pd.to_datetime(text_df["published_on"], utc=True)
            text_df["hour"] = text_df["timestamp"].dt.floor("H")

            agg = (
                text_df.groupby("hour")["combined_text"]
                .apply(lambda x: " [SEP] ".join(x.dropna().astype(str)))
                .rename("text_content")
                .reset_index()
                .rename(columns={"hour": "timestamp"})
                .set_index("timestamp")
            )

            self.df = self.df.join(agg, how="left")
            self.df["text_content"] = self.df["text_content"].fillna(
                "[NO_EVENT] market is quiet"
            )
            logger.info(
                f"  ✓ Text data aggregated ({len(agg)} unique hours, "
                f"{(self.df['text_content'] == '[NO_EVENT] market is quiet').sum()} filled)"
            )
        except FileNotFoundError:
            logger.warning("  ✗ huggingface_crypto_news.csv not found")
            self.df["text_content"] = "[NO_EVENT] market is quiet"
        except Exception as exc:
            logger.warning(f"  ✗ Error loading text data: {exc}")
            self.df["text_content"] = "[NO_EVENT] market is quiet"

    def _filter_time_range(self) -> None:
        before = len(self.df)
        self.df = self.df[
            (self.df.index >= self.time_start) & (self.df.index <= self.time_end)
        ]
        logger.info(
            f"Time filter: {before} → {len(self.df)} rows "
            f"({self.time_start.date()} to {self.time_end.date()})"
        )

    # =========================================================================
    # PHASE 2: IMAGE MAPPING & VALIDATION
    # =========================================================================

    def map_and_validate_images(self) -> None:
        """Map candlestick image paths and drop rows where image files are missing."""
        logger.info("=" * 70)
        logger.info("PHASE 2: Mapping and validating chart images")
        logger.info("=" * 70)

        asset_lower = self.asset.lower()
        asset_image_dir = self.image_dir / asset_lower

        def _image_path(ts):
            return str(asset_image_dir / f"{int(ts.timestamp())}.png")

        self.df["image_path"] = self.df.index.map(_image_path)

        exists = self.df["image_path"].apply(os.path.exists)
        found, missing = exists.sum(), (~exists).sum()
        logger.info(f"  Images found: {found} | missing: {missing}")

        if missing > 0:
            logger.warning(f"  ⚠ Dropping {missing} rows (missing images)")
            self.df = self.df[exists].copy()

        logger.info(f"  → Dataset after image validation: {len(self.df)} rows")

    # =========================================================================
    # PHASE 3: FEATURE ENGINEERING
    # =========================================================================

    def engineer_features(self) -> None:
        """
        Build all input features at time t.

        TIME-LAG NOTE (no leakage):
          All features use only information available at time t.
          return_1h = pct_change of close, uses close(t) vs close(t-1) — safe.
          is_post_ETF = deterministic calendar flag — safe.
        """
        logger.info("=" * 70)
        logger.info("PHASE 3: Feature engineering (input X at time t)")
        logger.info("=" * 70)

        # return_1h: percentage change from previous hour close (uses t-1 → t)
        self.df["return_1h"] = self.df["close"].pct_change() * 100
        self.df["return_1h"] = self.df["return_1h"].fillna(0.0)

        # is_post_ETF: binary calendar flag (no look-ahead — purely deterministic)
        self.df["is_post_ETF"] = (self.df.index >= ETF_APPROVAL_DATE).astype(int)
        logger.info(
            f"  ✓ is_post_ETF: {self.df['is_post_ETF'].sum()} post-ETF rows "
            f"/ {(self.df['is_post_ETF'] == 0).sum()} pre-ETF rows"
        )

        # Fill any remaining NaN in GDELT columns
        for col in ["gdelt_econ_volume", "gdelt_econ_tone", "gdelt_conflict_volume"]:
            self.df[col] = self.df[col].fillna(0)

        # Fill any remaining NaN in funding_rate (forward-fill then zero)
        self.df["funding_rate"] = self.df["funding_rate"].ffill().fillna(0.0)

        logger.info("  ✓ Feature engineering complete")

    # =========================================================================
    # PHASE 4: TARGET ENGINEERING
    # =========================================================================

    def engineer_targets(self) -> None:
        """
        Build 3 sentiment-proxy target columns aligned to time t+1.

        TIME-LAG DESIGN (critical — prevents data leakage):
        ┌──────────────────────────────────────────────────────────────────┐
        │  Features (X) at time t  →  Targets (y) represent time t+1      │
        │                                                                  │
        │  Step 1: Compute step-differences AT each row t:                │
        │    delta_funding(t)  = funding_rate(t) - funding_rate(t-1)      │
        │    delta_tone(t)     = gdelt_econ_tone(t) - gdelt_econ_tone(t-1)│
        │    delta_conflict(t) = gdelt_conflict_volume(t) - ...(t-1)      │
        │                                                                  │
        │  Step 2: Shift ALL target variables BACKWARD by 1 (.shift(-1))  │
        │    This makes row t hold the NEXT hour's (t+1) signal.          │
        │    The model learns: given X(t), predict what happens at t+1.   │
        │                                                                  │
        │  Step 3: Drop first row (NaN from .shift(1)) and                │
        │          last row (NaN from .shift(-1)).                        │
        └──────────────────────────────────────────────────────────────────┘
        """
        logger.info("=" * 70)
        logger.info("PHASE 4: Target engineering (output y at time t+1)")
        logger.info("=" * 70)

        # ------------------------------------------------------------------
        # Step 1: Step-differences (computed at each time t using t-1 data)
        # ------------------------------------------------------------------
        delta_funding = self.df["funding_rate"].diff()        # funding_rate(t) - funding_rate(t-1)
        delta_tone    = self.df["gdelt_econ_tone"].diff()     # tone(t) - tone(t-1)
        delta_conflict = self.df["gdelt_conflict_volume"].diff()  # conflict(t) - conflict(t-1)

        # return_1h is already a 1-step difference (close pct_change)
        current_return = self.df["return_1h"]

        # ------------------------------------------------------------------
        # Step 2: Shift backward by 1 — row t now holds t+1's signal
        #   .shift(-1) means: value at index i comes from index i+1
        # ------------------------------------------------------------------
        target_delta_funding  = delta_funding.shift(-1)
        target_return         = current_return.shift(-1)
        target_delta_tone     = delta_tone.shift(-1)
        target_delta_conflict = delta_conflict.shift(-1)
        
        target_raw_funding    = self.df["funding_rate"].shift(-1)

        # Stack the four raw target variables into a DataFrame for scaling
        target_df = pd.DataFrame(
            {
                "target_delta_funding":  target_delta_funding,
                "target_return":         target_return,
                "target_delta_tone":     target_delta_tone,
                "target_delta_conflict": target_delta_conflict,
            },
            index=self.df.index,
        )

        # ------------------------------------------------------------------
        # Step 3: Drop rows with NaN targets
        #   - First row: NaN from .diff() on delta_funding / delta_tone / delta_conflict
        #   - Last row:  NaN from .shift(-1) — no t+1 observation available
        # ------------------------------------------------------------------
        rows_before = len(self.df)
        valid_mask = target_df.notna().all(axis=1) & target_raw_funding.notna()
        self.df = self.df[valid_mask].copy()
        target_df = target_df[valid_mask].copy()
        target_raw_funding = target_raw_funding[valid_mask].copy()
        rows_dropped = rows_before - len(self.df)
        logger.info(f"  Dropped {rows_dropped} rows with NaN targets (first/last rows)")

        # ------------------------------------------------------------------
        # TARGET 1: y_baseline — absolute funding_rate level at t+1 (Option A)
        # ------------------------------------------------------------------
        self.df["y_baseline"] = target_raw_funding.values
        logger.info("  ✓ y_baseline = target_raw_funding (absolute level at t+1)")

        # ------------------------------------------------------------------
        # TARGET 2: y_heuristic — weighted Z-score composite
        # ------------------------------------------------------------------
        scaler_h = StandardScaler()
        z_scaled = scaler_h.fit_transform(target_df.values)  # shape (N, 4)

        # Columns order: delta_funding, return, delta_tone, delta_conflict
        # Weights:       +0.4,          +0.3,   +0.2,       -0.1
        weights = np.array([0.4, 0.3, 0.2, -0.1])
        self.df["y_heuristic"] = z_scaled @ weights
        logger.info(
            "  ✓ y_heuristic = 0.4*Z(delta_funding) + 0.3*Z(return) "
            "+ 0.2*Z(delta_tone) - 0.1*Z(delta_conflict)"
        )

        # ------------------------------------------------------------------
        # TARGET 3: y_pca — first principal component of Z-scored variables
        # ------------------------------------------------------------------
        scaler_p = StandardScaler()
        z_for_pca = scaler_p.fit_transform(target_df.values)  # shape (N, 4)

        pca = PCA(n_components=1, random_state=42)
        pc1 = pca.fit_transform(z_for_pca)          # shape (N, 1)
        self.df["y_pca"] = pc1[:, 0]

        explained = pca.explained_variance_ratio_[0] * 100
        logger.info(
            f"  ✓ y_pca = PC1 of Z-scored target variables "
            f"(explains {explained:.1f}% variance)"
        )

        logger.info(f"  → Final dataset shape: {self.df.shape}")
        logger.info(
            f"  Target ranges:\n"
            f"    y_baseline  [{self.df['y_baseline'].min():.5f}, {self.df['y_baseline'].max():.5f}]\n"
            f"    y_heuristic [{self.df['y_heuristic'].min():.4f}, {self.df['y_heuristic'].max():.4f}]\n"
            f"    y_pca       [{self.df['y_pca'].min():.4f}, {self.df['y_pca'].max():.4f}]"
        )

    # =========================================================================
    # PHASE 5: FINAL ASSEMBLY
    # =========================================================================

    def assemble_final_dataset(self) -> pd.DataFrame:
        """
        Select final columns and return a single flat DataFrame.

        Column order:
          META:     timestamp
          TABULAR:  return_1h, volume, funding_rate,
                    gdelt_econ_volume, gdelt_econ_tone, gdelt_conflict_volume,
                    is_post_ETF
          MODAL:    text_content, image_path
          TARGETS:  y_baseline, y_heuristic, y_pca
        """
        logger.info("=" * 70)
        logger.info("PHASE 5: Assembling final dataset")
        logger.info("=" * 70)

        final_columns = [
            # Tabular features
            "return_1h",
            "volume",
            "funding_rate",
            "gdelt_econ_volume",
            "gdelt_econ_tone",
            "gdelt_conflict_volume",
            "is_post_ETF",
            # Multimodal
            "text_content",
            "image_path",
            # Targets
            "y_baseline",
            "y_heuristic",
            "y_pca",
        ]

        existing = [c for c in final_columns if c in self.df.columns]
        missing  = [c for c in final_columns if c not in self.df.columns]

        if missing:
            logger.warning(f"  ⚠ Missing columns (skipped): {missing}")

        df_out = self.df[existing].copy().reset_index()  # timestamp becomes column

        cols = ["timestamp"] + existing
        df_out = df_out[cols]

        logger.info(f"  ✓ Final shape: {df_out.shape}")
        logger.info(f"  ✓ Columns: {list(df_out.columns)}")
        logger.info(f"\n  Sample row [0]:")
        logger.info(f"    timestamp   : {df_out.iloc[0]['timestamp']}")
        logger.info(f"    return_1h   : {df_out.iloc[0]['return_1h']:.4f}%")
        logger.info(f"    funding_rate: {df_out.iloc[0]['funding_rate']:.6f}")
        logger.info(f"    is_post_ETF : {df_out.iloc[0]['is_post_ETF']}")
        logger.info(f"    y_baseline  : {df_out.iloc[0]['y_baseline']:.6f}")
        logger.info(f"    y_heuristic : {df_out.iloc[0]['y_heuristic']:.4f}")
        logger.info(f"    y_pca       : {df_out.iloc[0]['y_pca']:.4f}")

        return df_out

    # =========================================================================
    # PHASE 6 (OPTIONAL): PUSH TO HUGGING FACE HUB
    # =========================================================================

    def push_to_hub(
        self,
        df: pd.DataFrame,
        repo_id: str,
        private: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Push a single-split Dataset to Hugging Face Hub.

        Automatically purges existing parquet shards before pushing so that
        schema changes (v3 → v5) do not cause a feature-mismatch error.
        """
        if not HF_AVAILABLE:
            raise ImportError("Install 'datasets' and 'huggingface_hub' to use push_to_hub.")

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError(
                "HF_TOKEN environment variable not set. "
                "Set it with: export HF_TOKEN='your_token_here'"
            )

        if dry_run:
            logger.info(f"[DRY RUN] Would push {len(df)} rows to {repo_id}")
            return

        # --- Purge ALL repo content to avoid schema-mismatch on schema change --
        # The datasets library validates new parquet features against BOTH the
        # existing parquet shards AND the dataset_infos.json / README card.
        # We must delete everything before re-pushing with the new v5 schema.
        api = HfApi(token=hf_token)
        try:
            repo_files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
            # Exclude .gitattributes — HF requires it to remain
            deletable = [f for f in repo_files if f != ".gitattributes"]
            if deletable:
                logger.info(
                    f"  Purging {len(deletable)} file(s) from {repo_id} "
                    f"to apply v5 schema ..."
                )
                api.delete_files(
                    repo_id=repo_id,
                    repo_type="dataset",
                    delete_patterns=deletable,
                )
                logger.info("  ✓ Repo purged — pushing fresh v5 dataset")
        except Exception as e:
            logger.warning(f"  Could not purge repo (may be new): {e}")
        # -----------------------------------------------------------------------

        logger.info(f"Pushing {len(df)} rows to {repo_id} ...")
        ds = Dataset.from_pandas(df)

        if "image_path" in ds.column_names:
            ds = ds.cast_column("image_path", HFImage())

        ds.push_to_hub(repo_id=repo_id, private=private, token=hf_token)
        logger.info(f"  ✓ Dataset pushed → https://huggingface.co/datasets/{repo_id}")

    # =========================================================================
    # MAIN ORCHESTRATION
    # =========================================================================

    def run(
        self,
        push_to_hub: bool = False,
        hub_repo_id: Optional[str] = None,
        hub_private: bool = False,
        hub_dry_run: bool = False,
    ) -> pd.DataFrame:
        """
        Run the complete pipeline and return a single flat DataFrame.

        Phases:
          1. Load & merge all data sources
          2. Map + validate chart images
          3. Engineer input features (X at time t)
          4. Engineer targets (y at time t+1)
          5. Assemble final DataFrame
          6. (Optional) Push to Hugging Face Hub

        Returns:
            pd.DataFrame with all features and 3 target columns.
        """
        if hub_repo_id is None:
            hub_repo_id = f"khanh252004/multimodal_crypto_sentiment_{self.asset.lower()}"

        logger.info("\n" + "=" * 70)
        logger.info("DataAligner v5 — full pipeline")
        logger.info(f"Asset: {self.asset} | Hub: {hub_repo_id}")
        logger.info("=" * 70 + "\n")

        try:
            self.load_all_data()
            self.map_and_validate_images()
            self.engineer_features()
            self.engineer_targets()
            df_final = self.assemble_final_dataset()

            if push_to_hub:
                self.push_to_hub(
                    df_final,
                    repo_id=hub_repo_id,
                    private=hub_private,
                    dry_run=hub_dry_run,
                )

            logger.info("\n" + "=" * 70)
            logger.info("✓ Pipeline complete!")
            logger.info("=" * 70)
            return df_final

        except Exception as exc:
            logger.error(f"\n✗ Pipeline failed: {exc}")
            raise


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DataAligner v5 — Multimodal Crypto Sentiment Dataset"
    )
    parser.add_argument("--asset", default="BTC", choices=["BTC", "ETH"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--image-dir", default="data/processed/images")
    parser.add_argument("--time-start", default="2020-01-01")
    parser.add_argument("--time-end", default="2025-01-31")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo-id", default=None)
    parser.add_argument("--hub-private", action="store_true")
    parser.add_argument("--hub-dry-run", action="store_true")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="If set, save final DataFrame to this CSV path",
    )

    args = parser.parse_args()

    aligner = DataAligner(
        asset=args.asset,
        data_dir=args.data_dir,
        image_dir=args.image_dir,
        time_start=args.time_start,
        time_end=args.time_end,
    )

    df_result = aligner.run(
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
        hub_private=args.hub_private,
        hub_dry_run=args.hub_dry_run,
    )

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_result.to_csv(out_path, index=False)
        logger.info(f"Saved to {out_path}")

    logger.info(f"\nFinal DataFrame: {df_result.shape}")
    logger.info(f"Columns: {list(df_result.columns)}")
