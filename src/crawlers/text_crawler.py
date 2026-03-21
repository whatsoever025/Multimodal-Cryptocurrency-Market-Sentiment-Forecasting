"""
Text Crawler for cryptocurrency news and social text collection.
Collects news from CryptoPanic and social posts from Reddit.
"""

import hashlib
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class TextCrawler:
    """
    Crawler for fetching crypto text data from news and social sources.
    """

    def __init__(self, base_path="data/raw", request_timeout=30):
        """
        Initialize TextCrawler and load environment variables.

        Args:
            base_path: Directory path for saving raw data files
            request_timeout: Timeout (seconds) for HTTP requests
        """
        self.project_root = Path(__file__).resolve().parents[2]
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.output_file = self.base_path / "crypto_text.csv"
        self.request_timeout = request_timeout

        # Load .env from project root.
        load_dotenv(self.project_root / ".env")

        self.reddit_user_agent = os.getenv("REDDIT_USER_AGENT")

        logger.info("TextCrawler initialized")
        logger.info(f"Output file: {self.output_file.absolute()}")

    @staticmethod
    def _align_timestamp_to_hour(dt):
        """Align timestamps to the nearest lower hour to match OHLCV granularity."""
        if dt is None:
            dt = datetime.utcnow()
        return pd.Timestamp(dt).floor("h")

    @staticmethod
    def _clean_text(text):
        """Remove URLs, emojis, and extra spaces from text."""
        if not text:
            return ""

        # Remove URLs.
        cleaned = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Remove non-ASCII characters (including most emojis).
        cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
        # Normalize whitespace.
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _text_hash(asset, text):
        """Generate a deterministic hash for deduplication."""
        payload = f"{asset.lower()}::{text.lower()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _default_user_agent():
        """Fallback User-Agent for public endpoints that reject default requests UA."""
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ThesisBot/1.0"


    def _fetch_reddit(self, post_limit=50):
        """
        Fetch posts from selected crypto subreddits using Reddit's public .json endpoint.

        Args:
            post_limit: Maximum number of posts to fetch per subreddit

        Returns:
            List of dictionaries with text records.
        """
        if not self.reddit_user_agent:
            logger.warning(
                "REDDIT_USER_AGENT missing. Skipping Reddit source. "
                "Set REDDIT_USER_AGENT in .env to avoid Reddit blocking requests."
            )
            return []

        records = []
        subreddits = ["CryptoCurrency", "Bitcoin", "Ethereum"]
        headers = {"User-Agent": self.reddit_user_agent or self._default_user_agent()}

        for subreddit_name in subreddits:
            fetched_for_subreddit = 0
            after = None

            try:
                while fetched_for_subreddit < post_limit:
                    page_size = min(100, post_limit - fetched_for_subreddit)
                    endpoint = f"https://www.reddit.com/r/{subreddit_name}.json"
                    params = {"limit": page_size}
                    if after:
                        params["after"] = after

                    response = requests.get(
                        endpoint,
                        params=params,
                        headers=headers,
                        timeout=self.request_timeout,
                    )
                    response.raise_for_status()

                    payload = response.json()
                    children = payload.get("data", {}).get("children", [])
                    if not children:
                        break

                    for post in children:
                        post_data = post.get("data", {})
                        title = post_data.get("title", "")
                        selftext = post_data.get("selftext", "")
                        combined_text = self._clean_text(f"{title} {selftext}".strip())
                        if not combined_text:
                            continue

                        created_utc = post_data.get("created_utc")
                        if created_utc is None:
                            aligned_ts = self._align_timestamp_to_hour(None)
                        else:
                            aligned_ts = self._align_timestamp_to_hour(
                                datetime.utcfromtimestamp(created_utc)
                            )

                        if subreddit_name.lower() == "bitcoin":
                            asset = "BTC"
                        elif subreddit_name.lower() == "ethereum":
                            asset = "ETH"
                        else:
                            asset = "BTC,ETH"

                        records.append(
                            {
                                "timestamp": aligned_ts,
                                "source": f"Reddit - r/{subreddit_name}",
                                "asset": asset,
                                "text": combined_text,
                                "user_sentiment": None,
                            }
                        )

                    fetched_for_subreddit += len(children)
                    after = payload.get("data", {}).get("after")
                    if not after:
                        break

                    # Aggressive Reddit throttling between pagination pages.
                    time.sleep(3)

                # Aggressive Reddit throttling between subreddit requests.
                time.sleep(3)

            except Exception as exc:
                logger.error(f"Reddit fetch failed for r/{subreddit_name}: {exc}")
                time.sleep(3)

        logger.info(f"Fetched {len(records)} text rows from Reddit")
        return records

    def _fetch_stocktwits(self):
        """
        Fetch social messages from StockTwits symbol streams.

        Returns:
            List of dictionaries with text records.
        """
        records = []
        symbols = [("BTC.X", "BTC"), ("ETH.X", "ETH")]
        headers = {"User-Agent": self.reddit_user_agent or self._default_user_agent()}

        for symbol, asset in symbols:
            endpoint = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
            try:
                response = requests.get(endpoint, headers=headers, timeout=self.request_timeout)
                response.raise_for_status()
                payload = response.json()

                for message in payload.get("messages", []):
                    body = self._clean_text(message.get("body", ""))
                    if not body:
                        continue

                    created_at = message.get("created_at")
                    parsed_dt = pd.to_datetime(created_at, utc=True, errors="coerce")
                    if pd.isna(parsed_dt):
                        aligned_ts = self._align_timestamp_to_hour(None)
                    else:
                        aligned_ts = self._align_timestamp_to_hour(parsed_dt.tz_convert(None))

                    entities = message.get("entities") or {}
                    sentiment = entities.get("sentiment") or {}
                    user_sentiment = sentiment.get("basic")

                    records.append(
                        {
                            "timestamp": aligned_ts,
                            "source": "StockTwits",
                            "asset": asset,
                            "text": body,
                            "user_sentiment": user_sentiment,
                        }
                    )

                # Conservative pacing between symbols.
                time.sleep(2)
            except Exception as exc:
                logger.error(f"StockTwits fetch failed for {symbol}: {exc}")
                time.sleep(2)

        logger.info(f"Fetched {len(records)} text rows from StockTwits")
        return records

    def _load_existing_hashes(self):
        """Load existing hashes from CSV to prevent duplicate inserts."""
        if not self.output_file.exists():
            return set()

        try:
            existing_df = pd.read_csv(self.output_file)
            if "text_hash" in existing_df.columns:
                return set(existing_df["text_hash"].dropna().astype(str).tolist())

            if {"asset", "text"}.issubset(existing_df.columns):
                rebuilt_hashes = {
                    self._text_hash(str(row["asset"]), str(row["text"]))
                    for _, row in existing_df.iterrows()
                }
                return rebuilt_hashes
        except Exception as exc:
            logger.warning(f"Could not read existing text dataset for dedupe: {exc}")

        return set()

    def save_records(self, records):
        """
        Save text records to CSV with hash-based deduplication.

        Args:
            records: List of text record dictionaries

        Returns:
            Number of newly saved rows.
        """
        if not records:
            logger.info("No text records to save")
            return 0

        existing_hashes = self._load_existing_hashes()
        new_rows = []

        for record in records:
            text_hash = self._text_hash(record["asset"], record["text"])
            if text_hash in existing_hashes:
                continue

            existing_hashes.add(text_hash)
            new_rows.append(
                {
                    "timestamp": pd.Timestamp(record["timestamp"]),
                    "source": record["source"],
                    "asset": record["asset"],
                    "text": record["text"],
                    "user_sentiment": record.get("user_sentiment"),
                    "text_hash": text_hash,
                }
            )

        if not new_rows:
            logger.info("No new rows after deduplication")
            return 0

        schema_cols = ["timestamp", "source", "asset", "text", "user_sentiment", "text_hash"]
        df_new = pd.DataFrame(new_rows)
        for col in schema_cols:
            if col not in df_new.columns:
                df_new[col] = pd.NA
        df_new = df_new[schema_cols].sort_values("timestamp").reset_index(drop=True)

        if self.output_file.exists():
            try:
                existing_df = pd.read_csv(self.output_file)
                for col in schema_cols:
                    if col not in existing_df.columns:
                        existing_df[col] = pd.NA
                existing_df = existing_df[schema_cols]
                combined_df = pd.concat([existing_df, df_new], ignore_index=True)
                combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
                combined_df.to_csv(self.output_file, index=False)
            except Exception as exc:
                logger.warning(
                    f"Failed to merge with existing dataset schema, falling back to append mode: {exc}"
                )
                df_new.to_csv(self.output_file, mode="a", header=False, index=False)
        else:
            df_new.to_csv(self.output_file, index=False)

        logger.info(f"Saved {len(df_new)} new text rows to {self.output_file}")
        return len(df_new)

    def crawl_all(self):
        """Fetch text data from all configured sources and store to CSV."""
        logger.info("=" * 80)
        logger.info("Starting text data crawl")
        logger.info("=" * 80)

        records = []
        try:
            records.extend(self._fetch_reddit(post_limit=50))
        except Exception as exc:
            logger.error(f"Reddit pipeline step failed: {exc}")

        try:
            records.extend(self._fetch_stocktwits())
        except Exception as exc:
            logger.error(f"StockTwits pipeline step failed: {exc}")

        inserted = self.save_records(records)

        logger.info("=" * 80)
        logger.info(f"Text crawl completed. New rows inserted: {inserted}")
        logger.info("=" * 80)

    def run(self):
        """Standardized run method for orchestrator compatibility."""
        logger.info("TextCrawler.run() started")

        records = []
        try:
            records.extend(self._fetch_reddit(post_limit=50))
        except Exception as exc:
            logger.error(f"Reddit pipeline step failed: {exc}")

        try:
            records.extend(self._fetch_stocktwits())
        except Exception as exc:
            logger.error(f"StockTwits pipeline step failed: {exc}")

        inserted = self.save_records(records)
        logger.info(f"TextCrawler.run() inserted {inserted} new rows")
        logger.info("TextCrawler.run() completed successfully")
