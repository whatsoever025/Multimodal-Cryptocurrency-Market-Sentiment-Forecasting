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
import praw
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

        self.cryptopanic_api_key = os.getenv("CRYPTOPANIC_API_KEY")
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = os.getenv("REDDIT_USER_AGENT")

        self.reddit = self._init_reddit_client()

        logger.info("TextCrawler initialized")
        logger.info(f"Output file: {self.output_file.absolute()}")

    def _init_reddit_client(self):
        """Initialize Reddit API client or return None if credentials are missing."""
        required = [
            self.reddit_client_id,
            self.reddit_client_secret,
            self.reddit_user_agent,
        ]
        if not all(required):
            logger.warning(
                "Reddit credentials missing. Skipping Reddit source. "
                "Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT in .env."
            )
            return None

        try:
            reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent=self.reddit_user_agent,
            )
            # Lightweight check to fail early on invalid credentials.
            _ = reddit.read_only
            logger.info("Reddit client initialized")
            return reddit
        except Exception as exc:
            logger.warning(f"Failed to initialize Reddit client. Skipping Reddit source: {exc}")
            return None

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

    def fetch_cryptopanic_news(self):
        """
        Fetch BTC/ETH news titles from CryptoPanic.

        Returns:
            List of dictionaries with text records.
        """
        if not self.cryptopanic_api_key:
            logger.warning(
                "CRYPTOPANIC_API_KEY missing. Skipping CryptoPanic source. "
                "Set CRYPTOPANIC_API_KEY in .env."
            )
            return []

        endpoint = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": self.cryptopanic_api_key,
            "currencies": "BTC,ETH",
            "kind": "news",
            "public": "true",
        }

        records = []

        try:
            response = requests.get(endpoint, params=params, timeout=self.request_timeout)
            response.raise_for_status()
            payload = response.json()

            for item in payload.get("results", []):
                title = self._clean_text(item.get("title", ""))
                if not title:
                    continue

                created_at = item.get("created_at")
                parsed_dt = pd.to_datetime(created_at, utc=True, errors="coerce")
                if pd.isna(parsed_dt):
                    aligned_ts = self._align_timestamp_to_hour(None)
                else:
                    aligned_ts = self._align_timestamp_to_hour(parsed_dt.tz_convert(None))

                currencies = item.get("currencies") or []
                assets = []
                for c in currencies:
                    code = (c.get("code") or "").upper().strip()
                    if code in {"BTC", "ETH"}:
                        assets.append(code)

                if not assets:
                    assets = ["BTC", "ETH"]

                for asset in assets:
                    records.append(
                        {
                            "timestamp": aligned_ts,
                            "source": "cryptopanic",
                            "asset": asset,
                            "text": title,
                        }
                    )

            logger.info(f"Fetched {len(records)} text rows from CryptoPanic")
        except Exception as exc:
            logger.error(f"CryptoPanic fetch failed: {exc}")

        # Strict rate limiting.
        time.sleep(2)
        return records

    def fetch_reddit_posts(self, post_limit=50):
        """
        Fetch hot and top posts from selected crypto subreddits.

        Args:
            post_limit: Number of posts to fetch per listing

        Returns:
            List of dictionaries with text records.
        """
        if self.reddit is None:
            return []

        records = []
        subreddits = ["CryptoCurrency", "Bitcoin"]

        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                for post in subreddit.hot(limit=post_limit):
                    text = self._clean_text(post.title)
                    if not text:
                        continue

                    asset = "BTC" if subreddit_name.lower() == "bitcoin" else "BTC,ETH"
                    aligned_ts = self._align_timestamp_to_hour(
                        datetime.utcfromtimestamp(post.created_utc)
                    )
                    records.append(
                        {
                            "timestamp": aligned_ts,
                            "source": f"reddit:{subreddit_name.lower()}:hot",
                            "asset": asset,
                            "text": text,
                        }
                    )

                # Strict rate limiting between listing calls.
                time.sleep(2)

                for post in subreddit.top(time_filter="day", limit=post_limit):
                    text = self._clean_text(post.title)
                    if not text:
                        continue

                    asset = "BTC" if subreddit_name.lower() == "bitcoin" else "BTC,ETH"
                    aligned_ts = self._align_timestamp_to_hour(
                        datetime.utcfromtimestamp(post.created_utc)
                    )
                    records.append(
                        {
                            "timestamp": aligned_ts,
                            "source": f"reddit:{subreddit_name.lower()}:top",
                            "asset": asset,
                            "text": text,
                        }
                    )

                # Strict rate limiting between subreddits.
                time.sleep(2)

            except Exception as exc:
                logger.error(f"Reddit fetch failed for r/{subreddit_name}: {exc}")
                time.sleep(2)

        logger.info(f"Fetched {len(records)} text rows from Reddit")
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
                    "text_hash": text_hash,
                }
            )

        if not new_rows:
            logger.info("No new rows after deduplication")
            return 0

        df_new = pd.DataFrame(new_rows)
        df_new = df_new.sort_values("timestamp").reset_index(drop=True)

        if self.output_file.exists():
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
        records.extend(self.fetch_cryptopanic_news())
        records.extend(self.fetch_reddit_posts(post_limit=50))

        inserted = self.save_records(records)

        logger.info("=" * 80)
        logger.info(f"Text crawl completed. New rows inserted: {inserted}")
        logger.info("=" * 80)

    def run(self):
        """Standardized run method for orchestrator compatibility."""
        logger.info("TextCrawler.run() started")
        self.crawl_all()
        logger.info("TextCrawler.run() completed successfully")
