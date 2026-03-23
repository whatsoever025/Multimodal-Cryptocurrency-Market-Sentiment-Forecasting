"""
Reddit Cryptocurrency Crawler

Fetches hourly discussions and posts from crypto-related subreddits:
- r/CryptoCurrency
- r/Bitcoin
- r/Ethereum

Uses Reddit's public .json endpoint (no PRAW library required).
Inherits from BaseCrawler for standardized data collection pipeline.
"""

import hashlib
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import requests

from .base import BaseCrawler, CrawlerConfig


class RedditCrawler(BaseCrawler):
    """
    Crawler for fetching cryptocurrency discussion posts from Reddit.
    
    Inherits from BaseCrawler for:
    - Environment variable loading (.env)
    - Logging setup
    - HTTP session management with connection pooling
    - Retry logic with exponential backoff
    - Rate limiting enforcement
    """

    # Target subreddits to monitor
    TARGET_SUBREDDITS = ["CryptoCurrency", "Bitcoin", "Ethereum"]

    # Asset detection patterns for automatic categorization
    ASSET_PATTERNS = {
        "BTC": [r"\bbtc\b", r"\bbitcoin\b"],
        "ETH": [r"\beth\b", r"\bethereum\b"],
    }

    def __init__(
        self,
        base_path: str = "data/raw",
        config: Optional[CrawlerConfig] = None,
        posts_per_subreddit: int = 100,
    ):
        """
        Initialize Reddit Crawler.

        Args:
            base_path: Directory for saving raw data files
            config: Optional CrawlerConfig for customization
            posts_per_subreddit: Number of posts to fetch per subreddit
        """
        super().__init__(base_path=base_path, config=config)

        self.posts_per_subreddit = posts_per_subreddit
        self.reddit_user_agent = self.get_env("REDDIT_USER_AGENT")
        
        if not self.reddit_user_agent:
            self.logger.warning(
                "REDDIT_USER_AGENT not set in .env. "
                "Reddit may block requests. Set REDDIT_USER_AGENT to continue."
            )

        self.output_file = self.base_path / "reddit_posts.csv"
        self.logger.info(f"RedditCrawler initialized. Output: {self.output_file}")

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch posts from target subreddits using Reddit's public .json endpoint.

        Returns:
            List of post records as dictionaries
        """
        if not self.reddit_user_agent:
            self.logger.error("REDDIT_USER_AGENT not available. Skipping Reddit.")
            return []

        all_records = []
        headers = {"User-Agent": self.reddit_user_agent}

        for subreddit_name in self.TARGET_SUBREDDITS:
            self.logger.info(f"Fetching posts from r/{subreddit_name}")
            records = self._fetch_subreddit_posts(
                subreddit_name, headers, self.posts_per_subreddit
            )
            all_records.extend(records)
            
            # Rate limiting between subreddits
            time.sleep(1)

        self.logger.info(f"Fetched {len(all_records)} total records from Reddit")
        return all_records

    def _fetch_subreddit_posts(
        self,
        subreddit_name: str,
        headers: Dict[str, str],
        post_limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch posts from a single subreddit.

        Args:
            subreddit_name: Name of subreddit (without r/)
            headers: HTTP headers with User-Agent
            post_limit: Maximum posts to fetch

        Returns:
            List of post records
        """
        records = []
        fetched_count = 0
        after = None

        while fetched_count < post_limit:
            page_size = min(100, post_limit - fetched_count)
            endpoint = f"https://www.reddit.com/r/{subreddit_name}.json"
            params = {"limit": page_size}
            
            if after:
                params["after"] = after

            try:
                response = self.request_with_retry("GET", endpoint, params=params, headers=headers)
                payload = response.json()
                
            except requests.RequestException as e:
                self.logger.error(f"Failed to fetch r/{subreddit_name}: {e}")
                break

            children = payload.get("data", {}).get("children", [])
            if not children:
                break

            for post in children:
                post_data = post.get("data", {})
                title = self._clean_text(post_data.get("title", ""))
                selftext = self._clean_text(post_data.get("selftext", ""))
                combined_text = f"{title} {selftext}".strip()

                if not combined_text:
                    continue

                created_utc = post_data.get("created_utc")
                timestamp = self._align_timestamp_to_hour(
                    datetime.utcfromtimestamp(created_utc) if created_utc else None
                )

                # Detect assets mentioned in the post
                assets = self._detect_assets(combined_text)

                record = {
                    "timestamp": timestamp.isoformat(),
                    "subreddit": subreddit_name,
                    "title": title,
                    "text": selftext,
                    "combined_text": combined_text,
                    "url": post_data.get("url", ""),
                    "score": post_data.get("score", 0),
                    "num_comments": post_data.get("num_comments", 0),
                    "author": post_data.get("author", ""),
                    "assets": ",".join(assets) if assets else "OTHER",
                    "text_hash": self._text_hash(combined_text),
                    "source": "reddit",
                }
                
                records.append(record)
                fetched_count += 1

            # Get next page marker
            after = payload.get("data", {}).get("after")
            if not after:
                break

            # Rate limiting between pages
            time.sleep(0.5)

        self.logger.info(f"Fetched {len(records)} posts from r/{subreddit_name}")
        return records

    def _detect_assets(self, text: str) -> List[str]:
        """
        Detect cryptocurrency assets mentioned in text.

        Args:
            text: Text to search

        Returns:
            List of detected asset codes (e.g., ['BTC', 'ETH'])
        """
        detected = set()
        text_lower = text.lower()

        for asset, patterns in self.ASSET_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected.add(asset)
                    break

        return sorted(list(detected))

    def validate(self, records: List[Dict[str, Any]]) -> bool:
        """
        Validate Reddit post records.

        Checks:
        - Required fields present (timestamp, subreddit, text)
        - Timestamp is valid ISO format
        - Text not empty

        Args:
            records: List of post records

        Returns:
            True if all records valid, False otherwise
        """
        if not records:
            self.logger.warning("No records to validate")
            return True

        required_fields = {"timestamp", "subreddit", "combined_text", "source"}

        for i, record in enumerate(records):
            missing = required_fields - set(record.keys())
            if missing:
                self.logger.error(f"Record {i} missing fields: {missing}")
                return False

            if not record.get("combined_text", "").strip():
                self.logger.error(f"Record {i} has empty combined_text")
                return False

            try:
                datetime.fromisoformat(record.get("timestamp", ""))
            except (ValueError, TypeError):
                self.logger.error(f"Record {i} has invalid timestamp: {record.get('timestamp')}")
                return False

        self.logger.info(f"Validated {len(records)} records successfully")
        return True

    def save(self, records: List[Dict[str, Any]], filename: Optional[str] = None) -> int:
        """
        Save posts to CSV file (overwrites existing data).

        Args:
            records: List of validated post records
            filename: Optional filename (defaults to reddit_posts.csv)

        Returns:
            Number of records saved
        """
        if filename:
            output_file = self.base_path / filename
        else:
            output_file = self.output_file

        try:
            df = pd.DataFrame(records)
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {len(records)} records to {output_file}")
            return len(records)

        except Exception as e:
            self.logger.error(f"Failed to save records: {e}")
            return 0

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean text by removing URLs, emojis, and extra whitespace.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove URLs
        cleaned = re.sub(r"https?://\S+|www\.\S+", " ", text)
        
        # Remove non-ASCII characters (including emojis)
        cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
        
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        
        return cleaned

    @staticmethod
    def _text_hash(text: str) -> str:
        """
        Generate deterministic hash for text deduplication.

        Args:
            text: Text to hash

        Returns:
            SHA256 hex digest
        """
        payload = text.lower()
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _align_timestamp_to_hour(dt: Optional[datetime] = None) -> datetime:
        """
        Align timestamp to the nearest lower hour.

        Args:
            dt: Datetime to align (defaults to now if None)

        Returns:
            Datetime aligned to lower hour
        """
        if dt is None:
            dt = datetime.utcnow()
        return pd.Timestamp(dt).floor("h").to_pydatetime()


if __name__ == "__main__":
    """
    Standalone execution: python reddit_crawler.py
    Fetches posts and saves to data/raw/reddit_posts.csv
    """
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        crawler = RedditCrawler()
        saved_count = crawler.run()
        
        if saved_count > 0:
            print(f"✓ Redis crawler completed successfully: {saved_count} records saved")
            sys.exit(0)
        else:
            print("✗ Reddit crawler failed or returned no data")
            sys.exit(0)  # Exit 0 for graceful degradation
            
    except Exception as e:
        print(f"✗ Reddit crawler encountered fatal error: {e}")
        logging.exception(e)
        sys.exit(0)  # Exit 0 for graceful degradation
