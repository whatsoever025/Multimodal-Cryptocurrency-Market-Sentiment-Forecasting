"""
StockTwits Cryptocurrency Crawler

Fetches hourly sentiment messages and discussions for:
- $BTC.X (Bitcoin)
- $ETH.X (Ethereum)

Uses the public StockTwits API endpoint.
Implements rate-limit handling and graceful degradation.
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


class StockTwitsCrawler(BaseCrawler):
    """
    Crawler for fetching sentiment messages from StockTwits.
    
    Inherits from BaseCrawler for:
    - Environment variable loading (.env)
    - Logging setup
    - HTTP session management
    - Retry logic with exponential backoff
    - Rate limiting enforcement
    """

    # StockTwits API configuration
    BASE_URL = "https://api.stocktwits.com/api/v3"
    
    # Symbols to monitor (cashtag format)
    TARGET_SYMBOLS = ["BTC.X", "ETH.X"]

    def __init__(
        self,
        base_path: str = "data/raw",
        config: Optional[CrawlerConfig] = None,
        messages_per_symbol: int = 100,
    ):
        """
        Initialize StockTwits Crawler.

        Args:
            base_path: Directory for saving raw data files
            config: Optional CrawlerConfig for customization
            messages_per_symbol: Number of messages to fetch per symbol
        """
        super().__init__(base_path=base_path, config=config)

        self.api_key = self.get_env("STOCKTWITS_API_KEY")
        self.messages_per_symbol = messages_per_symbol
        
        if not self.api_key:
            self.logger.warning(
                "STOCKTWITS_API_KEY not set in .env. "
                "Using public API (rate limits apply)."
            )

        self.output_file = self.base_path / "stocktwits_messages.csv"
        self.logger.info(f"StockTwitsCrawler initialized. Output: {self.output_file}")

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch messages from StockTwits API for target symbols.

        Returns:
            List of message records as dictionaries
        """
        all_records = []

        for symbol in self.TARGET_SYMBOLS:
            self.logger.info(f"Fetching messages for {symbol}")
            
            try:
                records = self._fetch_symbol_messages(symbol, self.messages_per_symbol)
                all_records.extend(records)
                
            except Exception as e:
                self.logger.error(f"Failed to fetch {symbol}: {e}")
                # Continue with next symbol (graceful degradation)

            # Rate limiting between symbols
            time.sleep(1)

        self.logger.info(f"Fetched {len(all_records)} total records from StockTwits")
        return all_records

    def _fetch_symbol_messages(
        self,
        symbol: str,
        message_limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch messages for a specific symbol from StockTwits.

        Args:
            symbol: StockTwits symbol (e.g., 'BTC.X')
            message_limit: Maximum messages to fetch

        Returns:
            List of message records
        """
        records = []
        fetched_count = 0
        max_id = None

        while fetched_count < message_limit:
            endpoint = f"{self.BASE_URL}/live/symbols/{symbol}"
            params = {"limit": min(30, message_limit - fetched_count)}
            
            if max_id:
                params["max"] = max_id

            if self.api_key:
                params["authorization"] = self.api_key

            try:
                response = self.request_with_retry("GET", endpoint, params=params)
                payload = response.json()
                
            except requests.RequestException as e:
                self.logger.error(f"Failed to fetch {symbol} messages: {e}")
                break

            messages = payload.get("message", {}).get("messages", [])
            if not messages:
                break

            for message in messages:
                created_utc = message.get("created_at")
                timestamp = self._parse_timestamp(created_utc) if created_utc else datetime.utcnow()
                
                # Align to hour
                timestamp = pd.Timestamp(timestamp).floor("h").to_pydatetime()

                record = {
                    "timestamp": timestamp.isoformat(),
                    "symbol": symbol,
                    "message": message.get("body", ""),
                    "user_id": message.get("user", {}).get("id"),
                    "user_username": message.get("user", {}).get("username"),
                    "message_id": message.get("id"),
                    "created_at": created_utc,
                    "conversation_id": message.get("conversation", {}).get("id"),
                    "sentiment": self._detect_sentiment(message.get("body", "")),
                    "message_hash": self._text_hash(message.get("body", "")),
                    "source": "stocktwits",
                }
                
                records.append(record)
                fetched_count += 1

            # Get max_id for pagination
            if messages:
                max_id = messages[-1].get("id")
            else:
                break

            # Rate limiting between pages
            time.sleep(0.5)

        self.logger.info(f"Fetched {len(records)} messages for {symbol}")
        return records

    def _detect_sentiment(self, text: str) -> str:
        """
        Simple sentiment detection from message text.
        
        Looks for bullish/bearish keywords.
        
        Args:
            text: Message text
            
        Returns:
            Sentiment: 'bullish', 'bearish', or 'neutral'
        """
        text_lower = text.lower()
        
        bullish_keywords = [
            r"\bto\s+the\s+moon\b", r"\blambo\b", r"\blong\b",
            r"\bbuy\b", r"\bhold\b", r"\bpump\b", r"\bgain\b"
        ]
        
        bearish_keywords = [
            r"\brekt\b", r"\brip\b", r"\bsell\b", r"\bshort\b",
            r"\bdump\b", r"\bcrash\b", r"\bloss\b"
        ]
        
        bullish_count = sum(1 for kw in bullish_keywords if re.search(kw, text_lower))
        bearish_count = sum(1 for kw in bearish_keywords if re.search(kw, text_lower))
        
        if bullish_count > bearish_count:
            return "bullish"
        elif bearish_count > bullish_count:
            return "bearish"
        else:
            return "neutral"

    def validate(self, records: List[Dict[str, Any]]) -> bool:
        """
        Validate StockTwits message records.

        Checks:
        - Required fields present
        - Timestamp is valid ISO format
        - Message not empty

        Args:
            records: List of message records

        Returns:
            True if all records valid, False otherwise
        """
        if not records:
            self.logger.warning("No records to validate")
            return True

        required_fields = {"timestamp", "symbol", "message", "source"}

        for i, record in enumerate(records):
            missing = required_fields - set(record.keys())
            if missing:
                self.logger.error(f"Record {i} missing fields: {missing}")
                return False

            if not record.get("message", "").strip():
                self.logger.error(f"Record {i} has empty message")
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
        Save messages to CSV file (overwrites existing data).

        Args:
            records: List of validated message records
            filename: Optional filename (defaults to stocktwits_messages.csv)

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
    def _text_hash(text: str) -> str:
        """
        Generate deterministic hash for deduplication.

        Args:
            text: Text to hash

        Returns:
            SHA256 hex digest
        """
        payload = text.lower()
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_timestamp(timestamp_str: str) -> datetime:
        """
        Parse StockTwits timestamp string to datetime.
        
        Handles ISO8601 format from API.

        Args:
            timestamp_str: ISO8601 timestamp string

        Returns:
            Parsed datetime object
        """
        try:
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return datetime.utcnow()


if __name__ == "__main__":
    """
    Standalone execution: python stocktwits_crawler.py
    Fetches messages and saves to data/raw/stocktwits_messages.csv
    """
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        crawler = StockTwitsCrawler()
        saved_count = crawler.run()
        
        if saved_count > 0:
            print(f"✓ StockTwits crawler completed successfully: {saved_count} records saved")
            sys.exit(0)
        else:
            print("✗ StockTwits crawler failed or returned no data")
            sys.exit(0)  # Exit 0 for graceful degradation
            
    except Exception as e:
        print(f"✗ StockTwits crawler encountered fatal error: {e}")
        logging.exception(e)
        sys.exit(0)  # Exit 0 for graceful degradation
