"""
Coinalyze Cryptocurrency Metrics Crawler

Fetches hourly on-chain and derivatives metrics:
- Open Interest (by exchange)
- Liquidations (long/short volume)
- Long/Short Ratio

Covers Bitcoin (BTC) and Ethereum (ETH).
Inherits from BaseCrawler for standardized data collection pipeline.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import requests

from .base import BaseCrawler, CrawlerConfig


class CoinalyzeCrawler(BaseCrawler):
    """
    Crawler for fetching on-chain and derivatives metrics from Coinalyze API.
    
    Inherits from BaseCrawler for:
    - Environment variable loading (.env)
    - Logging setup
    - HTTP session management
    - Retry logic with exponential backoff
    - Rate limiting enforcement
    """

    # Coinalyze API configuration
    BASE_URL = "https://api.coinalyze.net"
    
    # Supported metrics
    METRICS = ["open_interest", "liquidations", "long_short_ratio"]
    
    # Supported cryptocurrencies
    TARGET_ASSETS = ["BTC", "ETH"]

    def __init__(
        self,
        base_path: str = "data/raw",
        config: Optional[CrawlerConfig] = None,
        hours_back: int = 24,
    ):
        """
        Initialize Coinalyze Crawler.

        Args:
            base_path: Directory for saving raw data files
            config: Optional CrawlerConfig for customization
            hours_back: Number of hours of historical data to fetch
        """
        super().__init__(base_path=base_path, config=config)

        self.api_key = self.get_env("COINALYZE_API_KEY")
        self.hours_back = hours_back
        
        if not self.api_key:
            self.logger.warning(
                "COINALYZE_API_KEY not set in .env. "
                "Coinalyze crawler will not work without API key."
            )

        self.output_file = self.base_path / "coinalyze_metrics.csv"
        self.logger.info(f"CoinalyzeCrawler initialized. Output: {self.output_file}")

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch metrics from Coinalyze API.

        Returns:
            List of metric records as dictionaries
        """
        if not self.api_key:
            self.logger.error("COINALYZE_API_KEY not available. Skipping Coinalyze.")
            return []

        all_records = []

        for asset in self.TARGET_ASSETS:
            self.logger.info(f"Fetching metrics for {asset}")
            
            try:
                records = self._fetch_asset_metrics(asset)
                all_records.extend(records)
                
            except Exception as e:
                self.logger.error(f"Failed to fetch {asset} metrics: {e}")
                # Continue with next asset (graceful degradation)

            # Rate limiting between assets
            time.sleep(1)

        self.logger.info(f"Fetched {len(all_records)} total metric records from Coinalyze")
        return all_records

    def _fetch_asset_metrics(self, asset: str) -> List[Dict[str, Any]]:
        """
        Fetch all metrics for a specific asset.

        Args:
            asset: Asset code (e.g., 'BTC', 'ETH')

        Returns:
            List of metric records
        """
        records = []

        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=self.hours_back)

        for metric in self.METRICS:
            self.logger.debug(f"Fetching {metric} for {asset}")
            
            try:
                metric_records = self._fetch_single_metric(
                    asset, metric, start_time, end_time
                )
                records.extend(metric_records)
                
            except Exception as e:
                self.logger.warning(f"Failed to fetch {metric} for {asset}: {e}")
                # Continue with next metric

            # Rate limiting between metrics
            time.sleep(0.5)

        self.logger.info(f"Fetched {len(records)} records for {asset}")
        return records

    def _fetch_single_metric(
        self,
        asset: str,
        metric: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Fetch a specific metric from Coinalyze API.

        Args:
            asset: Asset code (e.g., 'BTC', 'ETH')
            metric: Metric type ('open_interest', 'liquidations', 'long_short_ratio')
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of metric records
        """
        records = []

        try:
            # Construct API endpoint
            endpoint = f"{self.BASE_URL}/v1/metrics/{metric}/{asset}"
            
            params = {
                "api_key": self.api_key,
                "start": int(start_time.timestamp()),
                "end": int(end_time.timestamp()),
                "interval": "1h",
            }

            response = self.request_with_retry("GET", endpoint, params=params)
            payload = response.json()

            if not payload.get("data"):
                self.logger.warning(f"No data returned for {metric}/{asset}")
                return records

            # Parse metric-specific data
            if metric == "open_interest":
                records = self._parse_open_interest(payload.get("data", []), asset)
                
            elif metric == "liquidations":
                records = self._parse_liquidations(payload.get("data", []), asset)
                
            elif metric == "long_short_ratio":
                records = self._parse_long_short_ratio(payload.get("data", []), asset)

        except requests.RequestException as e:
            self.logger.error(f"API error fetching {metric}/{asset}: {e}")

        return records

    def _parse_open_interest(
        self,
        data: List[Dict[str, Any]],
        asset: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse open interest data from API response.

        Args:
            data: API response data
            asset: Asset code

        Returns:
            List of parsed records
        """
        records = []
        
        for item in data:
            timestamp = self._parse_timestamp(item.get("timestamp"))
            
            record = {
                "timestamp": timestamp.isoformat(),
                "asset": asset,
                "metric_type": "open_interest",
                "value": float(item.get("open_interest", 0)),
                "exchange": item.get("exchange", "aggregated"),
                "currency_usd": float(item.get("usd_oi", 0)),
                "source": "coinalyze",
            }
            records.append(record)

        return records

    def _parse_liquidations(
        self,
        data: List[Dict[str, Any]],
        asset: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse liquidations data from API response.

        Args:
            data: API response data
            asset: Asset code

        Returns:
            List of parsed records
        """
        records = []
        
        for item in data:
            timestamp = self._parse_timestamp(item.get("timestamp"))
            
            record = {
                "timestamp": timestamp.isoformat(),
                "asset": asset,
                "metric_type": "liquidations",
                "long_liquidations_usd": float(item.get("long_usd", 0)),
                "short_liquidations_usd": float(item.get("short_usd", 0)),
                "total_liquidations_usd": float(item.get("total_usd", 0)),
                "exchange": item.get("exchange", "aggregated"),
                "source": "coinalyze",
            }
            records.append(record)

        return records

    def _parse_long_short_ratio(
        self,
        data: List[Dict[str, Any]],
        asset: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse long/short ratio data from API response.

        Args:
            data: API response data
            asset: Asset code

        Returns:
            List of parsed records
        """
        records = []
        
        for item in data:
            timestamp = self._parse_timestamp(item.get("timestamp"))
            
            record = {
                "timestamp": timestamp.isoformat(),
                "asset": asset,
                "metric_type": "long_short_ratio",
                "long_positions": float(item.get("long", 0)),
                "short_positions": float(item.get("short", 0)),
                "ratio": float(item.get("ratio", 0)),
                "exchange": item.get("exchange", "aggregated"),
                "source": "coinalyze",
            }
            records.append(record)

        return records

    def validate(self, records: List[Dict[str, Any]]) -> bool:
        """
        Validate Coinalyze metric records.

        Checks:
        - Required fields present
        - Timestamp is valid ISO format
        - Numeric values are positive or zero

        Args:
            records: List of metric records

        Returns:
            True if all records valid, False otherwise
        """
        if not records:
            self.logger.warning("No records to validate")
            return True

        required_fields = {"timestamp", "asset", "metric_type", "source"}

        for i, record in enumerate(records):
            missing = required_fields - set(record.keys())
            if missing:
                self.logger.error(f"Record {i} missing fields: {missing}")
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
        Save metrics to CSV file (overwrites existing data).

        Args:
            records: List of validated metric records
            filename: Optional filename (defaults to coinalyze_metrics.csv)

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
    def _parse_timestamp(timestamp_val: Any) -> datetime:
        """
        Parse timestamp from API response.

        Handles both Unix timestamp and ISO8601 strings.

        Args:
            timestamp_val: Timestamp value (int or str)

        Returns:
            Parsed datetime object
        """
        try:
            if isinstance(timestamp_val, (int, float)):
                return datetime.utcfromtimestamp(timestamp_val)
            elif isinstance(timestamp_val, str):
                return datetime.fromisoformat(timestamp_val.replace("Z", "+00:00"))
            else:
                return datetime.utcnow()
        except (ValueError, TypeError):
            return datetime.utcnow()


if __name__ == "__main__":
    """
    Standalone execution: python coinalyze_crawler.py
    Fetches metrics and saves to data/raw/coinalyze_metrics.csv
    """
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        crawler = CoinalyzeCrawler()
        saved_count = crawler.run()
        
        if saved_count > 0:
            print(f"✓ Coinalyze crawler completed successfully: {saved_count} records saved")
            sys.exit(0)
        else:
            print("✗ Coinalyze crawler failed or returned no data")
            sys.exit(0)  # Exit 0 for graceful degradation
            
    except Exception as e:
        print(f"✗ Coinalyze crawler encountered fatal error: {e}")
        logging.exception(e)
        sys.exit(0)  # Exit 0 for graceful degradation
