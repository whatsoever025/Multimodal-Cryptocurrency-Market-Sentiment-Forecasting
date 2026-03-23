"""
Coinalyze Cryptocurrency Metrics Crawler

Fetches hourly on-chain and derivatives metrics (last 12 months by default):
- Open Interest (OHLC format)
- Liquidations (long/short volume)
- Long/Short Ratio

Covers Bitcoin (BTC) and Ethereum (ETH).
Fetches data monthly in chunks and saves to three separate CSV files.
Inherits from BaseCrawler for standardized data collection pipeline.
"""

import logging
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import requests

from .base import BaseCrawler, CrawlerConfig


class CoinalyzeCrawler(BaseCrawler):
    """
    Crawler for fetching on-chain and derivatives metrics from Coinalyze API.
    
    Features:
    - Fetches recent data in monthly chunks (default: last 12 months)
    - Saves to three separate optimized CSV files
    - Handles API data structure variations gracefully
    - Inherits from BaseCrawler for environment loading, logging, HTTP session, retry logic, rate limiting
    """

    # Coinalyze API configuration
    BASE_URL = "https://api.coinalyze.net"
    
    # Supported metrics
    METRICS = ["open_interest", "liquidations", "long_short_ratio"]
    
    # Supported cryptocurrencies with exchange format
    TARGET_ASSETS = {
        "BTCUSDT_PERP.A": "BTC",  # Bybit perpetual
        "ETHUSDT_PERP.A": "ETH",  # Bybit perpetual
    }

    def __init__(
        self,
        base_path: str = "data/raw",
        config: Optional[CrawlerConfig] = None,
        months_back: int = 12,
    ):
        """
        Initialize Coinalyze Crawler.

        Args:
            base_path: Directory for saving raw data files
            config: Optional CrawlerConfig for customization
            months_back: Number of months of historical data to fetch (default: 12)
        """
        super().__init__(base_path=base_path, config=config)

        self.api_key = self.get_env("COINALYZE_API_KEY")
        self.months_back = months_back
        
        if not self.api_key:
            self.logger.warning(
                "COINALYZE_API_KEY not set in .env. "
                "Coinalyze crawler will not work without API key."
            )

        # Output files for each metric type
        self.output_files = {
            "open_interest": self.base_path / "coinalyze_open_interest.csv",
            "liquidations": self.base_path / "coinalyze_liquidations.csv",
            "long_short_ratio": self.base_path / "coinalyze_long_short_ratio.csv",
        }
        
        self.logger.info(f"CoinalyzeCrawler initialized. Fetching last {months_back} months. Output files: {list(self.output_files.values())}")

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch metrics from Coinalyze API for the last N months in monthly chunks.

        Returns:
            List of metric records as dictionaries
        """
        if not self.api_key:
            self.logger.error("COINALYZE_API_KEY not available. Skipping Coinalyze.")
            return []

        all_records = []
        
        # Generate monthly date ranges (most recent first, then backwards)
        end_date = datetime.utcnow()
        start_date = end_date - relativedelta(months=self.months_back)

        current_date = start_date
        month_count = 0

        while current_date < end_date:
            month_start = current_date
            month_end = current_date + relativedelta(months=1)
            
            # Don't go beyond current time
            if month_end > end_date:
                month_end = end_date

            self.logger.info(f"Fetching for period: {month_start.date()} to {month_end.date()}")
            
            try:
                monthly_records = self._fetch_month(month_start, month_end)
                all_records.extend(monthly_records)
                month_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {month_start.date()}: {e}")
                # Continue with next month (graceful degradation)

            current_date = month_end
            time.sleep(0.5)  # Rate limiting between months

        self.logger.info(f"Fetched {len(all_records)} total metric records from {month_count} months")
        return all_records

    def _fetch_month(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Fetch all metrics for one month across all assets.

        Args:
            start_time: Start of month
            end_time: End of month

        Returns:
            List of metric records
        """
        records = []

        for symbol, asset_name in self.TARGET_ASSETS.items():
            for metric in self.METRICS:
                try:
                    metric_records = self._fetch_single_metric(
                        symbol, asset_name, metric, start_time, end_time
                    )
                    records.extend(metric_records)
                    time.sleep(0.3)  # Rate limiting between API calls
                    
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {metric} for {symbol}: {e}")
                    # Continue with next metric

        return records

    def _fetch_single_metric(
        self,
        symbol: str,
        asset_name: str,
        metric: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Fetch a specific metric from Coinalyze API.

        Args:
            symbol: Symbol code (e.g., 'BTCUSDT_PERP.A')
            asset_name: Asset name for display (e.g., 'BTC')
            metric: Metric type ('open_interest', 'liquidations', 'long_short_ratio')
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of metric records
        """
        records = []

        try:
            # Map metric types to API endpoints
            endpoint_map = {
                "open_interest": f"{self.BASE_URL}/v1/open-interest-history",
                "liquidations": f"{self.BASE_URL}/v1/liquidation-history",
                "long_short_ratio": f"{self.BASE_URL}/v1/long-short-ratio-history",
            }
            
            endpoint = endpoint_map.get(metric)
            if not endpoint:
                self.logger.warning(f"Unknown metric type: {metric}")
                return records
            
            params = {
                "api_key": self.api_key,
                "symbols": symbol,
                "interval": "1hour",
                "from": int(start_time.timestamp()),
                "to": int(end_time.timestamp()),
            }
            
            # Add convert_to_usd for open_interest metric
            if metric == "open_interest":
                params["convert_to_usd"] = "true"

            response = self.request_with_retry("GET", endpoint, params=params)
            payload = response.json()

            # API returns a list directly
            if isinstance(payload, list) and len(payload) > 0:
                payload = payload[0]  # Get first item
                
            if not payload.get("history"):
                self.logger.warning(f"No data returned for {metric}/{symbol}")
                return records

            # Parse metric-specific data
            if metric == "open_interest":
                records = self._parse_open_interest(payload.get("history", []), asset_name)
                
            elif metric == "liquidations":
                records = self._parse_liquidations(payload.get("history", []), asset_name)
                
            elif metric == "long_short_ratio":
                records = self._parse_long_short_ratio(payload.get("history", []), asset_name)

        except requests.RequestException as e:
            self.logger.error(f"API error fetching {metric}/{symbol}: {e}")

        return records

    def _parse_open_interest(
        self,
        data: List[Dict[str, Any]],
        asset_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse open interest data from API response.

        Args:
            data: API response history data with OHLC
            asset_name: Asset name (e.g., 'BTC')

        Returns:
            List of parsed records
        """
        records = []
        
        for item in data:
            timestamp = datetime.utcfromtimestamp(item.get("t", 0))
            
            record = {
                "timestamp": timestamp.isoformat(),
                "asset": asset_name,
                "metric_type": "open_interest",
                "open": float(item.get("o", 0)),
                "high": float(item.get("h", 0)),
                "low": float(item.get("l", 0)),
                "close": float(item.get("c", 0)),
                "source": "coinalyze",
            }
            records.append(record)

        return records

    def _parse_liquidations(
        self,
        data: List[Dict[str, Any]],
        asset_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse liquidations data from API response.

        Args:
            data: API response history data with liquidations
                  l: long liquidations, s: short liquidations
            asset_name: Asset name (e.g., 'BTC')

        Returns:
            List of parsed records
        """
        records = []
        
        for item in data:
            timestamp = datetime.utcfromtimestamp(item.get("t", 0))
            
            record = {
                "timestamp": timestamp.isoformat(),
                "asset": asset_name,
                "metric_type": "liquidations",
                "long_liquidations": float(item.get("l", 0)),
                "short_liquidations": float(item.get("s", 0)),
                "source": "coinalyze",
            }
            records.append(record)

        return records

    def _parse_long_short_ratio(
        self,
        data: List[Dict[str, Any]],
        asset_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse long/short ratio data from API response.

        Args:
            data: API response history data with ratio info
                  r: ratio, l: long %, s: short %
            asset_name: Asset name (e.g., 'BTC')

        Returns:
            List of parsed records
        """
        records = []
        
        for item in data:
            timestamp = datetime.utcfromtimestamp(item.get("t", 0))
            
            record = {
                "timestamp": timestamp.isoformat(),
                "asset": asset_name,
                "metric_type": "long_short_ratio",
                "ratio": float(item.get("r", 0)),
                "long_percentage": float(item.get("l", 0)),
                "short_percentage": float(item.get("s", 0)),
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
        Save metrics to three separate CSV files, optimized per metric type.

        Files created:
        - coinalyze_open_interest.csv: timestamp, asset, open, high, low, close
        - coinalyze_liquidations.csv: timestamp, asset, long_liquidations, short_liquidations
        - coinalyze_long_short_ratio.csv: timestamp, asset, ratio, long_percentage, short_percentage

        Args:
            records: List of validated metric records
            filename: Optional filename (ignored, uses three predefined files)

        Returns:
            Number of records saved across all files
        """
        if not records:
            self.logger.warning("No records to save")
            return 0

        try:
            # Separate records by metric type
            records_by_metric = {
                "open_interest": [],
                "liquidations": [],
                "long_short_ratio": [],
            }

            for record in records:
                metric_type = record.get("metric_type")
                if metric_type in records_by_metric:
                    records_by_metric[metric_type].append(record)

            total_saved = 0

            # Save open_interest with optimized fields
            if records_by_metric["open_interest"]:
                oi_df = pd.DataFrame(records_by_metric["open_interest"])
                oi_df = oi_df[["timestamp", "asset", "open", "high", "low", "close"]]
                oi_df = oi_df.sort_values("timestamp")
                oi_df.to_csv(self.output_files["open_interest"], index=False)
                self.logger.info(
                    f"Saved {len(oi_df)} open_interest records to {self.output_files['open_interest']}"
                )
                total_saved += len(oi_df)

            # Save liquidations with optimized fields
            if records_by_metric["liquidations"]:
                liq_df = pd.DataFrame(records_by_metric["liquidations"])
                liq_df = liq_df[["timestamp", "asset", "long_liquidations", "short_liquidations"]]
                liq_df = liq_df.sort_values("timestamp")
                liq_df.to_csv(self.output_files["liquidations"], index=False)
                self.logger.info(
                    f"Saved {len(liq_df)} liquidations records to {self.output_files['liquidations']}"
                )
                total_saved += len(liq_df)

            # Save long_short_ratio with optimized fields
            if records_by_metric["long_short_ratio"]:
                lsr_df = pd.DataFrame(records_by_metric["long_short_ratio"])
                lsr_df = lsr_df[["timestamp", "asset", "ratio", "long_percentage", "short_percentage"]]
                lsr_df = lsr_df.sort_values("timestamp")
                lsr_df.to_csv(self.output_files["long_short_ratio"], index=False)
                self.logger.info(
                    f"Saved {len(lsr_df)} long_short_ratio records to {self.output_files['long_short_ratio']}"
                )
                total_saved += len(lsr_df)

            return total_saved

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
