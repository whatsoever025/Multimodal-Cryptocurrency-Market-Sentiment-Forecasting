"""
Sentiment Crawler for Cryptocurrency Market Sentiment Data
Fetches Fear & Greed Index from Alternative.me API.

Uses BaseCrawler inheritance for standardized data collection pipeline.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import requests

from .base import BaseCrawler, CrawlerConfig


class SentimentCrawler(BaseCrawler):
    """
    Crawler for fetching cryptocurrency market sentiment data.
    Focuses on Fear & Greed Index from Alternative.me API (no API key needed).
    
    Inherits from BaseCrawler for:
    - Environment variable loading (.env)
    - Logging setup
    - HTTP session management
    - Retry logic with exponential backoff
    - Rate limiting enforcement
    """
    
    BASE_URL = "https://api.alternative.me"

    def __init__(
        self,
        base_path: str = 'data/raw',
        config: Optional[CrawlerConfig] = None,
        limit: int = 0,
    ):
        """
        Initialize Sentiment Crawler.
        
        Args:
            base_path: Directory path for saving raw data files
            config: Optional CrawlerConfig for customization
            limit: Number of records to fetch (0 = all available history)
        """
        # Call parent init for standard setup (env loading, logging, etc.)
        super().__init__(base_path=base_path, config=config)
        
        self.limit = limit
        self.output_file = self.base_path / 'fear_greed_index.csv'
        self.logger.info(f"SentimentCrawler initialized. Output: {self.output_file}")
    
    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch Fear & Greed Index data from API.
        
        Returns:
            List of sentiment records as dictionaries
        """
        try:
            df = self._fetch_fear_greed_index(limit=self.limit)
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            self.logger.error(f"Failed to fetch sentiment data: {e}")
            return []
    
    def validate(self, records: List[Dict[str, Any]]) -> bool:
        """
        Validate sentiment data schema.
        
        Checks:
        - Required fields present: value, value_classification, timestamp
        - Numeric types correct
        - Value range (0-100 for Fear & Greed)

        Args:
            records: List of sentiment records

        Returns:
            True if all records valid, False otherwise
        """
        if not records:
            self.logger.warning("No records to validate")
            return True
        
        required_fields = {'value', 'value_classification', 'timestamp'}
        
        for i, record in enumerate(records):
            missing = required_fields - set(record.keys())
            if missing:
                self.logger.error(f"Record {i} missing fields: {missing}")
                return False
            
            try:
                value = int(record['value'])
                if not (0 <= value <= 100):
                    self.logger.error(f"Record {i}: value {value} out of range [0-100]")
                    return False
            except (ValueError, TypeError) as e:
                self.logger.error(f"Record {i}: invalid value - {e}")
                return False
        
        self.logger.info(f"Validated {len(records)} records successfully")
        return True
    
    def save(
        self,
        records: List[Dict[str, Any]],
        filename: Optional[str] = None,
    ) -> int:
        """
        Save sentiment records to CSV file (overwrites existing data).
        
        Args:
            records: List of sentiment records
            filename: Optional filename (defaults to fear_greed_index.csv)
            
        Returns:
            Number of rows saved
        """
        if filename:
            output_file = self.base_path / filename
        else:
            output_file = self.output_file

        if not records:
            self.logger.warning("No records to save")
            return 0
        
        try:
            df = pd.DataFrame(records)
            
            # Remove duplicates based on timestamp
            original_len = len(df)
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
            deduped = original_len - len(df)
            if deduped > 0:
                self.logger.info(f"Removed {deduped} duplicate records")
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            
            self.logger.info(f"Successfully saved {len(df)} records to {output_file}")
            return len(df)
            
        except Exception as e:
            self.logger.error(f"Failed to save records: {e}")
            return 0

    def _fetch_fear_greed_index(self, limit: int = 0) -> pd.DataFrame:
        """
        Fetch Fear & Greed Index historical data.
        
        Args:
            limit: Number of records to fetch (0 = all available history)
        
        Returns:
            DataFrame with Fear & Greed Index data
        """
        self.logger.info(f"Fetching Fear & Greed Index (limit={limit if limit > 0 else 'ALL'})")
        
        try:
            endpoint = f"{self.BASE_URL}/fng/"
            params = {'limit': limit}
            
            response = self.request_with_retry('GET', endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data:
                raise ValueError("Invalid response format from Fear & Greed API")
            
            records = []
            for entry in data['data']:
                timestamp_val = int(entry.get('timestamp', 0))
                record = {
                    'timestamp': timestamp_val,
                    'datetime': pd.Timestamp.utcfromtimestamp(timestamp_val).isoformat() if timestamp_val > 0 else None,
                    'value': int(entry.get('value', 0)),
                    'value_classification': entry.get('value_classification', 'Unknown'),
                    'time_until_update': entry.get('time_until_update'),
                    'source': 'alternative.me',
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            
            # Sort by timestamp (oldest first)
            if not df.empty:
                df = df.sort_values('timestamp').reset_index(drop=True)
                self.logger.info(
                    f"Successfully fetched {len(df)} Fear & Greed Index records. "
                    f"Range: {df['datetime'].min()} to {df['datetime'].max()}"
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching Fear & Greed Index: {e}", exc_info=True)
            return pd.DataFrame()


if __name__ == "__main__":
    """
    Standalone execution: python sentiment_crawler.py
    Fetches sentiment data and saves to data/raw/fear_greed_index.csv
    """
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        crawler = SentimentCrawler()
        saved_count = crawler.run()
        
        if saved_count > 0:
            print(f"✓ Sentiment crawler completed successfully: {saved_count} records saved")
            sys.exit(0)
        else:
            print("✗ Sentiment crawler failed or returned no data")
            sys.exit(0)  # Exit 0 for graceful degradation
            
    except Exception as e:
        print(f"✗ Sentiment crawler encountered fatal error: {e}")
        logging.exception(e)
        sys.exit(0)  # Exit 0 for graceful degradation
