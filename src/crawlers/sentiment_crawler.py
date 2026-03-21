"""
Sentiment Crawler for Cryptocurrency Market Sentiment Data
Fetches Fear & Greed Index from Alternative.me API.

Refactored to use BaseCrawler inheritance for consistency.
"""

import requests
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from .base import BaseCrawler, CrawlerConfig


class SentimentCrawler(BaseCrawler):
    """
    Crawler for fetching cryptocurrency market sentiment data.
    Focuses on Fear & Greed Index from Alternative.me API.
    
    Inherits from BaseCrawler for:
    - Standardized initialization and logging
    - Consistent error handling
    - Unified configuration
    """
    
    def __init__(self, base_path='data/raw', config: CrawlerConfig = None):
        """
        Initialize Sentiment Crawler.
        
        Args:
            base_path: Directory path for saving raw data files
            config: Optional CrawlerConfig for customization
        """
        # Call parent init for standard setup
        super().__init__(base_path=base_path, config=config)
        
        self.base_url = "https://api.alternative.me"
    
    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch Fear & Greed Index data from API.
        
        Returns:
            List of sentiment records as dictionaries
        """
        try:
            df = self.fetch_fear_greed_index(limit=0)
            return df.to_dict('records')
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
        """
        if not records:
            self.logger.warning("No records to validate")
            return True
        
        required_fields = {'value', 'value_classification', 'timestamp'}
        
        for i, record in enumerate(records):
            if not all(field in record for field in required_fields):
                missing = required_fields - set(record.keys())
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
    
    def save(self, records: List[Dict[str, Any]]) -> int:
        """
        Save sentiment records to CSV file.
        
        Args:
            records: List of sentiment records
            
        Returns:
            Number of rows saved
        """
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
            output_file = self.base_path / 'fear_greed_index.csv'
            df.to_csv(output_file, index=False)
            
            self.logger.info(f"Successfully saved {len(df)} records to {output_file}")
            return len(df)
            
        except Exception as e:
            self.logger.error(f"Failed to save records: {e}")
            raise IOError(str(e))
    
    def fetch_fear_greed_index(self, limit=0):
        """
        Fetch Fear & Greed Index historical data.
        
        Args:
            limit: Number of records to fetch (0 = all available history)
        
        Returns:
            DataFrame with Fear & Greed Index data
        """
        self.logger.info(f"Fetching Fear & Greed Index (limit={limit if limit > 0 else 'ALL'})")
        
        try:
            endpoint = f"{self.base_url}/fng/"
            params = {'limit': limit}
            
            response = self.request_with_retry('GET', endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data:
                raise ValueError("Invalid response format from Fear & Greed API")
            
            records = []
            for entry in tqdm(data['data'], desc="Processing Fear & Greed data"):
                record = {
                    'value': int(entry.get('value', 0)),
                    'value_classification': entry.get('value_classification', 'Unknown'),
                    'timestamp': int(entry.get('timestamp', 0)),
                    'time_until_update': entry.get('time_until_update', None)
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Sort by timestamp (oldest first)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Successfully fetched {len(df)} Fear & Greed Index records")
            if len(df) > 0:
                self.logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching Fear & Greed Index: {str(e)}")
            raise
    
    def fetch_sentiment_metrics(self):
        """
        Fetch current sentiment snapshot with detailed metrics.
        
        Returns:
            DataFrame with current sentiment metrics
        """
        logger.info("Fetching current sentiment metrics")
        
        try:
            # Get latest Fear & Greed reading with details
            endpoint = f"{self.base_url}/fng/"
            params = {'limit': 1, 'format': 'json'}
            
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data or len(data['data']) == 0:
                raise ValueError("No current sentiment data available")
            
            current = data['data'][0]
            metadata = data.get('metadata', {})
            
            record = {
                'value': int(current.get('value', 0)),
                'value_classification': current.get('value_classification', 'Unknown'),
                'timestamp': int(current.get('timestamp', 0)),
                'datetime': pd.to_datetime(int(current.get('timestamp', 0)), unit='s'),
                'time_until_update': current.get('time_until_update', None),
                'error': metadata.get('error', None)
            }
            
            df = pd.DataFrame([record])
            
            logger.info(f"Current Fear & Greed Index: {record['value']} ({record['value_classification']})")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching sentiment metrics: {str(e)}")
            raise
    
    def save_to_raw(self, df, filename):
        """
        Save DataFrame to CSV file in the data/raw/ directory.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        filepath = self.base_path / filename
        
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Successfully saved data to {filepath.absolute()}")
            logger.info(f"Saved {len(df)} rows, {len(df.columns)} columns")
            
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {str(e)}")
            raise
    
    def crawl_all(self):
        """
        Comprehensive crawl: fetch all sentiment data.
        """
        logger.info("=" * 80)
        logger.info("Starting Sentiment data crawl")
        logger.info("=" * 80)
        
        # 1. Fetch complete Fear & Greed Index history
        try:
            logger.info("\n--- Fetching Fear & Greed Index History ---")
            fg_history_df = self.fetch_fear_greed_index(limit=0)
            self.save_to_raw(fg_history_df, "fear_greed_index.csv")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed Index history: {e}")
        
        # 2. Fetch current sentiment snapshot
        try:
            logger.info("\n--- Fetching Current Sentiment Snapshot ---")
            current_sentiment_df = self.fetch_sentiment_metrics()
            self.save_to_raw(current_sentiment_df, "sentiment_current.csv")
        except Exception as e:
            logger.error(f"Failed to fetch current sentiment: {e}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Sentiment crawl completed!")
        logger.info(f"All data saved to: {self.base_path.absolute()}")
        logger.info("=" * 80)
    
    def run(self) -> int:
        """
        Standard orchestration flow for crawler. Uses parent's run() implementation.
        
        Returns:
            Number of records saved
        """
        self.logger.info("SentimentCrawler starting via run()")
        # Use parent's orchestration: fetch() -> validate() -> save()
        return super().run()
        
        # Execute the crawl
        self.crawl_all()
        
        logger.info("SentimentCrawler.run() completed successfully")
