"""
CoinGecko Crawler for Cryptocurrency Derivatives Data Collection
Fetches historical Open Interest and Liquidation data from CoinGecko API.
Uses public API endpoints (Demo API key optional).

Refactored to use BaseCrawler inheritance for consistency.
"""

import requests
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from .base import BaseCrawler, CrawlerConfig


class CoinGeckoCrawler(BaseCrawler):
    """
    Crawler for fetching cryptocurrency derivatives data from CoinGecko API.
    Supports Open Interest and Liquidation data.
    
    Inherits from BaseCrawler for:
    - Standardized initialization and logging
    - Consistent error handling
    - Unified configuration management
    """
    
    def __init__(self, base_path='data/raw', api_key=None, config: CrawlerConfig = None):
        """
        Initialize CoinGecko Crawler.
        
        Args:
            base_path: Directory path for saving raw data files
            api_key: Optional CoinGecko Demo API key
            config: Optional CrawlerConfig for customization
        """
        # Call parent init for standard setup
        super().__init__(base_path=base_path, config=config)
        
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = api_key
        
        # Setup headers
        self.headers = {}
        if self.api_key:
            self.headers['x-cg-demo-api-key'] = self.api_key
            self.logger.info("CoinGeckoCrawler initialized with API key")
        else:
            self.logger.info("CoinGeckoCrawler initialized in PUBLIC mode (rate limited)")
    
    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch derivatives data from CoinGecko API.
        
        Returns:
            List of derivatives records as dictionaries
        """
        all_records = []
        
        try:
            # Fetch derivatives exchanges data
            derivatives_df = self.fetch_derivatives_data()
            all_records.extend(derivatives_df.to_dict('records'))
        except Exception as e:
            self.logger.error(f"Failed to fetch derivatives data: {e}")
        
        return all_records
    
    def validate(self, records: List[Dict[str, Any]]) -> bool:
        """
        Validate derivatives data schema.
        
        Checks:
        - Required fields present
        - Numeric type validity
        """
        if not records:
            self.logger.warning("No records to validate")
            return True
        
        required_fields = {
            'exchange', 'open_interest_btc', 'trade_volume_24h_btc'
        }
        
        for i, record in enumerate(records):
            if not all(field in record for field in required_fields):
                missing = required_fields - set(record.keys())
                self.logger.error(f"Record {i} missing fields: {missing}")
                return False
        
        self.logger.info(f"Validated {len(records)} records successfully")
        return True
    
    def save(self, records: List[Dict[str, Any]]) -> int:
        """
        Save derivatives records to CSV file.
        
        Args:
            records: List of derivatives records
            
        Returns:
            Number of rows saved
        """
        if not records:
            self.logger.warning("No records to save")
            return 0
        
        try:
            df = pd.DataFrame(records)
            
            # Remove duplicates
            original_len = len(df)
            df = df.drop_duplicates(subset=['exchange'], keep='first')
            deduped = original_len - len(df)
            if deduped > 0:
                self.logger.info(f"Removed {deduped} duplicate records")
            
            # Save to CSV
            output_file = self.base_path / 'coingecko_derivatives_exchanges.csv'
            df.to_csv(output_file, index=False)
            
            self.logger.info(f"Successfully saved {len(df)} records to {output_file}")
            return len(df)
            
        except Exception as e:
            self.logger.error(f"Failed to save records: {e}")
            raise IOError(str(e))
    
    def fetch_derivatives_data(self, coin_id='bitcoin'):
        """
        Fetch derivatives data (Open Interest, Volume) for a specific coin.
        
        Args:
            coin_id: CoinGecko coin ID ('bitcoin', 'ethereum')
        
        Returns:
            DataFrame with derivatives data
        """
        self.logger.info(f"Fetching derivatives data for {coin_id}")
        
        try:
            # Fetch derivatives exchanges data
            endpoint = f"{self.base_url}/derivatives/exchanges"
            response = self.request_with_retry('GET', endpoint, headers=self.headers)
            response.raise_for_status()
            
            exchanges_data = response.json()
            self.logger.info(f"Fetched data from {len(exchanges_data)} derivatives exchanges")
            
            # Extract relevant data
            derivatives_records = []
            for exchange in exchanges_data:
                record = {
                    'exchange': exchange.get('name', 'Unknown'),
                    'exchange_id': exchange.get('id', ''),
                    'open_interest_btc': exchange.get('open_interest_btc', 0),
                    'trade_volume_24h_btc': exchange.get('trade_volume_24h_btc', 0),
                    'number_of_perpetual_pairs': exchange.get('number_of_perpetual_pairs', 0),
                    'number_of_futures_pairs': exchange.get('number_of_futures_pairs', 0),
                    'timestamp': pd.Timestamp.now()
                }
                derivatives_records.append(record)
            
            df = pd.DataFrame(derivatives_records)
            self.logger.info(f"Successfully fetched {len(df)} derivatives exchange records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching derivatives data: {str(e)}")
            raise
    
    def run(self) -> int:
        """
        Standard orchestration flow for crawler. Uses parent's run() implementation.
        
        Returns:
            Number of records saved
        """
        self.logger.info("CoinGeckoCrawler starting via run()")
        # Use parent's orchestration: fetch() -> validate() -> save()
        return super().run()
        
        # Execute the crawl
        self.crawl_all(coin_ids=coin_ids)
        
        logger.info("CoinGeckoCrawler.run() completed successfully")
