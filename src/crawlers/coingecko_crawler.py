"""
CoinGecko Crawler for Cryptocurrency Derivatives Data Collection
Fetches historical Open Interest and Liquidation data from CoinGecko API.
Uses public API endpoints (Demo API key optional).
"""

import requests
import pandas as pd
import time
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CoinGeckoCrawler:
    """
    Crawler for fetching cryptocurrency derivatives data from CoinGecko API.
    Supports Open Interest and Liquidation data.
    """
    
    def __init__(self, base_path='data/raw', api_key=None):
        """
        Initialize CoinGecko Crawler.
        
        Args:
            base_path: Directory path for saving raw data files
            api_key: Optional CoinGecko Demo API key
        """
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = api_key
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Setup headers
        self.headers = {}
        if self.api_key:
            self.headers['x-cg-demo-api-key'] = self.api_key
            logger.info("CoinGeckoCrawler initialized with API key")
        else:
            logger.info("CoinGeckoCrawler initialized in PUBLIC mode (rate limited)")
        
        logger.info(f"Data will be saved to: {self.base_path.absolute()}")
    
    def fetch_derivatives_data(self, coin_id='bitcoin'):
        """
        Fetch derivatives data (Open Interest, Volume) for a specific coin.
        
        Args:
            coin_id: CoinGecko coin ID ('bitcoin', 'ethereum')
        
        Returns:
            DataFrame with derivatives data
        """
        logger.info(f"Fetching derivatives data for {coin_id}")
        
        try:
            # Fetch derivatives exchanges data
            endpoint = f"{self.base_url}/derivatives/exchanges"
            response = requests.get(endpoint, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            exchanges_data = response.json()
            logger.info(f"Fetched data from {len(exchanges_data)} derivatives exchanges")
            
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
                    'year_established': exchange.get('year_established', None),
                    'country': exchange.get('country', 'Unknown'),
                    'timestamp': pd.Timestamp.now()
                }
                derivatives_records.append(record)
            
            df = pd.DataFrame(derivatives_records)
            df['datetime'] = df['timestamp']
            
            logger.info(f"Successfully fetched {len(df)} derivatives exchange records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching derivatives data for {coin_id}: {str(e)}")
            raise
    
    def fetch_coin_derivatives_tickers(self, coin_id='bitcoin'):
        """
        Fetch derivatives tickers for a specific coin.
        
        Args:
            coin_id: CoinGecko coin ID ('bitcoin', 'ethereum')
        
        Returns:
            DataFrame with derivatives tickers data
        """
        logger.info(f"Fetching derivatives tickers for {coin_id}")
        
        try:
            # Map coin_id to symbol for filtering
            coin_symbol_map = {'bitcoin': 'BTC', 'ethereum': 'ETH'}
            symbol = coin_symbol_map.get(coin_id, coin_id.upper())
            
            endpoint = f"{self.base_url}/derivatives"
            response = requests.get(endpoint, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            tickers_data = response.json()
            logger.info(f"Fetched {len(tickers_data)} total derivatives tickers")
            
            # Filter for specific coin
            filtered_tickers = []
            for ticker in tqdm(tickers_data, desc=f"Filtering {coin_id} tickers"):
                if symbol in ticker.get('symbol', '').upper():
                    record = {
                        'symbol': ticker.get('symbol', ''),
                        'contract_type': ticker.get('contract_type', ''),
                        'market': ticker.get('market', ''),
                        'index': ticker.get('index', 0),
                        'index_basis_percentage': ticker.get('index_basis_percentage', 0),
                        'bid_ask_spread': ticker.get('bid_ask_spread', 0),
                        'funding_rate': ticker.get('funding_rate', 0),
                        'open_interest_usd': ticker.get('open_interest_usd', 0),
                        'h24_volume': ticker.get('converted_volume', {}).get('usd', 0),
                        'last_traded': ticker.get('last_traded', None),
                        'timestamp': pd.Timestamp.now()
                    }
                    filtered_tickers.append(record)
            
            df = pd.DataFrame(filtered_tickers)
            if not df.empty:
                df['datetime'] = df['timestamp']
            
            logger.info(f"Successfully fetched {len(df)} derivatives tickers for {coin_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching derivatives tickers for {coin_id}: {str(e)}")
            raise
    
    def fetch_historical_derivatives(self, coin_id='bitcoin', days=90, type_filter='open_interest'):
        """
        Fetch historical derivatives data from CoinGecko (OPTION 3 - Historical Exchange APIs).
        Retrieves time-series derivatives metrics across all exchanges.
        
        Args:
            coin_id: CoinGecko coin ID ('bitcoin', 'ethereum')
            days: Number of days back to fetch (max ~365)
            type_filter: Type of data ('open_interest', 'trading_volume', etc.)
        
        Returns:
            DataFrame with historical derivatives data
        """
        import json
        from datetime import datetime, timedelta
        
        logger.info(f"Fetching {days}-day historical {type_filter} for {coin_id}")
        
        try:
            # CoinGecko historical derivatives endpoint
            endpoint = f"{self.base_url}/derivatives/historical"
            params = {
                'coin_id': coin_id,
                'days': days,
                'type': type_filter  # open_interest or trading_volume
            }
            
            response = requests.get(endpoint, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse historical data
            records = []
            for entry in tqdm(data, desc=f"{coin_id} historical {type_filter}"):
                timestamp = entry[0] if isinstance(entry, list) else entry.get('timestamp')
                value = entry[1] if isinstance(entry, list) else entry.get('value')
                
                records.append({
                    'timestamp': pd.to_datetime(timestamp, unit='ms') if isinstance(timestamp, (int, float)) else pd.to_datetime(timestamp),
                    f'{type_filter}_total': float(value) if value else 0,
                    'coin_id': coin_id,
                    'datetime': pd.to_datetime(timestamp, unit='ms') if isinstance(timestamp, (int, float)) else pd.to_datetime(timestamp)
                })
            
            df = pd.DataFrame(records)
            if not df.empty:
                df = df.sort_values('timestamp').reset_index(drop=True)
                logger.info(f"Successfully fetched {len(df)} historical {type_filter} records for {coin_id}")
                logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            else:
                logger.warning(f"No historical data found for {coin_id}")
            
            return df
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Historical data endpoint not available for {coin_id} (404)")
                return pd.DataFrame()
            else:
                logger.error(f"HTTP error fetching historical derivatives for {coin_id}: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Error fetching historical derivatives for {coin_id}: {str(e)}")
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
    
    def crawl_all(self, coin_ids=['bitcoin', 'ethereum']):
        """
        Comprehensive crawl: fetch derivatives data for specified coins.
        Includes historical trading volume data.
        
        Args:
            coin_ids: List of CoinGecko coin IDs to fetch
        """
        logger.info("=" * 80)
        logger.info("Starting CoinGecko derivatives data crawl")
        logger.info(f"Coins: {coin_ids}")
        logger.info("=" * 80)
        
        # Fetch global derivatives exchanges data (once)
        try:
            logger.info("\n--- Fetching Global Derivatives Exchanges Data (Current Snapshot) ---")
            derivatives_df = self.fetch_derivatives_data()
            self.save_to_raw(derivatives_df, "coingecko_derivatives_exchanges.csv")
            time.sleep(2)  # Rate limiting
        except Exception as e:
            logger.error(f"Failed to fetch derivatives exchanges data: {e}")
        
        # Fetch derivatives tickers and historical data for each coin
        for idx, coin_id in enumerate(coin_ids, 1):
            try:
                logger.info(f"\n--- Fetching {coin_id.upper()} Derivatives Tickers ({idx}/{len(coin_ids)}) ---")
                tickers_df = self.fetch_coin_derivatives_tickers(coin_id)
                
                if not tickers_df.empty:
                    self.save_to_raw(tickers_df, f"{coin_id}_derivatives_tickers.csv")
                else:
                    logger.warning(f"No derivatives tickers found for {coin_id}")
                
                # Rate limiting between tickers
                time.sleep(1)
                
                # Fetch historical trading volume
                try:
                    logger.info(f"\n--- Fetching {coin_id.upper()} Historical Trading Volume (90 days) ---")
                    volume_history = self.fetch_historical_derivatives(
                        coin_id=coin_id,
                        days=90,
                        type_filter='trading_volume'
                    )
                    
                    if not volume_history.empty:
                        self.save_to_raw(volume_history, f"{coin_id}_trading_volume_history.csv")
                    else:
                        logger.warning(f"No historical volume data for {coin_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to fetch historical volume for {coin_id}: {e}")
                
                # Rate limiting between coins
                if idx < len(coin_ids):
                    logger.info(f"\nWaiting 3 seconds before next coin (rate limit)...")
                    time.sleep(3)
                    
            except Exception as e:
                logger.error(f"Failed to fetch derivatives data for {coin_id}: {e}")
                time.sleep(3)
        
        logger.info("\n" + "=" * 80)
        logger.info("CoinGecko crawl completed!")
        logger.info(f"All data saved to: {self.base_path.absolute()}")
        logger.info("=" * 80)
    
    def run(self):
        """
        Standardized run method for orchestrator compatibility.
        """
        logger.info("CoinGeckoCrawler.run() started")
        
        # Default configuration
        coin_ids = ['bitcoin', 'ethereum']
        
        # Execute the crawl
        self.crawl_all(coin_ids=coin_ids)
        
        logger.info("CoinGeckoCrawler.run() completed successfully")
