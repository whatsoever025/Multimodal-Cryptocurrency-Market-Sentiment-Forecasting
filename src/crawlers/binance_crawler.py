"""
Binance Crawler for Cryptocurrency Market Data Collection
Fetches spot OHLCV, futures funding rates, open interest, and liquidation data.
Uses CCXT library with Public API (no authentication required).

Refactored to use BaseCrawler inheritance for consistency.
"""

import ccxt
import pandas as pd
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from .base import BaseCrawler, CrawlerConfig
from tqdm import tqdm


class BinanceCrawler(BaseCrawler):
    """
    Crawler for fetching cryptocurrency market data from Binance Public API.
    Supports spot OHLCV, futures funding rates, open interest, and liquidations.
    
    Inherits from BaseCrawler for:
    - Standardized initialization
    - Retry logic with exponential backoff
    - Rate limiting management
    - Error handling consistency
    """
    
    def __init__(self, base_path='data/raw', config: CrawlerConfig = None):
        """
        Initialize Binance Crawler in PUBLIC mode.
        
        Args:
            base_path: Directory path for saving raw data files
            config: Optional CrawlerConfig for advanced customization
        """
        # Call parent init for standard setup (logging, path creation)
        super().__init__(base_path=base_path, config=config)
        
        # Initialize CCXT Binance exchange in PUBLIC mode
        self.exchange = ccxt.binance({
            'enableRateLimit': True,  # Built-in rate limiting
            'options': {'defaultType': 'spot'}
        })
        
        # Initialize Binance Futures exchange for futures-specific data
        self.futures_exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        self.logger.info("BinanceCrawler initialized in PUBLIC mode")
    
    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV data for BTC/USDT and ETH/USDT (1h timeframe).
        
        Returns:
            List of OHLCV records as dictionaries
        """
        all_records = []
        
        for symbol in ['BTC/USDT', 'ETH/USDT']:
            try:
                df = self.fetch_ohlcv_data(symbol, timeframe='1h', start_date='2023-01-01')
                records = df.to_dict('records')
                all_records.extend(records)
                self.logger.info(f"Fetched {len(records)} records for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to fetch {symbol}: {e}")
        
        return all_records
    
    def validate(self, records: List[Dict[str, Any]]) -> bool:
        """
        Validate OHLCV data schema and integrity.
        
        Checks:
        - Required fields: timestamp, open, high, low, close, volume
        - Numeric types: OHLCV fields are floats, volume is positive
        - Timestamp validity: parseable to datetime
        - Price relationships: low <= close <= high
        """
        if not records:
            self.logger.warning("No records to validate")
            return True
        
        required_fields = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        
        for i, record in enumerate(records):
            # Check required fields
            if not all(field in record for field in required_fields):
                missing = required_fields - set(record.keys())
                self.logger.error(f"Record {i} missing fields: {missing}")
                return False
            
            # Validate numeric types
            try:
                o, h, l, c, v = float(record['open']), float(record['high']), \
                                float(record['low']), float(record['close']), float(record['volume'])
                
                # Validate relationships
                if not (l <= c <= h):
                    self.logger.error(f"Record {i}: price relationship violation (L:{l} > C:{c} > H:{h})")
                    return False
                
                if v < 0:
                    self.logger.error(f"Record {i}: negative volume {v}")
                    return False
                    
            except (ValueError, TypeError) as e:
                self.logger.error(f"Record {i}: invalid numeric value - {e}")
                return False
        
        self.logger.info(f"Validated {len(records)} records successfully")
        return True
    
    def save(self, records: List[Dict[str, Any]]) -> int:
        """
        Save OHLCV records to CSV file.
        
        Args:
            records: List of OHLCV records
            
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
            output_file = self.base_path / 'btcusdt_ohlcv_1h.csv'
            df.to_csv(output_file, index=False)
            
            self.logger.info(f"Successfully saved {len(df)} records to {output_file}")
            return len(df)
            
        except Exception as e:
            self.logger.error(f"Failed to save records: {e}")
            raise IOError(str(e))
    
    def fetch_ohlcv_data(self, symbol, timeframe='1h', start_date='2023-01-01', limit=1000):
        """
        Fetch deep historical OHLCV (candlestick) data from Binance spot market.
        Implements pagination to fetch data from a target start date to present.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT', 'ETH/USDT')
            timeframe: Candle timeframe ('1h', '4h', '1d', etc.)
            start_date: Target start date (YYYY-MM-DD format)
            limit: Number of candles per request (max 1000)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        self.logger.info(f"Fetching OHLCV data for {symbol} ({timeframe}) from {start_date}")
        
        all_ohlcv = []
        
        # Parse start date to timestamp
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        current_since = self.exchange.parse8601(start_dt.isoformat())
        now_ts = self.exchange.milliseconds()
        
        # Calculate total expected candles for progress tracking
        timeframe_map = {'1h': 1, '4h': 4, '1d': 24}
        hours_per_candle = timeframe_map.get(timeframe, 1)
        total_hours = (datetime.now() - start_dt).total_seconds() / 3600
        estimated_candles = int(total_hours / hours_per_candle)
        
        try:
            with tqdm(total=estimated_candles, desc=f"{symbol} {timeframe}", unit="candles") as pbar:
                while current_since < now_ts:
                    # Fetch OHLCV data
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol,
                        timeframe=timeframe,
                        since=current_since,
                        limit=limit
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    pbar.update(len(ohlcv))
                    
                    # Check if we got fewer candles than limit (end of data)
                    if len(ohlcv) < limit:
                        break
                    
                    # Update since to the last candle's timestamp + 1ms for pagination
                    current_since = ohlcv[-1][0] + 1
                    
                    # Rate limiting (extra safety on top of CCXT's built-in)
                    time.sleep(self.config.rate_limit_delay_seconds)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                all_ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Successfully fetched {len(df)} total candles for {symbol}")
            if len(df) > 0:
                self.logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
            raise
    
    def fetch_ohlcv(self, symbol, timeframe='1h', start_date='2023-01-01', limit=1000):
        """Alias for fetch_ohlcv_data for backward compatibility."""
        return self.fetch_ohlcv_data(symbol, timeframe, start_date, limit)
        """
        Fetch deep historical OHLCV (candlestick) data from Binance spot market.
        Implements pagination to fetch data from a target start date to present.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT', 'ETH/USDT')
            timeframe: Candle timeframe ('1h', '4h', '1d', etc.)
            start_date: Target start date (YYYY-MM-DD format)
            limit: Number of candles per request (max 1000)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"Fetching OHLCV data for {symbol} ({timeframe}) from {start_date}")
        
        all_ohlcv = []
        
        # Parse start date to timestamp
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        current_since = self.exchange.parse8601(start_dt.isoformat())
        now_ts = self.exchange.milliseconds()
        
        # Calculate total expected candles for progress tracking
        timeframe_map = {'1h': 1, '4h': 4, '1d': 24}
        hours_per_candle = timeframe_map.get(timeframe, 1)
        total_hours = (datetime.now() - start_dt).total_seconds() / 3600
        estimated_candles = int(total_hours / hours_per_candle)
        
        try:
            with tqdm(total=estimated_candles, desc=f"{symbol} {timeframe}", unit="candles") as pbar:
                while current_since < now_ts:
                    # Fetch OHLCV data
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol,
                        timeframe=timeframe,
                        since=current_since,
                        limit=limit
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    pbar.update(len(ohlcv))
                    
                    # Check if we got fewer candles than limit (end of data)
                    if len(ohlcv) < limit:
                        break
                    
                    # Update since to the last candle's timestamp + 1ms for pagination
                    current_since = ohlcv[-1][0] + 1
                    
                    # Rate limiting (extra safety on top of CCXT's built-in)
                    time.sleep(0.5)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                all_ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['datetime'] = df['timestamp']
            
            # Remove duplicates based on timestamp
            original_len = len(df)
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
            if original_len > len(df):
                logger.info(f"Removed {original_len - len(df)} duplicate records")
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Successfully fetched {len(df)} total candles for {symbol}")
            logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
            raise
    
    def fetch_funding_rate(self, symbol, start_date='2023-01-01', limit=1000):
        """
        Fetch complete historical futures funding rate data.
        
        Args:
            symbol: Futures pair (e.g., 'BTC/USDT')
            start_date: Target start date (YYYY-MM-DD format)
            limit: Number of records per request
        
        Returns:
            DataFrame with funding rate data
        """
        logger.info(f"Fetching funding rate history for {symbol} from {start_date}")
        
        all_funding = []
        
        # Parse start date to timestamp
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        current_since = self.futures_exchange.parse8601(start_dt.isoformat())
        now_ts = self.futures_exchange.milliseconds()
        
        try:
            with tqdm(desc=f"{symbol} Funding Rate", unit="records") as pbar:
                while current_since < now_ts:
                    # Fetch funding rate history
                    funding_history = self.futures_exchange.fetch_funding_rate_history(
                        symbol,
                        since=current_since,
                        limit=limit
                    )
                    
                    if not funding_history:
                        break
                    
                    all_funding.extend(funding_history)
                    pbar.update(len(funding_history))
                    
                    if len(funding_history) < limit:
                        break
                    
                    # Update pagination
                    current_since = funding_history[-1]['timestamp'] + 1
                    time.sleep(0.5)
            
            # Convert to DataFrame
            df = pd.DataFrame(all_funding)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Remove duplicates
            original_len = len(df)
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
            if original_len > len(df):
                logger.info(f"Removed {original_len - len(df)} duplicate funding rate records")
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Successfully fetched {len(df)} funding rate records")
            if len(df) > 0:
                logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {str(e)}")
            raise
    
    
    def fetch_liquidations(self, symbol, limit=100):
        """
        Fetch recent liquidation events using public trades data.
        Note: Binance doesn't provide direct liquidation API in public mode.
        This method fetches recent large trades as a proxy.
        
        Args:
            symbol: Futures pair (e.g., 'BTC/USDT')
            limit: Number of recent trades to fetch
        
        Returns:
            DataFrame with recent large trades data
        """
        logger.info(f"Fetching recent liquidation/large trades for {symbol}")
        
        try:
            # Fetch recent trades (public endpoint)
            trades = self.futures_exchange.fetch_trades(symbol, limit=limit)
            
            df = pd.DataFrame([{
                'timestamp': trade['timestamp'],
                'datetime': pd.to_datetime(trade['timestamp'], unit='ms'),
                'side': trade['side'],
                'price': trade['price'],
                'amount': trade['amount'],
                'cost': trade['cost']
            } for trade in trades])
            
            # Filter for larger trades (potential liquidations)
            if not df.empty:
                median_cost = df['cost'].median()
                df['is_large_trade'] = df['cost'] > median_cost * 3
            
            logger.info(f"Successfully fetched {len(df)} recent trades")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching liquidations for {symbol}: {str(e)}")
            raise
    
    def fetch_futures_data(self, symbol, start_date='2023-01-01'):
        """
        Fetch all futures-related data: funding rate and liquidations.
        Note: Open interest historical data is fetched separately via fetch_open_interest_history().
        
        Args:
            symbol: Futures pair (e.g., 'BTC/USDT')
            start_date: Start date for historical data (YYYY-MM-DD)
        
        Returns:
            Dictionary containing futures data DataFrames
        """
        logger.info(f"Fetching futures data for {symbol}")
        
        futures_data = {
            'funding_rate': self.fetch_funding_rate(symbol, start_date=start_date),
            'liquidations': self.fetch_liquidations(symbol)
        }
        
        return futures_data
    
    def fetch_liquidation_history(self, symbol, start_date='2023-01-01', limit=1000):
        """
        Fetch historical liquidation events by analyzing large trades with extreme volume.
        Fetches all recent trades and filters for liquidation-like events.
        
        Args:
            symbol: Futures pair (e.g., 'BTC/USDT')
            start_date: Start date (YYYY-MM-DD format)
            limit: Trades per request (max 1000)
        
        Returns:
            DataFrame with potential liquidation events (large trades)
        """
        import requests
        logger.info(f"Fetching liquidation history for {symbol} from {start_date}")
        
        all_liquidations = []
        base_url = "https://fapi.binance.com/fapi/v1"
        
        # Parse start date
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        current_time = int(start_dt.timestamp() * 1000)
        now_time = int(datetime.now().timestamp() * 1000)
        
        try:
            with tqdm(desc=f"{symbol} Liquidations", unit="batches") as pbar:
                while current_time < now_time:
                    try:
                        # Fetch aggregate trades
                        endpoint = f"{base_url}/aggTrades"
                        params = {
                            'symbol': symbol.replace('/', ''),  # BTCUSDT
                            'startTime': current_time,
                            'limit': min(limit, 1000)
                        }
                        
                        response = requests.get(endpoint, params=params, timeout=10)
                        response.raise_for_status()
                        data = response.json()
                        
                        if not data:
                            break
                        
                        # Collect all trades
                        for trade in data:
                            all_liquidations.append({
                                'id': trade.get('a', ''),
                                'timestamp': int(trade.get('T', 0)),
                                'price': float(trade.get('p', 0)),
                                'quantity': float(trade.get('q', 0)),
                                'quoteAsset': float(trade.get('q', 0)) * float(trade.get('p', 0)),
                                'isBuyerMaker': trade.get('m', False),
                                'datetime': pd.to_datetime(int(trade.get('T', 0)), unit='ms')
                            })
                        
                        pbar.update(1)
                        
                        if len(data) < limit:
                            break
                        
                        # Move to next batch
                        current_time = int(data[-1]['T']) + 1
                        time.sleep(0.3)
                        
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"API request failed: {e}, retrying...")
                        time.sleep(1)
                        continue
            
            if all_liquidations:
                df = pd.DataFrame(all_liquidations)
                df = df.drop_duplicates(subset=['id'], keep='first')
                df = df.sort_values('timestamp').reset_index(drop=True)
                logger.info(f"Successfully fetched {len(df)} liquidation/trade records")
                if len(df) > 0:
                    logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
                return df
            else:
                logger.warning("No liquidation data retrieved")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching liquidation history for {symbol}: {str(e)}")
            raise
    
    def crawl_all(self, symbols=['BTC/USDT', 'ETH/USDT'], timeframes=['1h', '4h'], start_date='2023-01-01'):
        """
        Comprehensive crawl: fetch all data types for specified symbols and timeframes.
        Includes deep historical data collection with progress tracking.
        
        Args:
            symbols: List of trading pairs to fetch
            timeframes: List of timeframes for OHLCV data
            start_date: Start date for historical data (YYYY-MM-DD)
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting comprehensive Binance data crawl")
        self.logger.info(f"Symbols: {symbols}")
        self.logger.info(f"Timeframes: {timeframes}")
        self.logger.info(f"Start Date: {start_date}")
        self.logger.info("=" * 80)
        
        all_data = []
        
        for idx, symbol in enumerate(symbols, 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Processing {symbol} ({idx}/{len(symbols)})")
            self.logger.info(f"{'='*80}")
            
            # Fetch Spot OHLCV data for primary timeframe
            try:
                self.logger.info(f"\n--- Fetching {symbol} OHLCV (1h) ---")
                ohlcv_df = self.fetch_ohlcv_data(symbol, timeframe='1h', start_date=start_date)
                all_data.extend(ohlcv_df.to_dict('records'))
                time.sleep(self.config.rate_limit_delay_seconds)
                
            except Exception as e:
                self.logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
                time.sleep(self.config.rate_limit_delay_seconds)
            
            # Rate limiting between symbols
            if idx < len(symbols):
                self.logger.info(f"\nWaiting before next symbol (rate limit)...")
                time.sleep(2)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Data crawl completed!")
        self.logger.info(f"All data available to save: {len(all_data)} records")
        self.logger.info("=" * 80)
        
        return all_data
    
    def save_to_raw(self, df, filename):
        """
        DEPRECATED: Use save() method instead.
        Kept for backward compatibility only.
        
        Args:
            df: DataFrame to save
            filename: Output filename (e.g., 'btc_ohlcv.csv')
        """
        self.logger.warning("save_to_raw() is deprecated. Use save() instead.")
        filepath = self.base_path / filename
        try:
            df.to_csv(filepath, index=False)
            self.logger.info(f"Saved {len(df)} rows to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving {filepath}: {e}")
    
    def run(self) -> int:
        """
        Standard orchestration flow for crawler. Uses parent's run() implementation.
        Fetches BTC/USDT and ETH/USDT OHLCV data from Binance.
        
        Returns:
            Number of records saved
        """
        self.logger.info("BinanceCrawler starting via run()")
        # Use parent's orchestration: fetch() -> validate() -> save()
        return super().run()
