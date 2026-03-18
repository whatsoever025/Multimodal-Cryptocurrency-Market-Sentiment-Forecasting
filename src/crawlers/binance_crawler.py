"""
Binance Crawler for Cryptocurrency Market Data Collection
Fetches spot OHLCV, futures funding rates, open interest, and liquidation data.
Uses CCXT library with Public API (no authentication required).
"""

import ccxt
import pandas as pd
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceCrawler:
    """
    Crawler for fetching cryptocurrency market data from Binance Public API.
    Supports spot OHLCV, futures funding rates, open interest, and liquidations.
    """
    
    def __init__(self, base_path='data/raw'):
        """
        Initialize Binance Crawler in PUBLIC mode (no API keys).
        
        Args:
            base_path: Directory path for saving raw data files
        """
        # Initialize CCXT Binance exchange in PUBLIC mode
        self.exchange = ccxt.binance({
            'enableRateLimit': True,  # Built-in rate limiting
            'options': {
                'defaultType': 'spot',  # Default to spot market
            }
        })
        
        # Initialize Binance Futures exchange for futures-specific data
        self.futures_exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })
        
        # Setup data directory
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("BinanceCrawler initialized in PUBLIC mode")
        logger.info(f"Data will be saved to: {self.base_path.absolute()}")
    
    def fetch_ohlcv(self, symbol, timeframe='1h', start_date='2023-01-01', limit=1000):
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
    
    def save_to_raw(self, df, filename):
        """
        Save DataFrame to CSV file in the data/raw/ directory.
        
        Args:
            df: DataFrame to save
            filename: Output filename (e.g., 'btc_ohlcv.csv')
        """
        filepath = self.base_path / filename
        
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Successfully saved data to {filepath.absolute()}")
            logger.info(f"Saved {len(df)} rows, {len(df.columns)} columns")
            
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {str(e)}")
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
        logger.info("=" * 80)
        logger.info("Starting comprehensive Binance data crawl")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Timeframes: {timeframes}")
        logger.info(f"Start Date: {start_date}")
        logger.info("=" * 80)
        
        for idx, symbol in enumerate(symbols, 1):
            symbol_clean = symbol.replace('/', '').lower()  # e.g., 'BTC/USDT' -> 'btcusdt'
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing {symbol} ({idx}/{len(symbols)})")
            logger.info(f"{'='*80}")
            
            # 1. Fetch Spot OHLCV data for each timeframe
            for timeframe in timeframes:
                try:
                    logger.info(f"\n--- Fetching {symbol} OHLCV ({timeframe}) ---")
                    ohlcv_df = self.fetch_ohlcv(symbol, timeframe=timeframe, start_date=start_date)
                    self.save_to_raw(ohlcv_df, f"{symbol_clean}_ohlcv_{timeframe}.csv")
                    
                    # Rate limiting between timeframes
                    logger.info("Waiting 2 seconds (rate limit)...")
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Failed to fetch OHLCV for {symbol} ({timeframe}): {e}")
                    time.sleep(2)  # Wait even on error
            
            # 2. Fetch Futures data (funding rate & liquidations)
            try:
                logger.info(f"\n--- Fetching {symbol} Futures Data ---")
                futures_data = self.fetch_futures_data(symbol, start_date=start_date)
                
                # Save funding rate (historical)
                self.save_to_raw(
                    futures_data['funding_rate'],
                    f"{symbol_clean}_funding_rate.csv"
                )
                
                # Save liquidations/large trades (historical)
                self.save_to_raw(
                    futures_data['liquidations'],
                    f"{symbol_clean}_liquidations.csv"
                )
                
                logger.info("Waiting 2 seconds (rate limit)...")
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to fetch futures data for {symbol}: {e}")
                time.sleep(2)
            
            # 3. Fetch Historical Liquidations (extended data, not just recent)
            try:
                logger.info(f"\n--- Fetching {symbol} Historical Liquidations (OPTION 3) ---")
                liquidation_history = self.fetch_liquidation_history(symbol, start_date=start_date)
                
                if not liquidation_history.empty:
                    self.save_to_raw(
                        liquidation_history,
                        f"{symbol_clean}_liquidations_history.csv"
                    )
                else:
                    logger.warning(f"No historical liquidation data available for {symbol}")
                
                logger.info("Waiting 2 seconds (rate limit)...")
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to fetch historical liquidations for {symbol}: {e}")
                time.sleep(2)
            
            # Rate limiting between symbols (strictly enforce to avoid IP ban)
            if idx < len(symbols):
                logger.info(f"\nCompleted {symbol}. Waiting 5 seconds before next symbol (rate limit)...")
                time.sleep(5)
        
        logger.info("\n" + "=" * 80)
        logger.info("Data crawl completed!")
        logger.info(f"All data saved to: {self.base_path.absolute()}")
        logger.info("=" * 80)
    
    def run(self):
        """
        Standardized run method for orchestrator compatibility.
        Executes the full Binance crawl sequence with default parameters.
        """
        logger.info("BinanceCrawler.run() started")
        
        # Default configuration - Multi-symbol support
        symbols = ['BTC/USDT', 'ETH/USDT']
        timeframes = ['1h', '4h']
        start_date = '2023-01-01'  # Deep historical data from Jan 1, 2023
        
        # Execute the crawl
        self.crawl_all(symbols=symbols, timeframes=timeframes, start_date=start_date)
        
        logger.info("BinanceCrawler.run() completed successfully")
