"""
Binance Vision Crawler for Historical Monthly Data Collection.

Downloads and extracts monthly historical data directly from Binance Vision for:
- klines (1h): OHLCV candlestick data
- fundingRate: Futures funding rates
- openInterestHist: Open interest history
- liquidationSnapshot: Liquidation events

Uses in-memory processing with pandas for efficient data handling.
"""

import os
import logging
import requests
import zipfile
import io
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

from .base import BaseCrawler, CrawlerConfig


class BinanceVisionCrawler(BaseCrawler):
    """
    Crawler for downloading and processing monthly historical data from Binance Vision.
    
    Supports:
    - Multiple data types: klines, fundingRate, openInterestHist, liquidationSnapshot
    - Multiple symbols: BTCUSDT, ETHUSDT
    - In-memory ZIP extraction for efficiency
    - Automatic data formatting and validation
    """
    
    # Base URL for Binance Vision data
    BASE_URL = "https://data.binance.vision/data/futures/um/monthly"
    
    # Target symbols and data types
    TARGET_SYMBOLS = ['BTCUSDT', 'ETHUSDT']
    TARGET_TYPES = ['klines', 'fundingRate', 'openInterestHist', 'liquidationSnapshot']
    
    def __init__(self, base_path: str = 'data/raw', config: CrawlerConfig = None):
        """
        Initialize Binance Vision Crawler.
        
        Args:
            base_path: Directory path for saving raw data files
            config: Optional CrawlerConfig for advanced customization
        """
        super().__init__(base_path=base_path, config=config)
        self.logger = logging.getLogger(__name__)
        self.logger.info("BinanceVisionCrawler initialized")
    
    def _generate_monthly_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        Generate a list of monthly date strings in YYYY-MM format.
        
        Args:
            start_date: Start date as string (e.g., '2023-01-01')
            end_date: End date as string (e.g., '2026-03-01')
        
        Returns:
            List of strings in YYYY-MM format representing each month
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        months = []
        current = start
        
        while current <= end:
            months.append(current.strftime('%Y-%m'))
            current += relativedelta(months=1)
        
        return months
    
    def _get_monthly_url(self, symbol: str, data_type: str, month_str: str) -> str:
        """
        Construct the Binance Vision monthly data download URL.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            data_type: Data type ('klines', 'fundingRate', 'openInterestHist', 'liquidationSnapshot')
            month_str: Month string in YYYY-MM format
        
        Returns:
            Full URL for downloading the monthly data ZIP file
        """
        if data_type == 'klines':
            # klines requires the timeframe folder
            return f"{self.BASE_URL}/klines/{symbol}/1h/{symbol}-1h-{month_str}.zip"
        else:
            # Other data types follow standard pattern
            return f"{self.BASE_URL}/{data_type}/{symbol}/{symbol}-{data_type}-{month_str}.zip"
    
    def _download_and_parse(self, url: str) -> Optional[pd.DataFrame]:
        """
        Download URL content and extract CSV from in-memory ZIP file.
        
        Args:
            url: URL to download from Binance Vision
        
        Returns:
            DataFrame containing the parsed CSV data, or None if 404 or error
        """
        try:
            response = requests.get(url, timeout=(5.0, 25.0))
            
            if response.status_code == 404:
                self.logger.info(f"Data not available: {url}")
                return None
            
            response.raise_for_status()
            
            # Open ZIP file in memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                # Get the first CSV file in the ZIP
                csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                
                if not csv_files:
                    self.logger.warning(f"No CSV file found in ZIP from {url}")
                    return None
                
                csv_filename = csv_files[0]
                
                # Read CSV directly into DataFrame
                with zip_file.open(csv_filename) as csv_file:
                    df = pd.read_csv(csv_file)
                    self.logger.debug(f"Parsed CSV: {csv_filename} with {len(df)} rows")
                    return df
        
        except requests.RequestException as e:
            self.logger.error(f"Request failed for {url}: {e}")
            return None
        except zipfile.BadZipFile as e:
            self.logger.error(f"Bad ZIP file from {url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing {url}: {e}")
            return None
    
    def _format_klines_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format klines data with proper column names and types.
        
        Args:
            df: Raw klines DataFrame from Binance Vision
        
        Returns:
            Formatted DataFrame with selected columns
        """
        # Rename columns according to Binance Vision klines format
        new_columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'count', 'taker_buy_volume', 
            'taker_buy_quote_volume', 'ignore'
        ]
        
        # Only rename if we have at least 12 columns
        if len(df.columns) >= len(new_columns):
            df.columns = new_columns[:len(df.columns)]
        
        # Keep only required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols].copy()
        
        return df
    
    def _format_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Format data based on type and ensure timestamp is properly converted.
        
        Args:
            df: Raw DataFrame from Binance Vision
            data_type: Type of data being processed
        
        Returns:
            Formatted DataFrame with timestamp converted to datetime
        """
        if df.empty:
            return df
        
        if data_type == 'klines':
            df = self._format_klines_data(df)
        
        # Convert timestamp column from milliseconds to datetime
        if 'timestamp' in df.columns:
            try:
                # Binance Vision timestamps are in milliseconds
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            except Exception as e:
                self.logger.warning(f"Could not convert timestamp for {data_type}: {e}")
        
        # Convert numeric columns to float
        numeric_cols = df.select_dtypes(include=['object']).columns
        for col in numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                self.logger.debug(f"Could not convert {col} to numeric: {e}")
        
        return df
    
    def _process_symbol_data_type(
        self, 
        symbol: str, 
        data_type: str, 
        start_date: str = '2023-01-01',
        end_date: str = '2026-03-01'
    ) -> Optional[pd.DataFrame]:
        """
        Download and merge all monthly data for a specific symbol and data type.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            data_type: Data type to process
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            Merged DataFrame for all months, or None if no data available
        """
        months = self._generate_monthly_dates(start_date, end_date)
        all_dfs = []
        
        self.logger.info(
            f"Processing {symbol} - {data_type}: {len(months)} months "
            f"({start_date} to {end_date})"
        )
        
        for month_str in months:
            url = self._get_monthly_url(symbol, data_type, month_str)
            
            df = self._download_and_parse(url)
            if df is not None:
                df = self._format_data(df, data_type)
                all_dfs.append(df)
                self.logger.debug(f"Downloaded {symbol} {data_type} for {month_str}: {len(df)} rows")
            else:
                self.logger.debug(f"Skipped {symbol} {data_type} for {month_str} (not available)")
        
        if not all_dfs:
            self.logger.warning(f"No data found for {symbol} - {data_type}")
            return None
        
        # Merge all monthly DataFrames
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        # Sort by timestamp and drop duplicates
        if 'timestamp' in merged_df.columns:
            merged_df = merged_df.sort_values('timestamp')
            merged_df = merged_df.drop_duplicates(subset=['timestamp'], keep='first')
        
        self.logger.info(
            f"Merged {symbol} - {data_type}: {len(merged_df)} total rows "
            f"after deduplication"
        )
        
        return merged_df
    
    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch method for BaseCrawler compatibility.
        
        Returns:
            List of records (empty, as data is saved directly to CSV)
        """
        self.run()
        return []
    
    def validate(self, records: List[Dict[str, Any]]) -> bool:
        """
        Validate method for BaseCrawler compatibility.
        
        Args:
            records: Records to validate
        
        Returns:
            True if validation passes
        """
        return True
    
    def save(self, records: List[Dict[str, Any]]) -> int:
        """
        Save method for BaseCrawler compatibility.
        Note: BinanceVisionCrawler saves data directly in run() method.
        
        Args:
            records: Records to save (unused - data saved during run)
        
        Returns:
            0 (not used in pipeline)
        """
        return 0
    
    def run(
        self,
        start_date: str = '2023-01-01',
        end_date: str = '2026-03-01'
    ) -> None:
        """
        Main pipeline: download, process, and save monthly data for all symbols and data types.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        self.logger.info("Starting BinanceVisionCrawler pipeline")
        
        # Ensure raw data directory exists
        raw_dir = Path(self.base_path)
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        failed_count = 0
        
        for symbol in self.TARGET_SYMBOLS:
            for data_type in self.TARGET_TYPES:
                try:
                    df = self._process_symbol_data_type(
                        symbol, 
                        data_type,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if df is not None and not df.empty:
                        # Save to CSV
                        output_file = raw_dir / f"{symbol}_{data_type}.csv"
                        df.to_csv(output_file, index=False)
                        self.logger.info(
                            f"Saved {symbol} - {data_type} to {output_file} "
                            f"({len(df)} rows)"
                        )
                        processed_count += 1
                    else:
                        self.logger.warning(
                            f"No data to save for {symbol} - {data_type}"
                        )
                        failed_count += 1
                
                except Exception as e:
                    self.logger.error(
                        f"Pipeline failed for {symbol} - {data_type}: {e}"
                    )
                    failed_count += 1
        
        self.logger.info(
            f"BinanceVisionCrawler pipeline completed: "
            f"{processed_count} succeeded, {failed_count} failed"
        )
