"""
Chart Generator for Converting OHLCV Data to Candlestick Chart Images
Generates professional candlestick charts for deep learning (ViT/ResNet) training.
Includes technical indicators (MA7, MA25, RSI, MACD) and uses multiprocessing for optimization.
Pre-calculates indicators on full dataset to avoid redundant computations.
"""

import pandas as pd
import numpy as np

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

import mplfinance as mpf
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from PIL import Image
import io
import warnings
import ta  # Technical Analysis Library

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Generates candlestick chart images from OHLCV data for deep learning models.
    Features: Moving averages, volume bars, sliding window, multiprocessing optimization.
    """
    
    def __init__(self, window_size=24, image_size=(224, 224), output_path='data/processed/images'):
        """
        Initialize Chart Generator.
        
        Args:
            window_size: Number of candles per chart (lookback period)
            image_size: Target image dimensions (width, height)
            output_path: Base directory for saving images
        """
        self.window_size = window_size
        self.image_size = image_size
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Configure chart style
        self.chart_style = mpf.make_mpf_style(
            marketcolors=mpf.make_marketcolors(
                up='#26a69a',    # Green for bullish candles
                down='#ef5350',  # Red for bearish candles
                edge='inherit',
                wick='inherit',
                volume='in',
                alpha=0.9
            ),
            gridcolor='#2a2e39',
            gridstyle='',
            y_on_right=False,
            rc={
                'axes.edgecolor': '#2a2e39',
                'axes.linewidth': 0,
                'axes.labelcolor': 'none',
                'xtick.color': 'none',
                'ytick.color': 'none',
                'figure.facecolor': '#1e222d',
                'axes.facecolor': '#1e222d'
            }
        )
        
        logger.info(f"ChartGenerator initialized: window_size={window_size}, image_size={image_size}")
        logger.info(f"Images will be saved to: {self.output_path.absolute()}")
    
    def calculate_indicators(self, df):
        """
        Calculate technical indicators on entire DataFrame.
        Pre-calculation avoids redundant per-window computations.
        
        Args:
            df: DataFrame with OHLCV data (must have 'close', 'high', 'low', 'volume')
        
        Returns:
            DataFrame with added indicator columns (MA7, MA25, RSI, MACD)
        """
        df = df.copy()
        
        # Calculate Moving Averages
        df['MA7'] = df['close'].rolling(window=7, min_periods=1).mean()
        df['MA25'] = df['close'].rolling(window=25, min_periods=1).mean()
        
        # Calculate RSI (Relative Strength Index, window=14)
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        
        # Calculate MACD (Moving Average Convergence Divergence)
        # The ta library returns a MACD object with methods: macd(), macd_signal(), macd_diff()
        macd_obj = ta.trend.MACD(df['close'])
        df['MACD'] = macd_obj.macd()
        df['MACD_signal'] = macd_obj.macd_signal()
        df['MACD_hist'] = macd_obj.macd_diff()
        
        logger.info(f"Indicators calculated: MA7, MA25, RSI(14), MACD")
        
        return df
    
    def create_chart_image(self, window_data):
        """
        Create a candlestick chart image from pre-calculated window data with indicators.
        Assumes indicators (MA7, MA25, RSI, MACD, MACD_signal, MACD_hist) are already calculated.
        
        Args:
            window_data: DataFrame containing window_size candles with pre-calculated indicators
        
        Returns:
            PIL Image object
        """
        # Prepare data
        chart_data = window_data.copy()
        chart_data.index = pd.to_datetime(chart_data.index)
        
        # Create additional plots for MA7, MA25 (on price panel)
        add_plots = [
            mpf.make_addplot(chart_data['MA7'], color='#42a5f5', width=1.5, 
                           ylabel='Price'),
            mpf.make_addplot(chart_data['MA25'], color='#ffa726', width=1.5),
            # MACD (Panel 2)
            mpf.make_addplot(chart_data['MACD'], panel=2, color='#2196F3', 
                           width=1.5, ylabel='MACD'),
            mpf.make_addplot(chart_data['MACD_signal'], panel=2, color='#FF9800', 
                           width=1.5),
            mpf.make_addplot(chart_data['MACD_hist'], panel=2, color='#4CAF50', 
                           type='bar', alpha=0.3, ylabel='MACD Hist'),
            # RSI (Panel 3)
            mpf.make_addplot(chart_data['RSI'], panel=3, color='#9C27B0', 
                           width=1.5, ylabel='RSI'),
            # RSI Overbought/Oversold levels
            mpf.make_addplot([70]*len(chart_data), panel=3, color='red', 
                           type='line', alpha=0.3, width=0.5),
            mpf.make_addplot([30]*len(chart_data), panel=3, color='green', 
                           type='line', alpha=0.3, width=0.5),
        ]
        
        # Create figure with 3 panels: Price/Volume, MACD, RSI
        fig, axes = mpf.plot(
            chart_data,
            type='candle',
            style=self.chart_style,
            volume=True,
            addplot=add_plots,
            returnfig=True,
            figsize=(6, 8),
            panel_ratios=(3, 1, 1.5, 1.5),  # Price:Volume:MACD:RSI
            datetime_format='%H:%M',
            xrotation=0,
            axisoff=True,
            closefig=True
        )
        
        # Clean up all axes
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.margins(0, 0)
            ax.set_facecolor('#1e222d')
        
        fig.patch.set_facecolor('#1e222d')
        plt.tight_layout(pad=0)
        
        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0, 
                   facecolor='#1e222d')
        buf.seek(0)
        
        # Convert to PIL Image
        img = Image.open(buf)
        
        # Resize to target size
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        
        plt.close(fig)
        buf.close()
        
        return img
    
    def generate_single_chart(self, args):
        """
        Generate a single chart (for multiprocessing).
        Uses pre-calculated indicators from window_data.
        
        Args:
            args: Tuple of (window_data, timestamp, filepath)
        
        Returns:
            Tuple of (success, timestamp, filepath)
        """
        try:
            window_data, timestamp, filepath = args
            
            # Create chart image with pre-calculated indicators
            img = self.create_chart_image(window_data)
            
            # Save image
            img.save(filepath, optimize=True)
            
            return (True, timestamp, str(filepath))
            
        except Exception as e:
            logger.error(f"Error generating chart at timestamp {args[1]}: {str(e)}")
            return (False, None, None)
    
    def generate_charts_from_csv(self, csv_path, symbol='btc', num_workers=None):
        """
        Generate chart images from CSV file using multiprocessing.
        Pre-calculates all indicators on the full DataFrame before windowing.
        
        Args:
            csv_path: Path to OHLCV CSV file
            symbol: Symbol name for output directory (e.g., 'btc', 'eth')
            num_workers: Number of parallel workers (None = auto)
        
        Returns:
            List of generated image paths
        """
        logger.info(f"Loading data from {csv_path}")
        
        # Load CSV data
        df = pd.read_csv(csv_path)
        
        # Ensure datetime column
        if 'datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("CSV must contain 'datetime' or 'timestamp' column")
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        logger.info(f"Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
        
        # **PRE-CALCULATE ALL INDICATORS ON FULL DATAFRAME**
        logger.info("Pre-calculating indicators on full dataset...")
        df = self.calculate_indicators(df)
        
        # **DROP NaN VALUES AFTER INDICATOR CALCULATION**
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN values. Remaining: {len(df)}")
        
        logger.info(f"Final dataset: {len(df)} records from {df.index.min()} to {df.index.max()}")
        
        # Create output directory
        output_dir = self.output_path / symbol
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # **PREPARE ARGUMENTS: ONLY (window_data, timestamp, filepath)**
        # This prevents passing the entire DataFrame to each worker (memory leak fix)
        args_list = []
        for idx in range(len(df)):
            # Extract window data
            start_idx = max(0, idx - self.window_size + 1)
            end_idx = idx + 1
            window_data = df.iloc[start_idx:end_idx].copy()
            
            # Skip if not enough data for a full window
            if len(window_data) < self.window_size // 2:
                continue
            
            # Get timestamp and create filepath
            timestamp = int(df.index[idx].timestamp())
            filepath = output_dir / f"{timestamp}.png"
            
            # Add to args list (only 3 items, not entire df)
            args_list.append((window_data, timestamp, str(filepath)))
        
        logger.info(f"Prepared {len(args_list)} chart arguments (from {len(df)} records)")
        
        # Determine number of workers
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
        
        logger.info(f"Generating {len(args_list)} charts using {num_workers} workers...")
        
        # Generate charts with multiprocessing
        generated_files = []
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(self.generate_single_chart, args_list),
                total=len(args_list),
                desc=f"Generating {symbol.upper()} charts"
            ))
        
        # Collect successful results
        generated_files = [filepath for success, ts, filepath in results if success]
        
        logger.info(f"Successfully generated {len(generated_files)} chart images")
        logger.info(f"Images saved to: {output_dir.absolute()}")
        
        return generated_files
    
    def generate_all_symbols(self, symbols_config=None):
        """
        Generate charts for multiple symbols.
        
        Args:
            symbols_config: Dict mapping symbol names to CSV paths
                           If None, uses default BTC and ETH paths
        
        Returns:
            Dict of symbol: list of generated files
        """
        if symbols_config is None:
            symbols_config = {
                'btc': 'data/raw/BTCUSDT_klines.csv',
                'eth': 'data/raw/ETHUSDT_klines.csv'
            }
        
        logger.info("=" * 80)
        logger.info("Starting chart generation for all symbols")
        logger.info(f"Symbols: {list(symbols_config.keys())}")
        logger.info("=" * 80)
        
        results = {}
        for symbol, csv_path in symbols_config.items():
            try:
                logger.info(f"\n--- Processing {symbol.upper()} ---")
                files = self.generate_charts_from_csv(csv_path, symbol=symbol)
                results[symbol] = files
            except Exception as e:
                logger.error(f"Failed to generate charts for {symbol}: {str(e)}")
                results[symbol] = []
        
        logger.info("\n" + "=" * 80)
        logger.info("Chart generation completed!")
        total_images = sum(len(files) for files in results.values())
        logger.info(f"Total images generated: {total_images}")
        logger.info("=" * 80)
        
        return results
    
    def run(self):
        """
        Standardized run method for orchestrator compatibility.
        """
        logger.info("ChartGenerator.run() started")
        
        # Generate charts for default symbols
        results = self.generate_all_symbols()
        
        logger.info("ChartGenerator.run() completed successfully")
        return results
