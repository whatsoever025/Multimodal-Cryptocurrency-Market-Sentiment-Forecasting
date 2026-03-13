"""
Chart Generator for Converting OHLCV Data to Candlestick Chart Images
Generates professional candlestick charts for deep learning (ViT/ResNet) training.
Includes technical indicators and uses multiprocessing for optimization.
"""

import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from PIL import Image
import io
import warnings

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
        Calculate technical indicators (Moving Averages).
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # Calculate Moving Averages
        df['MA7'] = df['close'].rolling(window=7, min_periods=1).mean()
        df['MA25'] = df['close'].rolling(window=25, min_periods=1).mean()
        
        return df
    
    def create_chart_image(self, window_data, timestamp):
        """
        Create a candlestick chart image from window data.
        
        Args:
            window_data: DataFrame containing window_size candles
            timestamp: Unix timestamp for filename
        
        Returns:
            PIL Image object
        """
        # Prepare data
        chart_data = window_data.copy()
        chart_data.index = pd.to_datetime(chart_data.index)
        
        # Calculate moving averages
        chart_data['MA7'] = chart_data['close'].rolling(window=7, min_periods=1).mean()
        chart_data['MA25'] = chart_data['close'].rolling(window=25, min_periods=1).mean()
        
        # Prepare additional plots (moving averages)
        add_plots = [
            mpf.make_addplot(chart_data['MA7'], color='#42a5f5', width=1.5),
            mpf.make_addplot(chart_data['MA25'], color='#ffa726', width=1.5)
        ]
        
        # Create figure
        fig, axes = mpf.plot(
            chart_data,
            type='candle',
            style=self.chart_style,
            volume=True,
            addplot=add_plots,
            returnfig=True,
            figsize=(6, 6),
            panel_ratios=(3, 1),  # Price:Volume ratio
            datetime_format='%H:%M',
            xrotation=0,
            axisoff=True,
            closefig=True
        )
        
        # Remove all margins, axes, labels
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
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0, facecolor='#1e222d')
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
        
        Args:
            args: Tuple of (index, row, df, symbol, output_dir)
        
        Returns:
            Tuple of (success, timestamp, filepath)
        """
        try:
            idx, row, df_subset, symbol, output_dir = args
            
            # Extract window data
            start_idx = max(0, idx - self.window_size + 1)
            window_data = df_subset.iloc[start_idx:idx + 1].copy()
            
            # Skip if not enough data
            if len(window_data) < self.window_size // 2:
                return (False, None, None)
            
            # Get timestamp
            timestamp = int(row['timestamp'].timestamp())
            
            # Create chart image
            img = self.create_chart_image(window_data, timestamp)
            
            # Save image
            filepath = output_dir / f"{timestamp}.png"
            img.save(filepath, optimize=True)
            
            return (True, timestamp, str(filepath))
            
        except Exception as e:
            logger.error(f"Error generating chart at index {idx}: {str(e)}")
            return (False, None, None)
    
    def generate_charts_from_csv(self, csv_path, symbol='btc', num_workers=None):
        """
        Generate chart images from CSV file using multiprocessing.
        
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
        
        # Create output directory
        output_dir = self.output_path / symbol
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare arguments for multiprocessing
        df_reset = df.reset_index()
        args_list = [
            (i, df_reset.iloc[i], df, symbol, output_dir)
            for i in range(len(df_reset))
        ]
        
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
                'btc': 'data/raw/btcusdt_ohlcv_1h.csv',
                'eth': 'data/raw/ethusdt_ohlcv_1h.csv'
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
