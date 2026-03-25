"""
Production Chart Generation Script
Generates all candlestick chart images from collected OHLCV data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing.chart_generator import ChartGenerator
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Generate charts for all symbols."""
    logger.info("=" * 80)
    logger.info("PRODUCTION CHART GENERATION")
    logger.info("=" * 80)
    
    # Initialize generator
    generator = ChartGenerator(
        window_size=24,        # 24-hour lookback window
        image_size=(224, 224), # Standard ViT/ResNet input size
        output_path='data/processed/images'
    )
    
    # Define symbols to process
    symbols_config = {
        'btc': 'data/raw/BTCUSDT_klines.csv',
        'eth': 'data/raw/ETHUSDT_klines.csv'
    }
    
    # Generate all charts
    results = generator.generate_all_symbols(symbols_config)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("GENERATION SUMMARY")
    logger.info("=" * 80)
    for symbol, files in results.items():
        logger.info(f"{symbol.upper()}: {len(files)} images generated")
    
    total = sum(len(files) for files in results.values())
    logger.info(f"\nTOTAL: {total} chart images ready for deep learning!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
