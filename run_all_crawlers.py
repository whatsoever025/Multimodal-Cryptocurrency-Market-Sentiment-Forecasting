"""
Central Orchestrator for Multi-Crawler Data Collection Pipeline
Single entry point for running all data crawlers (Binance, Twitter, Reddit, etc.)
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from crawlers.binance_crawler import BinanceCrawler
from crawlers.coingecko_crawler import CoinGeckoCrawler
from crawlers.gdelt_bq_crawler import GdeltBQCrawler
from crawlers.sentiment_crawler import SentimentCrawler
from crawlers.text_crawler import TextCrawler


# Centralized logging configuration
def setup_logging(log_file='crawler.log', log_level=logging.INFO):
    """
    Configure centralized logging for all crawlers.
    
    Args:
        log_file: Path to the log file
        log_level: Logging level (INFO, DEBUG, WARNING, ERROR)
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file).parent
    if log_path != Path('.'):
        log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 100)
    logger.info(f"Logging initialized - Session started at {datetime.now().isoformat()}")
    logger.info(f"Log file: {Path(log_file).absolute()}")
    logger.info("=" * 100)
    
    return logger


class CrawlerRegistry:
    """
    Registry pattern for managing multiple crawlers.
    Allows easy addition of new crawlers without modifying orchestrator logic.
    """
    
    def __init__(self, data_path='data/raw'):
        """
        Initialize the crawler registry.
        
        Args:
            data_path: Base path for saving raw data
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.crawlers = {}
        self.logger = logging.getLogger(__name__)
        
        # Register all available crawlers
        self._register_crawlers()
    
    def _register_crawlers(self):
        """
        Register all available crawler classes.
        Add new crawlers here as they are developed.
        """
        self.logger.info("Registering available crawlers...")
        
        # Register Binance crawler
        self.register('binance', BinanceCrawler, {
            'base_path': str(self.data_path)
        })
        
        # Register CoinGecko crawler
        self.register('coingecko', CoinGeckoCrawler, {
            'base_path': str(self.data_path)
        })
        
        # Register Sentiment crawler
        self.register('sentiment', SentimentCrawler, {
            'base_path': str(self.data_path)
        })

        # Register Text crawler
        self.register('text', TextCrawler, {
            'base_path': str(self.data_path)
        })
        
        # Register GDELT BigQuery crawler
        self.register('gdelt', GdeltBQCrawler, {
            'base_path': str(self.data_path)
        })
        
        # Future crawlers can be added here:
        # self.register('twitter', TwitterCrawler, {'base_path': str(self.data_path)})
        # self.register('reddit', RedditCrawler, {'base_path': str(self.data_path)})
        # self.register('news', NewsCrawler, {'base_path': str(self.data_path)})
        
        self.logger.info(f"Registered {len(self.crawlers)} crawler(s): {list(self.crawlers.keys())}")
    
    def register(self, name, crawler_class, init_params=None):
        """
        Register a crawler in the registry.
        
        Args:
            name: Unique identifier for the crawler (e.g., 'binance', 'twitter')
            crawler_class: The crawler class to instantiate
            init_params: Dictionary of initialization parameters for the crawler
        """
        if init_params is None:
            init_params = {}
        
        self.crawlers[name] = {
            'class': crawler_class,
            'params': init_params,
            'instance': None
        }
        
        self.logger.info(f"✓ Registered crawler: {name}")
    
    def get_crawler(self, name):
        """
        Get or create a crawler instance.
        
        Args:
            name: Crawler identifier
        
        Returns:
            Initialized crawler instance
        """
        if name not in self.crawlers:
            raise ValueError(f"Crawler '{name}' not registered. Available: {list(self.crawlers.keys())}")
        
        crawler_info = self.crawlers[name]
        
        # Lazy initialization: create instance only when needed
        if crawler_info['instance'] is None:
            self.logger.info(f"Initializing {name} crawler...")
            crawler_info['instance'] = crawler_info['class'](**crawler_info['params'])
        
        return crawler_info['instance']
    
    def run_crawler(self, name):
        """
        Run a specific crawler with error handling.
        
        Args:
            name: Crawler identifier
        
        Returns:
            True if successful, False if failed
        """
        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(f"Starting crawler: {name.upper()}")
        self.logger.info(f"{'=' * 100}")
        
        try:
            crawler = self.get_crawler(name)
            crawler.run()
            
            self.logger.info(f"✓ {name.upper()} crawler completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ {name.upper()} crawler failed with error: {str(e)}", exc_info=True)
            return False
    
    def run_all(self):
        """
        Run all registered crawlers with isolated error handling.
        If one crawler fails, others continue to run.
        
        Returns:
            Dictionary with results for each crawler
        """
        self.logger.info("\n" + "=" * 100)
        self.logger.info("RUNNING ALL REGISTERED CRAWLERS")
        self.logger.info(f"Total crawlers to run: {len(self.crawlers)}")
        self.logger.info("=" * 100)
        
        results = {}
        
        for crawler_name in self.crawlers.keys():
            success = self.run_crawler(crawler_name)
            results[crawler_name] = 'SUCCESS' if success else 'FAILED'
        
        # Summary
        self.logger.info("\n" + "=" * 100)
        self.logger.info("CRAWLER EXECUTION SUMMARY")
        self.logger.info("=" * 100)
        
        for crawler_name, status in results.items():
            status_symbol = "✓" if status == 'SUCCESS' else "✗"
            self.logger.info(f"{status_symbol} {crawler_name.upper()}: {status}")
        
        success_count = sum(1 for s in results.values() if s == 'SUCCESS')
        self.logger.info(f"\nTotal: {success_count}/{len(results)} crawlers succeeded")
        self.logger.info("=" * 100)
        
        return results
    
    def list_crawlers(self):
        """List all registered crawlers."""
        return list(self.crawlers.keys())


def parse_arguments():
    """
    Parse command-line arguments for crawler execution.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Multi-Crawler Data Collection Pipeline for Crypto Sentiment Forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all crawlers
  python run_all_crawlers.py
  
  # Run specific crawler
  python run_all_crawlers.py --source binance
  
  # Run multiple specific crawlers
  python run_all_crawlers.py --source binance --source twitter
  
  # List available crawlers
  python run_all_crawlers.py --list
  
  # Enable debug logging
  python run_all_crawlers.py --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        action='append',
        choices=['binance', 'coingecko', 'sentiment', 'text', 'gdelt', 'twitter', 'reddit', 'news'],
        help='Specific crawler(s) to run. Can be specified multiple times. If not specified, runs all crawlers.'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available crawlers and exit'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/raw',
        help='Base directory for saving raw data (default: data/raw)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='crawler.log',
        help='Path to log file (default: crawler.log)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """
    Main orchestrator function.
    Initializes registry, parses arguments, and runs selected crawlers.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Setup centralized logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_file=args.log_file, log_level=log_level)
    
    # Ensure data directory exists
    data_path = Path(args.data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory: {data_path.absolute()}")
    
    # Initialize crawler registry
    registry = CrawlerRegistry(data_path=args.data_path)
    
    # Handle --list flag
    if args.list:
        print("\nAvailable Crawlers:")
        print("-" * 40)
        for crawler_name in registry.list_crawlers():
            print(f"  • {crawler_name}")
        print("-" * 40)
        print(f"\nTotal: {len(registry.list_crawlers())} crawler(s)")
        return
    
    # Determine which crawlers to run
    if args.source:
        # Run specific crawler(s)
        logger.info(f"Running specific crawler(s): {args.source}")
        results = {}
        for crawler_name in args.source:
            success = registry.run_crawler(crawler_name)
            results[crawler_name] = 'SUCCESS' if success else 'FAILED'
        
        # Print summary
        logger.info("\n" + "=" * 100)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 100)
        for name, status in results.items():
            logger.info(f"{'✓' if status == 'SUCCESS' else '✗'} {name.upper()}: {status}")
        logger.info("=" * 100)
    else:
        # Run all crawlers
        logger.info("No specific crawler specified - running ALL crawlers")
        registry.run_all()
    
    logger.info("\n✓ Orchestrator execution completed!")
    logger.info(f"Session ended at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
