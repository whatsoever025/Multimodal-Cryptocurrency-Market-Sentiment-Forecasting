"""
Hugging Face Dataset Crawler for Cryptocurrency News Data

Fetches crypto news sentiment dataset from Hugging Face Hub.
Dataset: maryamfakhari/crypto-news-coindesk-2020-2025

Uses BaseCrawler inheritance for standardized data collection pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from .base import BaseCrawler, CrawlerConfig


class HuggingFaceCrawler(BaseCrawler):
    """
    Crawler for fetching cryptocurrency news data from Hugging Face datasets.
    
    Downloads the crypto-news-coindesk dataset and converts to CSV format.
    
    Inherits from BaseCrawler for:
    - Environment variable loading (.env)
    - Logging setup
    - HTTP session management
    - Retry logic with exponential backoff
    - Rate limiting enforcement
    """

    DATASET_ID = "maryamfakhari/crypto-news-coindesk-2020-2025"

    def __init__(
        self,
        base_path: str = "data/raw",
        config: Optional[CrawlerConfig] = None,
    ):
        """
        Initialize Hugging Face Crawler.

        Args:
            base_path: Directory path for saving raw data files
            config: Optional CrawlerConfig for customization
        """
        super().__init__(base_path=base_path, config=config)

        self.output_file = self.base_path / "huggingface_crypto_news.csv"
        self.logger.info(
            f"HuggingFaceCrawler initialized. Dataset: {self.DATASET_ID}"
        )
        self.logger.info(f"Output: {self.output_file}")

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch cryptocurrency news data from Hugging Face dataset.

        Returns:
            List of news records as dictionaries
        """
        try:
            from datasets import load_dataset

            self.logger.info(f"Loading dataset: {self.DATASET_ID}")
            dataset = load_dataset(self.DATASET_ID)

            # Handle different dataset structures
            # HF datasets typically have a 'train' split
            if isinstance(dataset, dict):
                # Multiple splits - use train split or first available
                if "train" in dataset:
                    data = dataset["train"]
                else:
                    split_name = list(dataset.keys())[0]
                    self.logger.info(f"Using split: {split_name}")
                    data = dataset[split_name]
            else:
                # Single dataset object
                data = dataset

            # Convert to list of dictionaries
            # HuggingFace datasets use to_list() instead of pandas-style to_dict()
            records = data.to_list()

            self.logger.info(f"Fetched {len(records)} records from Hugging Face")
            return records

        except ImportError:
            self.logger.error(
                "datasets library not installed. Install with: pip install datasets"
            )
            return []
        except Exception as e:
            self.logger.error(f"Failed to fetch data from Hugging Face: {e}")
            return []

    def validate(self, records: List[Dict[str, Any]]) -> bool:
        """
        Validate news data schema.

        Checks:
        - Records is a non-empty list
        - Each record is a dictionary
        - Records have content fields

        Args:
            records: List of news records

        Returns:
            True if all records valid, False otherwise
        """
        if not records:
            self.logger.warning("No records to validate")
            return True

        if not isinstance(records, list):
            self.logger.error(f"Records must be a list, got {type(records)}")
            return False

        for i, record in enumerate(records):
            if not isinstance(record, dict):
                self.logger.error(
                    f"Record {i} is not a dictionary: {type(record)}"
                )
                return False

            # Check that record is not empty
            if not record:
                self.logger.error(f"Record {i} is empty")
                return False

        self.logger.info(f"Validated {len(records)} records successfully")
        return True

    def save(
        self,
        records: List[Dict[str, Any]],
        filename: Optional[str] = None,
    ) -> int:
        """
        Save news records to CSV file.

        Args:
            records: List of news records
            filename: Optional filename (defaults to huggingface_crypto_news.csv)

        Returns:
            Number of rows saved
        """
        if filename:
            output_file = self.base_path / filename
        else:
            output_file = self.output_file

        if not records:
            self.logger.warning("No records to save")
            return 0

        try:
            df = pd.DataFrame(records)

            self.logger.info(f"DataFrame shape: {df.shape}")
            self.logger.info(f"Columns: {list(df.columns)}")

            # Remove duplicates if a unique identifier exists
            original_len = len(df)
            
            # Try common unique field names
            unique_fields = []
            for field in ["id", "url", "title", "news_id"]:
                if field in df.columns:
                    unique_fields = [field]
                    break

            if unique_fields:
                df = df.drop_duplicates(subset=unique_fields, keep="first")
                deduped = original_len - len(df)
                if deduped > 0:
                    self.logger.info(f"Removed {deduped} duplicate records")

            # Save to CSV
            df.to_csv(output_file, index=False)

            self.logger.info(f"Successfully saved {len(df)} records to {output_file}")
            return len(df)

        except Exception as e:
            self.logger.error(f"Failed to save data to CSV: {e}")
            return 0
