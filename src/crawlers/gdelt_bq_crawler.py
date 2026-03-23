"""
GDELT BigQuery Crawler for Macroeconomic Sentiment Data Collection.
Fetches crypto news sentiment data directly from Google BigQuery GDELT dataset.

Uses BaseCrawler inheritance for standardized data collection pipeline.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from google.cloud import bigquery

from .base import BaseCrawler, CrawlerConfig


class GdeltBQCrawler(BaseCrawler):
    """
    Crawler for fetching macroeconomic sentiment data from Google BigQuery GDELT dataset.
    Aggregates crypto news sentiment data by day from GDELT v2 GKG table.
    
    Inherits from BaseCrawler for:
    - Environment variable loading (.env)
    - Logging setup
    - Retry logic and error handling
    - Standardized interface
    """

    def __init__(
        self,
        base_path: str = "data/raw",
        config: Optional[CrawlerConfig] = None,
        days_back: int = 7,
    ):
        """
        Initialize GDELT BigQuery Crawler with credentials validation.

        Args:
            base_path: Directory path for saving raw data files
            config: Optional CrawlerConfig for customization
            days_back: Number of days of historical data to fetch

        Raises:
            EnvironmentError: If GOOGLE_APPLICATION_CREDENTIALS is not set
        """
        # Call parent init for standard setup (env loading, logging, etc.)
        super().__init__(base_path=base_path, config=config)
        
        # Validate Google Cloud credentials
        credentials_path = self.get_env("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            self.logger.warning(
                "GOOGLE_APPLICATION_CREDENTIALS not set in .env. "
                "GDELT crawler requires GCP service account credentials."
            )
            self.client = None
        else:
            # Initialize BigQuery client
            try:
                self.client = bigquery.Client()
                self.logger.info("BigQuery client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize BigQuery client: {e}")
                self.client = None

        self.days_back = days_back
        self.output_file = self.base_path / "gdelt_sentiment.csv"
        self.logger.info(f"GdeltBQCrawler initialized. Output: {self.output_file}")

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch GDELT sentiment data from BigQuery.
        
        Returns:
            List of sentiment records as dictionaries
        """
        if not self.client:
            self.logger.error("BigQuery client not available")
            return []

        try:
            df = self._fetch_historical_data()
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            self.logger.error(f"Failed to fetch GDELT data: {e}")
            return []

    def _fetch_historical_data(self) -> pd.DataFrame:
        """
        Fetch daily aggregated crypto news sentiment data from GDELT BigQuery.

        Executes daily queries for the last N days, filtering for 
        cryptocurrency-related news only.

        Returns:
            pandas.DataFrame with columns:
                - timestamp: Daily UTC timestamp
                - news_volume: Count of documents (news articles)
                - gdelt_mean_tone: Average tone score
                - gdelt_mean_positive: Average positive tone
                - gdelt_mean_negative: Average negative tone

        Raises:
            Exception: If BigQuery query execution fails
        """
        self.logger.info(f"Fetching {self.days_back} days of GDELT sentiment data from BigQuery")

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=self.days_back)

        # Convert to date strings for SQL
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')

        sql = f"""
SELECT
  TIMESTAMP(SAFE.PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8))) AS timestamp,
  COUNT(DocumentIdentifier) AS news_volume,
  AVG(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) AS gdelt_mean_tone,
  AVG(CAST(SPLIT(V2Tone, ',')[OFFSET(1)] AS FLOAT64)) AS gdelt_mean_positive,
  AVG(CAST(SPLIT(V2Tone, ',')[OFFSET(2)] AS FLOAT64)) AS gdelt_mean_negative
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  DATE >= CAST('{start_str}000000' AS INT64)
  AND DATE <= CAST('{end_str}235959' AS INT64)
  AND SourceCollectionIdentifier = 1
  AND (
    REGEXP_CONTAINS(AllNames, r'(?i)(bitcoin|ethereum|crypto|binance|coinbase|cryptocurrency)')
    OR REGEXP_CONTAINS(V2Themes, r'(?i)(bitcoin|ethereum|crypto|binance|coinbase|cryptocurrency)')
  )
GROUP BY timestamp
ORDER BY timestamp ASC;
        """

        try:
            self.logger.debug(f"Executing GDELT query for {start_str} to {end_str}...")
            query_job = self.client.query(sql)
            df = query_job.to_dataframe()
            
            self.logger.info(f"Retrieved {len(df)} days of crypto news")
            if not df.empty:
                self.logger.debug(
                    f"Tone range: {df['gdelt_mean_tone'].min():.2f} to {df['gdelt_mean_tone'].max():.2f}"
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"BigQuery query failed: {e}", exc_info=True)
            raise

    def validate(self, records: List[Dict[str, Any]]) -> bool:
        """
        Validate GDELT sentiment data schema and integrity.
        
        Checks:
        - Required fields present: timestamp, news_volume, tone scores
        - Numeric types correct
        - Timestamp validity

        Args:
            records: List of sentiment records

        Returns:
            True if validation passes, False otherwise
        """
        if not records:
            self.logger.warning("No records to validate")
            return True

        required_fields = {
            'timestamp', 'news_volume', 'gdelt_mean_tone',
            'gdelt_mean_positive', 'gdelt_mean_negative'
        }

        for i, record in enumerate(records):
            missing = required_fields - set(record.keys())
            if missing:
                self.logger.error(f"Record {i} missing fields: {missing}")
                return False

            try:
                # Validate numeric types
                int(record['news_volume'])
                float(record['gdelt_mean_tone'])
                float(record['gdelt_mean_positive'])
                float(record['gdelt_mean_negative'])

                if int(record['news_volume']) < 0:
                    self.logger.error(f"Record {i}: negative news_volume")
                    return False

            except (ValueError, TypeError) as e:
                self.logger.error(f"Record {i}: invalid numeric value - {e}")
                return False

            try:
                datetime.fromisoformat(record.get('timestamp', ''))
            except (ValueError, TypeError):
                self.logger.error(f"Record {i}: invalid timestamp format")
                return False

        self.logger.info(f"Validated {len(records)} records successfully")
        return True

    def save(self, records: List[Dict[str, Any]], filename: Optional[str] = None) -> int:
        """
        Save GDELT records to CSV file (overwrites existing data).
        
        Args:
            records: List of sentiment records
            filename: Optional filename (defaults to gdelt_sentiment.csv)
            
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

            # Remove duplicates based on timestamp
            original_len = len(df)
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
            deduped = original_len - len(df)
            if deduped > 0:
                self.logger.info(f"Removed {deduped} duplicate records")

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Save to CSV
            df.to_csv(output_file, index=False)

            self.logger.info(f"Successfully saved {len(df)} records to {output_file}")
            return len(df)

        except Exception as e:
            self.logger.error(f"Failed to save records: {e}")
            return 0


if __name__ == "__main__":
    """
    Standalone execution: python gdelt_bq_crawler.py
    Fetches sentiment data and saves to data/raw/gdelt_sentiment.csv
    """
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        crawler = GdeltBQCrawler()
        saved_count = crawler.run()
        
        if saved_count > 0:
            print(f"✓ GDELT crawler completed successfully: {saved_count} records saved")
            sys.exit(0)
        else:
            print("✗ GDELT crawler failed or returned no data")
            sys.exit(0)  # Exit 0 for graceful degradation
            
    except Exception as e:
        print(f"✗ GDELT crawler encountered fatal error: {e}")
        logging.exception(e)
        sys.exit(0)  # Exit 0 for graceful degradation
