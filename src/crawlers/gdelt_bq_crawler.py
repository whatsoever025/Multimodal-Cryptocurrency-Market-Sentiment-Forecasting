"""
GDELT BigQuery Crawler for Macroeconomic Sentiment Data Collection.
Fetches crypto news sentiment data directly from Google BigQuery GDELT dataset.

Refactored to use BaseCrawler inheritance for consistency.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from .base import BaseCrawler, CrawlerConfig

logger = logging.getLogger(__name__)


class GdeltBQCrawler(BaseCrawler):
    """
    Crawler for fetching macroeconomic sentiment data from Google BigQuery GDELT dataset.
    Aggregates crypto news sentiment data by day.
    
    Inherits from BaseCrawler for:
    - Standardized initialization
    - Consistent error handling
    - Unified logging
    """

    def __init__(self, base_path="data/raw", config: CrawlerConfig = None):
        """
        Initialize GDELT BigQuery Crawler with credentials validation.

        Args:
            base_path: Directory path for saving raw data files
            config: Optional CrawlerConfig for customization

        Raises:
            EnvironmentError: If GOOGLE_APPLICATION_CREDENTIALS is not set
        """
        # Call parent init for standard setup
        super().__init__(base_path=base_path, config=config)
        
        # Load environment variables from .env file in project root
        project_root = Path(__file__).resolve().parents[2]
        env_path = project_root / ".env"
        load_dotenv(env_path)

        # Validate Google Cloud credentials
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            error_msg = (
                "GOOGLE_APPLICATION_CREDENTIALS environment variable not set. "
                "Set GOOGLE_APPLICATION_CREDENTIALS to the path of your GCP service account JSON file. "
                "Example: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json"
            )
            self.logger.critical(error_msg)
            raise EnvironmentError(error_msg)

        # Initialize BigQuery client
        try:
            self.client = bigquery.Client()
            self.logger.info("BigQuery client initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize BigQuery client: {str(e)}"
            self.logger.critical(error_msg)
            raise EnvironmentError(error_msg)

        self.output_file = self.base_path / "gdelt_hourly_sentiment.csv"
        self.logger.info(f"Output file will be saved to: {self.output_file.absolute()}")

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch GDELT sentiment data from BigQuery.
        
        Returns:
            List of sentiment records as dictionaries
        """
        df = self.fetch_historical_data()
        return df.to_dict('records') if not df.empty else []

    def validate(self, records: List[Dict[str, Any]]) -> bool:
        """
        Validate GDELT sentiment data schema and integrity.
        
        Checks:
        - Required fields present: timestamp, news_volume, gdelt_mean_tone, gdelt_mean_positive, gdelt_mean_negative
        - Numeric types correct
        - Timestamp validity
        """
        if not records:
            self.logger.warning("No records to validate")
            return True

        required_fields = {
            'timestamp', 'news_volume', 'gdelt_mean_tone',
            'gdelt_mean_positive', 'gdelt_mean_negative'
        }

        for i, record in enumerate(records):
            if not all(field in record for field in required_fields):
                missing = required_fields - set(record.keys())
                self.logger.error(f"Record {i} missing fields: {missing}")
                return False

            try:
                # Validate numeric types
                nv = int(record['news_volume'])
                tone = float(record['gdelt_mean_tone'])
                pos = float(record['gdelt_mean_positive'])
                neg = float(record['gdelt_mean_negative'])

                if nv < 0:
                    self.logger.error(f"Record {i}: negative news_volume {nv}")
                    return False

            except (ValueError, TypeError) as e:
                self.logger.error(f"Record {i}: invalid numeric value - {e}")
                return False

        self.logger.info(f"Validated {len(records)} records successfully")
        return True

    def save(self, records: List[Dict[str, Any]]) -> int:
        """
        Save GDELT records to CSV file.
        
        Args:
            records: List of sentiment records
            
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
            df.to_csv(self.output_file, index=False)

            self.logger.info(f"Successfully saved {len(df)} records to {self.output_file}")
            return len(df)

        except Exception as e:
            self.logger.error(f"Failed to save records: {e}")
            raise IOError(str(e))
        """
        Fetch daily aggregated crypto news sentiment data from GDELT BigQuery.

        Executes monthly queries for 2026-03-01 to 2026-03-21 (TEST: 1 month),
        filtering for cryptocurrency-related news only.

        Returns:
            pandas.DataFrame with columns:
                - timestamp: Daily UTC timestamp (at 00:00:00)
                - news_volume: Count of documents (news articles)
                - gdelt_mean_tone: Average tone score
                - gdelt_mean_positive: Average positive tone
                - gdelt_mean_negative: Average negative tone

        Raises:
            Exception: If BigQuery query execution fails
        """
        self.logger.info("Fetching historical GDELT sentiment data from BigQuery with monthly chunking...")

        # TEST: Using 1 month of data (2026-03-01 to 2026-03-21)
        start_date = datetime(2026, 3, 1)
        end_date = datetime(2026, 3, 21)
        
        # Generate monthly ranges
        monthly_ranges = self._generate_monthly_ranges(start_date, end_date)
        logger.info(f"Querying {len(monthly_ranges)} months of data...")

        # List to accumulate DataFrames from each month
        all_dfs = []

        try:
            for month_idx, (month_start, month_end) in enumerate(monthly_ranges, 1):
                logger.info(f"Processing month {month_idx}/{len(monthly_ranges)}: {month_start.date()} to {month_end.date()}")
                
                # Convert dates to YYYYMMDD format for SQL
                start_str = month_start.strftime('%Y%m%d')
                end_str = month_end.strftime('%Y%m%d')

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
    REGEXP_CONTAINS(AllNames, r'(?i)(bitcoin|ethereum|crypto|binance|coinbase)')
    OR REGEXP_CONTAINS(V2Themes, r'(?i)(bitcoin|ethereum|crypto|binance|coinbase)')
    OR REGEXP_CONTAINS(DocumentIdentifier, r'(?i)(bitcoin|ethereum|crypto|binance|coinbase)')
  )
GROUP BY timestamp
ORDER BY timestamp ASC;
                """

                try:
                    logger.debug(f"Executing query for {month_start.date()} to {month_end.date()}...")
                    query_job = self.client.query(sql)
                    
                    # Convert result to DataFrame
                    df = query_job.to_dataframe()
                    
                    records = len(df)
                    logger.info(f"  Month {month_idx}: Retrieved {records} days of crypto news")
                    
                    if records > 0:
                        all_dfs.append(df)
                        logger.debug(f"  Tone range: {df['gdelt_mean_tone'].min():.2f} to {df['gdelt_mean_tone'].max():.2f}")
                    
                except Exception as e:
                    error_msg = f"Failed to query month {month_idx} ({month_start.date()} to {month_end.date()}): {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    raise

            # Combine all monthly DataFrames
            if all_dfs:
                logger.info(f"Combining {len(all_dfs)} months of data...")
                df_combined = pd.concat(all_dfs, ignore_index=True)
                df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
                
                total_records = len(df_combined)
                logger.info(f"Total records retrieved: {total_records}")
                logger.info(f"Date range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}")
                logger.info(f"News volume range: {df_combined['news_volume'].min()} to {df_combined['news_volume'].max()}")
                
                return df_combined
            else:
                logger.warning("No cryptocurrency news data found in any month")
                return pd.DataFrame(columns=['timestamp', 'news_volume', 'gdelt_mean_tone', 'gdelt_mean_positive', 'gdelt_mean_negative'])
                
        except Exception as e:
            error_msg = f"Failed to execute monthly BigQuery queries: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    def _generate_monthly_ranges(self, start_date, end_date):
        """
        Generate monthly date ranges for chunked querying.

        Args:
            start_date: datetime object for start date
            end_date: datetime object for end date

        Returns:
            list of tuples (start_date, end_date) for each month
        """
        ranges = []
        current = start_date

        while current <= end_date:
            # Last day of current month
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1)
            else:
                next_month = current.replace(month=current.month + 1)
            
            month_end = next_month - timedelta(days=1)
            
            # Don't go past end_date
            if month_end > end_date:
                month_end = end_date

            ranges.append((current, month_end))
            current = month_end + timedelta(days=1)

        return ranges

    def save(self, df):
        """
        Save DataFrame to CSV file.

        Args:
            df: pandas.DataFrame to save

        Returns:
            int: Number of rows saved

        Raises:
            IOError: If save operation fails
        """
        try:
            row_count = len(df)
            df.to_csv(self.output_file, index=False)
            logger.info(f"Successfully saved {row_count} records to {self.output_file}")
            return row_count
            
        except Exception as e:
            error_msg = f"Failed to save data to CSV: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise IOError(error_msg)

    def run(self) -> int:
        """
        Standard orchestration flow for crawler. Uses parent's run() implementation.
        Fetches GDELT sentiment data from BigQuery.
        
        Returns:
            Number of records saved
        """
        self.logger.info("GdeltBQCrawler starting via run()")
        # Use parent's orchestration: fetch() -> validate() -> save()
        return super().run()
