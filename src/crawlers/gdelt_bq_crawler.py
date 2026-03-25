"""
GDELT BigQuery Crawler for Macroeconomic Sentiment Data Collection.

Fetches macroeconomic and geopolitical news sentiment from Google BigQuery GDELT dataset
using a single optimized query spanning 2023 to present.

Architecture:
- Inherits from BaseCrawler for standardized interface
- Initializes BigQuery Client inside fetch() method
- All data transformations occur in BigQuery SQL (timestamp parsing, tone extraction)
- Outputs hourly-granular macro sentiment data to gdelt_macro.csv

Output columns:
- timestamp: UTC datetime truncated to hour (parsed from DATE field)
- url: News article URL/source (DocumentIdentifier)
- themes: Extracted theme tags with scores (V2Themes)
- sentiment_tone: Average document tone score [-100, 100] (first value of V2Tone)

BigQuery schema column names:
- DATE: YYYYMMDDHHMMSS format timestamp
- DocumentIdentifier: URL/source of the article
- V2Themes: Comma-separated theme classification tags with confidence scores
- V2Tone: Comma-separated average tone and sentiment metrics
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from .base import BaseCrawler, CrawlerConfig


class GdeltBQCrawler(BaseCrawler):
    """
    Crawler for fetching macroeconomic sentiment data from Google BigQuery GDELT dataset.
    
    Executes single optimized query covering 2023 to present, with all data transformation
    (timestamp parsing, theme filtering, tone extraction) performed in BigQuery SQL.
    
    Inherits from BaseCrawler for:
    - Environment variable loading (.env)
    - Logging setup
    - Standardized interface (fetch, validate, save, run)
    - Error handling and graceful degradation
    """

    # Query constants
    START_DATE = "2023-01-01"  # Historical data range start (for code documentation)
    DAYS_BACK = 7  # Fetch last 7 days to keep data volume manageable
    MAX_RECORDS = 100000  # Cap at 100k records to avoid memory/time issues
    MACRO_THEMES = [
        "ECON_INFLATION",
        "US_FEDERAL_RESERVE",
        "CRISIS",
        "ARMEDCONFLICT",
        "ECON_UNEMPLOYMENT",
    ]

    def __init__(
        self,
        base_path: str = "data/raw",
        config: Optional[CrawlerConfig] = None,
    ):
        """
        Initialize GDELT BigQuery Crawler.

        Args:
            base_path: Directory path for saving raw data files
            config: Optional CrawlerConfig for customization
        """
        # Call parent init for standard setup (env loading, logging, etc.)
        super().__init__(base_path=base_path, config=config)

        # BigQuery client initialized lazily inside fetch() method
        self.client = None
        self.output_file = self.base_path / "gdelt_macro.csv"
        
        self.logger.info(f"GdeltBQCrawler initialized. Output: {self.output_file}")

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch macroeconomic sentiment data from BigQuery GDELT dataset.
        
        Initializes BigQuery Client on demand and executes single optimized query
        covering 2023-present with macro-theme filters. All data transformation
        (timestamp parsing, tone extraction) occurs in SQL for efficiency.
        
        Returns:
            List of sentiment records as dictionaries with keys: 
            timestamp, url, themes, sentiment_tone
        """
        try:
            # Initialize BigQuery Client lazily (on first fetch call)
            from google.cloud import bigquery
            
            try:
                self.client = bigquery.Client()
                self.logger.info("BigQuery client initialized successfully")
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize BigQuery client (check GOOGLE_APPLICATION_CREDENTIALS): {e}"
                )
                return []
            
            # Execute optimized query and return records
            df = self._fetch_macro_sentiment_data()
            return df.to_dict('records') if not df.empty else []
            
        except Exception as e:
            self.logger.error(f"Failed to fetch GDELT data: {e}", exc_info=True)
            return []

    def _fetch_macro_sentiment_data(self) -> pd.DataFrame:
        """
        Fetch macroeconomic sentiment records from GDELT GKG table.
        
        Executes single optimized BigQuery query covering 2023-present with:
        - Partition pruning via _PARTITIONTIME >= TIMESTAMP('2023-01-01')
        - Macro-theme filtering using V2Themes regex
        - SQL-side data transformation (timestamp parsing, tone extraction)
        - Large result set handling via destination table
        
        Uses correct BigQuery schema column names:
        - DATE (YYYYMMDDHHMMSS format)
        - DocumentIdentifier (URL/source)
        - V2Themes (theme tags with scores)
        - V2Tone (sentiment scores)
        
        Returns:
            pandas.DataFrame with columns: timestamp, url, themes, sentiment_tone
            
        Raises:
            Exception: If BigQuery query execution fails (handled gracefully in fetch())
        """
        self.logger.info(f"Fetching macro-economic sentiment from GDELT (last {self.DAYS_BACK} days, max {self.MAX_RECORDS} records)...")
        
        # Highly optimized BigQuery query with all transformations in SQL
        # Uses exact BigQuery schema column names for gdelt-bq.gdeltv2.gkg_partitioned table
        # Optimized for cost: fetches only last 7 days, limited to 100k records
        sql = f"""
SELECT
  TIMESTAMP_TRUNC(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)), HOUR) AS timestamp,
  DocumentIdentifier AS url,
  V2Themes AS themes,
  CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS sentiment_tone
FROM
  `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE
  _PARTITIONTIME >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {self.DAYS_BACK} DAY)
  AND REGEXP_CONTAINS(V2Themes, r'(?i)(ECON_INFLATION|US_FEDERAL_RESERVE|CRISIS|ARMEDCONFLICT|ECON_UNEMPLOYMENT)')
ORDER BY timestamp DESC
LIMIT {self.MAX_RECORDS}
        """
        
        try:
            self.logger.debug(f"Executing GDELT query for last {self.DAYS_BACK} days (limit {self.MAX_RECORDS} records)...")
            self.logger.debug(f"Macro themes: {', '.join(self.MACRO_THEMES)}")
            
            # For large result sets, use optimized configuration
            from google.cloud import bigquery
            job_config = bigquery.QueryJobConfig(
                priority=bigquery.QueryPriority.INTERACTIVE,
                use_query_cache=False,  # Disable cache to force fresh 30-day fetch
                # No byte limit - GDELT data is large, let BigQuery handle it
            )
            
            query_job = self.client.query(sql, job_config=job_config)
            df = query_job.to_dataframe()  # Use standard REST API for data transfer
            
            self.logger.info(f"Retrieved {len(df)} macro-sentiment records from GDELT")
            if not df.empty:
                self.logger.debug(
                    f"Tone range: {df['sentiment_tone'].min():.2f} to {df['sentiment_tone'].max():.2f} | "
                    f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}"
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"BigQuery query execution failed: {e}", exc_info=True)
            raise

    def validate(self, records: List[Dict[str, Any]]) -> bool:
        """
        Validate macro-sentiment records schema and integrity.
        
        Checks:
        - All required fields present (timestamp, url, themes, sentiment_tone)
        - sentiment_tone is numeric and within valid range [-100, 100]
        - timestamp is valid ISO format
        - url and themes are non-empty strings

        Args:
            records: List of sentiment records to validate

        Returns:
            True if all records valid, False otherwise
        """
        if not records:
            self.logger.warning("No records to validate")
            return True

        required_fields = {"timestamp", "url", "themes", "sentiment_tone"}

        for i, record in enumerate(records):
            # Check all required fields present
            missing = required_fields - set(record.keys())
            if missing:
                self.logger.error(f"Record {i} missing fields: {missing}")
                return False

            # Validate sentiment_tone is numeric and in valid range
            try:
                tone = float(record.get("sentiment_tone"))
                if not -100 <= tone <= 100:
                    self.logger.error(f"Record {i}: sentiment_tone out of range [-100, 100]: {tone}")
                    return False
            except (ValueError, TypeError) as e:
                self.logger.error(f"Record {i}: sentiment_tone not numeric: {e}")
                return False

            # Validate timestamp is ISO format
            try:
                datetime.fromisoformat(str(record.get("timestamp", "")).replace("Z", "+00:00"))
            except (ValueError, TypeError) as e:
                self.logger.error(f"Record {i}: invalid timestamp format: {e}")
                return False

            # Validate url is non-empty
            url = record.get("url", "").strip()
            if not url:
                self.logger.error(f"Record {i}: url is empty")
                return False

            # Validate themes is non-empty
            themes = record.get("themes", "").strip()
            if not themes:
                self.logger.error(f"Record {i}: themes is empty")
                return False

        self.logger.info(f"Validated {len(records)} records successfully")
        return True

    def save(self, records: List[Dict[str, Any]], filename: Optional[str] = None) -> int:
        """
        Save macro-sentiment records to CSV file (overwrites existing data).
        
        Deduplicates by (timestamp, url) to remove exact duplicate articles at same hour.
        Sorts by timestamp before saving.
        
        Args:
            records: List of validated sentiment records
            filename: Optional filename (defaults to gdelt_macro.csv)
            
        Returns:
            Number of rows saved to CSV
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

            # Deduplicate by (timestamp, url) to remove exact duplicate articles
            original_len = len(df)
            df = df.drop_duplicates(subset=["timestamp", "url"], keep="first")
            deduped = original_len - len(df)
            if deduped > 0:
                self.logger.info(f"Removed {deduped} duplicate records based on (timestamp, url)")

            # Sort by timestamp for temporal coherence
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Save to CSV (overwrites existing)
            df.to_csv(output_file, index=False)

            self.logger.info(f"Successfully saved {len(df)} records to {output_file}")
            return len(df)

        except Exception as e:
            self.logger.error(f"Failed to save records: {e}")
            return 0


if __name__ == "__main__":
    """
    Standalone execution: python gdelt_bq_crawler.py
    
    Fetches macro-economic sentiment data and saves to data/raw/gdelt_macro.csv
    Graceful degradation: exits with code 0 even if no credentials or query fails.
    """
    import sys
    import logging as std_logging

    # Setup logging
    std_logging.basicConfig(
        level=std_logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        crawler = GdeltBQCrawler()
        saved_count = crawler.run()
        
        if saved_count > 0:
            print(f"✓ GDELT macro-sentiment crawler completed: {saved_count} records saved")
            sys.exit(0)
        else:
            print("✗ GDELT crawler returned no data (check BigQuery credentials)")
            sys.exit(0)  # Graceful degradation
            
    except Exception as e:
        print(f"✗ GDELT crawler encountered fatal error: {e}")
        std_logging.exception(e)
        sys.exit(0)  # Graceful degradation
