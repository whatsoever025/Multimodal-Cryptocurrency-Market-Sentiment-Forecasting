"""
Base Crawler Abstract Class - Foundation for all data crawlers.

Provides:
- Standardized interface (fetch, validate, save, run)
- HTTP session management with connection pooling
- Retry logic with exponential backoff
- Rate limiting integration
- Unified error handling
- Health monitoring hooks
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class CrawlerConfig:
    """Standardized crawler configuration."""

    # Timeout settings (connect_timeout, read_timeout)
    timeout_seconds: Tuple[float, float] = (5.0, 25.0)

    # Retry strategy
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Rate limiting (per crawler instance)
    rate_limit_delay_seconds: float = 1.0
    requests_per_minute: int = 60

    # Data collection
    batch_size: int = 100

    def __post_init__(self):
        """Validate configuration."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.timeout_seconds[0] <= 0 or self.timeout_seconds[1] <= 0:
            raise ValueError("timeout values must be positive")


class BaseCrawler(ABC):
    """
    Abstract base class for all cryptocurrency data crawlers.

    Guarantees:
    - Connection pooling (efficient HTTP requests)
    - Retry logic with exponential backoff
    - Rate limiting enforcement
    - Consistent error handling
    - Standard run() orchestration flow
    """

    def __init__(
        self,
        base_path: str = "data/raw",
        config: Optional[CrawlerConfig] = None,
    ):
        """
        Initialize base crawler with configuration.

        Args:
            base_path: Directory for saving raw data files
            config: Crawler configuration (uses defaults if None)
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.config = config or CrawlerConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # HTTP session with connection pooling
        self.session = self._init_session()

        # Rate limiting state (token bucket)
        self.tokens = float(self.config.requests_per_minute) / 60.0
        self.last_token_refill = time.time()
        self.minute_counter = 0
        self.minute_start = time.time()

        self.logger.debug(
            f"Initialized {self.__class__.__name__} with "
            f"timeout={self.config.timeout_seconds}s, "
            f"max_retries={self.config.max_retries}"
        )

    def _init_session(self) -> requests.Session:
        """
        Initialize reusable HTTP session with connection pooling.

        Reusing sessions significantly improves performance:
        - Connection pooling reduces TCP handshake overhead
        - SSL session reuse avoids repeated TLS handshakes
        - Typically 20-30% latency reduction for paginated requests
        """
        session = requests.Session()

        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=Retry(
                total=0,  # We handle retries manually
                connect=0,
                backoff_factor=0,
            ),
        )

        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    @abstractmethod
    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch data from source.

        Must return:
            List of record dictionaries. Each record should contain
            keys: timestamp, source, asset (at minimum).

        Raises:
            CrawlerException subclass on error
        """
        pass

    @abstractmethod
    def validate(self, records: List[Dict[str, Any]]) -> bool:
        """
        Validate data schema and integrity.

        Should check:
        - Required fields present
        - Timestamp validity
        - Data type correctness
        - No corrupted records

        Returns:
            True if all records valid, False if validation fails

        Logs:
            Error details for failed records
        """
        pass

    @abstractmethod
    def save(self, records: List[Dict[str, Any]]) -> int:
        """
        Save records to persistent storage.

        Args:
            records: List of validated records

        Returns:
            Number of rows successfully saved (may be less than input
            if deduplication occurs)

        Raises:
            IOError if save fails
        """
        pass

    def run(self) -> int:
        """
        Standard orchestration flow with error handling.

        Implements:
        1. Fetch from source
        2. Validate data
        3. Save to storage
        4. Return count of saved records

        Returns:
            Number of records saved (0 if any step fails)

        Note:
            Exceptions are caught and logged. Caller receives count only.
        """
        self.logger.info(f"Starting {self.__class__.__name__}")

        try:
            records = self.fetch()
            self.logger.info(f"Fetched {len(records)} records")

            if not records:
                self.logger.warning("No records fetched from source")
                return 0

            if not self.validate(records):
                self.logger.error("Data validation failed")
                return 0

            saved_count = self.save(records)
            self.logger.info(f"Completed: {saved_count} records saved")
            return saved_count

        except Exception as e:
            self.logger.error(
                f"Failed with exception: {e}",
                exc_info=True,
            )
            return 0

    def request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> requests.Response:
        """
        HTTP request with exponential backoff retry logic.

        Handles:
        - Rate limiting (HTTP 429): waits Retry-After
        - Server errors (HTTP 5xx): retries with exponential backoff
        - Connection errors: retries with exponential backoff
        - Client errors (HTTP 4xx, NOT 429): fails immediately

        Args:
            method: HTTP method ('GET', 'POST', etc.)
            url: Request URL
            **kwargs: Passed to requests.Session.request()

        Returns:
            requests.Response object

        Raises:
            requests.exceptions.RequestException on permanent failure

        Example:
            response = self.request_with_retry('GET', 'https://api.example.com/data')
            data = response.json()
        """
        # Apply rate limiting before each request
        self._rate_limit_wait()

        # Set timeout if not provided
        kwargs.setdefault("timeout", self.config.timeout_seconds)

        last_exception: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.session.request(method, url, **kwargs)

                # Handle rate limiting (429)
                if response.status_code == 429:
                    retry_after = int(
                        response.headers.get(
                            "Retry-After",
                            self.config.retry_delay_seconds,
                        )
                    )
                    self.logger.warning(
                        f"Rate limited (429). Waiting {retry_after}s "
                        f"(attempt {attempt + 1}/{self.config.max_retries + 1})"
                    )
                    time.sleep(retry_after)
                    continue

                # Handle server errors (5xx) with exponential backoff
                if 500 <= response.status_code < 600:
                    if attempt < self.config.max_retries:
                        delay = self.config.retry_delay_seconds * (2 ** attempt)
                        self.logger.warning(
                            f"Server error {response.status_code}. "
                            f"Retrying in {delay:.1f}s "
                            f"(attempt {attempt + 1}/{self.config.max_retries + 1})"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        response.raise_for_status()

                # Client errors (4xx) fail immediately
                response.raise_for_status()
                return response

            except (requests.Timeout, requests.ConnectionError) as e:
                last_exception = e

                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay_seconds * (2 ** attempt)
                    self.logger.warning(
                        f"Connection error: {type(e).__name__}. "
                        f"Retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{self.config.max_retries + 1})"
                    )
                    time.sleep(delay)
                else:
                    break

            except requests.HTTPError as e:
                # For 4xx errors, fail immediately unless it's 429 (handled above)
                if e.response.status_code < 500:
                    raise
                last_exception = e

        # All retries exhausted
        self.logger.error(
            f"Request failed after {self.config.max_retries + 1} attempts: {url}"
        )
        raise last_exception or requests.RequestException(
            f"Failed to {method} {url} after {self.config.max_retries + 1} retries"
        )

    def _rate_limit_wait(self):
        """
        Enforce rate limiting using token bucket algorithm.

        Supports two levels:
        1. Per-second rate (via token bucket with burst)
        2. Per-minute rate (absolute limit)

        Blocks caller if either limit would be exceeded.
        """
        # Refill tokens based on elapsed time
        now = time.time()
        elapsed = now - self.last_token_refill
        tokens_per_second = self.config.requests_per_minute / 60.0
        new_tokens = elapsed * tokens_per_second

        # Cap tokens at per-minute limit (allows burst)
        self.tokens = min(self.config.requests_per_minute, self.tokens + new_tokens)
        self.last_token_refill = now

        # Check per-minute hard limit
        if now - self.minute_start >= 60:
            self.minute_counter = 0
            self.minute_start = now

        if self.minute_counter >= self.config.requests_per_minute:
            sleep_until = self.minute_start + 60
            sleep_time = sleep_until - now
            self.logger.warning(
                f"Per-minute rate limit reached ({self.config.requests_per_minute} req/min). "
                f"Sleeping {sleep_time:.1f}s"
            )
            time.sleep(max(0, sleep_time))
            self.minute_counter = 0
            self.minute_start = time.time()

        # Check per-second rate (token bucket)
        while self.tokens < 1.0:
            sleep_time = (1.0 - self.tokens) / tokens_per_second
            self.logger.debug(f"Rate limiting: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)

            # Refill after sleep
            elapsed = time.time() - self.last_token_refill
            new_tokens = elapsed * tokens_per_second
            self.tokens = min(
                self.config.requests_per_minute,
                self.tokens + new_tokens,
            )
            self.last_token_refill = time.time()

        # Consume 1 token
        self.tokens -= 1.0
        self.minute_counter += 1

    def __del__(self):
        """Clean up session on object destruction."""
        if hasattr(self, "session"):
            self.session.close()
