"""
Exception Hierarchy for Crawler Framework.

Distinguishes between:
- Temporary failures (transient, recoverable)
- Permanent failures (won't succeed without intervention)
- Partial successes (some data collected despite errors)
"""

from typing import Any, Dict, List, Optional


class CrawlerException(Exception):
    """Base exception for all crawler errors."""

    pass


class TemporaryFailure(CrawlerException):
    """
    Transient error - should retry.

    Examples:
    - HTTP 429 (Rate Limited)
    - HTTP 503 (Service Unavailable)
    - Timeout
    - Connection refused
    - DNS resolution failure

    Caller should implement exponential backoff retry.
    """

    def __init__(
        self,
        message: str,
        retry_after_seconds: Optional[float] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.original_exception = original_exception


class PermanentFailure(CrawlerException):
    """
    Permanent error - won't succeed without intervention.

    Examples:
    - HTTP 401 (Unauthorized - invalid credentials)
    - HTTP 403 (Forbidden - insufficient permissions)
    - HTTP 404 (Not Found)
    - Invalid configuration
    - Authentication token expired

    Caller should log and skip, not retry.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.original_exception = original_exception


class PartialSuccess(CrawlerException):
    """
    Partial data collected despite errors.

    Example: Successfully fetched 900 of 1000 expected records before hitting
    rate limit. Partial data is still valuable and should be saved.

    Caller should:
    1. Save the records that succeeded
    2. Log the partial failure
    3. Optionally retry for missing data
    """

    def __init__(
        self,
        message: str,
        records: List[Dict[str, Any]],
        failed_count: int = 0,
    ):
        super().__init__(message)
        self.records = records
        self.failed_count = failed_count
        self.success_count = len(records)

    def __str__(self) -> str:
        return (
            f"{super().__str__()} "
            f"[Success: {self.success_count}, Failed: {self.failed_count}]"
        )


class ValidationFailure(CrawlerException):
    """
    Data validation failed - records don't match schema.

    Examples:
    - Required field missing
    - Invalid timestamp format
    - Data type mismatch
    - Null values in non-nullable fields
    """

    def __init__(
        self,
        message: str,
        failed_records: Optional[List[Dict[str, Any]]] = None,
        error_details: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.failed_records = failed_records or []
        self.error_details = error_details or []


class ConfigError(CrawlerException):
    """Configuration is invalid or incomplete."""

    def __init__(self, message: str, missing_keys: Optional[List[str]] = None):
        super().__init__(message)
        self.missing_keys = missing_keys or []


class DataStorageError(CrawlerException):
    """Error writing data to storage (CSV, database, etc.)."""

    pass


def classify_http_error(status_code: int) -> type:
    """
    Classify HTTP error as temporary or permanent.

    Args:
        status_code: HTTP status code

    Returns:
        Exception class: TemporaryFailure or PermanentFailure
    """
    # Temporary (4xx rate limit or 5xx server errors)
    if status_code == 429:  # Too Many Requests
        return TemporaryFailure
    if 500 <= status_code < 600:  # Server errors
        return TemporaryFailure

    # Permanent (other 4xx client errors)
    if 400 <= status_code < 500:
        return PermanentFailure

    # Unknown - assume temporary
    return TemporaryFailure
