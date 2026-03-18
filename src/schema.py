"""
Standard Data Schema for All Crawlers.

Provides:
- Canonical CrawlRecord format (all crawlers output this)
- Data type definitions (OHLCV, sentiment, text, etc.)
- CSV serialization with JSON values storage
- Schema validation helpers
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Types of data collected by crawlers."""

    OHLCV = "ohlcv"  # Candlestick: open, high, low, close, volume
    SENTIMENT = "sentiment"  # Index/classification: bullish/neutral/bearish
    TEXT = "text"  # News/social posts with text content
    FUNDING_RATE = "funding_rate"  # Futures funding rates
    OPEN_INTEREST = "open_interest"  # Futures open interest
    LIQUIDATIONS = "liquidations"  # Liquidation events
    DERIVATIVES_OI = "derivatives_oi"  # Exchange-level open interest
    TICKER = "ticker"  # Current price/volume snapshots


@dataclass
class CrawlRecord:
    """
    Canonical record format for all crawler output.

    Design:
    - timestamp: Always UTC, ISO 8601, hourly-aligned
    - asset: Standard code (BTC, ETH) for consistency
    - data_type: What kind of data (OHLCV, sentiment, etc.)
    - provider: Source API (binance, coingecko, cryptopanic)
    - source_id: Unique ID in provider's system
    - values: Provider-specific fields in JSON (flexible)
    """

    timestamp: datetime
    asset: str
    data_type: DataType
    provider: str
    source_id: str  # Unique ID for deduplication
    values: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate record on construction."""
        if not self.timestamp.tzinfo:
            # Ensure timezone-aware (UTC)
            logger.warning(f"Record timestamp not timezone-aware: {self.timestamp}")

        if not self.asset or len(self.asset) > 10:
            raise ValueError(f"Invalid asset code: {self.asset}")

        if not self.provider or len(self.provider) > 50:
            raise ValueError(f"Invalid provider: {self.provider}")

        if not self.source_id or len(self.source_id) > 500:
            raise ValueError(f"Invalid source_id: {self.source_id}")

    def to_csv_row(self) -> Dict[str, str]:
        """
        Convert to CSV row (all values as strings).

        Output columns:
        - timestamp: ISO 8601
        - asset: Standard code
        - data_type: Enum value
        - provider: Provider name
        - source_id: Record ID
        - values_json: JSON-serialized dict
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
            "data_type": self.data_type.value,
            "provider": self.provider,
            "source_id": self.source_id,
            "values_json": json.dumps(self.values),
        }

    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> "CrawlRecord":
        """Reconstruct CrawlRecord from CSV row."""
        values = {}
        if row.get("values_json"):
            try:
                values = json.loads(row["values_json"])
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse values_json: {e}")

        return cls(
            timestamp=datetime.fromisoformat(row["timestamp"]),
            asset=row["asset"],
            data_type=DataType(row["data_type"]),
            provider=row["provider"],
            source_id=row["source_id"],
            values=values,
        )

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get value from values dict with default."""
        return self.values.get(key, default)

    @property
    def csv_columns(self) -> List[str]:
        """Standard CSV column order for all records."""
        return [
            "timestamp",
            "asset",
            "data_type",
            "provider",
            "source_id",
            "values_json",
        ]


# ============================================================================
# RECORD BUILDERS - Factory functions for common data types
# ============================================================================


def make_ohlcv_record(
    timestamp: datetime,
    asset: str,
    provider: str,
    source_id: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
) -> CrawlRecord:
    """Build OHLCV record (candlestick)."""
    return CrawlRecord(
        timestamp=timestamp,
        asset=asset,
        data_type=DataType.OHLCV,
        provider=provider,
        source_id=source_id,
        values={
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
    )


def make_sentiment_record(
    timestamp: datetime,
    asset: str,
    provider: str,
    source_id: str,
    value: int,
    classification: str,
) -> CrawlRecord:
    """Build sentiment index record."""
    return CrawlRecord(
        timestamp=timestamp,
        asset=asset,
        data_type=DataType.SENTIMENT,
        provider=provider,
        source_id=source_id,
        values={
            "value": value,
            "classification": classification,
        },
    )


def make_text_record(
    timestamp: datetime,
    asset: str,
    provider: str,
    source_id: str,
    text: str,
    source_url: Optional[str] = None,
) -> CrawlRecord:
    """Build text/news record."""
    return CrawlRecord(
        timestamp=timestamp,
        asset=asset,
        data_type=DataType.TEXT,
        provider=provider,
        source_id=source_id,
        values={
            "text": text,
            "source_url": source_url,
        },
    )


def make_funding_rate_record(
    timestamp: datetime,
    asset: str,
    provider: str,
    source_id: str,
    funding_rate: float,
    funding_timestamp: Optional[datetime] = None,
) -> CrawlRecord:
    """Build futures funding rate record."""
    return CrawlRecord(
        timestamp=timestamp,
        asset=asset,
        data_type=DataType.FUNDING_RATE,
        provider=provider,
        source_id=source_id,
        values={
            "funding_rate": funding_rate,
            "funding_timestamp": funding_timestamp.isoformat()
            if funding_timestamp
            else None,
        },
    )


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================


def validate_ohlcv_values(values: Dict[str, Any]) -> bool:
    """Validate OHLCV record values."""
    required_keys = {"open", "high", "low", "close", "volume"}
    if not required_keys.issubset(values.keys()):
        logger.error(f"OHLCV missing keys: {required_keys - values.keys()}")
        return False

    for key in required_keys:
        try:
            value = float(values[key])
            if value < 0:
                logger.error(f"OHLCV {key} is negative: {value}")
                return False
        except (ValueError, TypeError):
            logger.error(f"OHLCV {key} is not numeric: {values[key]}")
            return False

    return True


def validate_sentiment_values(values: Dict[str, Any]) -> bool:
    """Validate sentiment record values."""
    if "value" not in values or "classification" not in values:
        logger.error("Sentiment missing value or classification")
        return False

    try:
        int(values["value"])
    except (ValueError, TypeError):
        logger.error(f"Sentiment value not numeric: {values['value']}")
        return False

    return True


def validate_text_values(values: Dict[str, Any]) -> bool:
    """Validate text record values."""
    if "text" not in values:
        logger.error("Text record missing text field")
        return False

    if not isinstance(values["text"], str) or not values["text"].strip():
        logger.error(f"Text record has empty text: {values.get('text')}")
        return False

    return True
