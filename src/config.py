"""
Centralized Configuration for Cryptocurrency Assets and API Tiers.

Provides:
- Single source of truth for supported assets
- API code mappings per provider (e.g., 'BTC' → 'bitcoin' for CoinGecko)
- Regex patterns for asset detection
- API tier limits for rate limiting
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import re


@dataclass
class Asset:
    """Cryptocurrency asset metadata."""

    code: str  # Standard code (BTC, ETH, SOL)
    name: str  # Full name (Bitcoin, Ethereum, Solana)
    icon_emoji: str = ""

    # Provider-specific codes/identifiers
    api_codes: Dict[str, str] = field(default_factory=dict)

    # Regex patterns for detection in text
    detection_patterns: List[str] = field(default_factory=list)

    def get_code_for_provider(self, provider: str) -> Optional[str]:
        """Get provider-specific code (e.g., 'ETH' → 'ethereum' for coingecko)."""
        return self.api_codes.get(provider)

    def detect_in_text(self, text: str) -> bool:
        """Check if asset mentioned in text using detection patterns."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.detection_patterns)


@dataclass
class APITierConfig:
    """Rate limit configuration for specific API tier."""

    provider: str
    tier_name: str
    requests_per_minute: int
    requests_per_second: float = 1.0
    burst_allowed: int = 5
    concurrent_connections: int = 10

    def __post_init__(self):
        # Calculate consistent per-second from per-minute
        self.requests_per_second = self.requests_per_minute / 60.0
        self.burst_allowed = max(self.burst_allowed, int(self.requests_per_second * 2))


# ============================================================================
# SUPPORTED ASSETS - Central Registry
# ============================================================================

SUPPORTED_ASSETS = {
    "BTC": Asset(
        code="BTC",
        name="Bitcoin",
        icon_emoji="₿",
        api_codes={
            "binance": "BTC/USDT",
            "binance_futures": "BTC/USDT",
            "coingecko": "bitcoin",
            "cryptopanic": "BTC",
            "stocktwits": "BTC.X",
        },
        detection_patterns=[
            r"\bbtc\b",
            r"\bbitcoin\b",
            r"btc/",
        ],
    ),
    "ETH": Asset(
        code="ETH",
        name="Ethereum",
        icon_emoji="Ξ",
        api_codes={
            "binance": "ETH/USDT",
            "binance_futures": "ETH/USDT",
            "coingecko": "ethereum",
            "cryptopanic": "ETH",
            "stocktwits": "ETH.X",
        },
        detection_patterns=[
            r"\beth\b",
            r"\bethereum\b",
            r"eth/",
        ],
    ),
    "SOL": Asset(
        code="SOL",
        name="Solana",
        icon_emoji="◎",
        api_codes={
            "binance": "SOL/USDT",
            "coingecko": "solana",
        },
        detection_patterns=[
            r"\bsol\b",
            r"\bsolana\b",
        ],
    ),
}


def get_asset(code: str) -> Optional[Asset]:
    """Get asset by standard code."""
    return SUPPORTED_ASSETS.get(code.upper())


def get_asset_code_for_provider(asset_code: str, provider: str) -> str:
    """
    Get provider-specific code for an asset.

    Args:
        asset_code: Standard code (e.g., 'BTC', 'ETH')
        provider: Provider name (e.g., 'binance', 'coingecko')

    Returns:
        Provider-specific code (e.g., 'BTC/USDT' for Binance)

    Raises:
        ValueError: If asset or provider not found
    """
    asset = get_asset(asset_code)
    if not asset:
        raise ValueError(
            f"Asset '{asset_code}' not in registry. "
            f"Available: {list(SUPPORTED_ASSETS.keys())}"
        )

    provider_code = asset.get_code_for_provider(provider)
    if provider_code is None:
        raise ValueError(
            f"Asset '{asset_code}' not available for provider '{provider}'. "
            f"Available providers: {list(asset.api_codes.keys())}"
        )

    return provider_code


def register_asset(asset: Asset):
    """Register new asset globally."""
    SUPPORTED_ASSETS[asset.code] = asset


def detect_assets_in_text(text: str) -> List[str]:
    """
    Detect which supported assets are mentioned in text.

    Args:
        text: Text to analyze

    Returns:
        List of detected asset codes (e.g., ['BTC', 'ETH'])
    """
    detected = []
    for code, asset in SUPPORTED_ASSETS.items():
        if asset.detect_in_text(text):
            detected.append(code)
    return detected


# ============================================================================
# API TIER CONFIGURATIONS - Rate Limits per Provider
# ============================================================================

API_TIERS = {
    # Binance
    "binance.spot": APITierConfig(
        provider="binance",
        tier_name="spot_api",
        requests_per_minute=1200,  # 20 req/sec
        concurrent_connections=20,
    ),
    "binance.futures": APITierConfig(
        provider="binance",
        tier_name="futures_api",
        requests_per_minute=1200,
        concurrent_connections=20,
    ),
    # CoinGecko
    "coingecko.free": APITierConfig(
        provider="coingecko",
        tier_name="free_tier",
        requests_per_minute=45,  # ~0.75 req/sec, very conservative
    ),
    "coingecko.pro": APITierConfig(
        provider="coingecko",
        tier_name="pro_tier",
        requests_per_minute=600,
        concurrent_connections=5,
    ),
    # CryptoPanic
    "cryptopanic.developer": APITierConfig(
        provider="cryptopanic",
        tier_name="developer_tier",
        requests_per_minute=180,  # 3 req/sec
        concurrent_connections=5,
    ),
    # Alternative.me (Fear & Greed)
    "alternative.me": APITierConfig(
        provider="alternative_me",
        tier_name="public",
        requests_per_minute=60,  # Conservative limit
        concurrent_connections=2,
    ),
    # Reddit (public endpoint)
    "reddit.public": APITierConfig(
        provider="reddit",
        tier_name="public_endpoint",
        requests_per_minute=30,  # Conservative - Reddit is strict
        concurrent_connections=4,
    ),
    # StockTwits
    "stocktwits.public": APITierConfig(
        provider="stocktwits",
        tier_name="public_endpoint",
        requests_per_minute=120,  # 2 req/sec
        concurrent_connections=5,
    ),
}


def get_api_tier(tier_id: str) -> Optional[APITierConfig]:
    """Get rate limit config for specific API tier."""
    return API_TIERS.get(tier_id)


def get_api_tier_by_provider(provider: str, tier_name: str = "default") -> Optional[APITierConfig]:
    """
    Get rate limit config by provider and tier name.

    Args:
        provider: Provider name (e.g., 'binance', 'coingecko')
        tier_name: Tier name (e.g., 'free', 'pro')

    Returns:
        APITierConfig or None if not found
    """
    # Try exact match
    key = f"{provider}.{tier_name}"
    if key in API_TIERS:
        return API_TIERS[key]

    # Try provider.default
    default_key = f"{provider}.default"
    if default_key in API_TIERS:
        return API_TIERS[default_key]

    # Return first matching provider
    for tier_id, config in API_TIERS.items():
        if tier_id.startswith(f"{provider}."):
            return config

    return None


# ============================================================================
# DATA SCHEMA CONSTANTS
# ============================================================================

# Standard columns for all crawler output
STANDARD_COLUMNS = [
    "timestamp",  # ISO 8601 UTC
    "asset",  # Standard code (BTC, ETH, etc.)
    "source",  # Crawler/provider name
    "source_id",  # Unique ID in source system
]

# CSV export columns
CSV_EXPORT_COLUMNS = STANDARD_COLUMNS + ["values_json"]

# Examples of asset signatures
EXAMPLE_ASSET_SIGNATURES = {
    "BTC": {"prices": "float", "volume": "float"},
    "sentiment_index": {"value": "int", "classification": "str"},
    "crypto_text": {"text": "str", "source": "str"},
}
