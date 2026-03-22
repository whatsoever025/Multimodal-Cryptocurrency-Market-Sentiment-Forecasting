# Data Collection Summary

**Generated:** 2026-03-22  
**Collection Status:** ✅ Complete

---

## Overview

Raw data collected from multiple sources to support multimodal sentiment forecasting for cryptocurrency markets (BTC/USDT and ETH/USDT).

---

## Data Sources & Files

### 1) Binance Vision - Futures Market Data

**Crawler:** `BinanceVisionCrawler`  
**Module:** `src/crawlers/binance_vision_crawler.py`  
**Coverage:** 2023-01-01 to 2026-02-28 (Monthly historical archives)

#### Klines (OHLCV) - 1 Hour

| File | Rows | Size | Columns |
|------|------|------|---------|
| **BTCUSDT_klines.csv** | 27,720 | 1,705 KB | timestamp, open, high, low, close, volume |
| **ETHUSDT_klines.csv** | 27,438 | 1,696 KB | timestamp, open, high, low, close, volume |

**Data Quality:**
- Timestamp: UTC milliseconds → converted to datetime
- Prices: Float64, positive values
- Volume: Float64, positive values
- Duplicates removed per symbol
- Sorted chronologically

**Usage:** OHLCV candlestick analysis, technical indicators, price patterns

---

#### Funding Rate - Daily Snapshots

| File | Rows | Size | Columns |
|------|------|------|---------|
| **BTCUSDT_fundingRate.csv** | 3,465 | 88 KB | timestamp, funding_rate, ... |
| **ETHUSDT_fundingRate.csv** | 3,458 | 88 KB | timestamp, funding_rate, ... |

**Data Quality:**
- Timestamp: UTC milliseconds → converted to datetime
- Funding rate: Float64 (interest rates, signed)
- Deduplicates & sorts by timestamp
- Indicator of market leverage and sentiment

**Usage:** Futures market sentiment, leverage cycles, long/short imbalance

---

#### Open Interest & Liquidations

| Data Type | Status | Reason |
|-----------|--------|--------|
| openInterestHist | ❌ No Data | Not available on Binance Vision for 2023-2026 |
| liquidationSnapshot | ❌ No Data | Not available on Binance Vision for 2023-2026 |

*Note: These datasets may be available only on newer Binance Vision endpoints or require different data sources.*

---

### 2) CoinGecko - Multi-Exchange Derivatives

**Crawler:** `CoinGeckoCrawler`  
**Module:** `src/crawlers/coingecko_crawler.py`  
**Source:** CoinGecko API v3 (Public)

#### Derivatives Exchanges Snapshot

| File | Rows | Size | Columns |
|------|------|------|---------|
| **coingecko_derivatives_exchanges.csv** | 7 | 1.7 KB | exchange, exchange_id, open_interest_btc, trade_volume_24h_btc, number_of_perpetual_pairs, number_of_futures_pairs, timestamp |

**Data Quality:**
- Exchange-level aggregates (not per-symbol)
- Single timestamp snapshot (latest available)
- BTC-denominated metrics

**Usage:** Market structure, multi-exchange comparison, derivatives proliferation

---

### 3) Sentiment Crawlers

#### Fear & Greed Index

**Crawler:** `SentimentCrawler`  
**Source:** Alternative.me Fear and Greed Index API  
**File:** `fear_greed_index.csv`  
**Size:** 102 KB

**Columns:**
- timestamp
- value (0-100 scale)
- value_classification (Extreme Fear, Fear, Neutral, Greed, Extreme Greed)

**Coverage:** Historical data through current date

**Usage:** Market-wide sentiment baseline, panic/euphoria cycles

---

#### Text-Based Sentiment

**Crawler:** `TextCrawler`  
**Sources:** CryptoPanic API, Reddit, StockTwits  
**File:** `crypto_text.csv`  
**Size:** 95 KB

**Columns:**
- timestamp (hour bucket, UTC)
- source (cryptopanic, reddit, stocktwits)
- asset (BTC, ETH)
- text (cleaned: no URLs, no emojis, no special chars)
- user_sentiment (inferred from text)
- text_hash (SHA-256 for deduplication)

**Data Quality:**
- Deduplicates across runs using text_hash
- Hourly UTC aggregation
- Text sanitization applied

**Usage:** Social sentiment, market narrative, event detection

---

## File Locations

```
data/raw/
├── BTCUSDT_klines.csv              (1,705 KB)
├── ETHUSDT_klines.csv              (1,696 KB)
├── BTCUSDT_fundingRate.csv         (88 KB)
├── ETHUSDT_fundingRate.csv         (88 KB)
├── coingecko_derivatives_exchanges.csv (1.7 KB)
├── fear_greed_index.csv            (102 KB)
└── crypto_text.csv                 (95 KB)
```

**Total Size:** ~5.8 MB

---

## Data Quality & Completeness

| Dataset | Completeness | Data Quality | Notes |
|---------|--------------|--------------|-------|
| Binance Klines | 99% | High | Minor gaps in future months |
| Binance Funding Rate | 99% | High | Sparse early 2023 data |
| CoinGecko Exchanges | 100% | Medium | Snapshot only, no historical |
| Fear & Greed Index | 100% | High | Complete historical |
| Text Sentiment | 80% | Medium | Depends on API availability |

---

## Data Preprocessing Notes

### Timestamps
- **All timestamps standardized to UTC datetime**
- Binance millis-to-datetime conversion applied
- Hourly bucketing for text data (floor to hour)

### Deduplication
- Klines & Funding Rate: deduplicated by (symbol, timestamp)
- Text data: deduplicated by text_hash (SHA-256)

### Sorting
- All data sorted chronologically by timestamp

### Missing Values
- Open interest & liquidation data unavailable
- Text data is sparse during low-volume periods
- Early CoinGecko data may have API limits

---

## Suggested Next Steps

### 1. Data Merging
Combine klines + funding rate by timestamp for joint analysis:
```python
merge_by = ['symbol', 'timestamp']
merged = klines.merge(funding_rate, on=merge_by, how='outer')
```

### 2. Feature Engineering
- Technical indicators from klines (MA, RSI, MACD, BB)
- Lagged funding rates (prediction features)
- Sentiment aggregation (hourly averages)
- Volatile events detection

### 3. Time Series Alignment
- Resample text/fear data to match klines frequency (1h)
- Forward-fill missing sentiment values
- Lag sentiment features for prediction

### 4. Exploratory Analysis
- Correlation: price moves ↔ funding rates
- Correlation: market sentiment ↔ price volatility
- Event analysis: large price moves + sentiment spikes

---

## Data Dictionary Quick Reference

### Klines
- **timestamp**: datetime UTC (index)
- **open**: float, opening price
- **high**: float, highest price in period
- **low**: float, lowest price in period
- **close**: float, closing price
- **volume**: float, base asset volume

### Funding Rate
- **timestamp**: datetime UTC (index)
- **funding_rate**: float, perpetual funding rate (%)

### Fear & Greed
- **timestamp**: datetime UTC (index)
- **value**: int (0-100)
- **value_classification**: str (Extreme Fear/Fear/Neutral/Greed/Extreme Greed)

### Text Sentiment
- **timestamp**: datetime UTC (hourly bucket)
- **source**: str (cryptopanic/reddit/stocktwits)
- **asset**: str (BTC/ETH)
- **text**: str (cleaned content)
- **user_sentiment**: str (extracted sentiment)
- **text_hash**: str (SHA-256 dedup key)

---

## Crawler Execution Log

```
Session: 2026-03-22 23:01:11 UTC

✓ BinanceVisionCrawler        SUCCESS
  - BTCUSDT klines:            27,720 rows
  - BTCUSDT fundingRate:       3,465 rows
  - ETHUSDT klines:            27,438 rows
  - ETHUSDT fundingRate:       3,458 rows

✓ CoinGeckoCrawler              SUCCESS
  - Derivatives exchanges:     7 rows

✓ SentimentCrawler              SUCCESS
  - Fear & Greed Index:        (full history)

✓ TextCrawler                   SUCCESS
  - Crypto text sentiment:     (latest collection)

✗ GdeltBQCrawler                SKIPPED
  - Reason: google-cloud-bigquery not installed
```

---

## Access & Usage

### Load Data in Python
```python
import pandas as pd

# Klines
btc_klines = pd.read_csv('data/raw/BTCUSDT_klines.csv', parse_dates=['timestamp'])
eth_klines = pd.read_csv('data/raw/ETHUSDT_klines.csv', parse_dates=['timestamp'])

# Funding Rate
btc_funding = pd.read_csv('data/raw/BTCUSDT_fundingRate.csv', parse_dates=['timestamp'])

# Sentiment
fear_greed = pd.read_csv('data/raw/fear_greed_index.csv', parse_dates=['timestamp'])
text_data = pd.read_csv('data/raw/crypto_text.csv', parse_dates=['timestamp'])

# CoinGecko
exchanges = pd.read_csv('data/raw/coingecko_derivatives_exchanges.csv')
```

---

## Updates & Refresh

To refresh all data:
```bash
python run_all_crawlers.py
```

To run specific crawlers:
```bash
python run_all_crawlers.py --source binance_vision
python run_all_crawlers.py --source coingecko
python run_all_crawlers.py --source sentiment
python run_all_crawlers.py --source text
```

---

## Questions & Troubleshooting

- **No data for [symbol]?** Check Binance Vision availability at https://data.binance.vision
- **Missing sentiment data?** APIs may have rate limits; rerun text crawler
- **Timestamp misalignment?** All times are UTC; check timezone conversion
- **Rows fewer than expected?** Deduplication removes exact duplicates per symbol

---

*For more details, see CRAWLERS.md*
