# Data Dictionary & Summary

## Overview

This document describes all raw data files used by the cryptocurrency market sentiment
forecasting system. The data spans **January 2020 to January 2025** across three modalities:
market data, news text, and candlestick chart images.

**Data used in the thesis (v5):**
- Crypto News Articles: 229,172 records (CoinDesk via HuggingFace, 2019–2025)
- Market & Funding Rate Data: ~89,136 records (Binance OHLCV + funding rates, 2020–2025)
- GDELT Macro Indicators: 43,909 hourly records (2020–2025)
- Generated Chart Images: 89,048 candlestick charts with technical indicators (BTC: 44,524 + ETH: 44,524)

---

## Data Files

### 0. **huggingface_crypto_news.csv** — Cryptocurrency News Dataset (CoinDesk)
- **Purpose:** Comprehensive crypto news dataset from CoinDesk, used as the text modality
- **Rows:** 229,172 news articles
- **Date Range:** 2019-10-29 to 2025-02-01 (5.3 years)
- **Source:** HuggingFace Hub — `khanh252004/multimodal_crypto_sentiment_btc` / `_eth`
- **Granularity:** Individual articles (timestamps at publication time)

#### Fields:
| Field | Type | Description |
|-------|------|-------------|
| `published_on` | datetime | Publication timestamp (UTC) |
| `title` | string | Article headline |
| `body` | string | Article body text |
| `combined_text` | string | Title + body concatenated (used for FinBERT encoding) |

#### Key Statistics:
- **Average articles per day:** ~120
- **Hours with at least one article:** ~84.9% of the hourly index
- **Missing hours:** Filled with `"[NO_EVENT] market is quiet"` placeholder during alignment

---

### 1. **BTCUSDT_klines.csv** — Bitcoin OHLCV Candles
- **Purpose:** Bitcoin hourly price data (OHLCV) — master hourly index for BTC
- **Rows:** 44,568
- **Date Range:** 2020-01-01 to 2025-01-31 (5 years)
- **Source:** Binance Vision API
- **Granularity:** Hourly

#### Fields:
| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | UTC timestamp (`YYYY-MM-DD HH:MM:SS`) |
| `open` | float | Opening price (USDT) |
| `high` | float | Highest price in the period (USDT) |
| `low` | float | Lowest price in the period (USDT) |
| `close` | float | Closing price (USDT) |
| `volume` | float | Trading volume (BTC) |

---

### 2. **BTCUSDT_fundingRate.csv** — Bitcoin Perpetual Funding Rates
- **Purpose:** Perpetual futures funding rates — key sentiment/positioning signal
- **Rows:** 5,574
- **Date Range:** 2020-01-01 to 2025-01-31
- **Source:** Binance Vision API
- **Granularity:** Every 8 hours (forward-filled to hourly in the alignment pipeline)

#### Fields:
| Field | Type | Description |
|-------|------|-------------|
| `calc_time` | int | Unix timestamp in milliseconds |
| `funding_interval_hours` | int | Funding interval in hours (always 8) |
| `last_funding_rate` | float | Funding rate (decimal, e.g., 0.0001 = 0.01%) |

#### Interpretation:
- **Positive rate:** Long positions pay shorts (bullish sentiment / excess leverage)
- **Negative rate:** Shorts pay longs (bearish sentiment)

---

### 3. **ETHUSDT_klines.csv** — Ethereum OHLCV Candles
- **Purpose:** Ethereum hourly price data — master hourly index for ETH
- **Rows:** 44,568
- **Date Range:** 2020-01-01 to 2025-01-31 (5 years)
- **Source:** Binance Vision API
- **Granularity:** Hourly

#### Fields: Same as BTCUSDT_klines.csv

---

### 4. **ETHUSDT_fundingRate.csv** — Ethereum Perpetual Funding Rates
- **Purpose:** ETH perpetual futures funding rates
- **Rows:** 5,574
- **Date Range:** 2020-01-01 to 2025-01-31
- **Source:** Binance Vision API
- **Granularity:** Every 8 hours (forward-filled to hourly)

#### Fields: Same as BTCUSDT_fundingRate.csv

---

### 5. **gdelt_exogenous_data.csv** — GDELT Exogenous Macro Indicators
- **Purpose:** Global macroeconomic and geopolitical news sentiment (exogenous signals)
- **Rows:** 43,909
- **Date Range:** 2020-01-01 to 2025-01-31 (5 years)
- **Source:** GDELT v2.1 BigQuery Dataset (pre-computed CSV)
- **Granularity:** Hourly (aggregated)
- **Themes:** Economy/inflation (`ECON_INFLATION`) + Conflict/politics (`ARMEDCONFLICT`)

#### Fields:
| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | UTC timestamp (ISO 8601) |
| `gdelt_econ_volume` | int | # articles on economy/inflation themes |
| `gdelt_econ_tone` | float | Average sentiment tone of economic articles (−100 to +100) |
| `gdelt_conflict_volume` | int | # articles on conflict/politics themes |

#### Tone Scale:
- **−100 to −10:** Negative news (crises, conflicts, policy concerns)
- **−10 to +10:** Neutral / mixed
- **+10 to +100:** Positive news (growth, stability)

---

## v5 Implementation: 13-Field Multimodal Structure

### Final Dataset Fields (After Alignment & Processing)

The **data_aligner.py** pipeline produces a **13-field multimodal dataset** (10 features + 3 targets):

#### 1. Meta Group (1 field)
| Field | Type | Source | Purpose |
|-------|------|--------|---------|
| `timestamp` | datetime | All sources (hourly index) | Time identifier (UTC) |

#### 2. Tabular Data Group (7 fields)
| Field | Type | Source | Purpose |
|-------|------|--------|---------|
| `return_1h` | float (%) | OHLCV | Hourly % price change |
| `volume` | float | OHLCV | Trading activity (asset units) |
| `funding_rate` | float | Binance funding rates | Derivatives sentiment (8-hour, forward-filled) |
| `gdelt_econ_volume` | int | GDELT exogenous | # macro economy articles |
| `gdelt_econ_tone` | float (−100 to +100) | GDELT exogenous | Sentiment tone of economic news |
| `gdelt_conflict_volume` | int | GDELT exogenous | # geopolitical/conflict articles |
| `is_post_ETF` | int (0 or 1) | Calendar date | Regime flag: 1 if timestamp ≥ 2024-01-01 |

#### 3. Textual Data Group (1 field)
| Field | Type | Source | Purpose |
|-------|------|--------|---------|
| `text_content` | string | CoinDesk news (hourly aggregated) | Articles joined with `[SEP]`; `"[NO_EVENT] market is quiet"` for empty hours |

#### 4. Visual Data Group (1 field)
| Field | Type | Source | Purpose |
|-------|------|--------|---------|
| `image_path` | image (224×224 PNG) | chart_generator.py | Candlestick + MA7/MA25/RSI(14)/MACD |

#### 5. Target Labels (3 fields)
| Field | Type | Description |
|-------|------|-------------|
| `y_baseline` | float | Raw funding rate at t+1 (shifted to t+8 in `dataset.py`; scaled ×1000 during training) |
| `y_heuristic` | float | Weighted Z-score composite at t+1: 0.4·Z(Δfunding) + 0.3·Z(return) + 0.2·Z(Δtone) − 0.1·Z(Δconflict) |
| `y_vol_adj_return` | float | Volatility-adjusted log return at t+1: log\_return(t+1) / (vol\_168h(t) + 1e-6) |

**Total: 1 meta + 7 tabular + 1 text + 1 visual + 3 targets = 13 columns per row**

### Data Alignment Process (Phases 1–5)

| Phase | Operation | Input | Output |
|-------|-----------|-------|--------|
| **1** | Load 4 sources | CSV files | Hourly index alignment |
| **2** | Map & validate images | Image directory | Drop rows with missing chart images |
| **3** | Feature engineering | Close, funding, GDELT | 7 tabular features (return_1h, is_post_ETF, etc.) |
| **4** | Target engineering | Returns, funding, GDELT | 3 target columns |
| **5** | Final assembly | All columns | 13-column DataFrame; push to HuggingFace Hub |

### Chronological Split

```
Timeline:
|────────────── Train + Val (85%) ──────────────|──── Test (15%) ────|
2020-01-02                                   2024-04-29         2025-01-31
```

- **Walk-forward folds:** Applied in-memory via `create_walk_forward_dataloaders` on the 85% window
- **Test set:** Strictly held out; 8-step buffer prevents overlap with training windows

---

## Generated Assets

### Candlestick Chart Images

**Location:** `data/processed/images/{btc,eth}/`

| Property | Value |
|----------|-------|
| BTC charts | 44,524 images |
| ETH charts | 44,524 images |
| Total | 89,048 images |
| Resolution | 224×224 pixels (standard ViT input) |
| Format | PNG |
| Date range | 2020-01-02 to 2025-01-31 |
| Lookback window | 24 candles per chart |

**Chart overlays:**
1. Candlestick (OHLC bars with wicks) on dark background
2. MA7 — 7-period moving average
3. MA25 — 25-period moving average
4. RSI(14) — oscillator panel
5. MACD — histogram panel

---

## Coverage Summary

| Data Source | Asset(s) | Coverage | Records | Notes |
|------------|----------|----------|---------|-------|
| CoinDesk News (HuggingFace) | BTC, ETH | 2019–2025 | 229,172 articles | Text modality |
| Binance Vision (klines) | BTC, ETH | 2020–2025 | 89,136 hourly rows | OHLCV |
| Binance Vision (funding) | BTC, ETH | 2020–2025 | 11,148 (8h) → hourly | Funding rate |
| GDELT Exogenous | Global macro | 2020–2025 | 43,909 hourly rows | Economy + conflict |
| Generated Chart Images | BTC, ETH | 2020–2025 | 89,048 images | 224×224 candlestick |

---

## Data Collection Methods

| Script | API / Source | Purpose |
|--------|-------------|---------|
| `src/crawlers/binance_vision_crawler.py` | Binance Vision (historical) | OHLCV klines + funding rates |
| `src/crawlers/huggingface_crawler.py` | HuggingFace Hub | CoinDesk news articles |
| *(GDELT CSV)* | Pre-computed from GDELT BigQuery | `gdelt_exogenous_data.csv` loaded directly |

---

**Generated:** 2026-05-24  
**Market Data Time Span:** 2020-01-01 to 2025-01-31  
**News Data Time Span:** 2019-10-29 to 2025-02-01  
**Total Chart Images:** 89,048 (BTC: 44,524 + ETH: 44,524)  
