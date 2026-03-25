# Data Dictionary & Summary

## Overview

This document summarizes all raw data files collected by the cryptocurrency market sentiment forecasting system. The data spans from **October 2019 to March 2026** and includes historical crypto news, market data, funding rates, liquidations, sentiment indices, macroeconomic news sentiment, and cryptocurrency discussion sentiment.

**Total Records Across All Files: ~515,000+**
- Crypto News Articles: 229,172 records (HuggingFace dataset, 2019-2025)
- Market & On-Chain Data: 105,487 records (2023-2026)
- Sentiment Data: 179,524 records (Reddit + crypto news, 2023-2026)
- Generated Chart Images: 89,048 candlestick charts with technical indicators (BTC: 44,524 + ETH: 44,524)

---

## Data Files

### 0. **huggingface_crypto_news.csv** - Cryptocurrency News Dataset (HuggingFace)
- **Purpose:** Comprehensive crypto news sentiment dataset from Coindesk
- **Rows:** 229,172 news articles
- **Date Range:** 2019-10-29 to 2025-02-01 (5.3 years)
- **Source:** HuggingFace Hub - `maryamfakhari/crypto-news-coindesk-2020-2025`
- **Granularity:** Individual articles (timestamps at publication time)

#### Fields:
| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique article identifier |
| `guid` | string | Global unique identifier |
| `published_on` | datetime | Publication timestamp (ISO 8601 with microseconds) |
| `title` | string | Article headline (cleaned) |
| `body` | string | Article body text (cleaned) |
| `url` | string | Source URL to original article |
| `imageurl` | string | URL to associated image/thumbnail |
| `tags` | string | Article tags/keywords |
| `categories` | string | Article categories (News, Analysis, etc.) |
| `source` | string | News source (Coindesk, etc.) |
| `upvotes` | int | Engagement metric (upvotes/reactions) |
| `downvotes` | int | Downvotes or negative reactions |
| `last_update` | datetime | Last modification timestamp |

#### Data Quality:
- **Total records:** 229,172 (100% unique by publication date + title)
- **Date coverage:** Full 5.3-year span 2019-10-29 to 2025-02-01 (no major gaps)
- **Missing values:** <0.1% (mostly in optional fields like tags/categories)
- **Time-aligned:** Can be merged with price data on hourly basis

#### Key Statistics:
- **Average articles per day:** ~120
- **Peak coverage:** 2021-2022 (bull market period)
- **Assets mentioned:** Primary focus on BTC/ETH market movements
- **Topics:** Market news, regulation, technology updates, price analysis

#### Use Cases:
- Long-term sentiment analysis (5+ year history)
- Training language models for crypto sentiment classification
- Identifying major news cycles and market catalysts
- Backtesting sentiment-based trading strategies

---

### 1. **reddit_posts.csv** - Reddit + Crypto News Sentiment (Combined)
- **Purpose:** Combined sentiment data from Reddit discussions and Coindesk crypto news articles
- **Rows:** 179,524 records (176,543 crypto news + 2,981 Reddit discussions)
- **Date Range:** 2023-01-01 to 2026-03-24 (3+ years)
- **Sources:** 
  - Crypto news: `maryamfakhari/crypto-news-coindesk-2020-2025` dataset from Hugging Face
  - Reddit: r/CryptoCurrency, r/Bitcoin, r/Ethereum (via public .json endpoint)
- **Granularity:** Hourly (timestamps aligned to hour boundary)

#### Fields:
| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | UTC timestamp in ISO 8601 format (YYYY-MM-DDTHH:MM:SS) |
| `subreddit` | string | Source: subreddit name (\"CryptoCurrency\", \"Bitcoin\", \"Ethereum\") or \"news\" for articles |
| `title` | string | Article/post title (cleaned, URLs removed) |
| `text` | string | Article body or post selftext (cleaned) |
| `combined_text` | string | Title + text concatenated (used for sentiment encoding) |
| `url` | string | Source URL to original content |
| `score` | int | Reddit upvotes or news engagement (upvotes for news) |
| `num_comments` | int | Reddit comment count or downvotes (for news sources) |
| `author` | string | Reddit username or news source name |
| `assets` | string | Detected crypto assets: \"BTC\", \"ETH\", \"BTC,ETH\", or \"OTHER\" |
| `text_hash` | string | SHA256 hash of combined_text (for deduplication, 100% unique) |
| `source` | string | Data source: \"reddit\" or \"crypto_news_sentiment\" |

#### Asset Detection (Regex-based):
- **BTC:** Matches patterns `\bbtc\b` or `\bbitcoin\b` → **72.1%** of records mention BTC
- **ETH:** Matches patterns `\beth\b` or `\bethereum\b` → **35.5%** of records mention ETH  
- **OTHER:** No BTC/ETH mentions → **7.4%** of records

#### Data Quality Metrics:
- **Total records:** 179,524 (100% unique text hashes)
- **Duplicates removed:** 281 (detected and removed during merge)
- **Date coverage:** Full 3-year span 2023-01-01 to 2026-03-24 (no gaps > 5 months)
- **Missing values:** <1% nulls (mostly in non-critical fields like text/body)
- **Source split:** 98.3% crypto news sentiment, 1.7% Reddit discussions

#### Sample Records:
```
Crypto News Record:
timestamp,subreddit,title,text,combined_text,url,score,num_comments,author,assets,text_hash,source
2023-06-15T14:00:00,news,Bitcoin ETF approval signals institutional adoption,"Bitcoin spot ETF application...",<combined>,https://coindesk.com/...,1200,0,coindesk,BTC,abc123def456...,crypto_news_sentiment

Reddit Record:
2026-03-24T01:00:00,CryptoCurrency,"Daily Crypto Discussion - March 24 2026","**Welcome to daily...",<combined>,https://reddit.com/r/Crypto.../...,13,51,AutoModerator,OTHER,881171a5ef5a...,reddit
```

#### Preprocessing Applied:
- Text cleaning: URL removal, HTML entity decoding, emoji removal, whitespace normalization
- Timestamp alignment: All timestamps floored to hour boundary
- Deduplication: Exact match duplicates removed by text_hash (kept first occurrence)
- Asset detection: Case-insensitive regex pattern matching

---

### 2. **BTCUSDT_klines.csv** - Bitcoin OHLCV Candles
- **Purpose:** Bitcoin hourly price data (OHLC) and trading volume
- **Rows:** 44,568
- **Date Range:** 2020-01-01 to 2025-01-31 (5 years)
- **Source:** Binance Vision API
- **Granularity:** Hourly

#### Fields:
| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | UTC timestamp in format `YYYY-MM-DD HH:MM:SS` |
| `open` | float | Opening price (USDT) |
| `high` | float | Highest price in the period (USDT) |
| `low` | float | Lowest price in the period (USDT) |
| `close` | float | Closing price (USDT) |
| `volume` | float | Trading volume (BTC) |

#### Sample Data:
```
timestamp,open,high,low,close,volume
2023-01-01 00:00:00,16537.5,16540.9,16504.0,16527.0,5381.399
2023-01-01 01:00:00,16527.1,16554.3,16524.1,16550.4,3210.826
```

---

### 3. **BTCUSDT_fundingRate.csv** - Bitcoin Perpetual Funding Rates
- **Purpose:** Bitcoin perpetual futures funding rates (8-hour intervals)
- **Rows:** 3,465
- **Date Range:** 2023-01-01 to 2026-02-28
- **Source:** Binance Vision API
- **Granularity:** Every 8 hours

#### Fields:
| Field | Type | Description |
|-------|------|-------------|
| `calc_time` | int | Unix timestamp in milliseconds |
| `funding_interval_hours` | int | Funding interval in hours (always 8) |
| `last_funding_rate` | float | Funding rate value (decimal, e.g., 0.0001 = 0.01%) |

#### Interpretation:
- **Positive rate:** Long positions pay shorts (bullish sentiment)
- **Negative rate:** Shorts pay longs (bearish sentiment)
- Rates capture market leverage and sentiment

#### Sample Data:
```
calc_time,funding_interval_hours,last_funding_rate
1672531200000,8,0.0001
1672560000008,8,0.0001
```

---

### 4. **ETHUSDT_klines.csv** - Ethereum OHLCV Candles
- **Purpose:** Ethereum hourly price data (OHLC) and trading volume
- **Rows:** 44,568
- **Date Range:** 2020-01-01 to 2025-01-31 (5 years)
- **Source:** Binance Vision API
- **Granularity:** Hourly

#### Fields: Same as BTCUSDT_klines.csv
| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | UTC timestamp |
| `open` | float | Opening price (USDT) |
| `high` | float | Highest price (USDT) |
| `low` | float | Lowest price (USDT) |
| `close` | float | Closing price (USDT) |
| `volume` | float | Trading volume (ETH) |

---

### 5. **ETHUSDT_fundingRate.csv** - Ethereum Perpetual Funding Rates
- **Purpose:** Ethereum perpetual futures funding rates
- **Rows:** 3,465
- **Date Range:** 2023-01-01 to 2026-02-28
- **Source:** Binance Vision API
- **Granularity:** Every 8 hours

#### Fields: Same as BTCUSDT_fundingRate.csv
| Field | Type | Description |
|-------|------|-------------|
| `calc_time` | int | Unix timestamp in milliseconds |
| `funding_interval_hours` | int | Funding interval (8 hours) |
| `last_funding_rate` | float | Funding rate (decimal percentage) |

---

### 6. **coinalyze_open_interest.csv** - Perpetual Open Interest
- **Purpose:** Track aggregate open interest on Bybit PERP (Bybit perpetuals exchange)
- **Rows:** 3,034
- **Date Range:** 2026-01-19 to 2026-03-23 (recent data, ~2 months)
- **Source:** Coinalyze API
- **Granularity:** Hourly
- **Assets:** BTC, ETH

#### Fields:
| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | UTC timestamp in ISO format `YYYY-MM-DDTHH:MM:SS` |
| `asset` | string | Crypto symbol (BTC or ETH) |
| `open` | float | Opening open interest (in USD) |
| `high` | float | Highest open interest (in USD) |
| `low` | float | Lowest open interest (in USD) |
| `close` | float | Closing open interest (in USD) |

#### Purpose:
Open interest represents the total value of outstanding perpetual contracts. Rising OI signals increased leverage and conviction in price moves; falling OI suggests unwinding of positions.

#### Sample Data:
```
timestamp,asset,open,high,low,close
2026-01-19T11:00:00,BTC,8827827806.378601,8834181033.232399,8827228099.167301,8830276929.0418
```

---

### 7. **coinalyze_liquidations.csv** - Perpetual Liquidations
- **Purpose:** Track hourly liquidations on Bybit PERP (long vs. short)
- **Rows:** 4,172
- **Date Range:** 2025-12-25 to 2026-03-23 (~3 months)
- **Source:** Coinalyze API
- **Granularity:** Hourly
- **Assets:** BTC, ETH

#### Fields:
| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | UTC timestamp in ISO format |
| `asset` | string | Crypto symbol (BTC or ETH) |
| `long_liquidations` | float | BTC/ETH liquidated from long positions (in units) |
| `short_liquidations` | float | BTC/ETH liquidated from short positions (in units) |

#### Purpose:
Liquidations indicate forced closures of leveraged positions. High liquidations suggest market participants were over-leveraged; pattern shifts indicate momentum or trend reversals.

#### Interpretation:
- **High long liquidations:** Suggests short squeeze or forced closure of bull positions
- **High short liquidations:** Suggests long squeeze or forced closure of bear positions

#### Sample Data:
```
timestamp,asset,long_liquidations,short_liquidations
2025-12-25T07:00:00,BTC,0.0,0.751
2025-12-25T08:00:00,BTC,0.0,0.818
```

---

### 8. **coinalyze_long_short_ratio.csv** - Perpetual Long/Short Ratio
- **Purpose:** Track trader sentiment via open position ratios on Bybit PERP
- **Rows:** 4,032
- **Date Range:** 2025-12-29 to 2026-03-23 (~3 months)
- **Source:** Coinalyze API
- **Granularity:** Hourly
- **Assets:** BTC, ETH

#### Fields:
| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | UTC timestamp in ISO format |
| `asset` | string | Crypto symbol (BTC or ETH) |
| `ratio` | float | Long Open Interest ÷ Short Open Interest (e.g., 2.72 = 2.72:1) |
| `long_percentage` | float | % of total OI in long positions (0-100) |
| `short_percentage` | float | % of total OI in short positions (0-100) |

#### Purpose:
Long/short ratio is a sentiment indicator. High ratios indicate bullish trader sentiment; low ratios indicate bearish sentiment. Extreme ratios often precede reversals.

#### Interpretation:
- **Ratio > 3.0:** Extremely bullish (potential reversal risk)
- **Ratio 1.5-3.0:** Moderately bullish
- **Ratio 1.0-1.5:** Neutral to slightly bullish
- **Ratio < 1.0:** More shorts than longs (bearish)

#### Sample Data:
```
timestamp,asset,ratio,long_percentage,short_percentage
2025-12-29T16:00:00,BTC,2.7216,73.13,26.87
2025-12-29T16:00:00,ETH,2.686,72.87,27.13
```

---

### 9. **fear_greed_index.csv** - Cryptocurrency Fear & Greed Index
- **Purpose:** Daily cryptocurrency market sentiment index
- **Rows:** 2,970
- **Date Range:** 2018-02-01 to 2026-03-24 (8+ years)
- **Source:** Alternative.me API
- **Granularity:** Daily
- **Update Frequency:** Daily (UTC)

#### Fields:
| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | int | Unix timestamp (seconds) |
| `datetime` | string | ISO 8601 datetime format with timezone |
| `value` | int | Fear & Greed Index value (0-100) |
| `value_classification` | string | Text classification: "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed" |
| `time_until_update` | float | Hours until next update (typically null) |
| `source` | string | Data source (alternative.me) |

#### Index Levels:
| Value Range | Classification |
|-------------|-----------------|
| 0-25 | **Extreme Fear** - High capitulation |
| 25-46 | **Fear** - Negative sentiment |
| 46-54 | **Neutral** - Balanced sentiment |
| 54-75 | **Greed** - Positive sentiment |
| 75-100 | **Extreme Greed** - Euphoria/overbought |

#### Methodology:
The Fear & Greed Index combines multiple signals:
- Volatility (25% weight)
- Market momentum and volume (25%)
- Social media intensity (15%)
- Market dominance (10%)
- Trending searches (10%)
- Surveys (15%)

#### Sample Data:
```
timestamp,datetime,value,value_classification,time_until_update,source
1517443200,2018-02-01T00:00:00+00:00,30,Fear,,alternative.me
1517529600,2018-02-02T00:00:00+00:00,15,Extreme Fear,,alternative.me
```

---

### 9. **gdelt_macro.csv.csv** - Macroeconomic News Sentiment
- **Purpose:** Sentiment analysis of global macroeconomic news (GDELT)
- **Rows:** 27,829
- **Date Range:** 2023-01-01 to 2026-03-24 (3+ years)
- **Source:** GDELT v2.1 BigQuery Dataset
- **Granularity:** Hourly (aggregated)
- **Filter:** Macro-economic themes only (ECON_INFLATION, US_FEDERAL_RESERVE, CRISIS, ARMEDCONFLICT, ECON_UNEMPLOYMENT)

#### Fields:
| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | UTC timestamp in ISO 8601 format |
| `news_volume` | int | Number of news articles processed in this hour with macro themes |
| `avg_sentiment_tone` | float | Average sentiment tone (-100 to +100); negative = negative sentiment, positive = positive |

#### Sentiment Tone Scale:
- **-100 to -50:** Very negative news (crises, conflicts, unemployment)
- **-50 to -10:** Negative news (economic concerns)
- **-10 to +10:** Neutral / mixed news
- **+10 to +50:** Positive news (growth, stability)
- **+50 to +100:** Very positive news (booming, prosperity)

#### Purpose:
Macroeconomic sentiment provides context for cryptocurrency market movements. Negative macro news often correlates with crypto sell-offs; positive macro sentiment supports risk-on sentiment.

#### Themes Tracked:
- `ECON_INFLATION` - Inflation related news
- `US_FEDERAL_RESERVE` - Federal Reserve policy
- `CRISIS` - Global economic crises
- `ARMEDCONFLICT` - Geopolitical conflicts
- `ECON_UNEMPLOYMENT` - Employment data/trends

#### Sample Data:
```
timestamp,news_volume,avg_sentiment_tone
2023-01-01T00:00:00+00:00,45,-12.3
2023-01-01T01:00:00+00:00,32,5.1
```

---

## Generated Assets

### Candlestick Chart Images

**Location:** `data/processed/images/{btc,eth}/`

#### Specifications:
- **BTC Charts:** 44,524 images
- **ETH Charts:** 44,524 images
- **Total:** 89,048 chart images
- **Image Size:** 224×224 pixels (standard for deep learning)
- **Format:** PNG
- **Date Range:** 2020-01-02 to 2025-01-31

#### Chart Contents:
1. **Candlestick Pattern:** OHLC bars with wicks
2. **Technical Indicators:**
   - **MA7:** 7-hour moving average (blue line)
   - **MA25:** 25-hour moving average (red line)
   - **RSI(14):** 14-period Relative Strength Index (oscillator at bottom)
   - **MACD:** Moving Average Convergence Divergence with signal line

#### Use Cases:
- Training computer vision models (ResNet, ViT, CNN)
- Multimodal prediction model input (combined with text sentiment)
- Pattern recognition and technical analysis
- Visualization of price action with indicators

#### Generation Stats:
- **Processing time:** ~25 minutes (12.25 min BTC + 12.19 min ETH)
- **Parallel workers:** 21
- **Generation rate:** ~60 images/second
- **Preprocessing:** Automatic indicator calculation on full dataset

---

### Coverage Summary:

| Data Source | Asset(s) | Coverage | Records | Reliability |
|------------|----------|----------|---------|-------------|
| **HuggingFace Crypto News** | **BTC, ETH** | **2019-2025** | **229,172** | **⭐⭐⭐⭐⭐ - Comprehensive historical dataset** |
| Binance Vision (klines) | BTC, ETH | 2020-2025 | 89,136 | ⭐⭐⭐⭐⭐ - Official exchange data |
| Binance Vision (funding) | BTC, ETH | 2023-2026 | 11,148 | ⭐⭐⭐⭐⭐ - Official exchange data |
| Coinalyze (OI, liquidations, L/S) | BTC, ETH | Recent (2-3 mo) | 11,238 | ⭐⭐⭐⭐ - Aggregator, good coverage |
| Fear & Greed Index | Crypto market | 2018-2026 | 2,970 | ⭐⭐⭐⭐⭐ - Well-established index |
| GDELT Macro Sentiment | Global macro | 2023-2026 | 27,829 | ⭐⭐⭐⭐ - News-based, real-time |
| Reddit Discussion | BTC, ETH | 2025-2026 | 2,981 | ⭐⭐⭐⭐ - Real-time Reddit data |
| **Generated Chart Images** | **BTC, ETH** | **2020-2025** | **89,048** | **✓ 224×224 px with technical indicators** |

---

## Time Alignment & Joins

### Recommended Join Keys:

1. **Price + Funding Rates:** 
   - Join on `timestamp` (both hourly)
   - Files: `BTCUSDT_klines.csv` ↔ `BTCUSDT_fundingRate.csv`

2. **Price + Sentiment:**
   - Join on hourly `timestamp`
   - Files: `BTCUSDT_klines.csv` ↔ `fear_greed_index.csv` (daily, up-sample or group)

3. **On-Chain Metrics (OI, Liquidations, L/S):**
   - All at hourly granularity
   - Join on `timestamp` and `asset` (BTC or ETH)
   - Files: `coinalyze_*.csv`

4. **Macro Sentiment:**
   - Hourly timestamp
   - Join with price data on `timestamp`
   - File: `gdelt_macro.csv.csv`

---

## Suggested Use Cases

### 1. **Price Prediction Model**
- Input: OHLCV (klines) + funding rates + Fear/Greed + macro sentiment
- Target: Next 1-24 hour price movement
- Files: BTC/ETH klines, funding rates, fear_greed, GDELT

### 2. **Liquidation Cascade Detection**
- Input: Liquidation volume + OI changes + L/S ratio extreme values
- Target: Detect potential price reversals
- Files: coinalyze_liquidations, coinalyze_open_interest, coinalyze_long_short_ratio

### 3. **Sentiment Features**
- Input: Fear/Greed classification + macro sentiment tone + news volume
- Use as: Feature for machine learning models
- Files: fear_greed_index, gdelt_macro

### 4. **Risk Management**
- Monitor funding rates for leverage extremes
- Track liquidation volumes for capitulation signals
- Check macro sentiment for systemic risk
- Files: Funding rates + liquidations + GDELT

---

## Data Collection Methods

| Crawler | API | Update Frequency | Notes |
|---------|-----|------------------|-------|
| `binance_vision_crawler.py` | Binance Vision (Historical) | One-time download | 37 months of 1-hour candles |
| `coinalyze_crawler.py` | Coinalyze REST API | Configurable (months_back) | Recent data, rolling window ~3 months |
| `sentiment_crawler.py` | Alternative.me (Free API) | All-time history | 8+ years of daily fear/greed |
| `gdelt_bq_crawler.py` | Google BigQuery GDELT | Recent (query-based) | Hourly macro news sentiment |

---

## Next Steps for Analysis

1. **Align time series:** Ensure all timestamps are UTC and hourly/daily granularity matches
2. **Handle missing data:** Use forward-fill or interpolation for hourly gaps
3. **Feature engineering:** Create lagged features, rolling averages, momentum indicators
4. **Correlation analysis:** Check which sentiment signals lead price movements
5. **Backtesting:** Validate prediction signals against actual price history
6. **Model training:** Combine multimodal data (price + sentiment + macro) for forecasting

---

**Generated:** 2026-03-25  
**Total Data Points:** 515,000+ records (229,172 HF news + 105,487 market/on-chain + 179,524 sentiment)  
**Total Chart Images:** 89,048 candlestick charts (BTC: 44,524 + ETH: 44,524) with technical indicators  
**Historical Data Time Span:** 2019-10-29 to 2025-02-01 (HuggingFace news articles)  
**Market Data Time Span:** 2020-01-01 to 2025-01-31 (Price + funding data)  
**Sentiment Data Time Span:** 2023-01-01 to 2026-03-24 (combined Reddit + macro analysis)  
**Fear & Greed Index Span:** 2018-02-01 to 2026-03-24 (8+ years)  

#### Recent Updates:
- **2026-03-25:** Complete data pipeline overhaul
  - Added HuggingFace `crypto-news-coindesk-2020-2025` dataset: 229,172 articles (2019-2025)
  - Extended crawler time ranges to match HuggingFace historical data
  - Generated 89,048 chart images (224×224 px) with MA7, MA25, RSI(14), MACD indicators
  - Updated BTC/ETH klines to 5-year span: 44,568 records each (2020-1-1 to 2025-01-31)
  - Sentiment crawler: 1,922 Fear & Greed records (filtered to 2019-2025 range)
