# Crawler Reference

This document summarizes all crawlers in this repository, including their data sources, output files, credentials, and execution patterns.

## Orchestrator

Main entry point: run_all_crawlers.py

- Runs one crawler:
  - python run_all_crawlers.py --source text
- Runs all registered crawlers:
  - python run_all_crawlers.py
- Lists available crawlers:
  - python run_all_crawlers.py --list

Current registered crawlers:
- binance
- coingecko
- sentiment
- text

## 1) BinanceCrawler

Module: src/crawlers/binance_crawler.py

Purpose:
- Collect market structure and derivatives context from Binance public APIs.

Data sources:
- Binance Spot OHLCV
- Binance Futures funding rate history
- Binance Futures open interest
- Binance Futures recent trades used as liquidation proxy

Key assets:
- BTC/USDT
- ETH/USDT

Typical outputs in data/raw:
- btcusdt_ohlcv_1h.csv
- btcusdt_ohlcv_4h.csv
- btcusdt_funding_rate.csv
- btcusdt_open_interest.csv
- btcusdt_liquidations.csv
- ethusdt_ohlcv_1h.csv
- ethusdt_ohlcv_4h.csv
- ethusdt_funding_rate.csv
- ethusdt_open_interest.csv
- ethusdt_liquidations.csv

Notes:
- Uses ccxt with built-in rate limiting plus extra sleep for safety.
- Supports deep historical pagination from a start date.

## 2) CoinGeckoCrawler

Module: src/crawlers/coingecko_crawler.py

Purpose:
- Collect derivatives exchange and ticker metadata from CoinGecko.

Data sources:
- CoinGecko derivatives exchanges endpoint
- CoinGecko derivatives tickers endpoint

Typical outputs in data/raw:
- coingecko_derivatives_exchanges.csv
- bitcoin_derivatives_tickers.csv
- ethereum_derivatives_tickers.csv

Credentials:
- Optional API key placeholder exists in .env.example as COINGECKO_API_KEY.
- Current implementation can accept api_key on class initialization.

Notes:
- Designed to run in public mode if no key is provided.

## 3) SentimentCrawler

Module: src/crawlers/sentiment_crawler.py

Purpose:
- Collect sentiment baselines from Fear and Greed Index.

Data source:
- Alternative.me Fear and Greed API

Typical outputs in data/raw:
- fear_greed_index.csv
- sentiment_current.csv

Notes:
- Pulls full history and latest snapshot.
- No API credentials required.

## 4) TextCrawler

Module: src/crawlers/text_crawler.py

Purpose:
- Collect text-based market narrative signals for BTC and ETH from news and social channels.

Data sources:
- CryptoPanic posts API (news)
- Reddit public JSON endpoint from:
  - r/CryptoCurrency
  - r/Bitcoin
  - r/Ethereum
- StockTwits symbol streams:
  - BTC.X
  - ETH.X

Keyword filter:
- BTC
- Bitcoin
- ETH
- Ethereum
- Bullish
- Bearish

Output in data/raw:
- crypto_text.csv

Output fields:
- timestamp
- source
- asset
- text
- user_sentiment
- text_hash

Design details:
- Timestamp is aligned to Binance-style hourly UTC bucket in milliseconds.
- Text cleaning removes URLs, emojis/non-ASCII, and special characters.
- Deduplication uses SHA-256 text_hash to avoid duplicate rows across runs.
- Historical backfill for CryptoPanic uses next-page pagination until limit.
- Polite rate limiting is applied for CryptoPanic, Reddit, and StockTwits requests.

Required credentials for full collection:
- CRYPTOPANIC_API_KEY
- REDDIT_USER_AGENT (required; use a non-default browser-like User-Agent)

StockTwits credentials:
- No API key required for the public symbol endpoint.

Behavior when missing setup:
- Missing CryptoPanic token: news collection is skipped with warning.
- Missing REDDIT_USER_AGENT: Reddit collection is skipped with warning.

## Environment and Dependencies

Credential template:
- .env.example

Dependencies:
- requirements.txt

Install dependencies:
- pip install -r requirements.txt

If you use a virtual environment on Windows:
- .venv/Scripts/Activate.ps1

## Suggested Run Order

For full raw-data refresh:
1. python run_all_crawlers.py --source binance
2. python run_all_crawlers.py --source coingecko
3. python run_all_crawlers.py --source sentiment
4. python run_all_crawlers.py --source text

Or run all together:
- python run_all_crawlers.py
