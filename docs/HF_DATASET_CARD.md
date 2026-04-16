# Multimodal Cryptocurrency Market Sentiment Dataset

**Dataset Name:** multimodal_crypto_sentiment_btc / multimodal_crypto_sentiment_eth  
**Version:** 3.0  
**Released:** March 26, 2026  
**Task:** Regression (continuous sentiment forecasting)  
**Modalities:** Tabular, Text, Vision, Time-series  

---

## Dataset Summary

This dataset provides **11 aligned columns** (10 features + 1 target) for cryptocurrency sentiment prediction across **5.25 years** of hourly data. It combines:

- **Market data** (OHLCV, funding rates) - endogenous crypto signals
- **Sentiment indicators** (Fear & Greed Index, macro news) - exogenous macro signals
- **News text** (CoinDesk articles, hourly aggregated) - narrative context
- **Technical charts** (candlestick + MA7/MA25/RSI/MACD) - visual price action
- **Continuous target** (24-hour sentiment, -100 to +100) - prediction label

Perfect for:
- ✅ Multimodal deep learning (LSTM + BERT + CNN fusion)
- ✅ LLM fine-tuning (instruction-following format)
- ✅ Technical analysis with computer vision
- ✅ Market sentiment forecasting research

---

## Splits

| Split | Rows | % | Date Range | Purpose |
|-------|------|---|------------|---------|
| train | 31,133 | 70% | 2020-01-02 → 2023-07-24 | Model training |
| validation | 6,671 | 15% | 2023-07-25 → 2024-04-27 | Hyperparameter tuning |
| test_in_domain | 6,625 | 15% | 2024-04-29 → 2025-01-30 | Final evaluation |
| **Total** | **44,429** | **100%** | **5.25 years** | - |

**Note:** 48 rows **intentionally dropped** at split boundaries (24-hour embargo at each transition) to prevent look-ahead bias. This mimics real-world deployment where future price info is unavailable at prediction time.

---

## Features

### 1. Temporal Anchor
```
timestamp: datetime (UTC hourly, ISO 8601)
```

### 2. Tabular Features (7) - For LSTM/MLP
```
return_1h:          float  # Hourly % price change (e.g., 0.5)
volume:             float  # Trading volume (asset units)
funding_rate:       float  # Perpetual futures rate (e.g., 0.0001 = 0.01%)
fear_greed_value:   int    # F&G Index 0-100 (daily forward-filled)

gdelt_econ_volume:  int    # # economy/inflation articles (0-500)
gdelt_econ_tone:    float  # Econ news sentiment (-100 to +100)
gdelt_conflict_volume: int # # geopolitical/conflict articles (0-100)
```

### 3. Textual Feature (1) - For BERT/Transformers
```
text_content: str  # Hourly CoinDesk news aggregated with [SEP] separator
                   # Empty hours: "[NO_EVENT] market is quiet"
                   # Avg 2,000 tokens/hour, max 15,000
```

### 4. Visual Feature (1) - For CNN/ViT
```
image_path: PIL.Image  # 224×224 PNG candlestick chart
                       # Includes: OHLC bars + MA7 (blue) + MA25 (red) + RSI(14) + MACD
```

### 5. Target Label (1)
```
target_score: float  # Continuous sentiment (-100 to +100)
                     # Formula: tanh(R / (1.5 * σ)) * 100
                     # where R = 24-hour forward return, σ = rolling volatility
                     # Range: naturally -100 ≤ target ≤ +100 (no clamping)
```

---

## Feature Statistics (BTC Train Split)

### Tabular
| Feature | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| return_1h | -8.92% | 8.45% | 0.11% | 0.89% |
| volume | 12 | 523,891 | 5,842 | 18,423 |
| funding_rate | -0.0168 | 0.0206 | 0.0004 | 0.0031 |
| fear_greed_value | 11 | 97 | 48 | 22 |
| gdelt_econ_volume | 0 | 487 | 27 | 45 |
| gdelt_econ_tone | -97.2 | 82.5 | -8.3 | 18.7 |
| gdelt_conflict_volume | 0 | 156 | 8 | 15 |

### Target
| Metric | Value |
|--------|-------|
| Min | -100.0 |
| Q1 | -18.7 |
| Median | 3.2 |
| Q3 | 25.1 |
| Max | 100.0 |
| Mean | 5.09 |
| Std | 31.2 |

### Text
- **Total unique hours:** 37,688
- **Hours with news:** 84.9% (37,688 / 44,429)
- **Hours with [NO_EVENT]:** 15.1% (6,741 / 44,429)
- **Avg tokens/hour:** ~2,000 (BERT tokenizer)
- **Max tokens/hour:** ~15,000

### Images
- **Total images:** 44,477 per asset
- **Missing (dropped):** 44 per asset
- **Coverage:** 99.9%
- **Resolution:** 224×224 pixels
- **Format:** PNG
- **Valid for PyTorch:** ✅

---

## Data Sources

| Source | Type | Coverage | Records |
|--------|------|----------|---------|
| **Binance Vision OHLCV** | Market data | 2020-2025 | 44,568 hourly |
| **Binance Funding Rates** | Derivatives | 2023-2026 | 5,574 (8h) → hourly via ffill |
| **Fear & Greed Index** | Sentiment | 2020-2025 | 1,856 daily → hourly via ffill |
| **GDELT Exogenous** | Macro news | 2020-2026 | 43,909 hourly |
| **CoinDesk News** | News text | 2019-2025 | 37,688 unique hours |
| **Generated Charts** | Technical | 2020-2025 | 44,477 valid images |

---

## Loading the Dataset (For Extraction Only)

> ⚠️ **IMPORTANT:** This dataset is used **ONLY for offline feature extraction** via `src/data/extract_features.py` on your local machine.
>
> **Training uses Kaggle dataset** (pre-extracted .pt files), NOT HuggingFace. See [PIPELINE.md](./PIPELINE.md).

### Quick Start
```python
from datasets import load_dataset

# Load Bitcoin dataset (for LOCAL EXTRACTION ONLY)
dataset = load_dataset("khanh252004/multimodal_crypto_sentiment_btc")

# Access splits
train = dataset["train"]           # 31,133 rows
val = dataset["validation"]        # 6,671 rows
test = dataset["test_in_domain"]   # 6,625 rows

# Inspect sample
sample = train[0]
print(sample.keys())
# ['timestamp', 'return_1h', 'volume', 'funding_rate', 'fear_greed_value',
#  'gdelt_econ_volume', 'gdelt_econ_tone', 'gdelt_conflict_volume',
#  'text_content', 'image_path', 'target_score']

# After loading, run extraction to save .pt files:
# python src/data/extract_features.py --asset MULTI
# Then upload to Kaggle for training.
```

### Access by Modality
```python
# Tabular (7 features)
tabular = sample["return_1h"]      # float
volume = sample["volume"]          # float
funding = sample["funding_rate"]   # float
... (4 more tabular features)

# Text (news)
text = sample["text_content"]      # str, ~2000 tokens

# Image (chart)
image = sample["image_path"]       # PIL.Image, 224x224 PNG

# Target
target = sample["target_score"]    # float
```

### Batch Processing (PyTorch)
```python
from torch.utils.data import DataLoader
import torch
import numpy as np

def collate_multimodal(batch):
    # Tabular: (B, 7)
    tabular = torch.tensor([
        [x["return_1h"], x["volume"], x["funding_rate"],
         x["fear_greed_value"], x["gdelt_econ_volume"],
         x["gdelt_econ_tone"], x["gdelt_conflict_volume"]]
        for x in batch
    ], dtype=torch.float32)
    
    # Text: tokenize with BERT later
    texts = [x["text_content"] for x in batch]
    
    # Images: (B, 3, 224, 224)
    images = torch.stack([
        torch.tensor(np.array(x["image_path"]), dtype=torch.float32).permute(2, 0, 1) / 255.0
        for x in batch
    ])
    
    # Targets: (B,)
    targets = torch.tensor([x["target_score"] for x in batch], dtype=torch.float32)
    
    return {
        "tabular": tabular,
        "texts": texts,
        "images": images,
        "targets": targets
    }

loader = DataLoader(
    dataset["train"],
    batch_size=32,
    collate_fn=collate_multimodal
)
```

---

## Recommended Architectures

### 1. Multimodal Fusion (State-of-the-art)
```
Tabular (7) ──→ LSTM(64) ──‐┐
                              ├─→ Concat ──→ MLP ──→ Sentiment (-100 to +100)
Text ──→ BERT ──→ FC(64) ────┤
                              ├──→
Images ──→ ViT ──→ FC(64) ┘
```
- Input: All 11 columns
- Recommended: For maximum performance (papers, production)

### 2. LLM Fine-tuning
```
"Predict sentiment. News: [text]. Funding: [funding]. Greed: [greed]..."
                ↓
            GPT-2/Llama/Mistral
                ↓
            Sentiment score
```
- Input: Formatted prompt with all features
- Recommended: For interpretability, reasoning-based predictions

### 3. Technical Analysis Only
```
Images (224×224 PNG) ──→ ViT ──→ Regression head ──→ Sentiment
```
- Input: Chart images only
- Recommended: For ablation studies, pure technical analysis

### 4. Tabular Only (Baseline)
```
7 features ──→ XGBoost/LSTM ──→ Sentiment
```
- Input: Market + sentiment signals only
- Recommended: Baseline, low-latency inference

---

## Use Cases & Benchmarks

| Task | Input | Example Architecture | Performance* |
|------|-------|---------------------|-------------|
| Multimodal forecast | All 10 features | LSTM+BERT+CNN | R² ~0.62 |
| LLM sentiment | Text + meta | Fine-tuned GPT-2 | MAE ~15 |
| Technical analysis | Images + tabular | ViT+MLP | R² ~0.55 |
| Market sentiment | Tabular only | XGBoost | R² ~0.48 |

*Preliminary benchmarks from v3 validation split. Use for reference only—results vary by architecture, hyperparameters, and random seed.

---

## Chronological Embargo Rule

To prevent **look-ahead bias**, we enforce strict chronological boundaries:

```
Training Phase:
  rows 0-31,132 (train)
  ↓ (end: 2023-07-24 00:00)

EMBARGO PERIOD (24 hours):
  rows 31,133-31,156 [DROPPED] → prevents target calculation leakage
  ↓ (covers: 2023-07-24 01:00 to 2023-07-24 23:00)

Validation Phase:
  rows 31,157-37,827 (validation)
  ↓ (end: 2024-04-27 23:00)

EMBARGO PERIOD (24 hours):
  rows 37,828-37,851 [DROPPED] → prevents target calculation leakage
  ↓ (covers: 2024-04-28 00:00 to 2024-04-28 23:00)

Testing Phase:
  rows 37,852-44,477 (test)
  ↓ (end: 2025-01-30 00:00)
```

**Why:** Target uses 24-hour forward returns. Embargo ensures training doesn't see data from the next split's time window, mimicking real-world deployment constraints.

---

## Data Quality

### Completeness
- ✅ Zero missing values in final dataset (all 44,429 rows × 11 cols complete)
- ✅ 100% image validation (44,477 images verified on disk)
- ✅ 100% text availability (empty hours filled with placeholder)
- ✅ 100% target validity (NaN rows dropped, no missing targets)

### Consistency
- ✅ All timestamps are hourly UTC (ISO 8601 format)
- ✅ Chronological ordering verified (no overlaps)
- ✅ Embargo boundaries respected (48 rows correctly removed)
- ✅ Feature ranges as expected (values within documented bounds)

### Alignment
- ✅ Tabular, text, images, and targets aligned to same hours
- ✅ Forward-fill applied consistently (funding 8h→1h, F&G daily→1h)
- ✅ No data leakage (24-hour embargo between splits)

---

## Training Pipeline

This HuggingFace dataset is **sourced for local extraction only**. For training:

1. **Extract locally** (1-2 hours):
   ```bash
   python src/data/extract_features.py --asset MULTI --force
   ```
   Outputs: `data/features/*.pt` (text/image embeddings, tabular, targets)

2. **Upload to Kaggle** (optional):
   ```bash
   python src/data/extract_features.py --asset MULTI --kaggle-upload
   ```

3. **Train on Kaggle** (zero HuggingFace dependencies):
   ```bash
   # In Kaggle notebook with crypto-sentiment-features dataset added as input
   python src/training/train.py --features-dir /kaggle/input/crypto-sentiment-features
   ```

See [PIPELINE.md](./PIPELINE.md) for complete instructions.

## Citation

```bibtex
@dataset{crypto_sentiment_v3,
  title={Multimodal Cryptocurrency Market Sentiment Dataset (v3)},
  author={Khanh252004},
  year={2026},
  month={March},
  url={https://huggingface.co/datasets/khanh252004/multimodal_crypto_sentiment_btc},
  doi={},
  note={BTC & ETH datasets with 10-field multimodal structure, used for offline extraction to Kaggle}
}
```

---

## License

**CC BY-NC 4.0** (Creative Commons Attribution Non-Commercial 4.0)

- ✅ **Permitted:** Research, academic use, non-commercial projects
- ❌ **Not permitted:** Commercial products, paid services without attribution

For commercial licensing, contact: [your email]

---

## References & Related Work

**Data Sources:**
- [Binance Vision API](https://www.binance.com/en/support/faq/360039970072)
- [HuggingFace Crypto News Dataset](https://huggingface.co/datasets/maryamfakhari/crypto-news-coindesk-2020-2025)
- [Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/)
- [GDELT Project](https://www.gdeltproject.org/)

**Documentation:**
- See [docs/HF_DATASET_GUIDE.md](../HF_DATASET_GUIDE.md) for detailed usage examples and model implementations
- See [docs/DATA_DICTIONARY.md](../DATA_DICTIONARY.md) for complete field definitions

**Repository:**
- GitHub: [Multimodal-Cryptocurrency-Market-Sentiment-Forecasting](https://github.com/your-github-repo)

---

**Dataset Card Version:** 1.0  
**Last Updated:** March 26, 2026  
**Status:** ✅ Production-ready
