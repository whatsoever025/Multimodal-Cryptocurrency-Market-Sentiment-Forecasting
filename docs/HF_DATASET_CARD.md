# Multimodal Cryptocurrency Market Sentiment Dataset

**Dataset Name:** multimodal_crypto_sentiment_btc / multimodal_crypto_sentiment_eth  
**Version:** 5.0  
**Released:** May 24, 2026  
**Task:** Multi-target Regression (continuous sentiment forecasting)  
**Modalities:** Tabular, Text, Vision, Time-series  

---

## Dataset Summary

This dataset provides **13 aligned columns** (10 features + 3 targets) for cryptocurrency sentiment prediction across **5.25 years** of hourly data. It combines:

- **Market data** (OHLCV, funding rates) - endogenous crypto signals
- **Regime data** (is_post_ETF flag) - institutional regime gating
- **Sentiment indicators** (macro news sentiment volume and tone) - exogenous macro signals
- **News text** (CoinDesk articles, hourly aggregated) - narrative context
- **Technical charts** (candlestick + MA7/MA25/RSI/MACD) - visual price action
- **Three continuous targets** (baseline, composite heuristic, and volatility-adjusted return at t+1) - prediction labels

Perfect for:
- ✅ Multimodal deep learning (LSTM + FinBERT + ViT fusion with learnable tokens)
- ✅ LLM fine-tuning (instruction-following format)
- ✅ Technical analysis with computer vision
- ✅ Market sentiment forecasting research

---

## Splits

The dataset is uploaded to the Hugging Face Hub divided chronologically into two splits:

| Split | Rows (approx.) | % | Date Range | Purpose |
|-------|------|---|------------|---------|
| train | ~37,764 | 85% | 2020-01-02 → 2024-04-28 | Walk-forward cross-validation |
| test | ~6,665 | 15% | 2024-04-29 → 2025-01-31 | Final evaluation (holdout) |
| **Total** | **~44,429** | **100%** | **5.25 years** | - |

**Note:** During training, `create_walk_forward_dataloaders` loads this full chronological sequence, automatically fitting StandardScaler (for tabular features) and RobustScaler (for targets) per-fold and per-asset on the training windows to prevent temporal and cross-asset data leakage.

---

## Features

### 1. Temporal Anchor
```
timestamp: datetime (UTC hourly, ISO 8601)
```

### 2. Tabular Features (7) - For LSTM/MLP
```
return_1h:             float  # Hourly % price change (e.g., 0.5)
volume:                float  # Trading volume (asset units)
funding_rate:          float  # Perpetual futures rate (e.g., 0.0001 = 0.01%)
gdelt_econ_volume:     int    # # economy/inflation articles (0-500)
gdelt_econ_tone:       float  # Econ news sentiment (-100 to +100)
gdelt_conflict_volume: int    # # geopolitical/conflict articles (0-100)
is_post_ETF:           int    # Binary flag: 1 if >= 2024-01-01 (post-ETF regime)
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

### 5. Target Labels (3)
```
y_baseline: float        # Raw funding rate at t+1 (shifted to t+8 in dataset.py, scaled * 1000.0)
y_heuristic: float       # Weighted Z-score composite of target changes at t+1 (clipped to [-5.0, 5.0])
y_vol_adj_return: float  # Volatility-Adjusted Log Return at t+1 (unscaled/raw)
```

---

## Feature Statistics (BTC Train Split)

### Tabular
| Feature | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| return_1h | -8.92% | 8.45% | 0.11% | 0.89% |
| volume | 12 | 523,891 | 5,842 | 18,423 |
| funding_rate | -0.0168 | 0.0206 | 0.0004 | 0.0031 |
| gdelt_econ_volume | 0 | 487 | 27 | 45 |
| gdelt_econ_tone | -97.2 | 82.5 | -8.3 | 18.7 |
| gdelt_conflict_volume | 0 | 156 | 8 | 15 |
| is_post_ETF | 0 | 1 | 0.20 | 0.40 |

### Targets
| Target | Min | Max | Mean | Std | Description |
|--------|-----|-----|------|-----|-------------|
| y_baseline | -0.0168 | 0.0206 | 0.0004 | 0.0031 | Absolute funding rate at t+1 |
| y_heuristic | -8.45 | 9.12 | 0.02 | 1.15 | Weighted Z-score composite at t+1 |
| y_vol_adj_return | -12.43 | 14.50 | 0.05 | 2.10 | Volatility-Adjusted return at t+1 |

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
| **GDELT Exogenous** | Macro news | 2020-2026 | 43,909 hourly |
| **CoinDesk News** | News text | 2019-2025 | 37,688 unique hours |
| **Generated Charts** | Technical | 2020-2025 | 44,477 valid images |

---

## Loading the Dataset (For Extraction Only)

> ⚠️ **IMPORTANT:** This dataset is used **ONLY for offline feature extraction** via `src/training/extract_features.py` on your local machine.
>
> **Training uses Kaggle dataset** (pre-extracted .pt files), NOT HuggingFace. See [docs/MODEL_ARCHITECTURE.md](./MODEL_ARCHITECTURE.md).

### Quick Start
```python
from datasets import load_dataset

# Load Bitcoin dataset (for LOCAL EXTRACTION ONLY)
dataset = load_dataset("khanh252004/multimodal_crypto_sentiment_btc")

# Access splits
train = dataset["train"]           # ~37,764 rows
test = dataset["test"]             # ~6,665 rows

# Inspect sample
sample = train[0]
print(sample.keys())
# ['timestamp', 'return_1h', 'volume', 'funding_rate', 'gdelt_econ_volume', 
#  'gdelt_econ_tone', 'gdelt_conflict_volume', 'is_post_ETF',
#  'text_content', 'image_path', 'y_baseline', 'y_heuristic', 'y_vol_adj_return']

# After loading, run extraction to save .pt files:
# python src/training/extract_features.py --asset MULTI
# Then upload to Kaggle for training.
```

### Access by Modality
```python
# Tabular (7 features)
return_1h = sample["return_1h"]          # float
volume = sample["volume"]                # float
funding = sample["funding_rate"]         # float
is_post_ETF = sample["is_post_ETF"]      # int (0 or 1)
... (3 more tabular features)

# Text (news)
text = sample["text_content"]            # str, ~2000 tokens

# Image (chart)
image = sample["image_path"]             # PIL.Image, 224x224 PNG

# Targets
y_base = sample["y_baseline"]            # float
y_heur = sample["y_heuristic"]           # float
y_vol = sample["y_vol_adj_return"]       # float
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
         x["gdelt_econ_volume"], x["gdelt_econ_tone"], 
         x["gdelt_conflict_volume"], x["is_post_ETF"]]
        for x in batch
    ], dtype=torch.float32)
    
    # Text: tokenize with BERT later
    texts = [x["text_content"] for x in batch]
    
    # Images: (B, 3, 224, 224)
    images = torch.stack([
        torch.tensor(np.array(x["image_path"]), dtype=torch.float32).permute(2, 0, 1) / 255.0
        for x in batch
    ])
    
    # Targets: (B, 3)
    targets = torch.tensor([
        [x["y_baseline"], x["y_heuristic"], x["y_vol_adj_return"]]
        for x in batch
    ], dtype=torch.float32)
    
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

### 1. Multimodal Fusion with [FUSION] Token (State-of-the-art)
```
Tabular (7 + 16 AssetID) ──→ MLP ─────────┐
                                          ├─→ [FUSION] Token Stack ──→ Cross-Modal Attention ──→ LSTM ──→ MLP Heads ──→ Predict targets
Text Embedding (256) ─────────────────────┤
Image Embedding (256) ────────────────────┘
```
- Input: Tabular, Text Embeddings, Image Embeddings
- Fusion: Learnable [FUSION] token extracts modality interactions, compressed via bottleneck, temporal modeling via LSTM.

### 2. Tabular Baselines
- Input: 7 features + asset ID embeddings
- Models: LightGBM, XGBoost, or LSTM baseline on tabular data only.

---

## Training Pipeline

This HuggingFace dataset is **sourced for local extraction only**. For training:

1. **Extract locally** (1-2 hours):
   ```bash
   python src/training/extract_features.py --asset MULTI --force
   ```
   Outputs: `data/features/{ASSET}/*.pt` (text/image embeddings, tabular, targets)

2. **Upload to Kaggle** (optional):
   ```bash
   python src/training/extract_features.py --asset MULTI --push-to-kaggle --kaggle-dataset-name crypto-sentiment-embeddings --kaggle-username <your-username> --kaggle-key <your-key>
   ```

3. **Train on Kaggle** (zero HuggingFace dependencies):
   ```bash
   python src/training/train.py --features-dir /kaggle/input/crypto-sentiment-embeddings
   ```

---

## Citation

```bibtex
@dataset{crypto_sentiment_v5,
  title={Multimodal Cryptocurrency Market Sentiment Dataset (v5)},
  author={Khanh252004},
  year={2026},
  month={May},
  url={https://huggingface.co/datasets/khanh252004/multimodal_crypto_sentiment_btc},
  note={BTC & ETH datasets with 13-field multimodal structure, used for offline extraction to Kaggle}
}
```

---

## License

**CC BY-NC 4.0** (Creative Commons Attribution Non-Commercial 4.0)

- ✅ **Permitted:** Research, academic use, non-commercial projects
- ❌ **Not permitted:** Commercial products, paid services without attribution

---

**Dataset Card Version:** 2.0  
**Last Updated:** May 24, 2026  
**Status:** ✅ Production-ready
