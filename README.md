# Multimodal Cryptocurrency Market Sentiment Forecasting

**Thesis project — Hanoi University of Science and Technology, School of Information and Communication Technology**

This repository contains the full implementation for the thesis *"MultimodalFusionNet: A Multimodal Deep Learning Approach for Cryptocurrency Market Forecasting"*. The project empirically tests whether fusing tabular, textual, and visual modalities improves short-horizon forecasting over well-tuned unimodal baselines — and reports a negative result.

---

## Architecture: MultimodalFusionNet (MFN)

MFN is an intermediate-fusion network with three modality branches and a learnable `[FUSION]` token:

```
Tabular features (7 cols)  →  TabularEncoder MLP  →  256-dim
FinBERT text embedding      →  (pre-extracted)     →  256-dim   ─→  CrossModalAttention
ViT image embedding         →  (pre-extracted)     →  256-dim         (4 tokens: [FUSION], text, image, tabular)
                                                                          ↓ [FUSION] token output
                                                                  Bottleneck Linear 256→64
                                                                          ↓
                                                                  Temporal LSTM (1 layer, 64-dim)
                                                                          ↓
                                                                  PredictionHead (64→32→1)
```

Three independent models are trained (one per target). All backbone encoders (FinBERT, ViT) are **frozen**; only the fusion components are trained.

### Tabular features (7 columns)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `return_1h` | Hourly price return (%) |
| 1 | `volume` | Trading volume |
| 2 | `funding_rate` | Perpetual swap funding rate |
| 3 | `gdelt_econ_volume` | GDELT economic event volume |
| 4 | `gdelt_econ_tone` | GDELT economic tone (−100 to +100) |
| 5 | `gdelt_conflict_volume` | GDELT conflict event volume |
| 6 | `is_post_ETF` | Binary flag: 1 if timestamp ≥ 2024-01-01 |

An additional 16-dim learnable asset embedding (BTC=0, ETH=1) is concatenated to the tabular input before the encoder, making the effective input size 23-dim.

### Targets

| Code name | Thesis name | Horizon | Benchmark |
|-----------|-------------|---------|-----------|
| `y_baseline` | `y_funding` | t+8h | Historical Mean |
| `y_heuristic` | Heuristic Market-State Composite | t+1h | Zero |
| `y_vol_adj_return` | Volatility-Adjusted Log Return | t+1h | Zero |

> **Note:** `y_baseline` in the code corresponds to `y_funding` in the thesis document. The code name predates the final thesis terminology.

---

## Dataset

- **Assets:** Bitcoin (BTC) and Ethereum (ETH)
- **Period:** January 2020 – January 2025 (5 years)
- **Granularity:** Hourly
- **Size:** 88,998 samples (44,499 per asset)
- **Sources:** Binance OHLCV + funding rates, GDELT macro data, CoinDesk news (via HuggingFace), candlestick chart images

Raw data files are in `data/raw/`. Pre-extracted embeddings (FinBERT text, ViT image, tabular `.pt` tensors) are stored in `data/features/BTC/` and `data/features/ETH/` (not committed — loaded from Kaggle input dataset during training).

---

## Training (Kaggle)

Training runs on **Kaggle** (T4 GPU, 16 GB VRAM) using Python 3.10 and PyTorch 2.0.0. The full dataset and pre-extracted feature tensors are uploaded as a Kaggle input dataset.

**Walk-forward cross-validation:** 5 folds, 85/15 train/val split, seed=42. Each fold resets model weights from scratch. Per-fold per-asset `StandardScaler` (tabular) and `RobustScaler` (targets) prevent look-ahead bias.

### Kaggle training commands

```bash
# Standard full-model training (all 3 targets, 5 folds)
python -m src.training.train \
    --features-dir /kaggle/input/<dataset-name>/features \
    --tabular-dir  /kaggle/input/<dataset-name>/features \
    --asset MULTI --num-folds 5 --seed 42

# Ablation: tabular-only (text + image zeroed)
python -m src.training.train \
    --features-dir /kaggle/input/<dataset-name>/features \
    --ablation tabular_only --seed 42

# Ablation: no_funding variant (drop funding_rate col)
python src/training/create_ablation_features.py \
    --input-dir /kaggle/input/<dataset-name>/features \
    --output-dir /kaggle/working \
    --variant no_funding
python -m src.training.train \
    --tabular-file tabular_features_no_funding.pt \
    --tabular-dir /kaggle/working \
    --ablation tabular_only --seed 42

# Run baselines (XGBoost, Linear, Historical Mean, Persistence)
python -m src.baseline.run_baselines \
    --features-dir /kaggle/input/<dataset-name>/features \
    --seed 42
```

---

## Local Usage (evaluation / data pipeline only)

Training requires Kaggle GPU. Locally you can run the data alignment pipeline and EDA.

### Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Run the data alignment pipeline

```bash
# BTC
python -m src.preprocessing.data_aligner --asset BTC

# ETH
python -m src.preprocessing.data_aligner --asset ETH
```

### Generate candlestick charts

```bash
python src/generate_charts.py
```

### Run significance tests (Diebold-Mariano)

```bash
# After obtaining prediction .npz files from Kaggle
python -m src.baseline.significance_tests
```

---

## Key Results

| Model | `y_funding` R²_OOS (vs. Hist. Mean) | `y_funding` R²_OOS (vs. Persistence) |
|-------|-------------------------------------|---------------------------------------|
| XGBoost | **0.78** | 0.13 |
| Tabular LSTM | 0.76 | — |
| MFN (full) | 0.52 | — |

The 1-hour targets (`y_heuristic`, `y_vol_adj_return`) yielded R²_OOS ≈ 0 for all models, consistent with weak-form market efficiency.

A modality ablation confirmed that removing text and image inputs **improved** out-of-sample accuracy. The funding rate's high autocorrelation (r ≈ 0.97) largely explains its apparent tractability.

---

## Repository Structure

```
.
├── data/
│   ├── raw/                    # Raw CSV files (OHLCV, funding rates, GDELT, news)
│   └── processed/              # Generated charts and embeddings (gitignored)
├── docs/
│   ├── MODEL_ARCHITECTURE.md
│   ├── DATA_DICTIONARY.md
│   └── HF_DATASET_CARD.md
├── src/
│   ├── crawlers/
│   │   ├── base.py                     # Abstract base class for all crawlers
│   │   ├── binance_vision_crawler.py   # Binance OHLCV + funding rate data
│   │   └── huggingface_crawler.py      # CoinDesk news via HuggingFace datasets
│   ├── preprocessing/
│   │   ├── data_aligner.py             # Multimodal alignment pipeline
│   │   └── chart_generator.py          # Candlestick chart renderer
│   ├── training/
│   │   ├── config.py                   # Hyperparameter configuration
│   │   ├── dataset.py                  # Walk-forward dataset + data loaders
│   │   ├── model.py                    # MultimodalFusionNet architecture
│   │   ├── train.py                    # Training loop + evaluation
│   │   ├── extract_features.py         # Offline FinBERT + ViT feature extraction
│   │   ├── create_ablation_features.py # no_funding variant creator
│   │   └── utils.py
│   ├── baseline/
│   │   ├── models.py                   # XGBoost, Linear, Historical Mean baselines
│   │   ├── run_baselines.py            # Baseline training + evaluation
│   │   ├── kaggle_baseline.py          # Self-contained single-file version for Kaggle
│   │   ├── significance_tests.py       # Diebold-Mariano test
│   │   └── metrics.py
│   └── generate_charts.py              # Entry-point script for chart generation
├── EDA_Dataset_Verification.ipynb
├── baseline_results.json               # Saved baseline evaluation results
├── requirements.txt
└── .env.example
```

---

## Reproducibility

| Setting | Value |
|---------|-------|
| Seed | 42 |
| Python | 3.10 |
| PyTorch | 2.0.0 |
| GPU (Kaggle) | NVIDIA T4, 16 GB |
| Walk-forward folds | 5 |
| Train/val split | 85% / 15% |
| Sequence length | 24 hours |
| Batch size | 128 |

The Kaggle notebook (link in thesis) contains the complete training run with all seeds fixed.
