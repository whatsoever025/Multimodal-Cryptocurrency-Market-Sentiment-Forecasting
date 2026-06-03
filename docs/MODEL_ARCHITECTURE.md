# MultimodalFusionNet Architecture Documentation

**Last Updated:** June 3, 2026  
**Model Version:** 5.1 (Hyperparameter Re-tuning — Plateau Fix)  
**Status:** Production-Ready

---

## Overview

**MultimodalFusionNet** is a production-grade multimodal sentiment forecasting architecture with **offline feature extraction**. It accepts pre-computed text and image embeddings (extracted via frozen FinBERT and Vision Transformer backbones) and fuses them with tabular features using cross-modal attention and a temporal LSTM for multi-target sentiment forecasting.

**Purpose:** Forecast cryptocurrency sentiment indicators on a continuous scale using combined textual, visual, and numerical market data with **zero I/O bottlenecks, lightweight trainable architecture (~330K params), and pure float32 stability**.

**Pipeline:**
1. **Offline Phase** (run once): Extract FinBERT text embeddings & Vision Transformer (ViT) image embeddings → save as `.pt` files in asset-specific directories (`BTC/` and `ETH/`).
2. **Training Phase** (fast & stable): Load pre-extracted embeddings + apply TabularEncoder, learnable `[FUSION]` token stacking, CrossModalAttention, Temporal LSTM, and Prediction Heads.

---

## Architecture Components

### 1. Input Specification

> **Note:** All inputs are loaded from disk during training. Text and image embeddings are pre-computed offline via `src/training/extract_features.py`. Tabular features are scaled dynamically in-memory using StandardScaler fitted on the training split of the current fold/asset.

| Modality | Shape | Type | Source | Description |
|----------|-------|------|--------|-------------|
| **Tabular (SCALED)** | (batch, seq_len, 7) | float32 | StandardScaler | Market features: return_1h, volume, funding_rate, gdelt_econ_volume, gdelt_econ_tone, gdelt_conflict_volume, is_post_ETF |
| **Text Embedding** | (batch, seq_len, 256) | float32 | Offline extracted | FinBERT [CLS] token embeddings projected to 256D (pre-computed) |
| **Image Embedding** | (batch, seq_len, 256) | float32 | Offline extracted | Vision Transformer (ViT) embeddings projected to 256D (pre-computed) |
| **Targets (SCALED)** | (batch, 3) | float32 | RobustScaler | Targets: y_baseline, y_heuristic, y_vol_adj_return (fitted on train split of current fold/asset) |

- **seq_len (sequence length):** 24 hours (default)  
- **Embedding Dimension:** 256 (FinBERT 768 → 256, ViT 768 → 256)  
- **Asset Embeddings:** 16-dimensional asset category embeddings are concatenated with the tabular features inside the model.

---

## Preprocessing & Dynamic Scaling Strategy

### 1. Tabular Feature Scaling (StandardScaler)
- **Applied to:** Tabular features (7 dimensions) before TabularEncoder.
- **Fitting:** Fitted dynamically per-fold and per-asset *only* on the training slice of that fold.
- **Data Leakage Prevention:** Prevents temporal look-ahead data leakage and cross-asset contamination (statistics are computed independently for BTC and ETH train partitions and applied to their respective validation/test partitions).

### 2. Target Scaling (RobustScaler)
- **Applied to:** Target sentiment scores for loss computation.
- **Fitting:** Fitted dynamically per-fold and per-asset *only* on the training slice of that fold.
- **Formulas/Pre-adjustments in `dataset.py`:**
  - `y_baseline` represents the raw funding rate. Since it is extremely small, it is scaled by 1000.0 (`target_scores[:, 0] = target_scores[:, 0] * 1000.0`).
  - `y_heuristic` represents a weighted Z-score composite. To avoid outlier distortion, it is clipped to `[-5.0, 5.0]` before scaling.
  - `y_vol_adj_return` represents the volatility-adjusted log return. It is kept raw/un-clipped.

---

## Detailed Model Layers

### 1. TabularEncoder
```
Input: (batch, seq_len, 7+16=23) numeric + asset embedding features
       ↓
Linear (23 → 64) + ReLU + Dropout (0.2)
       ↓
Linear (64 → 256) + ReLU + Dropout (0.2)
       ↓
Output: (batch, seq_len, 256)
```
- **Xavier initialization** is applied to all linear layers.
- Output dimension of **256** matches the text and image embeddings.

### 2. Learnable [FUSION] Token + Cross-Modal Attention
To combine the three modalities at each timestep, we stack a learnable fusion detector token alongside the modal embeddings:
```
Learnable [FUSION] Token: (1, 1, 256) → expands to (batch, seq_len, 1, 256)
  ↓
Input Stack: [ [FUSION] (256) + Text (256) + Image (256) + Tabular (256) ]
  ↓ (batch, seq_len, 4, 256)
Reshape to: (batch*seq_len, 4, 256)
  ↓
Multi-Head Self-Attention (4 tokens attend to each other)
  ├─ embed_dim: 256
  ├─ num_heads: 4
  ├─ mha_dropout: 0.1
  └─ batch_first: True
  ↓
Residual Connection + LayerNorm (Pre-LN Structure for gradient stability)
  ↓
Extract [FUSION] Token Only (position 0) → (batch*seq_len, 256)
  ↓
Reshape back to: (batch, seq_len, 256)
```

### 3. Bottleneck Layer
```
Input: (batch, seq_len, 256)  [Fused [FUSION] representation]
       ↓
Linear Projection: 256 → 128
       ↓
Output: (batch, seq_len, 128)
```
Compresses features before LSTM. **128** (up from 64) provides sufficient capacity
to encode 24-hour multimodal context without information bottleneck.

### 4. Temporal LSTM Layer
```
Input: (batch, seq_len, 128)
       ↓
LSTM Cell (1 layer, batch-first)
├─ Input Size: 128
├─ Hidden Size: 128
├─ Dropout: 0.3 (reduced from 0.5 — was too aggressive at low LR)
└─ Batch First: True
       ↓
Extract Final Hidden State h_n[-1] → (batch, 128)
```

### 5. Prediction Heads
Independent prediction heads are trained for each of the three targets.
```
Input: (batch, 128)  [LSTM final hidden state]
       ↓
Linear (128 → 64) + ReLU + Dropout (0.3)
       ↓
Linear (64 → 1)
       ↓
Output: (batch, 1)  [continuous predicted sentiment score]
```
> **Note:** Intermediate dimension is computed dynamically as `input_dim // 2`
> (here: 128 // 2 = 64), so it scales automatically with `lstm_hidden_dim`.

---

## Default Configuration

### DataConfig (`config.py`)
```python
asset: "MULTI"                          # "BTC", "ETH", or "MULTI" (combined)
seq_len: 24                             # 24-hour sliding window
batch_size: 128                         # Default batch size
```

### ModelConfig (`config.py`)
```python
hidden_dim: 256                         # Internal embedding dimension
bottleneck_dim: 128                     # Bottleneck dimension (raised from 64: more capacity for 24h context)
lstm_layers: 1                          # LSTM layers
lstm_hidden_dim: 128                    # LSTM hidden dimension (raised from 64)
lstm_dropout: 0.3                       # Reduced from 0.5 (was too aggressive at low LR)
attention_heads: 4                      # Cross-modal attention heads
mha_dropout: 0.1                        # MHA dropout (keeps backward dot-product stable)
encoder_dropout: 0.2                    # TabularEncoder dropout (reduced from 0.3)
head_dropout: 0.3                       # Prediction head dropout (reduced from 0.4)
grad_clip: 1.0                          # Gradient norm clipping (L2)
frozen_backbones: True                  # Freeze FinBERT & ViT
```

### TrainingConfig (`config.py`)
```python
max_epochs: 60                          # Training epochs
learning_rate: 1e-4                     # Raised from 1e-5: original was too low, caused plateau after epoch 2
                                        # Safe range for frozen-backbone head-only training: 1e-4 to 5e-4
weight_decay: 1e-3                      # Reduced from 1e-2: wd/lr ratio was 10x, counteracting gradient updates
accumulate_steps: 2                     # Gradient accumulation steps (effective batch = 256)
warmup_steps: 100                       # Reduced from 800: ~4 epochs (walk-forward fold ≈ 21 steps/epoch)
                                        # Original 800 steps = 38 epochs of warmup (LR never reached peak)
use_warmup: True                        # Enable warmup schedule
early_stopping_patience: 15             # Patience epochs (val loss min_delta=1e-4)
```

> **v5.1 Rationale — Plateau Fix:**  
> Walk-forward folds contain ~5,500 training samples (60% of one asset's 37,764 rows).  
> With `batch_size=128`, `accumulate_steps=2` → **~21 optimizer steps/epoch**.  
> The original `warmup_steps=800` therefore spanned **≈ 38 epochs of warmup**, meaning  
> the LR never reached its peak before cosine decay returned it to near-zero.  
> Combined with `weight_decay=1e-2` (10× the LR) actively counteracting gradient updates,  
> the model had essentially zero effective learning rate for 60% of training.

---

## Numerical Stability & Training Design

1. **Pure Float32 Training:** We do not use mixed-precision (AMP) or GradScaler. This avoids underflow/overflow errors in recurrent LSTM cells and attention backward passes.
2. **HuberLoss (`delta=1.0`):** Replaces MSE loss, providing robust gradients when dealing with noisy market indicators and outliers.
3. **Clamping Predictions:** Predictions are clamped to `[-150.0, 150.0]` before calculating the loss to prevent numerical spikes.
4. **Pre-LN Attention Structure:** Normalization occurs before attention computation rather than after, facilitating gradient flow in deeper parts of the network.
5. **Gradient Clipping:** L2 norm of gradients is clipped at `1.0` before optimization steps to prevent exploding gradients.
6. **Per-Fold Reset:** The weights of the trainable components are fully reset at the beginning of each walk-forward fold (`_reset_weights`), ensuring no leakage between folds.
7. **NaN Diagnostic checks:** Both pre-backward loss values and post-backward gradient checks are executed every batch. Problematic batches are serialized to disk (`problematic_batch_*.pt`) for inspection, halting execution on critical failures.

---

## Data Flow Diagram (Training Epoch)

```
[Pre-extracted Text (256D) & Image (256D) Tensors]
                   │
                   ▼ (Slice seq_len=24)
            [Text & Image Batch]
                   │
[Raw Tabular (7D)] ┼─► [StandardScaler (Per-fold/asset)] ──► [Scaled Tabular] ──► [TabularEncoder MLP] ──► [Tabular (256D)]
                   │                                                                                            │
                   └────────────────────────────────────────────────────────────────────────────────────────────┼──► Stack (3x256D)
                                                                                                                │
                                                                       [Learnable [FUSION] Token (1x256D)] ─────┘
                                                                                                │
                                                                                                ▼ (Concatenated: 4x256D)
                                                                                    [Cross-Modal Attention (Pre-LN)]
                                                                                                │
                                                                                                ▼ (Extract Fusion position 0)
                                                                                      [Bottleneck (256 → 64)]
                                                                                                │
                                                                                                ▼
                                                                                         [Temporal LSTM]
                                                                                                │
                                                                                                ▼ (Final hidden h_n)
                                                                                     [Prediction Head (MLP)]
                                                                                                │
                                                                                                ▼
                                                                                      [Clamped Prediction]
                                                                                                │
                                                                                     [HuberLoss vs Scaled Target]
                                                                                                │
                                                                                                ▼
                                                                             [Gradient Clip (1.0) & Optimizer Step]
```

---

**Generated:** 2026-05-24  
**Author:** Multimodal Crypto Sentiment Team  
**Status:** ✅ Production-ready and verified.
