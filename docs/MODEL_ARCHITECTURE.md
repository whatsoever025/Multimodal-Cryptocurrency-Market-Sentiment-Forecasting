# MultimodalFusionNet Architecture Documentation

**Last Updated:** April 15, 2026  
**Model Version:** 2.0 (Offline Feature Extraction)  
**Status:** Production-Ready

---

## Overview

**MultimodalFusionNet** is a production-grade multimodal sentiment forecasting architecture with **offline feature extraction**. It accepts pre-computed text and image embeddings (extracted via frozen FinBERT and ResNet50 backbones) and fuses them with tabular features using cross-modal attention and temporal LSTM for continuous sentiment score prediction.

**Purpose:** Forecast cryptocurrency sentiment on a continuous scale (-100 to +100) using combined textual, visual, and numerical market data with **zero I/O bottlenecks and zero float16 NaN issues**.

**Pipeline:**
1. **Offline Phase** (run once): Extract FinBERT text embeddings & ResNet50 image embeddings → save as `.pt` files
2. **Training Phase** (fast & stable): Load pre-extracted embeddings + apply TabularEncoder, CrossModalAttention, LSTM, PredictionHead

---

## Architecture Components

### 1. **Input Specification**

> **Note:** All inputs are loaded from disk during training. Text and image embeddings are pre-computed offline via `src/data/extract_features.py`. Tabular features are scaled in-memory using StandardScaler fitted on training split.

| Modality | Shape | Type | Source | Description |
|----------|-------|------|--------|-------------|
| **Tabular (SCALED)** | (batch, seq_len, 7) | float32 | StandardScaler | Market features: return_1h, volume, funding_rate, fear_greed, gdelt_econ_volume, gdelt_econ_tone, gdelt_conflict_volume (normalized: mean=0, std=1) |
| **Text Embedding** | (batch, seq_len, 256) | float32 | Offline extracted | FinBERT [CLS] token embeddings (pre-computed) |
| **Image Embedding** | (batch, seq_len, 256) | float32 | Offline extracted | ResNet50 avgpool embeddings (pre-computed) |
| **Target (SCALED)** | (batch,) | float32 | RobustScaler | Sentiment scores [-100, +100] normalized (median=0, IQR=1) |

**seq_len (sequence length):** 24 hours (default)  
**Embedding Dimension:** 256 (FinBERT 768 → 256, ResNet50 2048 → 256)  
**Scaling Strategy:** StandardScaler for tabular (learned on train split only), RobustScaler for targets (learned on train split only)

---

## Data Preprocessing & Scaling Strategy

### Tabular Feature Scaling (StandardScaler)
**Applied to:** Market features (7 dimensions) before TabularEncoder  
**Fitting:** Learned ONLY from training split during dataset initialization  
**Application:** Same statistics applied to train, validation, and test splits  
**Parameters:**
- `mean_` (shape: 7,) — Mean of each feature computed from training data
- `scale_` (shape: 7,) — Standard deviation of each feature computed from training data
- **Formula:** `X_scaled = (X_raw - mean_) / scale_`

**Purpose:** Normalize features to zero mean and unit variance, improving neural network convergence  
**Data Leakage Prevention:** Scaler's fit() ONLY sees training data; transform() applies to all splits

### Target Score Scaling (RobustScaler)
**Applied to:** Target sentiment scores [-100, +100] for loss computation  
**Fitting:** Learned ONLY from training split during dataset initialization  
**Application:** Same statistics applied to train, validation, and test splits  
**Parameters:**
- `center_` (scalar) — Median of target scores computed from training data
- `scale_` (scalar) — Interquartile range (IQR) computed from training data
- **Formula:** `y_scaled = (y_raw - median) / IQR`

**Purpose:** Robust to outliers in sentiment scores (extreme market events don't distort scale)  
**Rationale:** RobustScaler uses median/IQR instead of mean/std, better for sentiment with outliers  
**Data Leakage Prevention:** Scaler's fit() ONLY sees training data; transform() applies to all splits

### Implementation Details
**Location:** `src/training/dataset.py` → `CryptoMultimodalDataset` class  
**Key Method:** `_load_tabular_and_targets(split):`
- If `split == "train"`: Fit StandardScaler & RobustScaler on raw training data, apply scaling
- If `split == "validation"` or `"test_in_domain"`: Load scalers from training split, apply to current split
- Scaling happens ONCE during dataset initialization (in-memory, no disk I/O)

**Critical:** Both scalers are stored in the training dataset and reused by validation/test datasets to prevent data leakage. Validation and test data are scaled using training statistics.

---

### 2. **Modality Encoders**

> **Architecture Change (v2.0):** TextEncoder and ImageEncoder removed. Embeddings are pre-computed offline and passed directly to the model. Only TabularEncoder is trained.

#### A. Text & Image Embeddings (Offline, Pre-computed)

**Extraction Pipeline** (`src/data/extract_features.py`):
```
1. Text: (n_samples, max_text_length) tokens
   ↓ FinBERT Backbone (frozen) + [CLS] extraction
   ↓ Linear Projection (768 → 256)
   ↓ Save as text_embeddings_{split}.pt: (n_samples, 256)

2. Image: (n_samples, 3, 224, 224) pixels
   ↓ ResNet50 Backbone (frozen) + avgpool
   ↓ Linear Projection (2048 → 256)
   ↓ Save as image_embeddings_{split}.pt: (n_samples, 256)
```

**During Training:**
- Load pre-extracted embeddings from disk: `text_embeddings_train.pt`, `image_embeddings_train.pt`
- Slice by seq_len window: `text_embeddings[idx:idx+seq_len]` → (seq_len, 256)
- No backbone computation during training

**Benefits:**
- ✅ Zero I/O bottleneck (embeddings cached in memory)
- ✅ No redundant backbone inference
- ✅ 3-5x faster training per epoch
- ✅ 8-10GB VRAM vs 16GB (more headroom)

#### B. TabularEncoder (Only Trainable Encoder)
```
Input: (batch, seq_len, 7) numeric features (ALREADY SCALED by StandardScaler)
       ↓
Linear (7 → 64) + ReLU + Dropout (0.2)
       ↓
Linear (64 → 256) + ReLU + Dropout (0.2)
       ↓
Output: (batch, seq_len, 256)
```

**Key Properties:**
- Architecture: 2-layer MLP
- Activation: ReLU (applied after both linear layers)
- Weight Initialization: Xavier uniform on all linear layers
- Trainable: ✅ Yes (only encoder with learnable parameters)
- Input Preprocessing: Receives StandardScaler-normalized features (mean=0, std=1)
- **Output Dimension: 256** (matches text & image embedding dimensions)
- Dropout: 0.2 (after each layer)
- Note: Input is PRE-SCALED by StandardScaler (fit on training split) before reaching this encoder
- **Critical Design:** Output must be exactly 256D to match pre-extracted embeddings for modality stacking

---

### 3. **Cross-Modal Attention Layer**

```
Input: (batch, seq_len, 3, 256)  [Stack: text (256) + image (256) + tabular (256)]
       ↓
       All modalities MUST have matching dimensions (256D) ✓
       ↓
Reshape to (batch*seq_len, 3, 256)  [Treat 3 modalities as sequence]
       ↓
Multi-Head Self-Attention (3 modalities attend to each other)
       ├─ embed_dim: 256
       ├─ num_heads: 4
       ├─ dropout: 0.1
       └─ batch_first: True
       ↓
Residual Connection + LayerNorm
       ↓
Mean Pooling across modality dimension (average 3 modalities)
       → (batch*seq_len, 256)
       ↓
Reshape back to (batch, seq_len, 256)
       ↓
Output: (batch, seq_len, 256)  [fused representation]
```

**Key Properties:**
- Attention Type: Multi-Head Self-Attention using `torch.nn.MultiheadAttention`
- Number of Heads: 4 (configurable via `config.model.attention_heads`)
- Attention Dimension: 256 (matches embedding dimension)
- Dropout: 0.1 (configurable via `config.model.mha_dropout`)
- Modalities Treated As: Sequence of length 3 (text, image, tabular)
- Modality Fusion Strategy: Mean pooling across modality dimension (dim=1 after reshape)
- Residual Connection: Added before LayerNorm for gradient flow
- **Dimension Consistency:** All 3 modalities are 256D, ensuring proper attention computation

---

### 4. **Temporal LSTM Layer**

```
Input: (batch, seq_len, 256)  [Fused embeddings from cross-modal attention]
       ↓
LSTM Cell (2 layers, batch-first)
├─ Input Size: 256
├─ Hidden Size: 256
├─ Num Layers: 2
├─ Batch First: True
└─ Dropout: 0.2 (between layers, layer 1 → layer 2)
       ↓
Capture Temporal Dynamics (seq_len timesteps)
       ↓
Extract Final Hidden State h_n[-1]: (batch, 256)
       ↓
Output: (batch, 256)
```

**Key Properties:**
- Type: Unidirectional LSTM (left-to-right temporal modeling)
- Layers: 2
- Input/Hidden Dimension: 256 (fixed to embedding dimension)
- Dropout Between Layers: 0.2
- Output Selection: Final hidden state from last (2nd) layer
- Purpose: Capture temporal dynamics & market trends across 24-hour window

---

### 5. **Prediction Head (MLP)**

```
Input: (batch, 256)  [LSTM final hidden state]
       ↓
Linear (256 → 128) + ReLU + Dropout (0.2)
       ↓
Linear (128 → 64) + ReLU + Dropout (0.2)
       ↓
Linear (64 → 1)
       ↓
Output: (batch, 1)  [continuous sentiment score]
```

**Key Properties:**
- Architecture: 3-layer MLP
- Hidden Dimensions: 256 → 128 → 64 → 1
- Activation Functions: ReLU (hidden layers), None (output)
- Dropout: 0.2 (hidden layers)
- Output Range: Unrestricted float (model learns to predict [-100, +100] range)
- Weight Initialization: Xavier uniform on all linear layers
- Loss Function: MSE (Mean Squared Error)

---

## Default Configuration

### DataConfig
```python
asset: "MULTI"                          # "BTC", "ETH", or "MULTI" (combined)
seq_len: 24                             # 24-hour sliding window
batch_size: 128                         # Per-GPU batch size for training
max_text_length: 512                    # BERT token sequence length
image_size: 224                         # ResNet50 input size
shuffle_train: True
num_workers: 0                          # Kaggle compatibility (no multiprocessing)
pin_memory: True
prefetch_factor: 2
```

### ModelConfig
```python
hidden_dim: 256                         # Internal embedding dimension (MUST match embedding dimensions)
lstm_layers: 2                          # Temporal LSTM layers
lstm_dropout: 0.2                       # Dropout between LSTM layers
attention_heads: 4                      # Cross-modal attention heads
mha_dropout: 0.1                        # Attention layer dropout
encoder_dropout: 0.2                    # TabularEncoder dropout
head_dropout: 0.2                       # Prediction head dropout
grad_clip: 1.0                          # Gradient norm clipping (L2)
frozen_backbones: True                  # Freeze FinBERT & ResNet50
use_gradient_checkpointing: True        # Memory optimization for 16GB VRAM
```

**Design Note:** `hidden_dim=256` is **intentionally set to match** the 256D pre-extracted embeddings (FinBERT & ResNet50 projections). This ensures all 3 modalities (text, image, tabular) have identical feature dimensions for proper stacking in CrossModalAttention.

### TrainingConfig
```python
max_epochs: 60                          # Training epochs
learning_rate: 5e-5                     # Conservative for multimodal + AMP
weight_decay: 1e-5                      # L2 regularization
accumulate_steps: 2                     # Gradient accumulation (BS 128 → eff BS 256)
warmup_steps: 800                       # Steps for learning rate warmup
use_warmup: True                        # Enable learning rate warmup
scheduler_type: "cosine"                # Cosine annealing with warmup
```

### OptimizationConfig
```python
mixed_precision: True                   # ✅ Enable torch.cuda.amp.autocast
dtype: "float16"                        # "float16" or "bfloat16" precision
use_scaler: True                        # GradScaler for mixed precision
init_scale: 65536.0                     # Initial loss scale for GradScaler
growth_factor: 2.0                      # Loss scale growth factor
backoff_factor: 0.5                     # Loss scale backoff factor
growth_interval: 2000                   # Steps between growth checks
```

**Note:** v2.0 uses mixed precision (float16 + float32) with GradScaler, NOT pure float32. Mixed precision reduces VRAM while maintaining stability through automatic loss scaling.

---

## Model Statistics

### Parameter Counts (Offline Version)
- **FinBERT:** 109M (extracted offline, NOT in model)
- **ResNet50:** 24M (extracted offline, NOT in model)
- **Trainable Components:**
  - TabularEncoder: ~50K
  - CrossModalAttention: ~600K
  - TemporalLSTM: ~2.5M
  - PredictionHead: ~150K
  - **Total Trainable:** ~3.3M
- **Model Size:** Only ~13MB (vs ~6GB with backbones)

### Memory Usage (Kaggle 16GB with AMP)
- Model Parameters: ~13MB
- Optimizer States (AdamW): ~26MB
- Batch (BS=128, seq_len=24, float16 activations): ~6-8GB (embeddings + tabular)
- Pre-loaded Embeddings: ~2-3GB (text + image for split)
- GradScaler State: ~1MB
- **Total: ~10-12GB** (vs 12-13GB without offline extraction)
- **Headroom:** 4-6GB available for other operations ✅

**Note:** Larger batch size (128 vs 8) requires mixed precision to fit in 16GB.

---

## Training Pipeline

### Loss Function
```python
MSE Loss: L = mean((predictions - targets)²)
Note: Targets are continuous sentiment scores [-100, +100]
```

### Optimizer
```python
AdamW Optimizer:
  - Learning Rate: 1e-4 (higher than online version, only trainable layers)
  - Weight Decay: 1e-5 (L2 regularization)
  - Betas: (0.9, 0.999) (Adam standard)
  - Epsilon: 1e-8 (numerical stability)
Note: Applied only to ~3.3M trainable parameters
```

### Learning Rate Schedule
```
Warmup Phase (800 steps):
  Linear increase from 0 → 1e-4

Cosine Annealing Phase:
  Cosine decay from 1e-4 → ~0
  
Total Training Steps: len(train_loader) × max_epochs / accumulate_steps
Example: ~8000 batches × 60 epochs / 2 = ~240K steps
```

### Gradient Management (Mixed Precision)
```
Training Precision: float16 (activations) + float32 (weights)
Gradient Accumulation: 2 steps
  → Effective batch size: 128 × 2 = 256
  → Enables larger accumulation without float16 underflow

Automatic Loss Scaling (GradScaler):
  → Initial scale: 65536.0
  → Growth: ×2 every 2000 steps (if no NaNs)
  → Backoff: ÷2 if gradient overflow detected
  → Prevents float16 gradient underflow

Gradient Clipping: L2 norm ≤ 1.0
  → Applied BEFORE optimizer.step()
  → Prevents explosion in LSTM/Attention
  → torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

Optimizer: AdamW (mixed precision compatible)
  → Gradients computed in float16
  → Weight updates applied in float32
```

---

## Data Flow Example (Offline Version)

**Setup Phase** (run once, `src/data/extract_features.py`):
```
Dataset: 30K+ samples per split (train/val/test)
  ↓
Text Pipeline:
  (30K, 512) tokens → FinBERT Backbone → (30K, 768) CLS embeddings 
  → Linear Projection (768 → 256) → (30K, 256)
  → Save: text_embeddings_train.pt, text_embeddings_val.pt, text_embeddings_test.pt

Image Pipeline:
  (30K, 3, 224, 224) pixels → ResNet50 Backbone → (30K, 2048) avgpool 
  → Linear Projection (2048 → 256) → (30K, 256)
  → Save: image_embeddings_train.pt, image_embeddings_val.pt, image_embeddings_test.pt

Computational Cost: ~30-60 minutes (one-time, GPU-accelerated)
Output Size: ~750MB per split (text + image embeddings combined)
```

**Training Phase** (per batch, fast & stable):
```
Batch size: 128 samples, seq_len: 24

Load from memory (cached during epoch start):
  text_embeddings[idx:idx+24]  → (24, 256)  [pre-computed, no scaling needed]
  image_embeddings[idx:idx+24] → (24, 256)  [pre-computed, no scaling needed]
  tabular[idx:idx+24] → (24, 7) [RAW features, WILL BE SCALED]
  targets[idx:idx+24] → (24,) [RAW scores, WILL BE SCALED]
                              ↓
              APPLY IN-MEMORY SCALING (learned from training split):
        tabular_scaled: (24, 7) = (tabular_raw - mean_) / scale_  [StandardScaler]
        targets_scaled: (24,) = (targets_raw - median) / IQR      [RobustScaler]
                              ↓
          Stack into batch (128, 24, X):
        text_emb: (128, 24, 256)
        image_emb: (128, 24, 256)
        tabular_scaled: (128, 24, 7) ← NOW SCALED
                              ↓
        Encode tabular → TabularEncoder MLP → (128, 24, 256)
                              ↓
          Stack modalities: (128, 24, 3, 256) ✓ All dimensions match!
              [text(256) + image(256) + tabular(256)]
                              ↓
                    Cross-Modal Attention Layer
                    (3 modalities attend to each other)
                  (128, 24, 256) [fused representation]
                              ↓
                      Temporal LSTM (2 layers)
                   (128, 256) [final hidden state]
                              ↓
                      Prediction Head (MLP)
                        (128, 1) [logits]
                              ↓
                    Sentiment Scores (scaled space)
                              ↓
          Compute Loss: MSE(predictions_scaled, targets_scaled)
          Backward pass (float16 activations + float32 weights via AMP)
          Gradient scaling via GradScaler (automatic loss scaling)
          Gradient clipping (L2 norm ≤ 1.0)
          Optimizer step with warmup/cosine schedule
          Gradient accumulation every 2 steps → effective BS 256
```

**Performance Gains (Offline vs Online):**
- Setup cost: ~1 hour for 30K+ samples (one-time, amortized) 
- Training speedup: 3-5x faster per epoch (no backbone inference)
- Batch time: ~100-200ms per 128 samples (vs 500ms-1s with online encoding)
- VRAM reduction: ~16GB (with backbones) → ~10-12GB (offline + AMP)
- Total training speed for 60 epochs: ~2-3 hours vs ~8-10 hours

---

## Special Features (Offline Version v2.0)

### 1. VRAM Optimization
- ✅ **Offline extraction:** Backbones (109M + 24M) not in model during training
- ✅ **Model footprint reduced:** ~13MB (vs ~6GB with backbones)
- ✅ **Mixed precision (AMP):** float16 activations + float32 weights reduces memory
- ✅ **Automatic loss scaling:** GradScaler prevents float16 gradient underflow
- ✅ **Gradient accumulation:** Effective batch 256 from batch 128 with shared VRAM
- ✅ **Pre-loaded embeddings:** Cached in memory after loader, zero redundant I/O

### 2. Numerical Stability
- ✅ **Mixed precision with GradScaler:** Safely handles float16 without NaN overflow
- ✅ **Loss scaling:** Automatic adjustment prevents gradient underflow/overflow
- ✅ **Gradient clipping:** L2 norm ≤ 1.0 prevents LSTM/Attention explosion
- ✅ **Xavier initialization:** All trainable layers uniformly initialized
- ✅ **Conservative LR:** 5e-5 (safe for 3.3M trainable params)
- ✅ **Warmup phase:** 800 steps linear warmup prevents early instability

### 3. Reproducibility
- Seed: 42 (set at training start)
- Deterministic: Yes (reproducible embeddings extracted with fixed seed)
- Checkpoint Format: Best model + periodic epochs
- Config Logging: All hyperparameters saved with checkpoint

### 4. Monitoring & Debugging
- **Weights & Biases integration:** Real-time experiment tracking and visualization
- **Per-step logging** (every 100 steps): loss, learning rate, global step, loss scale (AMP)
- **Per-epoch validation:** MSE + MAE metrics on validation set
- **Checkpoint management:** Save best model + last 3 checkpoints (configurable)
- **tqdm progress bars:** Batch-level granularity with moving average loss
- **Gradient norm logging:** Monitor gradient norms for debugging (track clipping frequency)
- **Mixed precision monitoring:** Log loss scale changes from GradScaler

### 5. Pipeline Advantages
- ✅ **Separation of concerns:** Feature extraction once, train many times
- ✅ **Faster iteration:** Change model without re-extracting embeddings
- ✅ **Reproducible embeddings:** Same embeddings across all experiments
- ✅ **Easy ablation:** Test different architectures without backbone changes

---

## Known Limitations & Critical Issues

### ✅ FIXED: Dimension Mismatch (Previously Critical)

**Status:** ✅ **RESOLVED**

**Fix Applied:** Changed `config.ModelConfig.hidden_dim` default from 512 → 256

**Details:**
- Text embeddings: (batch, seq_len, **256**)
- Image embeddings: (batch, seq_len, **256**)
- Tabular encoder output: (batch, seq_len, **256**) ← NOW FIXED TO 256

**Why This Works:**
- All 3 modalities now have identical dimensions (256D)
- `torch.stack()` in CrossModalAttentionLayer now succeeds
- No dimension mismatches in forward pass
- Embeddings remain at 256D from offline extraction (FinBERT & ResNet50 projections)
- TabularEncoder final layer explicitly outputs 256D

**Code Change:**
```python
# config.py - ModelConfig
hidden_dim: int = 256  # MUST match embedding dimensions (256D)
```

**Architecture Now Consistent:**
- ✅ Text: 256D → CrossModalAttention: 256D → LSTM: 256D → Head: 256D
- ✅ Image: 256D → CrossModalAttention: 256D → LSTM: 256D → Head: 256D
- ✅ Tabular: 7D → Encoder: 256D → CrossModalAttention: 256D → LSTM: 256D → Head: 256D

---

## Other Limitations

1. **Offline Embeddings:** Cannot fine-tune FinBERT/ResNet50 (frozen by design)
   - *Workaround:* Re-run `extract_features.py` if need different backbone versions

2. **Single Modality Fusion:** No explicit per-modality attention weights
   - *Impact:* Equal importance for text/image/tabular (mean pooling only)
   - *Improvement:* Could add learned gating mechanisms for modality weighting

3. **Temporal Window:** Fixed 24-hour sliding window (cannot capture multi-day trends)
   - *Workaround:* Increase `seq_len` in config (may require tuning accumulation_steps and LR)

4. **Continuous Output:** No explicit constraint to [-100, +100] range
   - *Impact:* Model learns bounds empirically; predictions may exceed range
   - *Improvement:* Add output tagging or constrained regression loss

---

## Future Improvements

- [ ] **Modality-specific attention weights:** Learned importance scores for text/image/tabular
- [ ] **Variable-length sequences:** Adaptive seq_len based on market volatility
- [ ] **Multi-task learning:** Joint prediction of sentiment + price movement + volatility
- [ ] **Uncertainty quantification:** Output confidence intervals, not just point predictions
- [ ] **LoRA fine-tuning:** Lightweight adapters for backbone re-training without full extraction
- [ ] **Custom loss function:** Bounded regression to constrain [-100, +100] range
- [ ] **Dynamic modality fusion:** Learned gating mechanism instead of mean pooling
- [ ] **Temporal ensemble:** Combine predictions from multiple seq_len windows

---

## Implementation Details

### Files Overview
- **`src/data/extract_features.py`** - Offline feature extraction (run once)
- **`src/training/dataset.py`** - Loads pre-extracted embeddings + tabular data
- **`src/training/model.py`** - Lightweight model (TabularEncoder + Attention + LSTM + Head)
- **`src/training/train.py`** - Pure float32 training loop with gradient clipping

### Key Hyperparameters
| Parameter | Value | Reason |
|-----------|-------|--------|
| hidden_dim | 512 | Balanced representational power |
| learning_rate | 1e-4 | Safe for 3.3M params (pure float32) |
| accumulate_steps | 2 | Effective batch 16 from batch 8 |
| grad_clip | 1.0 | Prevents LSTM/Attention explosion |
| warmup_steps | 800 | Stable learning initialization |
| dropout | 0.2 | Regularization across layers |

## References

- **FinBERT:** https://huggingface.co/ProsusAI/finbert
- **ResNet50:** https://pytorch.org/vision/stable/models.html
- **Gradient Clipping:** https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- **Gradient Accumulation:** https://arxiv.org/abs/1711.00141
- **Offline Feature Extraction Pattern:** Common in production ML pipelines for reduced I/O + computational efficiency

---

**Last Modified:** April 15, 2026 (v2.0: Offline + Mixed Precision)  
**Author:** Multimodal Crypto Sentiment Team  
**Key Changes in v2.0:**
- **Offline extraction:** Pre-compute embeddings once, train many times
- **Removed backbones:** FinBERT (109M) and ResNet50 (24M) not in training model
- **Mixed precision:** float16 activations + float32 weights with GradScaler (NOT pure float32)
- **Automatic loss scaling:** GradScaler prevents gradient underflow/overflow
- **Explicit gradient clipping:** L2 norm ≤ 1.0 before optimizer.step()
- **Model size:** Reduced from 6GB (with backbones) to 13MB (offline only)
- **VRAM usage:** ~10-12GB (with batch_size=128, mixed precision)
- **Training speed:** 3-5x faster per epoch vs online encoding
- **Batch size:** Increased from 8 to 128 (AMP enables larger batches)

**v2.0 Benefits:**
✅ One-time feature extraction (amortized cost)
✅ Reproducible embeddings across experiments  
✅ Fast iteration: Change model without re-extracting
✅ Memory efficient: Small model + mixed precision
✅ Numerically stable: Automatic loss scaling + gradient clipping
✅ Scalable: Higher batch size with AMP reduces training time
