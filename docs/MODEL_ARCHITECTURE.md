# MultimodalFusionNet Architecture Documentation

**Last Updated:** April 12, 2026  
**Model Version:** 1.0  
**Status:** Production-Ready

---

## Overview

**MultimodalFusionNet** is a production-grade multimodal sentiment forecasting architecture designed for cryptocurrency market sentiment prediction. It fuses three modalities (text, image, tabular) using cross-modal attention and temporal LSTM for continuous sentiment score prediction.

**Purpose:** Forecast cryptocurrency sentiment on a continuous scale (-100 to +100) using combined textual, visual, and numerical market data.

---

## Architecture Components

### 1. **Input Specification**

| Modality | Shape | Type | Description |
|----------|-------|------|-------------|
| **Tabular** | (batch, seq_len, 7) | float32 | Market features: return_1h, volume, funding_rate, fear_greed, gdelt_econ_volume, gdelt_econ_tone, gdelt_conflict_volume |
| **Text** | (batch, seq_len, max_text_length) | int64 | BERT tokenized text (FinBERT) |
| **Text Mask** | (batch, seq_len, max_text_length) | int64 | Attention mask for BERT |
| **Images** | (batch, seq_len, 3, 224, 224) | float32 | Candlestick charts, normalized to [0, 1] |

**seq_len (sequence length):** 24 hours (default)  
**max_text_length:** 512 tokens

---

### 2. **Modality Encoders**

#### A. TimeDistributedTextEncoder
```
Input: (batch, seq_len, max_text_length) token IDs + attention mask
       ↓
FinBERT Backbone (frozen, 12 layers, 768 hidden dim)
       ↓
[CLS] Token Extraction (batch*seq_len, 768)
       ↓
Linear Projection (768 → hidden_dim)  [Xavier initialized]
       ↓
Dropout (0.2)
       ↓
Output: (batch, seq_len, hidden_dim)
```

**Key Properties:**
- Backbone: FinBERT (ProsusAI/finbert)
- Frozen: Yes (requires_grad=False)
- Gradient Checkpointing: Disabled (frozen models don't need it)
- Tokenizer: AutoTokenizer from HuggingFace
- Token Extraction: [CLS] token (index 0)

#### B. TimeDistributedImageEncoder
```
Input: (batch, seq_len, 3, 224, 224) normalized images
       ↓
ResNet50 Backbone (frozen, pretrained on ImageNet1K_V2)
       ↓
Average Pooling → Flatten (batch*seq_len, 2048)
       ↓
Linear Projection (2048 → hidden_dim)  [Xavier initialized]
       ↓
Dropout (0.2)
       ↓
Output: (batch, seq_len, hidden_dim)
```

**Key Properties:**
- Backbone: ResNet50 (ImageNet1K_V2 weights)
- Frozen: Yes (requires_grad=False)
- Input Size: 224×224 RGB images
- Image Normalization: Pixel-wise division by 255.0

#### C. TabularEncoder
```
Input: (batch, seq_len, 7) numeric features
       ↓
Linear (7 → 64) + ReLU + Dropout (0.2)
       ↓
Linear (64 → hidden_dim) + ReLU + Dropout (0.2)
       ↓
Output: (batch, seq_len, hidden_dim)
```

**Key Properties:**
- Architecture: 2-layer MLP
- Activation: ReLU
- Weight Initialization: Xavier uniform
- Trainable: Yes

---

### 3. **Cross-Modal Attention Layer**

```
Input: (batch, seq_len, 3, hidden_dim)  [3 modality stack: text, image, tabular]
       ↓
Reshape to (batch*seq_len, 3, hidden_dim)
       ↓
Multi-Head Self-Attention (3 modalities attend to each other)
       ├─ Heads: 4
       ├─ Dropout: 0.1
       └─ Batch-first: True
       ↓
Residual Connection + LayerNorm
       ↓
Mean Pooling across modality dimension
       ↓
Reshape back to (batch, seq_len, hidden_dim)
       ↓
Output: (batch, seq_len, hidden_dim)  [fused representation]
```

**Key Properties:**
- Attention Type: Multi-Head Self-Attention
- Number of Heads: 4
- Attention Dimension: hidden_dim
- Dropout: 0.1
- Modalities Treated As: Sequence of length 3

---

### 4. **Temporal LSTM Layer**

```
Input: (batch, seq_len, hidden_dim)
       ↓
LSTM Cell (2 layers)
├─ Input Size: hidden_dim
├─ Hidden Size: hidden_dim
├─ Num Layers: 2
├─ Batch First: True
└─ Dropout: 0.2 (between layers)
       ↓
Extract Final Hidden State (batch, hidden_dim)
       ↓
Output: (batch, hidden_dim)
```

**Key Properties:**
- Type: Bidirectional: No
- Layers: 2
- Dropout Between Layers: 0.2
- Output: Final hidden state from last layer

---

### 5. **Prediction Head (MLP)**

```
Input: (batch, hidden_dim)
       ↓
Linear (hidden_dim → 128) + ReLU + Dropout (0.2)
       ↓
Linear (128 → 64) + ReLU + Dropout (0.2)
       ↓
Linear (64 → 1)
       ↓
Output: (batch, 1)  [continuous sentiment score]
```

**Key Properties:**
- Architecture: 3-layer MLP
- Hidden Layers: 128 → 64
- Output Activation: None (raw continuous value)
- Output Range: Unrestricted (model learns to predict [-100, +100])
- Weight Initialization: Xavier uniform

---

## Default Configuration

### DataConfig
```python
asset: "MULTI"              # BTC + ETH combined
seq_len: 24                 # 24-hour sliding window
batch_size: 8               # Per-GPU batch size
max_text_length: 512        # BERT token sequence length
image_size: 224             # ResNet50 input size
shuffle_train: True
num_workers: 0              # Kaggle compatibility (no multiprocessing)
pin_memory: True
prefetch_factor: 2
```

### ModelConfig
```python
hidden_dim: 512             # Encoder output dimension
lstm_layers: 2              # Temporal layers
lstm_dropout: 0.2
attention_heads: 4          # Cross-modal attention heads
mha_dropout: 0.1            # Attention dropout
encoder_dropout: 0.2        # Encoder dropout
head_dropout: 0.2           # Prediction head dropout
grad_clip: 1.0              # Gradient norm clipping
frozen_backbones: True      # Freeze BERT & ResNet50
use_gradient_checkpointing: False  # Disabled (frozen models)
init_weights: True          # Xavier init for projections
```

### TrainingConfig
```python
max_epochs: 60
learning_rate: 1e-5         # Conservative for stability
weight_decay: 1e-5
accumulate_steps: 2         # Gradient accumulation (BS 8 → eff BS 16)
warmup_steps: 800
use_warmup: True
scheduler_type: "cosine"    # Cosine annealing with warmup
```

### OptimizationConfig
```python
mixed_precision: True       # AMP float16
dtype: "float16"
use_scaler: True            # GradScaler
init_scale: 65536.0
growth_factor: 2.0
backoff_factor: 0.5
growth_interval: 2000
```

---

## Model Statistics

### Parameter Counts
- **FinBERT:** ~109M (frozen)
- **ResNet50:** ~24M (frozen)
- **Trainable Projection/Attention/LSTM/Head:** ~1.5M
- **Total:** ~134.5M (only 1.1% trainable)

### Memory Usage (Kaggle 16GB)
- Model: ~2GB
- Batch: ~6-8GB (batch_size=8, seq_len=24)
- Optimizer States: ~2-3GB
- Total: ~12-13GB

---

## Training Pipeline

### Loss Function
```python
MSE Loss: L = (predictions - targets)² / batch_size
```

### Optimizer
```python
AdamW with:
  - Learning Rate: 1e-5
  - Weight Decay: 1e-5 (L2 regularization)
```

### Learning Rate Schedule
```
Warmup Phase (800 steps): Linear increase from 0 → 1e-5
Cosine Annealing Phase: Cosine decay from 1e-5 → 0
Total Steps: num_batches × max_epochs / accumulate_steps
```

### Gradient Management
```
Gradient Accumulation: 2 steps (effective batch size = 16)
Gradient Clipping: L2 norm ≤ 1.0
Mixed Precision: AMP with float16
```

---

## Data Flow Example

**Input:** Batch of 8 samples, seq_len=24, max_text_length=512

```
(8, 24, 512) tokens → FinBERT → (192, 768) → Projection → (8, 24, 512)
(8, 24, 3, 224, 224) images → ResNet50 → (192, 2048) → Projection → (8, 24, 512)
(8, 24, 7) tabular → MLP → (8, 24, 512)
                              ↓
                      Stack modalities
                        (8, 24, 3, 512)
                              ↓
                    Cross-Modal Attention
                        (8, 24, 512)
                              ↓
                         LSTM (2 layers)
                         (8, 512)
                              ↓
                      Prediction Head
                         (8, 1)
                              ↓
                    Sentiment Scores [-100, +100]
```

---

## Special Features

### 1. VRAM Optimization
- Frozen backbones (no gradient computation for 109M + 24M parameters)
- AMP mixed precision (float16 for forward/backward, float32 for loss)
- Gradient accumulation (effective larger batch without VRAM increase)
- No gradient checkpointing on frozen models

### 2. Robustness
- Xavier uniform weight initialization for all trainable layers
- Conservative learning rate (1e-5) to avoid NaN
- Gradient clipping at norm 1.0
- Warmup phase (800 steps) for stable learning

### 3. Reproducibility
- Seed: 42 (set at training start)
- Deterministic: Yes (with CUDA deterministic settings)
- Checkpoint saving: Best model + periodic epochs

### 4. Monitoring
- W&B integration for experiment tracking
- Per-step logging (every 100 steps)
- Per-epoch validation (MSE + MAE metrics)
- Checkpoint auto-save (keep last 3, plus best)

---

## Known Limitations

1. **Frozen Backbones:** No fine-tuning of BERT/ResNet50 (by design for memory)
2. **Single Modality Importance:** No explicit modality weighting
3. **Temporal Window:** Fixed 24-hour window (cannot capture longer trends)
4. **Continuous Output:** No explicit constraint to [-100, +100] range (model learns this)
5. **Training Speed:** BERT inference is slow (~100-200ms per batch)

---

## Future Improvements

- [ ] Modality-specific attention weights
- [ ] Variable-length sequences
- [ ] Multi-task learning (aux predictions)
- [ ] Uncertainty quantification
- [ ] Fine-tunable LoRA adapters for backbones
- [ ] Custom loss function for sentiment bounds

---

## References

- **FinBERT:** https://huggingface.co/ProsusAI/finbert
- **ResNet50:** https://pytorch.org/vision/stable/models.html
- **PyTorch AMP:** https://pytorch.org/docs/stable/amp.html
- **Gradient Accumulation:** https://arxiv.org/abs/1711.00141

---

**Last Modified:** April 12, 2026  
**Author:** Multimodal Crypto Sentiment Team
