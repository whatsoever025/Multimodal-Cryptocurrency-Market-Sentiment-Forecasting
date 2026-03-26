# Hugging Face Dataset Guide: Multimodal Cryptocurrency Market Sentiment

## 📊 Dataset Overview

Two curated datasets are available on Hugging Face Hub for cryptocurrency sentiment forecasting:

| Dataset | URL | Size | Assets | Splits | Purpose |
|---------|-----|------|--------|--------|---------|
| **BTC Sentiment** | [khanh252004/multimodal_crypto_sentiment_btc](https://huggingface.co/datasets/khanh252004/multimodal_crypto_sentiment_btc) | 44,429 rows | Bitcoin (BTCUSDT) | train / validation / test_in_domain | Bitcoin sentiment prediction |
| **ETH Sentiment** | [khanh252004/multimodal_crypto_sentiment_eth](https://huggingface.co/datasets/khanh252004/multimodal_crypto_sentiment_eth) | 44,429 rows | Ethereum (ETHUSDT) | train / validation / test_in_domain | Ethereum sentiment prediction |

**v3 Release:** March 26, 2026

---

## 🎯 Dataset Characteristics

### Multimodal Structure (10 Features + Target)

Each row contains **11 columns** combining market data, sentiment, and visuals:

#### 1. **Temporal Anchor (1 field)**
```python
timestamp: datetime  # UTC hourly (2020-01-02 21:00 → 2025-01-30 00:00)
```

#### 2. **Tabular Features (7 fields)** - For LSTM/MLP Encoders
```python
# Endogenous (market-driven)
return_1h: float  # Hourly % price change (0.5% = 0.5)

# Trading Activity
volume: float     # Hourly trading volume (asset units)

# Derivative Sentiment
funding_rate: float  # Perpetual funding rate (0.0001 = 0.01%, 8-hour intervals forward-filled)

# Macro/Market Sentiment
fear_greed_value: int  # Fear & Greed Index (0-100, daily forward-filled)

# Exogenous News Events (GDELT)
gdelt_econ_volume: int    # # economy/inflation articles (0-500)
gdelt_econ_tone: float    # Average sentiment (-100 to +100)
gdelt_conflict_volume: int  # # geopolitical/conflict articles (0-100)
```

#### 3. **Textual Feature (1 field)** - For BERT/Transformer Encoders
```python
text_content: string  # Hourly CoinDesk news aggregated with [SEP] separator
                      # Empty hours: "[NO_EVENT] market is quiet"
                      # Avg length: ~2,000 tokens
```

#### 4. **Visual Feature (1 field)** - For CNN/ViT Encoders
```python
image_path: PIL.Image  # 224×224 PNG candlestick chart
                       # Contains: OHLC bars + MA7 (blue) + MA25 (red) + RSI(14) + MACD
```

#### 5. **Target Label (1 field)**
```python
target_score: float  # Continuous sentiment (-100 to +100)
                     # Formula: tanh(future_returns / (1.5 * volatility)) * 100
                     # Horizon: 24 hours ahead
```

---

## 📥 Loading the Datasets

### Quick Start

```python
from datasets import load_dataset

# Load BTC dataset
dataset_btc = load_dataset("khanh252004/multimodal_crypto_sentiment_btc")

# Access splits
train_data = dataset_btc["train"]        # 31,133 rows
val_data = dataset_btc["validation"]    # 6,671 rows
test_data = dataset_btc["test_in_domain"]  # 6,625 rows

# Inspect a sample
sample = train_data[0]
print(sample.keys())
# Output: ['timestamp', 'return_1h', 'volume', 'funding_rate', 'fear_greed_value',
#          'gdelt_econ_volume', 'gdelt_econ_tone', 'gdelt_conflict_volume',
#          'text_content', 'image_path', 'target_score']
```

### Accessing Different Modalities

```python
from datasets import load_dataset
import pandas as pd
from PIL import Image

dataset = load_dataset("khanh252004/multimodal_crypto_sentiment_btc")

# --- TABULAR DATA (7 features) ---
sample = dataset["train"][0]
price_return = sample["return_1h"]           # float: -0.523 (%)
volume = sample["volume"]                    # float: 2211.0
funding = sample["funding_rate"]             # float: 0.0001
sentiment = sample["fear_greed_value"]       # int: 65
econ_news_count = sample["gdelt_econ_volume"] # int: 45
econ_sentiment = sample["gdelt_econ_tone"]   # float: -12.3
conflict_count = sample["gdelt_conflict_volume"]  # int: 8

# Create DataFrame for ML
df_tabular = dataset["train"].to_pandas()[
    ['timestamp', 'return_1h', 'volume', 'funding_rate', 
     'fear_greed_value', 'gdelt_econ_volume', 'gdelt_econ_tone', 
     'gdelt_conflict_volume']
]

# --- TEXTUAL DATA (1 feature) ---
news_text = sample["text_content"]  # str: "Bitcoin rallies... [SEP] Fed decision..."

# --- VISUAL DATA (1 feature) ---
chart_image = sample["image_path"]  # PIL.Image: 224x224 PNG
# Already loaded as PIL Image, ready for CV models

# --- TARGET ---
target = sample["target_score"]  # float: 45.2 (sentiment label)
```

### Batch Processing

```python
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

dataset = load_dataset("khanh252004/multimodal_crypto_sentiment_btc")

def collate_batch(batch):
    """Custom collator for multimodal data."""
    timestamps = [x["timestamp"] for x in batch]
    
    # Tabular features (7 cols) → tensor (B, 7)
    tabular = torch.tensor([
        [x["return_1h"], x["volume"], x["funding_rate"], 
         x["fear_greed_value"], x["gdelt_econ_volume"], 
         x["gdelt_econ_tone"], x["gdelt_conflict_volume"]]
        for x in batch
    ], dtype=torch.float32)
    
    # Text → list of strings (encode with BERT tokenizer later)
    texts = [x["text_content"] for x in batch]
    
    # Images → tensor (B, 3, 224, 224)
    images = torch.stack([
        torch.tensor(np.array(x["image_path"])).permute(2, 0, 1) / 255.0
        for x in batch
    ])
    
    # Targets → tensor (B,)
    targets = torch.tensor([x["target_score"] for x in batch], dtype=torch.float32)
    
    return {
        "timestamps": timestamps,
        "tabular": tabular,
        "texts": texts,
        "images": images,
        "targets": targets
    }

train_loader = DataLoader(
    dataset["train"],
    batch_size=32,
    shuffle=True,
    collate_fn=collate_batch
)

for batch in train_loader:
    print(batch["tabular"].shape)  # (32, 7)
    print(batch["images"].shape)   # (32, 3, 224, 224)
    print(batch["targets"].shape)  # (32,)
    break
```

---

## 🏗️ Architecture Examples

### 1. **Multimodal Fusion Model**

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torchvision.models as models

class MultimodalSentimentModel(nn.Module):
    """Fuses LSTM (tabular) + BERT (text) + ResNet (images)."""
    
    def __init__(self):
        super().__init__()
        
        # Branch 1: Tabular encoder (LSTM)
        self.lstm = nn.LSTM(input_size=7, hidden_size=64, num_layers=2, batch_first=True)
        
        # Branch 2: Text encoder (BERT)
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_fc = nn.Linear(768, 64)
        
        # Branch 3: Visual encoder (ResNet50)
        resnet = models.resnet50(pretrained=True)
        self.visual = nn.Sequential(*list(resnet.children())[:-1])  # Remove classification head
        self.visual_fc = nn.Linear(2048, 64)
        
        # Fusion: Concatenate all branches → MLP head
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Continuous output (-100 to +100)
        )
    
    def forward(self, tabular, text_embeddings, images):
        # Tabular: (B, 1, 7) → (B, 64)
        lstm_out, _ = self.lstm(tabular.unsqueeze(1))
        tabular_feat = lstm_out[:, -1, :]  # Last hidden state
        
        # Text: embeddings from BERT → (B, 64)
        text_feat = self.text_fc(text_embeddings)
        
        # Visual: ResNet50 → (B, 2048) → (B, 64)
        visual_feat = self.visual(images).squeeze(-1).squeeze(-1)
        visual_feat = self.visual_fc(visual_feat)
        
        # Concatenate all modalities
        combined = torch.cat([tabular_feat, text_feat, visual_feat], dim=1)  # (B, 192)
        
        # Final prediction
        output = self.fusion(combined)
        return output

# Training loop sketch
model = MultimodalSentimentModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()  # Regression loss

for epoch in range(10):
    for batch in train_loader:
        tabular = batch["tabular"]  # (B, 7)
        images = batch["images"]    # (B, 3, 224, 224)
        targets = batch["targets"]  # (B,)
        
        # Tokenize text with BERT
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        text_tokens = tokenizer(batch["texts"], return_tensors="pt", padding=True)
        text_embeddings = model.bert(**text_tokens).pooler_output  # (B, 768)
        
        # Forward pass
        predictions = model(tabular, text_embeddings, images)  # (B, 1)
        loss = criterion(predictions.squeeze(), targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 2. **Text-Only Model (LLM Fine-tuning)**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, Trainer, TrainingArguments

# Convert dataset to instruction-following format
def format_for_llm(dataset):
    def format_fn(row):
        prompt = f"""Analyze cryptocurrency market sentiment.

Market Data:
- Price return (1h): {row['return_1h']:.2f}%
- Volume: {row['volume']:,.0f}
- Funding rate: {row['funding_rate']:.4f}
- Fear/Greed: {row['fear_greed_value']}/100

Macro Context:
- Economy news: {row['gdelt_econ_volume']} articles (tone: {row['gdelt_econ_tone']:.1f})
- Conflict news: {row['gdelt_conflict_volume']} articles

News:
{row['text_content'][:500]}

Predict 24-hour sentiment (-100 to +100)."""
        
        answer = f"{row['target_score']:.1f}"
        
        return {"text": prompt + f"\n\nAnswer: {answer}"}
    
    return dataset.map(format_fn)

# Load and format
dataset = load_dataset("khanh252004/multimodal_crypto_sentiment_btc")
formatted = format_for_llm(dataset["train"])

# Fine-tune GPT-2 or Llama
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./crypto_sentiment_lm",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=1000,
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=formatted,
)

trainer.train()
```

### 3. **Image-Only Model (Technical Analysis)**

```python
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

dataset = load_dataset("khanh252004/multimodal_crypto_sentiment_btc")

# ResNet50 for image-based sentiment
class ImageSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.head = nn.Linear(2048, 1)
    
    def forward(self, x):
        features = self.backbone(x).squeeze(-1).squeeze(-1)
        output = self.head(features)
        return output

model = ImageSentimentModel()

# Training
train_images = [
    transforms.ToTensor()(sample["image_path"])
    for sample in dataset["train"]
]
train_targets = torch.tensor(
    [sample["target_score"] for sample in dataset["train"]]
)

# ... Standard training loop ...
```

---

## 📊 Dataset Statistics

### Size & Splits

| Split | Rows | % of Total | Date Range |
|-------|------|-----------|------------|
| train | 31,133 | 70.0% | 2020-01-02 21:00 → 2023-07-24 00:00 |
| validation | 6,671 | 15.0% | 2023-07-25 01:00 → 2024-04-27 23:00 |
| test_in_domain | 6,625 | 14.9% | 2024-04-29 00:00 → 2025-01-30 00:00 |
| **Total** | **44,429** | **100%** | **5.25 years** |

**Note:** 48 rows removed between splits (24-hour embargo at each boundary) to prevent look-ahead bias.

### Feature Statistics (BTC Train Split)

| Feature | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| return_1h (%) | -8.92 | 8.45 | 0.11 | 0.89 |
| volume | 12.0 | 523,891 | 5,842 | 18,423 |
| funding_rate | -0.0168 | 0.0206 | 0.00041 | 0.0031 |
| fear_greed_value | 11 | 97 | 48 | 22 |
| gdelt_econ_volume | 0 | 487 | 27 | 45 |
| gdelt_econ_tone | -97.2 | 82.5 | -8.3 | 18.7 |
| gdelt_conflict_volume | 0 | 156 | 8 | 15 |
| target_score | -100.0 | 100.0 | 5.09 | 31.2 |

### Text Statistics

| Metric | Value |
|--------|-------|
| Total unique hours with news | 37,688 |
| Hours with [NO_EVENT] placeholder | 7,868 |
| Avg tokens per hour | ~2,000 |
| Max tokens per hour | ~15,000 |
| News sources | CoinDesk |

### Image Statistics

| Metric | BTC | ETH |
|--------|-----|-----|
| Total images | 44,477 | 44,477 |
| Missing images (dropped) | 44 | 44 |
| Resolution | 224×224 px | 224×224 px |
| Format | PNG | PNG |
| Indicators | MA7, MA25, RSI, MACD | MA7, MA25, RSI, MACD |

---

## 🎓 Use Cases

### 1. **Multimodal Sentiment Forecasting**
Predict continuous 24-hour sentiment using all 10 features:
- **Input:** return_1h, volume, funding_rate, fear_greed_value, GDELT (3 fields), text, image
- **Output:** Continuous score (-100 to +100)
- **Architecture:** Multi-branch encoder fusion (LSTM + BERT + CNN) + MLP head
- **Recommended:** Used for research publications, production models

### 2. **Text-Only LLM Fine-tuning**
Fine-tune a language model for instruction-following:
- **Input:** Formatted prompt with market data + news text
- **Output:** Sentiment score prediction
- **Use case:** Reasoning-based predictions, interpretability
- **Recommended:** GPT-2, Llama, or Mistral models

### 3. **Image-Based Technical Analysis**
Train a CNN to predict sentiment from candlestick charts:
- **Input:** 224×224 candlestick chart images
- **Output:** Continuous sentiment score
- **Use case:** Pure technical analysis, ablation studies
- **Recommended:** ResNet, EfficientNet, Vision Transformer

### 4. **Tabular-Only Models**
Use only market data signals (7 features):
- **Input:** return_1h, volume, funding, fear/greed, GDELT (3 fields)
- **Output:** Continuous score
- **Use case:** Baseline models, low-latency inference
- **Recommended:** XGBoost, LightGBM, LSTM

### 5. **Multimodal Classification**
Convert to classification task (Bullish/Neutral/Bearish):
- **Thresholds:** target_score ≤ -20 (Bearish), -20 ≤ target_score ≤ 20 (Neutral), ≥ 20 (Bullish)
- **Use case:** Risk classification, automated trading signals
- **Recommended:** Any of the above architectures with cross-entropy loss

---

## 🔍 Data Quality & Validation

### Embargo Rule (Prevents Look-Ahead Bias)

```
Timeline visualization:
|-- Train (31,133 rows) --|[24h embargo]|-- Val (6,671) --|[24h embargo]|-- Test (6,625) --|
     2020-01 to 2023-07   | 2023-07-24   | 2023-07-25-2024-04 | 2024-04-28  | 2024-04-29-2025-01

Key: The 24-hour embargo before each split ensures that future price info (target calculation
     uses 24-hour forward returns) does not leak into training on past data.
```

### Image Validation

- ✅ All 44,477 rows per asset have corresponding 224×224 PNG images
- ✅ Images generated from official OHLCV data (Binance Vision)
- ✅ Technical indicators (MA7/MA25/RSI/MACD) automatically computed
- ❌ 44 rows per asset dropped due to missing images (pre-aligned)

### Text Coverage

- ✅ 84.9% of hours have CoinDesk news articles
- ✅ 15.1% of hours filled with `[NO_EVENT] market is quiet` placeholder
- ✅ Hourly aggregation handles multiple articles per hour

### No Missing Values in Final Dataset

All rows are complete across all 11 columns (after alignment/validation):
- NaN values in funding_rate, fear_greed_value filled via forward-fill
- GDELT fields use 0 for missing hours (indicates no news)
- target_score is continuous without NaNs (dropped rows with forward >24h)

---

## 📝 Citation

If using these datasets, please cite:

```bibtex
@dataset{crypto_sentiment_btc_eth_v3,
  title={Multimodal Cryptocurrency Market Sentiment Dataset (v3)},
  author={Khanh252004},
  year={2026},
  month={March},
  url={https://huggingface.co/datasets/khanh252004/multimodal_crypto_sentiment_btc},
  note={BTC & ETH datasets with 10 features + continuous sentiment target}
}
```

---

## 📚 References

**Data Sources:**
- Binance Vision API: OHLCV candles, funding rates
- HuggingFace Hub: `maryamfakhari/crypto-news-coindesk-2020-2025` (229,172 articles)
- Alternative.me: Fear & Greed Index (daily, 8+ years)
- GDELT v2.1: Global macroeconomic & geopolitical news (via BigQuery)
- Custom generation: 224×224 candlestick charts with MA7/MA25/RSI/MACD

**Related Papers:**
- Multimodal learning for finance: [...your papers...]
- Sentiment analysis in crypto: [...your papers...]
- Technical analysis with deep learning: [...your papers...]

---

**Last Updated:** March 26, 2026  
**v3 Release Date:** March 26, 2026  
**Status:** ✅ Production-ready, both datasets (BTC/ETH) live on HF Hub
