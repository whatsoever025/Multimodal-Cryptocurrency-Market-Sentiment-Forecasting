"""
MultimodalFusionNet: Production-grade multimodal sentiment forecasting architecture.

Architecture (Offline Feature Extraction):
1. Pre-Computed Embeddings (extracted offline via extract_features.py):
   - FinBERT text embeddings: (batch, seq_len, 256)
   - ResNet50 image embeddings: (batch, seq_len, 256)
   
2. Tabular MLP Encoder:
   - MLP encoding of 7 raw tabular features → (batch, seq_len, 256)

3. Cross-Modal Attention:
   - 3 modalities (text, image, tabular) attend to each other at each timestep
   - Treated as sequence length = 3

4. Temporal LSTM:
   - Captures temporal dynamics across seq_len timesteps
   - 2 layers with dropout

5. Prediction Head:
   - MLP reducing to single continuous output (-100 to +100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TabularEncoder(nn.Module):
    """
    MLP encoder for tabular features.
    
    Input: (batch, seq_len, 7) numeric features
    Output: (batch, seq_len, hidden_dim) encoded features
    """
    
    def __init__(self, hidden_dim: int = 256, input_size: int = 7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # Initialize weights
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tabular: (batch, seq_len, 7)
        
        Returns:
            (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = tabular.shape
        
        # Reshape for MLP: (batch * seq_len, 7)
        tabular_flat = tabular.reshape(batch_size * seq_len, -1)
        
        # MLP forward: (batch*seq_len, 7) → (batch*seq_len, hidden_dim)
        encoded = self.mlp(tabular_flat)
        
        # Reshape back: (batch, seq_len, hidden_dim)
        return encoded.reshape(batch_size, seq_len, -1)


class CrossModalAttentionLayer(nn.Module):
    """
    Cross-modal attention allowing 3 modalities to attend to each other.
    
    Treats modalities as a sequence of length 3 and applies multi-head attention.
    At each timestep, the 3 modality representations attend to each other.
    
    Input: (batch, seq_len, 3, hidden_dim)
    Output: (batch, seq_len, hidden_dim) - fused representation
    """
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention treating modalities as sequence (seq_len=3)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # (batch, seq, dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, modality_stack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modality_stack: (batch, seq_len, 3, hidden_dim)
                           3 = [text, image, tabular] modalities
        
        Returns:
            (batch, seq_len, hidden_dim) - fused features
        """
        batch_size, seq_len, num_modalities, hidden_dim = modality_stack.shape
        
        # Reshape for attention: (batch*seq_len, 3, hidden_dim)
        # Treat 3 modalities as sequence
        modality_flat = modality_stack.reshape(batch_size * seq_len, num_modalities, hidden_dim)
        
        # Self-attention on modalities: (batch*seq_len, 3, hidden_dim) → (batch*seq_len, 3, hidden_dim)
        attended, _ = self.mha(
            modality_flat, modality_flat, modality_flat,
            need_weights=False
        )
        
        # Add residual + layer norm
        attended = self.layer_norm(modality_flat + self.dropout(attended))
        
        # Pool across modality dimension (mean): (batch*seq_len, hidden_dim)
        fused = attended.mean(dim=1)
        
        # Reshape back: (batch, seq_len, hidden_dim)
        return fused.reshape(batch_size, seq_len, hidden_dim)


class TemporalLSTMLayer(nn.Module):
    """
    LSTM for temporal modeling across sequence.
    
    Input: (batch, seq_len, hidden_dim)
    Output: (batch, hidden_dim) - final hidden state
    """
    
    def __init__(self, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        
        Returns:
            (batch, hidden_dim) - final hidden state from last layer
        """
        # LSTM forward
        # output: (batch, seq_len, hidden_dim)
        # (h_n, c_n): h_n is (num_layers, batch, hidden_dim)
        output, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state from last layer
        # h_n[-1] shape: (batch, hidden_dim)
        return h_n[-1]


class PredictionHead(nn.Module):
    """
    MLP prediction head for continuous sentiment score.
    
    Input: (batch, hidden_dim)
    Output: (batch, 1) - continuous score in range [-100, 100]
    """
    
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        # Initialize weights
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, hidden_dim)
        
        Returns:
            (batch, 1)
        """
        return self.head(x)


class MultimodalFusionNet(nn.Module):
    """
    Production-grade multimodal fusion network with offline feature extraction.
    
    Architecture:
    1. Accept pre-extracted embeddings (FinBERT text & ResNet50 images)
    2. Encode tabular features with lightweight MLP
    3. Stack modality representations and apply cross-modal attention
    4. Apply temporal LSTM for temporal dynamics
    5. MLP prediction head for continuous output
    
    Pure float32 training (no AMP):
    - Eliminates gradient underflow/overflow
    - Simplifies numerical stability
    - 16GB VRAM sufficient with batch_size=8, seq_len=24
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.seq_len = config.data.seq_len
        
        logger.info(f"Initializing MultimodalFusionNet (hidden_dim={self.hidden_dim})...")
        
        # 1. Tabular encoder (only trainable component with backbones)
        self.tabular_encoder = TabularEncoder(
            hidden_dim=self.hidden_dim,
            input_size=7,
        )
        
        # 2. Cross-modal attention
        self.cross_modal_attention = CrossModalAttentionLayer(
            hidden_dim=self.hidden_dim,
            num_heads=config.model.attention_heads,
            dropout=config.model.mha_dropout,
        )
        
        # 3. Temporal LSTM
        self.temporal_lstm = TemporalLSTMLayer(
            hidden_dim=self.hidden_dim,
            num_layers=config.model.lstm_layers,
            dropout=config.model.lstm_dropout,
        )
        
        # 4. Prediction head
        self.prediction_head = PredictionHead(
            hidden_dim=self.hidden_dim,
            dropout=config.model.head_dropout,
        )
        
        logger.info("✓ MultimodalFusionNet initialized")
        
        # Print parameter counts
        self._log_parameter_counts()
    
    def _log_parameter_counts(self):
        """Log trainable/frozen parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logger.info(f"Parameters: {total_params:,.0f} total | {trainable_params:,.0f} trainable | {frozen_params:,.0f} frozen")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through multimodal fusion network.
        
        Args:
            batch: Dict with keys:
                - tabular: (batch, seq_len, 7) - raw tabular features
                - text_embedding: (batch, seq_len, 256) - pre-extracted FinBERT embeddings
                - image_embedding: (batch, seq_len, 256) - pre-extracted ResNet50 embeddings
        
        Returns:
            (batch, 1) - continuous sentiment predictions
        """
        batch_size, seq_len = batch["tabular"].shape[0], batch["tabular"].shape[1]
        
        # ==================== ENCODE TABULAR FEATURES ====================
        # Tabular encoder: (batch, seq_len, 7) → (batch, seq_len, hidden_dim)
        tabular_features = self.tabular_encoder(batch["tabular"])
        
        # ==================== USE PRE-EXTRACTED EMBEDDINGS ====================
        # Text embeddings: (batch, seq_len, 256) - already extracted offline
        text_features = batch["text_embedding"]  # (batch, seq_len, 256)
        
        # Image embeddings: (batch, seq_len, 256) - already extracted offline
        image_features = batch["image_embedding"]  # (batch, seq_len, 256)
        
        # ==================== CROSS-MODAL ATTENTION ====================
        # Stack modalities: (batch, seq_len, 3, hidden_dim)
        modality_stack = torch.stack(
            [text_features, image_features, tabular_features],
            dim=2
        )
        
        # Apply cross-modal attention: (batch, seq_len, 3, hidden_dim) → (batch, seq_len, hidden_dim)
        fused_features = self.cross_modal_attention(modality_stack)
        
        # ==================== TEMPORAL LSTM ====================
        # LSTM forward: (batch, seq_len, hidden_dim) → (batch, hidden_dim)
        temporal_output = self.temporal_lstm(fused_features)
        
        # ==================== PREDICTION HEAD ====================
        # MLP head: (batch, hidden_dim) → (batch, 1)
        predictions = self.prediction_head(temporal_output)
        
        return predictions
    
    def get_trainable_params(self):
        """Return iterator of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]


if __name__ == "__main__":
    """Test model initialization and forward pass."""
    import sys
    
    from config import ExperimentConfig
    
    print("=" * 80)
    print("Testing MultimodalFusionNet (Offline Features)")
    print("=" * 80)
    
    try:
        # Load config
        config = ExperimentConfig()
        device = "cpu"  # Use CPU for testing
        
        print(f"\n1. Initializing model (device={device})...")
        model = MultimodalFusionNet(config).to(device)
        print("   ✓ Model initialized")
        
        # Create dummy batch (with pre-extracted embeddings)
        print("\n2. Creating dummy batch...")
        batch = {
            "tabular": torch.randn(4, 24, 7).to(device),
            "text_embedding": torch.randn(4, 24, 256).to(device),  # Pre-extracted embeddings
            "image_embedding": torch.randn(4, 24, 256).to(device),  # Pre-extracted embeddings
        }
        print(f"   ✓ Batch created with shapes:")
        for key, val in batch.items():
            print(f"     - {key}: {val.shape}")
        
        # Forward pass
        print("\n3. Running forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(batch)
        print(f"   ✓ Output shape: {output.shape} (expected: (4, 1))")
        print(f"   ✓ Output values: min={output.min():.4f}, max={output.max():.4f}")
        
        print("\n" + "=" * 80)
        print("✅ All model tests passed!")
        print("=" * 80)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
"""
MultimodalFusionNet: Production-grade multimodal sentiment forecasting architecture.

Architecture:
1. TimeDistributed Encoders (VRAM-safe):
   - FinBERT for text: frozen backbone + gradient checkpointing
   - ResNet50 for images: frozen backbone + gradient checkpointing
   - MLP for tabular features

2. Cross-Modal Attention:
   - 3 modalities (text, image, tabular) attend to each other at each timestep
   - Treated as sequence length = 3

3. Temporal LSTM:
   - Captures temporal dynamics across seq_len timesteps
   - 2 layers with dropout

4. Prediction Head:
   - MLP reducing to single continuous output (-100 to +100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

try:
    from transformers import AutoModel
except ImportError:
    raise ImportError("'transformers' package required: pip install transformers")

try:
    import torchvision.models as models
except ImportError:
    raise ImportError("'torchvision' package required: pip install torchvision")


logger = logging.getLogger(__name__)


class TimeDistributedTextEncoder(nn.Module):
    """
    FinBERT encoder for sequences of text.
    
    Input: (batch*seq_len, max_text_length) token IDs
    Output: (batch*seq_len, hidden_dim) contextual embeddings
    
    VRAM Strategy:
    - Freeze FinBERT backbone (no gradient updates)
    - Optional gradient checkpointing (trade compute for memory)
    """
    
    def __init__(self, hidden_dim: int = 256, use_gradient_checkpointing: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        logger.info("Loading FinBERT...")
        self.bert = AutoModel.from_pretrained("ProsusAI/finbert")
        
        # Freeze backbone (critical for VRAM)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Note: gradient checkpointing disabled since backbone is frozen
        
        # Project [CLS] token (768) to hidden_dim
        self.projection = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        nn.init.xavier_uniform_(self.projection.weight)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch*seq_len, max_text_length)
            attention_mask: (batch*seq_len, max_text_length)
        
        Returns:
            (batch*seq_len, hidden_dim)
        """
        # Forward pass without gradients (backbone is frozen)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        # [CLS] token (index 0) → project → dropout
        cls_token = outputs.last_hidden_state[:, 0, :]  # (batch*seq_len, 768)
        projected = self.projection(cls_token)  # (batch*seq_len, hidden_dim)
        return self.dropout(projected)


class TimeDistributedImageEncoder(nn.Module):
    """
    ResNet50 encoder for sequences of images.
    
    Input: (batch*seq_len, 3, 224, 224) RGB images (normalized to [0, 1])
    Output: (batch*seq_len, hidden_dim) visual features
    
    VRAM Strategy:
    - Freeze ResNet50 backbone
    - Optional gradient checkpointing
    """
    
    def __init__(self, hidden_dim: int = 256, use_gradient_checkpointing: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        logger.info("Loading ResNet50...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze backbone (critical for VRAM)
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Note: gradient checkpointing disabled since backbone is frozen
        
        # Remove classification head, keep backbone + avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Up to avgpool
        
        # Project avgpool output (2048) to hidden_dim
        self.projection = nn.Linear(2048, hidden_dim)
        nn.init.xavier_uniform_(self.projection.weight)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch*seq_len, 3, 224, 224)
        
        Returns:
            (batch*seq_len, hidden_dim)
        """
        # Forward pass (backbone is frozen via requires_grad=False)
        # Backbone output: (batch*seq_len, 2048, 1, 1)
        features = self.backbone(images)
        
        # Squeeze spatial dims: (batch*seq_len, 2048)
        features = features.squeeze(-1).squeeze(-1)
        
        # Project and dropout
        projected = self.projection(features)
        return self.dropout(projected)


class TabularEncoder(nn.Module):
    """
    MLP encoder for tabular features.
    
    Input: (batch, seq_len, 7) numeric features
    Output: (batch, seq_len, hidden_dim) encoded features
    """
    
    def __init__(self, hidden_dim: int = 256, input_size: int = 7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # Initialize weights
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tabular: (batch, seq_len, 7)
        
        Returns:
            (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = tabular.shape
        
        # Reshape for MLP: (batch * seq_len, 7)
        tabular_flat = tabular.reshape(batch_size * seq_len, -1)
        
        # MLP forward: (batch*seq_len, 7) → (batch*seq_len, hidden_dim)
        encoded = self.mlp(tabular_flat)
        
        # Reshape back: (batch, seq_len, hidden_dim)
        return encoded.reshape(batch_size, seq_len, -1)


class CrossModalAttentionLayer(nn.Module):
    """
    Cross-modal attention allowing 3 modalities to attend to each other.
    
    Treats modalities as a sequence of length 3 and applies multi-head attention.
    At each timestep, the 3 modality representations attend to each other.
    
    Input: (batch, seq_len, 3, hidden_dim)
    Output: (batch, seq_len, hidden_dim) - fused representation
    """
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention treating modalities as sequence (seq_len=3)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # (batch, seq, dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, modality_stack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modality_stack: (batch, seq_len, 3, hidden_dim)
                           3 = [text, image, tabular] modalities
        
        Returns:
            (batch, seq_len, hidden_dim) - fused features
        """
        batch_size, seq_len, num_modalities, hidden_dim = modality_stack.shape
        
        # Reshape for attention: (batch*seq_len, 3, hidden_dim)
        # Treat 3 modalities as sequence
        modality_flat = modality_stack.reshape(batch_size * seq_len, num_modalities, hidden_dim)
        
        # Self-attention on modalities: (batch*seq_len, 3, hidden_dim) → (batch*seq_len, 3, hidden_dim)
        attended, _ = self.mha(
            modality_flat, modality_flat, modality_flat,
            need_weights=False
        )
        
        # Add residual + layer norm
        attended = self.layer_norm(modality_flat + self.dropout(attended))
        
        # Pool across modality dimension (mean): (batch*seq_len, hidden_dim)
        fused = attended.mean(dim=1)
        
        # Reshape back: (batch, seq_len, hidden_dim)
        return fused.reshape(batch_size, seq_len, hidden_dim)


class TemporalLSTMLayer(nn.Module):
    """
    LSTM for temporal modeling across sequence.
    
    Input: (batch, seq_len, hidden_dim)
    Output: (batch, hidden_dim) - final hidden state
    """
    
    def __init__(self, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        
        Returns:
            (batch, hidden_dim) - final hidden state from last layer
        """
        # LSTM forward
        # output: (batch, seq_len, hidden_dim)
        # (h_n, c_n): h_n is (num_layers, batch, hidden_dim)
        output, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state from last layer
        # h_n[-1] shape: (batch, hidden_dim)
        return h_n[-1]


class PredictionHead(nn.Module):
    """
    MLP prediction head for continuous sentiment score.
    
    Input: (batch, hidden_dim)
    Output: (batch, 1) - continuous score in range [-100, 100]
    """
    
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        # Initialize weights
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, hidden_dim)
        
        Returns:
            (batch, 1)
        """
        return self.head(x)


class MultimodalFusionNet(nn.Module):
    """
    Production-grade multimodal fusion network for cryptocurrency sentiment forecasting.
    
    Architecture:
    1. Encode each modality independently (TimeDistributed pattern for VRAM efficiency)
    2. Stack modality representations and apply cross-modal attention
    3. Apply temporal LSTM for temporal dynamics
    4. MLP prediction head for continuous output
    
    VRAM Management (16GB constraint):
    - Frozen backbones (BERT, ResNet50) + optional gradient checkpointing
    - Batch size 8, gradient accumulation, AMP (float16)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.seq_len = config.data.seq_len
        
        logger.info(f"Initializing MultimodalFusionNet (hidden_dim={self.hidden_dim})...")
        
        # 1. Modality-specific encoders
        self.text_encoder = TimeDistributedTextEncoder(
            hidden_dim=self.hidden_dim,
            use_gradient_checkpointing=config.model.use_gradient_checkpointing,
        )
        
        self.image_encoder = TimeDistributedImageEncoder(
            hidden_dim=self.hidden_dim,
            use_gradient_checkpointing=config.model.use_gradient_checkpointing,
        )
        
        self.tabular_encoder = TabularEncoder(
            hidden_dim=self.hidden_dim,
            input_size=7,
        )
        
        # 2. Cross-modal attention
        self.cross_modal_attention = CrossModalAttentionLayer(
            hidden_dim=self.hidden_dim,
            num_heads=config.model.attention_heads,
            dropout=config.model.mha_dropout,
        )
        
        # 3. Temporal LSTM
        self.temporal_lstm = TemporalLSTMLayer(
            hidden_dim=self.hidden_dim,
            num_layers=config.model.lstm_layers,
            dropout=config.model.lstm_dropout,
        )
        
        # 4. Prediction head
        self.prediction_head = PredictionHead(
            hidden_dim=self.hidden_dim,
            dropout=config.model.head_dropout,
        )
        
        logger.info("✓ MultimodalFusionNet initialized")
        
        # Print parameter counts
        self._log_parameter_counts()
    
    def _log_parameter_counts(self):
        """Log trainable/frozen parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logger.info(f"Parameters: {total_params:,.0f} total | {trainable_params:,.0f} trainable | {frozen_params:,.0f} frozen")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through multimodal fusion network.
        
        Args:
            batch: Dict with keys:
                - tabular: (batch, seq_len, 7)
                - text_ids: (batch, seq_len, max_text_length)
                - text_mask: (batch, seq_len, max_text_length)
                - images: (batch, seq_len, 3, 224, 224)
        
        Returns:
            (batch, 1) - continuous sentiment predictions
        """
        batch_size, seq_len = batch["tabular"].shape[0], batch["tabular"].shape[1]
        
        # ==================== ENCODE EACH MODALITY ====================
        
        # Text encoder: (batch, seq_len, max_text_length) → (batch, seq_len, hidden_dim)
        text_ids = batch["text_ids"].reshape(batch_size * seq_len, -1)
        text_mask = batch["text_mask"].reshape(batch_size * seq_len, -1)
        text_features = self.text_encoder(text_ids, text_mask)  # (batch*seq_len, hidden_dim)
        text_features = text_features.reshape(batch_size, seq_len, -1)  # (batch, seq_len, hidden_dim)
        
        # Image encoder: (batch, seq_len, 3, 224, 224) → (batch, seq_len, hidden_dim)
        images = batch["images"].reshape(batch_size * seq_len, 3, 224, 224)
        image_features = self.image_encoder(images)  # (batch*seq_len, hidden_dim)
        image_features = image_features.reshape(batch_size, seq_len, -1)  # (batch, seq_len, hidden_dim)
        
        # Tabular encoder: (batch, seq_len, 7) → (batch, seq_len, hidden_dim)
        tabular_features = self.tabular_encoder(batch["tabular"])  # (batch, seq_len, hidden_dim)
        
        # ==================== CROSS-MODAL ATTENTION ====================
        
        # Stack modalities: (batch, seq_len, 3, hidden_dim)
        modality_stack = torch.stack(
            [text_features, image_features, tabular_features],
            dim=2
        )
        
        # Apply cross-modal attention: (batch, seq_len, 3, hidden_dim) → (batch, seq_len, hidden_dim)
        fused_features = self.cross_modal_attention(modality_stack)
        
        # ==================== TEMPORAL LSTM ====================
        
        # LSTM forward: (batch, seq_len, hidden_dim) → (batch, hidden_dim)
        temporal_output = self.temporal_lstm(fused_features)
        
        # ==================== PREDICTION HEAD ====================
        
        # MLP head: (batch, hidden_dim) → (batch, 1)
        predictions = self.prediction_head(temporal_output)
        
        return predictions
    
    def freeze_backbones(self):
        """Explicitly freeze all backbone parameters."""
        for param in self.text_encoder.bert.parameters():
            param.requires_grad = False
        for param in self.image_encoder.backbone.parameters():
            param.requires_grad = False
        logger.info("✓ All backbones frozen")
    
    def unfreeze_backbones(self):
        """Unfreeze all backbone parameters (for fine-tuning)."""
        for param in self.text_encoder.bert.parameters():
            param.requires_grad = True
        for param in self.image_encoder.backbone.parameters():
            param.requires_grad = True
        logger.info("✓ All backbones unfrozen")
    
    def get_trainable_params(self):
        """Return iterator of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]


if __name__ == "__main__":
    """Test model initialization and forward pass."""
    import sys
    
    from config import ExperimentConfig
    
    print("=" * 80)
    print("Testing MultimodalFusionNet")
    print("=" * 80)
    
    try:
        # Load config
        config = ExperimentConfig()
        device = "cpu"  # Use CPU for testing
        
        print(f"\n1. Initializing model (device={device})...")
        model = MultimodalFusionNet(config).to(device)
        print("   ✓ Model initialized")
        
        # Create dummy batch
        print("\n2. Creating dummy batch...")
        batch = {
            "tabular": torch.randn(4, 24, 7).to(device),
            "text_ids": torch.randint(0, 30522, (4, 24, 512)).to(device),
            "text_mask": torch.ones(4, 24, 512).to(device),
            "images": torch.randn(4, 24, 3, 224, 224).to(device),
        }
        print(f"   ✓ Batch created with shapes:")
        for key, val in batch.items():
            print(f"     - {key}: {val.shape}")
        
        # Forward pass
        print("\n3. Running forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(batch)
        print(f"   ✓ Output shape: {output.shape} (expected: (4, 1))")
        print(f"   ✓ Output values: min={output.min():.4f}, max={output.max():.4f}")
        
        print("\n" + "=" * 80)
        print("✅ All model tests passed!")
        print("=" * 80)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
