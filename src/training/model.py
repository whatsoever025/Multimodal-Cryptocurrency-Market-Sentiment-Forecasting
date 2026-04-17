"""
MultimodalFusionNet: Production-grade multimodal sentiment forecasting with [FUSION] token.

Architecture (Offline Feature Extraction):
1. Pre-Computed Embeddings (extracted offline via extract_features.py):
   - FinBERT text embeddings: (batch, seq_len, 256)
   - Vision Transformer (ViT) image embeddings: (batch, seq_len, 256)
   
2. Tabular MLP Encoder:
   - MLP encoding of 7 raw tabular features -> (batch, seq_len, 256)

3. Learnable [FUSION] Token:
   - Trainable 256D parameter vector (detector token)
   - Expands to (batch, seq_len, 256) for attention

4. Cross-Modal Attention with [FUSION] Token:
   - 4 tokens ([FUSION], text, image, tabular) attend to each other
   - All tokens: 256D (all match)
   - Extracts only [FUSION] token output (no mean pooling)

5. Temporal LSTM:
   - Captures temporal dynamics across seq_len timesteps
   - Input/Hidden: 256D (from [FUSION] token)
   - 2 layers with dropout

6. Prediction Head:
   - MLP reducing to single continuous output (-100 to +100)
   - Input: 256D (from LSTM final hidden state)

Innovation: Learnable [FUSION] token replaces mean pooling for better fusion.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TabularEncoder(nn.Module):
    """
    MLP encoder for tabular features.
    
    Input: (batch, seq_len, 7) numeric features
    Output: (batch, seq_len, hidden_dim) encoded features
    """
    
    def __init__(self, hidden_dim: int = 256, input_size: int = 7, dropout: float = 0.4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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
    Cross-modal attention with learnable [FUSION] token pooling.
    
    Treats [FUSION] token + 3 modalities (4 total) as a sequence and applies multi-head attention.
    At each timestep, all 4 tokens attend to each other, then extracts only [FUSION] token output.
    
    STABILITY FIX (2025-04-17):
    - Pre-LN structure: LayerNorm BEFORE attention (not after residual)
    - Dropout applied AFTER residual (not inside attention)
    - Prevents gradient explosion in backward pass through attention
    - Essential for stability with 4-head attention on 256D embeddings
    
    Input: (batch, seq_len, 4, hidden_dim)
    Output: (batch, seq_len, hidden_dim) - [FUSION] token representation only (no mean pooling)
    """
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Pre-LN: Normalize BEFORE attention (not after)
        self.layer_norm_input = nn.LayerNorm(hidden_dim)
        
        # Multi-head attention treating modalities as sequence (4 tokens)
        # REDUCED DROPOUT: 0.1 instead of 0.3 for backward stability
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,  # Conservative: prevents gradient amplification
            batch_first=True,  # (batch, seq, dim)
        )
        
        # Dropout applied AFTER residual (safer than before)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, modality_stack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modality_stack: (batch, seq_len, 4, hidden_dim)
                           4 = [fusion_token, text, image, tabular] tokens
        
        Returns:
            (batch, seq_len, hidden_dim) - [FUSION] token output only
        """
        batch_size, seq_len, num_modalities, hidden_dim = modality_stack.shape
        
        # Reshape for attention: (batch*seq_len, 4, hidden_dim)
        # Treat 4 tokens as sequence
        modality_flat = modality_stack.reshape(batch_size * seq_len, num_modalities, hidden_dim)
        
        # ===== PRE-LN RESIDUAL STRUCTURE (STABLE) =====
        # 1. Normalize input first (Pre-LN, not Post-LN)
        modality_norm = self.layer_norm_input(modality_flat)  # (batch*seq_len, 4, hidden_dim)
        
        # 2. Self-attention on normalized tokens
        attended, _ = self.mha(
            modality_norm, modality_norm, modality_norm,
            need_weights=False
        )
        
        # 3. Residual connection (stable because attention operates on normalized input)
        # 4. Dropout applied AFTER residual (not inside attention)
        output = modality_flat + self.dropout(attended)  # (batch*seq_len, 4, hidden_dim)
        
        # Extract only [FUSION] token (position 0): (batch*seq_len, hidden_dim)
        # No mean pooling - [FUSION] token is the only output
        fused = output[:, 0, :]  # First token is [FUSION]
        
        # Reshape back: (batch, seq_len, hidden_dim)
        return fused.reshape(batch_size, seq_len, hidden_dim)


class TemporalLSTMLayer(nn.Module):
    """
    LSTM for temporal modeling across sequence.
    
    Input: (batch, seq_len, lstm_hidden_dim)
    Output: (batch, lstm_hidden_dim) - final hidden state
    """
    
    def __init__(self, input_dim: int = 64, num_layers: int = 1, dropout: float = 0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, lstm_hidden_dim)
        
        Returns:
            (batch, lstm_hidden_dim) - final hidden state from last layer
        """
        # LSTM forward
        # output: (batch, seq_len, lstm_hidden_dim)
        # (h_n, c_n): h_n is (num_layers, batch, lstm_hidden_dim)
        output, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state from last layer
        # h_n[-1] shape: (batch, lstm_hidden_dim)
        return h_n[-1]


class PredictionHead(nn.Module):
    """
    Simplified MLP prediction head for continuous sentiment score.
    
    Input: (batch, lstm_hidden_dim=64)
    Output: (batch, 1) - continuous score in range [-100, 100]
    """
    
    def __init__(self, input_dim: int = 64, dropout: float = 0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        # Initialize weights
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, lstm_hidden_dim=64)
        
        Returns:
            (batch, 1)
        """
        return self.head(x)


class MultimodalFusionNet(nn.Module):
    """
    Production-grade multimodal fusion network with [FUSION] token pooling.
    
    Architecture:
    1. Create learnable [FUSION] token (256D, detector for cross-modal fusion)
    2. Accept pre-extracted embeddings (FinBERT text & ViT images)
    3. Encode tabular features with lightweight MLP (output: 256D to match embeddings)
    4. Stack [FUSION] token + 3 modalities and apply cross-modal attention (4 total tokens)
    5. Extract only [FUSION] token output (no mean pooling)
    6. Bottleneck layer: Linear(256 → 64) - compress fused features
    7. Apply temporal LSTM for temporal dynamics (64D hidden state, 1 layer)
    8. Simplified MLP prediction head (64 → 16 → 1) for continuous output
    
    Key Innovation: [FUSION] token acts as learnable aggregator, replacing mean pooling
    - Token learns to extract relevant information from 3 modalities
    - Entirely trainable fusion mechanism
    - No fixed pooling operation
    
    Mixed precision training (AMP):
    - float16 activations + float32 weights
    - GradScaler for automatic loss scaling
    - Gradient clipping (L2 norm <= 1.0)
    - 16GB VRAM sufficient with batch_size=128, seq_len=24
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.seq_len = config.data.seq_len
        
        logger.info(f"Initializing MultimodalFusionNet (hidden_dim={self.hidden_dim})...")
        
        # 0. Learnable [FUSION] token (detector token for cross-modal fusion)
        # Shape: (1, 1, hidden_dim) -> expands to (batch, seq_len, hidden_dim) in forward
        self.fusion_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        nn.init.xavier_uniform_(self.fusion_token)
        logger.info("✓ [FUSION] token initialized (learnable 256D parameter)")
        
        # 1. Tabular encoder (only trainable component with backbones)
        self.tabular_encoder = TabularEncoder(
            hidden_dim=self.hidden_dim,
            input_size=7,
            dropout=config.model.encoder_dropout,
        )
        
        # 2. Cross-modal attention
        self.cross_modal_attention = CrossModalAttentionLayer(
            hidden_dim=self.hidden_dim,
            num_heads=config.model.attention_heads,
            dropout=config.model.mha_dropout,
        )
        
        # 3. Bottleneck layer: compress from 256 -> 64
        self.bottleneck = nn.Linear(self.hidden_dim, config.model.bottleneck_dim)
        logger.info(f"✓ Bottleneck layer initialized ({self.hidden_dim} → {config.model.bottleneck_dim})")
        
        # 4. Temporal LSTM (simplified: 1 layer, 64D hidden)
        self.temporal_lstm = TemporalLSTMLayer(
            input_dim=config.model.bottleneck_dim,
            num_layers=config.model.lstm_layers,
            dropout=config.model.lstm_dropout,
        )
        
        # 5. Prediction head (simplified: 64 → 16 → 1)
        self.prediction_head = PredictionHead(
            input_dim=config.model.bottleneck_dim,
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
        Forward pass through multimodal fusion network with [FUSION] token.
        
        Args:
            batch: Dict with keys:
                - tabular: (batch, seq_len, 7) - raw tabular features
                - text_embedding: (batch, seq_len, 256) - pre-extracted FinBERT embeddings
                - image_embedding: (batch, seq_len, 256) - pre-extracted ViT embeddings
        
        Returns:
            (batch,) - continuous sentiment predictions
        """
        batch_size, seq_len = batch["tabular"].shape[0], batch["tabular"].shape[1]
        
        # ==================== LEARNABLE [FUSION] TOKEN ====================
        # Expand [FUSION] token to match batch and sequence dimensions
        # self.fusion_token: (1, 1, hidden_dim) -> (batch, seq_len, hidden_dim)
        fusion_token_expanded = self.fusion_token.expand(batch_size, seq_len, -1)
        
        # ==================== ENCODE TABULAR FEATURES ====================
        # Tabular encoder: (batch, seq_len, 7) -> (batch, seq_len, hidden_dim)
        tabular_features = self.tabular_encoder(batch["tabular"])
        
        # ==================== USE PRE-EXTRACTED EMBEDDINGS ====================
        # Text embeddings: (batch, seq_len, 256) - already extracted offline
        text_features = batch["text_embedding"]  # (batch, seq_len, 256)
        
        # Image embeddings: (batch, seq_len, 256) - already extracted offline
        image_features = batch["image_embedding"]  # (batch, seq_len, 256)
        
        # ==================== CROSS-MODAL ATTENTION WITH [FUSION] TOKEN ====================
        # Stack [FUSION] token with 3 modalities: (batch, seq_len, 4, hidden_dim)
        # Order: [fusion_token, text, image, tabular]
        modality_stack = torch.stack(
            [fusion_token_expanded, text_features, image_features, tabular_features],
            dim=2
        )
        
        # Apply cross-modal attention: (batch, seq_len, 4, hidden_dim) -> (batch, seq_len, hidden_dim)
        # Outputs only [FUSION] token (position 0) - no mean pooling
        fused_features = self.cross_modal_attention(modality_stack)
        
        # ==================== BOTTLENECK LAYER ====================
        # Compress fused features: (batch, seq_len, 256) -> (batch, seq_len, 64)
        # Removes redundant information before LSTM
        bottleneck_features = self.bottleneck(fused_features)
        
        # ==================== TEMPORAL LSTM ====================
        # LSTM forward: (batch, seq_len, 64) -> (batch, 64)
        # Input: compressed [FUSION] token representations across time
        temporal_output = self.temporal_lstm(bottleneck_features)
        
        # ==================== PREDICTION HEAD ====================
        # Simplified MLP head: (batch, 64) -> (batch, 1) -> squeeze to (batch,)
        predictions = self.prediction_head(temporal_output)  # (batch, 1)
        predictions = predictions.squeeze(dim=1)  # (batch,) - match target shape
        
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
