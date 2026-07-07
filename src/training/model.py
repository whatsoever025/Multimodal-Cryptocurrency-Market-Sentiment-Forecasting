"""
MultimodalFusionNet (MFN): intermediate-fusion architecture for multimodal forecasting.

Architecture overview:
  1. Pre-extracted embeddings: FinBERT text (batch, seq_len, 256),
     ViT image (batch, seq_len, 256).
  2. TabularEncoder MLP: (batch, seq_len, 7+16) → (batch, seq_len, 256).
  3. Learnable [FUSION] token (256D) acts as a cross-modal aggregator.
  4. CrossModalAttentionLayer: 4-token self-attention ([FUSION], text, image, tabular)
     — only the [FUSION] token output is retained.
  5. Bottleneck Linear: 256 → 64.
  6. TemporalLSTM: (batch, seq_len, 64) → (batch, 64) final hidden state.
  7. PredictionHead: 64 → 1 per target (single or multi-target mode).
"""

import torch
import torch.nn as nn
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class TabularEncoder(nn.Module):
    """
    MLP encoder for tabular features.

    Input:  (batch, seq_len, input_size)  — raw tabular + 16-dim asset embedding
    Output: (batch, seq_len, hidden_dim)
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
            tabular: (batch, seq_len, input_size)
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
    Single-layer cross-modal attention with a learnable [FUSION] token.

    Treats the four modality tokens ([FUSION], text, image, tabular) as a
    4-element sequence and applies multi-head self-attention. Only the [FUSION]
    token output (position 0) is returned — no mean pooling.

    Uses Pre-LN structure (LayerNorm before attention) and zero dropout inside
    the attention backward path to avoid NaN gradients in scaled dot-product
    attention.

    Input:  (batch, seq_len, 4, hidden_dim)
    Output: (batch, seq_len, hidden_dim)  — [FUSION] token only
    """
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Pre-LN: Normalize BEFORE attention (not after)
        self.layer_norm_input = nn.LayerNorm(hidden_dim)
        
        # Multi-head attention treating modalities as sequence (4 tokens)
        # CRITICAL: dropout=0 inside attention (no dropout in backward pass of attention)
        # Dropout applied after residual instead (numerically safer)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,  # DISABLED: No dropout inside attention backward to prevent NaN
            batch_first=True,  # (batch, seq, dim)
        )
        
        # Dropout applied AFTER residual (safe, clean gradient flow)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, modality_stack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modality_stack: (batch, seq_len, 4, hidden_dim)
                            order: [fusion_token, text, image, tabular]
        Returns:
            (batch, seq_len, hidden_dim)  — [FUSION] token representation
        """
        batch_size, seq_len, num_modalities, hidden_dim = modality_stack.shape
        
        # Reshape for attention: (batch*seq_len, 4, hidden_dim)
        # Treat 4 tokens as sequence
        modality_flat = modality_stack.reshape(batch_size * seq_len, num_modalities, hidden_dim)
        
        # ===== PRE-LN RESIDUAL STRUCTURE (NUMERICALLY STABLE) =====
        # 1. Normalize input first (Pre-LN, not Post-LN)
        modality_norm = self.layer_norm_input(modality_flat)  # (batch*seq_len, 4, hidden_dim)
        
        # 2. Self-attention on normalized tokens (NO DROPOUT inside attention)
        attended, _ = self.mha(
            modality_norm, modality_norm, modality_norm,
            need_weights=False
        )
        
        # 3. Residual connection (stable because attention operates on normalized input)
        # 4. Dropout applied AFTER residual (not inside attention backward)
        # This prevents NaN from dropout creating sparse extreme gradients in attention backward
        output = modality_flat + self.dropout(attended)  # (batch*seq_len, 4, hidden_dim)
        
        # Extract only [FUSION] token (position 0): (batch*seq_len, hidden_dim)
        # No mean pooling - [FUSION] token is the only output
        fused = output[:, 0, :]  # First token is [FUSION]
        
        # Reshape back: (batch, seq_len, hidden_dim)
        return fused.reshape(batch_size, seq_len, hidden_dim)


class TemporalLSTMLayer(nn.Module):
    """
    Single-layer LSTM for temporal modelling across the 24-hour input window.

    Input:  (batch, seq_len, input_dim)
    Output: (batch, input_dim)  — final hidden state
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
            (batch, lstm_hidden_dim)
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
    Two-layer MLP prediction head. Input: (batch, input_dim). Output: (batch, 1)."""
    
    def __init__(self, input_dim: int = 64, dropout: float = 0.4):
        super().__init__()
        mid_dim = max(input_dim // 2, 16)  # Scale with input; floor at 16 for small configs
        self.head = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )
        # Initialize weights
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            (batch, 1)
        """
        return self.head(x)


class MultimodalFusionNet(nn.Module):
    """
    Full MultimodalFusionNet model.

    Combines a TabularEncoder, CrossModalAttentionLayer (with [FUSION] token),
    bottleneck linear, TemporalLSTM, and one or more PredictionHeads.

    Ablation modes zero-out modalities before the attention step:
        "full"         — all modalities active (default)
        "tabular_only" — text and image embeddings zeroed
        "no_text"      — text embedding zeroed
        "no_image"     — image embedding zeroed

    In multi-target mode (num_targets > 1), independent prediction heads are
    used to avoid gradient interference between targets.
    """
    
    def __init__(self, config, ablation_mode: str = "full", num_targets: int = 1):
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.seq_len = config.data.seq_len
        self.num_targets = num_targets

        # Ablation mode: controls which modalities are active
        # "full"         = all modalities (default)
        # "tabular_only" = zero-out text + image
        # "no_text"      = zero-out text only
        # "no_image"     = zero-out image only
        self.ablation_mode = ablation_mode

        logger.info(f"Initializing MultimodalFusionNet (hidden_dim={self.hidden_dim}, ablation={self.ablation_mode}, num_targets={self.num_targets})...")
        
        # 0. Learnable [FUSION] token (detector token for cross-modal fusion)
        # Shape: (1, 1, hidden_dim) -> expands to (batch, seq_len, hidden_dim) in forward
        self.fusion_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        # Initialize like BERT's [CLS] token: small normal distribution.
        # xavier_uniform_ is designed for weight matrices (fan_in/fan_out), not embedding vectors.
        nn.init.normal_(self.fusion_token, mean=0.0, std=0.02)
        logger.info("✓ [FUSION] token initialized (learnable 256D parameter, std=0.02)")
        
        # Asset embedding (2 assets: BTC=0, ETH=1, 16 dimensions)
        self.asset_embedding = nn.Embedding(2, 16)
        logger.info("✓ Asset embedding initialized (2 → 16 dimensions)")
        
        # 1. Tabular encoder (only trainable component with backbones)
        # input_size = tabular_features + 16 asset embedding
        # Base: 7+16=23 | Extended (ma7/ma25/rsi/macd): 11+16=27
        self.tabular_encoder = TabularEncoder(
            hidden_dim=self.hidden_dim,
            input_size=config.model.tabular_input_size,
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
        
        # 5. Prediction head(s)
        # num_targets=1 (default): single head, output (batch,) — single-target mode.
        # num_targets>1: independent head per target, output (batch, num_targets) — multi-target mode.
        # Separate heads prevent gradient interference between targets; each head specialises
        # on its own loss surface while sharing the full backbone.
        if self.num_targets > 1:
            self.prediction_heads = nn.ModuleList([
                PredictionHead(input_dim=config.model.bottleneck_dim, dropout=config.model.head_dropout)
                for _ in range(self.num_targets)
            ])
            logger.info(f"✓ {self.num_targets} independent prediction heads initialized (multi-target mode)")
        else:
            self.prediction_head = PredictionHead(
                input_dim=config.model.bottleneck_dim,
                dropout=config.model.head_dropout,
            )
            logger.info("✓ Single prediction head initialized (single-target mode)")

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
        Args:
            batch: Dict with keys:
                tabular:         (batch, seq_len, n_tab_features)
                text_embedding:  (batch, seq_len, 256)
                image_embedding: (batch, seq_len, 256)
                asset_id:        (batch,)  — 0=BTC, 1=ETH

        Returns:
            Single-target: (batch,)
            Multi-target:  (batch, num_targets)
        """
        batch_size, seq_len = batch["tabular"].shape[0], batch["tabular"].shape[1]
        
        # ==================== LEARNABLE [FUSION] TOKEN ====================
        # Expand [FUSION] token to match batch and sequence dimensions
        # self.fusion_token: (1, 1, hidden_dim) -> (batch, seq_len, hidden_dim)
        fusion_token_expanded = self.fusion_token.expand(batch_size, seq_len, -1)
        
        # ==================== PROCESS ASSET EMBEDDING ====================
        # batch["asset_id"]: (batch,) -> we convert to float vector (batch, 16)
        # then expand to (batch, seq_len, 16) to concat with tabular features
        asset_ids = batch.get("asset_id", torch.zeros(batch_size, dtype=torch.long, device=batch["tabular"].device))
        asset_emb = self.asset_embedding(asset_ids)  # (batch, 16)
        asset_emb_expanded = asset_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, 16)
        
        # Combine tabular and asset embedding
        tabular_combined = torch.cat([batch["tabular"], asset_emb_expanded], dim=2)  # (batch, seq_len, 23)
        
        # ==================== ENCODE TABULAR FEATURES ====================
        # Tabular encoder: (batch, seq_len, 23) -> (batch, seq_len, hidden_dim)
        tabular_features = self.tabular_encoder(tabular_combined)
        
        # ==================== USE PRE-EXTRACTED EMBEDDINGS ====================
        # Text embeddings: (batch, seq_len, 256) - already extracted offline
        text_features = batch["text_embedding"]  # (batch, seq_len, 256)
        
        # Image embeddings: (batch, seq_len, 256) - already extracted offline
        image_features = batch["image_embedding"]  # (batch, seq_len, 256)
        
        # ==================== ABLATION: ZERO-OUT DISABLED MODALITIES ====================
        if self.ablation_mode == "tabular_only":
            text_features = torch.zeros_like(text_features)
            image_features = torch.zeros_like(image_features)
        elif self.ablation_mode == "no_text":
            text_features = torch.zeros_like(text_features)
        elif self.ablation_mode == "no_image":
            image_features = torch.zeros_like(image_features)
        
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
        
        # ==================== PREDICTION HEAD(S) ====================
        # Single-target: (batch, 64) -> (batch, 1) -> (batch,)
        # Multi-target:  stack outputs of N independent heads -> (batch, num_targets)
        if self.num_targets > 1:
            predictions = torch.stack(
                [head(temporal_output).view(-1) for head in self.prediction_heads],
                dim=1,
            )  # (batch, num_targets)
        else:
            predictions = self.prediction_head(temporal_output).view(-1)  # (batch,)

        return predictions
    
    def get_trainable_params(self):
        """Return iterator of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]


if __name__ == "__main__":
    """Test model initialization and forward pass."""
    import sys
    import os
    # Allow running this file directly by adding the package root to sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from training.config import ExperimentConfig
    
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
        print(f"   ✓ Output shape: {output.shape} (expected: (4,))")
        print(f"   ✓ Output values: min={output.min():.4f}, max={output.max():.4f}")
        
        print("\n" + "=" * 80)
        print("✅ All model tests passed!")
        print("=" * 80)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
