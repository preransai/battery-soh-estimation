"""
Cross-Attention Multimodal Model for SOH Prediction

Improved architecture with:
- Multi-head cross-attention between numeric and semantic features
- Learnable adapter layers for pre-trained embeddings
- Residual connections and layer normalization
- Gradient checkpointing for memory efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class NumericEncoder(nn.Module):
    """
    Encodes numeric time-series features using CNN + Transformer.
    
    Architecture:
    1. 1D CNN for local pattern extraction
    2. Transformer encoder for global dependencies
    3. Pooling for fixed-size output
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 256,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim)
            
        Returns:
            pooled: (batch, output_dim) - pooled representation
            sequence: (batch, seq_len, output_dim) - full sequence
        """
        # CNN: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Back to (batch, seq_len, hidden_dim)
        x = x.transpose(1, 2)
        
        # Positional encoding + Transformer
        x = self.pos_encoding(x)
        x = self.transformer(x)
        
        # Project to output dimension
        sequence = self.output_proj(x)
        
        # Pooling: mean + max
        mean_pool = torch.mean(sequence, dim=1)
        max_pool, _ = torch.max(sequence, dim=1)
        pooled = (mean_pool + max_pool) / 2
        
        return pooled, sequence


class EmbeddingAdapter(nn.Module):
    """
    Adapter layer for fine-tuning frozen LLM embeddings.
    
    Uses bottleneck architecture to efficiently adapt
    pre-trained embeddings to battery domain.
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 bottleneck_dim: int = 64,
                 output_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        # Down projection
        self.down = nn.Linear(input_dim, bottleneck_dim)
        
        # Non-linearity
        self.activation = nn.GELU()
        
        # Up projection
        self.up = nn.Linear(bottleneck_dim, output_dim)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection if dimensions match
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, input_dim)
            
        Returns:
            (batch, output_dim)
        """
        # Bottleneck
        h = self.down(x)
        h = self.activation(h)
        h = self.up(h)
        h = self.dropout(h)
        
        # Residual connection
        if self.residual is not None:
            residual = self.residual(x)
        else:
            residual = x
        
        # Combine and normalize
        out = self.layer_norm(h + residual)
        
        return out


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention between numeric and semantic features.
    
    Query: Numeric features
    Key/Value: Semantic features
    
    This allows numeric representations to attend to relevant
    semantic information for prediction.
    """
    
    def __init__(self,
                 numeric_dim: int,
                 semantic_dim: int,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        assert numeric_dim % n_heads == 0, "numeric_dim must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.head_dim = numeric_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Query from numeric
        self.query = nn.Linear(numeric_dim, numeric_dim)
        
        # Key/Value from semantic (project to numeric dim)
        self.key = nn.Linear(semantic_dim, numeric_dim)
        self.value = nn.Linear(semantic_dim, numeric_dim)
        
        # Output projection
        self.output = nn.Linear(numeric_dim, numeric_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(numeric_dim)
    
    def forward(self, 
                numeric: torch.Tensor,
                semantic: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            numeric: (batch, numeric_dim) or (batch, seq_len, numeric_dim)
            semantic: (batch, semantic_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            output: Same shape as numeric
            attention: Optional attention weights
        """
        # Handle both pooled and sequence inputs
        if numeric.dim() == 2:
            numeric = numeric.unsqueeze(1)  # (batch, 1, numeric_dim)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, _ = numeric.shape
        
        # Expand semantic to match sequence length
        semantic = semantic.unsqueeze(1)  # (batch, 1, semantic_dim)
        
        # Compute Q, K, V
        Q = self.query(numeric)  # (batch, seq_len, numeric_dim)
        K = self.key(semantic)   # (batch, 1, numeric_dim)
        V = self.value(semantic) # (batch, 1, numeric_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.output(out)
        
        # Residual + LayerNorm
        out = self.layer_norm(out + numeric)
        
        if squeeze_output:
            out = out.squeeze(1)
        
        if return_attention:
            return out, attn.mean(dim=1)  # Average over heads
        
        return out, None


class GatedFusion(nn.Module):
    """
    Gated fusion of numeric and semantic features.
    
    Uses learned gates to dynamically weight the contribution
    of each modality based on input.
    """
    
    def __init__(self, 
                 numeric_dim: int,
                 semantic_dim: int,
                 output_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        total_dim = numeric_dim + semantic_dim
        
        # Gate networks
        self.numeric_gate = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.semantic_gate = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.Sigmoid()
        )
        
        # Projection layers
        self.numeric_proj = nn.Linear(numeric_dim, output_dim)
        self.semantic_proj = nn.Linear(semantic_dim, output_dim)
        
        # Output layer
        self.output = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_dim = output_dim
    
    def forward(self, 
                numeric: torch.Tensor,
                semantic: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            numeric: (batch, numeric_dim)
            semantic: (batch, semantic_dim)
            
        Returns:
            fused: (batch, output_dim)
            gates: Dictionary with gate values for interpretability
        """
        # Concatenate for gate computation
        combined = torch.cat([numeric, semantic], dim=-1)
        
        # Compute gates
        g_numeric = self.numeric_gate(combined)
        g_semantic = self.semantic_gate(combined)
        
        # Project features
        h_numeric = self.numeric_proj(numeric)
        h_semantic = self.semantic_proj(semantic)
        
        # Gated combination
        fused = g_numeric * h_numeric + g_semantic * h_semantic
        
        # Output
        out = self.output(fused)
        
        gates = {
            'numeric_gate': g_numeric.mean().item(),
            'semantic_gate': g_semantic.mean().item()
        }
        
        return out, gates


class CrossAttentionSOHModel(nn.Module):
    """
    Full cross-attention multimodal model for SOH prediction.
    
    Architecture:
    1. NumericEncoder: CNN + Transformer for time-series
    2. EmbeddingAdapter: Fine-tune LLM embeddings
    3. MultiHeadCrossAttention: Fuse modalities
    4. GatedFusion: Dynamic weighting
    5. SOH Regressor: Final prediction
    """
    
    def __init__(self,
                 numeric_input_dim: int = 3,  # voltage, current, temperature
                 semantic_input_dim: int = 768,  # SBERT embedding dim
                 hidden_dim: int = 256,
                 n_attention_heads: int = 4,
                 n_transformer_layers: int = 2,
                 dropout: float = 0.1,
                 use_cross_attention: bool = True,
                 use_gated_fusion: bool = True):
        super().__init__()
        
        self.use_cross_attention = use_cross_attention
        self.use_gated_fusion = use_gated_fusion
        
        # Numeric encoder
        self.numeric_encoder = NumericEncoder(
            input_dim=numeric_input_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=hidden_dim,
            n_heads=n_attention_heads,
            n_layers=n_transformer_layers,
            dropout=dropout
        )
        
        # Semantic adapter
        self.semantic_adapter = EmbeddingAdapter(
            input_dim=semantic_input_dim,
            bottleneck_dim=64,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # Cross-attention
        if use_cross_attention:
            self.cross_attention = MultiHeadCrossAttention(
                numeric_dim=hidden_dim,
                semantic_dim=hidden_dim,
                n_heads=n_attention_heads,
                dropout=dropout
            )
        
        # Gated fusion
        if use_gated_fusion:
            self.gated_fusion = GatedFusion(
                numeric_dim=hidden_dim,
                semantic_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout
            )
            regressor_input_dim = hidden_dim
        else:
            regressor_input_dim = hidden_dim * 2  # Concatenation
        
        # SOH regressor
        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self,
                numeric_seq: torch.Tensor,
                semantic_emb: torch.Tensor,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            numeric_seq: (batch, seq_len, numeric_input_dim)
            semantic_emb: (batch, semantic_input_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with predictions and optional attention
        """
        # Encode numeric
        numeric_pooled, numeric_seq_out = self.numeric_encoder(numeric_seq)
        
        # Adapt semantic embeddings
        semantic_adapted = self.semantic_adapter(semantic_emb)
        
        # Cross-attention
        attn_weights = None
        if self.use_cross_attention:
            numeric_attended, attn_weights = self.cross_attention(
                numeric_pooled, semantic_adapted, return_attention
            )
        else:
            numeric_attended = numeric_pooled
        
        # Fusion
        gates = None
        if self.use_gated_fusion:
            fused, gates = self.gated_fusion(numeric_attended, semantic_adapted)
        else:
            fused = torch.cat([numeric_attended, semantic_adapted], dim=-1)
        
        # Predict SOH
        soh_pred = self.regressor(fused).squeeze(-1)
        
        output = {
            'soh': soh_pred,
            'numeric_encoding': numeric_pooled,
            'semantic_encoding': semantic_adapted
        }
        
        if return_attention and attn_weights is not None:
            output['attention'] = attn_weights
        
        if gates is not None:
            output['gates'] = gates
        
        return output


class MultimodalDataset(Dataset):
    """Dataset for cross-attention model."""
    
    def __init__(self,
                 sequences: np.ndarray,
                 embeddings: np.ndarray,
                 targets: np.ndarray):
        """
        Args:
            sequences: (n_samples, seq_len, n_features)
            embeddings: (n_samples, embedding_dim)
            targets: (n_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.embeddings = torch.FloatTensor(embeddings)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'embedding': self.embeddings[idx],
            'target': self.targets[idx]
        }


class CrossAttentionTrainer:
    """Trainer for cross-attention model."""
    
    def __init__(self,
                 model: CrossAttentionSOHModel,
                 device: str = 'cpu',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            sequences = batch['sequence'].to(self.device)
            embeddings = batch['embedding'].to(self.device)
            targets = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(sequences, embeddings)
            loss = self.criterion(output['soh'], targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0
        
        for batch in val_loader:
            sequences = batch['sequence'].to(self.device)
            embeddings = batch['embedding'].to(self.device)
            targets = batch['target'].to(self.device)
            
            output = self.model(sequences, embeddings)
            loss = self.criterion(output['soh'], targets)
            
            total_loss += loss.item()
            # Use .tolist() instead of .numpy() to avoid numpy compatibility issues
            all_preds.extend(output['soh'].detach().cpu().tolist())
            all_targets.extend(targets.detach().cpu().tolist())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mae = float(np.mean(np.abs(all_preds - all_targets)))
        rmse = float(np.sqrt(np.mean((all_preds - all_targets) ** 2)))
        
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        return {
            'loss': total_loss / len(val_loader),
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              n_epochs: int = 100,
              early_stopping_patience: int = 10) -> Dict:
        """Full training loop."""
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            
            self.scheduler.step(val_metrics['loss'])
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                      f"Val R²={val_metrics['r2']:.4f}, MAE={val_metrics['mae']:.4f}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'final_metrics': self.evaluate(val_loader)
        }


if __name__ == "__main__":
    print("Cross-Attention Multimodal Model")
    print("=" * 50)
    
    # Test model
    model = CrossAttentionSOHModel(
        numeric_input_dim=3,
        semantic_input_dim=768,
        hidden_dim=256
    )
    
    # Test forward pass
    batch_size = 8
    seq_len = 500
    
    numeric_seq = torch.randn(batch_size, seq_len, 3)
    semantic_emb = torch.randn(batch_size, 768)
    
    output = model(numeric_seq, semantic_emb, return_attention=True)
    
    print(f"\nModel output:")
    print(f"  SOH predictions: {output['soh'].shape}")
    print(f"  Numeric encoding: {output['numeric_encoding'].shape}")
    print(f"  Semantic encoding: {output['semantic_encoding'].shape}")
    
    if 'attention' in output:
        print(f"  Attention weights: {output['attention'].shape}")
    
    if 'gates' in output:
        print(f"  Gates: {output['gates']}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
