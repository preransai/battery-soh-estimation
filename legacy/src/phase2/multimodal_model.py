"""
Phase-2 Step 4: Multimodal Model
Full multimodal architecture combining numeric and semantic features

Architecture:
1. Numeric Encoder (CNN-LSTM)
2. Feature-to-Text Conversion
3. LLM Embeddings (Sentence-BERT)
4. Semantic Encoder (MLP)
5. Cross-Attention Fusion
6. SOH Regression Head
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pickle
import json

# Project root
PROJECT_ROOT = '.'
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'phase2'))

from feature_to_text import FeatureToText, convert_features_to_text


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@dataclass
class MultimodalTrainingResult:
    """Container for multimodal training results."""
    model_name: str
    run_id: int
    train_loss: List[float]
    val_loss: List[float]
    test_mae: float
    test_rmse: float
    test_mape: float
    test_r2: float
    best_epoch: int
    training_time: float
    hyperparameters: Dict


# ============================================================================
# LLM EMBEDDING MODULE
# ============================================================================

class LLMEmbedder:
    """
    Generate embeddings using Sentence-BERT or similar models.
    Uses frozen embeddings (no fine-tuning).
    """
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2', device: str = None):
        """
        Initialize embedder.
        
        Args:
            model_name: Sentence-BERT model name
            device: Device for computation
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.embedding_dim = 768  # Default for MPNet
    
    def load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✓ Loaded {self.model_name} (dim={self.embedding_dim})")
        except ImportError:
            print("⚠️ sentence-transformers not installed. Using random embeddings.")
            self.model = None
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
        
        Returns:
            Array of shape (n_texts, embedding_dim)
        """
        if self.model is None:
            # Fallback to random embeddings for testing
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.astype(np.float32)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed_texts([text])[0]


# ============================================================================
# MULTIMODAL DATASET
# ============================================================================

class MultimodalDataset(Dataset):
    """Dataset for multimodal SOH prediction."""
    
    def __init__(self, numeric_features: np.ndarray, 
                 semantic_embeddings: np.ndarray,
                 targets: np.ndarray,
                 sequence_length: int = 10):
        """
        Args:
            numeric_features: Numeric features (n_samples, n_features)
            semantic_embeddings: LLM embeddings (n_samples, embedding_dim)
            targets: SOH targets (n_samples,)
            sequence_length: Sequence length for temporal modeling
        """
        self.numeric = torch.FloatTensor(numeric_features)
        self.semantic = torch.FloatTensor(semantic_embeddings)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = sequence_length
        
        # Create valid indices
        self.indices = list(range(0, len(numeric_features) - sequence_length + 1))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.seq_len
        
        # Numeric sequence
        x_numeric = self.numeric[start:end]  # (seq_len, n_features)
        
        # Semantic: use last embedding in sequence
        x_semantic = self.semantic[end - 1]  # (embedding_dim,)
        
        # Target: SOH at end of sequence
        y = self.targets[end - 1]
        
        return x_numeric, x_semantic, y


class MultimodalFeatureDataset(Dataset):
    """Dataset for non-sequential multimodal prediction."""
    
    def __init__(self, numeric_features: np.ndarray,
                 semantic_embeddings: np.ndarray,
                 targets: np.ndarray):
        """
        Args:
            numeric_features: Numeric features (n_samples, n_features)
            semantic_embeddings: LLM embeddings (n_samples, embedding_dim)
            targets: SOH targets (n_samples,)
        """
        self.numeric = torch.FloatTensor(numeric_features)
        self.semantic = torch.FloatTensor(semantic_embeddings)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.numeric)
    
    def __getitem__(self, idx):
        return self.numeric[idx], self.semantic[idx], self.targets[idx]


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class NumericEncoder(nn.Module):
    """CNN-LSTM encoder for numeric features."""
    
    def __init__(self, input_dim: int, cnn_filters: List[int] = [32, 64],
                 kernel_size: int = 3, lstm_hidden: int = 128,
                 lstm_layers: int = 2, output_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        
        # CNN layers
        cnn_layers = []
        in_channels = input_dim
        
        for out_channels in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_dim = output_dim
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        # CNN expects (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # CNN
        cnn_out = self.cnn(x)  # (batch, filters, seq_len)
        
        # Back to (batch, seq, features)
        cnn_out = cnn_out.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)
        
        # Use last hidden state
        hidden = h_n[-1, :, :]  # (batch, lstm_hidden)
        
        # Project
        out = self.output_proj(hidden)  # (batch, output_dim)
        
        return out


class SemanticEncoder(nn.Module):
    """MLP encoder for semantic embeddings."""
    
    def __init__(self, input_dim: int = 768, 
                 hidden_dims: List[int] = [512, 256, 128],
                 output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim
    
    def forward(self, x):
        return self.network(x)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion of numeric and semantic features."""
    
    def __init__(self, numeric_dim: int, semantic_dim: int,
                 attention_heads: int = 4, output_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.numeric_dim = numeric_dim
        self.semantic_dim = semantic_dim
        
        # Project to common dimension
        self.common_dim = output_dim
        
        self.numeric_proj = nn.Linear(numeric_dim, self.common_dim)
        self.semantic_proj = nn.Linear(semantic_dim, self.common_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.common_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_dim = output_dim
    
    def forward(self, numeric_features, semantic_features):
        # Project to common dimension
        numeric_proj = self.numeric_proj(numeric_features)  # (batch, common_dim)
        semantic_proj = self.semantic_proj(semantic_features)  # (batch, common_dim)
        
        # Reshape for attention: (batch, seq=1, dim)
        numeric_proj = numeric_proj.unsqueeze(1)
        semantic_proj = semantic_proj.unsqueeze(1)
        
        # Cross-attention: numeric attends to semantic
        attn_out, attn_weights = self.attention(
            query=numeric_proj,
            key=semantic_proj,
            value=semantic_proj
        )
        
        # Remove sequence dimension
        attn_out = attn_out.squeeze(1)  # (batch, common_dim)
        numeric_proj = numeric_proj.squeeze(1)
        
        # Concatenate and project
        combined = torch.cat([numeric_proj, attn_out], dim=1)  # (batch, common_dim * 2)
        output = self.output_proj(combined)  # (batch, output_dim)
        
        return output, attn_weights


class GatedFusion(nn.Module):
    """Gated fusion mechanism for combining modalities."""
    
    def __init__(self, numeric_dim: int, semantic_dim: int, output_dim: int = 256):
        super().__init__()
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(numeric_dim + semantic_dim, output_dim),
            nn.Sigmoid()
        )
        
        # Transform each modality
        self.numeric_transform = nn.Linear(numeric_dim, output_dim)
        self.semantic_transform = nn.Linear(semantic_dim, output_dim)
        
        self.output_dim = output_dim
    
    def forward(self, numeric_features, semantic_features):
        # Compute gate values
        combined = torch.cat([numeric_features, semantic_features], dim=1)
        gate_values = self.gate(combined)
        
        # Transform modalities
        numeric_trans = self.numeric_transform(numeric_features)
        semantic_trans = self.semantic_transform(semantic_features)
        
        # Gated combination
        output = gate_values * numeric_trans + (1 - gate_values) * semantic_trans
        
        return output, gate_values


class SOHRegressor(nn.Module):
    """Regression head for SOH prediction."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


# ============================================================================
# FULL MULTIMODAL MODEL
# ============================================================================

class MultimodalSOHModel(nn.Module):
    """
    Full multimodal model for SOH prediction.
    
    Architecture:
    - Numeric Encoder (CNN-LSTM) → 128-dim
    - Semantic Encoder (MLP) → 128-dim  
    - Cross-Attention Fusion → 256-dim
    - SOH Regressor → 1
    """
    
    def __init__(self, numeric_input_dim: int, semantic_input_dim: int = 768,
                 numeric_output_dim: int = 128, semantic_output_dim: int = 128,
                 fusion_type: str = 'cross_attention', fusion_output_dim: int = 256,
                 attention_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # Numeric encoder
        self.numeric_encoder = NumericEncoder(
            input_dim=numeric_input_dim,
            output_dim=numeric_output_dim,
            dropout=dropout
        )
        
        # Semantic encoder
        self.semantic_encoder = SemanticEncoder(
            input_dim=semantic_input_dim,
            output_dim=semantic_output_dim,
            dropout=dropout
        )
        
        # Fusion layer
        if fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(
                numeric_dim=numeric_output_dim,
                semantic_dim=semantic_output_dim,
                attention_heads=attention_heads,
                output_dim=fusion_output_dim,
                dropout=dropout
            )
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(
                numeric_dim=numeric_output_dim,
                semantic_dim=semantic_output_dim,
                output_dim=fusion_output_dim
            )
        elif fusion_type == 'concatenation':
            self.fusion = None
            fusion_output_dim = numeric_output_dim + semantic_output_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Regressor
        self.regressor = SOHRegressor(
            input_dim=fusion_output_dim,
            hidden_dims=[128, 64],
            dropout=dropout
        )
        
        # Store attention weights for visualization
        self.last_attention_weights = None
    
    def forward(self, numeric_seq, semantic_emb):
        # Encode numeric features
        numeric_enc = self.numeric_encoder(numeric_seq)  # (batch, 128)
        
        # Encode semantic features
        semantic_enc = self.semantic_encoder(semantic_emb)  # (batch, 128)
        
        # Fuse
        if self.fusion_type == 'concatenation':
            fused = torch.cat([numeric_enc, semantic_enc], dim=1)
            self.last_attention_weights = None
        else:
            fused, attn_weights = self.fusion(numeric_enc, semantic_enc)
            self.last_attention_weights = attn_weights
        
        # Predict SOH
        soh_pred = self.regressor(fused)
        
        return soh_pred
    
    def get_attention_weights(self):
        """Get last attention weights for visualization."""
        return self.last_attention_weights


class NumericOnlyModel(nn.Module):
    """Numeric-only model for ablation comparison."""
    
    def __init__(self, numeric_input_dim: int, output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        self.numeric_encoder = NumericEncoder(
            input_dim=numeric_input_dim,
            output_dim=output_dim,
            dropout=dropout
        )
        
        self.regressor = SOHRegressor(
            input_dim=output_dim,
            hidden_dims=[64, 32],
            dropout=dropout
        )
    
    def forward(self, numeric_seq, semantic_emb=None):
        # Ignore semantic
        numeric_enc = self.numeric_encoder(numeric_seq)
        soh_pred = self.regressor(numeric_enc)
        return soh_pred


# ============================================================================
# TRAINER
# ============================================================================

class MultimodalTrainer:
    """Trainer for multimodal SOH model."""
    
    def __init__(self, config: Dict, project_root: str = None):
        self.config = config
        self.project_root = project_root or PROJECT_ROOT
        self.train_config = config['training']
        self.mm_config = config['multimodal']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Output directory
        self.output_dir = os.path.join(self.project_root, config['paths']['multimodal_models'])
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Results
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # LLM Embedder
        self.embedder = LLMEmbedder(
            model_name=self.mm_config['llm_embeddings']['model'],
            device=str(self.device)
        )
    
    def prepare_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                     feature_cols: List[str], seq_length: int = 10) -> Dict:
        """
        Prepare multimodal data.
        
        Args:
            train_df: Training features DataFrame
            test_df: Test features DataFrame
            feature_cols: Numeric feature columns
            seq_length: Sequence length
        
        Returns:
            Dictionary with DataLoaders and embeddings
        """
        print("\nPreparing multimodal data...")
        
        # 1. Convert features to text
        print("  Converting features to text...")
        train_text_df = convert_features_to_text(train_df)
        test_text_df = convert_features_to_text(test_df)
        
        # 2. Generate LLM embeddings
        print("  Generating LLM embeddings...")
        self.embedder.load_model()
        
        train_texts = train_text_df['full_description'].tolist()
        test_texts = test_text_df['full_description'].tolist()
        
        train_embeddings = self.embedder.embed_texts(train_texts)
        test_embeddings = self.embedder.embed_texts(test_texts)
        
        print(f"  Train embeddings shape: {train_embeddings.shape}")
        print(f"  Test embeddings shape: {test_embeddings.shape}")
        
        # 3. Prepare numeric features
        X_train = train_df[feature_cols].values
        y_train = train_df['soh'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['soh'].values
        
        # 4. Validation split
        val_ratio = self.train_config.get('validation_ratio', 0.15)
        val_size = int(len(X_train) * val_ratio)
        
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        emb_val = train_embeddings[-val_size:]
        
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        emb_train = train_embeddings[:-val_size]
        
        print(f"\n  Data shapes:")
        print(f"    Train: {X_train.shape}, embeddings: {emb_train.shape}")
        print(f"    Val: {X_val.shape}, embeddings: {emb_val.shape}")
        print(f"    Test: {X_test.shape}, embeddings: {test_embeddings.shape}")
        
        # 5. Create datasets
        train_dataset = MultimodalDataset(X_train, emb_train, y_train, seq_length)
        val_dataset = MultimodalDataset(X_val, emb_val, y_val, seq_length)
        test_dataset = MultimodalDataset(X_test, test_embeddings, y_test, seq_length)
        
        batch_size = self.train_config['batch_size']
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'numeric_dim': X_train.shape[1],
            'semantic_dim': train_embeddings.shape[1],
            'train_text_df': train_text_df,
            'test_text_df': test_text_df
        }
    
    def create_model(self, numeric_dim: int, semantic_dim: int,
                     model_type: str = 'full') -> nn.Module:
        """
        Create multimodal model.
        
        Args:
            numeric_dim: Numeric feature dimension
            semantic_dim: Semantic embedding dimension
            model_type: 'full' or 'numeric_only'
        """
        if model_type == 'numeric_only':
            model = NumericOnlyModel(
                numeric_input_dim=numeric_dim,
                output_dim=128,
                dropout=0.2
            )
        else:
            model = MultimodalSOHModel(
                numeric_input_dim=numeric_dim,
                semantic_input_dim=semantic_dim,
                numeric_output_dim=self.mm_config['numeric_encoder']['output_dim'],
                semantic_output_dim=128,
                fusion_type=self.mm_config['fusion']['type'],
                fusion_output_dim=self.mm_config['fusion']['output_dim'],
                attention_heads=self.mm_config['fusion']['attention_heads'],
                dropout=0.2
            )
        
        return model.to(self.device)
    
    def train_model(self, model: nn.Module, data: Dict,
                    model_name: str, run_id: int) -> Tuple[MultimodalTrainingResult, nn.Module]:
        """Train multimodal model."""
        import time
        start_time = time.time()
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.train_config['learning_rate']
        )
        criterion = nn.MSELoss()
        
        # Early stopping
        patience = self.train_config['early_stopping']['patience']
        min_delta = self.train_config['early_stopping']['min_delta']
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        train_losses = []
        val_losses = []
        
        epochs = self.train_config['epochs']
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for numeric, semantic, target in data['train_loader']:
                numeric = numeric.to(self.device)
                semantic = semantic.to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                output = model(numeric, semantic)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(data['train_loader'])
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for numeric, semantic, target in data['val_loader']:
                    numeric = numeric.to(self.device)
                    semantic = semantic.to(self.device)
                    target = target.to(self.device)
                    
                    output = model(numeric, semantic)
                    loss = criterion(output, target)
                    val_loss += loss.item()
            
            val_loss /= len(data['val_loader'])
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Evaluate
        test_metrics = self.evaluate(model, data['test_loader'])
        
        training_time = time.time() - start_time
        
        result = MultimodalTrainingResult(
            model_name=model_name,
            run_id=run_id,
            train_loss=train_losses,
            val_loss=val_losses,
            test_mae=test_metrics['mae'],
            test_rmse=test_metrics['rmse'],
            test_mape=test_metrics['mape'],
            test_r2=test_metrics['r2'],
            best_epoch=best_epoch,
            training_time=training_time,
            hyperparameters=self.mm_config
        )
        
        return result, model
    
    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set."""
        model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for numeric, semantic, target in test_loader:
                numeric = numeric.to(self.device)
                semantic = semantic.to(self.device)
                
                output = model(numeric, semantic)
                
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(target.numpy())
        
        preds = np.array(all_preds)
        targets = np.array(all_targets)
        
        mae = np.mean(np.abs(preds - targets))
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        mape = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100
        
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'predictions': preds,
            'targets': targets
        }
    
    def run_experiments(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                        feature_cols: List[str], num_runs: int = 3) -> pd.DataFrame:
        """Run multimodal experiments."""
        print(f"\n{'=' * 80}")
        print("PHASE-2 STEP 4: MULTIMODAL MODEL".center(80))
        print('=' * 80)
        
        # Prepare data
        data = self.prepare_data(train_df, test_df, feature_cols)
        
        all_results = []
        best_model = None
        best_rmse = float('inf')
        
        # Train full multimodal model
        print(f"\n{'─' * 80}")
        print("Training: FULL MULTIMODAL (Numeric + Semantic)")
        print('─' * 80)
        
        for run in range(num_runs):
            seed = self.config['random_seed'] + run
            set_seed(seed)
            
            print(f"\n  Run {run + 1}/{num_runs} (seed={seed})")
            
            model = self.create_model(
                data['numeric_dim'],
                data['semantic_dim'],
                model_type='full'
            )
            
            result, trained_model = self.train_model(
                model, data, 'multimodal_full', run
            )
            
            all_results.append(result)
            
            print(f"  → MAE: {result.test_mae:.4f}, RMSE: {result.test_rmse:.4f}, R²: {result.test_r2:.4f}")
            
            if result.test_rmse < best_rmse:
                best_rmse = result.test_rmse
                best_model = trained_model
        
        # Train numeric-only for comparison
        print(f"\n{'─' * 80}")
        print("Training: NUMERIC ONLY (for comparison)")
        print('─' * 80)
        
        for run in range(num_runs):
            seed = self.config['random_seed'] + run
            set_seed(seed)
            
            print(f"\n  Run {run + 1}/{num_runs} (seed={seed})")
            
            model = self.create_model(
                data['numeric_dim'],
                data['semantic_dim'],
                model_type='numeric_only'
            )
            
            result, _ = self.train_model(model, data, 'numeric_only', run)
            all_results.append(result)
            
            print(f"  → MAE: {result.test_mae:.4f}, RMSE: {result.test_rmse:.4f}, R²: {result.test_r2:.4f}")
        
        self.results = all_results
        self.best_model = best_model
        self.data = data
        
        # Create results DataFrame
        results_df = self._create_results_df()
        
        return results_df
    
    def _create_results_df(self) -> pd.DataFrame:
        """Create results DataFrame."""
        data = []
        for r in self.results:
            data.append({
                'model': r.model_name,
                'run': r.run_id,
                'mae': r.test_mae,
                'rmse': r.test_rmse,
                'mape': r.test_mape,
                'r2': r.test_r2,
                'best_epoch': r.best_epoch,
                'training_time': r.training_time
            })
        
        return pd.DataFrame(data)
    
    def save_results(self, results_df: pd.DataFrame) -> None:
        """Save all results."""
        print(f"\n{'─' * 80}")
        print("SAVING RESULTS")
        print('─' * 80)
        
        # Save results
        results_path = os.path.join(self.output_dir, f'multimodal_results_{self.timestamp}.csv')
        results_df.to_csv(results_path, index=False)
        print(f"  ✓ Results: {results_path}")
        
        # Save best model
        if self.best_model is not None:
            model_path = os.path.join(self.output_dir, f'multimodal_best_{self.timestamp}.pt')
            torch.save({
                'model_state_dict': self.best_model.state_dict(),
                'timestamp': self.timestamp,
                'config': self.mm_config
            }, model_path)
            print(f"  ✓ Best model: {model_path}")
        
        # Save text data
        if hasattr(self, 'data') and self.data is not None:
            text_path = os.path.join(self.output_dir, f'text_descriptions_{self.timestamp}.csv')
            self.data['train_text_df'].to_csv(text_path, index=False)
            print(f"  ✓ Text descriptions: {text_path}")
    
    def print_summary(self, results_df: pd.DataFrame) -> None:
        """Print summary."""
        print(f"\n{'=' * 80}")
        print("MULTIMODAL MODEL SUMMARY".center(80))
        print('=' * 80)
        
        summary = results_df.groupby('model').agg({
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'r2': ['mean', 'std']
        })
        
        print("\nResults (mean ± std):\n")
        print(f"{'Model':<20} {'MAE':<20} {'RMSE':<20} {'R²':<20}")
        print('─' * 80)
        
        for model in summary.index:
            mae_m = summary.loc[model, ('mae', 'mean')]
            mae_s = summary.loc[model, ('mae', 'std')]
            rmse_m = summary.loc[model, ('rmse', 'mean')]
            rmse_s = summary.loc[model, ('rmse', 'std')]
            r2_m = summary.loc[model, ('r2', 'mean')]
            r2_s = summary.loc[model, ('r2', 'std')]
            
            print(f"{model:<20} {mae_m:.4f} ± {mae_s:.4f}   {rmse_m:.4f} ± {rmse_s:.4f}   {r2_m:.4f} ± {r2_s:.4f}")
        
        print('─' * 80)
        
        # Compare multimodal vs numeric-only
        mm_results = results_df[results_df['model'] == 'multimodal_full']
        num_results = results_df[results_df['model'] == 'numeric_only']
        
        if len(mm_results) > 0 and len(num_results) > 0:
            mm_rmse = mm_results['rmse'].mean()
            num_rmse = num_results['rmse'].mean()
            improvement = (num_rmse - mm_rmse) / num_rmse * 100
            
            print(f"\n✓ Multimodal improvement over numeric-only: {improvement:.2f}%")
        
        print('\n' + '=' * 80)


def run_multimodal(config: Dict, project_root: str,
                   train_path: str, test_path: str) -> pd.DataFrame:
    """Run complete multimodal experiments."""
    trainer = MultimodalTrainer(config, project_root)
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Feature columns
    exclude_cols = ['battery_id', 'cycle_number', 'original_index', 'soh', 'has_spikes']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    # Run experiments
    results_df = trainer.run_experiments(
        train_df, test_df, feature_cols,
        num_runs=config['training']['num_runs']
    )
    
    # Save and summarize
    trainer.save_results(results_df)
    trainer.print_summary(results_df)
    
    return results_df


if __name__ == "__main__":
    import yaml
    
    project_root = PROJECT_ROOT
    
    with open(os.path.join(project_root, 'phase2_config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    
    train_path = os.path.join(project_root, 'data/phase2/harmonized/train_features.csv')
    test_path = os.path.join(project_root, 'data/phase2/harmonized/test_features.csv')
    
    results_df = run_multimodal(config, project_root, train_path, test_path)
    
    print("\n" + "="*80)
    print("STEP 4 COMPLETE ✅")
    print("Proceed to Step 5: Ablation Studies")
    print("="*80)
