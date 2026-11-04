"""
Phase-2 Step 3: Baseline Models
LSTM, CNN-LSTM, and Transformer models for SOH prediction

All models are trained with:
- 3 runs for reproducibility
- Identical train/test splits
- Fixed random seeds
- Early stopping
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pickle
import json

# Add parent path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainingResult:
    """Container for training results."""
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
# DATASET
# ============================================================================

class SOHDataset(Dataset):
    """PyTorch Dataset for SOH prediction."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 sequence_length: int = 10, stride: int = 1):
        """
        Args:
            features: Feature array (n_samples, n_features)
            targets: SOH targets (n_samples,)
            sequence_length: Number of consecutive cycles to use
            stride: Step between sequences
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = sequence_length
        self.stride = stride
        
        # Create sequence indices
        self.indices = list(range(0, len(features) - sequence_length + 1, stride))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len
        
        # Get sequence of features
        x = self.features[start_idx:end_idx]  # (seq_len, n_features)
        
        # Target is SOH at the end of sequence
        y = self.targets[end_idx - 1]
        
        return x, y


class SOHFeatureDataset(Dataset):
    """Dataset using extracted physics features (not sequences)."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Args:
            features: Feature array (n_samples, n_features)
            targets: SOH targets (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# ============================================================================
# MODELS
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM model for SOH prediction."""
    
    def __init__(self, input_dim: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2,
                 bidirectional: bool = False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension depends on bidirectional
        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward final states
            hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            hidden = h_n[-1, :, :]
        
        out = self.fc(hidden)
        return out.squeeze(-1)


class CNNLSTMModel(nn.Module):
    """CNN-LSTM hybrid model for SOH prediction."""
    
    def __init__(self, input_dim: int, cnn_filters: List[int] = [32, 64, 64],
                 kernel_size: int = 3, pool_size: int = 2,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        # CNN layers
        cnn_layers = []
        in_channels = input_dim
        
        for i, out_channels in enumerate(cnn_filters):
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(pool_size) if i < len(cnn_filters) - 1 else nn.Identity()
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
        
        # Fully connected
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        # CNN expects (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # CNN
        cnn_out = self.cnn(x)  # (batch, filters, reduced_seq)
        
        # Back to (batch, seq, features)
        cnn_out = cnn_out.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)
        
        # Use last hidden state
        hidden = h_n[-1, :, :]
        
        out = self.fc(hidden)
        return out.squeeze(-1)


class TransformerModel(nn.Module):
    """Transformer encoder model for SOH prediction."""
    
    def __init__(self, input_dim: int, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Transformer
        transformer_out = self.transformer(x)  # (batch, seq, d_model)
        
        # Use mean pooling over sequence
        pooled = transformer_out.mean(dim=1)  # (batch, d_model)
        
        out = self.fc(pooled)
        return out.squeeze(-1)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MLPBaseline(nn.Module):
    """Simple MLP baseline using physics features."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.2):
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
        
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


# ============================================================================
# TRAINER
# ============================================================================

class BaselineTrainer:
    """Trainer for baseline models with reproducibility."""
    
    def __init__(self, config: Dict, project_root: str = None):
        """
        Args:
            config: Configuration dictionary
            project_root: Project root path
        """
        self.config = config
        self.project_root = project_root or '.'
        self.train_config = config['training']
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Output directory
        self.output_dir = os.path.join(self.project_root, config['paths']['baseline_models'])
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Results storage
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data."""
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Loaded train: {len(train_df)} samples")
        print(f"Loaded test: {len(test_df)} samples")
        
        return train_df, test_df
    
    def prepare_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                     feature_cols: List[str], target_col: str = 'soh',
                     sequence_mode: bool = True, seq_length: int = 10) -> Dict:
        """
        Prepare data for training.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            feature_cols: Columns to use as features
            target_col: Target column name
            sequence_mode: If True, create sequences; else use features directly
            seq_length: Sequence length (if sequence_mode)
        
        Returns:
            Dictionary with DataLoaders
        """
        # Extract features and targets
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
        
        # Validation split from training data
        val_ratio = self.train_config.get('validation_ratio', 0.15)
        val_size = int(len(X_train) * val_ratio)
        
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        print(f"\nData shapes:")
        print(f"  Train: {X_train.shape}")
        print(f"  Val: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        
        # Create datasets
        if sequence_mode:
            train_dataset = SOHDataset(X_train, y_train, seq_length)
            val_dataset = SOHDataset(X_val, y_val, seq_length)
            test_dataset = SOHDataset(X_test, y_test, seq_length)
        else:
            train_dataset = SOHFeatureDataset(X_train, y_train)
            val_dataset = SOHFeatureDataset(X_val, y_val)
            test_dataset = SOHFeatureDataset(X_test, y_test)
        
        # Create DataLoaders
        batch_size = self.train_config.get('batch_size', 16)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'input_dim': X_train.shape[1],
            'X_test': X_test,
            'y_test': y_test
        }
    
    def create_model(self, model_type: str, input_dim: int) -> nn.Module:
        """Create model based on type."""
        model_configs = self.config['baseline_models']
        
        if model_type == 'lstm':
            cfg = model_configs['lstm']
            model = LSTMModel(
                input_dim=input_dim,
                hidden_size=cfg['hidden_size'],
                num_layers=cfg['num_layers'],
                dropout=cfg['dropout'],
                bidirectional=cfg.get('bidirectional', False)
            )
        
        elif model_type == 'cnn_lstm':
            cfg = model_configs['cnn_lstm']
            model = CNNLSTMModel(
                input_dim=input_dim,
                cnn_filters=cfg['cnn']['filters'],
                kernel_size=cfg['cnn']['kernel_size'],
                pool_size=cfg['cnn']['pool_size'],
                lstm_hidden=cfg['lstm']['hidden_size'],
                lstm_layers=cfg['lstm']['num_layers'],
                dropout=cfg['lstm']['dropout']
            )
        
        elif model_type == 'transformer':
            cfg = model_configs['transformer']
            model = TransformerModel(
                input_dim=input_dim,
                d_model=cfg['d_model'],
                nhead=cfg['nhead'],
                num_layers=cfg['num_encoder_layers'],
                dim_feedforward=cfg['dim_feedforward'],
                dropout=cfg['dropout']
            )
        
        elif model_type == 'mlp':
            model = MLPBaseline(
                input_dim=input_dim,
                hidden_dims=[128, 64],
                dropout=0.2
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def train_model(self, model: nn.Module, data: Dict, 
                    model_name: str, run_id: int) -> TrainingResult:
        """
        Train a single model.
        
        Args:
            model: PyTorch model
            data: Dictionary with DataLoaders
            model_name: Name of the model
            run_id: Run identifier for reproducibility
        
        Returns:
            TrainingResult with metrics
        """
        import time
        start_time = time.time()
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=self.train_config['learning_rate'])
        criterion = nn.MSELoss()
        
        # Early stopping
        patience = self.train_config['early_stopping']['patience']
        min_delta = self.train_config['early_stopping']['min_delta']
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        # Training history
        train_losses = []
        val_losses = []
        
        epochs = self.train_config['epochs']
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in data['train_loader']:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(data['train_loader'])
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in data['val_loader']:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(data['val_loader'])
            val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                break
            
            # Progress
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Test evaluation
        test_metrics = self.evaluate(model, data['test_loader'])
        
        training_time = time.time() - start_time
        
        # Get hyperparameters
        hyperparameters = self.config['baseline_models'].get(model_name, {})
        
        result = TrainingResult(
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
            hyperparameters=hyperparameters
        )
        
        return result, model
    
    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set."""
        model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)
                
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.numpy())
        
        preds = np.array(all_preds)
        targets = np.array(all_targets)
        
        # Compute metrics
        mae = np.mean(np.abs(preds - targets))
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        mape = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100
        
        # R2
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
                        feature_cols: List[str], 
                        models: List[str] = ['lstm', 'cnn_lstm', 'transformer'],
                        num_runs: int = 3) -> pd.DataFrame:
        """
        Run all baseline experiments.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            feature_cols: Feature columns
            models: List of model types to train
            num_runs: Number of runs per model
        
        Returns:
            DataFrame with results
        """
        print(f"\n{'=' * 80}")
        print("PHASE-2 STEP 3: BASELINE MODELS".center(80))
        print('=' * 80)
        
        all_results = []
        best_models = {}
        
        for model_type in models:
            print(f"\n{'─' * 80}")
            print(f"Training: {model_type.upper()}")
            print('─' * 80)
            
            model_results = []
            best_run_rmse = float('inf')
            
            for run in range(num_runs):
                # Set seed for reproducibility
                seed = self.config['random_seed'] + run
                set_seed(seed)
                
                print(f"\n  Run {run + 1}/{num_runs} (seed={seed})")
                
                # Determine if sequence mode needed
                sequence_mode = model_type in ['lstm', 'cnn_lstm', 'transformer']
                seq_length = 10 if sequence_mode else 1
                
                # Prepare data
                data = self.prepare_data(
                    train_df, test_df, feature_cols,
                    sequence_mode=sequence_mode,
                    seq_length=seq_length
                )
                
                # Create model
                model = self.create_model(model_type, data['input_dim'])
                
                # Train
                result, trained_model = self.train_model(model, data, model_type, run)
                model_results.append(result)
                all_results.append(result)
                
                print(f"  → MAE: {result.test_mae:.4f}, RMSE: {result.test_rmse:.4f}, R²: {result.test_r2:.4f}")
                
                # Save best model
                if result.test_rmse < best_run_rmse:
                    best_run_rmse = result.test_rmse
                    best_models[model_type] = trained_model
            
            # Summary for this model
            maes = [r.test_mae for r in model_results]
            rmses = [r.test_rmse for r in model_results]
            r2s = [r.test_r2 for r in model_results]
            
            print(f"\n  {model_type.upper()} Summary ({num_runs} runs):")
            print(f"    MAE:  {np.mean(maes):.4f} ± {np.std(maes):.4f}")
            print(f"    RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
            print(f"    R²:   {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
        
        # Store results
        self.results = all_results
        self.best_models = best_models
        
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
        """Save all results to disk."""
        print(f"\n{'─' * 80}")
        print("SAVING RESULTS")
        print('─' * 80)
        
        # Save results DataFrame
        results_path = os.path.join(self.output_dir, f'baseline_results_{self.timestamp}.csv')
        results_df.to_csv(results_path, index=False)
        print(f"  ✓ Results saved: {results_path}")
        
        # Save best models
        for model_name, model in self.best_models.items():
            model_path = os.path.join(self.output_dir, f'{model_name}_best_{self.timestamp}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': model_name,
                'timestamp': self.timestamp,
                'config': self.config['baseline_models'].get(model_name, {})
            }, model_path)
            print(f"  ✓ {model_name} model saved: {model_path}")
        
        # Save summary statistics
        summary = results_df.groupby('model').agg({
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'r2': ['mean', 'std']
        }).round(4)
        
        summary_path = os.path.join(self.output_dir, f'baseline_summary_{self.timestamp}.csv')
        summary.to_csv(summary_path)
        print(f"  ✓ Summary saved: {summary_path}")
        
        # Save detailed results as JSON
        detailed = {
            'timestamp': self.timestamp,
            'config': {
                'training': self.train_config,
                'models': self.config['baseline_models']
            },
            'results': [
                {
                    'model': r.model_name,
                    'run': r.run_id,
                    'mae': float(r.test_mae),
                    'rmse': float(r.test_rmse),
                    'mape': float(r.test_mape),
                    'r2': float(r.test_r2),
                    'best_epoch': r.best_epoch,
                    'training_time': r.training_time
                }
                for r in self.results
            ]
        }
        
        json_path = os.path.join(self.output_dir, f'baseline_detailed_{self.timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(detailed, f, indent=2)
        print(f"  ✓ Detailed results saved: {json_path}")
    
    def print_summary(self, results_df: pd.DataFrame) -> None:
        """Print final summary."""
        print(f"\n{'=' * 80}")
        print("BASELINE MODELS SUMMARY".center(80))
        print('=' * 80)
        
        summary = results_df.groupby('model').agg({
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'r2': ['mean', 'std'],
            'training_time': 'mean'
        })
        
        print("\nResults (mean ± std across runs):\n")
        print(f"{'Model':<15} {'MAE':<20} {'RMSE':<20} {'R²':<20} {'Time (s)':<10}")
        print('─' * 85)
        
        for model in summary.index:
            mae_mean = summary.loc[model, ('mae', 'mean')]
            mae_std = summary.loc[model, ('mae', 'std')]
            rmse_mean = summary.loc[model, ('rmse', 'mean')]
            rmse_std = summary.loc[model, ('rmse', 'std')]
            r2_mean = summary.loc[model, ('r2', 'mean')]
            r2_std = summary.loc[model, ('r2', 'std')]
            time_mean = summary.loc[model, ('training_time', 'mean')]
            
            print(f"{model:<15} {mae_mean:.4f} ± {mae_std:.4f}   {rmse_mean:.4f} ± {rmse_std:.4f}   {r2_mean:.4f} ± {r2_std:.4f}   {time_mean:.1f}")
        
        print('─' * 85)
        
        # Best model
        best_idx = results_df['rmse'].idxmin()
        best = results_df.loc[best_idx]
        print(f"\n✓ Best single run: {best['model']} (run {best['run']}) with RMSE = {best['rmse']:.4f}")
        
        print('\n' + '=' * 80)


def run_baselines(config: Dict, project_root: str,
                  train_path: str, test_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Run complete baseline experiments.
    
    Args:
        config: Configuration dictionary
        project_root: Project root path
        train_path: Path to training features CSV
        test_path: Path to test features CSV
    
    Returns:
        (results_df, best_models)
    """
    trainer = BaselineTrainer(config, project_root)
    
    # Load data
    train_df, test_df = trainer.load_data(train_path, test_path)
    
    # Define feature columns (exclude non-features)
    exclude_cols = ['battery_id', 'cycle_number', 'original_index', 'soh', 'has_spikes']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
    
    # Run experiments
    results_df = trainer.run_experiments(
        train_df, test_df, feature_cols,
        models=['lstm', 'cnn_lstm', 'transformer'],
        num_runs=config['training']['num_runs']
    )
    
    # Save results
    trainer.save_results(results_df)
    
    # Print summary
    trainer.print_summary(results_df)
    
    return results_df, trainer.best_models


if __name__ == "__main__":
    import yaml
    
    project_root = '.'
    
    # Load config
    with open(os.path.join(project_root, 'phase2_config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    
    # Paths to harmonized data
    train_path = os.path.join(project_root, 'data/phase2/harmonized/train_features.csv')
    test_path = os.path.join(project_root, 'data/phase2/harmonized/test_features.csv')
    
    # Run baselines
    results_df, best_models = run_baselines(config, project_root, train_path, test_path)
    
    print("\n" + "="*80)
    print("STEP 3 COMPLETE ✅")
    print("Proceed to Step 4: Multimodal Model")
    print("="*80)
