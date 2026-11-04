"""
Baseline CNN-LSTM Model for SOH Prediction
Phase-1: Numeric-only baseline before multimodal fusion
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import pickle


class BatteryDataset(Dataset):
    """Dataset for battery SOH prediction with sequence data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, lookback: int = 10):
        """
        Args:
            features: Feature array (n_cycles, n_features)
            targets: SOH targets (n_cycles,)
            lookback: Number of past cycles to use for prediction
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.lookback = lookback
    
    def __len__(self):
        return len(self.features) - self.lookback
    
    def __getitem__(self, idx):
        # Get sequence of past lookback cycles
        x = self.features[idx:idx + self.lookback]
        # Predict SOH at current cycle (end of sequence)
        y = self.targets[idx + self.lookback]
        
        return x, y


class CNN_LSTM(nn.Module):
    """
    CNN-LSTM architecture for SOH prediction.
    
    Architecture:
    1. CNN layers: Extract spatial patterns from cycle sequences
    2. LSTM layers: Capture temporal degradation dynamics
    3. Fully connected: Regression to SOH value
    """
    
    def __init__(self, input_size: int, config: Dict):
        """
        Args:
            input_size: Number of features per cycle
            config: Model configuration dictionary
        """
        super(CNN_LSTM, self).__init__()
        
        model_config = config['model']['architecture']
        cnn_config = model_config['cnn']
        lstm_config = model_config['lstm']
        fc_config = model_config['fc']
        
        # CNN layers (1D convolution over cycle sequence)
        cnn_filters = cnn_config['filters']
        kernel_size = cnn_config['kernel_size']
        
        self.conv1 = nn.Conv1d(input_size, cnn_filters[0], kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(cnn_filters[0])
        self.pool1 = nn.MaxPool1d(cnn_config['pool_size'])
        
        self.conv2 = nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(cnn_filters[1])
        self.pool2 = nn.MaxPool1d(cnn_config['pool_size'])
        
        self.conv3 = nn.Conv1d(cnn_filters[1], cnn_filters[2], kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm1d(cnn_filters[2])
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_filters[2],
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            dropout=lstm_config['dropout'],
            batch_first=True
        )
        
        # Fully connected layers
        fc_dims = fc_config['hidden_dims']
        self.fc1 = nn.Linear(lstm_config['hidden_size'], fc_dims[0])
        self.fc2 = nn.Linear(fc_dims[0], fc_dims[1])
        self.fc3 = nn.Linear(fc_dims[1], 1)  # Output: SOH prediction
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(lstm_config['dropout'])
    
    def forward(self, x):
        # Input: (batch, lookback, features)
        # Reshape for CNN: (batch, features, lookback)
        x = x.permute(0, 2, 1)
        
        # CNN layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Reshape for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # LSTM layers
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last hidden state
        x = h_n[-1]  # (batch, hidden_size)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x.squeeze()


class BaselineTrainer:
    """Trainer for baseline CNN-LSTM model."""
    
    def __init__(self, config: Dict, device: str = None):
        """
        Args:
            config: Configuration dictionary
            device: 'cuda' or 'cpu', auto-detected if None
        """
        self.config = config
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.train_losses = []
        self.val_losses = []
    
    def prepare_data(self, features_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders for training.
        
        Args:
            features_df: Features DataFrame
        
        Returns:
            train_loader, test_loader
        """
        print(f"\n{'─' * 80}")
        print("PREPARING DATA FOR TRAINING")
        print('─' * 80)
        
        # Split data
        split_config = self.config['split']
        train_ratio = split_config['train_ratio']
        
        n_samples = len(features_df)
        n_train = int(n_samples * train_ratio)
        
        print(f"\nTime-based split:")
        print(f"  Train: cycles 0-{n_train-1} ({n_train} cycles, {train_ratio*100:.0f}%)")
        print(f"  Test: cycles {n_train}-{n_samples-1} ({n_samples-n_train} cycles, {(1-train_ratio)*100:.0f}%)")
        
        # Separate features and target
        self.feature_cols = [col for col in features_df.columns if col not in ['cycle_number', 'cycle_index', 'soh']]
        
        train_df = features_df.iloc[:n_train]
        test_df = features_df.iloc[n_train:]
        
        X_train = train_df[self.feature_cols].values
        y_train = train_df['soh'].values
        
        X_test = test_df[self.feature_cols].values
        y_test = test_df['soh'].values
        
        # Normalize features
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"\n✓ Features normalized using MinMaxScaler")
        print(f"  Number of features: {len(self.feature_cols)}")
        
        # Create datasets
        lookback = self.config['sequence']['lookback_cycles']
        
        train_dataset = BatteryDataset(X_train, y_train, lookback=lookback)
        test_dataset = BatteryDataset(X_test, y_test, lookback=lookback)
        
        print(f"\n✓ Created sequence datasets (lookback={lookback} cycles)")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        
        # Create data loaders
        batch_size = self.config['model']['training']['batch_size']
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def build_model(self, input_size: int) -> None:
        """Build and initialize model."""
        self.model = CNN_LSTM(input_size, self.config).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n{'─' * 80}")
        print("MODEL ARCHITECTURE")
        print('─' * 80)
        print(self.model)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print(f"\n{'=' * 80}")
        print("TRAINING BASELINE MODEL".center(80))
        print('=' * 80)
        
        training_config = self.config['model']['training']
        epochs = training_config['epochs']
        lr = training_config['learning_rate']
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Early stopping
        early_stop_config = training_config['early_stopping']
        patience = early_stop_config['patience']
        min_delta = early_stop_config['min_delta']
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"\nTraining configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {lr}")
        print(f"  Batch size: {training_config['batch_size']}")
        print(f"  Early stopping patience: {patience}")
        print()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✓ Restored best model from epoch {epoch + 1 - patience_counter}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary with metrics (MAE, RMSE, MAPE)
        """
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mae = np.mean(np.abs(actuals - predictions))
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        return metrics, predictions, actuals
    
    def save_model(self, save_dir: str, model_name: str = 'baseline_cnn_lstm') -> None:
        """Save model, scaler, and metadata."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, f'{model_name}.pt')
        torch.save(self.model.state_dict(), model_path)
        
        # Save scaler
        scaler_path = os.path.join(save_dir, f'{model_name}_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata_path = os.path.join(save_dir, f'{model_name}_metadata.pkl')
        metadata = {
            'feature_cols': self.feature_cols,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\n✓ Model saved to: {save_dir}")


if __name__ == "__main__":
    # Test training
    import sys
    sys.path.append('..')
    from utils import load_config
    
    config = load_config('../config.yaml')
    
    # Load features
    features_path = os.path.join(config['paths']['features'], 'features_B0005.csv')
    features_df = pd.read_csv(features_path)
    
    # Train model
    trainer = BaselineTrainer(config)
    train_loader, test_loader = trainer.prepare_data(features_df)
    
    input_size = len(trainer.feature_cols)
    trainer.build_model(input_size)
    trainer.train(train_loader, test_loader)
    
    # Evaluate
    metrics, preds, actuals = trainer.evaluate(test_loader)
    print(f"\n{'=' * 80}")
    print("BASELINE MODEL METRICS".center(80))
    print('=' * 80)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model
    trainer.save_model(config['paths']['models'])
