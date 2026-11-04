"""
Phase-2 Data Splits Module
Provides proper train/val/test splits for different evaluation scenarios
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os


def create_within_battery_split(features_df: pd.DataFrame, 
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
    """
    Create time-based split WITHIN each battery.
    
    This is the correct approach for validating model architectures,
    as it respects the temporal nature of degradation.
    
    Args:
        features_df: DataFrame with all batteries
        train_ratio: Fraction for training (default 0.7)
        val_ratio: Fraction for validation (default 0.15)
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
    """
    train_list = []
    val_list = []
    test_list = []
    
    for battery_id in features_df['battery_id'].unique():
        battery_df = features_df[features_df['battery_id'] == battery_id].copy()
        battery_df = battery_df.sort_values('cycle_number').reset_index(drop=True)
        
        n = len(battery_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_list.append(battery_df.iloc[:train_end])
        val_list.append(battery_df.iloc[train_end:val_end])
        test_list.append(battery_df.iloc[val_end:])
    
    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    print(f"Within-Battery Time-Split:")
    print(f"  Train: {len(train_df)} samples ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_df)} samples ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_df)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    for bat in features_df['battery_id'].unique():
        train_bat = train_df[train_df['battery_id'] == bat]
        test_bat = test_df[test_df['battery_id'] == bat]
        print(f"  {bat}: train={len(train_bat)}, test={len(test_bat)}")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


def create_cross_battery_split(features_df: pd.DataFrame,
                                train_batteries: List[str],
                                test_batteries: List[str],
                                val_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
    """
    Create cross-battery split for generalization testing.
    
    Train on some batteries, test on others.
    
    Args:
        features_df: DataFrame with all batteries
        train_batteries: List of battery IDs for training
        test_batteries: List of battery IDs for testing
        val_ratio: Fraction of training data for validation
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
    """
    train_df = features_df[features_df['battery_id'].isin(train_batteries)].copy()
    test_df = features_df[features_df['battery_id'].isin(test_batteries)].copy()
    
    # Split off validation from training
    val_size = int(len(train_df) * val_ratio)
    train_df = train_df.sort_values(['battery_id', 'cycle_number']).reset_index(drop=True)
    
    # Take validation from end of each training battery
    val_list = []
    train_list = []
    
    for bat in train_batteries:
        bat_df = train_df[train_df['battery_id'] == bat]
        n = len(bat_df)
        split = int(n * (1 - val_ratio))
        train_list.append(bat_df.iloc[:split])
        val_list.append(bat_df.iloc[split:])
    
    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    
    print(f"Cross-Battery Split:")
    print(f"  Train batteries: {train_batteries}")
    print(f"  Test batteries: {test_batteries}")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


def normalize_features(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                       test_df: pd.DataFrame, exclude_cols: List[str] = None,
                       method: str = 'z-score') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Normalize features using statistics from training set only.
    
    IMPORTANT: Use training statistics to normalize all sets to avoid data leakage.
    
    Args:
        train_df, val_df, test_df: DataFrames to normalize
        exclude_cols: Columns to exclude from normalization
        method: 'z-score' or 'minmax'
    
    Returns:
        (train_norm, val_norm, test_norm, norm_params)
    """
    if exclude_cols is None:
        exclude_cols = ['battery_id', 'cycle_number', 'original_index', 'soh', 'has_spikes']
    
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    norm_params = {}
    train_norm = train_df.copy()
    val_norm = val_df.copy()
    test_norm = test_df.copy()
    
    for col in feature_cols:
        train_values = train_df[col].values
        
        if method == 'z-score':
            mean = np.mean(train_values)
            std = np.std(train_values) + 1e-8
            
            train_norm[col] = (train_df[col] - mean) / std
            val_norm[col] = (val_df[col] - mean) / std
            test_norm[col] = (test_df[col] - mean) / std
            
            norm_params[col] = {'mean': mean, 'std': std}
        
        elif method == 'minmax':
            min_val = np.min(train_values)
            max_val = np.max(train_values)
            
            train_norm[col] = (train_df[col] - min_val) / (max_val - min_val + 1e-8)
            val_norm[col] = (val_df[col] - min_val) / (max_val - min_val + 1e-8)
            test_norm[col] = (test_df[col] - min_val) / (max_val - min_val + 1e-8)
            
            norm_params[col] = {'min': min_val, 'max': max_val}
    
    return train_norm, val_norm, test_norm, norm_params


def load_and_prepare_data(project_root: str, 
                          split_type: str = 'within_battery') -> Dict:
    """
    Load harmonized data and prepare proper splits.
    
    Args:
        project_root: Project root directory
        split_type: 'within_battery' or 'cross_battery'
    
    Returns:
        Dictionary with prepared data
    """
    # Load all harmonized features
    train_path = os.path.join(project_root, 'data/phase2/harmonized/train_features.csv')
    test_path = os.path.join(project_root, 'data/phase2/harmonized/test_features.csv')
    
    train_orig = pd.read_csv(train_path)
    test_orig = pd.read_csv(test_path)
    
    # Combine all data
    all_data = pd.concat([train_orig, test_orig], ignore_index=True)
    
    print(f"Total samples: {len(all_data)}")
    print(f"Batteries: {all_data['battery_id'].unique().tolist()}")
    
    if split_type == 'within_battery':
        splits = create_within_battery_split(all_data, train_ratio=0.7, val_ratio=0.15)
    elif split_type == 'cross_battery':
        splits = create_cross_battery_split(
            all_data,
            train_batteries=['B0005'],
            test_batteries=['B0006', 'B0007']
        )
    else:
        raise ValueError(f"Unknown split type: {split_type}")
    
    # Normalize using training statistics only
    exclude_cols = ['battery_id', 'cycle_number', 'original_index', 'soh', 'has_spikes']
    train_norm, val_norm, test_norm, norm_params = normalize_features(
        splits['train'], splits['val'], splits['test'], exclude_cols
    )
    
    feature_cols = [c for c in train_norm.columns if c not in exclude_cols]
    
    return {
        'train_df': train_norm,
        'val_df': val_norm,
        'test_df': test_norm,
        'feature_cols': feature_cols,
        'norm_params': norm_params,
        'split_type': split_type
    }


if __name__ == "__main__":
    project_root = '.'
    
    print("="*60)
    print("WITHIN-BATTERY SPLIT (for model validation)")
    print("="*60)
    data = load_and_prepare_data(project_root, 'within_battery')
    
    print(f"\nTrain SOH range: [{data['train_df']['soh'].min():.2f}, {data['train_df']['soh'].max():.2f}]")
    print(f"Test SOH range: [{data['test_df']['soh'].min():.2f}, {data['test_df']['soh'].max():.2f}]")
    
    print("\n" + "="*60)
    print("CROSS-BATTERY SPLIT (for generalization testing)")
    print("="*60)
    data2 = load_and_prepare_data(project_root, 'cross_battery')
    
    print(f"\nTrain SOH range: [{data2['train_df']['soh'].min():.2f}, {data2['train_df']['soh'].max():.2f}]")
    print(f"Test SOH range: [{data2['test_df']['soh'].min():.2f}, {data2['test_df']['soh'].max():.2f}]")
