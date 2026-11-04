"""
Combined Dataset: NASA + Mendeley

Unified data handling for multiple battery datasets with consistent
feature extraction and normalization.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pickle

# Import existing loaders
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import NASABatteryLoader
from phase3.mendeley_loader import MendeleyBatteryLoader
from phase2.data_harmonization import DataHarmonizer


@dataclass
class UnifiedCycle:
    """Unified cycle representation across datasets."""
    battery_id: str
    dataset: str  # 'NASA' or 'Mendeley'
    cycle_number: int
    voltage: np.ndarray
    current: np.ndarray
    temperature: np.ndarray
    time: np.ndarray
    capacity: float
    soh: float
    test_condition: str = ""


class CombinedDataset:
    """
    Combined dataset handler for NASA + Mendeley batteries.
    
    Features:
    - Unified cycle representation
    - Consistent feature extraction
    - Cross-dataset splitting
    - Data augmentation options
    """
    
    def __init__(self, config: Dict, project_root: str = None):
        """
        Initialize combined dataset.
        
        Args:
            config: Configuration dictionary
            project_root: Project root path
        """
        self.config = config
        self.project_root = project_root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.nasa_cycles = []
        self.mendeley_cycles = []
        self.all_cycles = []
        self.features_df = None
        
        self.nasa_initial_capacities = {}
        self.mendeley_initial_capacities = {}
        
    def load_nasa(self, battery_ids: List[str] = None) -> int:
        """
        Load NASA batteries.
        
        Args:
            battery_ids: List of battery IDs (e.g., ['B0005', 'B0006'])
            
        Returns:
            Number of cycles loaded
        """
        if battery_ids is None:
            battery_ids = ['B0005', 'B0006', 'B0007']
        
        nasa_path = os.path.join(self.project_root, 'NASA_DataSets')
        
        for bat_id in battery_ids:
            filepath = os.path.join(nasa_path, f'{bat_id}.mat')
            
            if not os.path.exists(filepath):
                print(f"⚠️ NASA battery not found: {filepath}")
                continue
            
            try:
                loader = NASABatteryLoader(filepath, bat_id)
                loader.load()
                
                discharge_cycles = loader.get_discharge_cycles()
                initial_cap = loader.get_initial_capacity()
                self.nasa_initial_capacities[bat_id] = initial_cap
                
                for cycle in discharge_cycles:
                    soh = (cycle['capacity'] / initial_cap) * 100 if initial_cap else 100
                    
                    unified = UnifiedCycle(
                        battery_id=bat_id,
                        dataset='NASA',
                        cycle_number=cycle['cycle_number'],
                        voltage=cycle['voltage'],
                        current=cycle['current'],
                        temperature=cycle.get('temperature', np.ones(len(cycle['voltage'])) * 25),
                        time=cycle['time'],
                        capacity=cycle['capacity'],
                        soh=min(soh, 100.0)
                    )
                    self.nasa_cycles.append(unified)
                
                print(f"✓ Loaded NASA {bat_id}: {len(discharge_cycles)} cycles")
                
            except Exception as e:
                print(f"✗ Error loading NASA {bat_id}: {e}")
        
        return len(self.nasa_cycles)
    
    def load_mendeley(self, data_path: str = None, battery_ids: List[str] = None) -> int:
        """
        Load Mendeley batteries.
        
        Args:
            data_path: Path to Mendeley data directory
            battery_ids: Optional list of specific battery files
            
        Returns:
            Number of cycles loaded
        """
        if data_path is None:
            data_path = os.path.join(self.project_root, 'Mendeley_DataSets')
        
        if not os.path.exists(data_path):
            print(f"⚠️ Mendeley data not found: {data_path}")
            print("  Please download from: https://data.mendeley.com/datasets/cp3473x7xv/3")
            return 0
        
        # Find CSV files
        import glob
        if battery_ids:
            files = [os.path.join(data_path, f'{bid}.csv') for bid in battery_ids]
        else:
            files = glob.glob(os.path.join(data_path, '*.csv'))
        
        for filepath in files:
            if not os.path.exists(filepath):
                continue
            
            bat_id = os.path.basename(filepath).replace('.csv', '')
            
            try:
                loader = MendeleyBatteryLoader(filepath, bat_id)
                loader.load()
                cycles = loader.get_discharge_cycles()
                
                if not cycles:
                    continue
                
                initial_cap = loader.initial_capacity
                self.mendeley_initial_capacities[bat_id] = initial_cap
                
                for cycle in cycles:
                    soh = (cycle.capacity / initial_cap) * 100 if initial_cap else 100
                    
                    unified = UnifiedCycle(
                        battery_id=bat_id,
                        dataset='Mendeley',
                        cycle_number=cycle.cycle_number,
                        voltage=cycle.voltage,
                        current=cycle.current,
                        temperature=cycle.temperature,
                        time=cycle.time,
                        capacity=cycle.capacity,
                        soh=min(soh, 100.0),
                        test_condition=cycle.test_condition
                    )
                    self.mendeley_cycles.append(unified)
                
                print(f"✓ Loaded Mendeley {bat_id}: {len(cycles)} cycles")
                
            except Exception as e:
                print(f"✗ Error loading Mendeley {bat_id}: {e}")
        
        return len(self.mendeley_cycles)
    
    def combine(self) -> List[UnifiedCycle]:
        """
        Combine all loaded cycles.
        
        Returns:
            List of all unified cycles
        """
        self.all_cycles = self.nasa_cycles + self.mendeley_cycles
        print(f"\n📊 Combined dataset: {len(self.all_cycles)} total cycles")
        print(f"   NASA: {len(self.nasa_cycles)} cycles")
        print(f"   Mendeley: {len(self.mendeley_cycles)} cycles")
        return self.all_cycles
    
    def extract_features(self, target_length: int = 500) -> pd.DataFrame:
        """
        Extract features from all cycles.
        
        Args:
            target_length: Target sequence length for resampling
            
        Returns:
            DataFrame with extracted features
        """
        if not self.all_cycles:
            self.combine()
        
        features_list = []
        
        for cycle in self.all_cycles:
            try:
                features = self._extract_cycle_features(cycle, target_length)
                features_list.append(features)
            except Exception as e:
                print(f"Skipping cycle {cycle.battery_id}_{cycle.cycle_number}: {e}")
        
        self.features_df = pd.DataFrame(features_list)
        print(f"✓ Extracted features for {len(self.features_df)} cycles")
        
        return self.features_df
    
    def _extract_cycle_features(self, cycle: UnifiedCycle, target_length: int) -> Dict:
        """Extract features from a single cycle."""
        # Resample to target length
        orig_len = len(cycle.voltage)
        if orig_len < 10:
            raise ValueError("Cycle too short")
        
        indices = np.linspace(0, orig_len - 1, target_length).astype(int)
        
        voltage = cycle.voltage[indices]
        current = cycle.current[indices]
        temperature = cycle.temperature[indices] if len(cycle.temperature) == orig_len else np.ones(target_length) * 25
        if len(temperature) != target_length:
            temp_indices = np.linspace(0, len(temperature) - 1, target_length).astype(int)
            temperature = cycle.temperature[temp_indices]
        
        # Compute ICA (dQ/dV)
        try:
            dv = np.diff(voltage)
            dv[dv == 0] = 1e-6
            dq = np.abs(np.diff(current)) * np.diff(cycle.time[indices]) / 3600
            ica = dq / np.abs(dv)
            ica = np.clip(ica, 0, 100)  # Limit outliers
        except:
            ica = np.zeros(target_length - 1)
        
        # DCIR estimation
        try:
            current_changes = np.abs(np.diff(current))
            voltage_changes = np.abs(np.diff(voltage))
            valid_mask = current_changes > 0.01
            if valid_mask.any():
                dcir = np.median(voltage_changes[valid_mask] / current_changes[valid_mask])
            else:
                dcir = 0.0
        except:
            dcir = 0.0
        
        features = {
            'battery_id': cycle.battery_id,
            'dataset': cycle.dataset,
            'cycle_number': cycle.cycle_number,
            'soh': cycle.soh,
            'capacity': cycle.capacity,
            
            # Voltage stats
            'voltage_mean': np.mean(voltage),
            'voltage_std': np.std(voltage),
            'voltage_min': np.min(voltage),
            'voltage_max': np.max(voltage),
            'voltage_range': np.max(voltage) - np.min(voltage),
            
            # Current stats
            'current_mean': np.mean(current),
            'current_std': np.std(current),
            
            # Temperature stats
            'temp_mean': np.mean(temperature),
            'temp_max': np.max(temperature),
            'temp_delta': np.max(temperature) - np.min(temperature),
            'temp_std': np.std(temperature),
            
            # ICA features
            'ica_peak_voltage': voltage[np.argmax(ica) + 1] if len(ica) > 0 else 0,
            'ica_peak_magnitude': np.max(ica) if len(ica) > 0 else 0,
            'ica_mean': np.mean(ica) if len(ica) > 0 else 0,
            'ica_std': np.std(ica) if len(ica) > 0 else 0,
            
            # DCIR
            'dcir': dcir,
            
            # Time
            'discharge_time': cycle.time[-1] - cycle.time[0] if len(cycle.time) > 1 else 0,
            
            # Test condition (for Mendeley)
            'test_condition': cycle.test_condition
        }
        
        return features
    
    def create_splits(self, 
                      strategy: str = 'cross_dataset',
                      train_datasets: List[str] = None,
                      test_datasets: List[str] = None,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Create train/val/test splits.
        
        Args:
            strategy: 'cross_dataset', 'cross_battery', or 'within_battery'
            train_datasets: Datasets for training (for cross_dataset strategy)
            test_datasets: Datasets for testing (for cross_dataset strategy)
            train_ratio: Training ratio (for within_battery)
            val_ratio: Validation ratio (for within_battery)
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        if self.features_df is None:
            self.extract_features()
        
        if strategy == 'cross_dataset':
            # Train on NASA, test on Mendeley (or vice versa)
            train_datasets = train_datasets or ['NASA']
            test_datasets = test_datasets or ['Mendeley']
            
            train_df = self.features_df[self.features_df['dataset'].isin(train_datasets)]
            test_df = self.features_df[self.features_df['dataset'].isin(test_datasets)]
            
            # Split train into train/val
            n_train = int(len(train_df) * (1 - val_ratio / (train_ratio + val_ratio)))
            train_df = train_df.sort_values(['battery_id', 'cycle_number'])
            val_df = train_df.iloc[n_train:]
            train_df = train_df.iloc[:n_train]
            
        elif strategy == 'cross_battery':
            # Leave-one-battery-out or specified split
            batteries = self.features_df['battery_id'].unique()
            n_train = int(len(batteries) * train_ratio)
            
            train_batteries = batteries[:n_train]
            test_batteries = batteries[n_train:]
            
            train_df = self.features_df[self.features_df['battery_id'].isin(train_batteries)]
            test_df = self.features_df[self.features_df['battery_id'].isin(test_batteries)]
            
            n_val = int(len(train_df) * val_ratio)
            val_df = train_df.iloc[-n_val:]
            train_df = train_df.iloc[:-n_val]
            
        else:  # within_battery
            # Time-based split within each battery
            train_list, val_list, test_list = [], [], []
            
            for bat_id in self.features_df['battery_id'].unique():
                bat_df = self.features_df[self.features_df['battery_id'] == bat_id]
                bat_df = bat_df.sort_values('cycle_number')
                
                n = len(bat_df)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                
                train_list.append(bat_df.iloc[:n_train])
                val_list.append(bat_df.iloc[n_train:n_train+n_val])
                test_list.append(bat_df.iloc[n_train+n_val:])
            
            train_df = pd.concat(train_list, ignore_index=True)
            val_df = pd.concat(val_list, ignore_index=True)
            test_df = pd.concat(test_list, ignore_index=True)
        
        print(f"\n📊 Split ({strategy}):")
        print(f"   Train: {len(train_df)} samples")
        print(f"   Val: {len(val_df)} samples")
        print(f"   Test: {len(test_df)} samples")
        
        return {'train': train_df, 'val': val_df, 'test': test_df}
    
    def save(self, output_dir: str):
        """Save combined dataset to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.features_df is not None:
            self.features_df.to_csv(os.path.join(output_dir, 'combined_features.csv'), index=False)
        
        # Save metadata
        metadata = {
            'nasa_batteries': list(self.nasa_initial_capacities.keys()),
            'mendeley_batteries': list(self.mendeley_initial_capacities.keys()),
            'n_nasa_cycles': len(self.nasa_cycles),
            'n_mendeley_cycles': len(self.mendeley_cycles),
            'n_total_cycles': len(self.all_cycles)
        }
        
        with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Saved combined dataset to {output_dir}")


if __name__ == "__main__":
    print("Combined Dataset Handler")
    print("=" * 50)
    print("\nUsage:")
    print("  dataset = CombinedDataset(config)")
    print("  dataset.load_nasa(['B0005', 'B0006', 'B0007'])")
    print("  dataset.load_mendeley('/path/to/mendeley')")
    print("  dataset.combine()")
    print("  features = dataset.extract_features()")
    print("  splits = dataset.create_splits(strategy='within_battery')")
