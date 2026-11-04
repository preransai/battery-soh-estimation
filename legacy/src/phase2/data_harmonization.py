"""
Phase-2 Step 2: Multi-Dataset Harmonization
Standardize data across NASA batteries for consistent training/testing
"""

import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import pickle
from datetime import datetime

from data_loader import NASABatteryLoader, CycleData
from cycle_segmentation import CycleSegmentation
from feature_extraction import FeatureExtractor


@dataclass
class HarmonizedCycle:
    """Standardized cycle data after harmonization."""
    battery_id: str
    cycle_number: int
    original_cycle_index: int
    
    # Raw time-series (resampled to common rate)
    time: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    temperature: np.ndarray
    
    # Computed values
    capacity: float
    soh: float
    
    # Quality flags
    has_spikes: bool
    is_complete: bool
    data_quality_score: float  # 0-1, higher is better


class DataHarmonizer:
    """
    Harmonize battery data across different sources.
    
    Standardizes:
    - Sampling rates
    - Voltage/current normalization
    - Cycle numbering
    - Feature computation
    """
    
    def __init__(self, config: Dict, project_root: str = None):
        """
        Initialize harmonizer.
        
        Args:
            config: Configuration dictionary
            project_root: Project root directory for resolving paths
        """
        self.config = config
        self.harm_config = config['harmonization']
        self.project_root = project_root or '.'
        
        # Storage
        self.batteries = {}  # battery_id -> list of HarmonizedCycle
        self.initial_capacities = {}  # battery_id -> initial capacity
        self.features_df = None
        
        # Output directory
        self.output_dir = os.path.join(self.project_root, config['paths']['harmonized_data'])
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_and_harmonize_nasa(self, battery_configs: List[Dict]) -> Dict[str, List[HarmonizedCycle]]:
        """
        Load and harmonize multiple NASA batteries.
        
        Args:
            battery_configs: List of battery configuration dicts
        
        Returns:
            Dictionary mapping battery_id to list of harmonized cycles
        """
        print(f"\n{'=' * 80}")
        print("PHASE-2 STEP 2: MULTI-DATASET HARMONIZATION".center(80))
        print('=' * 80)
        
        for battery_config in battery_configs:
            battery_id = battery_config['battery_id']
            battery_path = battery_config['path']
            
            # Resolve path
            if not os.path.isabs(battery_path):
                battery_path = os.path.join(self.project_root, battery_path)
            
            print(f"\n{'─' * 80}")
            print(f"Processing: {battery_id}")
            print('─' * 80)
            
            # Load battery
            harmonized = self._harmonize_nasa_battery(battery_path, battery_id)
            self.batteries[battery_id] = harmonized
            
            print(f"  ✓ Harmonized {len(harmonized)} cycles")
        
        # Compute global statistics
        self._compute_global_stats()
        
        return self.batteries
    
    def _harmonize_nasa_battery(self, battery_path: str, battery_id: str) -> List[HarmonizedCycle]:
        """Harmonize a single NASA battery."""
        
        # Load raw data
        loader = NASABatteryLoader(battery_path, battery_id)
        loader.load()
        discharge_cycles = loader.get_discharge_cycles()
        
        print(f"  Loaded {len(discharge_cycles)} discharge cycles")
        
        # Compute initial capacity (average of first 5 cycles)
        capacities = [c.capacity for c in discharge_cycles[:5] if c.capacity is not None]
        initial_capacity = np.mean(capacities) if capacities else 2.0
        self.initial_capacities[battery_id] = initial_capacity
        
        print(f"  Initial capacity: {initial_capacity:.4f} Ah")
        
        # Process each cycle
        harmonized_cycles = []
        
        for i, cycle in enumerate(discharge_cycles):
            # Skip cycles with missing data
            if cycle.capacity is None or cycle.capacity <= 0:
                continue
            
            # Resample to common time base
            resampled = self._resample_cycle(cycle)
            
            if resampled is None:
                continue
            
            # Compute quality score
            quality_score, has_spikes = self._compute_quality_score(cycle)
            
            # Compute SOH
            soh = (cycle.capacity / initial_capacity) * 100.0
            
            # Create harmonized cycle
            harm_cycle = HarmonizedCycle(
                battery_id=battery_id,
                cycle_number=i,
                original_cycle_index=cycle.cycle_index,
                time=resampled['time'],
                voltage=resampled['voltage'],
                current=resampled['current'],
                temperature=resampled['temperature'],
                capacity=cycle.capacity,
                soh=soh,
                has_spikes=has_spikes,
                is_complete=True,
                data_quality_score=quality_score
            )
            
            harmonized_cycles.append(harm_cycle)
        
        print(f"  Processed {len(harmonized_cycles)} valid cycles")
        
        return harmonized_cycles
    
    def _resample_cycle(self, cycle: CycleData, target_length: int = 500) -> Optional[Dict]:
        """
        Resample cycle to common length.
        
        Args:
            cycle: Original cycle data
            target_length: Target number of samples
        
        Returns:
            Dictionary with resampled arrays or None if failed
        """
        try:
            # Original time vector
            t_orig = cycle.time - cycle.time[0]  # Start from 0
            
            if len(t_orig) < 10:
                return None
            
            # Target time vector (evenly spaced)
            t_new = np.linspace(0, t_orig[-1], target_length)
            
            # Interpolate each signal
            f_v = interp1d(t_orig, cycle.voltage, kind='linear', fill_value='extrapolate')
            f_i = interp1d(t_orig, cycle.current, kind='linear', fill_value='extrapolate')
            f_t = interp1d(t_orig, cycle.temperature, kind='linear', fill_value='extrapolate')
            
            return {
                'time': t_new,
                'voltage': f_v(t_new),
                'current': f_i(t_new),
                'temperature': f_t(t_new)
            }
        
        except Exception as e:
            print(f"    Warning: Resampling failed for cycle {cycle.cycle_index}: {e}")
            return None
    
    def _compute_quality_score(self, cycle: CycleData) -> Tuple[float, bool]:
        """
        Compute data quality score for a cycle.
        
        Returns:
            (quality_score, has_spikes)
        """
        score = 1.0
        
        # Check for spikes
        spike_threshold = self.config['audit']['max_spike_threshold']
        voltage_diff = np.abs(np.diff(cycle.voltage))
        max_spike = np.max(voltage_diff)
        has_spikes = max_spike > spike_threshold
        
        if has_spikes:
            score -= 0.2
        
        # Check data point count
        if cycle.data_points < 200:
            score -= 0.2
        elif cycle.data_points < 100:
            score -= 0.4
        
        # Check voltage range
        v_range = cycle.voltage.max() - cycle.voltage.min()
        if v_range < 1.0:  # Less than 1V range - partial discharge
            score -= 0.3
        
        return max(0.0, score), has_spikes
    
    def _compute_global_stats(self) -> None:
        """Compute statistics across all batteries."""
        print(f"\n{'─' * 80}")
        print("GLOBAL STATISTICS")
        print('─' * 80)
        
        total_cycles = sum(len(cycles) for cycles in self.batteries.values())
        print(f"  Total cycles across all batteries: {total_cycles}")
        
        for battery_id, cycles in self.batteries.items():
            soh_values = [c.soh for c in cycles]
            quality_scores = [c.data_quality_score for c in cycles]
            
            print(f"\n  {battery_id}:")
            print(f"    Cycles: {len(cycles)}")
            print(f"    SOH range: [{min(soh_values):.2f}, {max(soh_values):.2f}]%")
            print(f"    Mean quality score: {np.mean(quality_scores):.2f}")
            print(f"    Cycles with spikes: {sum(1 for c in cycles if c.has_spikes)}")
    
    def extract_features(self) -> pd.DataFrame:
        """
        Extract physics-guided features from all harmonized cycles.
        
        Returns:
            DataFrame with features for all cycles
        """
        print(f"\n{'─' * 80}")
        print("EXTRACTING PHYSICS-GUIDED FEATURES")
        print('─' * 80)
        
        all_features = []
        
        for battery_id, cycles in self.batteries.items():
            print(f"\n  Processing {battery_id}...")
            
            initial_capacity = self.initial_capacities[battery_id]
            
            for cycle in cycles:
                features = self._extract_cycle_features(cycle, initial_capacity)
                features['battery_id'] = battery_id
                all_features.append(features)
        
        self.features_df = pd.DataFrame(all_features)
        
        # Reorder columns
        cols = ['battery_id', 'cycle_number', 'soh', 'capacity'] + \
               [c for c in self.features_df.columns if c not in ['battery_id', 'cycle_number', 'soh', 'capacity']]
        self.features_df = self.features_df[cols]
        
        print(f"\n  ✓ Extracted {len(self.features_df)} feature vectors")
        print(f"  ✓ Features per cycle: {len(self.features_df.columns)}")
        
        return self.features_df
    
    def _extract_cycle_features(self, cycle: HarmonizedCycle, initial_capacity: float) -> Dict:
        """Extract features from a single harmonized cycle."""
        features = {
            'cycle_number': cycle.cycle_number,
            'original_index': cycle.original_cycle_index,
            'soh': cycle.soh,
            'capacity': cycle.capacity,
            'data_quality': cycle.data_quality_score,
            'has_spikes': int(cycle.has_spikes),
        }
        
        # Voltage features
        features['voltage_mean'] = np.mean(cycle.voltage)
        features['voltage_std'] = np.std(cycle.voltage)
        features['voltage_min'] = np.min(cycle.voltage)
        features['voltage_max'] = np.max(cycle.voltage)
        features['voltage_range'] = features['voltage_max'] - features['voltage_min']
        
        # Current features
        features['current_mean'] = np.mean(np.abs(cycle.current))
        features['current_std'] = np.std(cycle.current)
        
        # Temperature features
        features['temp_mean'] = np.mean(cycle.temperature)
        features['temp_max'] = np.max(cycle.temperature)
        features['temp_delta'] = np.max(cycle.temperature) - np.min(cycle.temperature)
        features['temp_std'] = np.std(cycle.temperature)
        
        # Discharge time
        features['discharge_time'] = cycle.time[-1] - cycle.time[0]
        
        # ICA features (simplified)
        ica_features = self._compute_ica_features(cycle)
        features.update(ica_features)
        
        # DCIR approximation
        features['dcir'] = self._compute_dcir(cycle)
        
        return features
    
    def _compute_ica_features(self, cycle: HarmonizedCycle) -> Dict:
        """Compute ICA (dQ/dV) features."""
        try:
            # Compute cumulative charge
            dt = np.diff(cycle.time, prepend=cycle.time[0])
            charge = np.cumsum(np.abs(cycle.current) * dt) / 3600.0  # Ah
            
            # Sort by voltage
            sort_idx = np.argsort(cycle.voltage)
            v_sorted = cycle.voltage[sort_idx]
            q_sorted = charge[sort_idx]
            
            # Smooth
            if len(q_sorted) > 11:
                q_smooth = savgol_filter(q_sorted, 11, 3)
            else:
                q_smooth = q_sorted
            
            # Compute dQ/dV
            dqdv = np.gradient(q_smooth, v_sorted)
            
            # Smooth dQ/dV
            if len(dqdv) > 11:
                dqdv_smooth = savgol_filter(dqdv, 11, 3)
            else:
                dqdv_smooth = dqdv
            
            # Find peak
            peak_idx = np.argmax(np.abs(dqdv_smooth))
            
            return {
                'ica_peak_voltage': v_sorted[peak_idx],
                'ica_peak_magnitude': np.abs(dqdv_smooth[peak_idx]),
                'ica_mean': np.mean(np.abs(dqdv_smooth)),
                'ica_std': np.std(dqdv_smooth)
            }
        
        except Exception:
            return {
                'ica_peak_voltage': 0.0,
                'ica_peak_magnitude': 0.0,
                'ica_mean': 0.0,
                'ica_std': 0.0
            }
    
    def _compute_dcir(self, cycle: HarmonizedCycle) -> float:
        """Compute DC internal resistance approximation."""
        try:
            current_abs = np.abs(cycle.current)
            
            # Find significant current region
            high_current_mask = current_abs > 0.5
            
            if np.sum(high_current_mask) < 10:
                return 0.0
            
            # Find largest current change
            current_diff = np.diff(current_abs)
            max_jump_idx = np.argmax(np.abs(current_diff))
            
            if max_jump_idx > 0 and max_jump_idx < len(cycle.voltage) - 1:
                delta_v = np.abs(cycle.voltage[max_jump_idx] - cycle.voltage[max_jump_idx + 1])
                delta_i = np.abs(current_abs[max_jump_idx + 1] - current_abs[max_jump_idx])
                
                if delta_i > 0.1:
                    return delta_v / delta_i
            
            return 0.0
        
        except Exception:
            return 0.0
    
    def normalize_features(self, method: str = 'z-score') -> pd.DataFrame:
        """
        Normalize features for model training.
        
        Args:
            method: 'z-score' or 'minmax'
        
        Returns:
            Normalized features DataFrame
        """
        if self.features_df is None:
            raise ValueError("Must extract features first!")
        
        # Columns to normalize (exclude identifiers and target)
        exclude_cols = ['battery_id', 'cycle_number', 'original_index', 'soh', 'has_spikes']
        feature_cols = [c for c in self.features_df.columns if c not in exclude_cols]
        
        df_normalized = self.features_df.copy()
        
        self.normalization_params = {}
        
        for col in feature_cols:
            values = df_normalized[col].values
            
            if method == 'z-score':
                mean = np.mean(values)
                std = np.std(values) + 1e-8
                df_normalized[col] = (values - mean) / std
                self.normalization_params[col] = {'mean': mean, 'std': std, 'method': 'z-score'}
            
            elif method == 'minmax':
                min_val = np.min(values)
                max_val = np.max(values)
                df_normalized[col] = (values - min_val) / (max_val - min_val + 1e-8)
                self.normalization_params[col] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
        
        print(f"\n  ✓ Normalized {len(feature_cols)} features using {method}")
        
        return df_normalized
    
    def create_train_test_split(self, train_battery: str = 'B0005', 
                                 test_batteries: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split by battery.
        
        Args:
            train_battery: Battery ID for training
            test_batteries: List of battery IDs for testing
        
        Returns:
            (train_df, test_df)
        """
        if self.features_df is None:
            raise ValueError("Must extract features first!")
        
        if test_batteries is None:
            test_batteries = [b for b in self.batteries.keys() if b != train_battery]
        
        train_df = self.features_df[self.features_df['battery_id'] == train_battery].copy()
        test_df = self.features_df[self.features_df['battery_id'].isin(test_batteries)].copy()
        
        print(f"\n  Train/Test Split:")
        print(f"    Train: {train_battery} ({len(train_df)} cycles)")
        print(f"    Test: {test_batteries} ({len(test_df)} cycles)")
        
        return train_df, test_df
    
    def save_harmonized_data(self) -> None:
        """Save all harmonized data to disk."""
        print(f"\n{'─' * 80}")
        print("SAVING HARMONIZED DATA")
        print('─' * 80)
        
        # Save features DataFrame
        features_path = os.path.join(self.output_dir, f'harmonized_features_{self.timestamp}.csv')
        self.features_df.to_csv(features_path, index=False)
        print(f"  ✓ Features saved: {features_path}")
        
        # Save normalization parameters
        if hasattr(self, 'normalization_params'):
            norm_path = os.path.join(self.output_dir, f'normalization_params_{self.timestamp}.pkl')
            with open(norm_path, 'wb') as f:
                pickle.dump(self.normalization_params, f)
            print(f"  ✓ Normalization params saved: {norm_path}")
        
        # Save initial capacities
        cap_path = os.path.join(self.output_dir, f'initial_capacities_{self.timestamp}.pkl')
        with open(cap_path, 'wb') as f:
            pickle.dump(self.initial_capacities, f)
        print(f"  ✓ Initial capacities saved: {cap_path}")
        
        # Save harmonized cycles (pickle for full data)
        cycles_path = os.path.join(self.output_dir, f'harmonized_cycles_{self.timestamp}.pkl')
        with open(cycles_path, 'wb') as f:
            pickle.dump(self.batteries, f)
        print(f"  ✓ Harmonized cycles saved: {cycles_path}")
        
        print(f"\n  All data saved to: {self.output_dir}")
    
    def print_summary(self) -> None:
        """Print harmonization summary."""
        print(f"\n{'=' * 80}")
        print("HARMONIZATION SUMMARY".center(80))
        print('=' * 80)
        
        print(f"\nBatteries processed: {len(self.batteries)}")
        
        for battery_id, cycles in self.batteries.items():
            print(f"\n{battery_id}:")
            print(f"  Cycles: {len(cycles)}")
            print(f"  Initial capacity: {self.initial_capacities[battery_id]:.4f} Ah")
            
            soh_values = [c.soh for c in cycles]
            print(f"  SOH range: {min(soh_values):.2f}% → {max(soh_values):.2f}%")
            
            quality = [c.data_quality_score for c in cycles]
            print(f"  Quality score: {np.mean(quality):.2f} (mean)")
        
        if self.features_df is not None:
            print(f"\nFeatures extracted: {len(self.features_df.columns)} per cycle")
            print(f"Total samples: {len(self.features_df)}")
        
        print('\n' + '=' * 80)


def run_harmonization(config: Dict, project_root: str = None) -> DataHarmonizer:
    """
    Run complete harmonization pipeline.
    
    Args:
        config: Configuration dictionary
        project_root: Project root directory
    
    Returns:
        DataHarmonizer instance with processed data
    """
    harmonizer = DataHarmonizer(config, project_root)
    
    # Get battery configs
    battery_configs = [
        config['datasets']['nasa_train'],
        config['datasets']['nasa_test_1'],
        config['datasets']['nasa_test_2']
    ]
    
    # Load and harmonize
    harmonizer.load_and_harmonize_nasa(battery_configs)
    
    # Extract features
    harmonizer.extract_features()
    
    # Normalize
    harmonizer.normalize_features(method='z-score')
    
    # Create train/test split
    train_df, test_df = harmonizer.create_train_test_split(
        train_battery='B0005',
        test_batteries=['B0006', 'B0007']
    )
    
    # Save everything
    harmonizer.save_harmonized_data()
    
    # Print summary
    harmonizer.print_summary()
    
    return harmonizer


if __name__ == "__main__":
    import yaml
    
    project_root = '.'
    
    # Load config
    with open(os.path.join(project_root, 'phase2_config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    
    # Run harmonization
    harmonizer = run_harmonization(config, project_root)
    
    print("\n" + "="*80)
    print("STEP 2 COMPLETE ✅")
    print("Proceed to Step 3: Baseline Models")
    print("="*80)
