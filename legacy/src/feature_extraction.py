"""
Physics-Guided Feature Extraction for Battery SOH Prediction
- SOH calculation
- ICA (Incremental Capacity Analysis): dQ/dV
- DCIR (DC Internal Resistance)
- Temperature features (ΔT, mean, max, std)
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional
import pandas as pd
from data_loader import CycleData


class FeatureExtractor:
    """
    Extract physics-guided features from battery cycles.
    
    Features extracted per cycle:
    - SOH (State of Health) %
    - ICA peak positions and magnitudes
    - DCIR (approximate)
    - Temperature statistics (ΔT, mean, max, std)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_config = config['features']
        
        # SOH configuration
        self.rated_capacity = config['dataset']['rated_capacity']
        self.initial_capacity_cycles = self.feature_config['soh']['initial_capacity_cycles']
        self.initial_capacity = None
        
        # ICA configuration
        self.ica_config = self.feature_config['ica']
        self.smoothing_method = self.ica_config['smoothing']
        self.window_length = self.ica_config['window_length']
        self.polyorder = self.ica_config['polyorder']
        self.voltage_range = self.ica_config['voltage_range']
        
        # DCIR configuration
        self.dcir_config = self.feature_config['dcir']
        self.current_threshold = self.dcir_config['current_threshold']
    
    def extract_all_features(self, cycles: List[CycleData]) -> pd.DataFrame:
        """
        Extract all features from a list of cycles.
        
        Args:
            cycles: List of CycleData objects
        
        Returns:
            DataFrame with all features per cycle
        """
        print(f"\n{'=' * 80}")
        print("EXTRACTING PHYSICS-GUIDED FEATURES".center(80))
        print('=' * 80)
        
        # Calculate initial capacity from first N cycles
        self._compute_initial_capacity(cycles)
        
        features_list = []
        
        for i, cycle in enumerate(cycles):
            if i % 20 == 0:
                print(f"Processing cycle {i+1}/{len(cycles)}...")
            
            features = self._extract_cycle_features(cycle, cycle_number=i)
            features_list.append(features)
        
        # Create DataFrame
        df = pd.DataFrame(features_list)
        
        print(f"\n✓ Extracted features for {len(cycles)} cycles")
        print(f"✓ Feature dimensions: {df.shape}")
        print(f"✓ Features: {list(df.columns)}")
        
        return df
    
    def _compute_initial_capacity(self, cycles: List[CycleData]) -> None:
        """Compute initial capacity as average of first N cycles."""
        n = min(self.initial_capacity_cycles, len(cycles))
        initial_capacities = [cycles[i].capacity for i in range(n)]
        self.initial_capacity = np.mean(initial_capacities)
        
        print(f"\n📊 Initial capacity: {self.initial_capacity:.4f} Ah")
        print(f"   (averaged from first {n} cycles)")
    
    def _extract_cycle_features(self, cycle: CycleData, cycle_number: int) -> Dict:
        """Extract all features from a single cycle."""
        features = {
            'cycle_number': cycle_number,
            'cycle_index': cycle.cycle_index,
        }
        
        # 1. SOH
        soh = self._compute_soh(cycle)
        features['soh'] = soh
        features['capacity'] = cycle.capacity
        
        # 2. ICA features
        ica_features = self._compute_ica_features(cycle)
        features.update(ica_features)
        
        # 3. DCIR
        dcir = self._compute_dcir(cycle)
        features['dcir'] = dcir
        
        # 4. Temperature features
        temp_features = self._compute_temperature_features(cycle)
        features.update(temp_features)
        
        # 5. Basic statistics
        basic_features = self._compute_basic_features(cycle)
        features.update(basic_features)
        
        return features
    
    def _compute_soh(self, cycle: CycleData) -> float:
        """
        Compute State of Health (SOH) as percentage.
        
        SOH = (Current Capacity / Initial Capacity) × 100
        """
        if self.initial_capacity is None or self.initial_capacity == 0:
            return 100.0
        
        soh = (cycle.capacity / self.initial_capacity) * 100.0
        return soh
    
    def _compute_ica_features(self, cycle: CycleData) -> Dict:
        """
        Compute Incremental Capacity Analysis (ICA) features.
        
        ICA = dQ/dV
        
        Returns:
            Dictionary with ICA peak positions and magnitudes
        """
        try:
            # Extract voltage and current for discharge
            voltage = cycle.voltage
            current = cycle.current
            time = cycle.time
            
            # Compute charge (Q) by integrating current over time
            # Q = ∫ I dt
            dt = np.diff(time, prepend=time[0])
            # For discharge, current is typically negative, take absolute value
            charge = np.cumsum(np.abs(current) * dt) / 3600.0  # Convert to Ah
            
            # Filter data within voltage range of interest
            v_min, v_max = self.voltage_range
            mask = (voltage >= v_min) & (voltage <= v_max)
            
            if np.sum(mask) < self.window_length:
                # Not enough data points in range
                return self._get_default_ica_features()
            
            voltage_filtered = voltage[mask]
            charge_filtered = charge[mask]
            
            # Sort by voltage (should be decreasing for discharge)
            sort_idx = np.argsort(voltage_filtered)
            voltage_sorted = voltage_filtered[sort_idx]
            charge_sorted = charge_filtered[sort_idx]
            
            # Compute dQ/dV using numerical differentiation
            # First smooth the charge curve
            if len(charge_sorted) >= self.window_length:
                charge_smooth = savgol_filter(
                    charge_sorted,
                    window_length=self.window_length,
                    polyorder=self.polyorder
                )
            else:
                charge_smooth = charge_sorted
            
            # Compute derivative dQ/dV
            dqdv = np.gradient(charge_smooth, voltage_sorted)
            
            # Smooth the dQ/dV curve
            if len(dqdv) >= self.window_length:
                dqdv_smooth = savgol_filter(
                    dqdv,
                    window_length=self.window_length,
                    polyorder=self.polyorder
                )
            else:
                dqdv_smooth = dqdv
            
            # Find peaks in dQ/dV curve
            peaks, properties = find_peaks(
                np.abs(dqdv_smooth),
                prominence=self.ica_config['peak_detection']['prominence'],
                width=self.ica_config['peak_detection']['width']
            )
            
            # Extract top 3 peaks (most prominent)
            if len(peaks) > 0:
                # Sort by prominence
                prominences = properties['prominences']
                top_indices = np.argsort(prominences)[-3:][::-1]  # Top 3, descending
                
                peak_voltages = voltage_sorted[peaks[top_indices]]
                peak_magnitudes = np.abs(dqdv_smooth[peaks[top_indices]])
                
                # Pad if less than 3 peaks
                while len(peak_voltages) < 3:
                    peak_voltages = np.append(peak_voltages, 0.0)
                    peak_magnitudes = np.append(peak_magnitudes, 0.0)
                
                return {
                    'ica_peak1_voltage': peak_voltages[0],
                    'ica_peak1_magnitude': peak_magnitudes[0],
                    'ica_peak2_voltage': peak_voltages[1],
                    'ica_peak2_magnitude': peak_magnitudes[1],
                    'ica_peak3_voltage': peak_voltages[2],
                    'ica_peak3_magnitude': peak_magnitudes[2],
                    'ica_num_peaks': len(peaks)
                }
            else:
                return self._get_default_ica_features()
        
        except Exception as e:
            # If ICA computation fails, return default values
            print(f"Warning: ICA computation failed for cycle {cycle.cycle_index}: {e}")
            return self._get_default_ica_features()
    
    def _get_default_ica_features(self) -> Dict:
        """Return default ICA features when computation fails."""
        return {
            'ica_peak1_voltage': 0.0,
            'ica_peak1_magnitude': 0.0,
            'ica_peak2_voltage': 0.0,
            'ica_peak2_magnitude': 0.0,
            'ica_peak3_voltage': 0.0,
            'ica_peak3_magnitude': 0.0,
            'ica_num_peaks': 0
        }
    
    def _compute_dcir(self, cycle: CycleData) -> float:
        """
        Compute DC Internal Resistance (DCIR) approximation.
        
        DCIR ≈ ΔV / ΔI
        
        We use the voltage drop at a significant current change.
        """
        try:
            voltage = cycle.voltage
            current = cycle.current
            
            # Find indices where current magnitude exceeds threshold
            high_current_mask = np.abs(current) > self.current_threshold
            
            if np.sum(high_current_mask) < 2:
                return 0.0
            
            # Find the initial voltage drop (first significant current change)
            # Look for transition from low to high current
            current_abs = np.abs(current)
            current_diff = np.diff(current_abs)
            
            # Find largest current jump
            max_jump_idx = np.argmax(np.abs(current_diff))
            
            if max_jump_idx > 0 and max_jump_idx < len(voltage) - 1:
                delta_v = voltage[max_jump_idx] - voltage[max_jump_idx + 1]
                delta_i = current_abs[max_jump_idx + 1] - current_abs[max_jump_idx]
                
                if delta_i > 0:
                    dcir = np.abs(delta_v / delta_i)
                    return dcir
            
            return 0.0
        
        except Exception as e:
            return 0.0
    
    def _compute_temperature_features(self, cycle: CycleData) -> Dict:
        """
        Compute temperature-related features.
        
        Returns:
            Dictionary with ΔT, mean, max, std
        """
        temp = cycle.temperature
        
        return {
            'temp_delta': np.max(temp) - np.min(temp),
            'temp_mean': np.mean(temp),
            'temp_max': np.max(temp),
            'temp_std': np.std(temp)
        }
    
    def _compute_basic_features(self, cycle: CycleData) -> Dict:
        """Compute basic statistical features from voltage and current."""
        return {
            'voltage_mean': np.mean(cycle.voltage),
            'voltage_std': np.std(cycle.voltage),
            'voltage_min': np.min(cycle.voltage),
            'voltage_max': np.max(cycle.voltage),
            'current_mean': np.mean(np.abs(cycle.current)),
            'current_std': np.std(cycle.current),
            'discharge_time': cycle.time[-1] - cycle.time[0]
        }
    
    def save_features(self, features_df: pd.DataFrame, output_path: str) -> None:
        """Save features to CSV file."""
        features_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved features to: {output_path}")
    
    def load_features(self, input_path: str) -> pd.DataFrame:
        """Load features from CSV file."""
        df = pd.read_csv(input_path)
        print(f"\n✓ Loaded features from: {input_path}")
        print(f"  Shape: {df.shape}")
        return df
    
    def print_feature_summary(self, features_df: pd.DataFrame) -> None:
        """Print summary statistics of extracted features."""
        print(f"\n{'=' * 80}")
        print("FEATURE SUMMARY STATISTICS".center(80))
        print('=' * 80)
        
        # SOH trend
        print(f"\n🔋 SOH Statistics:")
        print(f"  Initial SOH: {features_df['soh'].iloc[0]:.2f}%")
        print(f"  Final SOH: {features_df['soh'].iloc[-1]:.2f}%")
        print(f"  SOH range: [{features_df['soh'].min():.2f}, {features_df['soh'].max():.2f}]%")
        print(f"  Mean degradation rate: {(features_df['soh'].iloc[0] - features_df['soh'].iloc[-1]) / len(features_df):.3f}% per cycle")
        
        # ICA peaks
        print(f"\n📈 ICA Peak Analysis:")
        print(f"  Peak 1 voltage range: [{features_df['ica_peak1_voltage'].min():.3f}, {features_df['ica_peak1_voltage'].max():.3f}] V")
        print(f"  Peak 1 magnitude range: [{features_df['ica_peak1_magnitude'].min():.4f}, {features_df['ica_peak1_magnitude'].max():.4f}]")
        print(f"  Average peaks detected: {features_df['ica_num_peaks'].mean():.1f}")
        
        # DCIR
        dcir_nonzero = features_df[features_df['dcir'] > 0]['dcir']
        if len(dcir_nonzero) > 0:
            print(f"\n⚡ DCIR Statistics:")
            print(f"  Mean: {dcir_nonzero.mean():.6f} Ω")
            print(f"  Std: {dcir_nonzero.std():.6f} Ω")
            print(f"  Range: [{dcir_nonzero.min():.6f}, {dcir_nonzero.max():.6f}] Ω")
        
        # Temperature
        print(f"\n🌡️  Temperature Statistics:")
        print(f"  Mean ΔT: {features_df['temp_delta'].mean():.2f} ± {features_df['temp_delta'].std():.2f} °C")
        print(f"  Mean temperature: {features_df['temp_mean'].mean():.2f} °C")
        
        print('\n' + '=' * 80 + '\n')


def extract_and_save_features(cycles: List[CycleData], config: Dict, output_path: str) -> pd.DataFrame:
    """
    Convenience function to extract and save features.
    
    Args:
        cycles: List of CycleData objects
        config: Configuration dictionary
        output_path: Path to save features CSV
    
    Returns:
        Features DataFrame
    """
    extractor = FeatureExtractor(config)
    features_df = extractor.extract_all_features(cycles)
    extractor.print_feature_summary(features_df)
    extractor.save_features(features_df, output_path)
    
    return features_df


if __name__ == "__main__":
    # Test feature extraction
    import sys
    import os
    sys.path.append('..')
    from utils import load_config
    from data_loader import load_battery_data
    from cycle_segmentation import segment_and_filter_cycles
    
    config = load_config('../config.yaml')
    
    # Load and segment cycles
    loader = load_battery_data(config)
    filtered_cycles = segment_and_filter_cycles(loader, config)
    
    # Extract features
    features_path = os.path.join(config['paths']['features'], 'features_B0005.csv')
    features_df = extract_and_save_features(filtered_cycles, config, features_path)
    
    print(f"\n✓ Phase-1 Step 3 COMPLETE: Features extracted and saved")
    print(f"\nFeature columns:")
    for col in features_df.columns:
        print(f"  - {col}")
