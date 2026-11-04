"""
NASA Battery Data Loader for B0005
Loads .mat files and extracts cycle-wise data
"""

import numpy as np
import scipy.io as sio
from typing import Dict, List, Tuple, Optional
import os
from dataclasses import dataclass


@dataclass
class CycleData:
    """Data structure for a single battery cycle."""
    cycle_index: int
    cycle_type: str  # 'charge', 'discharge', 'impedance'
    time: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    temperature: np.ndarray
    capacity: Optional[float] = None
    ambient_temperature: Optional[float] = None
    data_points: int = 0
    
    def __post_init__(self):
        self.data_points = len(self.time)


class NASABatteryLoader:
    """
    Loader for NASA Battery Dataset (.mat files).
    
    Specifically designed for B0005, B0006, B0007, B0018 format.
    """
    
    def __init__(self, mat_file_path: str, battery_id: str = "B0005"):
        """
        Initialize loader.
        
        Args:
            mat_file_path: Path to .mat file
            battery_id: Battery identifier (e.g., 'B0005')
        """
        self.mat_file_path = mat_file_path
        self.battery_id = battery_id
        self.raw_data = None
        self.cycles = []
        
        if not os.path.exists(mat_file_path):
            raise FileNotFoundError(f"Battery file not found: {mat_file_path}")
    
    def load(self) -> None:
        """Load .mat file into memory."""
        print(f"Loading battery data: {self.battery_id}")
        print(f"File: {self.mat_file_path}")
        
        try:
            self.raw_data = sio.loadmat(self.mat_file_path)
            print(f"✓ Loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load .mat file: {e}")
    
    def parse_cycles(self) -> List[CycleData]:
        """
        Parse all cycles from raw data.
        
        Returns:
            List of CycleData objects
        """
        if self.raw_data is None:
            self.load()
        
        print(f"\nParsing cycles from {self.battery_id}...")
        
        # Extract battery data structure
        battery_data = self.raw_data[self.battery_id]
        cycle_array = battery_data[0, 0]['cycle'][0]
        
        total_cycles = len(cycle_array)
        print(f"Total cycles found: {total_cycles}")
        
        self.cycles = []
        
        for i in range(total_cycles):
            cycle = cycle_array[i]
            
            # Extract metadata
            cycle_type = str(cycle['type'][0])
            
            # Extract ambient temperature
            ambient_temp = None
            if len(cycle['ambient_temperature'][0]) > 0:
                ambient_temp = float(cycle['ambient_temperature'][0][0])
            
            # Extract data
            data_struct = cycle['data'][0, 0]
            
            # Parse based on cycle type
            if cycle_type in ['charge', 'discharge']:
                time = data_struct['Time'][0].flatten()
                voltage = data_struct['Voltage_measured'][0].flatten()
                current = data_struct['Current_measured'][0].flatten()
                temperature = data_struct['Temperature_measured'][0].flatten()
                
                # Capacity is only available for discharge cycles
                capacity = None
                if cycle_type == 'discharge' and 'Capacity' in data_struct.dtype.names:
                    capacity_array = data_struct['Capacity']
                    if capacity_array.size > 0:
                        capacity = float(capacity_array[0][0])
                
                cycle_data = CycleData(
                    cycle_index=i,
                    cycle_type=cycle_type,
                    time=time,
                    voltage=voltage,
                    current=current,
                    temperature=temperature,
                    capacity=capacity,
                    ambient_temperature=ambient_temp
                )
                
                self.cycles.append(cycle_data)
            
            elif cycle_type == 'impedance':
                # We can parse impedance data if needed later
                # For Phase-1, we focus on charge/discharge
                pass
        
        print(f"✓ Parsed {len(self.cycles)} charge/discharge cycles")
        
        # Count by type
        charge_count = sum(1 for c in self.cycles if c.cycle_type == 'charge')
        discharge_count = sum(1 for c in self.cycles if c.cycle_type == 'discharge')
        
        print(f"  - Charge cycles: {charge_count}")
        print(f"  - Discharge cycles: {discharge_count}")
        
        return self.cycles
    
    def get_discharge_cycles(self) -> List[CycleData]:
        """
        Get only discharge cycles (used for SOH calculation).
        
        Returns:
            List of discharge CycleData objects
        """
        if not self.cycles:
            self.parse_cycles()
        
        discharge_cycles = [c for c in self.cycles if c.cycle_type == 'discharge']
        print(f"\nExtracted {len(discharge_cycles)} discharge cycles")
        
        return discharge_cycles
    
    def get_cycle_summary(self) -> Dict:
        """
        Get summary statistics of all cycles.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.cycles:
            self.parse_cycles()
        
        discharge_cycles = [c for c in self.cycles if c.cycle_type == 'discharge']
        
        if not discharge_cycles:
            return {}
        
        capacities = [c.capacity for c in discharge_cycles if c.capacity is not None]
        data_points = [c.data_points for c in discharge_cycles]
        voltages = [c.voltage for c in discharge_cycles]
        temperatures = [c.temperature for c in discharge_cycles]
        
        summary = {
            'total_cycles': len(self.cycles),
            'discharge_cycles': len(discharge_cycles),
            'capacity': {
                'initial': capacities[0] if capacities else None,
                'final': capacities[-1] if capacities else None,
                'mean': np.mean(capacities) if capacities else None,
                'std': np.std(capacities) if capacities else None,
                'min': np.min(capacities) if capacities else None,
                'max': np.max(capacities) if capacities else None,
            },
            'data_points': {
                'mean': np.mean(data_points),
                'min': np.min(data_points),
                'max': np.max(data_points),
            },
            'voltage': {
                'mean': np.mean([v.mean() for v in voltages]),
                'global_min': np.min([v.min() for v in voltages]),
                'global_max': np.max([v.max() for v in voltages]),
            },
            'temperature': {
                'mean': np.mean([t.mean() for t in temperatures]),
                'global_min': np.min([t.min() for t in temperatures]),
                'global_max': np.max([t.max() for t in temperatures]),
            }
        }
        
        return summary
    
    def print_summary(self) -> None:
        """Print formatted summary of the battery data."""
        summary = self.get_cycle_summary()
        
        print(f"\n{'=' * 80}")
        print(f"BATTERY DATA SUMMARY: {self.battery_id}".center(80))
        print('=' * 80)
        
        print(f"\n📊 Cycle Counts:")
        print(f"  Total cycles: {summary['total_cycles']}")
        print(f"  Discharge cycles: {summary['discharge_cycles']}")
        
        print(f"\n🔋 Capacity Analysis:")
        cap = summary['capacity']
        print(f"  Initial capacity: {cap['initial']:.4f} Ah")
        print(f"  Final capacity: {cap['final']:.4f} Ah")
        print(f"  Capacity fade: {(cap['initial'] - cap['final']):.4f} Ah ({((cap['initial'] - cap['final']) / cap['initial'] * 100):.2f}%)")
        print(f"  Range: [{cap['min']:.4f}, {cap['max']:.4f}] Ah")
        
        print(f"\n⚡ Voltage Statistics:")
        volt = summary['voltage']
        print(f"  Mean voltage: {volt['mean']:.3f} V")
        print(f"  Range: [{volt['global_min']:.3f}, {volt['global_max']:.3f}] V")
        
        print(f"\n🌡️  Temperature Statistics:")
        temp = summary['temperature']
        print(f"  Mean temperature: {temp['mean']:.2f} °C")
        print(f"  Range: [{temp['global_min']:.2f}, {temp['global_max']:.2f}] °C")
        
        print(f"\n📈 Data Points per Cycle:")
        dp = summary['data_points']
        print(f"  Mean: {dp['mean']:.0f}")
        print(f"  Range: [{dp['min']}, {dp['max']}]")
        
        print('\n' + '=' * 80 + '\n')


def load_battery_data(config: Dict) -> NASABatteryLoader:
    """
    Convenience function to load battery data from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        NASABatteryLoader instance with loaded data
    """
    dataset_config = config['dataset']
    
    loader = NASABatteryLoader(
        mat_file_path=dataset_config['path'],
        battery_id=dataset_config['battery_id']
    )
    
    loader.load()
    loader.parse_cycles()
    loader.print_summary()
    
    return loader


if __name__ == "__main__":
    # Test the loader
    import sys
    sys.path.append('..')
    from utils import load_config
    
    config = load_config('../config.yaml')
    loader = load_battery_data(config)
    
    # Get discharge cycles
    discharge_cycles = loader.get_discharge_cycles()
    
    print(f"\n✓ Successfully loaded {len(discharge_cycles)} discharge cycles")
