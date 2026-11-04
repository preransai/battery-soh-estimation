"""
Cycle Segmentation and Filtering
Phase-1: Focus on discharge cycles with strict quality checks
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from data_loader import CycleData
import os


@dataclass
class FilteringStats:
    """Statistics from cycle filtering process."""
    total_cycles: int
    passed_cycles: int
    failed_min_points: int
    failed_voltage_range: int
    failed_spikes: int
    failed_incomplete: int
    
    def print_report(self):
        """Print formatted filtering report."""
        print(f"\n{'=' * 80}")
        print("CYCLE FILTERING REPORT".center(80))
        print('=' * 80)
        print(f"\nTotal cycles processed: {self.total_cycles}")
        print(f"✓ Passed quality checks: {self.passed_cycles} ({self.passed_cycles/self.total_cycles*100:.1f}%)")
        print(f"\nRejection reasons:")
        print(f"  ❌ Insufficient data points (< 100): {self.failed_min_points}")
        print(f"  ❌ Voltage out of range: {self.failed_voltage_range}")
        print(f"  ❌ Voltage spikes detected: {self.failed_spikes}")
        print(f"  ❌ Incomplete cycles: {self.failed_incomplete}")
        print('=' * 80 + '\n')


class CycleSegmentation:
    """
    Segment and filter battery cycles according to Phase-1 criteria.
    
    Filtering Criteria:
    - Remove cycles with < 100 data points
    - Remove cycles with voltage < 2.0V or > 4.3V
    - Detect and remove cycles with sudden spikes
    - Remove incomplete cycles
    """
    
    def __init__(self, config: Dict):
        """
        Initialize segmentation.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.filtering_config = config['filtering']
        
        self.min_points = self.filtering_config['min_data_points']
        self.v_min = self.filtering_config['voltage_min']
        self.v_max = self.filtering_config['voltage_max']
        self.detect_spikes = self.filtering_config['detect_spikes']
        self.spike_threshold = self.filtering_config['spike_threshold']
        
        self.stats = None
    
    def filter_cycles(self, cycles: List[CycleData]) -> List[CycleData]:
        """
        Apply all filtering criteria to cycles.
        
        Args:
            cycles: List of CycleData objects (discharge cycles only)
        
        Returns:
            List of filtered CycleData objects that pass all checks
        """
        print(f"\n{'─' * 80}")
        print("APPLYING CYCLE FILTERING CRITERIA")
        print('─' * 80)
        
        total = len(cycles)
        failed_min_points = 0
        failed_voltage_range = 0
        failed_spikes = 0
        failed_incomplete = 0
        
        filtered_cycles = []
        
        for i, cycle in enumerate(cycles):
            # Check 1: Minimum data points
            if cycle.data_points < self.min_points:
                failed_min_points += 1
                continue
            
            # Check 2: Voltage range
            if not self._check_voltage_range(cycle):
                failed_voltage_range += 1
                continue
            
            # Check 3: Detect spikes
            if self.detect_spikes and self._has_spikes(cycle):
                failed_spikes += 1
                continue
            
            # Check 4: Check completeness (has capacity value)
            if cycle.capacity is None or cycle.capacity <= 0:
                failed_incomplete += 1
                continue
            
            # Passed all checks
            filtered_cycles.append(cycle)
        
        # Store statistics
        self.stats = FilteringStats(
            total_cycles=total,
            passed_cycles=len(filtered_cycles),
            failed_min_points=failed_min_points,
            failed_voltage_range=failed_voltage_range,
            failed_spikes=failed_spikes,
            failed_incomplete=failed_incomplete
        )
        
        self.stats.print_report()
        
        return filtered_cycles
    
    def _check_voltage_range(self, cycle: CycleData) -> bool:
        """Check if voltage stays within acceptable range."""
        v_min_cycle = np.min(cycle.voltage)
        v_max_cycle = np.max(cycle.voltage)
        
        return (v_min_cycle >= self.v_min) and (v_max_cycle <= self.v_max)
    
    def _has_spikes(self, cycle: CycleData) -> bool:
        """
        Detect sudden voltage spikes/drops.
        
        A spike is defined as a sudden change > spike_threshold between
        consecutive measurements.
        """
        voltage_diff = np.abs(np.diff(cycle.voltage))
        max_diff = np.max(voltage_diff)
        
        return max_diff > self.spike_threshold
    
    def save_segmented_cycles(self, cycles: List[CycleData], output_dir: str) -> None:
        """
        Save segmented cycles to disk.
        
        Each cycle saved as: cycle_XXX.npz containing:
        - time, voltage, current, temperature arrays
        - capacity (scalar)
        - metadata
        
        Args:
            cycles: List of filtered CycleData objects
            output_dir: Directory to save cycles
        """
        print(f"\n{'─' * 80}")
        print(f"SAVING SEGMENTED CYCLES")
        print('─' * 80)
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, cycle in enumerate(cycles):
            filename = f"cycle_{i:03d}.npz"
            filepath = os.path.join(output_dir, filename)
            
            np.savez(
                filepath,
                time=cycle.time,
                voltage=cycle.voltage,
                current=cycle.current,
                temperature=cycle.temperature,
                capacity=cycle.capacity,
                cycle_index=cycle.cycle_index,
                cycle_type=cycle.cycle_type,
                data_points=cycle.data_points,
                ambient_temperature=cycle.ambient_temperature if cycle.ambient_temperature else 0.0
            )
        
        print(f"✓ Saved {len(cycles)} cycles to: {output_dir}")
        
        # Save metadata
        metadata_file = os.path.join(output_dir, "metadata.npz")
        np.savez(
            metadata_file,
            total_cycles=len(cycles),
            cycle_indices=[c.cycle_index for c in cycles],
            capacities=[c.capacity for c in cycles]
        )
        print(f"✓ Saved metadata to: {metadata_file}")
    
    def load_segmented_cycles(self, input_dir: str) -> List[CycleData]:
        """
        Load previously saved segmented cycles.
        
        Args:
            input_dir: Directory containing saved cycles
        
        Returns:
            List of CycleData objects
        """
        print(f"\nLoading segmented cycles from: {input_dir}")
        
        cycle_files = sorted([f for f in os.listdir(input_dir) if f.startswith('cycle_') and f.endswith('.npz')])
        
        cycles = []
        for filename in cycle_files:
            filepath = os.path.join(input_dir, filename)
            data = np.load(filepath)
            
            cycle = CycleData(
                cycle_index=int(data['cycle_index']),
                cycle_type=str(data['cycle_type']),
                time=data['time'],
                voltage=data['voltage'],
                current=data['current'],
                temperature=data['temperature'],
                capacity=float(data['capacity']),
                ambient_temperature=float(data['ambient_temperature']) if data['ambient_temperature'] != 0.0 else None,
                data_points=int(data['data_points'])
            )
            
            cycles.append(cycle)
        
        print(f"✓ Loaded {len(cycles)} cycles")
        
        return cycles
    
    def analyze_cycle_quality(self, cycles: List[CycleData]) -> Dict:
        """
        Analyze quality metrics of filtered cycles.
        
        Returns:
            Dictionary with quality metrics
        """
        data_points = [c.data_points for c in cycles]
        voltages_min = [c.voltage.min() for c in cycles]
        voltages_max = [c.voltage.max() for c in cycles]
        voltage_ranges = [c.voltage.max() - c.voltage.min() for c in cycles]
        temp_deltas = [c.temperature.max() - c.temperature.min() for c in cycles]
        capacities = [c.capacity for c in cycles]
        
        quality_metrics = {
            'num_cycles': len(cycles),
            'data_points': {
                'mean': np.mean(data_points),
                'std': np.std(data_points),
                'min': np.min(data_points),
                'max': np.max(data_points)
            },
            'voltage': {
                'min_global': np.min(voltages_min),
                'max_global': np.max(voltages_max),
                'mean_range': np.mean(voltage_ranges),
                'std_range': np.std(voltage_ranges)
            },
            'temperature_delta': {
                'mean': np.mean(temp_deltas),
                'std': np.std(temp_deltas),
                'min': np.min(temp_deltas),
                'max': np.max(temp_deltas)
            },
            'capacity': {
                'initial': capacities[0],
                'final': capacities[-1],
                'fade_ah': capacities[0] - capacities[-1],
                'fade_percent': (capacities[0] - capacities[-1]) / capacities[0] * 100
            }
        }
        
        return quality_metrics
    
    def print_quality_report(self, cycles: List[CycleData]) -> None:
        """Print formatted quality analysis report."""
        metrics = self.analyze_cycle_quality(cycles)
        
        print(f"\n{'=' * 80}")
        print("CYCLE QUALITY ANALYSIS".center(80))
        print('=' * 80)
        
        print(f"\n📊 Number of clean cycles: {metrics['num_cycles']}")
        
        print(f"\n📈 Data Points per Cycle:")
        dp = metrics['data_points']
        print(f"  Mean: {dp['mean']:.0f} ± {dp['std']:.0f}")
        print(f"  Range: [{dp['min']}, {dp['max']}]")
        
        print(f"\n⚡ Voltage Characteristics:")
        v = metrics['voltage']
        print(f"  Global range: [{v['min_global']:.3f}, {v['max_global']:.3f}] V")
        print(f"  Mean discharge range: {v['mean_range']:.3f} ± {v['std_range']:.3f} V")
        
        print(f"\n🌡️  Temperature Variation (ΔT):")
        t = metrics['temperature_delta']
        print(f"  Mean ΔT: {t['mean']:.2f} ± {t['std']:.2f} °C")
        print(f"  Range: [{t['min']:.2f}, {t['max']:.2f}] °C")
        print(f"  Note: Low variance expected for room temperature testing")
        
        print(f"\n🔋 Capacity Degradation:")
        c = metrics['capacity']
        print(f"  Initial: {c['initial']:.4f} Ah")
        print(f"  Final: {c['final']:.4f} Ah")
        print(f"  Fade: {c['fade_ah']:.4f} Ah ({c['fade_percent']:.2f}%)")
        
        print('\n' + '=' * 80 + '\n')


def segment_and_filter_cycles(loader, config: Dict, save_dir: str = None) -> List[CycleData]:
    """
    Convenience function to segment and filter cycles from loader.
    
    Args:
        loader: NASABatteryLoader instance
        config: Configuration dictionary
        save_dir: Optional directory to save segmented cycles
    
    Returns:
        List of filtered CycleData objects
    """
    # Get discharge cycles
    discharge_cycles = loader.get_discharge_cycles()
    
    # Create segmentation instance and filter
    segmentation = CycleSegmentation(config)
    filtered_cycles = segmentation.filter_cycles(discharge_cycles)
    
    # Print quality report
    segmentation.print_quality_report(filtered_cycles)
    
    # Save if requested
    if save_dir:
        segmentation.save_segmented_cycles(filtered_cycles, save_dir)
    
    return filtered_cycles


if __name__ == "__main__":
    # Test segmentation
    import sys
    sys.path.append('..')
    from utils import load_config
    from data_loader import load_battery_data
    
    config = load_config('../config.yaml')
    loader = load_battery_data(config)
    
    # Segment and filter
    output_dir = config['paths']['processed_data']
    filtered_cycles = segment_and_filter_cycles(loader, config, save_dir=output_dir)
    
    print(f"\n✓ Phase-1 Step 2 COMPLETE: {len(filtered_cycles)} clean cycles ready")
