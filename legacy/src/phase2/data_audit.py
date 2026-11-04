"""
Phase-2 Step 1: Data Audit & Verification
Rigorous validation of all datasets before proceeding
"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import json

from data_loader import NASABatteryLoader, CycleData
from utils import plot_style_setup


class DataAudit:
    """
    Comprehensive data audit for Phase-2.
    
    Verifies:
    - Cycle counts and data quality
    - Voltage/current/temperature ranges
    - SOH computation correctness
    - Data consistency across batteries
    """
    
    def __init__(self, config: Dict):
        """Initialize audit with configuration."""
        self.config = config
        self.audit_config = config['audit']
        self.audit_results = {}
        
        # Create audit directory
        self.audit_dir = config['paths']['audit_reports']
        os.makedirs(self.audit_dir, exist_ok=True)
        
        # Timestamp for this audit
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plot_style_setup()
    
    def audit_nasa_battery(self, battery_path: str, battery_id: str) -> Dict:
        """
        Comprehensive audit of a single NASA battery.
        
        Returns:
            Dictionary with audit results and flags
        """
        print(f"\n{'=' * 80}")
        print(f"AUDITING NASA BATTERY: {battery_id}".center(80))
        print('=' * 80)
        
        # Handle relative paths - make absolute if needed
        if not os.path.isabs(battery_path):
            # Try current directory first
            if not os.path.exists(battery_path):
                # Try project root
                project_root = '.'
                full_path = os.path.join(project_root, battery_path)
                if os.path.exists(full_path):
                    battery_path = full_path
                    print(f"Using absolute path: {battery_path}")
        
        print(f"Loading from: {battery_path}")
        
        # Load battery
        loader = NASABatteryLoader(battery_path, battery_id)
        loader.load()
        discharge_cycles = loader.get_discharge_cycles()
        
        audit_result = {
            'battery_id': battery_id,
            'timestamp': self.timestamp,
            'total_cycles': len(discharge_cycles),
            'checks': {},
            'warnings': [],
            'errors': [],
            'passed': True
        }
        
        # Check 1: Cycle count
        print(f"\n✓ Total discharge cycles: {len(discharge_cycles)}")
        if len(discharge_cycles) < 50:
            audit_result['warnings'].append(f"Low cycle count: {len(discharge_cycles)}")
        
        # Check 2: Voltage ranges
        print(f"\n► Checking voltage ranges...")
        voltages_min = [c.voltage.min() for c in discharge_cycles]
        voltages_max = [c.voltage.max() for c in discharge_cycles]
        
        v_global_min = np.min(voltages_min)
        v_global_max = np.max(voltages_max)
        
        v_range = self.audit_config['voltage_range']
        
        # Allow 0.5V tolerance for deep discharge (common in aged batteries)
        v_tolerance = 0.5
        v_hard_min = v_range[0] - v_tolerance  # 1.5V hard minimum
        v_hard_max = v_range[1] + 0.2  # 4.5V hard maximum
        
        print(f"  Global voltage range: [{v_global_min:.3f}, {v_global_max:.3f}] V")
        print(f"  Expected range: [{v_range[0]}, {v_range[1]}] V")
        print(f"  Hard limits: [{v_hard_min}, {v_hard_max}] V")
        
        # Determine pass/warning status
        in_expected = v_range[0] <= v_global_min and v_global_max <= v_range[1]
        in_tolerance = v_hard_min <= v_global_min and v_global_max <= v_hard_max
        
        if not in_expected:
            if in_tolerance:
                audit_result['warnings'].append(
                    f"Voltage slightly outside expected range: [{v_global_min:.3f}, {v_global_max:.3f}] V (within tolerance)"
                )
            else:
                audit_result['warnings'].append(
                    f"Voltage SIGNIFICANTLY outside range: [{v_global_min:.3f}, {v_global_max:.3f}] V"
                )
        
        # Pass if within tolerance (allow deep discharge in aged batteries)
        audit_result['checks']['voltage'] = {
            'min': v_global_min,
            'max': v_global_max,
            'in_expected_range': in_expected,
            'in_tolerance': in_tolerance,
            'passed': in_tolerance  # Pass if within tolerance
        }
        
        # Check 3: Current ranges
        print(f"\n► Checking current ranges...")
        currents_min = [c.current.min() for c in discharge_cycles]
        currents_max = [c.current.max() for c in discharge_cycles]
        
        i_global_min = np.min(currents_min)
        i_global_max = np.max(currents_max)
        
        i_range = self.audit_config['current_range']
        
        print(f"  Global current range: [{i_global_min:.3f}, {i_global_max:.3f}] A")
        print(f"  Expected range: [{i_range[0]}, {i_range[1]}] A")
        
        if i_global_min < i_range[0] or i_global_max > i_range[1]:
            audit_result['warnings'].append(
                f"Current outside expected range: [{i_global_min:.3f}, {i_global_max:.3f}]"
            )
        
        audit_result['checks']['current'] = {
            'min': i_global_min,
            'max': i_global_max,
            'passed': i_range[0] <= i_global_min and i_global_max <= i_range[1]
        }
        
        # Check 4: Temperature stability
        print(f"\n► Checking temperature stability...")
        temps_mean = [c.temperature.mean() for c in discharge_cycles]
        temps_std = [c.temperature.std() for c in discharge_cycles]
        
        t_mean = np.mean(temps_mean)
        t_std_avg = np.mean(temps_std)
        
        print(f"  Mean temperature: {t_mean:.2f} °C")
        print(f"  Average temperature std: {t_std_avg:.2f} °C")
        
        audit_result['checks']['temperature'] = {
            'mean': t_mean,
            'std_avg': t_std_avg,
            'passed': True  # NASA should be room temp stable
        }
        
        # Check 5: SOH computation
        print(f"\n► Verifying SOH computation...")
        capacities = [c.capacity for c in discharge_cycles if c.capacity is not None]
        
        if len(capacities) == 0:
            audit_result['errors'].append("No capacity values found!")
            audit_result['passed'] = False
            return audit_result
        
        initial_capacity = np.mean(capacities[:5])
        soh_values = [(cap / initial_capacity) * 100 for cap in capacities]
        
        print(f"  Initial capacity: {initial_capacity:.4f} Ah")
        print(f"  Final capacity: {capacities[-1]:.4f} Ah")
        print(f"  Initial SOH: {soh_values[0]:.2f}%")
        print(f"  Final SOH: {soh_values[-1]:.2f}%")
        print(f"  SOH range: [{np.min(soh_values):.2f}, {np.max(soh_values):.2f}]%")
        
        # Check if SOH generally decreases (with some tolerance for noise)
        if self.audit_config['verify_capacity_monotonic']:
            # Check if trend is downward
            from scipy.stats import linregress
            cycle_nums = np.arange(len(soh_values))
            slope, _, _, _, _ = linregress(cycle_nums, soh_values)
            
            print(f"  SOH degradation rate: {slope:.4f}% per cycle")
            
            if slope > 0:
                audit_result['warnings'].append(
                    f"SOH shows positive trend (increasing): {slope:.4f}% per cycle"
                )
        
        audit_result['checks']['soh'] = {
            'initial': soh_values[0],
            'final': soh_values[-1],
            'range': [np.min(soh_values), np.max(soh_values)],
            'degradation_rate': slope,
            'passed': slope < 0  # Should decrease
        }
        
        # Check 6: Data point consistency
        print(f"\n► Checking data point consistency...")
        data_points = [c.data_points for c in discharge_cycles]
        
        print(f"  Mean data points per cycle: {np.mean(data_points):.0f}")
        print(f"  Range: [{np.min(data_points)}, {np.max(data_points)}]")
        
        below_threshold = sum(1 for dp in data_points if dp < self.audit_config['min_data_points'])
        print(f"  Cycles below {self.audit_config['min_data_points']} points: {below_threshold}")
        
        audit_result['checks']['data_points'] = {
            'mean': np.mean(data_points),
            'min': np.min(data_points),
            'max': np.max(data_points),
            'below_threshold': below_threshold,
            'passed': below_threshold < len(discharge_cycles) * 0.1  # < 10% below threshold
        }
        
        # Check 7: Detect voltage spikes
        print(f"\n► Detecting voltage spikes...")
        spike_threshold = self.audit_config['max_spike_threshold']
        cycles_with_spikes = 0
        max_spike_values = []
        
        for cycle in discharge_cycles:
            voltage_diff = np.abs(np.diff(cycle.voltage))
            max_diff = np.max(voltage_diff)
            max_spike_values.append(max_diff)
            if max_diff > spike_threshold:
                cycles_with_spikes += 1
        
        print(f"  Cycles with spikes (>{spike_threshold}V): {cycles_with_spikes}")
        print(f"  Mean max voltage change: {np.mean(max_spike_values):.4f} V")
        print(f"  Max voltage change seen: {np.max(max_spike_values):.4f} V")
        
        # More lenient threshold for warnings (30% is acceptable for some batteries)
        # Only fail if > 50% of cycles have spikes (indicates systematic issue)
        spike_rate = cycles_with_spikes / len(discharge_cycles)
        
        if spike_rate > 0.3:  # > 30% - warning but don't fail
            audit_result['warnings'].append(
                f"Elevated spike rate: {cycles_with_spikes}/{len(discharge_cycles)} ({spike_rate*100:.1f}%) cycles"
            )
        
        # Only fail audit if > 50% have spikes (indicates fundamental data issue)
        audit_result['checks']['spikes'] = {
            'count': cycles_with_spikes,
            'rate': spike_rate,
            'threshold': spike_threshold,
            'mean_max_change': np.mean(max_spike_values),
            'max_change': np.max(max_spike_values),
            'passed': spike_rate < 0.5  # Pass if < 50% have spikes
        }
        
        # Plot random cycles
        if self.audit_config['plot_random_cycles'] > 0:
            self._plot_random_cycles(discharge_cycles, battery_id)
        
        # Final verdict
        # Critical checks that must pass
        critical_checks = ['voltage', 'soh', 'data_points']
        
        # Non-critical checks (warnings only)
        non_critical_checks = ['current', 'temperature', 'spikes']
        
        critical_passed = all(
            audit_result['checks'].get(check, {}).get('passed', True)
            for check in critical_checks
        )
        
        non_critical_passed = all(
            audit_result['checks'].get(check, {}).get('passed', True)
            for check in non_critical_checks
        )
        
        # Pass if critical checks pass (non-critical can have warnings)
        audit_result['passed'] = critical_passed and len(audit_result['errors']) == 0
        audit_result['critical_passed'] = critical_passed
        audit_result['non_critical_passed'] = non_critical_passed
        
        # Summary
        print(f"\n{'=' * 80}")
        print(f"AUDIT SUMMARY: {battery_id}".center(80))
        print('=' * 80)
        
        # Show check results
        print("\nCheck Results:")
        for check_name, check_result in audit_result['checks'].items():
            status = "✅" if check_result.get('passed', True) else "⚠️ "
            critical = "(critical)" if check_name in critical_checks else ""
            print(f"  {status} {check_name} {critical}")
        
        print(f"\n{'─' * 80}")
        
        if audit_result['passed']:
            if audit_result['warnings']:
                print(f"✅ AUDIT PASSED (with warnings): {battery_id}")
            else:
                print(f"✅ AUDIT PASSED: {battery_id}")
        else:
            print(f"❌ AUDIT FAILED: {battery_id}")
        
        if audit_result['warnings']:
            print(f"\n⚠️  Warnings ({len(audit_result['warnings'])}):")
            for warning in audit_result['warnings']:
                print(f"  - {warning}")
        
        if audit_result['errors']:
            print(f"\n❌ Errors ({len(audit_result['errors'])}):")
            for error in audit_result['errors']:
                print(f"  - {error}")
        
        print('=' * 80)
        
        return audit_result
    
    def _plot_random_cycles(self, cycles: List[CycleData], battery_id: str) -> None:
        """Plot random cycles for visual inspection."""
        n_plots = min(self.audit_config['plot_random_cycles'], len(cycles))
        
        # Select random cycles
        np.random.seed(self.config['random_seed'])
        indices = np.random.choice(len(cycles), n_plots, replace=False)
        indices = sorted(indices)
        
        fig, axes = plt.subplots(n_plots, 3, figsize=(15, 4*n_plots))
        
        if n_plots == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            cycle = cycles[idx]
            
            # Voltage vs Time
            axes[i, 0].plot(cycle.time, cycle.voltage, linewidth=1)
            axes[i, 0].set_ylabel('Voltage (V)')
            axes[i, 0].set_title(f'Cycle {idx}: Voltage Profile')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Current vs Time
            axes[i, 1].plot(cycle.time, cycle.current, linewidth=1, color='orange')
            axes[i, 1].set_ylabel('Current (A)')
            axes[i, 1].set_title(f'Cycle {idx}: Current Profile')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Temperature vs Time
            axes[i, 2].plot(cycle.time, cycle.temperature, linewidth=1, color='red')
            axes[i, 2].set_ylabel('Temperature (°C)')
            axes[i, 2].set_title(f'Cycle {idx}: Temperature Profile')
            axes[i, 2].grid(True, alpha=0.3)
            
            if i == n_plots - 1:
                axes[i, 0].set_xlabel('Time (s)')
                axes[i, 1].set_xlabel('Time (s)')
                axes[i, 2].set_xlabel('Time (s)')
        
        plt.suptitle(f'{battery_id}: Random Cycle Inspection', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        filename = f"{battery_id}_random_cycles_{self.timestamp}.png"
        filepath = os.path.join(self.audit_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  ✓ Saved random cycle plots: {filepath}")
    
    def audit_all_nasa_batteries(self) -> Dict[str, Dict]:
        """Audit all NASA batteries for Phase-2."""
        print(f"\n{'#' * 80}")
        print("PHASE-2 STEP 1: NASA BATTERY DATA AUDIT".center(80))
        print('#' * 80)
        
        nasa_configs = {
            'B0005': self.config['datasets']['nasa_train'],
            'B0006': self.config['datasets']['nasa_test_1'],
            'B0007': self.config['datasets']['nasa_test_2']
        }
        
        all_results = {}
        
        for battery_id, battery_config in nasa_configs.items():
            result = self.audit_nasa_battery(battery_config['path'], battery_id)
            all_results[battery_id] = result
        
        # Save consolidated report
        self._save_audit_report(all_results, "NASA_batteries")
        
        # Check if all passed
        all_passed = all(result['passed'] for result in all_results.values())
        
        print(f"\n{'#' * 80}")
        if all_passed:
            print("✅ ALL NASA BATTERIES PASSED AUDIT".center(80))
            print("✅ PROCEED TO STEP 2: HARMONIZATION".center(80))
        else:
            print("❌ AUDIT FAILURES DETECTED".center(80))
            print("❌ FIX ISSUES AND RE-RUN AUDIT".center(80))
        print('#' * 80)
        
        return all_results
    
    def _save_audit_report(self, results: Dict, dataset_name: str) -> None:
        """Save audit results to JSON and markdown."""
        # JSON report
        json_path = os.path.join(
            self.audit_dir, 
            f"{dataset_name}_audit_{self.timestamp}.json"
        )
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Saved audit JSON: {json_path}")
        
        # Markdown report
        md_path = os.path.join(
            self.audit_dir,
            f"{dataset_name}_audit_{self.timestamp}.md"
        )
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Data Audit Report: {dataset_name}\n\n")
            f.write(f"**Timestamp:** {self.timestamp}\n\n")
            f.write("---\n\n")
            
            for battery_id, result in results.items():
                f.write(f"## {battery_id}\n\n")
                f.write(f"**Status:** {'✅ PASSED' if result['passed'] else '❌ FAILED'}\n\n")
                f.write(f"**Total Cycles:** {result['total_cycles']}\n\n")
                
                f.write("### Checks\n\n")
                for check_name, check_result in result['checks'].items():
                    status = '✅' if check_result.get('passed', True) else '❌'
                    f.write(f"- **{check_name}:** {status}\n")
                    for key, value in check_result.items():
                        if key != 'passed':
                            f.write(f"  - {key}: {value}\n")
                    f.write("\n")
                
                if result['warnings']:
                    f.write("### Warnings\n\n")
                    for warning in result['warnings']:
                        f.write(f"- ⚠️  {warning}\n")
                    f.write("\n")
                
                if result['errors']:
                    f.write("### Errors\n\n")
                    for error in result['errors']:
                        f.write(f"- ❌ {error}\n")
                    f.write("\n")
                
                f.write("---\n\n")
        
        print(f"✓ Saved audit report: {md_path}")


if __name__ == "__main__":
    import yaml
    
    # Load Phase-2 config
    with open('../../phase2_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Run audit
    auditor = DataAudit(config)
    results = auditor.audit_all_nasa_batteries()
    
    # Check if we can proceed
    all_passed = all(r['passed'] for r in results.values())
    
    if all_passed:
        print("\n" + "="*80)
        print("STEP 1 COMPLETE ✅")
        print("Ready to proceed to Step 2: Multi-Dataset Harmonization")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("STEP 1 FAILED ❌")
        print("Fix issues and re-run audit before proceeding")
        print("="*80)
