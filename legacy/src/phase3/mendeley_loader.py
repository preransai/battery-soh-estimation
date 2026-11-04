"""
Mendeley Battery Dataset Loader

Dataset: Lithium-ion battery degradation dataset based on a realistic forklift operation profile
Source: https://data.mendeley.com/datasets/

Structure:
- Cell1/, Cell2/, Cell3/
  - Round00/, Round01/, ...
    - RPT.csv (Reference Performance Test data)

Columns: Cell, Round, Time, Current, Voltage, Energy, Temperature
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import glob


@dataclass
class MendeleyCycle:
    """Single discharge cycle from Mendeley dataset."""
    cell_id: str
    round_number: int
    cycle_number: int
    voltage: np.ndarray
    current: np.ndarray
    temperature: np.ndarray
    time: np.ndarray
    capacity: float
    energy: float


class MendeleyBatteryLoader:
    """
    Loader for Mendeley Forklift Battery Dataset.
    
    Structure:
    - CellX/RoundYY/RPT.csv
    - Columns: Cell, Round, Time, Current, Voltage, Energy, Temperature
    
    Usage:
        loader = MendeleyBatteryLoader('/path/to/Cell1')
        loader.load()
        cycles = loader.get_discharge_cycles()
    """
    
    def __init__(self, cell_path: str, cell_id: str = None):
        """
        Initialize loader.
        
        Args:
            cell_path: Path to CellX directory
            cell_id: Cell identifier (auto-detected if not provided)
        """
        self.cell_path = cell_path
        self.cell_id = cell_id or os.path.basename(cell_path)
        self.raw_data = None
        self.cycles = []
        self.initial_capacity = None
        self.round_capacities = {}
        
    def load(self) -> bool:
        """
        Load all RPT.csv files from the cell directory.
        
        Returns:
            True if loaded successfully
        """
        # Find all Round directories
        round_dirs = sorted(glob.glob(os.path.join(self.cell_path, 'Round*')))
        
        if not round_dirs:
            print(f"No Round directories found in {self.cell_path}")
            return False
        
        print(f"Loading {self.cell_id}: {len(round_dirs)} rounds found")
        
        all_data = []
        for round_dir in round_dirs:
            rpt_file = os.path.join(round_dir, 'RPT.csv')
            
            if os.path.exists(rpt_file):
                try:
                    df = pd.read_csv(rpt_file)
                    round_num = int(os.path.basename(round_dir).replace('Round', ''))
                    df['round_dir'] = round_num
                    all_data.append(df)
                except Exception as e:
                    print(f"Error loading {rpt_file}: {e}")
        
        if all_data:
            self.raw_data = pd.concat(all_data, ignore_index=True)
            print(f"  Loaded {len(self.raw_data)} total data points from {len(all_data)} rounds")
            return True
        
        return False
    
    def parse_cycles(self) -> List[MendeleyCycle]:
        """
        Parse raw data into discharge cycles.
        
        The Mendeley dataset uses current sign to indicate charge/discharge:
        - Negative current = discharge
        - Positive current = charge
        
        Returns:
            List of MendeleyCycle objects
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        cycles = []
        cycle_count = 0
        
        # Process each round
        for round_num in sorted(self.raw_data['round_dir'].unique()):
            round_data = self.raw_data[self.raw_data['round_dir'] == round_num].copy()
            round_data = round_data.sort_values('Time')
            
            # Find discharge segments (negative current)
            current = round_data['Current'].values
            is_discharge = current < -0.01  # Small threshold
            
            # Find segment boundaries
            changes = np.diff(is_discharge.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1
            
            # Handle edge cases
            if len(starts) == 0 and len(ends) == 0:
                # No clear discharge segments
                continue
            
            if is_discharge[0]:
                starts = np.insert(starts, 0, 0)
            if is_discharge[-1] and len(ends) < len(starts):
                ends = np.append(ends, len(current))
            
            # Extract each discharge segment
            for start, end in zip(starts, ends):
                if end - start < 50:  # Skip very short segments
                    continue
                
                segment = round_data.iloc[start:end]
                
                voltage = segment['Voltage'].values
                curr = np.abs(segment['Current'].values)
                temp = segment['Temperature'].values
                time = segment['Time'].values
                
                # Calculate capacity (Ah)
                dt = np.diff(time, prepend=time[0])
                capacity = np.sum(curr * dt) / 3600
                
                # Calculate energy (Wh)
                energy = np.sum(voltage * curr * dt) / 3600
                
                if capacity < 0.01:  # Skip very low capacity segments
                    continue
                
                cycle = MendeleyCycle(
                    cell_id=self.cell_id,
                    round_number=int(round_num),
                    cycle_number=cycle_count,
                    voltage=voltage,
                    current=curr,
                    temperature=temp,
                    time=time - time[0],  # Normalize time to start at 0
                    capacity=capacity,
                    energy=energy
                )
                
                cycles.append(cycle)
                cycle_count += 1
                
                # Track capacity per round
                if round_num not in self.round_capacities:
                    self.round_capacities[round_num] = []
                self.round_capacities[round_num].append(capacity)
        
        self.cycles = cycles
        
        # Set initial capacity (max of first few cycles)
        if cycles:
            first_caps = [c.capacity for c in cycles[:5]]
            self.initial_capacity = max(first_caps) if first_caps else 1.0
        
        print(f"  Parsed {len(cycles)} discharge cycles")
        return cycles
    
    def get_discharge_cycles(self) -> List[MendeleyCycle]:
        """Get parsed discharge cycles."""
        if not self.cycles:
            self.parse_cycles()
        return self.cycles
    
    def get_soh_series(self) -> pd.DataFrame:
        """
        Get SOH for each cycle.
        
        Returns:
            DataFrame with cycle_number, capacity, soh
        """
        if not self.cycles:
            self.parse_cycles()
        
        if not self.initial_capacity or self.initial_capacity == 0:
            return pd.DataFrame()
        
        data = []
        for cycle in self.cycles:
            soh = (cycle.capacity / self.initial_capacity) * 100
            data.append({
                'cell_id': cycle.cell_id,
                'round_number': cycle.round_number,
                'cycle_number': cycle.cycle_number,
                'capacity': cycle.capacity,
                'soh': min(soh, 100.0),
                'energy': cycle.energy
            })
        
        return pd.DataFrame(data)
    
    def get_round_summary(self) -> pd.DataFrame:
        """Get capacity summary per round."""
        if not self.round_capacities:
            self.parse_cycles()
        
        data = []
        for round_num, caps in sorted(self.round_capacities.items()):
            data.append({
                'round': round_num,
                'mean_capacity': np.mean(caps),
                'max_capacity': np.max(caps),
                'n_cycles': len(caps)
            })
        
        return pd.DataFrame(data)


class MendeleyDatasetLoader:
    """
    Load all cells from the Mendeley forklift battery dataset.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize loader.
        
        Args:
            dataset_path: Path to the root dataset directory containing Cell1, Cell2, Cell3
        """
        self.dataset_path = dataset_path
        self.cells = {}
        self.all_cycles = []
        
    def load_all_cells(self) -> Dict[str, MendeleyBatteryLoader]:
        """Load all cells from the dataset."""
        cell_dirs = sorted(glob.glob(os.path.join(self.dataset_path, 'Cell*')))
        
        print(f"Found {len(cell_dirs)} cells in {self.dataset_path}")
        
        for cell_dir in cell_dirs:
            cell_id = os.path.basename(cell_dir)
            loader = MendeleyBatteryLoader(cell_dir, cell_id)
            
            if loader.load():
                loader.parse_cycles()
                self.cells[cell_id] = loader
                self.all_cycles.extend(loader.cycles)
        
        print(f"\nTotal: {len(self.all_cycles)} discharge cycles from {len(self.cells)} cells")
        return self.cells
    
    def get_combined_soh(self) -> pd.DataFrame:
        """Get combined SOH data from all cells."""
        dfs = []
        for cell_id, loader in self.cells.items():
            df = loader.get_soh_series()
            if len(df) > 0:
                dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    
    def get_dataset_summary(self) -> Dict:
        """Get summary statistics for the entire dataset."""
        summary = {
            'n_cells': len(self.cells),
            'n_total_cycles': len(self.all_cycles),
            'cells': {}
        }
        
        for cell_id, loader in self.cells.items():
            soh_df = loader.get_soh_series()
            summary['cells'][cell_id] = {
                'n_cycles': len(loader.cycles),
                'n_rounds': len(loader.round_capacities),
                'initial_capacity': loader.initial_capacity,
                'soh_range': [soh_df['soh'].min(), soh_df['soh'].max()] if len(soh_df) > 0 else [0, 0]
            }
        
        return summary


def explore_mendeley_dataset(dataset_path: str) -> Dict:
    """
    Quick exploration of Mendeley dataset structure.
    """
    info = {
        'path': dataset_path,
        'cells': [],
        'total_rounds': 0,
        'sample_columns': []
    }
    
    cell_dirs = glob.glob(os.path.join(dataset_path, 'Cell*'))
    
    for cell_dir in cell_dirs:
        cell_id = os.path.basename(cell_dir)
        round_dirs = glob.glob(os.path.join(cell_dir, 'Round*'))
        
        info['cells'].append({
            'id': cell_id,
            'n_rounds': len(round_dirs)
        })
        info['total_rounds'] += len(round_dirs)
        
        # Sample first CSV
        if round_dirs and not info['sample_columns']:
            rpt_file = os.path.join(round_dirs[0], 'RPT.csv')
            if os.path.exists(rpt_file):
                df = pd.read_csv(rpt_file, nrows=5)
                info['sample_columns'] = list(df.columns)
    
    return info


if __name__ == "__main__":
    print("Mendeley Battery Loader")
    print("=" * 50)
    
    # Example path
    dataset_path = "./MENDELEY_Lithium-ion battery degradation dataset based on a realistic forklift operation profile"
    
    if os.path.exists(dataset_path):
        info = explore_mendeley_dataset(dataset_path)
        print(f"\nDataset: {info['path']}")
        print(f"Cells: {len(info['cells'])}")
        print(f"Total rounds: {info['total_rounds']}")
        print(f"Columns: {info['sample_columns']}")
    else:
        print(f"Dataset not found at: {dataset_path}")
