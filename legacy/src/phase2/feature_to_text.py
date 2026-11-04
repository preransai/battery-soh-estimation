"""
Phase-2 Step 4: Feature-to-Text Conversion
Converts physics features to natural language descriptions
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Project root
PROJECT_ROOT = '.'


@dataclass
class TextDescription:
    """Container for text descriptions of battery state."""
    battery_id: str
    cycle_number: int
    soh: float
    
    # Individual descriptions
    soh_text: str
    capacity_text: str
    ica_text: str
    resistance_text: str
    temperature_text: str
    
    # Combined summary
    full_description: str
    
    # Structured diagnosis
    diagnosis: str


class FeatureToText:
    """
    Convert physics features to natural language descriptions.
    
    Uses rule-based templates for consistent, interpretable text generation.
    """
    
    def __init__(self, templates_path: str = None):
        """
        Initialize converter.
        
        Args:
            templates_path: Path to templates YAML file
        """
        if templates_path is None:
            # Try both relative to current root and absolute via PROJECT_ROOT if available
            candidates = [
                os.path.join(os.getcwd(), 'src/phase2/text_templates.yaml'),
                os.path.join(os.path.dirname(__file__), 'text_templates.yaml')
            ]
            for c in candidates:
                if os.path.exists(c):
                    templates_path = c
                    break
            if templates_path is None:
                templates_path = 'src/phase2/text_templates.yaml' # Fallback
        
        with open(templates_path, 'r') as f:
            self.templates = yaml.safe_load(f)
        
        # Reference values for comparisons (will be set from training data)
        self.reference_values = {
            'initial_capacity': 2.0,  # Ah
            'initial_ica_peak_v': 3.7,  # V
            'initial_ica_magnitude': 1.0,
            'initial_dcir': 0.05  # Ohms
        }
    
    def set_reference_values(self, initial_features: Dict) -> None:
        """
        Set reference values from initial cycle features.
        
        Args:
            initial_features: Dictionary with initial cycle features
        """
        self.reference_values.update({
            'initial_capacity': initial_features.get('capacity', 2.0),
            'initial_ica_peak_v': initial_features.get('ica_peak_voltage', 3.7),
            'initial_ica_magnitude': initial_features.get('ica_peak_magnitude', 1.0),
            'initial_dcir': initial_features.get('dcir', 0.05)
        })
    
    def convert_single(self, features: Dict) -> TextDescription:
        """
        Convert a single cycle's features to text description.
        
        Args:
            features: Dictionary with cycle features
        
        Returns:
            TextDescription object
        """
        # Extract features
        soh = features.get('soh', 100.0)
        capacity = features.get('capacity', 2.0)
        ica_peak_v = features.get('ica_peak_voltage', 3.7)
        ica_magnitude = features.get('ica_peak_magnitude', 1.0)
        dcir = features.get('dcir', 0.05)
        temp_delta = features.get('temp_delta', 1.0)
        cycle_number = features.get('cycle_number', 0)
        battery_id = features.get('battery_id', 'unknown')
        
        # Generate individual descriptions
        soh_text = self._generate_soh_text(soh)
        capacity_text = self._generate_capacity_text(capacity)
        ica_text = self._generate_ica_text(ica_peak_v, ica_magnitude)
        resistance_text = self._generate_resistance_text(dcir)
        temperature_text = self._generate_temperature_text(temp_delta)
        
        # Generate full description
        full_description = self._generate_full_description(
            soh, soh_text, capacity_text, ica_text, resistance_text, temperature_text
        )
        
        # Generate diagnosis
        diagnosis = self._generate_diagnosis(soh, capacity, ica_peak_v, dcir, temp_delta)
        
        return TextDescription(
            battery_id=battery_id,
            cycle_number=cycle_number,
            soh=soh,
            soh_text=soh_text,
            capacity_text=capacity_text,
            ica_text=ica_text,
            resistance_text=resistance_text,
            temperature_text=temperature_text,
            full_description=full_description,
            diagnosis=diagnosis
        )
    
    def convert_batch(self, features_df: pd.DataFrame) -> List[TextDescription]:
        """
        Convert a DataFrame of features to text descriptions.
        
        Args:
            features_df: DataFrame with cycle features
        
        Returns:
            List of TextDescription objects
        """
        # Set reference from first cycle
        first_row = features_df.iloc[0].to_dict()
        self.set_reference_values(first_row)
        
        descriptions = []
        
        for _, row in features_df.iterrows():
            desc = self.convert_single(row.to_dict())
            descriptions.append(desc)
        
        return descriptions
    
    def _generate_soh_text(self, soh: float) -> str:
        """Generate SOH status text."""
        templates = self.templates['soh_status']
        
        if soh >= 95:
            return templates['excellent']['template'].format(soh=soh)
        elif soh >= 85:
            return templates['good']['template'].format(soh=soh)
        elif soh >= 75:
            return templates['moderate']['template'].format(soh=soh)
        elif soh >= 65:
            return templates['degraded']['template'].format(soh=soh)
        else:
            return templates['poor']['template'].format(soh=soh)
    
    def _generate_capacity_text(self, capacity: float) -> str:
        """Generate capacity fade text."""
        initial = self.reference_values['initial_capacity']
        ratio = capacity / initial
        fade_pct = (1 - ratio) * 100
        
        templates = self.templates['capacity_fade']
        
        if ratio > 0.95:
            return templates['minimal']['template']
        elif ratio > 0.85:
            return templates['slight']['template'].format(fade_pct=fade_pct)
        elif ratio > 0.75:
            return templates['moderate']['template'].format(fade_pct=fade_pct)
        elif ratio > 0.65:
            return templates['significant']['template'].format(fade_pct=fade_pct)
        else:
            return templates['severe']['template'].format(fade_pct=fade_pct)
    
    def _generate_ica_text(self, peak_v: float, magnitude: float) -> str:
        """Generate ICA analysis text."""
        initial_peak = self.reference_values['initial_ica_peak_v']
        initial_mag = self.reference_values['initial_ica_magnitude']
        
        shift = abs(peak_v - initial_peak)
        mag_ratio = magnitude / (initial_mag + 1e-8)
        
        # Peak shift text
        templates = self.templates['ica_peak']
        if shift < 0.05:
            shift_text = templates['stable']['template'].format(peak_v=peak_v)
        elif shift < 0.1:
            shift_text = templates['slight_shift']['template'].format(peak_v=peak_v, shift=shift)
        elif shift < 0.2:
            shift_text = templates['moderate_shift']['template'].format(peak_v=peak_v, shift=shift)
        else:
            shift_text = templates['significant_shift']['template'].format(peak_v=peak_v, shift=shift)
        
        # Magnitude text
        mag_templates = self.templates['ica_magnitude']
        if mag_ratio > 0.8:
            mag_text = mag_templates['strong']['template'].format(magnitude=magnitude)
        elif mag_ratio > 0.6:
            mag_text = mag_templates['reduced']['template'].format(magnitude=magnitude)
        elif mag_ratio > 0.4:
            mag_text = mag_templates['weak']['template'].format(magnitude=magnitude)
        else:
            mag_text = mag_templates['very_weak']['template'].format(magnitude=magnitude)
        
        return f"{shift_text} {mag_text}"
    
    def _generate_resistance_text(self, dcir: float) -> str:
        """Generate internal resistance text."""
        templates = self.templates['dcir']
        
        if dcir < 0.05:
            return templates['low']['template'].format(dcir=dcir)
        elif dcir < 0.1:
            return templates['normal']['template'].format(dcir=dcir)
        elif dcir < 0.2:
            return templates['elevated']['template'].format(dcir=dcir)
        elif dcir < 0.5:
            return templates['high']['template'].format(dcir=dcir)
        else:
            return templates['very_high']['template'].format(dcir=dcir)
    
    def _generate_temperature_text(self, delta_t: float) -> str:
        """Generate temperature behavior text."""
        templates = self.templates['temperature']
        
        if delta_t < 2.0:
            return templates['stable']['template'].format(delta_t=delta_t)
        elif delta_t < 5.0:
            return templates['mild_variation']['template'].format(delta_t=delta_t)
        elif delta_t < 10.0:
            return templates['moderate_variation']['template'].format(delta_t=delta_t)
        else:
            return templates['high_variation']['template'].format(delta_t=delta_t)
    
    def _generate_full_description(self, soh: float, soh_text: str, 
                                    capacity_text: str, ica_text: str,
                                    resistance_text: str, temperature_text: str) -> str:
        """Generate combined full description."""
        summary_templates = self.templates['summary']
        
        if soh >= 85:
            template = summary_templates['healthy']
        elif soh >= 70:
            template = summary_templates['degrading']
        else:
            template = summary_templates['critical']
        
        return template.format(
            soh=soh,
            capacity_text=capacity_text,
            ica_text=ica_text,
            resistance_text=resistance_text,
            temperature_text=temperature_text
        ).strip()
    
    def _generate_diagnosis(self, soh: float, capacity: float, 
                           ica_peak_v: float, dcir: float, delta_t: float) -> str:
        """Generate diagnostic summary."""
        factors = []
        
        # Check each factor
        initial_cap = self.reference_values['initial_capacity']
        if capacity / initial_cap < 0.85:
            factors.append("capacity fade")
        
        initial_peak = self.reference_values['initial_ica_peak_v']
        if abs(ica_peak_v - initial_peak) > 0.1:
            factors.append("electrode structural change")
        
        if dcir > 0.1:
            factors.append("increased internal resistance")
        
        if delta_t > 5.0:
            factors.append("thermal stress")
        
        if not factors:
            factors = ["normal aging"]
        
        factors_text = ", ".join(factors)
        
        # Generate recommendation
        diag_templates = self.templates['diagnosis']['recommendations']
        if soh >= 85:
            rec_text = diag_templates['healthy']['template']
        elif soh >= 70:
            rec_text = diag_templates['monitor']['template']
        else:
            remaining_cycles = int((soh - 60) / 0.2)  # Rough estimate
            rec_text = diag_templates['replace']['template'].format(cycles=max(10, remaining_cycles))
        
        return f"Diagnosis: {factors_text}. {rec_text}"
    
    def get_text_for_embedding(self, description: TextDescription, 
                               mode: str = 'full') -> str:
        """
        Get text string suitable for LLM embedding.
        
        Args:
            description: TextDescription object
            mode: 'full', 'summary', or 'diagnosis'
        
        Returns:
            Text string for embedding
        """
        if mode == 'full':
            return description.full_description
        elif mode == 'summary':
            return f"{description.soh_text} {description.capacity_text} {description.ica_text}"
        elif mode == 'diagnosis':
            return description.diagnosis
        else:
            return description.full_description
    
    def descriptions_to_dataframe(self, descriptions: List[TextDescription]) -> pd.DataFrame:
        """Convert list of TextDescriptions to DataFrame."""
        data = []
        for desc in descriptions:
            data.append({
                'battery_id': desc.battery_id,
                'cycle_number': desc.cycle_number,
                'soh': desc.soh,
                'soh_text': desc.soh_text,
                'capacity_text': desc.capacity_text,
                'ica_text': desc.ica_text,
                'resistance_text': desc.resistance_text,
                'temperature_text': desc.temperature_text,
                'full_description': desc.full_description,
                'diagnosis': desc.diagnosis
            })
        
        return pd.DataFrame(data)


def convert_features_to_text(features_df: pd.DataFrame, 
                            templates_path: str = None) -> pd.DataFrame:
    """
    Convenience function to convert features DataFrame to text.
    
    Args:
        features_df: DataFrame with physics features
        templates_path: Path to templates file
    
    Returns:
        DataFrame with text descriptions
    """
    converter = FeatureToText(templates_path)
    descriptions = converter.convert_batch(features_df)
    return converter.descriptions_to_dataframe(descriptions)


if __name__ == "__main__":
    # Test the converter
    import pandas as pd
    
    # Sample features
    sample_features = pd.DataFrame([
        {
            'battery_id': 'B0005',
            'cycle_number': 10,
            'soh': 98.5,
            'capacity': 1.97,
            'ica_peak_voltage': 3.68,
            'ica_peak_magnitude': 0.95,
            'dcir': 0.045,
            'temp_delta': 1.5
        },
        {
            'battery_id': 'B0005',
            'cycle_number': 100,
            'soh': 82.3,
            'capacity': 1.65,
            'ica_peak_voltage': 3.55,
            'ica_peak_magnitude': 0.72,
            'dcir': 0.12,
            'temp_delta': 3.2
        },
        {
            'battery_id': 'B0005',
            'cycle_number': 160,
            'soh': 68.1,
            'capacity': 1.36,
            'ica_peak_voltage': 3.42,
            'ica_peak_magnitude': 0.45,
            'dcir': 0.25,
            'temp_delta': 5.8
        }
    ])
    
    # Convert
    text_df = convert_features_to_text(sample_features)
    
    print("="*80)
    print("FEATURE-TO-TEXT CONVERSION TEST")
    print("="*80)
    
    for _, row in text_df.iterrows():
        print(f"\n--- Cycle {row['cycle_number']} (SOH: {row['soh']:.1f}%) ---")
        print(f"\nFull Description:\n{row['full_description']}")
        print(f"\nDiagnosis: {row['diagnosis']}")
        print("-"*80)
