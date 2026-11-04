
#!/usr/bin/env python3
"""
Example client for Battery SOH Prediction API
"""

import requests
import numpy as np

API_URL = "http://localhost:8000"

def predict_from_cycle(voltage, current, temperature=None):
    """
    Predict SOH from raw cycle data.

    Args:
        voltage: List of voltage measurements (V)
        current: List of current measurements (A)
        temperature: Optional list of temperature measurements (C)

    Returns:
        Predicted SOH (%)
    """
    payload = {
        "cycle_data": {
            "voltage": list(voltage),
            "current": list(current),
            "temperature": list(temperature) if temperature else None
        },
        "include_explanation": True
    }

    response = requests.post(f"{API_URL}/predict", json=payload)

    if response.ok:
        return response.json()
    else:
        raise Exception(f"API error: {response.status_code} - {response.text}")


def predict_from_features(features):
    """
    Predict SOH from pre-extracted features.

    Args:
        features: Dict with feature values

    Returns:
        Predicted SOH (%)
    """
    payload = {
        "features": features,
        "include_explanation": True
    }

    response = requests.post(f"{API_URL}/predict", json=payload)

    if response.ok:
        return response.json()
    else:
        raise Exception(f"API error: {response.status_code} - {response.text}")


if __name__ == "__main__":
    # Example 1: From features
    result = predict_from_features({
        "voltage_mean": 3.5,
        "voltage_std": 0.2,
        "capacity": 1.8,
        "discharge_time": 3600
    })

    print(f"Predicted SOH: {result['soh']:.1f}%")
    print(f"Explanation: {result['explanation']}")

    # Example 2: From raw cycle
    voltage = np.linspace(4.2, 3.0, 100).tolist()
    current = [-1.5] * 100

    result = predict_from_cycle(voltage, current)
    print(f"Predicted SOH: {result['soh']:.1f}%")
