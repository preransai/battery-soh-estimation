"""
Utility functions for Phase-1 Battery SOH Prediction
"""

import numpy as np
import yaml
import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # Create logger
    logger = logging.getLogger('BatterySOH')
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_config.get('save_to_file', False):
        log_file = log_config.get('log_file', 'logs/phase1.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories for outputs."""
    paths = config.get('paths', {})
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Created directory structure:")
    for key, path in paths.items():
        print(f"  - {key}: {path}")


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def normalize_data(data: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize data using specified method.
    
    Args:
        data: Input data array
        method: 'minmax' or 'zscore'
    
    Returns:
        Normalized data and normalization parameters
    """
    if method == 'minmax':
        data_min = np.min(data)
        data_max = np.max(data)
        normalized = (data - data_min) / (data_max - data_min + 1e-8)
        params = {'min': data_min, 'max': data_max, 'method': 'minmax'}
    
    elif method == 'zscore':
        data_mean = np.mean(data)
        data_std = np.std(data)
        normalized = (data - data_mean) / (data_std + 1e-8)
        params = {'mean': data_mean, 'std': data_std, 'method': 'zscore'}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_data(normalized_data: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Denormalize data using stored parameters."""
    method = params['method']
    
    if method == 'minmax':
        return normalized_data * (params['max'] - params['min']) + params['min']
    
    elif method == 'zscore':
        return normalized_data * params['std'] + params['mean']
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def detect_outliers(data: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers in data.
    
    Args:
        data: Input data array
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection
    
    Returns:
        Boolean mask where True indicates outlier
    """
    if method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outliers = z_scores > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return outliers


def plot_style_setup():
    """Setup matplotlib style for consistent plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10


def save_plot(fig, filename: str, config: Dict[str, Any], dpi: int = 300):
    """Save plot to file."""
    plot_dir = config['paths']['plots']
    filepath = os.path.join(plot_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved plot: {filepath}")


def print_section_header(title: str, char: str = "="):
    """Print formatted section header."""
    width = 80
    print("\n" + char * width)
    print(f"{title.center(width)}")
    print(char * width + "\n")


def print_subsection_header(title: str):
    """Print formatted subsection header."""
    print(f"\n{'─' * 80}")
    print(f"► {title}")
    print('─' * 80)


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Returns:
        Dictionary with MAE, RMSE, MAPE, R²
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Print metrics in formatted table."""
    print(f"\n{title}:")
    print("─" * 40)
    for metric_name, value in metrics.items():
        if metric_name == 'MAPE':
            print(f"  {metric_name:<10}: {value:>10.2f}%")
        else:
            print(f"  {metric_name:<10}: {value:>10.4f}")
    print("─" * 40)
