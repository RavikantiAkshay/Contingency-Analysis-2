import os
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    """
    Returns the centralized configuration dictionary for the EEQ401 project.

    All tuneable parameters, paths, thresholds, and solver settings are defined here.
    Validates parameter types and numeric boundaries before returning.

    Returns:
        cfg (dict): Configuration dictionary.
    """
    cfg = {
        'load_scales': {
            'normal': 1.0,
            'light': 0.7,
            'heavy': 1.3,
        },
        'campaign_sweep': {
            'mode': 'randomized_targeted',
            'min_scale': 0.80,
            'max_scale': 1.20,
            'num_base_scenarios': 100,
        },
        'security': {
            'v_safe_min': 0.95,       # p.u.
            'v_safe_max': 1.10,       # p.u.
            'v_alert_min': 0.90,      # p.u.
            'v_alert_max': 1.15,      # p.u.
            'line_safe_max': 100.0,   # % loading (continuous thermal rating)
            'line_alert_max': 120.0,  # % loading (short-term emergency rating)
        },
        'solver': {
            'max_iterations': 100,
            'tolerance': 0.0001,
        },
        'ml': {
            'cv_folds': 5,
            'random_seed': 42,
            'test_size': 0.2,
        },
        'paths': {
            'data_dir': 'data',
            'tabular_dir': os.path.join('data', 'tabular'),
            'tabular_master': os.path.join('data', 'tabular', 'master'),
            'tabular_base_cases': os.path.join('data', 'tabular', 'base_cases'),
            'processed': os.path.join('data', 'tabular', 'master'),
            'ml_ready': os.path.join('data', 'ml_ready'),
            'models': 'models',
            'logs': 'logs',
        },
        'network': {
            'exclude_slack': True,    # Do not disable slack generator as a contingency
            'exclude_vsource': True,  # Backward compatibility key
        },
    }

    # Parameter Validation
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Dict[str, Any]) -> None:
    """Internal validator for configuration dictionary."""
    # Check load scales
    for scale_name, scale_val in cfg['load_scales'].items():
        if not isinstance(scale_val, (int, float)) or scale_val <= 0:
            raise ValueError(f"Invalid load scale '{scale_name}': must be a positive number.")

    # Check security thresholds
    sec = cfg['security']
    if not (sec['v_alert_min'] <= sec['v_safe_min'] < sec['v_safe_max'] <= sec['v_alert_max']):
        raise ValueError("Invalid voltage thresholds: must satisfy v_alert_min <= v_safe_min < v_safe_max <= v_alert_max.")
    if not (0 < sec['line_safe_max'] <= sec['line_alert_max']):
        raise ValueError("Invalid line loading thresholds: must satisfy line_safe_max <= line_alert_max.")

    # Check solver settings
    sol = cfg['solver']
    if not isinstance(sol['max_iterations'], int) or sol['max_iterations'] <= 0:
        raise ValueError("max_iterations must be a positive integer.")
    if not isinstance(sol['tolerance'], (int, float)) or sol['tolerance'] <= 0:
        raise ValueError("tolerance must be a positive number.")
