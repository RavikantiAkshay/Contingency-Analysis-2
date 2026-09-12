from typing import Tuple, List, Dict, Any


def classify_security(
    features: Dict[str, Any],
    converged: bool,
    cfg: Dict[str, Any]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Classifies system operational state as Safe, Alert, or Critical based on threshold policies.

    Args:
        features (dict): Extracted features dictionary from extract_results().
        converged (bool): Solution convergence flag.
        cfg (dict): Configuration dictionary containing security thresholds.

    Returns:
        label (str): Operational security classification ('Safe', 'Alert', or 'Critical').
        violations (list[dict]): Detailed list of constraint violations recorded.

    Raises:
        ValueError: If security configuration thresholds are missing or invalid.
    """
    if 'security' not in cfg:
        raise ValueError("Configuration dictionary missing mandatory 'security' section.")

    sec_cfg = cfg['security']
    v_safe_min = sec_cfg['v_safe_min']
    v_safe_max = sec_cfg['v_safe_max']
    v_alert_min = sec_cfg['v_alert_min']
    v_alert_max = sec_cfg['v_alert_max']
    line_safe_max = sec_cfg['line_safe_max']
    line_alert_max = sec_cfg['line_alert_max']

    violations = []

    # Non-convergence policy -> Critical
    if not converged:
        violations.append({
            'type': 'non_convergence',
            'element': 'system',
            'value': None,
            'limit': None,
            'severity': 'Critical'
        })
        return 'Critical', violations

    severity_level = 0  # 0: Safe, 1: Alert, 2: Critical

    # 1. Bus Voltage Violations
    bus_voltages = features.get('bus_voltages', [])
    bus_table = features.get('bus_table')

    for idx, v in enumerate(bus_voltages, 1):
        # Determine if we have valid bus-specific limits from PyPOWER for this bus
        has_dynamic_limits = False
        bus_v_max = None
        bus_v_min = None
        
        if bus_table is not None and not bus_table.empty:
            # We are assuming bus_table is indexed or ordered such that row i corresponds to Bus i+1
            # Or we can just lookup by bus_id
            bus_row = bus_table[bus_table['bus_id'] == idx]
            if not bus_row.empty:
                if 'v_max_pu' in bus_row.columns and 'v_min_pu' in bus_row.columns:
                    bus_v_max = float(bus_row.iloc[0]['v_max_pu'])
                    bus_v_min = float(bus_row.iloc[0]['v_min_pu'])
                    # Basic sanity check to ensure limits were actually provided in the dataset
                    if bus_v_max > 0 and bus_v_min > 0 and bus_v_max > bus_v_min:
                        has_dynamic_limits = True

        # Evaluate voltage against system threshold bands
        if v < v_alert_min or v > v_alert_max:
            severity_level = max(severity_level, 2)
            limit = v_alert_min if v < v_alert_min else v_alert_max
            violations.append({
                'type': 'voltage_underflow' if v < v_alert_min else 'voltage_overflow',
                'element': f'Bus_{idx}',
                'value': round(v, 4),
                'limit': limit,
                'severity': 'Critical'
            })
        elif v < v_safe_min or v > v_safe_max:
            severity_level = max(severity_level, 1)
            limit = v_safe_min if v < v_safe_min else v_safe_max
            violations.append({
                'type': 'voltage_underflow' if v < v_safe_min else 'voltage_overflow',
                'element': f'Bus_{idx}',
                'value': round(v, 4),
                'limit': limit,
                'severity': 'Alert'
            })

    # 2. Branch Thermal Loading Violations
    line_loadings = features.get('line_loadings', [])
    for idx, loading in enumerate(line_loadings, 1):
        if loading > line_alert_max:
            severity_level = max(severity_level, 2)
            violations.append({
                'type': 'thermal_overload',
                'element': f'Branch_{idx}',
                'value': round(loading, 2),
                'limit': line_alert_max,
                'severity': 'Critical'
            })
        elif loading > line_safe_max:
            severity_level = max(severity_level, 1)
            violations.append({
                'type': 'thermal_overload',
                'element': f'Branch_{idx}',
                'value': round(loading, 2),
                'limit': line_safe_max,
                'severity': 'Alert'
            })

    # Map severity integer to label string
    severity_map = {0: 'Safe', 1: 'Alert', 2: 'Critical'}
    label = severity_map[severity_level]

    return label, violations
