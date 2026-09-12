import os
import json
from typing import Dict, Any, List
import pandas as pd


def save_results(
    results: Dict[str, Any],
    label: str,
    violations: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    case_name: str = "normal_base_case",
    output_dir: str = None
) -> Dict[str, str]:
    """
    Persists full load flow simulation results (P, Q, V, I, Loadings, Security Status) to disk.

    Updates/overwrites saved CSV and JSON files for the specified case_name every time it is run.

    Args:
        results (dict): Results dictionary returned by extract_results().
        label (str): Operational security classification ('Safe', 'Alert', 'Critical').
        violations (list[dict]): Constraint violation details.
        cfg (dict): Configuration dictionary containing output paths.
        case_name (str): Unique identifier for the simulation run (default: "normal_base_case").
        output_dir (str, optional): Target output directory. If None, uses tabular_base_cases dir.

    Returns:
        saved_paths (dict): Dictionary mapping file types to saved file paths.
    """
    target_dir = output_dir if output_dir is not None else cfg['paths'].get('tabular_base_cases', cfg['paths']['processed'])

    os.makedirs(target_dir, exist_ok=True)

    saved_paths = {}

    # 1. Save Bus P, Q, V, Angle CSV
    if 'bus_table' in results and isinstance(results['bus_table'], pd.DataFrame) and not results['bus_table'].empty:
        bus_csv_path = os.path.join(target_dir, f"{case_name}_buses.csv")
        results['bus_table'].to_csv(bus_csv_path, index=False)
        saved_paths['bus_csv'] = bus_csv_path

    # 2. Save Branch P, Q, I, Loading CSV
    if 'branch_table' in results and isinstance(results['branch_table'], pd.DataFrame) and not results['branch_table'].empty:
        branch_csv_path = os.path.join(target_dir, f"{case_name}_branches.csv")
        results['branch_table'].to_csv(branch_csv_path, index=False)
        saved_paths['branch_csv'] = branch_csv_path

    # 3. Save Summary JSON
    summary_data = {
        'case_name': case_name,
        'convergence': results.get('convergence', False),
        'security_label': label,
        'violations_count': len(violations),
        'violations': violations,
        'metrics': {
            'total_load_mw': round(results.get('total_load', 0.0) / 1000.0, 3),
            'total_load_mvar': round(results.get('total_load_mvar', 0.0) / 1000.0, 3),
            'total_gen_mw': round(results.get('total_gen_mw', 0.0), 3),
            'total_gen_mvar': round(results.get('total_gen_mvar', 0.0), 3),
            'losses_mw': round(results.get('total_gen_mw', 0.0) - results.get('total_load', 0.0) / 1000.0, 3),
            'losses_mvar': round(results.get('total_gen_mvar', 0.0) - results.get('total_load_mvar', 0.0) / 1000.0, 3),
            'min_voltage_pu': round(results.get('min_voltage', 0.0), 4),
            'max_voltage_pu': round(results.get('max_voltage', 0.0), 4),
            'max_line_loading_pct': round(results.get('max_line_loading', 0.0), 2),
            'max_voltage_deviation_pu': round(results.get('max_voltage_deviation', 0.0), 4),
        }
    }

    summary_json_path = os.path.join(target_dir, f"{case_name}_summary.json")
    with open(summary_json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    saved_paths['summary_json'] = summary_json_path

    return saved_paths
