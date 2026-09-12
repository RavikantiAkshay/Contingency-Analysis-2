"""
Master Dataset Builder.
Assembles simulation campaign results into a single consolidated Pandas DataFrame
matching LLD Section 9.1 schema (65 columns) and exports master_dataset.csv.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def build_dataset(
    all_results: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    filename: str = 'master_dataset.csv',
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Converts raw simulation campaign results into the master pandas DataFrame.

    Args:
        all_results (list of dicts): Campaign output from run_simulation_campaign().
        cfg (dict): Configuration dict containing output paths.
        filename (str): Target CSV filename (default: master_dataset.csv).
        output_dir (str, optional): Target directory path. If None, uses cfg['paths']['processed'].

    Returns:
        df (pd.DataFrame): Consolidated master dataset matching LLD §9.1 schema.
    """
    rows = []

    for item in all_results:
        case_id = item['case_id']
        ctype = item['contingency_type']
        celem = str(item['contingency_element'])
        converged = item['convergence']
        res = item.get('results', {})
        label = item.get('label', 'Critical')

        row = {
            'case_id': case_id,
            'contingency_type': ctype,
            'contingency_element': celem,
            'convergence': converged,
        }

        if converged and res:
            # 14 Bus Voltages
            bus_v = res.get('bus_voltages', [])
            for b_idx in range(14):
                row[f'V_bus{b_idx + 1}'] = bus_v[b_idx] if b_idx < len(bus_v) else np.nan

            # 14 Bus Angles
            bus_a = res.get('bus_angles', [])
            for b_idx in range(14):
                row[f'angle_bus{b_idx + 1}'] = bus_a[b_idx] if b_idx < len(bus_a) else np.nan

            # 20 Line Loadings
            line_l = res.get('line_loadings', [])
            for l_idx in range(20):
                row[f'line_loading_{l_idx + 1}'] = line_l[l_idx] if l_idx < len(line_l) else np.nan

            # 5 Generator Powers
            gen_p = res.get('gen_powers', [])
            for g_idx in range(5):
                row[f'gen_P_{g_idx + 1}'] = gen_p[g_idx] if g_idx < len(gen_p) else np.nan

            # 5 Derived Features
            row['min_bus_voltage'] = res.get('min_voltage', np.nan)
            row['max_bus_voltage'] = res.get('max_voltage', np.nan)
            row['max_line_loading'] = res.get('max_line_loading', np.nan)

            # Max voltage deviation from nominal 1.0 p.u.
            if bus_v:
                v_devs = [abs(v - 1.0) for v in bus_v]
                row['max_voltage_deviation'] = max(v_devs)
            else:
                row['max_voltage_deviation'] = np.nan

            row['total_load'] = res.get('total_load', np.nan) / 1000.0 if res.get('total_load') else np.nan  # MW
        else:
            # Non-convergent case encoding per LLD §9.3
            for b_idx in range(14):
                row[f'V_bus{b_idx + 1}'] = np.nan
                row[f'angle_bus{b_idx + 1}'] = np.nan
            for l_idx in range(20):
                row[f'line_loading_{l_idx + 1}'] = np.nan
            for g_idx in range(5):
                row[f'gen_P_{g_idx + 1}'] = np.nan

            row['min_bus_voltage'] = np.nan
            row['max_bus_voltage'] = np.nan
            row['max_line_loading'] = np.nan
            row['max_voltage_deviation'] = np.nan
            row['total_load'] = np.nan
            label = 'Critical'

        row['label'] = label
        rows.append(row)

    df = pd.DataFrame(rows)

    # Verify column count = 63 per refined schema
    expected_cols = (
        ['case_id', 'contingency_type', 'contingency_element', 'convergence'] +
        [f'V_bus{i+1}' for i in range(14)] +
        [f'angle_bus{i+1}' for i in range(14)] +
        [f'line_loading_{i+1}' for i in range(20)] +
        [f'gen_P_{i+1}' for i in range(5)] +
        ['min_bus_voltage', 'max_bus_voltage', 'max_line_loading', 'max_voltage_deviation', 'total_load'] +
        ['label']
    )
    df = df[expected_cols]

    # Save master dataset CSV
    target_dir = output_dir if output_dir is not None else cfg.get('paths', {}).get('processed', os.path.join('data', 'processed'))
    os.makedirs(target_dir, exist_ok=True)
    master_csv_path = os.path.join(target_dir, filename)
    df.to_csv(master_csv_path, index=False)
    print(f"[DATASET BUILD] Master dataset successfully written to '{master_csv_path}' ({len(df)} rows x {len(df.columns)} columns).")

    return df
