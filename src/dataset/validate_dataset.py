"""
Master Dataset Integrity Validator.
Performs verification checks on built DataFrame per LLD Section 3.12.1 specifications.
"""

import pandas as pd
from typing import Dict, Any, Tuple


def validate_dataset(dataset: pd.DataFrame, meta: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Performs comprehensive dataset integrity validation per LLD §3.12.1.

    Args:
        dataset (pd.DataFrame): Built master dataset DataFrame.
        meta (dict): Network metadata dictionary.
        cfg (dict): Configuration dict.

    Returns:
        is_valid (bool): True if all checks pass.
        report (dict): Summary report of validation results.
    """
    report = {
        'total_rows': len(dataset),
        'total_columns': len(dataset.columns),
        'unique_case_ids': dataset['case_id'].nunique(),
        'has_duplicates': False,
        'valid_column_count': (len(dataset.columns) == 63),
        'valid_labels': True,
        'valid_contingency_types': True,
        'converged_cases': int(dataset['convergence'].sum()),
        'non_converged_cases': int((~dataset['convergence']).sum()),
        'errors': [],
    }

    # 1. Duplicate case_ids check
    if report['unique_case_ids'] != report['total_rows']:
        report['has_duplicates'] = True
        report['errors'].append(f"Duplicate case_ids found! Total rows: {report['total_rows']}, Unique IDs: {report['unique_case_ids']}")

    # 2. Column count check
    if not report['valid_column_count']:
        report['errors'].append(f"Expected 63 columns per refined schema, got {report['total_columns']}")

    # 3. Label values check
    valid_labels = {'Safe', 'Alert', 'Critical'}
    found_labels = set(dataset['label'].dropna().unique())
    if not found_labels.issubset(valid_labels):
        report['valid_labels'] = False
        report['errors'].append(f"Invalid labels found: {found_labels - valid_labels}")

    # 4. Contingency type check
    valid_ctypes = {'none', 'line', 'generator'}
    found_ctypes = set(dataset['contingency_type'].dropna().unique())
    if not found_ctypes.issubset(valid_ctypes):
        report['valid_contingency_types'] = False
        report['errors'].append(f"Invalid contingency types found: {found_ctypes - valid_ctypes}")

    # 5. Converged cases numerical integrity check
    converged_df = dataset[dataset['convergence'] == True]
    voltage_cols = [f'V_bus{i+1}' for i in range(14)]
    if converged_df[voltage_cols].isna().any().any():
        report['errors'].append("NaN values found in voltage columns for converged cases!")

    # 6. Non-converged cases critical label check
    non_conv_df = dataset[dataset['convergence'] == False]
    if not non_conv_df.empty:
        non_conv_labels = set(non_conv_df['label'].unique())
        if non_conv_labels != {'Critical'}:
            report['errors'].append(f"Non-converged cases must be labeled 'Critical'! Found: {non_conv_labels}")

    is_valid = (len(report['errors']) == 0)
    return is_valid, report
