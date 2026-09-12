from typing import Dict, Any, Optional


def extract_results(
    raw_results: Optional[Dict[str, Any]],
    converged: bool = True,
    meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extracts and standardizes electrical quantities from solved simulation output.

    Args:
        raw_results (dict or None): Dictionary returned by solver containing tables/lists.
        converged (bool): Whether power flow converged.
        meta (dict, optional): Network metadata dictionary.

    Returns:
        results (dict): Standardized results dictionary compliant with LLD Section 3.8.
    """
    if not converged or raw_results is None:
        return {
            'convergence': False,
            'bus_voltages': [],
            'bus_angles': [],
            'line_loadings': [],
            'gen_powers': [],
            'min_voltage': None,
            'max_voltage': None,
            'max_line_loading': None,
            'max_voltage_deviation': None,
            'total_load': None,
            'total_load_mvar': None,
            'total_gen_mw': None,
            'total_gen_mvar': None,
            'bus_table': None,
            'branch_table': None,
        }

    return {
        'convergence': True,
        'bus_voltages': raw_results.get('bus_voltages', []),
        'bus_angles': raw_results.get('bus_angles', []),
        'line_loadings': raw_results.get('line_loadings', []),
        'gen_powers': raw_results.get('gen_powers', []),
        'min_voltage': raw_results.get('min_voltage'),
        'max_voltage': raw_results.get('max_voltage'),
        'max_line_loading': raw_results.get('max_line_loading'),
        'max_voltage_deviation': raw_results.get('max_voltage_deviation'),
        'total_load': raw_results.get('total_load'),
        'total_load_mvar': raw_results.get('total_load_mvar'),
        'total_gen_mw': raw_results.get('total_gen_mw'),
        'total_gen_mvar': raw_results.get('total_gen_mvar'),
        'bus_table': raw_results.get('bus_table'),
        'branch_table': raw_results.get('branch_table'),
    }
