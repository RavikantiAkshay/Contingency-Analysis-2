import importlib
from typing import Dict, Any, Optional, Tuple


# Map of supported PyPOWER case names to their module/function paths.
# Extensible: add new cases here to include them in the network pool.
SUPPORTED_CASES = {
    'case9': 'pypower.case9',
    'case14': 'pypower.case14',
    'case30': 'pypower.case30',
    'case39': 'pypower.case39',
    'case57': 'pypower.case57',
    'case118': 'pypower.case118',
    'case300': 'pypower.case300',
}


def load_network(case_name: str = 'case14', mpc: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads a PyPOWER case by name and extracts network metadata.

    Parameterized to support any standard PyPOWER test case. Defaults to case14
    for backward compatibility with the existing simulation pipeline.

    Args:
        case_name (str): Name of the PyPOWER case to load (e.g., 'case14', 'case30').
        mpc (dict, optional): Custom PyPOWER case dict. If provided, case_name is ignored.

    Returns:
        mpc (dict): Raw PyPOWER case dictionary.
        meta (dict): Network metadata dictionary containing:
            - network_name (str): Case name used to load this network
            - n_bus (int): Total bus count
            - n_line (int): Total branch count
            - n_gen (int): Total generator count
            - baseMVA (float): System base MVA
            - bus_names (list): List of bus IDs as strings
            - line_names (list): List of branch identifiers (from_bus-to_bus)
            - gen_names (list): List of generator identifiers
            - vsource_name (str): Identifier of slack/swing generator
    """
    if mpc is None:
        if case_name not in SUPPORTED_CASES:
            raise ValueError(
                f"Unsupported case name '{case_name}'. "
                f"Supported: {list(SUPPORTED_CASES.keys())}"
            )
        module = importlib.import_module(SUPPORTED_CASES[case_name])
        case_fn = getattr(module, case_name)
        mpc = case_fn()

    n_bus = len(mpc['bus'])
    n_branch = len(mpc['branch'])
    n_gen = len(mpc['gen'])

    # Derive branch names dynamically from the mpc struct (no hardcoding)
    line_names = []
    for b in mpc['branch']:
        f_bus = int(b[0])
        t_bus = int(b[1])
        line_names.append(f"{f_bus}-{t_bus}")

    # Identify slack bus generator
    slack_bus_id = None
    for bus in mpc['bus']:
        if int(bus[1]) == 3:  # BUS_TYPE == 3 (slack/reference)
            slack_bus_id = int(bus[0])
            break

    # Build generator names and identify slack generator
    gen_names = []
    vsource_name = "Gen.1"
    for i, g in enumerate(mpc['gen']):
        gen_bus = int(g[0])
        name = f"Gen.{i+1}"
        gen_names.append(name)
        if gen_bus == slack_bus_id:
            vsource_name = name

    meta = {
        'network_name': case_name,
        'n_bus': n_bus,
        'n_line': n_branch,
        'n_gen': n_gen,
        'baseMVA': float(mpc['baseMVA']),
        'bus_names': [str(int(b[0])) for b in mpc['bus']],
        'line_names': line_names,
        'gen_names': gen_names,
        'vsource_name': vsource_name,
    }

    return mpc, meta
