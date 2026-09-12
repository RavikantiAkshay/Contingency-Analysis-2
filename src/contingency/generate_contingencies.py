"""
Enumerate N-1 contingencies from PyPOWER network.
"""
from typing import Dict, Any, List, Optional
from pypower.api import case14


def generate_contingencies(
    meta: Dict[str, Any],
    mpc: Optional[Dict[str, Any]] = None,
    contingency_type: str = "line"
) -> List[Dict[str, Any]]:
    """
    Enumerates N-1 contingencies (line and/or generator) from the network.

    Args:
        meta (dict): Network metadata dictionary from load_network().
        mpc (dict, optional): PyPOWER case dict. If None, loads case14().
        contingency_type (str): Type of contingencies to generate: 'line', 'generator', or 'all'.

    Returns:
        contingencies (list of dicts): List of contingency descriptors.
    """
    if mpc is None:
        mpc = case14()

    contingencies = []

    # 1. Branch / Line Contingencies
    if contingency_type in ("line", "all"):
        for idx, branch in enumerate(mpc['branch']):
            # Only include enabled branches
            if branch[10] > 0:
                bus1 = int(branch[0])
                bus2 = int(branch[1])
                line_name = meta['line_names'][idx] if ('line_names' in meta and idx < len(meta['line_names'])) else f"Branch {bus1}-{bus2}"
                contingencies.append({
                    'type': 'line',
                    'element_name': line_name,
                    'element_idx': idx,
                    'bus1': bus1,
                    'bus2': bus2,
                    'description': f"Outage of {line_name} (Bus {bus1} to Bus {bus2})",
                })

    # 2. Generator Contingencies
    if contingency_type in ("generator", "all"):
        for idx, gen in enumerate(mpc['gen']):
            gen_bus = int(gen[0])
            gen_name = meta['gen_names'][idx] if ('gen_names' in meta and idx < len(meta['gen_names'])) else f"Gen at Bus {gen_bus}"
            # Check if slack generator exclusion applies
            is_slack = (idx == 0) or (gen_bus == 1)
            if is_slack and meta.get('exclude_slack', True):
                continue
            if gen[7] > 0:  # Enabled generator
                contingencies.append({
                    'type': 'generator',
                    'element_name': gen_name,
                    'element_idx': idx,
                    'bus': gen_bus,
                    'description': f"Outage of {gen_name}",
                })

    if not contingencies:
        raise RuntimeError("No valid contingencies generated from network state.")

    return contingencies
