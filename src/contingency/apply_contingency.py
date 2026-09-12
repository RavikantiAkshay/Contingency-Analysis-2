"""
Apply N-1 contingency to PyPOWER MPC dictionary.
"""
import copy
from typing import Dict, Any


def apply_contingency(mpc: Dict[str, Any], contingency: Dict[str, Any]) -> Dict[str, Any]:
    """
    Modifies PyPOWER MPC dictionary by disabling a specified branch or generator.

    Args:
        mpc (dict): Original PyPOWER case dictionary.
        contingency (dict): Contingency descriptor containing 'type' and 'element_idx'.

    Returns:
        modified_mpc (dict): Deep-copied PyPOWER case dictionary with outaged element disabled.
    """
    modified_mpc = copy.deepcopy(mpc)
    ctype = contingency.get('type')
    idx = contingency.get('element_idx')

    if ctype == 'line':
        if idx is None or idx < 0 or idx >= len(modified_mpc['branch']):
            raise ValueError(f"Branch index {idx} out of bounds for matrix of size {len(modified_mpc['branch'])}.")
        modified_mpc['branch'][idx, 10] = 0  # Status 0 = offline
    elif ctype == 'generator':
        if idx is None or idx < 0 or idx >= len(modified_mpc['gen']):
            raise ValueError(f"Generator index {idx} out of bounds for matrix of size {len(modified_mpc['gen'])}.")
        modified_mpc['gen'][idx, 7] = 0  # Status 0 = offline
    else:
        raise ValueError(f"Unknown contingency type: {ctype}")

    return modified_mpc
