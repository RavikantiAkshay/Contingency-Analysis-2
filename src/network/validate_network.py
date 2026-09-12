"""
Network Validator — Dynamically validates PyPOWER case dictionary and metadata integrity
for any transmission network topology (IEEE 14, IEEE 30, IEEE 57, IEEE 118, etc.).
"""

from typing import Tuple, List, Dict, Any, Optional


def validate_network(meta: Dict[str, Any], mpc: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Verifies the integrity of the loaded PyPOWER network.

    Args:
        meta (dict): Network metadata dictionary returned by load_network().
        mpc (dict): PyPOWER case dictionary.

    Returns:
        is_valid (bool): True if all checks pass, False otherwise.
        issues (list[str]): List of error/warning strings explaining any validation failures.
    """
    issues = []

    if mpc is None or not isinstance(mpc, dict):
        issues.append("Invalid or missing PyPOWER case dict (mpc).")
        return False, issues

    bus = mpc.get('bus')
    branch = mpc.get('branch')
    gen = mpc.get('gen')

    if bus is None or len(bus) == 0:
        issues.append("PyPOWER case bus matrix is empty or missing.")
    elif len(bus) != meta.get('n_bus'):
        issues.append(f"Bus count mismatch: metadata has {meta.get('n_bus')}, mpc matrix has {len(bus)}.")

    if branch is None or len(branch) == 0:
        issues.append("PyPOWER case branch matrix is empty or missing.")
    elif len(branch) != meta.get('n_line'):
        issues.append(f"Branch count mismatch: metadata has {meta.get('n_line')}, mpc matrix has {len(branch)}.")

    if gen is None or len(gen) == 0:
        issues.append("PyPOWER case gen matrix is empty or missing.")
    elif len(gen) != meta.get('n_gen'):
        issues.append(f"Generator count mismatch: metadata has {meta.get('n_gen')}, mpc matrix has {len(gen)}.")

    # Check for any inoperable (disabled) branches in the base case
    if branch is not None:
        disabled_branches = [i for i, br in enumerate(branch) if br[10] <= 0]
        if disabled_branches:
            for idx in disabled_branches:
                issues.append(f"Branch {idx} is disabled in base circuit.")

    is_valid = len(issues) == 0
    return is_valid, issues
