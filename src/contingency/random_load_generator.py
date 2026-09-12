"""
EEQ401 — Randomized Targeted Load Generator & Base Case Feasibility Filter

Implements non-uniform, randomized load pattern generation across target buses:
1. Selects random target buses (2, 3, or more load buses).
2. Assigns independent load scaling factors in [min_scale, max_scale] (e.g. 0.80x to 1.20x).
3. Evaluates base case power flow convergence and security limits.
4. Discards invalid / non-convergent / impermissible base cases before contingency sweeps.

Reference: User Request & Design Specifications.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from src.simulation.run_load_flow import run_load_flow
from src.labeling.classify_security import classify_security


def generate_random_load_pattern(
    mpc: Dict[str, Any],
    min_scale: float = 0.80,
    max_scale: float = 1.20,
    rng: Optional[np.random.Generator] = None
) -> Tuple[Dict[int, float], str]:
    """
    Generates a randomized load pattern across target buses.

    Args:
        mpc (dict): PyPOWER case dict.
        min_scale (float): Minimum scaling factor (default: 0.80).
        max_scale (float): Maximum scaling factor (default: 1.20).
        rng (np.random.Generator, optional): Random number generator.

    Returns:
        pattern (dict): Mapping bus_id -> scale_factor.
        description (str): Human-readable pattern summary.
    """
    if rng is None:
        rng = np.random.default_rng()

    bus_table = mpc['bus']
    load_bus_ids = [int(b[0]) for b in bus_table if b[2] > 0]

    if not load_bus_ids:
        return {}, "No load buses"

    # Step 1: Randomly select number of target buses (from 2 up to all load buses)
    k_targets = int(rng.integers(low=2, high=len(load_bus_ids) + 1))

    # Step 2: Randomly select k_targets from available load buses
    target_buses = rng.choice(load_bus_ids, size=k_targets, replace=False)

    # Step 3: Sample random load scale for each selected target bus
    pattern = {}
    details = []
    for bus_id in target_buses:
        scale = round(float(rng.uniform(min_scale, max_scale)), 3)
        pattern[int(bus_id)] = scale
        details.append(f"Bus{bus_id}:{scale:.2f}x")

    desc = f"Random load on {len(target_buses)} buses ({', '.join(details)})"
    return pattern, desc


def find_valid_base_case(
    mpc: Dict[str, Any],
    cfg: Dict[str, Any],
    max_attempts: int = 200,
    min_scale: float = 0.80,
    max_scale: float = 1.20,
    rng: Optional[np.random.Generator] = None,
    existing_patterns: Optional[set] = None
) -> Tuple[Dict[int, float], Dict[str, Any], str]:
    """
    Repeatedly generates random load patterns until a UNIQUE, VALID, CONVERGENT, and PERMISSIBLE
    base case load flow solution is found.

    Returns:
        valid_pattern (dict): Per-bus load scale mapping.
        base_results (dict): Electrical results dictionary for the valid base case.
        pattern_desc (str): Description of the load pattern.
    """
    if rng is None:
        rng = np.random.default_rng()

    if existing_patterns is None:
        existing_patterns = set()

    for attempt in range(1, max_attempts + 1):
        pattern, pattern_desc = generate_random_load_pattern(mpc, min_scale, max_scale, rng)

        pattern_key = frozenset((b, round(s, 2)) for b, s in pattern.items())
        if pattern_key in existing_patterns:
            continue

        # Solve AC power flow for candidate base case (no contingency)
        converged, results = run_load_flow(mpc, load_scale=pattern)

        if not converged or results is None:
            continue

        # Classify security status of candidate base case
        label, violations = classify_security(results, converged, cfg)

        if label == 'Critical':
            # Discard base case if loading itself causes severe failure/collapse
            continue

        # Valid, unique, permissible base case found
        existing_patterns.add(pattern_key)
        return pattern, results, pattern_desc

    raise RuntimeError(f"Could not find a unique valid base case after {max_attempts} attempts.")
