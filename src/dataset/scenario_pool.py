"""
EEQ401 — Scenario Pool Builder and Persistence Utilities for Dataset Generation.

Replays the deterministic random load pattern generator (seed=42)
used in run_simulation_campaign to reconstruct the exact per-bus load
scaling dictionary for base scenarios.

Provides utilities for saving, loading, and combining scenario pools
across master and expansion datasets (set1 through set5).
"""

import os
import json
import copy
from typing import Dict, Any, List, Optional
import numpy as np

from src.network.load_network import load_network
from src.contingency.generate_contingencies import generate_contingencies
from src.contingency.random_load_generator import generate_random_load_pattern
from src.simulation.run_load_flow import run_load_flow
from src.results.extract_results import extract_results
from src.labeling.classify_security import classify_security


def rebuild_scenario_pool(
    cfg: Dict[str, Any],
    num_scenarios: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Deterministically replays the campaign's random load generator to
    reconstruct the base-scenario load patterns.

    For each valid scenario, enumerates all 20 line contingencies and
    stores the scenario descriptor dict.

    Returns:
        pool (list of dicts): Each entry contains:
            - scenario_id (str): e.g. 'scenario_001'
            - load_pattern (dict): {bus_id: scale_factor, ...}
            - contingencies (list of dicts): The 20 N-1 line contingency descriptors.
    """
    mpc, meta = load_network()
    contingencies = generate_contingencies(meta, mpc, contingency_type="line")

    rng = np.random.default_rng(seed)
    sweep_cfg = cfg.get('campaign_sweep', {})
    min_s = sweep_cfg.get('min_scale', 0.80)
    max_s = sweep_cfg.get('max_scale', 1.20)

    existing_patterns = set()
    pool = []

    for sc_idx in range(num_scenarios):
        while True:
            pattern, desc = generate_random_load_pattern(mpc, min_s, max_s, rng)
            pattern_key = frozenset((b, round(s, 2)) for b, s in pattern.items())
            if pattern_key in existing_patterns:
                continue

            converged, raw = run_load_flow(mpc=mpc, solver_type="ac_nr", load_scale=pattern)
            if not converged or raw is None:
                continue

            base_results = extract_results(raw, converged, meta=meta)
            label, _ = classify_security(base_results, converged, cfg)
            if label == 'Critical':
                continue

            existing_patterns.add(pattern_key)
            break

        scenario_id = f"scenario_{sc_idx + 1:03d}"
        pool.append({
            'scenario_id': scenario_id,
            'load_pattern': pattern,
            'contingencies': contingencies,
        })

    return pool


def save_scenario_pool(pool: List[Dict[str, Any]], filepath: str) -> None:
    """
    Saves scenario pool to JSON file with serializable format.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    serializable_pool = []
    for sc in pool:
        entry = {
            'scenario_id': sc['scenario_id'],
            'load_pattern': {str(k): float(v) for k, v in sc['load_pattern'].items()},
            'contingencies': sc.get('contingencies', [])
        }
        serializable_pool.append(entry)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_pool, f, indent=2)


def load_scenario_pool(filepath: str) -> List[Dict[str, Any]]:
    """
    Loads scenario pool from JSON file, converting bus IDs back to integers,
    and normalizing contingency descriptors so element_idx is always present.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_pool = json.load(f)

    LINE_NAME_TO_IDX = {
        '1-2': 0, '1-5': 1, '2-3': 2, '2-4': 3, '2-5': 4,
        '3-4': 5, '4-5': 6, '4-7': 7, '4-9': 8, '5-6': 9,
        '6-11': 10, '6-12': 11, '6-13': 12, '7-8': 13, '7-9': 14,
        '9-10': 15, '9-14': 16, '10-11': 17, '12-13': 18, '13-14': 19
    }

    pool = []
    for sc in raw_pool:
        contingencies = []
        for c in sc.get('contingencies', []):
            c_copy = dict(c)
            if 'element_idx' not in c_copy:
                ename = c_copy.get('element_name') or c_copy.get('element')
                if ename in LINE_NAME_TO_IDX:
                    c_copy['element_idx'] = LINE_NAME_TO_IDX[ename]
                elif isinstance(ename, int):
                    c_copy['element_idx'] = ename
            contingencies.append(c_copy)

        entry = {
            'scenario_id': sc['scenario_id'],
            'load_pattern': {int(k): float(v) for k, v in sc['load_pattern'].items()},
            'contingencies': contingencies
        }
        pool.append(entry)

    return pool


def load_combined_pools(
    sources: List[str],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Load and combine scenario pools from multiple sources.
    """
    combined_pool = []
    tabular_dir = cfg.get('paths', {}).get('tabular_dir', os.path.join('data', 'tabular'))

    for src in sources:
        src_lower = src.lower()
        if src_lower == 'master':
            master_cache = os.path.join(tabular_dir, 'master', 'master_pool.json')
            if os.path.exists(master_cache):
                pool = load_scenario_pool(master_cache)
            else:
                pool = rebuild_scenario_pool(cfg, num_scenarios=100, seed=42)
                try:
                    save_scenario_pool(pool, master_cache)
                except Exception:
                    pass
            combined_pool.extend(pool)
        elif src_lower.startswith('set'):
            set_path = os.path.join(tabular_dir, src_lower, f"{src_lower}_pool.json")
            if not os.path.exists(set_path):
                raise FileNotFoundError(f"Scenario pool file not found for source '{src}': {set_path}")
            pool = load_scenario_pool(set_path)
            combined_pool.extend(pool)
        else:
            if os.path.exists(src):
                pool = load_scenario_pool(src)
                combined_pool.extend(pool)
            else:
                raise ValueError(f"Unknown data source: {src}")

    return combined_pool
