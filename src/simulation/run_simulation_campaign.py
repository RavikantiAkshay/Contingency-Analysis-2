"""
Automated Simulation Campaign Orchestrator.
Executes systematic load flow simulations across randomized, targeted load patterns
and N-1 line contingencies.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from src.network.load_network import load_network
from src.network.validate_network import validate_network
from src.contingency.generate_contingencies import generate_contingencies
from src.contingency.apply_contingency import apply_contingency
from src.contingency.random_load_generator import generate_random_load_pattern
from src.simulation.run_load_flow import run_load_flow
from src.results.extract_results import extract_results
from src.labeling.classify_security import classify_security


def run_simulation_campaign(
    cfg: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
    mpc: Optional[Dict[str, Any]] = None,
    contingency_type: str = "line",
    num_scenarios: int = 100,
    min_scale: float = 0.80,
    max_scale: float = 1.20,
    seed: int = 42,
    existing_patterns: Optional[set] = None,
    scenario_prefix: str = "scenario",
    rng: Optional[np.random.Generator] = None,
    progress_step: int = 10,
    include_base_cases: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes automated simulation campaign over randomized, targeted base load scenarios and N-1 contingencies.

    Workflow per scenario:
    1. Randomly selects load targets (2, 3, or all load buses) and scaling factors in [min_scale, max_scale].
    2. Filters base case feasibility: verifies AC power flow convergence and confirms base case is permissible (not Critical).
       If base case fails/overloads, discards and resamples.
    3. Sweeps serial line contingencies (Line 1-2, 1-5, etc.) on the verified valid base case.

    Returns:
        all_results (list of dicts): List of structured result objects for each case.
        sim_log (dict): Execution log summary.
    """
    if mpc is None or meta is None:
        mpc, meta = load_network()

    is_valid, issues = validate_network(meta, mpc)
    if not is_valid:
        raise RuntimeError(f"Network validation failed: {issues}")

    if rng is None:
        rng = np.random.default_rng(seed)

    sweep_cfg = cfg.get('campaign_sweep', {})
    min_s = sweep_cfg.get('min_scale', min_scale)
    max_s = sweep_cfg.get('max_scale', max_scale)

    contingencies = generate_contingencies(meta, mpc, contingency_type=contingency_type)

    all_results = []
    cases_per_scen = (1 + len(contingencies)) if include_base_cases else len(contingencies)
    total_cases = num_scenarios * cases_per_scen
    converged_count = 0
    failed_count = 0
    discarded_base_cases = 0
    if existing_patterns is None:
        existing_patterns = set()

    print(f"\n[CAMPAIGN START] Executing {num_scenarios} randomized base scenarios x {cases_per_scen} cases = {total_cases} total simulations...")
    print(f"  Load scale range: {min_s:.2f}x to {max_s:.2f}x (Random target bus selection)")

    for sc_idx in range(num_scenarios):
        scenario_id = f"{scenario_prefix}_{sc_idx+1:03d}"

        # 1. Find a valid, unique, permissible base case load pattern
        attempt_count = 0
        while True:
            attempt_count += 1
            pattern, desc = generate_random_load_pattern(mpc, min_s, max_s, rng)

            pattern_key = frozenset((b, round(s, 2)) for b, s in pattern.items())
            if pattern_key in existing_patterns:
                continue

            # Solve base case AC power flow
            converged, raw_base_results = run_load_flow(mpc=mpc, solver_type="ac_nr", load_scale=pattern)
            if not converged or raw_base_results is None:
                discarded_base_cases += 1
                continue

            base_results = extract_results(raw_base_results, converged, meta=meta)
            base_label, base_violations = classify_security(base_results, converged, cfg)

            # If base case itself is Critical (exceeds allowed operating limits prior to contingency), discard!
            if base_label == 'Critical':
                discarded_base_cases += 1
                continue

            # Valid unique pattern found
            existing_patterns.add(pattern_key)
            break

        if include_base_cases:
            converged_count += 1
            base_case_id = f"{scenario_id}_base"

            all_results.append({
                'case_id': base_case_id,
                'scenario': scenario_id,
                'load_pattern': pattern,
                'load_description': desc,
                'contingency_type': 'none',
                'contingency_element': 'none',
                'contingency_name': 'Base Case (No Outage)',
                'convergence': True,
                'results': base_results,
                'label': base_label,
                'violations': base_violations,
            })

        # 2. Sweep Line Contingencies on this valid base load scenario
        for c in contingencies:
            elem_idx = c['element_idx']
            bus_pair = f"{c['bus1']}-{c['bus2']}"
            cont_case_id = f"{scenario_id}_{c['type']}_{bus_pair}"

            target_mpc = apply_contingency(mpc, c)
            converged, raw_results = run_load_flow(mpc=target_mpc, solver_type="ac_nr", load_scale=pattern)
            results = extract_results(raw_results, converged, meta=meta)
            label, violations = classify_security(results, converged, cfg)

            if converged:
                converged_count += 1
            else:
                failed_count += 1

            all_results.append({
                'case_id': cont_case_id,
                'scenario': scenario_id,
                'load_pattern': pattern,
                'load_description': desc,
                'contingency_type': c['type'],
                'contingency_element': bus_pair,
                'contingency_name': c['element_name'],
                'convergence': converged,
                'results': results,
                'label': label,
                'violations': violations,
            })

        if (sc_idx + 1) % progress_step == 0 or (sc_idx + 1) == num_scenarios:
            curr_cases = (sc_idx + 1) * (1 + len(contingencies))
            print(f"  Progress: Completed Scenario {sc_idx+1}/{num_scenarios} ({curr_cases}/{total_cases} cases | Discarded Base Patterns: {discarded_base_cases})")

    sim_log = {
        'total_cases': total_cases,
        'converged_cases': converged_count,
        'failed_cases': failed_count,
        'num_scenarios': num_scenarios,
        'discarded_base_cases': discarded_base_cases,
        'num_contingencies': len(contingencies),
        'existing_patterns': existing_patterns,
    }

    print(f"[CAMPAIGN COMPLETE] Finished {total_cases} cases across {num_scenarios} valid base scenarios. Converged: {converged_count}, Failed: {failed_count}.\n")
    return all_results, sim_log
