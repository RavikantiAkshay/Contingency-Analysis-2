#!/usr/bin/env python3
"""
EEQ401 — Interactive / Single Line Contingency Analysis
Allows selecting a specific transmission line (by bus1 and bus2, branch name, or index),
outaging that line, executing AC Newton-Raphson power flow, and classifying security status.
"""

import sys
import argparse
from config.config import get_config
from src.network.load_network import load_network
from src.network.validate_network import validate_network
from src.contingency.generate_contingencies import generate_contingencies
from src.contingency.apply_contingency import apply_contingency
from src.simulation.run_load_flow import run_load_flow
from src.labeling.classify_security import classify_security
from src.results.save_results import save_results


def find_branch_contingency(contingencies, bus1=None, bus2=None, branch_idx=None, branch_name=None):
    """Matches a line contingency from the enumerated list based on user criteria."""
    for c in contingencies:
        if c['type'] != 'line':
            continue
        if branch_idx is not None and c['element_idx'] == branch_idx:
            return c
        if branch_name and (c['element_name'].lower() == branch_name.lower() or f"branch {c['bus1']}-{c['bus2']}".lower() == branch_name.lower()):
            return c
        if bus1 is not None and bus2 is not None:
            if (c['bus1'] == bus1 and c['bus2'] == bus2) or (c['bus1'] == bus2 and c['bus2'] == bus1):
                return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Run single line contingency analysis on IEEE 14-bus network.")
    parser.add_argument("--bus1", type=int, help="From-bus ID (e.g. 1)")
    parser.add_argument("--bus2", type=int, help="To-bus ID (e.g. 2)")
    parser.add_argument("--branch_idx", type=int, help="Branch matrix index (0 to 19)")
    parser.add_argument("--branch_name", type=str, help="Branch name (e.g. 'Branch 1-2')")
    parser.add_argument("--load_scale", type=float, default=1.0, help="Load scaling factor (default: 1.0)")
    args = parser.parse_args()

    print("========================================================================")
    print(" EEQ401: Single Line Contingency Analysis (IEEE 14-bus)")
    print("========================================================================")

    # 1. Load Config and Network
    cfg = get_config()
    mpc, meta = load_network()
    is_valid, issues = validate_network(meta, mpc)
    if not is_valid:
        print(f"FAILED: Network validation errors: {issues}")
        sys.exit(1)

    # 2. Enumerate Line Contingencies
    line_contingencies = generate_contingencies(meta, mpc, contingency_type="line")

    # 3. Resolve Target Contingency
    target_contingency = find_branch_contingency(
        line_contingencies,
        bus1=args.bus1,
        bus2=args.bus2,
        branch_idx=args.branch_idx,
        branch_name=args.branch_name
    )

    # Interactive prompt if no arguments specified or match not found via args
    if target_contingency is None:
        print("\nAvailable Line Contingencies:")
        for idx, c in enumerate(line_contingencies):
            print(f"  [{idx:2d}] {c['element_name']} (Bus {c['bus1']} <-> Bus {c['bus2']})")

        print("\nSelect a line contingency to outage:")
        user_input = input("Enter line index (0-19) or Bus pair (e.g. '1 2'): ").strip()
        
        parts = user_input.split()
        if len(parts) == 1 and parts[0].isdigit():
            b_idx = int(parts[0])
            target_contingency = find_branch_contingency(line_contingencies, branch_idx=b_idx)
        elif len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            b1, b2 = int(parts[0]), int(parts[1])
            target_contingency = find_branch_contingency(line_contingencies, bus1=b1, bus2=b2)

    if target_contingency is None:
        print("\nERROR: Invalid branch selection. Exiting.")
        sys.exit(1)

    print(f"\n---> SELECTED CONTINGENCY: {target_contingency['description']}")
    print(f"     Target Element        : {target_contingency['element_name']} (Index {target_contingency['element_idx']})")
    print(f"     Load Scale            : {args.load_scale}x")

    # 4. Apply Contingency
    contingency_mpc = apply_contingency(mpc, target_contingency)

    # 5. Execute AC Newton-Raphson Load Flow
    converged, results = run_load_flow(mpc=contingency_mpc, solver_type="ac_nr", load_scale=args.load_scale)
    if not converged:
        print("\nFAILED: Post-contingency load flow DID NOT CONVERGE! System is voltage-unstable / collapsed.")
        label = "Critical"
        violations = [{'type': 'convergence_failure', 'severity': 'Critical', 'element': target_contingency['element_name'], 'value': 'Non-convergent'}]
    else:
        print("     SUCCESS: AC Newton-Raphson converged successfully.")
        label, violations = classify_security(results, converged, cfg)

    print(f"\n========================================================================")
    print(f" POST-CONTINGENCY SECURITY ASSESSMENT: [{label.upper()}]")
    print(f"========================================================================")
    if converged:
        print(f"  Total System Demand   : {results['total_load']/1000.0:.2f} MW  |  {results['total_load_mvar']/1000.0:.2f} MVAr")
        print(f"  Total Generation      : {results['total_gen_mw']:.2f} MW  |  {results['total_gen_mvar']:.2f} MVAr")
        print(f"  Minimum Bus Voltage   : {results['min_voltage']:.4f} p.u.")
        print(f"  Maximum Bus Voltage   : {results['max_voltage']:.4f} p.u.")
        print(f"  Maximum Line Loading  : {results['max_line_loading']:.2f} %")
    
    print(f"  Total Violations      : {len(violations)}")
    if violations:
        print("  Violation Details:")
        for v in violations:
            print(f"    - [{v['severity']}] {v['type']} on {v['element']}: value={v['value']}")

    # 6. Save Results
    case_name = f"contingency_line_{target_contingency['bus1']}_{target_contingency['bus2']}"
    if results is not None:
        saved_paths = save_results(results, label, violations, cfg, case_name=case_name)
        print("\nSaved Output Artifacts:")
        for k, p in saved_paths.items():
            print(f"  - {k}: {p}")


if __name__ == '__main__':
    main()
