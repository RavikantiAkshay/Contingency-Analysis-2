#!/usr/bin/env python3
"""
EEQ401 — Base Case Load Flow Analysis (Normal, Light, Heavy)
Executes baseline power flow analysis on the IEEE 14-bus transmission network
for multiple operating conditions without contingencies.
"""

import sys
from config.config import get_config
from src.network.load_network import load_network
from src.network.validate_network import validate_network
from src.simulation.run_load_flow import run_load_flow
from src.results.save_results import save_results
from src.labeling.classify_security import classify_security


def main():
    print("========================================================================")
    print(" EEQ401: Base Case Load Flow Analysis (Normal, Light, Heavy)")
    print("========================================================================")

    # 1. Load Central Configuration
    cfg = get_config()
    print(f"\n[1/4] Loading PyPOWER Network ('case14')...")
    mpc, meta = load_network()
    is_valid, issues = validate_network(meta, mpc)
    if not is_valid:
        print(f"FAILED: Network validation errors detected: {issues}")
        sys.exit(1)
    print(f"SUCCESS: PyPOWER Network validated ({meta['n_bus']} Buses, {meta['n_line']} Branches, {meta['n_gen']} Generators).")

    # 2. Iterate over Load Conditions
    load_scales = cfg['load_scales']
    print(f"\n[2/4] Executing AC Newton-Raphson Power Flow for {len(load_scales)} conditions...")

    all_results = {}

    for condition_name, scale_factor in load_scales.items():
        print(f"\n---> CONDITION: {condition_name.upper()} (Load Scale: {scale_factor}x)")

        # Execute Power Flow Solution
        converged, results = run_load_flow(mpc=mpc, solver_type="ac_nr", load_scale=scale_factor)
        if not converged:
            print(f"FAILED: Base case power flow did not converge for {condition_name} condition!")
            continue
        print("     SUCCESS: AC Newton-Raphson converged successfully.")

        # Classify Security State
        label, violations = classify_security(results, converged, cfg)
        print(f"     SECURITY CLASS: [{label.upper()}] with {len(violations)} violations.")

        # Save Results to Disk (Self-describing filename: case14_load1.00x_normal)
        case_name = f"case14_load{scale_factor:.2f}x_{condition_name}"
        saved_paths = save_results(
            results, label, violations, cfg,
            case_name=case_name,
            output_dir=cfg['paths']['tabular_base_cases']
        )
        for key, path in saved_paths.items():
            print(f"     Saved {key}: {path}")

        all_results[condition_name] = (results, label, violations)

    print("\n========================================================================")
    print(" SUMMARY REPORT OF BASE CASES")
    print("========================================================================")
    for condition_name, data in all_results.items():
        results, label, violations = data
        print(f"\n--- {condition_name.upper()} LOAD BASE CASE ---")
        print(f"  Security Status       : [{label.upper()}]")
        print(f"  Total System Demand   : {results['total_load']/1000.0:.2f} MW  |  {results['total_load_mvar']/1000.0:.2f} MVAr")
        print(f"  Total Generation      : {results['total_gen_mw']:.2f} MW  |  {results['total_gen_mvar']:.2f} MVAr")
        print(f"  Minimum Bus Voltage   : {results['min_voltage']:.4f} p.u.")
        print(f"  Maximum Bus Voltage   : {results['max_voltage']:.4f} p.u.")
        print(f"  Maximum Line Loading  : {results['max_line_loading']:.2f} %")
        if violations:
            print("  Top Violations:")
            for v in violations[:3]:
                print(f"    - [{v['severity']}] {v['type']} on {v['element']}: value={v['value']}")
            if len(violations) > 3:
                print(f"    ... and {len(violations) - 3} more.")
    print("\n========================================================================")


if __name__ == '__main__':
    main()
