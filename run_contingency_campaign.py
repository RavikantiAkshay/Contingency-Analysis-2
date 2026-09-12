#!/usr/bin/env python3
"""
EEQ401 — Automated N-1 Line Contingency Simulation Campaign & Master Dataset Builder
Sweeps load scales from 0.70x to 1.30x in steps of 0.01 (61 load levels) across all 20 line contingencies + base cases.
Builds, validates, and exports the master_dataset.csv (1,281 cases x 63 columns).
"""

import sys
import time
from config.config import get_config
from src.network.load_network import load_network
from src.network.validate_network import validate_network
from src.simulation.run_simulation_campaign import run_simulation_campaign
from src.dataset.build_dataset import build_dataset
from src.dataset.validate_dataset import validate_dataset


def main():
    start_time = time.time()
    print("========================================================================")
    print(" EEQ401: Automated Line Contingency Simulation Campaign")
    print("========================================================================")

    # 1. Load Config & Network
    cfg = get_config()
    mpc, meta = load_network()

    is_valid, issues = validate_network(meta, mpc)
    if not is_valid:
        print(f"FAILED: Network validation errors: {issues}")
        sys.exit(1)

    print("\nNetwork Loaded Successfully:")
    print(f"  Buses       : {meta['n_bus']}")
    print(f"  Branches    : {meta['n_line']} transmission lines/transformers")
    print(f"  Generators  : {meta['n_gen']}")
    print(f"  Sweep Range : {cfg['campaign_sweep']['min_scale']}x to {cfg['campaign_sweep']['max_scale']}x ({cfg['campaign_sweep']['num_base_scenarios']} randomized base scenarios)")

    # 2. Run Automated Simulation Campaign (Line Contingencies Only)
    all_results, sim_log = run_simulation_campaign(
        cfg=cfg,
        meta=meta,
        mpc=mpc,
        contingency_type="line",
        num_scenarios=cfg['campaign_sweep']['num_base_scenarios']
    )

    # 3. Build Consolidated Master Dataset CSV
    print("\n------------------------------------------------------------------------")
    print(" Assembling Master Dataset CSV...")
    print("------------------------------------------------------------------------")
    dataset = build_dataset(all_results, cfg)

    # 4. Validate Dataset Integrity
    print("\n------------------------------------------------------------------------")
    print(" Validating Dataset Integrity (LLD §3.12)...")
    print("------------------------------------------------------------------------")
    is_valid, report = validate_dataset(dataset, meta, cfg)

    elapsed_time = time.time() - start_time

    print("\n========================================================================")
    print(" CAMPAIGN EXECUTION & DATASET SUMMARY")
    print("========================================================================")
    print(f"  Total Execution Time   : {elapsed_time:.2f} seconds")
    print(f"  Total Dataset Rows     : {report['total_rows']}")
    print(f"  Total Dataset Columns  : {report['total_columns']} (Refined Schema Spec: 63)")
    print(f"  Converged Cases        : {report['converged_cases']}")
    print(f"  Non-Converged Cases    : {report['non_converged_cases']}")
    print(f"  Dataset Integrity      : {'PASSED' if is_valid else 'FAILED'}")

    # Security Label Distribution
    label_counts = dataset['label'].value_counts().to_dict()
    print("\n  Security Class Label Distribution:")
    for lbl in ['Safe', 'Alert', 'Critical']:
        cnt = label_counts.get(lbl, 0)
        pct = (cnt / len(dataset)) * 100.0 if len(dataset) > 0 else 0
        print(f"    - {lbl:8s}: {cnt:5d} cases ({pct:5.1f}%)")

    if not is_valid:
        print(f"\n  VALIDATION ERRORS DETECTED:")
        for err in report['errors']:
            print(f"    - {err}")
        sys.exit(1)

    print("\nMaster dataset successfully created and saved at 'data/processed/master_dataset.csv'!")


if __name__ == '__main__':
    main()
