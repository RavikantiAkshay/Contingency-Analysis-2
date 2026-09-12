#!/usr/bin/env python3
"""
EEQ401 — Multi-Dataset Generation Campaign (100k, 200k, 300k Contingency Cases)

Generates 3 distinct large-scale datasets, separated into dedicated directories:
  data/tabular/dataset_100k/ (5,000 scenarios  x 20 lines = 100,000 cases)
  data/tabular/dataset_200k/ (10,000 scenarios x 20 lines = 200,000 cases)
  data/tabular/dataset_300k/ (15,000 scenarios x 20 lines = 300,000 cases)

Grand Total: 30,000 unique scenarios (600,000 line contingency cases).

Guarantees:
  - Strict zero-repetition: No scenario load pattern is ever repeated across any set.
  - Conforms to schema specification (63 columns).
  - Parallel multi-process execution with live tqdm progress bars.
  - Validates integrity and saves scenario pool JSON + summary metrics.
"""

import os
import sys
import time
import json
import copy
import argparse
import warnings
import multiprocessing as mp
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Suppress PyPOWER solver rank / divide-by-zero warnings during non-convergent outages
warnings.filterwarnings('ignore')

from config.config import get_config
from src.network.load_network import load_network
from src.network.validate_network import validate_network
from src.contingency.generate_contingencies import generate_contingencies
from src.contingency.random_load_generator import generate_random_load_pattern
from src.dataset.validate_dataset import validate_dataset
from src.dataset.scenario_pool import save_scenario_pool
from pypower.api import runpf, ppoption

EXPECTED_COLUMNS = (
    ['case_id', 'contingency_type', 'contingency_element', 'convergence'] +
    [f'V_bus{i+1}' for i in range(14)] +
    [f'angle_bus{i+1}' for i in range(14)] +
    [f'line_loading_{i+1}' for i in range(20)] +
    [f'gen_P_{i+1}' for i in range(5)] +
    ['min_bus_voltage', 'max_bus_voltage', 'max_line_loading', 'max_voltage_deviation', 'total_load'] +
    ['label']
)

NORM_AMPS = [
    800.0, 400.0, 300.0, 350.0, 300.0, 300.0, 300.0, 300.0,
    200.0, 250.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0,
    150.0, 100.0, 100.0, 100.0
]
SQRT3_144 = np.sqrt(3) * 144.0


def _worker_validate_pattern(item: Tuple[Dict[int, float], Dict[str, Any]]) -> Tuple[Dict[int, float], bool]:
    """
    Validates candidate base loading pattern with power flow feasibility check.
    """
    pattern, base_mpc = item
    opt = ppoption(VERBOSE=0, OUT_ALL=0)
    tmp = copy.deepcopy(base_mpc)
    bus = tmp['bus']
    for b_id, s in pattern.items():
        mask = (bus[:, 0] == int(b_id))
        bus[mask, 2] *= float(s)
        bus[mask, 3] *= float(s)

    res, succ = runpf(tmp, opt)
    if not succ or res is None:
        return pattern, False

    v_mags = res['bus'][:, 7]
    min_v = float(np.min(v_mags))
    max_v = float(np.max(v_mags))
    if min_v < 0.90 or max_v > 1.15:
        return pattern, False

    return pattern, True


def _worker_simulate_scenarios(task_chunk: List[Tuple[int, Dict[int, float], str, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]]) -> List[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Worker process function that simulates line contingencies across a batch of scenarios.
    """
    opt = ppoption(VERBOSE=0, OUT_ALL=0)
    chunk_results = []

    for sc_idx, pattern, prefix, base_mpc, contingencies, sec_cfg in task_chunk:
        scenario_id = f"{prefix}_sc{sc_idx+1:05d}"
        rows = []

        # Pre-scale base MPC bus loads for this scenario
        scaled_mpc = copy.deepcopy(base_mpc)
        bus = scaled_mpc['bus']
        for b_id, scale in pattern.items():
            mask = (bus[:, 0] == int(b_id))
            bus[mask, 2] *= float(scale)
            bus[mask, 3] *= float(scale)

        p_total_mw = float(np.sum(bus[:, 2]))

        # Security thresholds
        v_s_min, v_s_max = sec_cfg['v_safe_min'], sec_cfg['v_safe_max']
        v_a_min, v_a_max = sec_cfg['v_alert_min'], sec_cfg['v_alert_max']
        l_s_max, l_a_max = sec_cfg['line_safe_max'], sec_cfg['line_alert_max']

        # Sweep all 20 line outages
        for c in contingencies:
            b_idx = c['element_idx']
            bus_pair = f"{c['bus1']}-{c['bus2']}"
            cont_case_id = f"{scenario_id}_line_{bus_pair}"

            target_mpc = {
                'bus': bus.copy(),
                'branch': scaled_mpc['branch'].copy(),
                'gen': scaled_mpc['gen'].copy(),
                'baseMVA': scaled_mpc['baseMVA'],
                'version': scaled_mpc['version']
            }
            target_mpc['branch'][b_idx, 10] = 0  # Outage line

            results, success = runpf(target_mpc, opt)
            converged = bool(success)

            row = {
                'case_id': cont_case_id,
                'contingency_type': 'line',
                'contingency_element': bus_pair,
                'convergence': converged,
            }

            if converged and results is not None:
                r_bus = results['bus']
                r_branch = results['branch']
                r_gen = results['gen']

                v_mag = r_bus[:, 7]   # VM
                v_ang = r_bus[:, 8]   # VA
                gen_p = r_gen[:, 1]   # PG

                p_flow = r_branch[:, 13]  # PF
                q_flow = r_branch[:, 14]  # QF
                s_mva = np.sqrt(p_flow**2 + q_flow**2)
                cur_amps = s_mva * 1000.0 / SQRT3_144
                loading_pct = (cur_amps / NORM_AMPS) * 100.0
                loading_pct[r_branch[:, 10] == 0] = 0.0

                for b in range(14):
                    row[f'V_bus{b+1}'] = float(v_mag[b])
                    row[f'angle_bus{b+1}'] = float(v_ang[b])
                for l in range(20):
                    row[f'line_loading_{l+1}'] = float(loading_pct[l])
                for g in range(5):
                    row[f'gen_P_{g+1}'] = float(gen_p[g])

                min_v = float(np.min(v_mag))
                max_v = float(np.max(v_mag))
                max_load = float(np.max(loading_pct))
                row['min_bus_voltage'] = min_v
                row['max_bus_voltage'] = max_v
                row['max_line_loading'] = max_load
                row['max_voltage_deviation'] = float(np.max(np.abs(v_mag - 1.0)))
                row['total_load'] = p_total_mw

                # Classify security status
                if min_v < v_a_min or max_v > v_a_max or max_load > l_a_max:
                    label = 'Critical'
                elif min_v < v_s_min or max_v > v_s_max or max_load > l_s_max:
                    label = 'Alert'
                else:
                    label = 'Safe'
            else:
                for b in range(14):
                    row[f'V_bus{b+1}'] = np.nan
                    row[f'angle_bus{b+1}'] = np.nan
                for l in range(20):
                    row[f'line_loading_{l+1}'] = np.nan
                for g in range(5):
                    row[f'gen_P_{g+1}'] = np.nan
                row['min_bus_voltage'] = np.nan
                row['max_bus_voltage'] = np.nan
                row['max_line_loading'] = np.nan
                row['max_voltage_deviation'] = np.nan
                row['total_load'] = np.nan
                label = 'Critical'

            row['label'] = label
            rows.append(row)

        pool_entry = {
            'scenario_id': scenario_id,
            'load_pattern': {str(k): float(v) for k, v in pattern.items()},
            'total_load_mw': round(p_total_mw, 2),
            'contingencies': contingencies
        }
        chunk_results.append((rows, pool_entry))

    return chunk_results


def generate_unique_patterns_parallel(
    num_patterns: int,
    base_mpc: Dict[str, Any],
    rng: np.random.Generator,
    existing_patterns: set,
    pool: mp.Pool,
    min_scale: float = 0.80,
    max_scale: float = 1.20,
    desc: str = "Sampling base scenarios"
) -> List[Dict[int, float]]:
    """
    Generates unique, permissible pre-contingency load patterns in parallel with live tqdm progress.
    """
    accepted_patterns = []

    with tqdm(total=num_patterns, desc=desc, unit="scen", dynamic_ncols=True) as pbar:
        while len(accepted_patterns) < num_patterns:
            needed = num_patterns - len(accepted_patterns)
            batch_size = max(100, min(needed * 2, 2000))
            candidates = []

            while len(candidates) < batch_size:
                pattern, _ = generate_random_load_pattern(base_mpc, min_scale, max_scale, rng)
                pkey = frozenset((b, round(s, 2)) for b, s in pattern.items())
                if pkey in existing_patterns:
                    continue
                existing_patterns.add(pkey)
                candidates.append((pattern, base_mpc))

            results = pool.map(_worker_validate_pattern, candidates)
            for pattern, is_valid in results:
                if is_valid and len(accepted_patterns) < num_patterns:
                    accepted_patterns.append(pattern)
                    pbar.update(1)
                    if len(accepted_patterns) >= num_patterns:
                        break

    return accepted_patterns


def generate_large_scale_datasets(
    datasets_spec: List[Dict[str, Any]] = None,
    base_seed: int = 1000,
    base_output_dir: str = os.path.join('data', 'tabular'),
    batch_chunk_size: int = 50,
    num_workers: int = None
):
    if datasets_spec is None:
        datasets_spec = [
            {'name': 'dataset_100k', 'num_scenarios': 5000},
            {'name': 'dataset_200k', 'num_scenarios': 10000},
            {'name': 'dataset_300k', 'num_scenarios': 15000},
        ]

    start_total_time = time.time()
    cfg = get_config()
    base_mpc, meta = load_network()

    is_valid, issues = validate_network(meta, base_mpc)
    if not is_valid:
        raise RuntimeError(f"Network validation failed: {issues}")

    contingencies = generate_contingencies(meta, base_mpc, contingency_type="line")
    n_lines = len(contingencies)

    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, 12)

    total_scenarios_all = sum(d['num_scenarios'] for d in datasets_spec)
    total_cases_all = total_scenarios_all * n_lines

    print("=" * 80)
    print(" EEQ401: LARGE-SCALE MULTI-DATASET CAMPAIGN GENERATOR")
    print("=" * 80)
    print(f"  Datasets to Generate   : {len(datasets_spec)}")
    for d in datasets_spec:
        print(f"    - {d['name']:15s}: {d['num_scenarios']:,} scenarios x {n_lines} lines = {d['num_scenarios'] * n_lines:,} cases")
    print(f"  Total Scenarios        : {total_scenarios_all:,}")
    print(f"  Grand Total Cases      : {total_cases_all:,}")
    print(f"  Parallel CPU Workers   : {num_workers}")
    print(f"  Base Output Directory  : {base_output_dir}")
    print("=" * 80)

    seen_patterns = set()
    rng = np.random.default_rng(base_seed)
    all_set_summaries = []

    # Initialize persistent process pool across all sets
    with mp.Pool(processes=num_workers) as pool:
        for set_idx, spec in enumerate(datasets_spec, start=1):
            set_name = spec['name']
            num_scenarios = spec['num_scenarios']
            target_cases = num_scenarios * n_lines
            set_folder = os.path.join(base_output_dir, set_name)
            os.makedirs(set_folder, exist_ok=True)
            set_start_time = time.time()

            print(f"\n" + "#" * 80)
            print(f"  GENERATING {set_name.upper()} ({set_idx}/{len(datasets_spec)})")
            print(f"  Target Destination : {set_folder}")
            print(f"  Target Cases       : {target_cases:,} ({num_scenarios:,} scenarios x {n_lines} lines)")
            print("#" * 80)

            # 1. Generate unique patterns with live progress bar
            t_pat0 = time.time()
            patterns = generate_unique_patterns_parallel(
                num_patterns=num_scenarios,
                base_mpc=base_mpc,
                rng=rng,
                existing_patterns=seen_patterns,
                pool=pool,
                desc=f"  [1/4] Sampling base scenarios"
            )
            print(f"  [OK] Sampled {len(patterns):,} unique patterns in {time.time() - t_pat0:.1f}s (Global registry size: {len(seen_patterns):,}).")

            # 2. Prepare parallel tasks in chunks
            tasks = [
                (i, patterns[i], set_name, base_mpc, contingencies, cfg['security'])
                for i in range(num_scenarios)
            ]
            task_chunks = [
                tasks[i:i + batch_chunk_size]
                for i in range(0, len(tasks), batch_chunk_size)
            ]

            all_rows = []
            all_pool = []

            # 3. Simulate all line outages with live progress bar
            with tqdm(total=num_scenarios, desc=f"  [2/4] Simulating outages ({target_cases:,} cases)", unit="scen", dynamic_ncols=True) as pbar:
                for chunk_result in pool.imap(_worker_simulate_scenarios, task_chunks):
                    for rows, pool_entry in chunk_result:
                        all_rows.extend(rows)
                        all_pool.append(pool_entry)
                    pbar.update(len(chunk_result))

            # 4. Assemble and write CSV
            print(f"  [3/4] Assembling DataFrame and saving CSV...")
            df = pd.DataFrame(all_rows)
            df = df[EXPECTED_COLUMNS]
            csv_filename = f"{set_name}.csv"
            csv_path = os.path.join(set_folder, csv_filename)
            df.to_csv(csv_path, index=False)
            print(f"  [OK] Successfully wrote {len(df):,} rows x {len(df.columns)} columns to '{csv_path}'.")

            # 5. Save Scenario Pool JSON
            pool_filename = os.path.join(set_folder, f"{set_name}_pool.json")
            save_scenario_pool(all_pool, pool_filename)
            print(f"  [OK] Saved scenario pool to '{pool_filename}' ({len(all_pool):,} scenarios).")

            # 6. Validate integrity
            print(f"  [4/4] Validating dataset integrity...")
            is_valid, report = validate_dataset(df, meta, cfg)
            if not is_valid:
                raise RuntimeError(f"Validation failed for {set_name}: {report['errors']}")
            print(f"  [OK] Integrity PASSED: 63 columns, {report['total_rows']:,} rows, 0 duplicate IDs.")

            label_counts = df['label'].value_counts().to_dict()
            set_elapsed = time.time() - set_start_time

            summary = {
                'set_name': set_name,
                'folder': set_folder,
                'dataset_csv': csv_path,
                'scenario_pool_json': pool_filename,
                'num_scenarios': num_scenarios,
                'num_contingency_cases': target_cases,
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'converged_cases': report['converged_cases'],
                'non_converged_cases': report['non_converged_cases'],
                'label_distribution': {
                    lbl: label_counts.get(lbl, 0) for lbl in ['Safe', 'Alert', 'Critical']
                },
                'elapsed_seconds': round(set_elapsed, 2)
            }

            with open(os.path.join(set_folder, "summary.json"), 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)

            all_set_summaries.append(summary)
            print(f"  ✓ {set_name} completed in {set_elapsed:.1f}s ({set_elapsed/60.0:.2f} mins).")

    # Global manifest
    total_elapsed = time.time() - start_total_time
    manifest = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'num_sets': len(datasets_spec),
        'total_scenarios': total_scenarios_all,
        'total_cases': total_cases_all,
        'global_unique_scenarios': len(seen_patterns),
        'total_execution_time_seconds': round(total_elapsed, 2),
        'total_execution_time_minutes': round(total_elapsed / 60.0, 2),
        'sets': all_set_summaries
    }

    manifest_path = os.path.join(base_output_dir, "sets_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print(" ALL DATASETS GENERATED SUCCESSFULLY")
    print("=" * 80)
    print(f"  Total Scenarios        : {total_scenarios_all:,}")
    print(f"  Grand Total Cases      : {total_cases_all:,}")
    print(f"  Total Execution Time   : {total_elapsed:.1f}s ({total_elapsed/60.0:.2f} mins)")
    print(f"  Global Manifest Saved  : {manifest_path}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate large-scale power flow contingency datasets.")
    parser.add_argument('--workers', type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument('--output_dir', type=str, default=os.path.join('data', 'tabular'), help="Output base directory")
    parser.add_argument('--test', action='store_true', help="Run small test with 5, 10, 15 scenarios")
    args = parser.parse_args()

    spec = None
    if args.test:
        spec = [
            {'name': 'dataset_100k', 'num_scenarios': 5},
            {'name': 'dataset_200k', 'num_scenarios': 10},
            {'name': 'dataset_300k', 'num_scenarios': 15},
        ]

    generate_large_scale_datasets(
        datasets_spec=spec,
        base_output_dir=args.output_dir,
        num_workers=args.workers
    )
