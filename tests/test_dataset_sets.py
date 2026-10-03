"""
Automated unit tests for multi-set dataset generation and cross-set uniqueness validation.
"""
import os
import tempfile
import numpy as np
import pandas as pd
from config.config import get_config
from src.network.load_network import load_network
from src.simulation.run_simulation_campaign import run_simulation_campaign
from src.dataset.build_dataset import build_dataset
from src.dataset.validate_dataset import validate_dataset
from src.dataset.scenario_pool import save_scenario_pool, load_scenario_pool


def test_multi_set_uniqueness_and_validation():
    cfg = get_config()
    mpc, meta = load_network()

    seen_patterns = set()

    # 1. Simulate Set 1 with 2 scenarios
    results_set1, log_set1 = run_simulation_campaign(
        cfg=cfg,
        meta=meta,
        mpc=mpc,
        contingency_type="line",
        num_scenarios=2,
        seed=101,
        existing_patterns=seen_patterns,
        scenario_prefix="set1_scenario"
    )

    assert len(results_set1) == 42
    assert len(seen_patterns) == 2

    # 2. Simulate Set 2 with 2 scenarios sharing the same seen_patterns registry
    results_set2, log_set2 = run_simulation_campaign(
        cfg=cfg,
        meta=meta,
        mpc=mpc,
        contingency_type="line",
        num_scenarios=2,
        seed=102,
        existing_patterns=seen_patterns,
        scenario_prefix="set2_scenario"
    )

    assert len(results_set2) == 42
    # Total unique patterns in the registry should now be 4
    assert len(seen_patterns) == 4

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build and validate Set 1
        df_set1 = build_dataset(results_set1, cfg, filename="set1_dataset.csv", output_dir=tmpdir)
        assert df_set1.shape == (42, 63)
        valid1, rep1 = validate_dataset(df_set1, meta, cfg)
        assert valid1, f"Set 1 validation errors: {rep1['errors']}"

        # Build and validate Set 2
        df_set2 = build_dataset(results_set2, cfg, filename="set2_dataset.csv", output_dir=tmpdir)
        assert df_set2.shape == (42, 63)
        valid2, rep2 = validate_dataset(df_set2, meta, cfg)
        assert valid2, f"Set 2 validation errors: {rep2['errors']}"

        # Check that case_ids across both sets have NO overlap
        set1_cases = set(df_set1['case_id'])
        set2_cases = set(df_set2['case_id'])
        assert set1_cases.isdisjoint(set2_cases), "Collision found in case_ids between set1 and set2!"

        # Test scenario pool save and load
        pool_data = [
            {'scenario_id': 'sc_001', 'load_pattern': {2: 1.1, 3: 0.9}, 'contingencies': []}
        ]
        pool_path = os.path.join(tmpdir, "test_pool.json")
        save_scenario_pool(pool_data, pool_path)
        loaded = load_scenario_pool(pool_path)
        assert len(loaded) == 1
        assert loaded[0]['load_pattern'] == {2: 1.1, 3: 0.9}
        assert isinstance(list(loaded[0]['load_pattern'].keys())[0], int)


def test_load_existing_pools_into_registry():
    from run_generate_dataset_sets import load_existing_pools_into_registry
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 2 subdirectories with pools
        sub1 = os.path.join(tmpdir, "set1")
        sub2 = os.path.join(tmpdir, "set2")
        os.makedirs(sub1, exist_ok=True)
        os.makedirs(sub2, exist_ok=True)

        pool1 = [
            {'scenario_id': 'sc1', 'load_pattern': {'2': 1.05, '3': 0.95}, 'contingencies': []}
        ]
        pool2 = [
            {'scenario_id': 'sc2', 'load_pattern': {'4': 1.10, '5': 0.90}, 'contingencies': []}
        ]
        save_scenario_pool(pool1, os.path.join(sub1, "set1_pool.json"))
        save_scenario_pool(pool2, os.path.join(sub2, "set2_pool.json"))

        seen_patterns, summaries = load_existing_pools_into_registry(tmpdir)
        assert len(seen_patterns) == 2
        k1 = frozenset([(2, 1.05), (3, 0.95)])
        k2 = frozenset([(4, 1.10), (5, 0.90)])
        assert k1 in seen_patterns
        assert k2 in seen_patterns

