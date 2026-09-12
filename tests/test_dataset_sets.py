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
