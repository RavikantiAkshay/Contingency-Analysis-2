"""
Automated unit tests for dataset building and integrity validation.
"""
from config.config import get_config
from src.network.load_network import load_network
from src.simulation.run_simulation_campaign import run_simulation_campaign
from src.dataset.build_dataset import build_dataset
from src.dataset.validate_dataset import validate_dataset


def test_build_and_validate_small_dataset():
    cfg = get_config()
    mpc, meta = load_network()
    
    # Run small campaign over 2 randomized scenarios for testing speed
    all_results, sim_log = run_simulation_campaign(cfg, meta, mpc, contingency_type="line", num_scenarios=2)
    
    # Total expected cases: 2 scenarios * (1 base + 20 lines) = 42 cases
    assert len(all_results) == 42, f"Expected 42 cases, got {len(all_results)}"

    dataset = build_dataset(all_results, cfg, filename='test_master_dataset.csv')
    assert dataset.shape[0] == 42, f"Expected 42 rows in dataset, got {dataset.shape[0]}"
    assert dataset.shape[1] == 63, f"Expected 63 columns per refined schema, got {dataset.shape[1]}"

    is_valid, report = validate_dataset(dataset, meta, cfg)
    assert is_valid, f"Dataset validation failed: {report['errors']}"
