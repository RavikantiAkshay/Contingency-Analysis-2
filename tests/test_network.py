import pytest
from config.config import get_config
from src.network.load_network import load_network
from src.network.validate_network import validate_network


def test_load_and_validate_network():
    cfg = get_config()

    # Test loading
    mpc, meta = load_network()
    assert meta['n_bus'] == 14, f"Expected 14 buses, got {meta['n_bus']}"
    assert meta['n_line'] == 20, f"Expected 20 branches, got {meta['n_line']}"
    assert meta['n_gen'] == 5, f"Expected 5 generators, got {meta['n_gen']}"
    assert meta['vsource_name'] == "Gen.1" or len(meta['vsource_name']) > 0

    # Test validation
    is_valid, issues = validate_network(meta, mpc)
    assert is_valid, f"Network validation failed with issues: {issues}"
    assert len(issues) == 0


def test_load_and_validate_case30():
    mpc30, meta30 = load_network('case30')
    assert meta30['n_bus'] == 30
    assert meta30['n_line'] == 41
    assert meta30['n_gen'] == 6

    is_valid, issues = validate_network(meta30, mpc30)
    assert is_valid, f"case30 validation failed with issues: {issues}"
