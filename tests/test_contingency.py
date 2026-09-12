"""
Automated unit tests for N-1 contingency generation and application.
"""
from config.config import get_config
from src.network.load_network import load_network
from src.contingency.generate_contingencies import generate_contingencies
from src.contingency.apply_contingency import apply_contingency


def test_generate_line_contingencies():
    mpc, meta = load_network()
    contingencies = generate_contingencies(meta, mpc, contingency_type="line")
    assert len(contingencies) == 20, f"Expected 20 line contingencies, got {len(contingencies)}"
    first = contingencies[0]
    assert first['type'] == 'line'
    assert first['element_idx'] == 0
    assert 'bus1' in first and 'bus2' in first


def test_apply_line_contingency():
    mpc, meta = load_network()
    contingencies = generate_contingencies(meta, mpc, contingency_type="line")
    target = contingencies[0]  # First line contingency

    modified_mpc = apply_contingency(mpc, target)
    assert mpc['branch'][0, 10] == 1, "Base mpc was mutated!"
    assert modified_mpc['branch'][0, 10] == 0, "Target branch status was not set to 0!"


def test_random_load_generator():
    from src.contingency.random_load_generator import generate_random_load_pattern, find_valid_base_case
    cfg = get_config()
    mpc, meta = load_network()

    pattern, desc = generate_random_load_pattern(mpc, min_scale=0.80, max_scale=1.20)
    assert len(pattern) >= 2, "Expected at least 2 target buses in random load pattern."
    for bus_id, scale in pattern.items():
        assert 0.80 <= scale <= 1.20, f"Scale {scale} out of range [0.80, 1.20]"

    valid_pattern, base_res, desc = find_valid_base_case(mpc, cfg, min_scale=0.80, max_scale=1.20)
    assert valid_pattern is not None
    assert base_res['min_voltage'] > 0.80
