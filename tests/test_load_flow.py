import pytest
from config.config import get_config
from src.network.load_network import load_network
from src.simulation.run_load_flow import run_load_flow
from src.results.extract_results import extract_results
from src.labeling.classify_security import classify_security


def test_normal_load_flow():
    cfg = get_config()

    mpc, meta = load_network()
    converged, raw_results = run_load_flow(mpc=mpc)
    results = extract_results(raw_results, converged, meta=meta)
    assert converged is True, "Normal condition power flow failed to converge!"
    assert results['convergence'] is True
    assert len(results['bus_voltages']) == 14
    assert len(results['bus_angles']) == 14
    assert len(results['line_loadings']) == 20
    assert len(results['gen_powers']) == 5

    # Sanity checks on voltages
    for v in results['bus_voltages']:
        assert 0.8 <= v <= 1.2, f"Bus voltage out of expected sanity range: {v}"

    # Classification test
    label, violations = classify_security(results, converged, cfg)
    assert label in ['Safe', 'Alert', 'Critical']
    assert isinstance(violations, list)


def test_custom_nr_load_flow():
    cfg = get_config()
    mpc, meta = load_network()

    # Test custom NR solver
    converged, raw_results = run_load_flow(mpc=mpc, solver_type="custom_nr")
    results = extract_results(raw_results, converged, meta=meta)
    assert converged is True, "Custom NR power flow failed to converge!"
    assert results['convergence'] is True
    assert len(results['bus_voltages']) == 14
    assert len(results['line_loadings']) == 20

    # Sanity checks on voltages
    for v in results['bus_voltages']:
        assert 0.8 <= v <= 1.2, f"Custom NR bus voltage out of expected sanity range: {v}"
