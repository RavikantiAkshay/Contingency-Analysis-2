import copy
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from pypower.api import case14, runpf, ppoption

BRANCH_NAMES = [
    "Line.line_1_2", "Line.line_1_5", "Line.line_2_3", "Line.line_2_4",
    "Line.line_2_5", "Line.line_3_4", "Line.line_4_5", "Transformer.t_4_7",
    "Transformer.t_4_9", "Transformer.t_5_6", "Line.line_6_11", "Line.line_6_12",
    "Line.line_6_13", "Line.line_7_8", "Line.line_7_9", "Line.line_9_10",
    "Line.line_9_14", "Line.line_10_11", "Line.line_12_13", "Line.line_13_14"
]

NORM_AMPS = [
    800.0, 400.0, 300.0, 350.0, 300.0, 300.0, 300.0, 300.0,
    200.0, 250.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0,
    150.0, 100.0, 100.0, 100.0
]


class PyPowerSolver:
    """
    Generalized AC Newton-Raphson Power Flow engine using PyPOWER case14.
    """
    def __init__(self, mpc: Optional[Dict[str, Any]] = None):
        self.base_mpc = mpc if mpc is not None else case14()
        self.ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
        self.branch_names = BRANCH_NAMES
        self.norm_amps = NORM_AMPS

    def solve(
        self,
        load_scale: float = 1.0,
        outaged_branch: Optional[str] = None,
        outaged_gen: Optional[str] = None,
        mpc: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        base = mpc if mpc is not None else self.base_mpc
        target_mpc = copy.deepcopy(base)

        # Scale active and reactive loads (supports float, dict, or list/array)
        if isinstance(load_scale, dict):
            for bus_id, scale in load_scale.items():
                mask = (target_mpc['bus'][:, 0] == int(bus_id))
                target_mpc['bus'][mask, 2] *= float(scale)
                target_mpc['bus'][mask, 3] *= float(scale)
        elif isinstance(load_scale, (list, np.ndarray)):
            target_mpc['bus'][:, 2] *= load_scale
            target_mpc['bus'][:, 3] *= load_scale
        else:
            target_mpc['bus'][:, 2] *= float(load_scale)
            target_mpc['bus'][:, 3] *= float(load_scale)

        # Apply branch outage if specified
        if outaged_branch:
            try:
                branch_idx = self.branch_names.index(outaged_branch) if isinstance(outaged_branch, str) else int(outaged_branch)
                target_mpc['branch'][branch_idx, 10] = 0  # Status 0 = offline
            except (ValueError, IndexError):
                pass

        # Apply generator outage if specified
        if outaged_gen:
            try:
                gen_idx = int(outaged_gen.replace("Gen.", "").replace("Gen at Bus ", "")) - 1 if isinstance(outaged_gen, str) else int(outaged_gen)
                target_mpc['gen'][gen_idx, 7] = 0  # Status 0 = offline
            except (ValueError, IndexError):
                pass

        # Run AC Newton-Raphson Power Flow
        results, success = runpf(target_mpc, self.ppopt)

        if not success:
            return False, None

        return True, self._extract_results(results)

    def _extract_results(self, results: Dict) -> Dict[str, Any]:
        bus_data = results['bus']
        branch_data = results['branch']
        gen_data = results['gen']

        # Build Bus DataFrame
        buses_df = pd.DataFrame({
            'bus_id': bus_data[:, 0].astype(int),
            'v_mag_pu': bus_data[:, 7],
            'v_angle_deg': bus_data[:, 8],
            'p_load_mw': bus_data[:, 2],
            'q_load_mvar': bus_data[:, 3],
            'v_max_pu': bus_data[:, 11],
            'v_min_pu': bus_data[:, 12]
        })

        # Map Generators to Buses
        gen_p = np.zeros(len(buses_df))
        gen_q = np.zeros(len(buses_df))
        for g in gen_data:
            bus_idx = np.where(buses_df['bus_id'] == int(g[0]))[0][0]
            gen_p[bus_idx] += g[1]
            gen_q[bus_idx] += g[2]

        buses_df['p_gen_mw'] = gen_p
        buses_df['q_gen_mvar'] = gen_q
        buses_df['p_net_mw'] = buses_df['p_gen_mw'] - buses_df['p_load_mw']
        buses_df['q_net_mvar'] = buses_df['q_gen_mvar'] - buses_df['q_load_mvar']

        # Build Branch DataFrame dynamically matching network branch count
        n_branches = len(branch_data)
        if len(self.branch_names) == n_branches:
            b_names = self.branch_names
            n_amps = self.norm_amps
        else:
            b_names = []
            n_amps = []
            for b in branch_data:
                f_bus, t_bus = int(b[0]), int(b[1])
                tap = b[8]
                prefix = "Transformer.t" if (tap != 0 and tap != 1.0) else "Line.line"
                b_names.append(f"{prefix}_{f_bus}_{t_bus}")
                rate_a = b[5]
                val_mva = float(rate_a) if rate_a > 0 else 99.0
                n_amps.append(val_mva * 1000.0 / (np.sqrt(3) * 144.0))

        branches_df = pd.DataFrame({
            'branch_name': b_names,
            'bus1': branch_data[:, 0].astype(int),
            'bus2': branch_data[:, 1].astype(int),
            'p_flow_mw': branch_data[:, 13],
            'q_flow_mvar': branch_data[:, 14],
            'status': ['ENABLED' if s else 'DISABLED' for s in branch_data[:, 10]]
        })

        # Calculate Apparent Power (MVA) and Thermal Loading
        s_mva = np.sqrt(branches_df['p_flow_mw']**2 + branches_df['q_flow_mvar']**2)
        branches_df['norm_amps'] = n_amps
        branches_df['current_amps'] = s_mva * 1000 / (np.sqrt(3) * 144)
        branches_df['loading_pct'] = (branches_df['current_amps'] / branches_df['norm_amps']) * 100
        branches_df.loc[branches_df['status'] == 'DISABLED', 'loading_pct'] = 0.0

        v_dev = (buses_df['v_mag_pu'] - 1.0).abs().max()
        gen_p_list = gen_data[:, 1].tolist()

        return {
            'bus_table': buses_df,
            'branch_table': branches_df,
            'bus_voltages': buses_df['v_mag_pu'].tolist(),
            'bus_angles': buses_df['v_angle_deg'].tolist(),
            'line_loadings': branches_df['loading_pct'].tolist(),
            'gen_powers': gen_p_list,
            'min_voltage': buses_df['v_mag_pu'].min(),
            'max_voltage': buses_df['v_mag_pu'].max(),
            'max_line_loading': branches_df['loading_pct'].max(),
            'max_voltage_deviation': float(v_dev),
            'total_load': buses_df['p_load_mw'].sum() * 1000,
            'total_load_kvar': buses_df['q_load_mvar'].sum() * 1000,
            'total_load_mw': buses_df['p_load_mw'].sum(),
            'total_load_mvar': buses_df['q_load_mvar'].sum(),
            'total_gen_mw': buses_df['p_gen_mw'].sum(),
            'total_gen_mvar': buses_df['q_gen_mvar'].sum(),
            'convergence': True,
        }


def run_load_flow(
    mpc: Optional[Dict[str, Any]] = None,
    solver_type: str = "pypower",
    load_scale: float = 1.0,
    outaged_branch: Optional[str] = None,
    outaged_gen: Optional[str] = None,
    return_solved_mpc: bool = False
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Solves AC Newton-Raphson load flow for specified load scale and contingency state.

    Args:
        mpc (dict, optional): PyPOWER case dict. If None, loads case14().
        solver_type (str): Solver backend (default: 'pypower').
        load_scale (float): Load scaling factor (e.g. 1.0 for normal, 0.7 for light).
        outaged_branch (str, optional): Identifier of branch to outage.
        outaged_gen (str, optional): Identifier of generator to outage.
        return_solved_mpc (bool): If True, returns the raw solved PyPOWER mpc dict
            instead of the extracted results dict. Used by the graph pipeline.

    Returns:
        converged (bool): Whether load flow converged.
        results (dict or None): Electrical results dictionary (or raw solved mpc if return_solved_mpc=True).
    """
    if solver_type == "custom_nr":
        from src.simulation.custom_nr_solver import CustomNRSolver
        custom_solver = CustomNRSolver(mpc=mpc)
        converged, solved_mpc = custom_solver.solve(load_scale=load_scale, outaged_branch=outaged_branch, outaged_gen=outaged_gen, mpc=mpc)
        if not converged:
            return False, None
        if return_solved_mpc:
            return True, solved_mpc
        pypower_solver = PyPowerSolver(mpc=mpc)
        return True, pypower_solver._extract_results(solved_mpc)

    solver = PyPowerSolver(mpc=mpc)

    if return_solved_mpc:
        base = mpc if mpc is not None else solver.base_mpc
        target_mpc = copy.deepcopy(base)
        if isinstance(load_scale, dict):
            for bus_id, scale in load_scale.items():
                mask = (target_mpc['bus'][:, 0] == int(bus_id))
                target_mpc['bus'][mask, 2] *= float(scale)
                target_mpc['bus'][mask, 3] *= float(scale)
        elif isinstance(load_scale, (list, np.ndarray)):
            target_mpc['bus'][:, 2] *= load_scale
            target_mpc['bus'][:, 3] *= load_scale
        else:
            target_mpc['bus'][:, 2] *= float(load_scale)
            target_mpc['bus'][:, 3] *= float(load_scale)
        if outaged_branch is not None:
            try:
                branch_idx = solver.branch_names.index(outaged_branch) if isinstance(outaged_branch, str) else int(outaged_branch)
                target_mpc['branch'][branch_idx, 10] = 0
            except (ValueError, IndexError):
                pass
        if outaged_gen is not None:
            try:
                gen_idx = int(outaged_gen.replace("Gen.", "").replace("Gen at Bus ", "")) - 1 if isinstance(outaged_gen, str) else int(outaged_gen)
                target_mpc['gen'][gen_idx, 7] = 0
            except (ValueError, IndexError):
                pass
        results, success = runpf(target_mpc, solver.ppopt)
        if not success:
            return False, None
        return True, results

    return solver.solve(load_scale=load_scale, outaged_branch=outaged_branch, outaged_gen=outaged_gen, mpc=mpc)
