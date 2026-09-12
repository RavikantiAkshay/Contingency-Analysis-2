"""
Custom Standalone AC Newton-Raphson Power Flow Solver.
Pure Python/NumPy implementation of the AC Newton-Raphson algorithm.

Computes exact AC power flow solutions, bus voltage magnitudes/angles,
line active/reactive power flows, and convergence status without external solver dependencies.

Reference: EEQ401 — ML Architecture & LLD Section 4.
"""

import copy
import numpy as np
from typing import Dict, Any, Tuple, Optional


def build_ybus(mpc: Dict[str, Any]) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Builds the Nodal Admittance Matrix (Y_bus) from a PyPOWER case dictionary.

    Returns:
        Ybus (np.ndarray): Complex (n_bus, n_bus) admittance matrix.
        bus_map (dict): Mapping from bus_id to 0-indexed matrix index.
    """
    bus = mpc['bus']
    branch = mpc['branch']
    baseMVA = float(mpc['baseMVA'])

    n_bus = len(bus)
    bus_ids = bus[:, 0].astype(int)
    bus_map = {bid: idx for idx, bid in enumerate(bus_ids)}

    Ybus = np.zeros((n_bus, n_bus), dtype=np.complex128)

    # 1. Branch admittances
    for br in branch:
        status = int(br[10])
        if status <= 0:
            continue  # Line is outaged/disabled

        f_bus = int(br[0])
        t_bus = int(br[1])
        if f_bus not in bus_map or t_bus not in bus_map:
            continue

        i = bus_map[f_bus]
        j = bus_map[t_bus]

        r = float(br[2])
        x = float(br[3])
        b_shunt = float(br[4])  # total line charging susceptance
        tap = float(br[8]) if float(br[8]) != 0.0 else 1.0

        z = complex(r, x)
        y = 1.0 / z if abs(z) > 1e-12 else 0.0
        y_shunt = complex(0.0, b_shunt / 2.0)

        # Off-diagonal & Diagonal updates considering transformer tap ratios
        Ybus[i, i] += (y / (tap**2)) + y_shunt
        Ybus[j, j] += y + y_shunt
        Ybus[i, j] -= y / tap
        Ybus[j, i] -= y / tap

    # 2. Bus shunt admittances
    for idx, b in enumerate(bus):
        gs = float(b[4]) / baseMVA
        bs = float(b[5]) / baseMVA
        Ybus[idx, idx] += complex(gs, bs)

    return Ybus, bus_map


def solve_ac_nr_custom(
    mpc: Dict[str, Any],
    max_iter: int = 20,
    tol: float = 1e-8
) -> Tuple[bool, Dict[str, Any]]:
    """
    Executes pure AC Newton-Raphson Power Flow solution on a PyPOWER case struct.

    Args:
        mpc (dict): PyPOWER case dict.
        max_iter (int): Maximum NR iterations allowed (default 20).
        tol (float): Power mismatch convergence tolerance in per-unit (default 1e-8).

    Returns:
        converged (bool): True if AC NR converged within tolerance, False otherwise.
        mpc_solved (dict): Solved PyPOWER case dict with updated bus and branch matrices.
    """
    mpc_solved = copy.deepcopy(mpc)
    bus = mpc_solved['bus']
    branch = mpc_solved['branch']
    gen = mpc_solved['gen']
    baseMVA = float(mpc_solved['baseMVA'])

    n_bus = len(bus)
    Ybus, bus_map = build_ybus(mpc_solved)
    G = Ybus.real
    B = Ybus.imag

    # Classify buses & map indices
    bus_ids = bus[:, 0].astype(int)
    bus_types = bus[:, 1].astype(int)  # 1=PQ, 2=PV, 3=Slack

    slack_idx = [i for i, t in enumerate(bus_types) if t == 3]
    pv_idx = [i for i, t in enumerate(bus_types) if t == 2]
    pq_idx = [i for i, t in enumerate(bus_types) if t == 1]

    # Non-slack buses (for P equations) and PQ buses (for Q equations)
    pv_pq_idx = pv_idx + pq_idx  # dTheta variables
    
    # State vectors: Voltage Magnitude V and Angle Theta (radians)
    V = bus[:, 7].copy()      # VM (p.u.)
    Va = np.deg2rad(bus[:, 8]) # VA (rad)

    # Calculate Specified Power Injections (P_spec, Q_spec in p.u.)
    P_spec = np.zeros(n_bus, dtype=np.float64)
    Q_spec = np.zeros(n_bus, dtype=np.float64)

    # Subtract bus loads
    P_spec -= bus[:, 2] / baseMVA
    Q_spec -= bus[:, 3] / baseMVA

    # Add generator outputs
    for g in gen:
        g_status = int(g[7])
        if g_status > 0:
            g_bus = int(g[0])
            if g_bus in bus_map:
                idx = bus_map[g_bus]
                P_spec[idx] += g[1] / baseMVA
                Q_spec[idx] += g[2] / baseMVA

    converged = False

    for iteration in range(max_iter):
        # 1. Calculate Calculated Active and Reactive Power Injections
        P_calc = np.zeros(n_bus, dtype=np.float64)
        Q_calc = np.zeros(n_bus, dtype=np.float64)

        for i in range(n_bus):
            theta_i = Va[i]
            V_i = V[i]
            for j in range(n_bus):
                theta_j = Va[j]
                V_j = V[j]
                dtheta = theta_i - theta_j
                P_calc[i] += V_i * V_j * (G[i, j] * np.cos(dtheta) + B[i, j] * np.sin(dtheta))
                Q_calc[i] += V_i * V_j * (G[i, j] * np.sin(dtheta) - B[i, j] * np.cos(dtheta))

        # 2. Power Mismatches
        dP = P_spec[pv_pq_idx] - P_calc[pv_pq_idx]
        dQ = Q_spec[pq_idx] - Q_calc[pq_idx]

        mismatch = np.concatenate([dP, dQ])
        max_mismatch = np.max(np.abs(mismatch)) if len(mismatch) > 0 else 0.0

        if max_mismatch < tol:
            converged = True
            break

        # 3. Form Jacobian Matrix J = [J11 J12; J21 J22]
        n_pv_pq = len(pv_pq_idx)
        n_pq = len(pq_idx)

        J11 = np.zeros((n_pv_pq, n_pv_pq), dtype=np.float64)
        J12 = np.zeros((n_pv_pq, n_pq), dtype=np.float64)
        J21 = np.zeros((n_pq, n_pv_pq), dtype=np.float64)
        J22 = np.zeros((n_pq, n_pq), dtype=np.float64)

        # J11 = dP / dTheta
        for r, i in enumerate(pv_pq_idx):
            for c, j in enumerate(pv_pq_idx):
                if i == j:
                    J11[r, c] = -Q_calc[i] - B[i, i] * (V[i]**2)
                else:
                    dtheta = Va[i] - Va[j]
                    J11[r, c] = V[i] * V[j] * (G[i, j] * np.sin(dtheta) - B[i, j] * np.cos(dtheta))

        # J12 = dP / dV (for PQ buses)
        for r, i in enumerate(pv_pq_idx):
            for c, j in enumerate(pq_idx):
                if i == j:
                    J12[r, c] = (P_calc[i] / V[i]) + G[i, i] * V[i]
                else:
                    dtheta = Va[i] - Va[j]
                    J12[r, c] = V[i] * (G[i, j] * np.cos(dtheta) + B[i, j] * np.sin(dtheta))

        # J21 = dQ / dTheta (for PQ buses)
        for r, i in enumerate(pq_idx):
            for c, j in enumerate(pv_pq_idx):
                if i == j:
                    J21[r, c] = P_calc[i] - G[i, i] * (V[i]**2)
                else:
                    dtheta = Va[i] - Va[j]
                    J21[r, c] = -V[i] * V[j] * (G[i, j] * np.cos(dtheta) + B[i, j] * np.sin(dtheta))

        # J22 = dQ / dV (for PQ buses)
        for r, i in enumerate(pq_idx):
            for c, j in enumerate(pq_idx):
                if i == j:
                    J22[r, c] = (Q_calc[i] / V[i]) - B[i, i] * V[i]
                else:
                    dtheta = Va[i] - Va[j]
                    J22[r, c] = V[i] * (G[i, j] * np.sin(dtheta) - B[i, j] * np.cos(dtheta))

        # Assemble full Jacobian matrix
        J_top = np.hstack([J11, J12])
        J_bot = np.hstack([J21, J22])
        J = np.vstack([J_top, J_bot])

        # 4. Solve J * dx = mismatch
        try:
            dx = np.linalg.solve(J, mismatch)
        except np.linalg.LinAlgError:
            converged = False
            break

        # 5. Update State Variables
        dtheta_update = dx[:n_pv_pq]
        dV_update = dx[n_pv_pq:]

        Va[pv_pq_idx] += dtheta_update
        V[pq_idx] += dV_update

    # Write solved state back to mpc_solved
    bus[:, 7] = V
    bus[:, 8] = np.rad2deg(Va)

    # Ensure branch matrix has 15 columns (PyPOWER standard format for PF, QF, PT, QT)
    if branch.shape[1] < 15:
        padded_branch = np.zeros((len(branch), 15), dtype=np.float64)
        padded_branch[:, :branch.shape[1]] = branch
        branch = padded_branch
        mpc_solved['branch'] = branch

    # Compute line flows for solved state
    for br_idx, br in enumerate(branch):
        if int(br[10]) <= 0:
            branch[br_idx, 13] = 0.0
            branch[br_idx, 14] = 0.0
            continue

        f_bus = bus_map[int(br[0])]
        t_bus = bus_map[int(br[1])]
        r, x = float(br[2]), float(br[3])
        b_shunt = float(br[4])
        tap = float(br[8]) if float(br[8]) != 0.0 else 1.0

        z = complex(r, x)
        y = 1.0 / z if abs(z) > 1e-12 else 0.0

        V_f = V[f_bus] * np.exp(1j * Va[f_bus])
        V_t = V[t_bus] * np.exp(1j * Va[t_bus])

        # Current from 'from' bus to 'to' bus
        I_ft = (V_f / (tap**2) - V_t / tap) * y + V_f * complex(0, b_shunt / 2.0)
        S_ft = V_f * np.conj(I_ft) * baseMVA

        branch[br_idx, 13] = S_ft.real
        branch[br_idx, 14] = S_ft.imag

    return converged, mpc_solved


class CustomNRSolver:
    """
    Standalone AC Newton-Raphson Solver wrapper compatible with PyPowerSolver interface.
    """
    def __init__(self, mpc: Optional[Dict[str, Any]] = None):
        from pypower.api import case14
        self.base_mpc = mpc if mpc is not None else case14()

    def solve(
        self,
        load_scale: Any = 1.0,
        outaged_branch: Optional[Any] = None,
        outaged_gen: Optional[Any] = None,
        mpc: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        base = mpc if mpc is not None else self.base_mpc
        target_mpc = copy.deepcopy(base)

        # Scale loads
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

        # Apply outaged branch
        if outaged_branch is not None:
            b_idx = int(outaged_branch) if isinstance(outaged_branch, int) else None
            if b_idx is None and isinstance(outaged_branch, str):
                for idx, br in enumerate(target_mpc['branch']):
                    name = f"Line.line_{int(br[0])}_{int(br[1])}"
                    if name == outaged_branch:
                        b_idx = idx
                        break
            if b_idx is not None and 0 <= b_idx < len(target_mpc['branch']):
                target_mpc['branch'][b_idx, 10] = 0

        converged, mpc_solved = solve_ac_nr_custom(target_mpc)
        return converged, mpc_solved
