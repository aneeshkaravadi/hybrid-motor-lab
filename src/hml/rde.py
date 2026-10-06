"""Detonation physics for rotating detonation engine (RDE) studies.

1. Chapman-Jouguet (CJ) detonation state from full chemical equilibrium (Cantera).
2. A one-gamma model calibrated to that CJ speed, used to compare ideal cycle
   efficiencies: Brayton (constant pressure), Humphrey (constant volume) and
   Fickett-Jacobs (detonation). This is the "pressure-gain combustion" argument
   for why detonation engines can beat conventional combustors.
3. First-cut annulus sizing from the empirical correlations of Bykovskii,
   Zhdan & Vedernikov (J. Propulsion & Power 22(6), 2006). These need the
   detonation cell size, which must come from experiments.
"""
from __future__ import annotations

from dataclasses import dataclass

import cantera as ct
import numpy as np
from scipy.optimize import fsolve

from .thermo import Reactant, _products_phase, element_moles


@dataclass
class CJResult:
    D: float  # CJ detonation speed, m/s
    T2: float
    P2: float
    rho_ratio: float  # rho2 / rho1
    gamma2: float  # equilibrium products, at the CJ state
    T1: float
    P1: float
    c1: float  # reactant (frozen) sound speed
    gamma1: float


def _reactant_gas(reactants: list[tuple[Reactant, float]], T1: float, P1: float) -> ct.Solution:
    """Unburned mixture as an ideal gas. Every reactant must be a Cantera species (or air)."""
    b, _ = element_moles(reactants)
    gas = ct.Solution(thermo="ideal-gas", species=_products_phase(tuple(sorted(b))).species())
    X = {}
    for r, y in reactants:
        if r.name == "air":
            X["O2"] = X.get("O2", 0) + 0.21 * y / r.molar_mass
            X["N2"] = X.get("N2", 0) + 0.79 * y / r.molar_mass
        else:
            X[r.cantera_species] = X.get(r.cantera_species, 0) + y / r.molar_mass
    gas.TPX = T1, P1, X
    return gas


def _eq_sound_speed(gas: ct.Solution) -> float:
    s, P, rho0 = gas.entropy_mass, gas.P, gas.density
    gas.SP = s, P * 1.0005
    gas.equilibrate("SP")
    rho1 = gas.density
    gas.SP = s, P
    gas.equilibrate("SP")
    return np.sqrt(0.0005 * P / (rho1 - rho0))


def cj_state(reactants: list[tuple[Reactant, float]], T1: float = 298.15, P1: float = ct.one_atm) -> CJResult:
    """Solve mass, momentum and energy across the wave with u2 = equilibrium sound speed."""
    unburned = _reactant_gas(reactants, T1, P1)
    rho1, h1 = unburned.density, unburned.enthalpy_mass
    c1, g1 = unburned.sound_speed, unburned.cp_mass / unburned.cv_mass
    b, _ = element_moles(reactants)
    burned = _products_phase(tuple(sorted(b)))

    def state(T2, P2):
        burned.TPX = 3000.0, P1, {e: n for e, n in b.items()}
        burned.equilibrate("TP")
        burned.TP = T2, P2
        burned.equilibrate("TP")
        return burned.density, burned.enthalpy_mass, _eq_sound_speed(burned)

    def residual(x):
        T2, P2 = x[0] * 1000.0, x[1] * P1
        rho2, h2, a2 = state(T2, P2)
        u2 = a2
        w1 = rho2 * u2 / rho1
        mom = (P1 + rho1 * w1**2) - (P2 + rho2 * u2**2)
        energy = (h1 + 0.5 * w1**2) - (h2 + 0.5 * u2**2)
        return [mom / (rho1 * w1**2), energy / (0.5 * w1**2)]

    sol, _, ier, msg = fsolve(residual, [3.0, 18.0], full_output=True, xtol=1e-10)
    if ier != 1:
        raise RuntimeError(f"CJ solve failed: {msg}")
    T2, P2 = sol[0] * 1000.0, sol[1] * P1
    rho2, _, a2 = state(T2, P2)
    D = rho2 * a2 / rho1
    return CJResult(D, T2, P2, rho2 / rho1, burned.cp_mass / burned.cv_mass, T1, P1, c1, g1)


# ---------------------------------------------------------------- one-gamma cycles

@dataclass
class OneGamma:
    """Perfect gas with a single gamma and heat release q, matched to a CJ speed."""

    gamma: float
    R: float  # J/kg/K
    T1: float
    q: float  # J/kg

    @property
    def cp(self) -> float:
        return self.gamma * self.R / (self.gamma - 1.0)

    @classmethod
    def from_cj(cls, cj: CJResult, R: float, gamma: float | None = None) -> OneGamma:
        g = cj.gamma2 if gamma is None else gamma
        c1 = np.sqrt(g * R * cj.T1)
        M = cj.D / c1
        H = ((M**2 - 1.0) / (2.0 * M)) ** 2  # inverts M_CJ = sqrt(1+H) + sqrt(H)
        q = 2.0 * g * R * cj.T1 * H / (g**2 - 1.0)
        return cls(g, R, cj.T1, q)

    def m_cj(self, T: float) -> float:
        H = (self.gamma**2 - 1.0) * self.q / (2.0 * self.gamma * self.R * T)
        return np.sqrt(1.0 + H) + np.sqrt(H)

    def efficiency(self, cycle: str, pi_c: float = 1.0) -> float:
        """Ideal thermal efficiency after isentropic precompression by pressure ratio pi_c."""
        g, T1 = self.gamma, self.T1
        tau = pi_c ** ((g - 1.0) / g)
        T2 = T1 * tau  # after compression
        if cycle == "brayton":
            return 1.0 - 1.0 / tau
        if cycle == "humphrey":
            cv = self.R / (g - 1.0)
            T3 = T2 + self.q / cv
            # Constant-volume burn raises P by T3/T2; expand back to P1.
            T4 = T3 * (1.0 / (pi_c * T3 / T2)) ** ((g - 1.0) / g)
            return 1.0 - self.cp * (T4 - T1) / self.q
        if cycle == "fickett-jacobs":
            M = self.m_cj(T2)
            p_ratio = (1.0 + g * M**2) / (1.0 + g)  # P_CJ / P2
            t_ratio = (1.0 + g * M**2) ** 2 / ((1.0 + g) ** 2 * M**2)  # T_CJ / T2
            T_cj = T2 * t_ratio
            T4 = T_cj * (1.0 / (pi_c * p_ratio)) ** ((g - 1.0) / g)
            return 1.0 - self.cp * (T4 - T1) / self.q
        raise ValueError(cycle)


# ---------------------------------------------------------------- RDE sizing

@dataclass
class RDESizing:
    cell_size_mm: float
    fill_height_mm: tuple[float, float, float]  # low, nominal, high
    min_channel_width_mm: float
    min_length_mm: float
    min_diameter_mm: float
    diameter_mm: float
    feasible: bool  # requested diameter at least the empirical minimum
    waves_for_diameter: tuple[int, int]  # range for the requested diameter
    wave_frequency_khz: tuple[float, float]


def bykovskii_sizing(cell_size_mm: float, diameter_mm: float, D_cj: float, velocity_deficit: float = 0.85) -> RDESizing:
    """Empirical first-cut geometry. Real RDE waves usually run 80-90% of CJ speed.

    Correlations (Bykovskii et al. 2006): critical fill height h* = (12 +/- 5) lambda,
    minimum channel width 0.2 h*, minimum length 2 h*, minimum diameter 40 lambda,
    spacing between waves l = (7 +/- 2) h*.
    """
    lam = cell_size_mm
    h_lo, h_nom, h_hi = 7 * lam, 12 * lam, 17 * lam
    circumference = np.pi * diameter_mm
    # fewest waves: largest h* and largest spacing; most waves: smallest of both
    n_min = max(1, int(np.floor(circumference / (9 * h_hi))))
    n_max = max(1, int(np.floor(circumference / (5 * h_lo))))
    D = velocity_deficit * D_cj
    f = lambda n: n * D / (circumference / 1000.0) / 1000.0
    return RDESizing(lam, (h_lo, h_nom, h_hi), 0.2 * h_nom, 2 * h_nom, 40 * lam, diameter_mm,
                     diameter_mm >= 40 * lam, (n_min, n_max), (f(n_min), f(n_max)))
