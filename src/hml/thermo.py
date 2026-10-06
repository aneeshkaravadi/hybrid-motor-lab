"""Equilibrium combustion and nozzle expansion with Cantera.

The chamber is an adiabatic, constant-pressure (HP) equilibrium. The nozzle is
an isentropic expansion, either with the composition held fixed ("frozen") or
re-equilibrated at every pressure ("shifting"). This is the same model NASA CEA
uses for its rocket problem, solved here with Cantera's NASA thermo database.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import cantera as ct
import numpy as np
from scipy.optimize import brentq

G0 = 9.80665
R_UNIVERSAL = ct.gas_constant / 1000.0  # J/(mol K)

ATOMIC_MASS = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007, "Al": 26.982, "Cl": 35.45}


@dataclass(frozen=True)
class Reactant:
    """A propellant ingredient defined by its formula and enthalpy of formation.

    If ``cantera_species`` is given, the enthalpy is taken from Cantera's NASA
    database at 298.15 K instead of ``dhf_kj_mol``.
    """

    name: str
    formula: dict[str, float]
    dhf_kj_mol: float = 0.0
    cantera_species: str | None = None

    @property
    def molar_mass(self) -> float:  # kg/mol
        return sum(ATOMIC_MASS[e] * n for e, n in self.formula.items()) / 1000.0

    def enthalpy_j_per_kg(self) -> float:
        if self.cantera_species:
            gas = _products_phase(tuple(sorted(self.formula)))
            gas.TPX = 298.15, ct.one_atm, {self.cantera_species: 1.0}
            return gas.enthalpy_mass
        return self.dhf_kj_mol * 1000.0 / self.molar_mass


# Common ingredients. Gas-phase enthalpies come straight from the NASA database.
O2 = Reactant("O2", {"O": 2}, cantera_species="O2")
N2O = Reactant("N2O", {"N": 2, "O": 1}, cantera_species="N2O")
H2 = Reactant("H2", {"H": 2}, cantera_species="H2")
CH4 = Reactant("CH4", {"C": 1, "H": 4}, cantera_species="CH4")
C2H4 = Reactant("C2H4", {"C": 2, "H": 4}, cantera_species="C2H4")
AIR = Reactant("air", {"N": 2 * 3.76 / 4.76, "O": 2 * 1 / 4.76})  # 1 mol of 21% O2 / 79% N2, h(298 K) = 0

# Paraffin wax modeled as n-C32H66. Its enthalpy of formation is uncertain
# (typical estimates fall between about -1.9 and -2.2 MJ/kg). We use -2.0 MJ/kg;
# tests/test_physics.py shows c* moves well under 1% across that range.
PARAFFIN_MOLAR_MASS = (32 * 12.011 + 66 * 1.008) / 1000.0
PARAFFIN = Reactant("paraffin", {"C": 32, "H": 66}, dhf_kj_mol=-2.0e3 * PARAFFIN_MOLAR_MASS)


@lru_cache(maxsize=None)
def _products_phase(elements: tuple[str, ...]) -> ct.Solution:
    """Ideal-gas phase with every neutral NASA species made only of ``elements``."""
    allowed = set(elements)
    species = [
        s for s in ct.Species.list_from_file("nasa_gas.yaml")
        if set(s.composition) <= allowed and s.charge == 0
    ]
    return ct.Solution(thermo="ideal-gas", species=species)


@dataclass
class State:
    T: float
    P: float
    rho: float
    h: float  # J/kg
    s: float  # J/kg/K
    gamma: float
    mw: float  # kg/kmol
    X: dict[str, float] = field(repr=False, default_factory=dict)


def _snapshot(gas: ct.Solution) -> State:
    major = {n: x for n, x in zip(gas.species_names, gas.X) if x > 1e-3}
    return State(gas.T, gas.P, gas.density, gas.enthalpy_mass, gas.entropy_mass,
                 gas.cp_mass / gas.cv_mass, gas.mean_molecular_weight, major)


def element_moles(reactants: list[tuple[Reactant, float]]) -> tuple[dict[str, float], float]:
    """Moles of each element and mixture enthalpy (J/kg) for (reactant, mass fraction) pairs."""
    b: dict[str, float] = {}
    h = 0.0
    for r, y in reactants:
        moles = y / r.molar_mass
        for e, n in r.formula.items():
            b[e] = b.get(e, 0.0) + n * moles
        h += y * r.enthalpy_j_per_kg()
    return b, h


def equilibrate_hp(reactants: list[tuple[Reactant, float]], P: float) -> ct.Solution:
    """Adiabatic constant-pressure equilibrium of a reactant mixture (mass fractions)."""
    b, h = element_moles(reactants)
    gas = _products_phase(tuple(sorted(b)))
    # Seed with atoms carrying the exact element ratios, relax to a sane state,
    # then impose the reactant enthalpy and solve the HP equilibrium.
    gas.TPX = 3000.0, P, {e: n for e, n in b.items()}
    gas.equilibrate("TP")
    gas.HP = h, P
    gas.equilibrate("HP")
    return gas


def bipropellant(ox: Reactant, fuel: Reactant, of: float) -> list[tuple[Reactant, float]]:
    return [(ox, of / (1.0 + of)), (fuel, 1.0 / (1.0 + of))]


@dataclass
class RocketPerformance:
    chamber: State
    throat: State
    exit: State
    cstar: float  # m/s
    cf: float  # thrust coefficient at the given ambient pressure
    cf_vac: float
    isp: float  # s, at ambient
    isp_vac: float
    area_ratio: float
    v_exit: float


def _expand(gas: ct.Solution, s0: float, h0: float, P: float, comp0, frozen: bool) -> tuple[State, float]:
    if frozen:
        gas.TPX = gas.T, gas.P, comp0
        gas.SP = s0, P
    else:
        gas.SP = s0, P
        gas.equilibrate("SP")
    v = np.sqrt(max(2.0 * (h0 - gas.enthalpy_mass), 0.0))
    return _snapshot(gas), v


def _sound_speed(gas: ct.Solution, frozen: bool) -> float:
    if frozen:
        return gas.sound_speed
    # Equilibrium sound speed: (dP/drho) along an equilibrium isentrope.
    s, P = gas.entropy_mass, gas.P
    rho0 = gas.density
    gas.SP = s, P * 1.001
    gas.equilibrate("SP")
    rho1 = gas.density
    gas.SP = s, P
    gas.equilibrate("SP")
    return np.sqrt(0.001 * P / (rho1 - rho0))


def rocket(reactants: list[tuple[Reactant, float]], pc: float, area_ratio: float,
           p_ambient: float = 0.0, frozen: bool = False) -> RocketPerformance:
    """Ideal rocket performance (infinite-area chamber), like CEA's 'rocket' problem."""
    gas = equilibrate_hp(reactants, pc)
    ch = _snapshot(gas)
    comp0 = gas.X.copy()
    s0, h0 = gas.entropy_mass, gas.enthalpy_mass

    def throat_residual(p):
        _, v = _expand(gas, s0, h0, p, comp0, frozen)
        return v - _sound_speed(gas, frozen)

    pt = brentq(throat_residual, 0.40 * pc, 0.75 * pc, xtol=1e-6 * pc)
    th, vt = _expand(gas, s0, h0, pt, comp0, frozen)
    flux_t = th.rho * vt  # mass flux at the throat, kg/m^2/s

    def area_residual(p):
        st, v = _expand(gas, s0, h0, p, comp0, frozen)
        return flux_t / (st.rho * v) - area_ratio

    pe = brentq(area_residual, pc * 0.005 / area_ratio**1.2, pt * 0.999, xtol=1e-9 * pc)
    ex, ve = _expand(gas, s0, h0, pe, comp0, frozen)

    cstar = pc / flux_t
    cf_vac = (ve + pe * area_ratio / flux_t) / cstar
    cf = cf_vac - p_ambient * area_ratio / pc
    return RocketPerformance(ch, th, ex, cstar, cf, cf_vac, cstar * cf / G0,
                             cstar * cf_vac / G0, area_ratio, ve)


@dataclass
class PerformanceTable:
    """c* and vacuum C_F on an (O/F, Pc) grid, for fast lookup inside time-stepping sims."""

    of: np.ndarray
    pc: np.ndarray
    cstar: np.ndarray  # shape (len(of), len(pc))
    cf_vac: np.ndarray
    area_ratio: float

    def lookup(self, of: float, pc: float) -> tuple[float, float]:
        from scipy.interpolate import RegularGridInterpolator
        if not hasattr(self, "_ic"):
            self._ic = RegularGridInterpolator((self.of, self.pc), self.cstar, bounds_error=False, fill_value=None)
            self._if = RegularGridInterpolator((self.of, self.pc), self.cf_vac, bounds_error=False, fill_value=None)
        pt = [[np.clip(of, self.of[0], self.of[-1]), np.clip(pc, self.pc[0], self.pc[-1])]]
        return float(self._ic(pt)[0]), float(self._if(pt)[0])


def build_table(ox: Reactant, fuel: Reactant, of_values, pc_values, area_ratio: float) -> PerformanceTable:
    of_values, pc_values = np.asarray(of_values, float), np.asarray(pc_values, float)
    cs = np.zeros((len(of_values), len(pc_values)))
    cf = np.zeros_like(cs)
    for i, of in enumerate(of_values):
        for j, pc in enumerate(pc_values):
            r = rocket(bipropellant(ox, fuel, of), pc, area_ratio)
            cs[i, j], cf[i, j] = r.cstar, r.cf_vac
    return PerformanceTable(of_values, pc_values, cs, cf, area_ratio)
