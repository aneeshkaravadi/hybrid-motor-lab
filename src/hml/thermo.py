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
from scipy.optimize import brentq, minimize_scalar

G0 = 9.80665
R_UNIVERSAL = ct.gas_constant / 1000.0  # J/(mol K)

ATOMIC_MASS = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007, "Al": 26.982, "Cl": 35.45, "S": 32.06,
               "Mg": 24.305}


@dataclass(frozen=True)
class Reactant:
    """A propellant ingredient defined by its formula and enthalpy of formation.

    If ``cantera_species`` is given, the enthalpy is taken from Cantera's NASA
    database (gas or condensed) at 298.15 K instead of ``dhf_kj_mol``.
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
            sp = _nasa_species()[self.cantera_species]
            return sp.thermo.h(298.15) / sp.molecular_weight
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
def _nasa_species() -> dict[str, ct.Species]:
    return {sp.name: sp for f in ("nasa_gas.yaml", "nasa_condensed.yaml") for sp in ct.Species.list_from_file(f)}


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
    """Moles of each element per kg and mixture enthalpy (J/kg) for (reactant, mass fraction) pairs.

    The fractions are normalized to sum to 1 (as CEA does), so weight percentages work too.
    """
    b: dict[str, float] = {}
    h = 0.0
    total = sum(y for _, y in reactants)
    for r, y in reactants:
        y = y / total
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


# ---------------------------------------------------------------- condensed products
#
# Aluminum burns to Al2O3, which is liquid in the chamber and freezes in the
# nozzle at 2327 K. Leaving it out of the equilibrium forces the aluminum into
# gaseous suboxides and gets the temperature, c* and Isp wrong. Here each NASA
# condensed species made of the elements present is its own phase next to the
# gas, and Cantera's multiphase solver finds the equilibrium. Like CEA, only
# condensed species within their temperature range take part, and condensed
# products take up no volume.

# Aluminized-propellant ingredients. The ammonium perchlorate enthalpy is the
# value NASA's CEA uses (-70690.010 cal/mol, printed in its Example 5 output),
# since Cantera's condensed data doesn't include AP.
AP = Reactant("NH4ClO4(I)", {"N": 1, "H": 4, "Cl": 1, "O": 4}, dhf_kj_mol=-70690.010 * 4.184e-3)
ALUMINUM = Reactant("Al", {"Al": 1}, cantera_species="AL(cr)")
# The hydrocarbon binder from CEA's Example 5 (NASA RP-1311 Part II).
CEA_BINDER = Reactant("CHOS binder", {"C": 1, "H": 1.86955, "O": 0.031256, "S": 0.008415}, dhf_kj_mol=-2999.082 * 4.184e-3)
MGO = Reactant("MgO(cr)", {"Mg": 1, "O": 1}, dhf_kj_mol=-143785.851 * 4.184e-3)
WATER_LIQUID = Reactant("H2O(L)", {"H": 2, "O": 1}, cantera_species="H2O(L)")


@lru_cache(maxsize=None)
def _condensed_phases(elements: tuple[str, ...]) -> tuple[tuple[float, float, ct.Solution], ...]:
    """Every NASA condensed species made only of ``elements``, as (T_min, T_max, phase).

    Each gets a negligible molar volume. Cantera's default density for these
    phases (0.001 kg/m^3) would charge them a huge pressure-volume penalty at
    chamber pressure, and they'd never form.
    """
    out = []
    for sp in ct.Species.list_from_file("nasa_condensed.yaml"):
        if set(sp.composition) <= set(elements):
            d = dict(sp.input_data)
            d["equation-of-state"] = {"model": "constant-volume", "molar-volume": 1e-3}  # m^3/kmol
            phase = ct.Solution(thermo="fixed-stoichiometry", species=[ct.Species.from_dict(d)])
            out.append((sp.thermo.min_temp, sp.thermo.max_temp, phase))
    return tuple(out)


@dataclass
class MultiphaseState:
    """Equilibrium products per kg of propellant."""

    T: float
    P: float
    h: float  # J/kg
    s: float  # J/kg/K
    rho: float  # kg/m^3, with the condensed products taking no volume
    condensed: dict[str, float]  # kg of each condensed species per kg of propellant
    X: dict[str, float] = field(repr=False, default_factory=dict)  # mole fractions over all species, as CEA prints


class Equilibrium:
    """Gas and condensed products of one propellant in chemical equilibrium."""

    def __init__(self, reactants: list[tuple[Reactant, float]]):
        self.b, self.h0 = element_moles(reactants)  # mol of each element per kg, J/kg
        self.elements = tuple(sorted(self.b))
        self.gas = _products_phase(self.elements)
        self.candidates = _condensed_phases(self.elements)
        self._key = None

    def tp(self, T: float, P: float) -> MultiphaseState:
        key = tuple(i for i, (lo, hi, _) in enumerate(self.candidates) if lo <= T <= hi)
        if key != self._key:  # a new set of condensed phases: start again from the gas alone
            self._key = key
            self._phases = [self.candidates[i][2] for i in key]
            self.gas.TPX = T, P, self.b
            self.gas.equilibrate("TP")
            atoms = sum(x * sum(self.gas.species(k).composition.values()) for k, x in enumerate(self.gas.X))
            self._mix = ct.Mixture([(self.gas, sum(self.b.values()) / 1000.0 / atoms)] +
                                   [(ph, 0.0) for ph in self._phases])
        mix = self._mix
        mix.T, mix.P = T, P
        try:
            mix.equilibrate("TP", solver="gibbs", max_steps=2000)
        except ct.CanteraError:
            mix.equilibrate("TP", solver="vcs", max_steps=2000)
        phases = [self.gas] + self._phases
        n = [mix.phase_moles(i) for i in range(len(phases))]
        H = sum(ni * ph.enthalpy_mole for ni, ph in zip(n, phases))
        S = sum(ni * ph.entropy_mole for ni, ph in zip(n, phases))
        total = sum(n)
        X = {sp: n[0] * x / total for sp, x in zip(self.gas.species_names, self.gas.X) if x > 1e-6}
        cond = {}
        for ni, ph in zip(n[1:], self._phases):
            if ni > 0:
                X[ph.species_names[0]] = ni / total
                cond[ph.species_names[0]] = ni * ph.mean_molecular_weight
        rho = P / (n[0] * ct.gas_constant * T)
        return MultiphaseState(T, P, H, S, rho, cond, X)

    def _solve_T(self, prop: str, target: float, P: float, guess: float) -> MultiphaseState:
        """State at P where h or s equals ``target``.

        When the target falls inside a phase change (alumina freezing at 2327 K, say),
        no single temperature hits it: the temperature holds while the phases trade
        places. The state is then a lever-rule mix of the two sides.
        """
        f = lambda T: getattr(self.tp(T, P), prop) - target  # noqa: E731
        lo, hi = 0.9 * guess, 1.1 * guess
        while f(lo) > 0:
            lo *= 0.9
        while f(hi) < 0:
            hi *= 1.1
        T = brentq(f, lo, hi, xtol=1e-3)
        st = self.tp(T, P)
        scale = abs(target) * 1e-6 + 1e-3
        if abs(getattr(st, prop) - target) < scale:
            return st
        a, c = self.tp(T - 0.01, P), self.tp(T + 0.01, P)  # the two sides of the transition
        w = (target - getattr(a, prop)) / (getattr(c, prop) - getattr(a, prop))
        mix = lambda u, v: (1 - w) * u + w * v  # noqa: E731
        cond = {k: mix(a.condensed.get(k, 0.0), c.condensed.get(k, 0.0)) for k in set(a.condensed) | set(c.condensed)}
        return MultiphaseState(T, P, mix(a.h, c.h), mix(a.s, c.s), 1.0 / mix(1 / a.rho, 1 / c.rho), cond)

    def hp(self, P: float, guess: float = 3000.0) -> MultiphaseState:
        """Adiabatic, constant-pressure combustion of the reactants."""
        return self._solve_T("h", self.h0, P, guess)

    def sp(self, s: float, P: float, guess: float) -> MultiphaseState:
        return self._solve_T("s", s, P, guess)


@dataclass
class MultiphaseRocket:
    chamber: MultiphaseState
    throat: MultiphaseState
    exit: MultiphaseState
    cstar: float
    cf: float
    cf_vac: float
    isp: float
    isp_vac: float
    area_ratio: float
    v_exit: float

    @property
    def condensed_fraction(self) -> float:
        """Mass fraction of the chamber products that is condensed."""
        return sum(self.chamber.condensed.values())


def rocket_multiphase(reactants: list[tuple[Reactant, float]], pc: float, area_ratio: float,
                      p_ambient: float = 0.0) -> MultiphaseRocket:
    """Ideal shifting-equilibrium rocket with condensed products, like CEA's 'rocket' problem.

    The particles are assumed to keep up with the gas in speed and temperature,
    which real two-phase flow doesn't quite do, so this is an upper bound.
    The throat is where the mass flux peaks (for an equilibrium mixture with
    condensed products that's where the flow reaches its equilibrium sound speed).
    """
    eq = Equilibrium(reactants)
    ch = eq.hp(pc)
    guess = {"T": ch.T}

    def at(p):
        st = eq.sp(ch.s, p, guess["T"] * (p / pc) ** 0.15)
        return st, np.sqrt(max(2.0 * (ch.h - st.h), 0.0))

    def flux(p):
        st, v = at(p)
        return st.rho * v

    res = minimize_scalar(lambda p: -flux(p), bounds=(0.45 * pc, 0.75 * pc), method="bounded",
                          options={"xatol": 1e-5 * pc})
    th, vt = at(res.x)
    flux_t = th.rho * vt
    pe = brentq(lambda p: flux_t / flux(p) - area_ratio, pc * 0.005 / area_ratio**1.2, res.x * 0.999,
                xtol=1e-8 * pc)
    ex, ve = at(pe)
    cstar = pc / flux_t
    cf_vac = (ve + pe * area_ratio / flux_t) / cstar
    cf = cf_vac - p_ambient * area_ratio / pc
    return MultiphaseRocket(ch, th, ex, cstar, cf, cf_vac, cstar * cf / G0, cstar * cf_vac / G0, area_ratio, ve)
