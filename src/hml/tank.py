"""Self-pressurizing nitrous oxide tank and injector.

Nitrous oxide is stored as a saturated liquid under its own vapor, at its vapor
pressure (about 5 MPa at 20 C), so it needs no pressurant gas. As liquid leaves
through the injector, some of what's left boils to fill the space, and the heat
that takes comes out of the whole tank. Colder nitrous has a lower vapor
pressure, so the feed pressure and the oxidizer flow both fall during the burn.

Tank (equilibrium model). The tank holds mass m and internal energy U in a
fixed volume V, with liquid and vapor saturated at one temperature. That
temperature comes from the density m/V and the specific energy U/m, using
CoolProp's implementation of the Lemmon & Span reference equation of state for
N2O (J. Chem. Eng. Data 51, 2006). Liquid leaves from the bottom and carries its
enthalpy out with it, and no heat comes in through the walls:

    dm/dt = -mdot,     dU/dt = -mdot * h_liquid

Injector. Three standard models for the mass flux G from the tank down to the
chamber pressure p2 (mdot = Cd * A * G):

    SPI   single-phase incompressible liquid:  G = sqrt(2 rho_l (p1 - p2))
    HEM   homogeneous equilibrium: the liquid flashes to a liquid-vapor mix that
          expands isentropically to p2,  G = rho_2 sqrt(2 (h1 - h2)).  Below a
          critical p2 the flow chokes and G stops growing.
    Dyer  non-homogeneous non-equilibrium (Dyer et al., AIAA 2007-5702): bubbles
          need time to grow, so the real flow sits between the two,
          G = (k G_SPI + G_HEM) / (1 + k),  k = sqrt((p1 - p2) / (p_v - p2)).
          In a self-pressurized tank p1 = p_v, so k = 1 and it's the plain average.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import CoolProp.CoolProp as CP
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .grain import PortGeometry
from .hybrid import HybridResult, RegressionLaw
from .thermo import N2O, PerformanceTable, Reactant

FLUID = "NitrousOxide"
_AS = CP.AbstractState("HEOS", FLUID)


@dataclass(frozen=True)
class Saturation:
    T: float
    p: float
    rho_l: float
    rho_g: float
    h_l: float  # J/kg, CoolProp's reference state
    h_g: float
    s_l: float  # J/kg/K

    @property
    def u_l(self) -> float:
        return self.h_l - self.p / self.rho_l

    @property
    def u_g(self) -> float:
        return self.h_g - self.p / self.rho_g


def saturation(T: float) -> Saturation:
    _AS.update(CP.QT_INPUTS, 0.0, T)
    p, rho_l, h_l, s_l = _AS.p(), _AS.rhomass(), _AS.hmass(), _AS.smass()
    _AS.update(CP.QT_INPUTS, 1.0, T)
    return Saturation(T, p, rho_l, _AS.rhomass(), h_l, _AS.hmass(), s_l)


@dataclass(frozen=True)
class TankState:
    sat: Saturation
    liquid_mass: float
    vapor_mass: float

    @property
    def T(self) -> float:
        return self.sat.T

    @property
    def p(self) -> float:
        return self.sat.p


class N2OTank:
    """Rigid, adiabatic tank of saturated nitrous oxide."""

    def __init__(self, volume: float, mass: float, temperature: float):
        sat = saturation(temperature)
        x = (volume / mass - 1 / sat.rho_l) / (1 / sat.rho_g - 1 / sat.rho_l)  # vapor mass fraction
        if not 0.0 <= x < 1.0:
            lo, hi = sat.rho_g * volume, sat.rho_l * volume
            raise ValueError(f"{mass} kg won't sit as liquid plus vapor in {volume * 1e3:.2f} L at {temperature} K: "
                             f"it needs between {lo:.2f} and {hi:.2f} kg")
        self.volume = volume
        self.mass = mass
        self.energy = mass * ((1 - x) * sat.u_l + x * sat.u_g)

    def state(self) -> TankState:
        _AS.update(CP.DmassUmass_INPUTS, self.mass / self.volume, self.energy / self.mass)
        x = _AS.Q()
        if not 0.0 <= x <= 1.0:  # CoolProp reports a single-phase state as -1
            raise ValueError("the tank is no longer two-phase (the liquid has run out)")
        sat = saturation(_AS.T())
        return TankState(sat, (1 - x) * self.mass, x * self.mass)

    def liquid_fill(self) -> float:
        """Fraction of the tank volume that is liquid."""
        st = self.state()
        return st.liquid_mass / st.sat.rho_l / self.volume

    def drain_liquid(self, mdot: float, dt: float) -> None:
        h_l = self.state().sat.h_l
        self.mass -= mdot * dt
        self.energy -= mdot * h_l * dt


def _hem_flux(sat: Saturation, p: float) -> float:
    """HEM mass flux of saturated liquid expanding isentropically to p, before any choking."""
    _AS.update(CP.PSmass_INPUTS, p, sat.s_l)
    return _AS.rhomass() * np.sqrt(max(2 * (sat.h_l - _AS.hmass()), 0.0))


@lru_cache(maxsize=8)
def _hem_choke(T: float) -> tuple[float, float]:
    """Downstream pressure where HEM flow from saturated liquid at T chokes, and the choked mass flux.

    The flux rises from zero as p2 drops below the vapor pressure, peaks (near 70%
    of it for nitrous), then falls as the mixture's density drops faster than its
    speed rises. Below that peak the flow is choked and stays at the peak value.
    """
    sat = saturation(T)
    res = minimize_scalar(lambda p: -_hem_flux(sat, p), bounds=(0.1 * sat.p, sat.p), method="bounded",
                          options={"xatol": 1e-7 * sat.p})
    return float(res.x), float(-res.fun)


@dataclass
class Injector:
    area: float  # total orifice area, m^2
    cd: float = 0.75
    model: str = "dyer"

    def mass_flux(self, st: TankState, p_down: float) -> float:
        """Ideal mass flux (kg/m^2/s), before the discharge coefficient."""
        dp = st.p - p_down
        if dp <= 0:
            return 0.0
        g_spi = np.sqrt(2 * st.sat.rho_l * dp)
        if self.model == "spi":
            return float(g_spi)
        p_choke, g_choke = _hem_choke(st.T)
        g_hem = g_choke if p_down <= p_choke else _hem_flux(st.sat, p_down)
        if self.model == "hem":
            return g_hem
        if self.model == "dyer":
            k = np.sqrt(dp / (st.sat.p - p_down))  # 1 for a saturated tank
            return float((k * g_spi + g_hem) / (1 + k))
        raise ValueError(f"unknown injector model {self.model!r}")

    def mass_flow(self, st: TankState, p_down: float) -> float:
        return self.cd * self.area * self.mass_flux(st, p_down)


def liquid_n2o(temperature: float) -> Reactant:
    """Liquid N2O at the tank temperature, on the NASA enthalpy scale that thermo.py uses.

    CoolProp and NASA put the zero of enthalpy in different places, so this takes
    the NASA enthalpy of N2O gas at 298.15 K and adds CoolProp's difference
    between saturated liquid at ``temperature`` and the (near-ideal) gas at
    298.15 K and 100 Pa.
    """
    _AS.update(CP.PT_INPUTS, 100.0, 298.15)
    h_gas_298 = _AS.hmass()
    dh = saturation(temperature).h_l - h_gas_298
    h = N2O.enthalpy_j_per_kg() + dh
    return Reactant(f"N2O(l) at {temperature:.0f} K", dict(N2O.formula), dhf_kj_mol=h * N2O.molar_mass / 1000.0)


# ---------------------------------------------------------------- a hybrid fed from the tank

@dataclass
class BlowdownResult(HybridResult):
    mdot_ox: np.ndarray = field(default_factory=lambda: np.zeros(0))
    p_tank: np.ndarray = field(default_factory=lambda: np.zeros(0))
    T_tank: np.ndarray = field(default_factory=lambda: np.zeros(0))
    liquid_runout_time: float | None = None
    vapor_left: float = 0.0  # kg of vapor still in the tank when the burn ends


def simulate_blowdown(port: PortGeometry, length: float, law: RegressionLaw, table: PerformanceTable,
                      tank: N2OTank, injector: Injector, throat_diameter: float, burn_time: float = 60.0,
                      p_ambient: float = 101325.0, eta_cstar: float = 0.95, dt: float = 0.01) -> BlowdownResult:
    """Like ``hybrid.simulate``, but the oxidizer comes from ``tank`` through ``injector``.

    The injector flow depends on the chamber pressure and the chamber pressure on
    the total flow, so the two are solved together every step. The burn ends when
    the tank runs out of liquid (the vapor phase that follows is left out).
    ``tank`` is drained in place.
    """
    At = np.pi * throat_diameter**2 / 4
    xs, A, P = port.curves(400)
    x = t = 0.0
    out = {k: [] for k in ("t", "pc", "F", "of", "mf", "g", "x", "mo", "pt", "Tt")}
    fuel = ox_total = 0.0
    runout = None
    while t < burn_time and x < xs[-1]:
        st = tank.state()
        Ap, Pp = float(np.interp(x, xs, A)), float(np.interp(x, xs, P))

        def flows(pc):
            mo = injector.mass_flow(st, pc)
            return mo, law.fuel_density * law.a * (mo / Ap) ** law.n * Pp * length

        def residual(pc):
            mo, mf = flows(pc)
            if mo <= 0:
                return pc
            cstar, _ = table.lookup(mo / mf, pc)
            return pc - (mo + mf) * eta_cstar * cstar / At

        if residual(p_ambient) >= 0:
            raise ValueError("the injector can't supply enough flow to pressurize this chamber")
        pc = brentq(residual, p_ambient, st.p, xtol=10.0)
        mo, mf = flows(pc)
        if mo * dt >= st.liquid_mass:
            runout = t
            break
        _, cf_vac = table.lookup(mo / mf, pc)
        cf = cf_vac - p_ambient * table.area_ratio / pc
        for k, v in zip(out, (t, pc, max(cf * pc * At, 0.0), mo / mf, mf, mo / Ap, x, mo, st.p, st.T)):
            out[k].append(v)
        fuel += mf * dt
        ox_total += mo * dt
        tank.drain_liquid(mo, dt)
        x += law.a * (mo / Ap) ** law.n * dt
        t += dt
    arr = {k: np.array(v) for k, v in out.items()}
    return BlowdownResult(arr["t"], arr["pc"], arr["F"], arr["of"], arr["mf"], arr["g"], arr["x"], fuel, ox_total,
                          arr["mo"], arr["pt"], arr["Tt"], runout, tank.state().vapor_mass)
