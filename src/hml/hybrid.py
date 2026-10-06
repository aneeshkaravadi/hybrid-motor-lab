"""Quasi-steady hybrid rocket ballistics with a space-averaged regression law.

The solid fuel regresses at  r_dot = a * G_ox^n,  where G_ox = m_dot_ox / A_port
is the oxidizer mass flux through the port. Fuel mass flow is
m_dot_f = rho_f * r_dot * P * L (burning perimeter P, grain length L).

Because G_ox falls as the port opens, O/F drifts during the burn. With a
circular port, m_dot_f is proportional to r^(1-2n): for n > 0.5 the fuel flow
drops and O/F rises over time. Shaped ports, which 3D printing makes
practical, front-load fuel flow (more thrust early) but, for n > 0.5, make the
drift worse. ``throttle_schedule`` and ``graded_fuel_profile`` compute the two
remedies that do hold O/F flat.

The chamber pressure solves  Pc = (m_dot_ox + m_dot_f) * eta_c* * c*(O/F, Pc) / A_t.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .grain import PortGeometry
from .thermo import PerformanceTable

G0 = 9.80665


@dataclass
class RegressionLaw:
    name: str
    a: float  # m/s per (kg/m^2/s)^n
    n: float
    fuel_density: float  # kg/m^3
    source: str = ""

    @classmethod
    def from_cgs(cls, name, a_mm_s, n, fuel_density, source=""):
        """Convert the common literature form r[mm/s] = a * G[g/cm^2/s]^n to SI."""
        # G[g/cm^2/s] = G[kg/m^2/s] / 10
        return cls(name, a_mm_s * 1e-3 * 10.0 ** (-n), n, fuel_density, source)


# Karabeyoglu et al., "Scale-up tests of high regression rate paraffin-based
# hybrid rocket fuels", J. Propulsion and Power 20(6), 2004 (paraffin/GOX).
PARAFFIN_GOX = RegressionLaw.from_cgs("paraffin / GOX", 0.488, 0.62, 924.5,
                                      "Karabeyoglu et al., JPP 20(6), 2004")


@dataclass
class HybridResult:
    t: np.ndarray
    pc: np.ndarray
    thrust: np.ndarray
    of: np.ndarray
    mdot_fuel: np.ndarray
    g_ox: np.ndarray
    web: np.ndarray
    fuel_mass: float
    ox_mass: float

    @property
    def total_impulse(self) -> float:
        return float(np.trapezoid(self.thrust, self.t))

    @property
    def isp(self) -> float:
        return self.total_impulse / ((self.fuel_mass + self.ox_mass) * G0)

    def of_spread(self) -> float:
        """(max - min) / mean of O/F over the burn: a single number for O/F drift."""
        return float((self.of.max() - self.of.min()) / self.of.mean())


def simulate(port: PortGeometry, length: float, law: RegressionLaw, table: PerformanceTable,
             mdot_ox: float | Callable[[float], float], throat_diameter: float, burn_time: float,
             p_ambient: float = 101325.0, eta_cstar: float = 0.95, dt: float = 0.01,
             a_scale: Callable[[float], float] | None = None) -> HybridResult:
    """Time-march the burn.

    ``mdot_ox`` may be a constant or a function of web burned x (a throttle schedule).
    ``a_scale(x)`` multiplies the regression coefficient at web x: a fuel whose
    composition changes with radius, which only additive manufacturing can make.
    """
    At = np.pi * throat_diameter**2 / 4
    xs, A, P = port.curves(400)
    ox = mdot_ox if callable(mdot_ox) else (lambda _x, m=mdot_ox: m)
    x, t = 0.0, 0.0
    pc = 2e6
    out = {k: [] for k in ("t", "pc", "F", "of", "mf", "g", "x")}
    fuel = ox_total = 0.0
    while t < burn_time and x < xs[-1]:
        Ap, Pp = float(np.interp(x, xs, A)), float(np.interp(x, xs, P))
        mo = ox(x)
        g = mo / Ap
        rdot = law.a * (a_scale(x) if a_scale else 1.0) * g**law.n
        mf = law.fuel_density * rdot * Pp * length
        of = mo / mf
        for _ in range(20):  # fixed point on Pc; c* depends weakly on pressure
            cstar, cf_vac = table.lookup(of, pc)
            pc_new = (mo + mf) * eta_cstar * cstar / At
            if abs(pc_new - pc) < 1.0:
                break
            pc = pc_new
        cf = cf_vac - p_ambient * table.area_ratio / pc
        for k, v in zip(out, (t, pc, max(cf * pc * At, 0.0), of, mf, g, x)):
            out[k].append(v)
        fuel += mf * dt
        ox_total += mo * dt
        x += rdot * dt
        t += dt
    return HybridResult(np.array(out["t"]), np.array(out["pc"]), np.array(out["F"]), np.array(out["of"]),
                        np.array(out["mf"]), np.array(out["g"]), np.array(out["x"]), fuel, ox_total)


# ---------------------------------------------------------------- regression along the port
#
# The averaged law above uses one G_ox for the whole grain. Really, every bit of
# fuel that burns near the head end flows down the port too, so the mass flux,
# and with it the regression rate, grows toward the aft end. Marching along the
# port with the total flux G = mdot(z) / A(z):
#
#   d(mdot)/dz = rho_f a (mdot / A)^n P
#
# Inside one axial cell A and P are constant, so this integrates exactly:
#
#   mdot_out^(1-n) = mdot_in^(1-n) + (1-n) rho_f a P A^(-n) dz
#
# and down the whole port mdot^(1-n) is just a cumulative sum.

@dataclass
class AxialHybridResult(HybridResult):
    z: np.ndarray = field(default_factory=lambda: np.zeros(0))  # cell centers, m from the head end
    web_z: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # web burned, shape (time, cell)
    burn_through_time: float | None = None  # when the aft end first reached the casing, if it did


def total_flux_law(law: RegressionLaw, port: PortGeometry, length: float, mdot_ox: float) -> RegressionLaw:
    """Total-flux coefficient that gives the same fuel flow at ignition as ``law``.

    Published laws like PARAFFIN_GOX are fits against the space-averaged oxidizer
    flux. A total-flux law with the same n needs a smaller a, chosen here so the
    two models start the burn with the same fuel flow (the port is uniform at
    ignition, so the closed form above applies along the whole length).
    """
    xs, A, P = port.curves(400)
    A0, P0, n = float(A[0]), float(P[0]), law.n
    mf = law.fuel_density * law.a * (mdot_ox / A0) ** n * P0 * length
    a = ((mdot_ox + mf) ** (1 - n) - mdot_ox ** (1 - n)) / ((1 - n) * law.fuel_density * P0 * A0 ** (-n) * length)
    return RegressionLaw(f"{law.name} (total flux)", a, n, law.fuel_density, law.source)


def fuel_added_per_cell(mdot_ox: float, A: np.ndarray, P: np.ndarray, a: np.ndarray, law: RegressionLaw,
                        dz: float, flux: str = "total") -> np.ndarray:
    """Fuel mass flow added in each axial cell, head end first, for port areas A and perimeters P."""
    n, rho = law.n, law.fuel_density
    if flux == "oxidizer":  # the averaged law applied locally: no feedback from upstream fuel
        return rho * a * (mdot_ox / A) ** n * P * dz
    if flux != "total":
        raise ValueError(f"flux must be 'total' or 'oxidizer', not {flux!r}")
    m = (mdot_ox ** (1 - n) + np.cumsum((1 - n) * rho * a * P * A ** (-n) * dz)) ** (1 / (1 - n))
    return np.diff(np.r_[mdot_ox, m])


def simulate_axial(port: PortGeometry, length: float, law: RegressionLaw, table: PerformanceTable,
                   mdot_ox: float | Callable[[float], float], throat_diameter: float, burn_time: float,
                   p_ambient: float = 101325.0, eta_cstar: float = 0.95, dt: float = 0.01, cells: int = 40,
                   flux: str = "total", a_scale: Callable[[float], float] | None = None) -> AxialHybridResult:
    """Like ``simulate``, but each of ``cells`` axial slices of the grain regresses at its own rate.

    With ``flux="total"`` the regression follows the local total mass flux (use a
    law from ``total_flux_law``). With ``flux="oxidizer"`` every slice sees the
    same oxidizer flux, which reproduces ``simulate`` exactly. A throttle
    schedule ``mdot_ox(x)`` and ``a_scale(x)`` are evaluated at the mean web.
    The burn stops early if the aft end reaches the casing.
    """
    At = np.pi * throat_diameter**2 / 4
    xs, A_of_x, P_of_x = port.curves(400)
    ox = mdot_ox if callable(mdot_ox) else (lambda _x, m=mdot_ox: m)
    dz = length / cells
    z = (np.arange(cells) + 0.5) * dz
    x = np.zeros(cells)
    t, pc = 0.0, 2e6
    out = {k: [] for k in ("t", "pc", "F", "of", "mf", "g", "x", "xz")}
    fuel = ox_total = 0.0
    burn_through = None
    while t < burn_time:
        if x.max() >= xs[-1]:
            burn_through = t
            break
        A, P = np.interp(x, xs, A_of_x), np.interp(x, xs, P_of_x)
        mo = ox(float(x.mean()))
        a = law.a * (np.array([a_scale(v) for v in x]) if a_scale else np.ones(cells))
        added = fuel_added_per_cell(mo, A, P, a, law, dz, flux)
        mf = float(added.sum())
        of = mo / mf
        for _ in range(20):  # fixed point on Pc, as in simulate()
            cstar, cf_vac = table.lookup(of, pc)
            pc_new = (mo + mf) * eta_cstar * cstar / At
            if abs(pc_new - pc) < 1.0:
                break
            pc = pc_new
        cf = cf_vac - p_ambient * table.area_ratio / pc
        for k, v in zip(out, (t, pc, max(cf * pc * At, 0.0), of, mf, float(np.mean(mo / A)), float(x.mean()), x.copy())):
            out[k].append(v)
        fuel += mf * dt
        ox_total += mo * dt
        x = x + added / (law.fuel_density * P * dz) * dt  # each cell's regression rate
        t += dt
    return AxialHybridResult(np.array(out["t"]), np.array(out["pc"]), np.array(out["F"]), np.array(out["of"]),
                             np.array(out["mf"]), np.array(out["g"]), np.array(out["x"]), fuel, ox_total,
                             z, np.array(out["xz"]), burn_through)


# ---------------------------------------------------------------- holding O/F constant
#
#   O/F = m_ox / (rho_f a G^n P L),  G = m_ox / A   =>   O/F = m_ox^(1-n) A^n / (rho_f a P L)
#
# Any port that grows self-similarly has P proportional to A^(1/2), so for n > 1/2
# O/F always rises during the burn. Port shape alone cannot fix that at fixed m_ox.
# Two knobs can: throttle the oxidizer, or change the fuel's regression rate with radius.

def throttle_schedule(port: PortGeometry, length: float, law: RegressionLaw, target_of: float):
    """Oxidizer flow as a function of web x that holds O/F at ``target_of``."""
    xs, A, P = port.curves(400)

    def mdot_ox(x):
        Ap, Pp = np.interp(x, xs, A), np.interp(x, xs, P)
        return float((target_of * law.fuel_density * law.a * Pp * length / Ap**law.n) ** (1.0 / (1.0 - law.n)))

    return mdot_ox


def graded_fuel_profile(port: PortGeometry, length: float, law: RegressionLaw, mdot_ox: float, target_of: float):
    """Regression-coefficient multiplier a(x)/a0 that holds O/F at ``target_of`` at fixed oxidizer flow."""
    xs, A, P = port.curves(400)

    def a_scale(x):
        Ap, Pp = np.interp(x, xs, A), np.interp(x, xs, P)
        return float(mdot_ox ** (1 - law.n) * Ap**law.n / (target_of * law.fuel_density * law.a * Pp * length))

    return a_scale
