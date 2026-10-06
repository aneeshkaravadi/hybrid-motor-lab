"""Quasi-steady 0-D solid rocket motor ballistics.

At every instant, mass produced by the burning surface equals mass leaving
through the throat (chamber filling time is milliseconds, far shorter than the
burn):

    rho_p * r(Pc) * A_b  =  Pc * A_t / c*      with  r = a * Pc^n

which gives the classic closed form  Pc = (rho_p * a * A_b * c* / A_t)^(1/(1-n)).
Thrust is F = C_F * Pc * A_t. Ignition transients are not modeled.

``simulate`` assumes the burn rate depends only on Pc. ``simulate_erosive``
adds erosive burning: gas flowing fast along the port scours the burning
surface and makes it burn faster, most at the nozzle end where all the gas from
upstream is flowing past. It uses Mukunda & Paul's correlation ("Universal
behaviour in erosive burning of solid propellants", Combustion and Flame 109,
1997, Eq. 12), which they fit to about 450 data points across composite and
double-base propellants:

    r / r0 = 1 + 0.023 (g^0.8 - g_th^0.8)   for g > g_th = 35, else 1
    g  = g0 (Re0 / 1000)^-0.125,   g0 = G / (rho_p r0),   Re0 = rho_p r0 d / mu

where r0 = a Pc^n is the non-erosive rate, G the mass flux in the port, d the
port's hydraulic diameter and mu the gas viscosity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .grain import Segment
from .nozzle import thrust_coefficient

G0 = 9.80665


@dataclass
class SolidPropellant:
    name: str
    density: float  # kg/m^3
    a: float  # burn-rate coefficient, m/s/Pa^n
    n: float  # pressure exponent
    cstar: float  # m/s, delivered (already includes combustion efficiency)
    gamma: float  # effective ratio of specific heats for the nozzle
    # Combustion-gas viscosity, used only for erosive burning. 8e-5 Pa s is about
    # air's viscosity near 3000 K; the correlation only feels it to the 1/8 power.
    viscosity: float = 8e-5


# Illustrative ammonium-perchlorate composite. Real formulations differ; fit
# a and n to strand-burner or static-fire data before trusting absolute numbers.
EXAMPLE_APCP = SolidPropellant("example APCP", density=1700.0, a=3.5e-5, n=0.35, cstar=1500.0, gamma=1.20)


@dataclass
class SolidResult:
    t: np.ndarray
    pc: np.ndarray
    thrust: np.ndarray
    mdot: np.ndarray
    burn_area: np.ndarray
    web: np.ndarray
    propellant_mass: float

    @property
    def total_impulse(self) -> float:
        return float(np.trapezoid(self.thrust, self.t))

    @property
    def isp(self) -> float:
        return self.total_impulse / (self.propellant_mass * G0)

    @property
    def max_pressure(self) -> float:
        return float(self.pc.max())


def simulate(segments: list[Segment], prop: SolidPropellant, throat_diameter: float, area_ratio: float,
             p_ambient: float = 101325.0, dx: float = 1e-4) -> SolidResult:
    """March in web distance (not time) so thin and thick grains are resolved equally."""
    At = np.pi * throat_diameter**2 / 4
    curves = [s.port.curves(300) for s in segments]
    web_max = max(c[0][-1] for c in curves)
    xs = np.arange(0.0, web_max + dx, dx)
    t, pcs, F, mdots, Abs = [0.0], [], [], [], []
    for i, x in enumerate(xs):
        Ab = sum(s.burn_area(x, c)[0] for s, c in zip(segments, curves))
        if Ab <= 0:
            xs = xs[:i]
            break
        pc = (prop.density * prop.a * Ab * prop.cstar / At) ** (1.0 / (1.0 - prop.n))
        r = prop.a * pc**prop.n
        cf = thrust_coefficient(prop.gamma, area_ratio, pc, p_ambient)
        pcs.append(pc)
        F.append(max(cf * pc * At, 0.0))
        mdots.append(pc * At / prop.cstar)
        Abs.append(Ab)
        if i < len(xs) - 1:
            t.append(t[-1] + dx / r)
    t = np.array(t[: len(pcs)])
    mass = sum(s.propellant_volume() for s in segments) * prop.density
    return SolidResult(t, np.array(pcs), np.array(F), np.array(mdots), np.array(Abs), xs[: len(pcs)], mass)


# ---------------------------------------------------------------- erosive burning

def erosive_ratio(G, r0: float, prop: SolidPropellant, hydraulic_diameter):
    """Erosive burn-rate ratio r / r0 from Mukunda & Paul (1997), Eq. 12."""
    G, d = np.asarray(G, float), np.asarray(hydraulic_diameter, float)
    g0 = G / (prop.density * r0)
    re0 = prop.density * r0 * d / prop.viscosity
    g = g0 * (re0 / 1000.0) ** -0.125
    return np.where(g > 35.0, 1.0 + 0.023 * (np.maximum(g, 35.0) ** 0.8 - 35.0**0.8), 1.0)


@dataclass
class ErosiveResult(SolidResult):
    z: np.ndarray = field(default_factory=lambda: np.zeros(0))  # cell centers from the head end, m
    web_z: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # core web burned, (time, cell)
    peak_flux: np.ndarray = field(default_factory=lambda: np.zeros(0))  # largest port mass flux, kg/m^2/s
    peak_ratio: np.ndarray = field(default_factory=lambda: np.zeros(0))  # largest r / r0 anywhere
    # Peak port mass flux over the throat's, Pc / c*. Above 1 the port itself would
    # choke, the pressure along the port is far from uniform, and the model doesn't apply.
    port_choke_ratio: np.ndarray = field(default_factory=lambda: np.zeros(0))


def simulate_erosive(segments: list[Segment], prop: SolidPropellant, throat_diameter: float, area_ratio: float,
                     p_ambient: float = 101325.0, dx: float = 1e-4, cells_per_segment: int = 40,
                     erosive: bool = True) -> ErosiveResult:
    """Like ``simulate``, but each segment is split into axial cells that burn back at their own rate.

    At each step the chamber pressure is solved so the gas made along the whole
    grain (with each cell's erosive rate set by the flux of the gas made
    upstream of it) equals the nozzle flow. End faces burn at the non-erosive
    rate. With ``erosive=False`` every cell burns at r0, which reproduces
    ``simulate``. Each step burns at most ``dx`` of web in the fastest cell.
    """
    At = np.pi * throat_diameter**2 / 4
    curves = [s.port.curves(300) for s in segments]
    case_area = [np.pi * s.port.diameter**2 / 4 for s in segments]
    dz = [s.length / cells_per_segment for s in segments]
    zc = []
    z0 = 0.0
    for s, d in zip(segments, dz):
        zc.append((np.arange(cells_per_segment) + 0.5) * d)  # local cell centers
        z0 += s.length
    x = [np.zeros(cells_per_segment) for _ in segments]  # core web of each cell
    x_end = np.zeros(len(segments))  # web burned off each uninhibited end face
    t = 0.0
    out = {k: [] for k in ("t", "pc", "F", "mdot", "Ab", "x", "xz", "G", "eta", "choke")}

    def geometry():
        """Per segment: port area, perimeter, active length of each cell, and the two end-face areas."""
        geo = []
        for i, (s, c) in enumerate(zip(segments, curves)):
            xs, A, P = c
            done = x[i] >= xs[-1]  # burned through to the casing
            Ap = np.where(done, case_area[i], np.interp(x[i], xs, A))
            Pp = np.where(done, 0.0, np.interp(x[i], xs, P))
            if s.inhibited_ends:
                active = np.full(cells_per_segment, dz[i])
                faces = (0.0, 0.0)
            else:
                lo, hi = x_end[i], s.length - x_end[i]
                active = np.clip(np.minimum(zc[i] + dz[i] / 2, hi) - np.maximum(zc[i] - dz[i] / 2, lo), 0.0, None)
                live = np.nonzero(active > 0)[0]
                faces = (0.0, 0.0) if live.size == 0 else (case_area[i] - Ap[live[0]], case_area[i] - Ap[live[-1]])
            geo.append((Ap, Pp, active, faces))
        return geo

    def march(pc, geo):
        """Gas made along the grain at chamber pressure pc, head end first."""
        r0 = prop.a * pc**prop.n
        m, rates, G_max, eta_max = 0.0, [], 0.0, 1.0
        for Ap, Pp, active, faces in geo:
            m += prop.density * r0 * faces[0]
            r = np.empty(cells_per_segment)
            for j in range(cells_per_segment):
                G = m / Ap[j]
                eta = float(erosive_ratio(G, r0, prop, 4 * Ap[j] / max(Pp[j], 1e-12))) if erosive else 1.0
                r[j] = r0 * eta
                m += prop.density * r[j] * Pp[j] * active[j]
                if active[j] > 0 and Pp[j] > 0:
                    G_max, eta_max = max(G_max, G), max(eta_max, eta)
            m += prop.density * r0 * faces[1]
            G_max = max(G_max, m / Ap[-1])
            rates.append(r)
        return m, r0, rates, G_max, eta_max

    while True:
        geo = geometry()
        Ab = sum(float(np.sum(Pp * active)) + sum(faces) for _, Pp, active, faces in geo)
        if Ab <= 0:
            break
        pc_lo = (prop.density * prop.a * Ab * prop.cstar / At) ** (1.0 / (1.0 - prop.n))  # no erosion
        pc_hi = pc_lo
        while march(pc_hi, geo)[0] > pc_hi * At / prop.cstar:
            pc_hi *= 1.5
        def balance(p, geo=geo):
            return march(p, geo)[0] - p * At / prop.cstar

        pc = brentq(balance, pc_lo * (1 - 1e-9), pc_hi, xtol=1.0) if erosive else pc_lo
        _, r0, rates, G_max, eta_max = march(pc, geo)
        cf = thrust_coefficient(prop.gamma, area_ratio, pc, p_ambient)
        for k, v in zip(out, (t, pc, max(cf * pc * At, 0.0), pc * At / prop.cstar, Ab,
                              float(np.mean(np.concatenate(x))), np.concatenate(x), G_max, eta_max,
                              G_max * prop.cstar / pc)):
            out[k].append(v)
        dt = dx / max(r0, max(float(r.max()) for r in rates))
        for i, s in enumerate(segments):
            x[i] = x[i] + rates[i] * dt
            if not s.inhibited_ends:
                x_end[i] += r0 * dt
        t += dt
    mass = sum(s.propellant_volume() for s in segments) * prop.density
    z = np.concatenate([zc[i] + sum(s.length for s in segments[:i]) for i in range(len(segments))])
    return ErosiveResult(np.array(out["t"]), np.array(out["pc"]), np.array(out["F"]), np.array(out["mdot"]),
                         np.array(out["Ab"]), np.array(out["x"]), mass, z, np.array(out["xz"]),
                         np.array(out["G"]), np.array(out["eta"]), np.array(out["choke"]))
