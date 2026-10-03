"""Quasi-steady 0-D solid rocket motor ballistics.

At every instant, mass produced by the burning surface equals mass leaving
through the throat (chamber filling time is milliseconds, far shorter than the
burn):

    rho_p * r(Pc) * A_b  =  Pc * A_t / c*      with  r = a * Pc^n

which gives the classic closed form  Pc = (rho_p * a * A_b * c* / A_t)^(1/(1-n)).
Thrust is F = C_F * Pc * A_t. Erosive burning and ignition transients are
not modelled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
