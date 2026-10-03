"""Nozzle performance and contour generation.

``thrust_coefficient`` is the ideal-gas, one-gamma C_F. ``bell_contour`` builds
an approximate Rao thrust-optimized bell with the common quadratic-Bezier
method: a circular arc of 1.5*Rt upstream of the throat, an arc of 0.382*Rt
downstream, then a Bezier curve from the inflection angle theta_n to the exit
angle theta_e. theta_n and theta_e normally come from Rao's charts for the
chosen area ratio and length fraction; they are inputs here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


def exit_mach(gamma: float, area_ratio: float) -> float:
    g = gamma

    def f(M):
        return (1 / M) * ((2 / (g + 1)) * (1 + (g - 1) / 2 * M**2)) ** ((g + 1) / (2 * (g - 1))) - area_ratio

    return brentq(f, 1.0001, 50.0)


def thrust_coefficient(gamma: float, area_ratio: float, pc: float, p_ambient: float) -> float:
    g = gamma
    Me = exit_mach(g, area_ratio)
    pe_pc = (1 + (g - 1) / 2 * Me**2) ** (-g / (g - 1))
    momentum = np.sqrt(2 * g**2 / (g - 1) * (2 / (g + 1)) ** ((g + 1) / (g - 1)) * (1 - pe_pc ** ((g - 1) / g)))
    return momentum + (pe_pc - p_ambient / pc) * area_ratio


@dataclass
class Contour:
    x: np.ndarray  # axial position, m (throat at x = 0)
    r: np.ndarray  # inner wall radius, m
    throat_radius: float

    @property
    def area_ratio(self) -> float:
        return (self.r[-1] / self.throat_radius) ** 2

    @property
    def length(self) -> float:
        return float(self.x[-1])


def conical_length(rt: float, area_ratio: float, half_angle_deg: float = 15.0) -> float:
    re = rt * np.sqrt(area_ratio)
    return (re - rt) / np.tan(np.radians(half_angle_deg))


def bell_contour(throat_radius: float, area_ratio: float, theta_n_deg: float = 25.0,
                 theta_e_deg: float = 10.0, length_fraction: float = 0.8,
                 chamber_radius: float | None = None, convergent_half_angle_deg: float = 45.0,
                 n: int = 200) -> Contour:
    """Full inner contour: convergent cone, throat arcs, then the bell."""
    rt = throat_radius
    re = rt * np.sqrt(area_ratio)
    tn, te = np.radians(theta_n_deg), np.radians(theta_e_deg)
    Ln = length_fraction * conical_length(rt, area_ratio)

    # upstream throat arc, radius 1.5 rt, from -convergent angle up to the throat
    tc = np.radians(convergent_half_angle_deg)
    a1 = np.linspace(-np.pi / 2 - tc, -np.pi / 2, n // 4)
    x1 = 1.5 * rt * np.cos(a1)
    y1 = 1.5 * rt * np.sin(a1) + 2.5 * rt

    # downstream arc, radius 0.382 rt, from the throat to theta_n
    a2 = np.linspace(-np.pi / 2, tn - np.pi / 2, n // 4)
    x2 = 0.382 * rt * np.cos(a2)
    y2 = 0.382 * rt * np.sin(a2) + 1.382 * rt

    # Bezier from N (end of arc) to E (exit), control point Q at the tangent-line intersection
    Nx, Ny = x2[-1], y2[-1]
    Ex, Ey = Ln, re
    m1, m2 = np.tan(tn), np.tan(te)
    c1, c2 = Ny - m1 * Nx, Ey - m2 * Ex
    Qx = (c2 - c1) / (m1 - m2)
    Qy = (m1 * c2 - m2 * c1) / (m1 - m2)
    s = np.linspace(0, 1, n)
    xb = (1 - s) ** 2 * Nx + 2 * (1 - s) * s * Qx + s**2 * Ex
    yb = (1 - s) ** 2 * Ny + 2 * (1 - s) * s * Qy + s**2 * Ey

    xs = [x1, x2[1:], xb[1:]]
    ys = [y1, y2[1:], yb[1:]]
    if chamber_radius is not None:
        # straight convergent cone from the chamber wall down to the start of arc 1
        y0, x0 = y1[0], x1[0]
        dx = (chamber_radius - y0) / np.tan(tc)
        xc = np.linspace(x0 - dx, x0, n // 4, endpoint=False)
        yc = y0 + (x0 - xc) * np.tan(tc)
        xs.insert(0, xc)
        ys.insert(0, yc)
    return Contour(np.concatenate(xs), np.concatenate(ys), rt)


def export_nozzle_cad(contour: Contour, wall: float, step_path: str, stl_path: str | None = None,
                      flange_radius: float | None = None, flange_thickness: float = 0.0):
    """Revolve the wall (inner contour offset outward by ``wall``) into a solid; write STEP/STL.

    Inputs are in metres; the CAD is built in millimetres, the unit STEP and STL readers assume.
    """
    import build123d as bd

    mm = 1000.0
    x, r = contour.x * mm, contour.r * mm
    wall *= mm
    flange_radius = flange_radius * mm if flange_radius else None
    flange_thickness *= mm
    # outward normal offset of the inner wall
    dx, dr = np.gradient(x), np.gradient(r)
    norm = np.hypot(dx, dr)
    xo, ro = x - wall * dr / norm, r + wall * dx / norm
    # Smooth splines through ~80 points keep the STEP light and the surface fair.
    idx = np.unique(np.linspace(0, len(x) - 1, 80).astype(int))
    inner = [bd.Vector(x[i], 0, r[i]) for i in idx]
    outer = [bd.Vector(xo[i], 0, ro[i]) for i in idx][::-1]
    with bd.BuildPart() as part:
        with bd.BuildSketch(bd.Plane.XZ):
            with bd.BuildLine(bd.Plane.XZ):
                bd.Spline(*inner)
                bd.Line(inner[-1], outer[0])
                bd.Spline(*outer)
                bd.Line(outer[-1], inner[0])
            bd.make_face()
        bd.revolve(axis=bd.Axis.X)
        if flange_radius:
            with bd.BuildSketch(bd.Plane.YZ.offset(x[0])):
                bd.Circle(flange_radius)
                bd.Circle(r[0], mode=bd.Mode.SUBTRACT)
            bd.extrude(amount=flange_thickness)
    bd.export_step(part.part, step_path)
    if stl_path:
        bd.export_stl(part.part, stl_path, tolerance=0.05, angular_tolerance=0.2)
    return part.part
