"""Grain burn-back for arbitrary port cross-sections.

A burning surface that regresses at the same speed everywhere moves like a
wavefront (Huygens' principle): after burning a web distance x, the port is
every point within distance x of the original port surface. So one distance
field d(x, y) to the initial surface gives the whole burn history:

    A_port(x) = area of {d <= x} inside the casing
    P(x)      = dA_port/dx   (the burning perimeter)

openMotor does the same with fast marching. Here the port boundary is found to
sub-pixel accuracy (marching squares on a 4x supersampled coverage field) and
distances are measured to that curve with a KD-tree, which avoids the
"scalloped" over-estimate of perimeter that a plain pixel distance transform
gives near the start of the burn. Any shape you can describe, including ones
only 3D printing can make, works.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import contourpy
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

InsideFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class PortGeometry:
    """Port cross-section on a square grid covering the casing diameter."""

    diameter: float  # m, grain outer diameter
    port_mask: np.ndarray  # bool, True = initially open (port), sampled at cell centers
    name: str = "custom"
    coverage: np.ndarray | None = None  # fraction of each cell that is port, 0..1
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return self.port_mask.shape[0]

    @property
    def dx(self) -> float:
        return self.diameter / self.n

    def _coords(self):
        c = (np.arange(self.n) + 0.5) * self.dx - self.diameter / 2
        return np.meshgrid(c, c, indexing="ij")

    @property
    def casing_mask(self) -> np.ndarray:
        X, Y = self._coords()
        return X**2 + Y**2 <= (self.diameter / 2) ** 2

    def distance(self) -> np.ndarray:
        """Signed distance (m) from each cell center to the port surface; negative inside the port."""
        if "d" in self._cache:
            return self._cache["d"]
        cov = self.coverage if self.coverage is not None else self.port_mask.astype(float)
        lines = contourpy.contour_generator(z=cov).lines(0.5)
        pts = []
        for ln in lines:  # densify so point spacing is 0.1 cell
            seg = np.diff(ln, axis=0)
            steps = np.maximum(np.ceil(np.hypot(*seg.T) / 0.1).astype(int), 1)
            for p0, s, k in zip(ln[:-1], seg, steps):
                pts.append(p0 + s * (np.arange(k)[:, None] / k))
        pts = np.vstack(pts)  # (x = column j, y = row i) in cell units
        jj, ii = np.meshgrid(np.arange(self.n), np.arange(self.n))
        dist, _ = cKDTree(pts).query(np.column_stack([jj.ravel(), ii.ravel()]), workers=-1)
        d = dist.reshape(self.n, self.n) * self.dx
        d = np.where(cov >= 0.5, -d, d)
        self._cache["d"] = d
        return d

    def web(self) -> float:
        """Largest web distance before the flame reaches the casing everywhere."""
        return float(self.distance()[self.casing_mask].max())

    def port_area(self) -> float:
        cov = self.coverage if self.coverage is not None else self.port_mask.astype(float)
        return float((cov * self.casing_mask).sum() * self.dx**2)

    def curves(self, n_steps: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Web distance x, port area A(x) and burning perimeter P(x), all in SI units."""
        key = ("curves", n_steps)
        if key in self._cache:
            return self._cache[key]
        d = self.distance()[self.casing_mask]
        cell = self.dx**2
        # Each cell is burnt in proportion to how far the front has passed through it.
        xf = np.arange(0.0, self.web() + self.dx, self.dx / 2)
        Af = np.array([np.clip((xi - d) / self.dx + 0.5, 0.0, 1.0).sum() * cell for xi in xf])
        Pf = ndimage.gaussian_filter1d(np.gradient(Af, xf), 1.0, mode="nearest")
        x = np.linspace(0.0, xf[-1], n_steps)
        out = (x, np.interp(x, xf, Af), np.interp(x, xf, Pf))
        self._cache[key] = out
        return out


def from_shape(diameter: float, inside: InsideFn, name: str, n: int = 401, supersample: int = 4) -> PortGeometry:
    """Rasterize ``inside(X, Y) -> bool`` (coordinates in m, origin at the grain axis)."""
    dx = diameter / n
    c = (np.arange(n) + 0.5) * dx - diameter / 2
    X, Y = np.meshgrid(c, c, indexing="ij")
    mask = inside(X, Y)
    offs = (np.arange(supersample) + 0.5) / supersample - 0.5
    cov = np.zeros((n, n))
    for ox in offs:
        for oy in offs:
            cov += inside(X + ox * dx, Y + oy * dx)
    return PortGeometry(diameter, mask, name, cov / supersample**2)


def _polar(X, Y):
    return np.hypot(X, Y), np.arctan2(Y, X)


def tubular(diameter: float, core_diameter: float, n: int = 401) -> PortGeometry:
    return from_shape(diameter, lambda X, Y: np.hypot(X, Y) <= core_diameter / 2, "tubular", n)


def star(diameter: float, points: int, r_outer: float, r_inner: float, n: int = 401) -> PortGeometry:
    """Straight-sided star port with ``points`` tips at r_outer and valleys at r_inner."""
    half = np.pi / points
    tip = np.array([r_outer, 0.0])
    valley = np.array([r_inner * np.cos(half), r_inner * np.sin(half)])
    edge = valley - tip

    def inside(X, Y):
        R, TH = _polar(X, Y)
        phi = np.abs(np.mod(TH, 2 * half) - half)  # angle from the nearest tip
        px, py = R * np.cos(phi), R * np.sin(phi)
        cross = edge[0] * (py - tip[1]) - edge[1] * (px - tip[0])
        return ((cross >= 0) | (R <= r_inner)) & (R <= r_outer)  # origin side of the tip-valley edge

    return from_shape(diameter, inside, f"{points}-point star", n)


def _fins(X, Y, count, start, length, width):
    out = np.zeros(X.shape, bool)
    for k in range(count):
        a = 2 * np.pi * k / count
        u = X * np.cos(a) + Y * np.sin(a)
        v = -X * np.sin(a) + Y * np.cos(a)
        out |= (u >= 0) & (u <= start + length) & (np.abs(v) <= width / 2)
    return out


def finocyl(diameter: float, core_diameter: float, fins: int, fin_length: float, fin_width: float,
            n: int = 401) -> PortGeometry:
    def inside(X, Y):
        return (np.hypot(X, Y) <= core_diameter / 2) | _fins(X, Y, fins, core_diameter / 2, fin_length, fin_width)

    return from_shape(diameter, inside, f"finocyl x{fins}", n)


def moonburner(diameter: float, core_diameter: float, offset: float, n: int = 401) -> PortGeometry:
    return from_shape(diameter, lambda X, Y: np.hypot(X - offset, Y) <= core_diameter / 2, "moonburner", n)


def wagon_wheel(diameter: float, hub_diameter: float, spokes: int, spoke_length: float,
                spoke_width: float, tip_diameter: float, n: int = 401) -> PortGeometry:
    """Spokes with round bulbs at the tips: easy to 3D print, hard to cast."""
    rc = hub_diameter / 2 + spoke_length

    def inside(X, Y):
        m = (np.hypot(X, Y) <= hub_diameter / 2) | _fins(X, Y, spokes, hub_diameter / 2, spoke_length, spoke_width)
        for k in range(spokes):
            a = 2 * np.pi * k / spokes
            m |= np.hypot(X - rc * np.cos(a), Y - rc * np.sin(a)) <= tip_diameter / 2
        return m

    return from_shape(diameter, inside, f"wagon wheel x{spokes}", n)


@dataclass
class Segment:
    """One grain segment: a port shape extruded over a length, with optional burning ends."""

    port: PortGeometry
    length: float  # m
    inhibited_ends: bool = False

    def burn_area(self, x: float, curves=None) -> tuple[float, float]:
        """Burning area (m^2) and port area (m^2) after web x."""
        xs, A, P = curves if curves is not None else self.port.curves()
        if x >= xs[-1]:
            return 0.0, float(np.pi * self.port.diameter**2 / 4)
        A_port = float(np.interp(x, xs, A))
        perim = float(np.interp(x, xs, P))
        if self.inhibited_ends:
            return perim * self.length, A_port
        L = self.length - 2 * x
        if L <= 0:
            return 0.0, A_port
        A_end = np.pi * self.port.diameter**2 / 4 - A_port
        return perim * L + 2 * A_end, A_port

    def propellant_volume(self) -> float:
        return (np.pi * self.port.diameter**2 / 4 - self.port.port_area()) * self.length


def bates_area_exact(D: float, d: float, L: float, x: float) -> float:
    """Closed-form BATES burning area (core + both ends), used to verify the raster method."""
    r = d / 2 + x
    Lx = L - 2 * x
    if r >= D / 2 or Lx <= 0:
        return 0.0
    return 2 * np.pi * r * Lx + 2 * np.pi * ((D / 2) ** 2 - r**2)
