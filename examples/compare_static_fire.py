"""Compare a measured thrust curve against the solid-motor model.

    python examples/compare_static_fire.py data/static_fires/my_motor.csv [--fit]

With --fit, the burn-rate law (a, n) is fitted to the measured curve by least
squares on thrust vs time; use the fitted values to predict another motor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from hml import grain, solid

SHAPES = {
    "tubular": lambda s: grain.tubular(s["diameter_mm"] / 1e3, s["core_mm"] / 1e3),
    "star": lambda s: grain.star(s["diameter_mm"] / 1e3, s["points"], s["r_outer_mm"] / 1e3, s["r_inner_mm"] / 1e3),
    "finocyl": lambda s: grain.finocyl(s["diameter_mm"] / 1e3, s["core_mm"] / 1e3, s["fins"], s["fin_length_mm"] / 1e3,
                                       s["fin_width_mm"] / 1e3),
}


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".eng":
        rows = []
        lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith(";")]
        for ln in lines[1:]:  # first line is the motor header
            parts = ln.split()
            if len(parts) == 2:
                rows.append((float(parts[0]), float(parts[1])))
        data = np.array(rows)
    else:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def build(cfg):
    segs = []
    for s in cfg["segments"]:
        for _ in range(s.get("count", 1)):
            segs.append(grain.Segment(SHAPES[s["shape"]](s), s["length_mm"] / 1e3, s.get("inhibited_ends", False)))
    return segs


def run(cfg, segs, a=None, n=None):
    p = dict(cfg["propellant"])
    if a is not None:
        p["a"], p["n"] = a, n
    prop = solid.SolidPropellant("fit", **p)
    return solid.simulate(segs, prop, cfg["throat_mm"] / 1e3, cfg["area_ratio"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("curve", type=Path)
    ap.add_argument("--fit", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(args.curve.with_suffix(".json").read_text())
    t_m, F_m = read_curve(args.curve)
    segs = build(cfg)
    r = run(cfg, segs)
    label = "model (config a, n)"
    if args.fit:
        def loss(v):
            rr = run(cfg, segs, np.exp(v[0]), v[1])
            return float(np.mean((np.interp(t_m, rr.t, rr.thrust, right=0.0) - F_m) ** 2))
        res = minimize(loss, [np.log(cfg["propellant"]["a"]), cfg["propellant"]["n"]], method="Nelder-Mead")
        a_fit, n_fit = float(np.exp(res.x[0])), float(res.x[1])
        r = run(cfg, segs, a_fit, n_fit)
        label = f"model fit: a = {a_fit:.3e}, n = {n_fit:.3f}"
        print(json.dumps({"a": a_fit, "n": n_fit}))
    I_m = np.trapezoid(F_m, t_m)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_m, F_m, "k", label=f"measured: {I_m:.0f} N s")
    ax.plot(r.t, r.thrust, "C3--", label=f"{label}: {r.total_impulse:.0f} N s")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("thrust (N)")
    ax.legend()
    out = args.curve.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"impulse error {100 * (r.total_impulse - I_m) / I_m:+.1f}%  ->  {out}")


if __name__ == "__main__":
    main()
