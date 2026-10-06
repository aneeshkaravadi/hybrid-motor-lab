"""Regenerate every figure and number quoted in the README.

    python examples/make_figures.py

Takes about a minute (most of it is building the Cantera performance table).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hml import grain, hybrid, rde, solid, tank, thermo
from hml.nozzle import bell_contour, export_nozzle_cad

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
results: dict = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


# ------------------------------------------------------------------ 1. port gallery + burn-back
D = 0.075
ports = [
    grain.tubular(D, 0.025),
    grain.star(D, 6, 0.025, 0.012),
    grain.finocyl(D, 0.022, 5, 0.012, 0.004),
    grain.wagon_wheel(D, 0.016, 6, 0.012, 0.003, 0.008),
]
fig, axes = plt.subplots(2, 4, figsize=(12, 5.6))
for ax, p in zip(axes[0], ports):
    d = np.where(p.casing_mask, p.distance(), np.nan)
    ax.imshow(np.where(p.port_mask, np.nan, d) * 1000, cmap="magma_r", origin="lower")
    ax.contour(d * 1000, levels=np.linspace(2, p.web() * 1000, 6), colors="w", linewidths=0.6)
    ax.set_title(p.name)
    ax.set_xticks([]), ax.set_yticks([])
for ax, p in zip(axes[1], ports):
    x, A, P = p.curves(300)
    ax.plot(x * 1000, P * 1000, color="C3")
    ax.set_xlabel("web burned (mm)")
    ax.set_ylabel("burning perimeter (mm)")
fig.suptitle("Burn-back of four port shapes (white lines: flame front every few mm)")
save(fig, "port_burnback.png")

# ------------------------------------------------------------------ 2. solid thrust shaping
prop = solid.EXAMPLE_APCP
cases = {
    "4x BATES, open ends (roughly neutral)": [grain.Segment(grain.tubular(0.054, 0.020), 0.07) for _ in range(4)],
    "tubular, inhibited ends (progressive)": [grain.Segment(grain.tubular(0.054, 0.020), 0.28, True)],
    "6-point star (fast, then sliver tail-off)": [grain.Segment(grain.star(0.054, 6, 0.019, 0.009), 0.28, True)],
    # The narrow sustainer goes at the head end so it only carries its own gas. At the
    # nozzle end, all the boost grain's gas would have to squeeze through a port smaller
    # than the throat (see the erosive burning section below).
    "finocyl boost + tubular sustain": [grain.Segment(grain.tubular(0.054, 0.010), 0.16, True),
                                        grain.Segment(grain.finocyl(0.054, 0.030, 6, 0.008, 0.003), 0.12, True)],
}
fig, ax = plt.subplots(figsize=(7.5, 4.2))
results["solid"] = {}
for label, segs in cases.items():
    r = solid.simulate(segs, prop, throat_diameter=0.015 if "sustain" in label else 0.017, area_ratio=6.0)
    ax.plot(r.t, r.thrust, label=f"{label}: {r.total_impulse:.0f} N s")
    results["solid"][label] = {"impulse_Ns": round(r.total_impulse, 1), "burn_s": round(float(r.t[-1]), 2),
                               "max_pc_MPa": round(r.max_pressure / 1e6, 2), "isp_s": round(r.isp, 1)}
ax.set_xlabel("time (s)")
ax.set_ylabel("thrust (N)")
ax.set_title("Same 54 mm motor case, four grain designs (example APCP)")
ax.legend(fontsize=8)
save(fig, "solid_thrust_shaping.png")

# Erosive burning: fast gas along the port makes the propellant burn faster, most at the nozzle end.
bates = cases["4x BATES, open ends (roughly neutral)"]
plain = solid.simulate(bates, prop, 0.017, 6.0)
ero = solid.simulate_erosive(bates, prop, 0.017, 6.0)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
axes[0].plot(plain.t, plain.pc / 1e6, label="burn rate from pressure only")
axes[0].plot(ero.t, ero.pc / 1e6, label="with erosive burning")
axes[0].set_xlabel("time (s)")
axes[0].set_ylabel("chamber pressure (MPa)")
axes[0].set_title("4x BATES, 20 mm core, 17 mm throat")
axes[0].legend(fontsize=8)
for k, ts in enumerate((0.0, 0.5, 1.0)):
    i = int(np.searchsorted(ero.t, ts))
    rate = np.diff(ero.web_z[i:i + 2], axis=0)[0] / np.diff(ero.t[i:i + 2])[0]
    axes[1].plot(ero.z * 100, rate / (prop.a * ero.pc[i] ** prop.n), ".", ms=3, color=f"C{k}", label=f"t = {ts:.1f} s")
axes[1].set_xlabel("distance from the head end (cm)")
axes[1].set_ylabel("burn rate / pressure-only rate")
axes[1].set_title("only the grains near the nozzle erode")
axes[1].legend(fontsize=8)
sweep_ero = []
for dc in (0.018, 0.020, 0.022, 0.024, 0.026, 0.028, 0.030):
    segs = [grain.Segment(grain.tubular(0.054, dc), 0.07) for _ in range(4)]
    a, b = solid.simulate(segs, prop, 0.017, 6.0), solid.simulate_erosive(segs, prop, 0.017, 6.0)
    sweep_ero.append(((dc / 0.017) ** 2, 100 * (b.pc[0] / a.pc[0] - 1), 100 * (b.max_pressure / a.max_pressure - 1)))
sweep_ero = np.array(sweep_ero)
axes[2].plot(sweep_ero[:, 0], sweep_ero[:, 1], "o-", label="pressure at ignition")
axes[2].plot(sweep_ero[:, 0], sweep_ero[:, 2], "s-", label="peak pressure")
axes[2].set_xlabel("port-to-throat area ratio (core diameter 18-30 mm)")
axes[2].set_ylabel("rise from erosive burning (%)")
axes[2].set_title("why ports should be 2x the throat")
axes[2].legend(fontsize=8)
fig.suptitle("Erosive burning (Mukunda & Paul's correlation): gas from upstream scours the grains near the nozzle")
save(fig, "erosive_burning.png")
boost_first = [cases["finocyl boost + tubular sustain"][1], cases["finocyl boost + tubular sustain"][0]]
choked = solid.simulate_erosive(boost_first, prop, 0.015, 6.0)
results["erosive"] = {
    "bates_p0_MPa": [round(float(plain.pc[0]) / 1e6, 2), round(float(ero.pc[0]) / 1e6, 2)],
    "bates_pmax_MPa": [round(plain.max_pressure / 1e6, 2), round(ero.max_pressure / 1e6, 2)],
    "bates_peak_rate_ratio": round(float(ero.peak_ratio[0]), 2),
    "bates_port_flux_over_throat": round(float(ero.port_choke_ratio.max()), 2),
    "sweep_port_to_throat": [round(v, 2) for v in sweep_ero[:, 0]],
    "sweep_p0_rise_pct": [round(v, 1) for v in sweep_ero[:, 1]],
    "sweep_pmax_rise_pct": [round(v, 1) for v in sweep_ero[:, 2]],
    "boost_first_port_flux_over_throat": round(float(choked.port_choke_ratio[0]), 2),
    "boost_first_peak_port_flux": round(float(choked.peak_flux[0])),
    "throat_flux_at_ignition": round(float(choked.pc[0]) / prop.cstar),
}

# ------------------------------------------------------------------ 3. hybrid thermochemistry + O/F drift
of_grid = np.linspace(1.0, 4.0, 16)
pc_grid = np.array([0.5e6, 1e6, 2e6, 3e6, 5e6])
table = thermo.build_table(thermo.O2, thermo.PARAFFIN, of_grid, pc_grid, area_ratio=5.0)
fig, ax = plt.subplots(figsize=(6.5, 4))
for j, pc in enumerate(pc_grid):
    ax.plot(of_grid, table.cstar[:, j], label=f"Pc = {pc / 1e6:.1f} MPa")
ax.set_xlabel("O/F (mass)")
ax.set_ylabel("c* (m/s)")
ax.set_title("GOX / paraffin characteristic velocity (Cantera equilibrium)")
ax.legend(fontsize=8)
save(fig, "gox_paraffin_cstar.png")
i_best = int(np.argmax(table.cstar[:, 2]))
results["gox_paraffin_best_of_2MPa"] = {"of": float(of_grid[i_best]), "cstar": round(float(table.cstar[i_best, 2]), 1)}

law = hybrid.PARAFFIN_GOX
L, Dg, mdot_ox, dt_throat = 0.40, 0.10, 0.50, 0.030
designs = {
    "tubular 35 mm": grain.tubular(Dg, 0.035),
    "6-point star": grain.star(Dg, 6, 0.030, 0.012),
    "5-fin finocyl": grain.finocyl(Dg, 0.026, 5, 0.012, 0.004),
    "6-spoke wagon wheel": grain.wagon_wheel(Dg, 0.022, 6, 0.014, 0.004, 0.010),
}
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
results["hybrid"] = {}
for label, port in designs.items():
    r = hybrid.simulate(port, L, law, table, mdot_ox, dt_throat, burn_time=8.0)
    axes[0].plot(r.t, r.of, label=label)
    axes[1].plot(r.t, r.thrust, label=label)
    results["hybrid"][label] = {"of_start": round(float(r.of[0]), 2), "of_end": round(float(r.of[-1]), 2),
                                "of_spread": round(r.of_spread(), 3), "impulse_Ns": round(r.total_impulse, 0),
                                "isp_s": round(r.isp, 1), "fuel_kg": round(r.fuel_mass, 2)}
axes[0].axhline(of_grid[i_best], color="k", ls=":", lw=1, label="peak-c* O/F at 2 MPa")
axes[0].set_ylabel("O/F")
axes[1].set_ylabel("thrust (N)")
for ax in axes:
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=8)
fig.suptitle(f"GOX/paraffin hybrid, {mdot_ox} kg/s oxidizer, {L * 100:.0f} cm grain: port shape sets the O/F history")
save(fig, "hybrid_of_drift.png")

# Sweep star geometry to find the flattest O/F history: the kind of design study
# that only matters once the port can be printed instead of cast.
sweep = []
for points in (4, 5, 6, 7, 8):
    for depth in (0.006, 0.010, 0.014, 0.018):
        port = grain.star(Dg, points, 0.012 + depth, 0.012, n=301)
        r = hybrid.simulate(port, L, law, table, mdot_ox, dt_throat, burn_time=8.0, dt=0.02)
        sweep.append((points, depth, r.of_spread(), float(r.of.mean())))
sweep = np.array(sweep)
best = sweep[np.argmin(sweep[:, 2])]
results["star_sweep_best"] = {"points": int(best[0]), "depth_mm": best[1] * 1000, "of_spread": round(best[2], 3),
                              "mean_of": round(best[3], 2)}
fig, ax = plt.subplots(figsize=(6.5, 4))
for depth in np.unique(sweep[:, 1]):
    m = sweep[:, 1] == depth
    ax.plot(sweep[m, 0], sweep[m, 2] * 100, "o-", label=f"arm depth {depth * 1000:.0f} mm")
ax.axhline(results["hybrid"]["tubular 35 mm"]["of_spread"] * 100, color="k", ls="--", lw=1, label="tubular baseline")
ax.set_xlabel("star points")
ax.set_ylabel("O/F drift over burn, (max-min)/mean (%)")
ax.set_title("Star sweep: no star beats the plain tube on O/F drift (n = 0.62)")
ax.legend(fontsize=8)
save(fig, "star_sweep.png")

# Remedies: throttle the oxidizer, or grade the fuel's regression rate with radius.
base_port = designs["tubular 35 mm"]
base = hybrid.simulate(base_port, L, law, table, mdot_ox, dt_throat, burn_time=8.0)
target = float(base.of.mean())
throttle = hybrid.throttle_schedule(base_port, L, law, target)
graded = hybrid.graded_fuel_profile(base_port, L, law, mdot_ox, target)
r_thr = hybrid.simulate(base_port, L, law, table, throttle, dt_throat, burn_time=8.0)
r_grd = hybrid.simulate(base_port, L, law, table, mdot_ox, dt_throat, burn_time=8.0, a_scale=graded)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for r, lab in ((base, "fixed oxidizer, uniform fuel"), (r_thr, "oxidizer throttle schedule"), (r_grd, "radially graded printed fuel")):
    axes[0].plot(r.t, r.of, label=f"{lab} (drift {r.of_spread() * 100:.1f}%)")
xw = np.linspace(0, base.web[-1], 100)
axes[1].plot(xw * 1000, [throttle(v) for v in xw], color="C1")
axes[1].set_xlabel("web burned (mm)")
axes[1].set_ylabel("oxidizer flow (kg/s)")
axes[1].set_title("throttle schedule")
axes[2].plot(xw * 1000, [graded(v) for v in xw], color="C2")
axes[2].set_xlabel("web burned (mm)")
axes[2].set_ylabel("regression coefficient a(x) / a0")
axes[2].set_title("fuel grading needed")
axes[0].set_xlabel("time (s)")
axes[0].set_ylabel("O/F")
axes[0].legend(fontsize=7)
fig.suptitle("Holding O/F flat: what port shape cannot do, throttling or graded fuel can")
save(fig, "of_remedies.png")
results["remedies"] = {"target_of": round(target, 2),
                       "baseline_drift": round(base.of_spread(), 3),
                       "throttle_drift": round(r_thr.of_spread(), 4),
                       "graded_drift": round(r_grd.of_spread(), 4),
                       "throttle_range_kg_s": [round(throttle(0.0), 3), round(throttle(float(base.web[-1])), 3)],
                       "grading_range": [round(graded(0.0), 3), round(graded(float(base.web[-1])), 3)]}

# Regression along the port: the averaged law burns every slice the same, but
# fuel from the head end adds to the flux further down, so the aft end opens faster.
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
results["axial_regression"] = {}
for Lg in (0.2, 0.4, 0.8):
    tl = hybrid.total_flux_law(law, base_port, Lg, mdot_ox)
    avg = hybrid.simulate(base_port, Lg, law, table, mdot_ox, dt_throat, burn_time=8.0)
    ax_r = hybrid.simulate_axial(base_port, Lg, tl, table, mdot_ox, dt_throat, burn_time=8.0)
    if Lg == L:
        for k, ts in enumerate((2.0, 4.0, 6.0, 8.0 - 0.01)):
            i = int(np.searchsorted(ax_r.t, ts))
            axes[0].plot(ax_r.z * 100, ax_r.web_z[i] * 1000, color=f"C{k}", label=f"t = {ts:.0f} s")
            axes[0].axhline(avg.web[i] * 1000, color=f"C{k}", ls=":", lw=1)
    axes[1].plot(ax_r.z / Lg, ax_r.web_z[-1] * 1000, label=f"{Lg * 100:.0f} cm grain")
    results["axial_regression"][f"{Lg * 100:.0f}cm"] = {
        "web_head_mm": round(float(ax_r.web_z[-1, 0]) * 1000, 1), "web_aft_mm": round(float(ax_r.web_z[-1, -1]) * 1000, 1),
        "web_averaged_model_mm": round(float(avg.web[-1]) * 1000, 1),
        "aft_vs_head_rate_at_ignition": round(float(ax_r.web_z[1, -1] / ax_r.web_z[1, 0]), 3),
        "total_flux_a_over_averaged_a": round(tl.a / law.a, 3)}
axes[1].axhline(avg.web[-1] * 1000, color="k", ls=":", lw=1, label="averaged law (any length)")
axes[0].set_xlabel("distance from the head end (cm)")
axes[0].set_ylabel("web burned (mm)")
axes[0].set_title(f"{L * 100:.0f} cm grain (dotted: averaged law)")
axes[0].legend(fontsize=8)
axes[1].set_xlabel("position along the grain, head (0) to aft (1)")
axes[1].set_ylabel("web burned after 8 s (mm)")
axes[1].set_title("longer grains open more unevenly")
axes[1].legend(fontsize=8)
fig.suptitle("Fuel burned near the head end adds to the flux downstream, so the aft end regresses faster")
save(fig, "axial_regression.png")

# Nitrous oxide from a self-pressurizing tank instead of a constant oxidizer flow.
T_tank0 = 293.15
n2o_table = thermo.build_table(tank.liquid_n2o(T_tank0), thermo.PARAFFIN, np.linspace(2.0, 12.0, 11),
                               np.array([0.5e6, 1e6, 2e6, 3e6, 4e6]), area_ratio=5.0)
n2o_port, n2o_law, L_n2o, throat_n2o = grain.tubular(0.10, 0.035), hybrid.PARAFFIN_N2O, 0.20, 0.024
inj = tank.Injector(area=1.4e-5, cd=0.75, model="dyer")
tk = tank.N2OTank(volume=0.005, mass=3.5, temperature=T_tank0)
fill = tk.liquid_fill()
bd = tank.simulate_blowdown(n2o_port, L_n2o, n2o_law, n2o_table, tk, inj, throat_n2o)
cf_run = hybrid.simulate(n2o_port, L_n2o, n2o_law, n2o_table, float(bd.mdot_ox[0]), throat_n2o, burn_time=float(bd.t[-1]) + 0.01)
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
ax = axes[0, 0]
ax.plot(bd.t, bd.p_tank / 1e6, label="tank")
ax.plot(bd.t, bd.pc / 1e6, label="chamber")
ax.set_ylabel("pressure (MPa)")
ax2 = ax.twinx()
ax2.plot(bd.t, bd.T_tank - 273.15, "C3--", lw=1)
ax2.set_ylabel("tank temperature (C, dashed)", color="C3")
ax.set_title("the tank cools as liquid boils to fill the space")
ax.legend(fontsize=8, loc="lower left")
ax = axes[0, 1]
ax.plot(bd.t, bd.of, label=f"tank-fed (drift {bd.of_spread() * 100:.1f}%)")
ax.plot(cf_run.t, cf_run.of, "--", label=f"constant {bd.mdot_ox[0]:.2f} kg/s (drift {cf_run.of_spread() * 100:.1f}%)")
ax.set_ylabel("O/F")
ax.set_title("falling flow cancels most of the port's O/F rise")
ax.legend(fontsize=8)
ax = axes[1, 0]
ax.plot(bd.t, bd.thrust, label="tank-fed")
ax.plot(cf_run.t, cf_run.thrust, "--", label="constant oxidizer flow")
ax.set_ylabel("thrust (N)")
ax.set_xlabel("time (s)")
ax.set_title("thrust follows the tank pressure down")
ax.legend(fontsize=8)
ax = axes[1, 1]
st0 = tank.N2OTank(0.005, 3.5, T_tank0).state()
pcs = np.linspace(0.2e6, 0.98 * st0.p, 200)
for model, ls in (("spi", "--"), ("hem", ":"), ("dyer", "-")):
    m = tank.Injector(1.0, 1.0, model)
    ax.plot(pcs / 1e6, [m.mass_flux(st0, p) / 1000 for p in pcs], ls, label=model.upper() if model != "dyer" else "Dyer (used)")
ax.set_xlabel("chamber pressure (MPa)")
ax.set_ylabel("ideal mass flux (t/m^2/s)")
ax.set_title(f"injector models, saturated N2O at {T_tank0 - 273.15:.0f} C")
ax.legend(fontsize=8)
for a in axes[0]:
    a.set_xlabel("time (s)")
fig.suptitle(f"N2O/paraffin hybrid fed from a {tk.volume * 1e3:.0f} L tank of 3.5 kg nitrous ({fill * 100:.0f}% liquid at 20 C)")
save(fig, "n2o_blowdown.png")
spi_dyer = tank.Injector(1.0, 1.0, "dyer").mass_flux(st0, 2e6) / tank.Injector(1.0, 1.0, "spi").mass_flux(st0, 2e6)
results["n2o_blowdown"] = {
    "liquid_fill": round(fill, 3), "burn_s": round(float(bd.t[-1]), 2),
    "tank_MPa": [round(float(bd.p_tank[0]) / 1e6, 2), round(float(bd.p_tank[-1]) / 1e6, 2)],
    "tank_C": [round(float(bd.T_tank[0]) - 273.15, 1), round(float(bd.T_tank[-1]) - 273.15, 1)],
    "chamber_MPa": [round(float(bd.pc[0]) / 1e6, 2), round(float(bd.pc[-1]) / 1e6, 2)],
    "mdot_ox": [round(float(bd.mdot_ox[0]), 3), round(float(bd.mdot_ox[-1]), 3)],
    "of": [round(float(bd.of[0]), 2), round(float(bd.of[-1]), 2)], "of_spread": round(bd.of_spread(), 3),
    "of_constant_flow": [round(float(cf_run.of[0]), 2), round(float(cf_run.of[-1]), 2)],
    "of_spread_constant_flow": round(cf_run.of_spread(), 3),
    "thrust_N": [round(float(bd.thrust[0])), round(float(bd.thrust[-1]))],
    "impulse_Ns": round(bd.total_impulse), "isp_s": round(bd.isp, 1),
    "oxidizer_used_kg": round(bd.ox_mass, 2), "vapor_left_kg": round(bd.vapor_left, 2),
    "dyer_over_spi_at_2MPa": round(spi_dyer, 3),
    "best_of_2MPa": float(n2o_table.of[int(np.argmax(n2o_table.cstar[:, 2]))]),
}

# ------------------------------------------------------------------ 4. detonation / RDE
mixtures = {
    "H2-O2": thermo.bipropellant(thermo.O2, thermo.H2, 32 / 4.032),
    "H2-air": thermo.bipropellant(thermo.AIR, thermo.H2, 0.5 * 4.76 * 28.851 / 2.016),
    "C2H4-air": thermo.bipropellant(thermo.AIR, thermo.C2H4, 3 * 4.76 * 28.851 / 28.054),
    "CH4-O2": thermo.bipropellant(thermo.O2, thermo.CH4, 64 / 16.043),
}
literature = {"H2-O2": 2836, "H2-air": 1968, "C2H4-air": 1825}  # 1 atm, 298 K, stoichiometric
results["cj"] = {}
cj_h2air = None
for name, mix in mixtures.items():
    cj = rde.cj_state(mix)
    results["cj"][name] = {"D_m_s": round(cj.D, 1), "T_K": round(cj.T2), "P_ratio": round(cj.P2 / cj.P1, 2),
                           "literature_D": literature.get(name)}
    if name == "H2-air":
        cj_h2air = cj

R_h2air = 8314.46 / 20.91  # unburned stoichiometric H2-air, kg/kmol -> J/kg/K
model = rde.OneGamma.from_cj(cj_h2air, R=R_h2air)
pis = np.linspace(1, 30, 120)
fig, ax = plt.subplots(figsize=(6.5, 4))
for cyc, ls in (("fickett-jacobs", "-"), ("humphrey", "--"), ("brayton", ":")):
    ax.plot(pis, [model.efficiency(cyc, p) * 100 for p in pis], ls, label=cyc.replace("-", "–").title())
ax.set_xlabel("precompression pressure ratio")
ax.set_ylabel("ideal thermal efficiency (%)")
ax.set_title("Why detonate? H2-air, one-gamma model fit to the CJ speed")
ax.legend()
save(fig, "cycle_efficiency.png")
results["cycle_at_pi"] = {str(p): {c: round(model.efficiency(c, p) * 100, 1) for c in ("fickett-jacobs", "humphrey", "brayton")}
                          for p in (1, 2, 5, 10)}

# Sizing depends on the detonation cell size lambda, which must come from
# experiments (e.g. the Caltech Detonation Database). Sweep it instead of guessing.
lams = np.linspace(1, 20, 60)
fig, ax = plt.subplots(figsize=(6.5, 4))
siz = [rde.bykovskii_sizing(l, 150.0, cj_h2air.D) for l in lams]
ax.plot(lams, [z.min_diameter_mm for z in siz], label="minimum annulus diameter (40 lambda)")
ax.fill_between(lams, [z.fill_height_mm[0] for z in siz], [z.fill_height_mm[2] for z in siz], alpha=0.3,
                label="critical fill height h* = (12 +/- 5) lambda")
ax.plot(lams, [z.min_length_mm for z in siz], "--", label="minimum chamber length (2 h*)")
ax.set_xlabel("detonation cell size lambda (mm)")
ax.set_ylabel("mm")
ax.set_title("First-cut RDE geometry from Bykovskii's correlations")
ax.legend(fontsize=8)
save(fig, "rde_sizing.png")
results["rde_sizing_examples"] = {f"lambda_{l}mm_D150": rde.bykovskii_sizing(l, 150.0, cj_h2air.D).__dict__ for l in (1.0, 2.0, 5.0, 10.0)}

# ------------------------------------------------------------------ 5. nozzle CAD
rt = 0.015
contour = bell_contour(rt, area_ratio=8.0, theta_n_deg=27.0, theta_e_deg=10.0, chamber_radius=0.035)
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(contour.x * 1000, contour.r * 1000, "k")
ax.plot(contour.x * 1000, -contour.r * 1000, "k")
ax.set_aspect("equal")
ax.set_xlabel("axial (mm)")
ax.set_ylabel("radius (mm)")
ax.set_title(f"80% bell, area ratio {contour.area_ratio:.2f}, throat radius {rt * 1000:.0f} mm")
save(fig, "nozzle_contour.png")
cad = ROOT / "cad"
export_nozzle_cad(contour, wall=0.003, step_path=str(cad / "bell_nozzle_e8.step"), stl_path=str(cad / "bell_nozzle_e8.stl"),
                  flange_radius=0.045, flange_thickness=0.006)
results["nozzle"] = {"area_ratio": round(contour.area_ratio, 3), "length_mm": round(contour.length * 1000, 1)}

(ROOT / "docs" / "results.json").write_text(json.dumps(results, indent=2, default=float))
print(json.dumps(results, indent=2, default=float))
