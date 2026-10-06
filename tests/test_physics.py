"""Checks against closed-form answers and published reference values."""
import numpy as np
import pytest

from hml import grain, hybrid, rde, solid, thermo
from hml.nozzle import bell_contour, exit_mach, thrust_coefficient


# ---------------------------------------------------------------- thermochemistry
@pytest.mark.parametrize("ox, fuel, of, expected_K", [
    (thermo.O2, thermo.H2, 32 / 4.032, 3080),                 # stoichiometric H2/O2, 1 atm
    (thermo.AIR, thermo.CH4, 2 * 4.76 * 28.851 / 16.043, 2226),  # stoichiometric CH4/air
    (thermo.O2, thermo.CH4, 64 / 16.043, 3050),               # stoichiometric CH4/O2
])
def test_adiabatic_flame_temperature(ox, fuel, of, expected_K):
    gas = thermo.equilibrate_hp(thermo.bipropellant(ox, fuel, of), 101325.0)
    assert gas.T == pytest.approx(expected_K, abs=25)


def test_rocket_throat_is_sonic_and_area_ratio_hit():
    r = thermo.rocket(thermo.bipropellant(thermo.O2, thermo.PARAFFIN, 2.4), 3e6, 6.0, p_ambient=101325.0)
    assert 0.5 < r.throat.P / 3e6 < 0.62  # pt/pc for gamma ~ 1.1-1.3
    assert r.area_ratio == 6.0
    assert 1700 < r.cstar < 1900
    assert r.isp_vac > r.isp


def test_paraffin_enthalpy_sensitivity_is_small():
    """Our paraffin heat of formation is uncertain; show c* barely cares."""
    base = thermo.rocket(thermo.bipropellant(thermo.O2, thermo.PARAFFIN, 2.4), 3e6, 6.0).cstar
    for dh in (-1.9e3, -2.2e3):
        fuel = thermo.Reactant("paraffin", {"C": 32, "H": 66}, dhf_kj_mol=dh * thermo.PARAFFIN_MOLAR_MASS)
        c = thermo.rocket(thermo.bipropellant(thermo.O2, fuel, 2.4), 3e6, 6.0).cstar
        assert abs(c - base) / base < 0.01


# NASA CEA Example 5 (RP-1311 Part II): an aluminized AP composite, HP equilibrium.
CEA_EX5 = [(thermo.AP, 72.06), (thermo.CEA_BINDER, 18.58), (thermo.ALUMINUM, 9.0), (thermo.MGO, 0.2),
           (thermo.WATER_LIQUID, 0.16)]
CEA_EX5_T = {500: 2722.99, 250: 2706.53, 125: 2686.16, 50: 2653.00, 5: 2540.82}  # psia: K
CEA_EX5_X_500PSIA = {"AL2O3(L)": 0.03672, "HCL": 0.13187, "H2": 0.32150, "CO": 0.26456, "N2": 0.06833,
                     "H2O": 0.14650, "CO2": 0.01778}


def test_condensed_equilibrium_matches_cea_example_5():
    eq = thermo.Equilibrium(CEA_EX5)
    assert eq.h0 / 4184.0 == pytest.approx(-484.77, abs=0.02)  # CEA's mixture enthalpy, cal/g
    for psia, T in CEA_EX5_T.items():
        st = eq.hp(psia * 6894.757)
        assert st.T == pytest.approx(T, abs=5.0)
        if psia == 500:
            for sp, x in CEA_EX5_X_500PSIA.items():
                assert st.X[sp] == pytest.approx(x, rel=0.01), sp
    # leave out the condensed phase and the aluminum ends up as gaseous chlorides, 500 K too cold
    assert thermo.equilibrate_hp(CEA_EX5, 500 * 6894.757).T < CEA_EX5_T[500] - 400


def test_multiphase_rocket_is_the_gas_rocket_when_nothing_condenses():
    mix = [(thermo.AP, 72.06), (thermo.CEA_BINDER, 18.58)]
    gas = thermo.rocket(mix, 6.9e6, 10.0, 101325.0)
    multi = thermo.rocket_multiphase(mix, 6.9e6, 10.0, 101325.0)
    assert multi.chamber.condensed == {}
    assert multi.cstar == pytest.approx(gas.cstar, rel=1e-4)
    assert multi.isp_vac == pytest.approx(gas.isp_vac, rel=1e-4)


# ---------------------------------------------------------------- detonation
@pytest.mark.parametrize("name, mix, D_lit", [
    ("H2-O2", thermo.bipropellant(thermo.O2, thermo.H2, 32 / 4.032), 2836),
    ("H2-air", thermo.bipropellant(thermo.AIR, thermo.H2, 0.5 * 4.76 * 28.851 / 2.016), 1968),
    ("C2H4-air", thermo.bipropellant(thermo.AIR, thermo.C2H4, 3 * 4.76 * 28.851 / 28.054), 1825),
])
def test_cj_speed_matches_literature(name, mix, D_lit):
    assert rde.cj_state(mix).D == pytest.approx(D_lit, rel=0.005)


def test_one_gamma_round_trip_and_cycle_ordering():
    cj = rde.cj_state(thermo.bipropellant(thermo.AIR, thermo.H2, 0.5 * 4.76 * 28.851 / 2.016))
    m = rde.OneGamma.from_cj(cj, R=8314.46 / 20.91)
    assert m.m_cj(cj.T1) * np.sqrt(m.gamma * m.R * cj.T1) == pytest.approx(cj.D, rel=1e-9)
    for pi in (1.0, 3.0, 10.0):
        fj, hu, br = (m.efficiency(c, pi) for c in ("fickett-jacobs", "humphrey", "brayton"))
        assert fj > hu > br  # detonation > constant-volume > constant-pressure
    assert m.efficiency("brayton", 1.0) == pytest.approx(0.0, abs=1e-12)


def test_rde_sizing_flags_undersized_annulus():
    assert not rde.bykovskii_sizing(10.0, 150.0, 1970.0).feasible
    assert rde.bykovskii_sizing(2.0, 150.0, 1970.0).feasible


# ---------------------------------------------------------------- grain geometry
def test_tubular_perimeter_matches_circle():
    D, d = 0.1, 0.035
    x, A, P = grain.tubular(D, d).curves(300)
    exact = 2 * np.pi * (d / 2 + x)
    mid = slice(10, 280)
    assert np.max(np.abs(P[mid] - exact[mid]) / exact[mid]) < 0.01
    assert A[0] == pytest.approx(np.pi * d**2 / 4, rel=1e-3)
    assert grain.tubular(D, d).web() == pytest.approx((D - d) / 2, rel=0.01)


def test_bates_burn_area_matches_closed_form():
    D, d, L = 0.075, 0.025, 0.12
    seg = grain.Segment(grain.tubular(D, d), L)
    c = seg.port.curves(300)
    for x in (0.002, 0.008, 0.015, 0.022):
        assert seg.burn_area(x, c)[0] == pytest.approx(grain.bates_area_exact(D, d, L, x), rel=0.01)


def test_star_web_is_distance_from_valley_region_to_casing():
    s = grain.star(0.075, 6, 0.025, 0.012)
    assert s.port_mask.any() and s.web() > (0.075 / 2 - 0.025)


# ---------------------------------------------------------------- ballistics
def test_solid_mass_balance():
    segs = [grain.Segment(grain.tubular(0.054, 0.02), 0.07) for _ in range(4)]
    r = solid.simulate(segs, solid.EXAMPLE_APCP, 0.017, 6.0)
    burned = np.trapezoid(r.mdot, r.t)
    assert burned == pytest.approx(r.propellant_mass, rel=0.03)


def test_erosive_ratio_matches_mukunda_paul_by_hand():
    prop = solid.EXAMPLE_APCP
    r0, d = 0.006, 0.02
    re0 = prop.density * r0 * d / prop.viscosity
    # pick the port flux that makes g exactly 100 (and 30, under the threshold of 35)
    G100 = 100 * (re0 / 1000) ** 0.125 * prop.density * r0
    assert solid.erosive_ratio(G100, r0, prop, d) == pytest.approx(1 + 0.023 * (100**0.8 - 35**0.8), rel=1e-12)
    assert solid.erosive_ratio(0.3 * G100, r0, prop, d) == 1.0


@pytest.mark.parametrize("name", ["bates", "star"])
def test_axial_solid_model_without_erosion_is_the_closed_form(name):
    if name == "bates":  # uninhibited ends, so the cells at each end burn away
        segs, throat = [grain.Segment(grain.tubular(0.054, 0.02), 0.07) for _ in range(4)], 0.017
    else:
        segs, throat = [grain.Segment(grain.star(0.054, 6, 0.019, 0.009), 0.28, True)], 0.017
    ref = solid.simulate(segs, solid.EXAMPLE_APCP, throat, 6.0)
    ax = solid.simulate_erosive(segs, solid.EXAMPLE_APCP, throat, 6.0, erosive=False)
    assert ax.pc == pytest.approx(ref.pc, rel=1e-9)
    assert ax.t == pytest.approx(ref.t, rel=1e-9)


def test_erosive_burning_raises_the_start_pressure_and_conserves_mass():
    segs = [grain.Segment(grain.tubular(0.054, 0.02), 0.07) for _ in range(4)]
    ref = solid.simulate(segs, solid.EXAMPLE_APCP, 0.017, 6.0)
    ero = solid.simulate_erosive(segs, solid.EXAMPLE_APCP, 0.017, 6.0)
    assert ero.pc[0] > 1.1 * ref.pc[0]
    assert ero.peak_ratio[0] > 1.3 and ero.port_choke_ratio.max() < 1
    # the head-end grain sees too little flow to erode; the nozzle-end one burns faster toward the nozzle
    web = ero.web_z[len(ero.t) // 4]
    head, aft = web[:40], web[120:]
    assert np.ptp(head) == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(aft) > 0) and aft[-1] > 1.05 * aft[0] > 1.05 * head[0]
    assert np.trapezoid(ero.mdot, ero.t) == pytest.approx(ero.propellant_mass, rel=0.03)


def test_erosive_model_flags_a_port_that_would_choke():
    """All the boost grain's gas has to squeeze through the sustainer's port, which is smaller than the throat."""
    boost = grain.Segment(grain.finocyl(0.054, 0.030, 6, 0.008, 0.003), 0.12, True)
    sustain = grain.Segment(grain.tubular(0.054, 0.010), 0.16, True)
    bad = solid.simulate_erosive([boost, sustain], solid.EXAMPLE_APCP, 0.015, 6.0)
    good = solid.simulate_erosive([sustain, boost], solid.EXAMPLE_APCP, 0.015, 6.0)
    assert bad.port_choke_ratio[0] > 1 > good.port_choke_ratio.max()


def test_hybrid_of_rises_for_n_above_half_and_remedies_flatten_it():
    table = thermo.build_table(thermo.O2, thermo.PARAFFIN, np.linspace(1.5, 3.5, 5), [1e6, 2e6, 3e6], 5.0)
    port = grain.tubular(0.1, 0.035)
    law = hybrid.PARAFFIN_GOX
    base = hybrid.simulate(port, 0.4, law, table, 0.5, 0.03, 6.0, dt=0.02)
    assert base.of[-1] > base.of[0]
    thr = hybrid.simulate(port, 0.4, law, table, hybrid.throttle_schedule(port, 0.4, law, 2.4), 0.03, 6.0, dt=0.02)
    grd = hybrid.simulate(port, 0.4, law, table, 0.5, 0.03, 6.0, dt=0.02,
                          a_scale=hybrid.graded_fuel_profile(port, 0.4, law, 0.5, 2.4))
    assert thr.of_spread() < 1e-6 and grd.of_spread() < 1e-6


def test_axial_fuel_flow_matches_an_ode_solve():
    """The cell-by-cell closed form against a numerical solution of d(mdot)/dz = rho a (mdot/A)^n P."""
    from scipy.integrate import solve_ivp
    law = hybrid.PARAFFIN_GOX
    A, P, L, mo = np.pi * 0.0175**2, 2 * np.pi * 0.0175, 0.4, 0.5
    sol = solve_ivp(lambda z, m: law.fuel_density * law.a * (m / A) ** law.n * P, (0, L), [mo], rtol=1e-11, atol=1e-12,
                    dense_output=True)
    cells = 40
    added = hybrid.fuel_added_per_cell(mo, np.full(cells, A), np.full(cells, P), np.full(cells, law.a), law, L / cells)
    z_edges = np.linspace(0, L, cells + 1)[1:]
    assert mo + np.cumsum(added) == pytest.approx(sol.sol(z_edges)[0], rel=1e-8)


@pytest.fixture(scope="module")
def small_table():
    return thermo.build_table(thermo.O2, thermo.PARAFFIN, np.linspace(1.5, 3.5, 5), [1e6, 2e6, 3e6], 5.0)


def test_axial_model_with_oxidizer_flux_is_the_averaged_model(small_table):
    port, law = grain.tubular(0.1, 0.035), hybrid.PARAFFIN_GOX
    avg = hybrid.simulate(port, 0.4, law, small_table, 0.5, 0.03, 4.0, dt=0.02)
    ax = hybrid.simulate_axial(port, 0.4, law, small_table, 0.5, 0.03, 4.0, dt=0.02, flux="oxidizer")
    assert ax.of == pytest.approx(avg.of, rel=1e-9)
    assert ax.fuel_mass == pytest.approx(avg.fuel_mass, rel=1e-9)


def test_axial_regression_conserves_fuel_and_opens_the_aft_end_faster(small_table):
    port, law, L, mo = grain.tubular(0.1, 0.035), hybrid.PARAFFIN_GOX, 0.4, 0.5
    tl = hybrid.total_flux_law(law, port, L, mo)
    avg = hybrid.simulate(port, L, law, small_table, mo, 0.03, 6.0, dt=0.02)
    ax = hybrid.simulate_axial(port, L, tl, small_table, mo, 0.03, 6.0, dt=0.02)
    assert ax.mdot_fuel[0] == pytest.approx(avg.mdot_fuel[0], rel=1e-9)  # same fuel flow at ignition
    assert ax.web_z[-1, -1] > 1.1 * ax.web_z[-1, 0]  # aft end ahead of the head end
    # fuel burned so far equals the volume the port has grown by (to the last recorded state)
    xs, A, _ = port.curves(400)
    grown = law.fuel_density * np.sum(np.interp(ax.web_z[-1], xs, A) - A[0]) * L / len(ax.z)
    assert grown == pytest.approx(ax.fuel_mass - ax.mdot_fuel[-1] * 0.02, rel=2e-3)


def test_regression_unit_conversion():
    # 0.488 mm/s at G = 1 g/cm^2/s = 10 kg/m^2/s
    law = hybrid.PARAFFIN_GOX
    assert law.a * 10.0**law.n == pytest.approx(0.488e-3, rel=1e-9)


# ---------------------------------------------------------------- nozzle
def test_exit_mach_and_cf_textbook():
    # gamma 1.4, area ratio 4 -> Me = 2.94 (isentropic tables)
    assert exit_mach(1.4, 4.0) == pytest.approx(2.94, abs=0.01)
    # optimally expanded CF for gamma 1.2, eps 10 is about 1.6
    g, eps = 1.2, 10.0
    Me = exit_mach(g, eps)
    pe_pc = (1 + (g - 1) / 2 * Me**2) ** (-g / (g - 1))
    assert thrust_coefficient(g, eps, 1.0, pe_pc) == pytest.approx(1.6, abs=0.05)


def test_bell_contour_hits_area_ratio_and_is_monotonic():
    c = bell_contour(0.015, 8.0, chamber_radius=0.035)
    assert c.area_ratio == pytest.approx(8.0, rel=1e-6)
    throat = np.argmin(c.r)
    assert c.r[throat] == pytest.approx(0.015, rel=1e-3)
    assert np.all(np.diff(c.r[throat:]) >= -1e-12)
