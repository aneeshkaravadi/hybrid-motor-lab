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
