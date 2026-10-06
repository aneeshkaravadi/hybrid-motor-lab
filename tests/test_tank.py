"""Nitrous tank and injector, checked against hand calculations and an independent formulation of the drain."""
import numpy as np
import pytest

CP = pytest.importorskip("CoolProp.CoolProp")

from hml import grain, hybrid, tank, thermo  # noqa: E402

V, M0, T0 = 0.005, 3.5, 293.15


def test_tank_starts_saturated_with_the_right_fill():
    tk = tank.N2OTank(V, M0, T0)
    st = tk.state()
    assert st.T == pytest.approx(T0, abs=1e-6)
    assert st.p == pytest.approx(CP.PropsSI("P", "T", T0, "Q", 0, "NitrousOxide"), rel=1e-9)
    rho_l = CP.PropsSI("D", "T", T0, "Q", 0, "NitrousOxide")
    rho_g = CP.PropsSI("D", "T", T0, "Q", 1, "NitrousOxide")
    assert tk.liquid_fill() == pytest.approx((M0 / V - rho_g) / (rho_l - rho_g), rel=1e-6)
    with pytest.raises(ValueError, match="won't sit as liquid plus vapor"):
        tank.N2OTank(V, 1.01 * rho_l * V, T0)


def test_drain_matches_an_ode_for_the_tank_temperature():
    """The tank's mass-and-energy bookkeeping (with a CoolProp flash every step) against a different formulation.

    Draining saturated liquid at fixed volume, the evaporation needed to refill the space and the energy balance
    give dT/dm_out = -v_l h_fg / (v_fg C - u_fg B), with B and C the mass-weighted slopes of v and u along the
    saturation curve (derived in DERIVATIONS.md). Integrated here with solve_ivp, it never touches the flash.
    """
    from scipy.integrate import solve_ivp
    drained, steps = 2.0, 2000
    tk = tank.N2OTank(V, M0, T0)
    for _ in range(steps):
        tk.drain_liquid(drained / steps, 1.0)

    def slope(m_out, y):
        T, e = y[0], 1e-3
        s, lo, hi = tank.saturation(T), tank.saturation(T - e), tank.saturation(T + e)
        v_l, v_g = 1 / s.rho_l, 1 / s.rho_g
        m_g = (V - (M0 - m_out) * v_l) / (v_g - v_l)
        m_l = M0 - m_out - m_g
        B = m_l * (1 / hi.rho_l - 1 / lo.rho_l) / (2 * e) + m_g * (1 / hi.rho_g - 1 / lo.rho_g) / (2 * e)
        C = m_l * (hi.u_l - lo.u_l) / (2 * e) + m_g * (hi.u_g - lo.u_g) / (2 * e)
        return [-v_l * (s.h_g - s.h_l) / ((v_g - v_l) * C - (s.u_g - s.u_l) * B)]

    sol = solve_ivp(slope, (0.0, drained), [T0], rtol=1e-10, atol=1e-10)
    assert tk.state().T == pytest.approx(sol.y[0, -1], abs=0.01)
    assert T0 - sol.y[0, -1] > 10  # the tank really does cool a lot


def test_injector_models():
    st = tank.N2OTank(V, M0, T0).state()
    inj = {m: tank.Injector(1e-5, cd=0.8, model=m) for m in ("spi", "hem", "dyer")}
    p2 = 2.0e6
    assert inj["spi"].mass_flow(st, p2) == pytest.approx(0.8e-5 * np.sqrt(2 * st.sat.rho_l * (st.p - p2)), rel=1e-12)
    # a saturated tank has p1 = p_v, so Dyer's k = 1 and it averages SPI and HEM
    assert inj["dyer"].mass_flux(st, p2) == pytest.approx((inj["spi"].mass_flux(st, p2) + inj["hem"].mass_flux(st, p2)) / 2)
    # with almost no pressure drop, almost no vapor forms, so HEM approaches the liquid-only flow
    tiny = st.p * (1 - 1e-3)
    assert inj["hem"].mass_flux(st, tiny) / inj["spi"].mass_flux(st, tiny) == pytest.approx(1.0, abs=5e-3)
    # HEM chokes: below the critical downstream pressure the flow stops growing
    g = [inj["hem"].mass_flux(st, f * st.p) for f in (0.2, 0.4, 0.6, 0.9)]
    assert g[0] == pytest.approx(g[1], rel=1e-9) and g[1] == pytest.approx(g[2], rel=1e-9) and g[3] < g[2]


def test_tank_fed_burn_holds_together():
    ox = tank.liquid_n2o(T0)
    table = thermo.build_table(ox, thermo.PARAFFIN, np.linspace(3.0, 9.0, 4), [1e6, 2e6, 3e6], 5.0)
    tk = tank.N2OTank(V, M0, T0)
    r = tank.simulate_blowdown(grain.tubular(0.1, 0.035), 0.2, hybrid.PARAFFIN_N2O, table, tk,
                               tank.Injector(1.4e-5), 0.024, dt=0.02)
    assert r.liquid_runout_time is not None
    assert np.all(r.pc < r.p_tank)
    assert np.all(np.diff(r.p_tank) < 0) and np.all(np.diff(r.mdot_ox) < 0)
    assert r.ox_mass + tk.mass == pytest.approx(M0, rel=1e-12)
    # when the liquid runs out the tank is full of saturated vapor at its final temperature
    T_end = tk.state().T
    assert r.vapor_left == pytest.approx(CP.PropsSI("D", "T", T_end, "Q", 1, "NitrousOxide") * V, rel=0.02)


def test_liquid_n2o_enthalpy_sits_below_the_gas_by_about_the_heat_of_vaporization():
    gas = thermo.N2O.enthalpy_j_per_kg()
    liq = tank.liquid_n2o(T0).enthalpy_j_per_kg()
    h_fg = CP.PropsSI("H", "T", T0, "Q", 1, "NitrousOxide") - CP.PropsSI("H", "T", T0, "Q", 0, "NitrousOxide")
    # gas at 298 K vs liquid at 293 K: the heat of vaporization plus a little sensible and real-gas enthalpy
    assert h_fg < gas - liq < 1.6 * h_fg
