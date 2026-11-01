"""Run NASA CEA on the cases the thermochemistry tests check, and write tests/data/cea_reference.json.

    pip install cea                    # NASA's open-source CEA, https://github.com/nasa/cea (Apache-2.0)
    python tools/cea_reference.py

The tests only read the JSON, so CI never needs CEA or the network. Every case uses the same reactants,
mass fractions and enthalpies as the model it checks (see src/hml/thermo.py), and CEA gets its full
product list, condensed species included, so where the gas-only model leaves something out it shows.
"""
from __future__ import annotations

import json
from pathlib import Path

import cea
import numpy as np

G0 = 9.80665
ATM_BAR = 1.01325
OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "cea_reference.json"
# Same atomic masses as thermo.ATOMIC_MASS, so custom reactants get the same molar masses as in the model.
AM = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007, "S": 32.06}


def custom(name, formula, h, units):
    mw = sum(AM[e] * n for e, n in formula.items())
    return cea.Reactant(name=name, formula=formula, molecular_weight=mw, enthalpy=h, enthalpy_units=units,
                        temperature=298.15)


PARAFFIN_MW = 32 * AM["C"] + 66 * AM["H"]
PARAFFIN = custom("paraffin", {"C": 32, "H": 66}, -2.0e3 * PARAFFIN_MW / 1000.0, "kJ/mol")  # -2.0 MJ/kg
AIR = custom("air", {"N": 2 * 3.76 / 4.76, "O": 2 / 4.76}, 0.0, "kJ/mol")  # thermo.AIR: 21% O2, 79% N2
BINDER = custom("CHOS-Binder", {"C": 1.0, "H": 1.86955, "O": 0.031256, "S": 0.008415}, -2999.082, "cal/mol")
AIR_MW = 2 * 3.76 / 4.76 * AM["N"] + 2 / 4.76 * AM["O"]


def mixes(species):
    return cea.Mixture(species), cea.Mixture(species, products_from_reactants=True)


def condensed_mass_fraction(soln, solver, prod, point=0):
    """Mass fraction of all condensed products at one station (CEA lists the gases first)."""
    condensed = set(prod.species_names[solver.num_gas:])
    return float(sum(v[point] for k, v in soln.mass_fractions.items() if k in condensed))


def hp_flames():
    """Adiabatic flame temperature at 1 atm, stoichiometric, reactants at 298.15 K."""
    cases = {
        "H2-O2": (["H2", "O2"], 32 / 4.032),
        "CH4-air": (["CH4", AIR], 2 * 4.76 * AIR_MW / 16.043),
        "CH4-O2": (["CH4", "O2"], 64 / 16.043),
    }
    out = {}
    for name, (species, of) in cases.items():
        reac, prod = mixes(species)
        w = np.array([1.0, of]) / (1.0 + of)
        h0 = reac.calc_property(cea.ENTHALPY, w, 298.15) / cea.R
        solver = cea.EqSolver(prod, reactants=reac)
        soln = cea.EqSolution(solver)
        solver.solve(soln, cea.HP, h0, ATM_BAR, w)
        assert soln.converged, name
        out[name] = {"of": of, "T_K": float(soln.T)}
    return out


def rocket(species, w, pc_bar, eps):
    reac, prod = mixes(species)
    solver = cea.RocketSolver(prod, reactants=reac)
    soln = cea.RocketSolution(solver)
    hc = reac.calc_property(cea.ENTHALPY, w, np.full(len(w), 298.15)) / cea.R
    solver.solve(soln, w, pc_bar, supar=[eps], hc=hc, iac=True)
    assert soln.converged
    cstar = float(soln.c_star[0])
    return {"Tc_K": float(soln.T[0]), "cstar": cstar, "cf_vac": float(soln.Isp_vacuum[-1]) / cstar,
            "isp_vac_s": float(soln.Isp_vacuum[-1]) / G0,
            "condensed_chamber": condensed_mass_fraction(soln, solver, prod),
            "condensed_exit": condensed_mass_fraction(soln, solver, prod, -1)}


def paraffin_gox():
    """Paraffin (-2.0 MJ/kg) with gaseous O2, infinite-area chamber, shifting, area ratio 5: the hybrid table."""
    out = []
    grid = {20.0: [0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 2.0, 2.2, 2.6, 3.0, 3.5, 4.0],
            5.0: [1.2, 2.2, 3.0, 4.0], 50.0: [1.2, 2.2, 3.0, 4.0]}
    for pc_bar, ofs in grid.items():
        for of in ofs:
            w = np.array([1.0, of]) / (1.0 + of)
            out.append({"of": of, "pc_MPa": pc_bar / 10.0, **rocket([PARAFFIN, "O2"], w, pc_bar, 5.0)})
    return out


def aluminized():
    """AP and CEA Example 5's binder in Example 5's ratio, plus aluminum, at 1000 psia and area ratio 10."""
    pc_bar = 1000 * 6894.757 / 1e5
    out = []
    for al in np.round(np.arange(0.0, 0.241, 0.03), 2):
        species = ["NH4CLO4(I)", BINDER] + (["AL(cr)"] if al > 0 else [])
        w = [(1 - al) * 72.06 / 90.64, (1 - al) * 18.58 / 90.64] + ([al] if al > 0 else [])
        r = rocket(species, np.array(w), pc_bar, 10.0)
        r["isp_sl_s"] = (r["isp_vac_s"] * G0 - 101325.0 * 10.0 * r["cstar"] / (pc_bar * 1e5)) / G0
        out.append({"al_pct": round(float(al) * 100), **r})
    return out


def cj():
    """Chapman-Jouguet detonations, stoichiometric, 298.15 K and 1 atm."""
    cases = {
        "H2-O2": (["H2", "O2"], 32 / 4.032),
        "H2-air": (["H2", AIR], 0.5 * 4.76 * AIR_MW / 2.016),
        "C2H4-air": (["C2H4", AIR], 3 * 4.76 * AIR_MW / 28.054),
        "CH4-O2": (["CH4", "O2"], 64 / 16.043),
    }
    out = {}
    for name, (species, of) in cases.items():
        reac, prod = mixes(species)
        solver = cea.DetonationSolver(prod, reactants=reac)
        soln = cea.DetonationSolution(solver)
        solver.solve(soln, np.array([1.0, of]) / (1.0 + of), 298.15, ATM_BAR)
        assert soln.converged, name
        out[name] = {"of": of, "D_m_s": float(soln.velocity), "T_K": float(soln.T), "P_ratio": float(soln.P_P1)}
    return out


if __name__ == "__main__":
    data = {"source": f"NASA CEA {cea.__version__} (pip install cea, https://github.com/nasa/cea), "
                      "written by tools/cea_reference.py",
            "hp_flames_1atm": hp_flames(), "paraffin_gox_eps5": paraffin_gox(),
            "aluminized_1000psia_eps10": aluminized(), "cj_1atm": cj()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print(json.dumps(data, indent=1))
