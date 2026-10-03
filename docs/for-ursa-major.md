# For the Ursa Major team

You build printed liquid engines and solid motors. Two parts of this repo map to that:

1. **Solid-motor ballistics with arbitrary grain geometry** (`hml/grain.py`, `hml/solid.py`): burn-back for any port shape, multi-segment grains, and fitting a burn-rate law to static-fire data. ([figure](figures/solid_thrust_shaping.png))
2. **Nozzle contour to STEP, plus a metal-PBF printability check** (`hml/nozzle.py`, [DfAM report](dfam_report.md)).

**What it doesn't do:** structural, thermal or erosion analysis. Propellant constants are illustrative.

**A 3-week project I could do for you, remote:** a thrust-curve inverse-design tool (a target curve in, candidate grain geometries out) or an automated printability gate for nozzle CAD.

— Aneesh Karavadi · aneesh.karavadi@gmail.com
