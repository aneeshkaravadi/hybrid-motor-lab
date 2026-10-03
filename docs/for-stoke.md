# For the Stoke Space team

1. **Thermochemistry on Cantera** (`hml/thermo.py`). Adiabatic equilibrium plus frozen or shifting isentropic expansion: the same problem NASA CEA solves. It returns c*, C_F and Isp for any propellant pair made of species in the NASA database. Checked against textbook flame temperatures (within about 3 K).
2. **Printable nozzle CAD** (`hml/nozzle.py`). A Rao-style bell contour goes straight to a STEP solid. A DfAM check on the exported mesh shows the part is printable in metal powder-bed fusion with 2.5% support area when built exit-up. ([report](dfam_report.md))

**What it doesn't do:** regenerative cooling, real-gas effects at injector conditions, and boundary-layer losses.

**A 3-week project I could do for you, remote:** an O/F × Pc × area-ratio performance-map generator with frozen and shifting bounds, wrapped as a tested library. Or a parametric nozzle-contour-to-CAD pipeline with automatic printability checks.

— Aneesh Karavadi · aneesh.karavadi@gmail.com
