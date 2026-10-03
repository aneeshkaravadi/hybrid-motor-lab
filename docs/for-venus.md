# For the Venus Aerospace team

You fly rotating detonation engines, so the parts of this repo worth two minutes are:

1. **`hml/rde.py`: Chapman–Jouguet solver on full equilibrium chemistry.** It matches published CJ speeds to within 1 m/s (H₂/O₂ 2837, H₂/air 1969, C₂H₄/air 1824 m/s). Runs in milliseconds.
2. **Cycle comparison.** A one-gamma model is calibrated to that CJ speed and compares Fickett–Jacobs, Humphrey and Brayton efficiency against precompression. For H₂/air with no compressor: 25%, 23% and 0%. That is the pressure-gain argument, made quantitative. ([figure](figures/cycle_efficiency.png), [derivation](../DERIVATIONS.md#7-why-detonate-cycle-efficiencies-rdeonegamma))
3. **First-cut annulus sizing** from Bykovskii's correlations (fill height, channel width, minimum diameter, wave count, frequency). It is swept over cell size rather than pretending λ is known.

**What it doesn't do:** no unsteady CFD, no wave-number bifurcation physics, no injector or mixing model. The cycle numbers are ideal limits.

**A 3-week project I could do for you, remote:** extend the CJ and one-gamma tools to your propellant combination and operating pressures. Add a parametric sweep (equivalence ratio, inlet temperature, pressure) that outputs CJ state, ideal Isp and Bykovskii geometry bounds as a lookup table your engineers can query. The deliverable is a tested Python package plus a short report.

— Aneesh Karavadi · aneesh.karavadi@gmail.com
