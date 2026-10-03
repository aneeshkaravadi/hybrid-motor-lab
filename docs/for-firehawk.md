# For the Firehawk Aerospace team

You 3D-print propellant and fly hybrid motors, so the relevant part is the hybrid model and what printed port geometry can and can't do.

1. **Arbitrary port burn-back** (`hml/grain.py`). Any shape you can describe (star, finocyl, wagon wheel, or your own) becomes a burn-back history $A(x)$, $P(x)$ with sub-pixel accuracy (0.1% against the exact circle).
2. **The O/F finding.** With $\dot r = aG^n$, any self-similar port gives $O/F \propto A^{\,n-1/2}$. For $n > 0.5$, shaped ports front-load fuel (+3–5% impulse here) but *widen* the O/F swing (70–88% vs 21% for a tube). A 20-design star sweep found none that beats the tube. ([figures](figures/star_sweep.png))
3. **Inverse design.** It computes either the oxidizer throttle schedule, or the **radial regression-rate grading of a printed fuel**, that holds O/F constant. For the example motor: a throttle range of 0.73 to 0.41 kg/s, or a coefficient rising from 0.87× to 1.07× outward. ([figure](figures/of_remedies.png))

**What it doesn't do:** the regression law is space-averaged (no axial G variation), with no oxidizer-tank blowdown, and the thermochemistry is gas-phase only. The regression coefficients are paraffin/GOX from the literature, not your fuel.

**A 3-week project I could do for you, remote:** fit the regression law to your static-fire data (the fitting script already exists for solids), then run a port-geometry and grading trade study for one of your motors. The deliverable is a ranked design table (impulse, O/F swing, residual sliver) plus the code.

— Aneesh Karavadi · aneesh.karavadi@gmail.com
