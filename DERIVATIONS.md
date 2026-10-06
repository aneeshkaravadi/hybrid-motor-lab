# Derivations

Every equation the code uses, derived from the conservation laws, with a pointer to where it lives. If an interviewer asks "where does that come from?", the answer is in here.

---

## 1. Combustion chamber: adiabatic equilibrium (`thermo.equilibrate_hp`)

The chamber is modeled as constant pressure with no heat loss, so the products have the same total enthalpy as the reactants:

$$h_\text{products}(T_c, P_c) = \sum_i y_i\, h_i^\text{reactant}(298\ \text{K})$$

where $y_i$ are mass fractions and each $h_i$ includes its heat of formation. Among all product mixtures that have the reactants' element totals and this enthalpy, nature picks the one that **minimizes Gibbs free energy** at $(T, P)$. Cantera solves that minimization over every neutral NASA species made of the elements present.

**Implementation detail.** Cantera needs a starting composition with the correct element ratios. The code seeds the gas with free atoms (C, H, O, N) in the exact proportions, equilibrates at 3000 K so the state is physical, then imposes the reactant enthalpy and solves the HP equilibrium.

**Check:** stoichiometric H₂/O₂ at 1 atm gives 3077 K, CH₄/air 2225 K and CH₄/O₂ 3052 K. Textbook values are about 3080, 2226 and 3050 K (`tests/test_physics.py`).

---

## 2. Nozzle flow (`thermo.rocket`)

Steady, adiabatic flow with no shaft work conserves stagnation enthalpy:

$$h_0 = h + \tfrac{1}{2}v^2 \quad\Rightarrow\quad v = \sqrt{2\,(h_0 - h)}$$

The expansion is assumed isentropic ($s = s_c$). At each pressure, the composition is either held at chamber values (**frozen**) or re-equilibrated (**shifting**).

**Throat.** Mass flux $\rho v$ peaks where the flow reaches the local speed of sound, $v = a$. The code finds the pressure $p_t$ where $v(p) = a(p)$. For shifting flow it uses the *equilibrium* sound speed, $a^2 = (\partial P/\partial\rho)_s$ along an equilibrium isentrope, found by finite difference.

**Characteristic velocity.** This measures combustion performance alone, independent of the nozzle:

$$c^* = \frac{P_c A_t}{\dot m} = \frac{P_c}{\rho_t v_t}$$

**Exit.** Find $p_e$ such that the area ratio $\varepsilon = \frac{\rho_t v_t}{\rho_e v_e}$, which follows from mass conservation ($\dot m$ is the same at throat and exit).

**Thrust coefficient and Isp.** Thrust is the momentum flux plus the pressure imbalance at the exit:

$$F = \dot m v_e + (p_e - p_a)A_e,\qquad C_F = \frac{F}{P_c A_t},\qquad I_{sp} = \frac{c^* C_F}{g_0}$$

### One-gamma $C_F$ (`nozzle.thrust_coefficient`)

For a calorically perfect gas, the same steps give the closed form used in Sutton:

$$C_F = \sqrt{\frac{2\gamma^2}{\gamma-1}\left(\frac{2}{\gamma+1}\right)^{\frac{\gamma+1}{\gamma-1}}\left[1-\left(\frac{p_e}{P_c}\right)^{\frac{\gamma-1}{\gamma}}\right]} + \left(\frac{p_e}{P_c}-\frac{p_a}{P_c}\right)\varepsilon$$

The exit Mach number comes from the isentropic area relation

$$\varepsilon = \frac{1}{M_e}\left[\frac{2}{\gamma+1}\left(1+\frac{\gamma-1}{2}M_e^2\right)\right]^{\frac{\gamma+1}{2(\gamma-1)}}$$

and then $p_e/P_c = \left(1 + \frac{\gamma-1}{2}M_e^2\right)^{-\gamma/(\gamma-1)}$.

---

## 3. Grain burn-back (`grain.PortGeometry`)

**Assumption:** the propellant surface regresses normal to itself at the same speed everywhere. By Huygens' principle, after a web distance $x$ the flame front is the set of points exactly distance $x$ from the original surface. So compute one **distance field** $d(\mathbf{r})$ and then:

$$A_\text{port}(x) = \text{area}\{\,d \le x\,\} \cap \text{casing}, \qquad P(x) = \frac{dA_\text{port}}{dx}$$

**Why $P = dA/dx$:** moving a front of length $P$ outward by $dx$ sweeps an area $P\,dx$.

**Accuracy.** A plain pixel distance transform over-estimates the perimeter early in the burn by about 10%, because the front is a union of tiny circles around boundary pixels. The code avoids that in three steps:

1. Rasterize each shape 4×4 supersampled, giving a "coverage" fraction per cell.
2. Extract the boundary at coverage 0.5 with marching squares (sub-pixel accurate).
3. Measure distances to that curve with a KD-tree.

A circular port then matches $P = 2\pi(r_0 + x)$ to 0.1% mid-burn (`test_tubular_perimeter_matches_circle`).

**Segment burning area.** With uninhibited ends, the core length shrinks as both end faces burn back: $L(x) = L_0 - 2x$. Each end face is a ring of area $A_\text{case} - A_\text{port}(x)$:

$$A_b = P(x)\,(L_0 - 2x) + 2\,[A_\text{case} - A_\text{port}(x)]$$

For a BATES grain this matches the closed form $2\pi r L + 2\pi(R^2 - r^2)$ with $r = r_0 + x$ (`test_bates_burn_area_matches_closed_form`).

---

## 4. Solid motor chamber pressure (`solid.simulate`)

Gas leaves the chamber in milliseconds, much faster than the grain changes, so the chamber is **quasi-steady**: mass generated equals mass leaving.

$$\underbrace{\rho_p\, \dot r\, A_b}_{\text{gas from burning propellant}} = \underbrace{\frac{P_c A_t}{c^*}}_{\text{choked throat}}, \qquad \dot r = a P_c^{\,n}$$

$$\Rightarrow\quad P_c = \left(\frac{\rho_p\, a\, A_b\, c^*}{A_t}\right)^{\frac{1}{1-n}}$$

**Why $n < 1$ is required.** Suppose $P_c$ rises a little. Gas production grows like $P_c^{\,n}$, but outflow grows like $P_c^{\,1}$. If $n < 1$, outflow wins and pressure returns to equilibrium. If $n \ge 1$, the motor runs away. The $1/(1-n)$ exponent also shows why burning area matters so much: with $n = 0.35$, 10% more $A_b$ means about 16% more pressure.

The simulation steps in web distance (not time) and converts with $dt = dx / \dot r$, so fast and slow phases are resolved equally.

**Check:** integrated $\dot m\,dt$ equals the propellant mass loaded, within 3% (`test_solid_mass_balance`).

---

## 5. Hybrid motor O/F drift and how to stop it (`hybrid.py`)

Fuel regression follows the oxidizer mass flux: $\dot r = a\, G_{ox}^{\,n}$ with $G_{ox} = \dot m_{ox}/A_\text{port}$. The fuel mass flow is $\dot m_f = \rho_f\, \dot r\, P\, L$. Therefore

$$\frac{O}{F} = \frac{\dot m_{ox}}{\rho_f\, a\, (\dot m_{ox}/A)^n\, P\, L} = \frac{\dot m_{ox}^{\,1-n}\, A^{n}}{\rho_f\, a\, P\, L}$$

**Why port shape can't fix the drift.** Any port that grows into a scaled copy of itself has $A \propto s^2$ and $P \propto s$ (where $s$ is a length scale), so $P \propto A^{1/2}$. Substituting:

$$\frac{O}{F} \propto A^{\,n - 1/2}$$

For paraffin, $n = 0.62 > 0.5$, so O/F always rises as the port opens. Shaped ports start with a much larger perimeter, then round off toward a circle. That front-loads fuel flow, which helps thrust, but makes the O/F swing *larger*. The star sweep in the README confirms it: no star beats the plain tube.

**Remedy 1: throttle the oxidizer.** Solve the O/F equation for $\dot m_{ox}$ at a target O/F:

$$\dot m_{ox}(x) = \left(\frac{(O/F)\,\rho_f\, a\, P(x)\, L}{A(x)^{n}}\right)^{\frac{1}{1-n}}$$

**Remedy 2: grade the fuel.** Keep $\dot m_{ox}$ fixed and let the regression coefficient vary with depth, which a printed fuel grain can do:

$$\frac{a(x)}{a_0} = \frac{\dot m_{ox}^{\,1-n}\, A(x)^{n}}{(O/F)\,\rho_f\, a_0\, P(x)\, L}$$

Both are inverse solutions of the same model, so in simulation they hold O/F exactly flat *by construction*. The useful output is the schedule: how much throttle range, or how much regression-rate grading, the hardware must actually deliver.

### Regression along the port (`hybrid.simulate_axial`)

The averaged law gives every slice of the grain the same $G_{ox}$. But fuel burned near the head end joins the flow, so the total mass flow $\dot m(z)$ grows along the port. With the regression following the total flux $G = \dot m/A$, the fuel added over a length $dz$ is

$$\frac{d\dot m}{dz} = \rho_f\, a\, \left(\frac{\dot m}{A}\right)^{n} P$$

Inside one axial cell, $A$ and $P$ are fixed, so this separates:

$$\dot m^{-n}\, d\dot m = \rho_f\, a\, P A^{-n}\, dz \quad\Rightarrow\quad \dot m_\text{out}^{\,1-n} = \dot m_\text{in}^{\,1-n} + (1-n)\,\rho_f\, a\, P A^{-n}\, \Delta z$$

Down the whole port, $\dot m^{1-n}$ is a cumulative sum starting from $\dot m_{ox}$ at the head end, with no step-size error in $z$. Each cell's regression rate is its added fuel divided by $\rho_f P\, \Delta z$, and each cell burns back on its own copy of the port's $A(x)$ and $P(x)$ curves.

**Matching the coefficient.** Published laws like Karabeyoglu's are fits against the averaged oxidizer flux. A total-flux law with the same $n$ needs a smaller $a'$. At ignition the port is uniform, so the closed form holds along the whole length $L$, and requiring the same starting fuel flow $\dot m_f$ gives

$$a' = \frac{(\dot m_{ox} + \dot m_f)^{1-n} - \dot m_{ox}^{1-n}}{(1-n)\,\rho_f\, P A^{-n} L}$$

(`hybrid.total_flux_law`). That is a modeling choice for comparing the two, not a fit to data.

**Checks:** the cell-by-cell closed form matches `scipy.integrate.solve_ivp` to $10^{-8}$. Using the oxidizer flux in every cell reproduces the averaged model exactly. The fuel burned equals the volume the port has grown by, within 0.2% (`tests/test_physics.py`).

**Units.** The literature writes $\dot r[\text{mm/s}] = a\, G[\text{g/cm}^2\text{s}]^n$. Since $G_\text{cgs} = G_\text{SI}/10$, the SI coefficient is $a_\text{SI} = 10^{-3}\, a \cdot 10^{-n}$ (`test_regression_unit_conversion`).

---

## 6. Chapman–Jouguet detonation (`rde.cj_state`)

In the wave frame, reactants enter at speed $w_1$ and products leave at $u_2$. The conservation laws across the wave are:

| | |
|---|---|
| mass | $\rho_1 w_1 = \rho_2 u_2$ |
| momentum | $P_1 + \rho_1 w_1^2 = P_2 + \rho_2 u_2^2$ |
| energy | $h_1 + \tfrac12 w_1^2 = h_2 + \tfrac12 u_2^2$ |

Mass and momentum together give the **Rayleigh line**, which is straight in the $(P, 1/\rho)$ plane. Energy gives the **Hugoniot curve** of possible burned states. For most wave speeds the line cuts the curve twice or not at all. The slowest self-sustaining detonation is where the line is **tangent** to the equilibrium Hugoniot, and tangency is equivalent to

$$u_2 = a_{2,\text{eq}}$$

(the products leave at exactly their equilibrium sound speed). The code solves momentum and energy for $(T_2, P_2)$ with that condition and full equilibrium chemistry.

**Check:** stoichiometric H₂/O₂ 2837 m/s (literature 2836), H₂/air 1969 (1968), C₂H₄/air 1824 (1825), all at 1 atm and 298 K.

---

## 7. Why detonate? Cycle efficiencies (`rde.OneGamma`)

To compare cycles cleanly, model the gas with one $\gamma$ and a heat release $q$. For a perfect gas the CJ Mach number is

$$M_{CJ} = \sqrt{1+H} + \sqrt{H}, \qquad H = \frac{(\gamma^2-1)\,q}{2\gamma R T_1}$$

**Calibration.** Squaring $M - \sqrt H = \sqrt{1+H}$ gives $\sqrt H = (M^2-1)/(2M)$. So from the equilibrium CJ speed and the products' $\gamma$, the code backs out the $q$ that makes this model detonate at the right speed.

**CJ state in the one-gamma model:**

$$\frac{P_{CJ}}{P_1} = \frac{1+\gamma M^2}{1+\gamma}, \qquad \frac{T_{CJ}}{T_1} = \frac{(1+\gamma M^2)^2}{(1+\gamma)^2 M^2}$$

**Fickett–Jacobs cycle.** Detonate, expand isentropically back to $P_1$, then reject heat at constant pressure. The temperature after expansion is

$$T_4 = T_{CJ}\left(\frac{P_1}{P_{CJ}}\right)^{\frac{\gamma-1}{\gamma}} = \frac{T_1}{M^2}\left(\frac{1+\gamma M^2}{1+\gamma}\right)^{\frac{\gamma+1}{\gamma}}$$

$$\eta_{FJ} = 1 - \frac{c_p\,(T_4 - T_1)}{q}$$

**Other cycles.** With precompression ratio $\pi$, the code compresses isentropically first. Brayton (constant-pressure burn) gives $\eta = 1 - \pi^{-(\gamma-1)/\gamma}$, which is **zero with no precompression**. Humphrey (constant-volume burn) and Fickett–Jacobs both do useful work even at $\pi = 1$, because the combustion itself raises the pressure. That is "pressure-gain combustion", the reason RDEs exist.

**Result for H₂/air:** at $\pi = 1$, FJ 25%, Humphrey 23%, Brayton 0%. At $\pi = 10$: 50%, 48%, 36%.

These are ideal-cycle numbers. Real RDEs lose a lot to unsteady flow, mixing and non-ideal waves.

---

## 8. RDE geometry (`rde.bykovskii_sizing`)

Rotating detonation waves need enough fresh mixture ahead of them. Bykovskii, Zhdan & Vedernikov (J. Propulsion & Power 22(6), 2006) correlated experiments in terms of the **detonation cell size** λ:

| Quantity | Correlation |
|---|---|
| critical fill height | $h^* = (12 \pm 5)\,\lambda$ |
| min channel width | $\approx 0.2\,h^*$ |
| min chamber length | $\approx 2\,h^*$ |
| min diameter | $\approx 40\,\lambda$ |
| spacing between waves | $\ell = (7 \pm 2)\,h^*$ |

The number of waves is about $\pi d / \ell$, and wave frequency is $n\,D/(\pi d)$, using $D \approx 0.85\,D_{CJ}$ since real waves run slow.

**λ cannot be computed reliably from first principles. It has to come from experiments** (for example the Caltech Detonation Database). That is why the README sweeps λ rather than quoting one value.

---

## 9. Bell nozzle contour (`nozzle.bell_contour`)

This is Rao's thrust-optimized parabola approximation, built in three pieces:

1. A circular arc of radius $1.5 R_t$ upstream of the throat.
2. A circular arc of radius $0.382 R_t$ downstream, ending at the inflection angle $\theta_n$.
3. A quadratic Bézier curve from that point $N$ to the exit point $E = (L_n, R_e)$.

The Bézier control point $Q$ is where the tangent lines at $N$ (slope $\tan\theta_n$) and at $E$ (slope $\tan\theta_e$) intersect. That makes the curve leave $N$ and arrive at $E$ at exactly those angles. The length is a fraction (80% here) of a 15° cone of the same area ratio. $\theta_n$ and $\theta_e$ come from Rao's charts for the chosen area ratio.

---

## What the models leave out (say this before someone else does)

- **Ballistics:** erosive burning, ignition and tail-off transients, two-phase flow losses, nozzle erosion, and heat loss to the walls.
- **Hybrids:** `simulate_axial` resolves the flux along the port, but its total-flux coefficient is matched to the averaged law at ignition rather than fit to data. Both models ignore the oxidizer-tank blowdown and the injector.
- **Thermochemistry:** gas-phase only (no condensed products such as Al₂O₃ or soot). The paraffin heat of formation is uncertain, but a test shows it moves c* by less than 1%.
- **Detonation:** the one-gamma cycle analysis is an idealization. Real RDEs run 10–20% below CJ speed, and the sizing depends on an empirical cell size.
