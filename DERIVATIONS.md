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

### Condensed products (`thermo.Equilibrium`, `thermo.rocket_multiphase`)

Aluminum burns to Al₂O₃, which is liquid in the chamber and freezes in the nozzle at 2327 K. Gibbs minimization still applies, but now over the gas plus every NASA condensed species made of the elements present, each as its own pure phase. Three details, all following CEA:

- **Temperature ranges.** A condensed species only takes part between its data's temperature limits, so liquid alumina exists above 2327 K and solid below. Extrapolating each phase's polynomial outside its range could make the wrong phase win.
- **No volume.** CEA treats condensed products as taking up no volume. Cantera's default density for a pure condensed phase (0.001 kg/m³) would add $(P - P_\text{ref})\,v$ to its Gibbs energy, about $3\times10^8$ J/mol at 34 bar, and alumina could never form. Each phase gets a negligible molar volume instead.
- **HP and SP by temperature.** For a trial $T$, Cantera's multiphase solver finds the $(T, P)$ equilibrium. A root find on $T$ then matches the reactants' enthalpy (chamber) or the chamber entropy (nozzle).

**Freezing in the nozzle.** At 2327 K the mixture's entropy jumps by the latent heat of fusion, so for some pressures no single temperature matches the chamber entropy. Physically, the temperature holds at 2327 K while the alumina freezes. The model takes the two sides of the jump and mixes them by the lever rule, $w = (s - s_-)/(s_+ - s_-)$.

**Throat.** With condensed products in equilibrium, the flow chokes where the mass flux $\rho v$ peaks, with $\rho$ the mass per gas volume. That is where the velocity equals the mixture's equilibrium sound speed. The code maximizes $\rho v$ along the isentrope directly.

**Checks:** NASA's CEA Example 5 (RP-1311 Part II) is an aluminized AP composite with 9% Al. The model's HP temperatures are within 5 K of CEA's at all five pressures from 500 to 5 psia, and the product mole fractions (Al₂O₃(L), HCl, H₂, CO, N₂, H₂O, CO₂) are within 1%. With no aluminum, nothing condenses and `rocket_multiphase` matches the gas-only `rocket` to $10^{-4}$ in $c^*$ and $I_{sp}$, even though the two find the throat differently (`tests/test_physics.py`).

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

### Erosive burning (`solid.simulate_erosive`)

Gas flowing fast along the port raises the heat transfer to the burning surface, so the propellant burns faster than $aP_c^n$. Mukunda & Paul (Combustion and Flame 109, 1997) showed that one curve fits the data for most composite and double-base propellants when written in terms of

$$g_0 = \frac{G}{\rho_p r_0}, \qquad Re_0 = \frac{\rho_p r_0 d}{\mu}, \qquad g = g_0 \left(\frac{Re_0}{1000}\right)^{-0.125}$$

where $r_0 = aP_c^n$ is the pressure-only rate, $G$ the mass flux in the port, $d$ the port's hydraulic diameter $4A/P$, and $\mu$ the gas viscosity. $g_0$ compares the flow along the surface with the flow leaving it. Their Eq. 12 is

$$\frac{r}{r_0} = 1 + 0.023\left(g^{0.8} - g_{th}^{0.8}\right) \quad \text{for } g > g_{th} = 35, \qquad \text{else } 1$$

**Solving it.** Each segment is split into axial cells, each with its own web, port area $A_j$ and perimeter $P_j$ from the burn-back curves. For a trial $P_c$, march from the closed head end. The flow arriving at cell $j$ is everything made upstream, so $G_j = \dot m_{j-1}/A_j$. That sets $r_j = r_0\,\eta(g_j)$, and the cell adds $\rho_p r_j P_j \Delta z$ to the flow. End faces add gas where they sit and burn at $r_0$, since the flow passes them rather than along them. A root solve finds the $P_c$ where the total equals $P_c A_t/c^*$. Each cell then burns back by $r_j\,\Delta t$, with $\Delta t$ set so the fastest cell moves 0.1 mm.

**When the port chokes.** The most mass flux any section can pass is about what the throat passes, $P_c/c^*$, since it's the same gas from the same chamber. If the march asks for more than that somewhere in the port, the port itself chokes, the pressure along the port is far from uniform, and a 0-D chamber model doesn't apply. The result reports that ratio (`port_choke_ratio`).

**Checks:** the correlation by hand. With erosion off, the cell model reproduces the closed form exactly for BATES (whose cells burn away from the ends) and star grains. With erosion on, mass still balances within 3%, the head-end grain of a BATES stack doesn't erode while the nozzle-end one burns faster toward the nozzle, and a port narrower than the throat gets flagged (`tests/test_physics.py`).

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

## 6. Nitrous oxide tank and injector (`tank.py`)

Nitrous oxide is stored as a saturated liquid under its own vapor, so the tank pressure is the vapor pressure at the tank temperature, about 5 MPa at 20 °C. No pressurant gas is needed.

**Tank state.** The tank is rigid (volume $V$) and adiabatic, and holds mass $m$ and internal energy $U$. Liquid and vapor are in equilibrium at one temperature $T$, so with vapor mass fraction $x$:

$$\frac{V}{m} = (1-x)\,v_l(T) + x\,v_g(T), \qquad \frac{U}{m} = (1-x)\,u_l(T) + x\,u_g(T)$$

That's two equations for $T$ and $x$. CoolProp solves them directly from density and energy (a "DU flash") with the Lemmon & Span equation of state for N₂O.

**Draining.** Liquid leaves from the bottom and carries its enthalpy, flow work included:

$$\frac{dm}{dt} = -\dot m, \qquad \frac{dU}{dt} = -\dot m\, h_l$$

**Why the tank cools.** Write the same drain with $T$ as the unknown. Removing $dm_\text{out}$ of liquid while $dm_e$ evaporates ($dm_l = -dm_\text{out} - dm_e$, $dm_g = dm_e$), the fixed volume and the energy balance give

$$v_l\,dm_l + v_g\,dm_g + B\,dT = 0, \qquad u_l\,dm_l + u_g\,dm_g + C\,dT = -h_l\,dm_\text{out}$$

where $B = m_l v_l' + m_g v_g'$ and $C = m_l u_l' + m_g u_g'$ use slopes along the saturation curve. Eliminating $dm_e$ and using $u_{fg} + p\,v_{fg} = h_{fg}$:

$$\frac{dT}{dm_\text{out}} = -\frac{v_l\, h_{fg}}{v_{fg}\, C - u_{fg}\, B}$$

The numerator is the latent heat of the vapor that has to form to fill the space the liquid left, and that's what cools the tank. The test integrates this ODE on its own and compares it with the tank's mass-and-energy bookkeeping. After draining 2 kg from a 5 L tank, the two agree within 0.01 K (`tests/test_tank.py`).

**Injector.** The mass flow is $\dot m = C_d A\, G$, with three standard models for the ideal mass flux $G$ from the tank ($p_1$) to the chamber ($p_2$):

- **SPI** (single-phase incompressible): $G = \sqrt{2\rho_l\,(p_1 - p_2)}$. It treats the nitrous as liquid the whole way through, so it over-predicts.
- **HEM** (homogeneous equilibrium): the liquid flashes to a liquid-vapor mixture that expands isentropically, $G = \rho_2\sqrt{2(h_1 - h_2)}$ at $s_2 = s_1$. As $p_2$ drops, $G$ rises and then peaks, near $0.7\,p_1$ for nitrous at 20 °C. Past that, the mixture's density would fall faster than its speed rises, so the flow is choked and stays at the peak. With a tiny pressure drop almost no vapor forms, and HEM approaches SPI (a test checks this).
- **Dyer** (non-homogeneous non-equilibrium; Dyer et al., AIAA 2007-5702): bubbles need time to grow, so real flow through a short orifice falls between the two, $G = \frac{k\,G_\text{SPI} + G_\text{HEM}}{1 + k}$ with $k = \sqrt{(p_1 - p_2)/(p_v - p_2)}$. In a self-pressurized tank $p_1 = p_v$, so $k = 1$ and it's the plain average.

**Coupling to the motor.** The injector flow depends on $P_c$, and $P_c$ on the total flow, so every step solves $P_c = (\dot m_{ox}(P_c) + \dot m_f)\,\eta_{c^*}\, c^*/A_t$ with a root finder (`tank.simulate_blowdown`).

**Liquid nitrous in the Cantera table.** CoolProp and the NASA tables put the zero of enthalpy in different places. The liquid's NASA-scale enthalpy is the NASA value for N₂O gas at 298.15 K plus CoolProp's difference between saturated liquid at the tank temperature and the near-ideal gas at 298.15 K and 100 Pa. At 20 °C that's 255 kJ/kg below the gas, mostly the heat of vaporization (170 kJ/kg).

---

## 7. Chapman–Jouguet detonation (`rde.cj_state`)

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

## 8. Why detonate? Cycle efficiencies (`rde.OneGamma`)

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

## 9. RDE geometry (`rde.bykovskii_sizing`)

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

## 10. Bell nozzle contour (`nozzle.bell_contour`)

This is Rao's thrust-optimized parabola approximation, built in three pieces:

1. A circular arc of radius $1.5 R_t$ upstream of the throat.
2. A circular arc of radius $0.382 R_t$ downstream, ending at the inflection angle $\theta_n$.
3. A quadratic Bézier curve from that point $N$ to the exit point $E = (L_n, R_e)$.

The Bézier control point $Q$ is where the tangent lines at $N$ (slope $\tan\theta_n$) and at $E$ (slope $\tan\theta_e$) intersect. That makes the curve leave $N$ and arrive at $E$ at exactly those angles. The length is a fraction (80% here) of a 15° cone of the same area ratio. $\theta_n$ and $\theta_e$ come from Rao's charts for the chosen area ratio.

---

## What the models leave out (say this before someone else does)

- **Ballistics:** ignition and tail-off transients, two-phase flow losses, nozzle erosion, and heat loss to the walls. Erosive burning is an empirical correlation with an estimated gas viscosity. End faces burn at the pressure-only rate, and the pressure is taken as uniform along the port, which holds unless the port nears choking (flagged).
- **Hybrids:** `simulate_axial` resolves the flux along the port, but its total-flux coefficient is matched to the averaged law at ignition rather than fit to data. The nitrous tank is adiabatic and always in equilibrium. Real tanks lag behind equilibrium, and the walls give heat back to the cooling liquid. The burn also stops when the liquid runs out instead of burning the vapor left behind.
- **Thermochemistry:** condensed products stay in equilibrium with the gas, and the droplets are assumed to keep up with it in speed and temperature, so aluminized $I_{sp}$ is an upper bound. Real motors lose a few percent to particle lag. The paraffin heat of formation is uncertain, but a test shows it moves c* by less than 1%.
- **Detonation:** the one-gamma cycle analysis is an idealization. Real RDEs run 10–20% below CJ speed, and the sizing depends on an empirical cell size.
