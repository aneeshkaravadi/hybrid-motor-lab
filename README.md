# hybrid-motor-lab

[![tests](https://github.com/aneeshkaravadi/hybrid-motor-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/aneeshkaravadi/hybrid-motor-lab/actions/workflows/ci.yml)

Rocket motor models in Python, from the combustion chemistry all the way to a nozzle you can 3D print. It covers solid motors, hybrids, and a detour into rotating detonation engines.

<!-- TODO(Aneesh): photo of the club rocket / motor here, e.g.
![Our L1 rocket on the pad](docs/photos/rocket_on_pad.jpg)
-->

## Why I built this

I led propulsion design for my high school's rocketry club, and for most of that time a motor was a thrust curve on a website that I picked from and trusted. I wanted to be able to compute that curve myself, starting from the grain shape and the propellant. Once the solid-motor part worked, hybrids were the obvious next step, because a 3D-printed fuel grain can have any port shape you want. I got into detonation engines after reading about companies flying them and wanting to understand why anyone would bother.

## What's in it

- `thermo.py`: equilibrium combustion and nozzle expansion with [Cantera](https://cantera.org), basically the NASA CEA "rocket" problem
- `grain.py`: burn-back for any port shape you can describe (tube, star, finocyl, wagon wheel, or your own function)
- `solid.py` and `hybrid.py`: quasi-steady ballistics, including fuel regression $\dot r = aG^n$ for hybrids
- `rde.py`: Chapman–Jouguet detonation states, ideal cycle comparison, and rough RDE sizing
- `nozzle.py`: a Rao-style bell contour that exports straight to STEP

The math behind all of it is written out in [DERIVATIONS.md](DERIVATIONS.md).

## The result I didn't expect

I assumed printed port shapes would let you hold a hybrid's O/F ratio steady through the burn, since that was the whole appeal. They don't, at least not for paraffin. Every star, finocyl and wagon wheel I tried front-loaded the fuel flow, which helped thrust a little (3 to 5% more impulse), but made the O/F swing much worse than a plain tube (70 to 88% instead of 21%).

![O/F drift for four port shapes](docs/figures/hybrid_of_drift.png)

It took me a while to see why. For any port that grows into a scaled copy of itself, O/F goes like $A^{n-1/2}$, and paraffin has $n = 0.62$, so O/F has to rise no matter what shape you start with. I ran a sweep of 20 star designs to be sure and none of them beat the tube.

What does work is running the model backwards. To hold O/F flat you either throttle the oxidizer down over the burn (0.73 to 0.41 kg/s for my test case), or you print a fuel whose regression rate increases by about 24% from the port outward. That second one is a fuel grain you could only make by printing it.

![O/F remedies](docs/figures/of_remedies.png)

## Detonation, briefly

The detonation speeds from my CJ solver land within 1 m/s of published values for H₂/O₂, H₂/air and C₂H₄/air, which was the moment I started trusting it. The interesting part is the cycle comparison. For H₂/air with no compressor at all, an ideal detonation cycle still gets about 25% thermal efficiency while a normal constant-pressure (Brayton) cycle gets zero, because the detonation does its own compression. That's the argument for RDEs in one plot.

<img src="docs/figures/cycle_efficiency.png" width="60%">

The sizing part depends on the detonation cell size, which you can't really calculate and have to get from experiments, so I sweep it instead of pretending to know it.

## Solid motors

Same 54 mm case, four grain designs, four very different thrust curves. The propellant numbers here are illustrative, not a real formulation.

![Solid thrust shaping](docs/figures/solid_thrust_shaping.png)

<!-- TODO(Aneesh): once you have real data, add a section here, e.g.
## Checking it against real motors
Fit a and n on one motor with examples/compare_static_fire.py, then predict a second motor of the same propellant
(thrustcurve.org .eng file or club static-fire CSV in data/static_fires/). Show the overlay plot and the impulse error.
-->

## A nozzle that ends in hardware

`nozzle.py` builds the bell contour and exports it with build123d as [`cad/bell_nozzle_e8.step`](cad/bell_nozzle_e8.step). I ran the exported mesh through a DfAM check for metal powder-bed printing ([report](docs/dfam_report.md)). Walls pass easily, and standing it on its flange cuts the support area from 18% to 2.5%.

<!-- TODO(Aneesh): screenshot of the STEP open in SolidWorks/Fusion, e.g.
<img src="docs/photos/nozzle_in_solidworks.png" width="60%">
-->

## Things I got wrong along the way

- My first burn-back version used a plain pixel distance transform, and it overestimated the burning perimeter by about 10% early in the burn, because the flame front ends up being a bunch of tiny circles around boundary pixels. I switched to finding the boundary at sub-pixel accuracy and measuring distances to that, which got it to within 0.1% of the exact circle.
- The first nozzle STEP file was in metres while CAD programs assume millimetres, so it opened 1000 times too small, and the STL was 78 MB. Building the geometry in mm with spline walls fixed both (the STEP is now 69 KB).
- I labelled one of my solid grain cases "near-neutral star" before actually looking at the curve. It isn't neutral at all, so it's now labelled for what it does.
- The exit-pressure solver crashed with a negative temperature because I let it search down to absurdly low pressures, so the bracket now scales with the area ratio.

## Running it

```bash
pip install -e ".[dev,cad]"
pytest -q                          # 18 checks against known answers, a few seconds
python examples/make_figures.py    # regenerates every figure and number above
```

The tests compare against things I could look up independently: textbook flame temperatures, published CJ speeds, the exact BATES burning area, isentropic flow tables, and a mass balance on the solid motor.

## What's next

Things I want to add are tracked in [issues](https://github.com/aneeshkaravadi/hybrid-motor-lab/issues): validation against real static-fire data, erosive burning, and oxidizer tank blowdown for the hybrid model.

---

Aneesh Karavadi, engineering at UNT (TAMS). I used Claude Code to write a lot of the implementation, but I picked the problems and the validation targets, and I checked the results, so the mistakes are mine.
