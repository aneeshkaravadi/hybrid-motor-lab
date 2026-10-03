# For the X-Bow Systems team

You additively manufacture solid propellant. Printing removes the casting constraints on grain geometry, so the useful tool is fast burn-back for any shape.

1. **Arbitrary-geometry burn-back** (`hml/grain.py`). Describe a port with a function, get $A_b(x)$ for multi-segment grains with inhibited or open ends. Checked against the BATES closed form (within 1%) and the exact tubular perimeter (0.1%).
2. **Thrust-profile shaping.** Four grain designs in the same 54 mm case give four profiles: neutral, progressive, fast star with sliver tail-off, and boost–sustain. ([figure](figures/solid_thrust_shaping.png))
3. **Fitting a burn-rate law to test data.** `examples/compare_static_fire.py` fits $a$ and $n$ to a measured thrust curve. On a synthetic check it recovers the true values exactly. The honest validation is to fit on one motor and *predict* another made from the same propellant.

**What it doesn't do:** erosive burning, ignition transients, grain structural loads and two-phase losses. The propellant constants in the examples are illustrative.

**A 3-week project I could do for you, remote:** a grain-geometry optimizer for a target thrust-time curve. Given a case envelope and your $a$, $n$, $c^*$ (shared under NDA or replaced with surrogates), it searches printable port families and reports the closest match, the sliver fraction and peak pressure.

— Aneesh Karavadi · aneesh.karavadi@gmail.com
