# Static-fire and flight data

Drop measured thrust curves here to validate the solid-motor model.

Two formats are read by `examples/compare_static_fire.py`:

- **CSV** with columns `time_s,thrust_N` (from a load cell)
- **RASP `.eng`** files, the format thrustcurve.org publishes for commercial motors

Each data file needs a matching `<name>.json` describing the grain, nozzle and
propellant so the model can be run for the same motor:

```json
{
  "segments": [{"shape": "tubular", "diameter_mm": 54, "core_mm": 20, "length_mm": 70, "count": 4}],
  "throat_mm": 17.0,
  "area_ratio": 6.0,
  "propellant": {"density": 1700, "a": 3.5e-5, "n": 0.35, "cstar": 1500, "gamma": 1.2}
}
```

`a` and `n` are rarely published for commercial reloads. The script can fit
them to the measured curve (`--fit`), and then predict a *different* motor made
from the same propellant. That cross-prediction is the honest validation; fitting
one curve and plotting it back proves very little.
