# DfAM report: `cad/bell_nozzle_e8.stl`

Measured with a trimesh-based DfAM script ([dfam-check](https://github.com/earthtojake/text-to-cad), MIT), comparing against the **PBF-LB metal (SLM/DMLS)** column of its process-limits table (Hubs / EOS design guides). A machine- or alloy-specific datasheet would override these defaults.

| Check | Measured | Limit (metal PBF) | Status |
|---|---|---|---|
| Watertight, single body | yes, 1 body | required | ✅ pass |
| Units | bbox 111 × 91 × 91 mm | millimetres | ✅ pass |
| Minimum wall thickness | 2.61 mm (5th percentile 2.84 mm) | ≥ 0.4 mm supported / 0.5 mm unsupported | ✅ pass |
| Support area, as modelled (axis horizontal) | 18.0% of surface below 45° | self-supporting at 45° | ❓ reorient |
| Support area, best orientation (axis vertical, exit up) | **2.5%** of surface | self-supporting at 45° | ✅ preferred |

**Build orientation.** Stand the nozzle on its flange with the exit pointing up. The bell wall flares out at 10–27° from vertical, well inside the 45° self-supporting limit. The 45° convergent cone is right at the limit. The little remaining support is under the flange and at the throat.

**Not checked by this tool:**
- powder removal from internal passages (none in this part)
- residual stress and distortion of the thin divergent skirt
- surface roughness at the throat, which matters for heat transfer and erosion
- alloy choice for the thermal load

The 3 mm wall is a placeholder for a heat-sink test article, not a thermal design.

Reproduce:

```bash
python dfam_tool.py measure cad/bell_nozzle_e8.stl --angle-limit 45
python dfam_tool.py orientations cad/bell_nozzle_e8.stl --angle-limit 45
```
