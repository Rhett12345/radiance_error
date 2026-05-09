# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Satellite radiance transfer-model sensitivity experiment. Given transfer models between FY-4A/B AGRI channels, this code propagates input-radiance uncertainty through the models via Monte Carlo simulation and reports output error statistics.

## How to run

```bash
cd ~/transfer_model
python run_sensitivity.py
python run_sensitivity.py --data "fy4a_fy4b_convolution/*_sat_rad.csv" --coeff transfer_coeff_fy4a_fy4b_v1.csv --outdir ./output
```

Programmatic import:

```python
from transfer_error import main
main(["data1.csv", "data2.csv"], "coeffs.csv", "./output/")
```

Overwrite `N_MC` in `config.py` (default 500) for quick smoke tests.

## Architecture

The package has six modules with a strict dependency order:

```
config  →  physics  →  sensitivity  →  main
                   ↘  io_utils    ↗
                   ↘  plotting    ↗
```

**`config.py`** — single source of truth. All physical constants (C1_UM, C2_UM, JAC_SCALE), all 48 channel wavelengths for FY-4A/B/C, channel-type classification (`reflective` = ch01–ch06; `ir` = ch07/ch09–ch15; ch08 excluded as low-gain duplicate), and perturbation grids. Everything that might need tuning lives here.

**`physics.py`** — pure functions, no side effects. Jacobian `L_λ = L_ν × 1e8 / λ²` (two factors of 1e4: cm⁻²→m⁻² and |dν/dλ|), Planck radiance, brightness temperature inversion, and `dL/dTb` derivative. All wavelengths in µm. Radiance always W·m⁻²·sr⁻¹·µm⁻¹ after Jacobian.

**`io_utils.py`** — loads raw radiance CSVs (columns like `fy4a_ch01_Radiance`, original units W·cm⁻²·sr⁻¹·cm) and the transfer-coefficient CSV. Leaves Jacobian transform to the caller. Handles the fact that different CSV files contain different channel subsets (reflective in `*3_sat_rad.csv`, IR in plain `*_sat_rad.csv`).

**`sensitivity.py`** — core experiment. `apply_model()` evaluates linear (`coeff1·x + intercept`) or quadratic (`+ coeff2·x²`) transfer functions. `_run_one_channel()` vectorizes the MC: generates `(N_MC, N_scenes)` noise array, applies the model, aggregates dy and dTb statistics. Noise model differs by channel type:
- Reflective: `σ = perturbation_frac × x` (scene-dependent)
- IR: `σ = |dL/dTb(Tb_typical)| × ΔK` (constant, computed from mean scene radiance)

**`plotting.py`** — two standalone figure functions. Reflective plot: `rel_err_mean (%)` vs perturbation fraction (%). IR plot: `dTb_mean (K)` vs ΔK. Both at 300 dpi.

**`main.py`** — orchestrator. Entry point `main(data_csv_list, coeff_csv, output_dir)` loads data → loads coefficients → runs experiment → saves CSV + two PNGs.

## Key numeric details

- **C1_UM derivation**: `2hc² × 1e24`, NOT `2h(c × 1e6)²`. The 1e24 factor comes from λ⁻⁵ conversion (1e30) times dλ conversion (1e-6). Getting this wrong produces radiance 12 orders of magnitude too small.
- **Jacobian**: Both cm⁻²→m⁻² (×1e4) and |dν/dλ| (×1e4/λ²) are baked into the single factor `1e8/λ²`.
- **NaN handling**: `brightness_temperature()` returns NaN for non-positive radiance; downstream aggregation uses `nanmean`/`nanpercentile`.
- **Exp overflow guard**: Both `planck_radiance()` and `dL_dTb()` clip `c2/(λT)` to 500 to avoid overflow for very cold scenes.

## Relationship to transfer_refactor

The sibling project `transfer_refactor/` fits the transfer models (using sklearn). This project consumes its output CSV (`transfer_coeff_fy4a_fy4b_v1.csv`) and evaluates how input noise propagates through those fitted models. The wavelength catalog and Jacobian conversion in `config.py` and `physics.py` mirror `transfer_refactor/fy4_transfer/channel_catalog.py` and `io.py`.

## Dependencies

numpy, scipy, matplotlib. No sklearn — all model evaluation is explicit numpy arithmetic.
每次回复前需要输出“打报告”三个字符