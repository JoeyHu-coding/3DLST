# 3DLST Reproducible Analysis Code

This repository contains a compact public code package for the 3DLST analysis
of building footprint, building height, and hottest-month land surface
temperature (LST) anomalies across global cities.

The package is intentionally small. It keeps the study-facing statistical
methods and omits local drafting utilities, Word-generation scripts, temporary
QA scripts, exploratory notebooks, and raw geospatial preprocessing helpers.
Those omitted files are not needed to reproduce the core city-level estimates
once the analysis-ready grid matrix is available.

## Files

- `run_analysis.py` - command-line entry point that coordinates the selected
  pipeline stages and writes the run manifest.
- `lst_common.py` - shared configuration, input validation, feature creation,
  model tiers, regression helpers, and output utilities.
- `core_analysis.py` - city-level total building-volume response (BVR),
  footprint response (BVR-F), height response (BVR-H), footprint-height
  contrast, contribution accounting, and pooled fixed-effect diagnostics.
- `mechanism_analysis.py` - sequential adjustment, surface-context path models,
  grid-level interactions, city-level moderators, and leave-one-climate-group-
  out ridge reconstruction.
- `robustness_analysis.py` - same-volume horizontal versus vertical morphology
  contrast, the 12 footprint-threshold by minimum-grid checks, control-variable
  diagnostics, and spatial block bootstrap.
- `nonlinear_analysis.py` - grouped out-of-fold HGBR prediction, model-based
  Monte Carlo Shapley attribution, and accumulated local effects.
- `climate_typology.py` - risk-first Type I-IV assignment and Global
  North/Global South composition statistics.
- `sensor_validation.py` - analysis-ready-table checks for spatial support,
  Landsat-GSHTD correspondence, and viewing-angle interactions.
- `tests/` - compact synthetic tests for the public statistical functions.
- `requirements.txt` - minimal Python dependencies.
- `README.md` - this guide.

## Required Input

The scripts expect an analysis-ready 1 km grid matrix as a Parquet file. The
matrix must include these core columns:

```text
UID, Ts_anomaly, BF, MBH_m, lnF, lnH, lnV, eligible_hvca_main,
terrain_mean_m, slope_mean_deg, relief_p90_p10_m, p_water_1km,
grid_col, grid_row
```

Additional optional columns are used when present for surface-context and
moderator diagnostics:

```text
p_veg_1km, p_bare_1km, p_crop_1km, p_built_1km, core_veg_30m_1km,
ai_built_1km, VF, VVD_m, MVH_m, population, GDP, GDPpc, KG,
Country, Continent, climate_macro, income_group, climate_income_regime,
city_size_class, fid_shared_by_multiple_uid
```

The nonlinear, typology, and sensor modules accept separate analysis-ready
city or matched-observation tables. Their function docstrings define the
required columns. Raw image downloading, raster construction, manuscript
assembly, and reviewer-response utilities are intentionally outside this
public package.

## Installation

```bash
python -m pip install -r requirements.txt
```

Run the compact test suite with:

```bash
python -m pytest -q
```

## Run

From the repository root:

```bash
python run_analysis.py \
  --matrix data/hierarchical_vca_analysis_matrix.parquet \
  --output-dir results/3dlst_all \
  --steps all
```

Run only the core city-level models:

```bash
python run_analysis.py \
  --matrix data/hierarchical_vca_analysis_matrix.parquet \
  --output-dir results/3dlst_core \
  --steps core
```

For a fast smoke test:

```bash
python run_analysis.py \
  --matrix data/hierarchical_vca_analysis_matrix.parquet \
  --output-dir results/3dlst_smoke \
  --steps all \
  --max-cities 10 \
  --bootstrap-replicates 5
```

## Main Outputs

The entry point writes CSV and JSON files under the selected output directory:

- `core/city_total_bvr.csv` - city-level total building-volume response.
- `core/city_pathway_contrast.csv` - city-level BVR-F, BVR-H, and Delta.
- `core/city_contribution_decomposition.csv` - contribution reconstruction of
  the total BVR from footprint and height components.
- `core/summary_by_model_tier.csv` - distribution summaries by adjustment tier.
- `core/pooled_fixed_effects.csv` - pooled within-city fixed-effect diagnostics
  with city-clustered standard errors.
- `mechanisms/sequential_adjustment.csv` - attenuation across adjustment tiers.
- `mechanisms/surface_context_path_models.csv` - surface-context associations
  along footprint and height paths.
- `mechanisms/grid_interactions.csv` and `mechanisms/city_moderators.csv` -
  moderator diagnostics used to support the surface-context interpretation.
- `robustness/same_volume_matched_contrast.csv` - same-volume horizontal versus
  vertical morphology contrast.
- `robustness/sensitivity_summary.csv` - footprint threshold, sample-size,
  and minimum-grid sensitivity checks across the full 4 x 3 design.
- `robustness/spatial_block_bootstrap.csv` - within-city spatial block bootstrap
  support for the footprint-height contrast.
- `run_manifest.json` - command, input, output, and run metadata.

The additional modules expose DataFrame-in/DataFrame-out functions so that the
prepared city and sensor tables can be analyzed without local path assumptions:

```python
from climate_typology import classify_risk_first_typology
from nonlinear_analysis import grouped_oof_predictions
from sensor_validation import compare_temperature_products
```

## Method Scope

The primary inferential unit is the city-level ordinary least-squares estimate.
The response is within-city hottest-month LST anomaly. Building volume density
is defined as footprint fraction times mean building height. The pathway model
estimates the footprint path conditional on height and controls (BVR-F) and the
height path conditional on footprint and controls (BVR-H) on the same log
response scale. Reported 1% and 10% differences use the exact multipliers
`log(1.01)` and `log(1.10)`.

All estimates describe static 2020 within-city spatial association. The code
does not estimate temporal construction effects, air-temperature effects,
thermal-comfort outcomes, or causal impacts of new buildings.

## Suggested Open Research Text

```text
The analysis code is available at https://github.com/JoeyHu-coding/3DLST.
The analysis-ready grid matrix, figure source data, and supplementary source
tables are archived separately from the code repository.
```
