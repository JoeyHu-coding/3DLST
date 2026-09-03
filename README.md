# 3DLST Reproducible Analysis Code

This repository contains the compact public analysis code for the 3DLST study
of building footprint, building height, and hottest-month land surface
temperature (LST) anomalies across global cities.

The matching data release is archived in the existing Zenodo record series:
[10.5281/zenodo.20755781](https://doi.org/10.5281/zenodo.20755781). This concept
DOI resolves to the latest version. A manuscript should cite the
version-specific DOI displayed on the release page.

## Scope

The package starts from two analysis-ready research tables:

- `analysis_grid_core.parquet` supports the city-level linear, mechanism, and
  robustness analyses.
- `city_analysis_core.csv` supports the fixed nonlinear prediction and
  risk-first typology analyses.

The repository intentionally excludes raw satellite and third-party rasters,
geospatial preprocessing, manuscript assembly, reviewer-response utilities,
exploratory notebooks, and local QA scripts. The Zenodo release also includes
compact main-figure source tables and validation summaries for checking the
reported results. It does not claim to recreate third-party source products.

## Tested environment

The manuscript-facing run used Python 3.13.9 with:

```text
numpy 2.3.2
pandas 2.3.2
pyarrow 21.0.0
scipy 1.16.2
statsmodels 0.14.5
scikit-learn 1.7.2
```

Create an environment and install the runtime dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, replace `.venv/bin/python` with `.venv/Scripts/python.exe`.

For tests:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

## Grid-matrix workflow

Run all grid-matrix stages:

```bash
python run_analysis.py \
  --matrix data/analysis_grid_core.parquet \
  --output-dir results/grid_analysis \
  --steps all
```

`--steps all` means all grid-matrix stages: core city estimates, surface-context
diagnostics, and robustness checks. Use `core`, `mechanisms`, or `robustness`
to run a subset. A short smoke test can use `--max-cities 10` and
`--bootstrap-replicates 5`.

Principal outputs include:

- `core/city_total_bvr.csv`
- `core/city_pathway_contrast.csv`
- `core/city_contribution_decomposition.csv`
- `mechanisms/sequential_adjustment.csv`
- `mechanisms/surface_context_path_models.csv`
- `mechanisms/grid_interactions.csv`
- `mechanisms/city_moderators.csv`
- `robustness/same_volume_matched_contrast.csv`
- `robustness/sensitivity_summary.csv`
- `robustness/spatial_block_bootstrap.csv`
- `run_manifest.json`

## City-level workflow

Run the retained fixed nonlinear models and the final risk-first typology:

```bash
python run_city_analysis.py \
  --city-table data/city_analysis_core.csv \
  --output-dir results/city_analysis
```

This writes held-out predictions and performance from the released frozen
folds, full-sample Monte Carlo Shapley values and summaries, accumulated local
effects (ALE), and Global South/Global North typology summaries. These are
model-dependent predictive or descriptive associations, not causal effects.

The expensive nested tuning comparison is not repeated by this compact entry
point. Its final performance table is preserved in the Zenodo figure-source
package. The retained fixed model is the configuration used for the displayed
nonlinear diagnostics.

## Analysis definitions

Building volume density is footprint fraction multiplied by mean building
height. The primary inferential unit is the city-level ordinary least-squares
estimate. The response is within-city hottest-month LST anomaly. The pathway
model estimates the footprint response conditional on height and controls
(BVR-F) and the height response conditional on footprint and controls (BVR-H)
on the same log-response scale.

The linear workflow reports 1% morphology differences with the exact
`log(1.01)` multiplier. The typology uses the common 10% volume scale. All
estimates describe static 2020 within-city spatial associations; they do not
estimate temporal construction effects, air-temperature effects,
thermal-comfort outcomes, or causal impacts of new buildings.

## Files

- `run_analysis.py`: grid-matrix command-line workflow.
- `run_city_analysis.py`: fixed nonlinear and risk-first typology workflow.
- `lst_common.py`: shared configuration, validation, and regression helpers.
- `core_analysis.py`: total response and footprint-height decomposition.
- `mechanism_analysis.py`: surface-context diagnostics and moderators.
- `robustness_analysis.py`: threshold, same-volume, and spatial-bootstrap checks.
- `nonlinear_analysis.py`: grouped prediction, Shapley attribution, and ALE.
- `climate_typology.py`: risk-first classification and region summaries.
- `sensor_validation.py`: reusable tabular sensor and scale checks.
- `tests/`: synthetic behavioral and repository-contract tests.

## Suggested Open Research text

```text
The analysis code is available at https://github.com/JoeyHu-coding/3DLST.
The analysis-ready data and figure source tables are archived at the
version-specific DOI reported for the associated Zenodo release.
```
