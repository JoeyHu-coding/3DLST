"""Shared configuration, input handling, and statistical helpers for 3DLST.

This module contains reusable definitions used by the public analysis scripts.
It does not run any analysis by itself.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised only when dependencies are missing.
    pq = None


UID = "UID"
RESPONSE = "Ts_anomaly"

# The internal result scale follows the archived analysis: response per 10%
# multiplicative increment. Per-1% display values can be obtained by dividing
# these fields by 10 as a close small-increment display conversion.
LOG_10PCT = float(np.log(1.10))

PRIMARY_TIER_ID = "M1_terrain_water"
MAIN_BF_THRESHOLD = 0.01
MIN_ESTIMABLE_ROWS = 30
MIN_STABLE_ROWS = 100
MIN_LOG_RANGE = 0.2
NEAR_ZERO_C_10PCT = 0.01
DOMINANCE_SHARE = 0.65

DEFAULT_BOOTSTRAP_REPLICATES = 200
DEFAULT_BOOT_BLOCK_SIZE = 5
RANDOM_STATE = 20260518


@dataclass(frozen=True)
class ModelTier:
    """Adjustment tier used for city-level models."""

    tier_id: str
    label: str
    controls: tuple[str, ...]
    spatial_terms: bool = False


MODEL_TIERS = [
    ModelTier("M0_morphology", "Morphology only", ()),
    ModelTier(
        "M1_terrain_water",
        "Terrain and water",
        ("terrain_mean_m", "slope_mean_deg", "relief_p90_p10_m", "p_water_1km"),
    ),
    ModelTier(
        "M2_vegetation",
        "Terrain, water, and vegetation",
        (
            "terrain_mean_m",
            "slope_mean_deg",
            "relief_p90_p10_m",
            "p_water_1km",
            "p_veg_1km",
            "VF",
            "VVD_m",
            "MVH_m",
        ),
    ),
    ModelTier(
        "M3_landcover",
        "Terrain, water, vegetation, and land cover",
        (
            "terrain_mean_m",
            "slope_mean_deg",
            "relief_p90_p10_m",
            "p_water_1km",
            "p_veg_1km",
            "VF",
            "VVD_m",
            "MVH_m",
            "p_bare_1km",
            "p_crop_1km",
            "p_built_1km",
        ),
    ),
    ModelTier(
        "M4_spatial",
        "Terrain, water, and spatial trend",
        ("terrain_mean_m", "slope_mean_deg", "relief_p90_p10_m", "p_water_1km"),
        True,
    ),
    ModelTier(
        "M5_full",
        "Full adjustment",
        (
            "terrain_mean_m",
            "slope_mean_deg",
            "relief_p90_p10_m",
            "p_water_1km",
            "p_veg_1km",
            "VF",
            "VVD_m",
            "MVH_m",
            "p_bare_1km",
            "p_crop_1km",
            "p_built_1km",
        ),
        True,
    ),
]


CONTEXT_FIRST = [
    "Country",
    "Continent",
    "KG",
    "climate_macro",
    "income_group",
    "climate_income_regime",
    "population",
    "GDP",
    "GDPpc",
    "city_size_class",
]

MEAN_CONTEXT = [
    "BF",
    "MBH_m",
    "lnF",
    "lnH",
    "lnV",
    "p_veg_1km",
    "p_water_1km",
    "p_built_1km",
    "p_bare_1km",
    "p_crop_1km",
    "ai_built_1km",
    "VF",
    "VVD_m",
    "MVH_m",
    "terrain_mean_m",
    "slope_mean_deg",
    "relief_p90_p10_m",
]

SURFACE_CONTEXT_OUTCOMES = [
    ("p_veg_1km", "ESA vegetation cover fraction"),
    ("VF", "3D vegetation footprint fraction"),
    ("VVD_m", "3D vegetation volume density"),
    ("MVH_m", "Mean vegetation height"),
    ("p_water_1km", "ESA water fraction"),
    ("p_bare_1km", "ESA bare fraction"),
    ("p_crop_1km", "ESA crop fraction"),
    ("p_built_1km", "ESA built fraction"),
    ("core_veg_30m_1km", "Core vegetation area"),
    ("ai_built_1km", "Built aggregation index"),
]

GRID_MODERATORS = [
    ("p_veg_1km", "vegetation cover"),
    ("VF", "3D vegetation cover"),
    ("VVD_m", "vegetation volume density"),
    ("MVH_m", "vegetation height"),
    ("p_water_1km", "water cover"),
    ("p_bare_1km", "bare land cover"),
    ("p_crop_1km", "cropland cover"),
    ("p_built_1km", "built land-cover fraction"),
    ("ai_built_1km", "built aggregation"),
    ("terrain_mean_m", "terrain elevation"),
    ("slope_mean_deg", "slope"),
    ("relief_p90_p10_m", "local relief"),
    ("grid_distance_to_city_center_z", "grid distance to city center"),
]

FINAL_CITY_DRIVER_FEATURES = (
    "mean_BF",
    "mean_MBH_m",
    "mean_p_water_1km",
    "mean_p_bare_1km",
    "mean_p_crop_1km",
    "log_GDPpc",
    "log_population",
    "mean_VF",
    "mean_MVH_m",
    "mean_p_built_1km",
    "mean_ai_built_1km",
    "mean_terrain_mean_m",
    "mean_slope_mean_deg",
)

BOOTSTRAP_METRICS = [
    "BVR_10pct",
    "BVR_F_10pct",
    "BVR_H_10pct",
    "Delta_F_minus_H_10pct",
    "C_F_10pct",
    "C_H_10pct",
]


def now_utc() -> str:
    """Return a compact UTC timestamp for provenance metadata."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    """Write JSON with stable indentation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV table and create the parent directory if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def infer_project_root() -> Path:
    """Infer a useful default root for both local and public-repo layouts."""

    here = Path(__file__).resolve()
    candidates = [Path.cwd()]
    candidates.extend(parent for parent in here.parents[:4])
    for candidate in candidates:
        if (candidate / "output").exists() or (candidate / "data").exists():
            return candidate
    return Path.cwd()


def default_matrix_path(project_root: Path) -> Path:
    """Return the local project default analysis matrix path."""

    return project_root / "output" / "intermediate" / "HVCA" / "hierarchical_vca_analysis_matrix.parquet"


def normal_pvalue(t_value: float) -> float:
    """Two-sided normal-reference p-value used for compact coefficient tables."""

    if not math.isfinite(t_value):
        return math.nan
    return float(math.erfc(abs(t_value) / math.sqrt(2.0)))


def ci95(est: float, se: float) -> tuple[float, float]:
    """Return a normal-reference 95% confidence interval."""

    if not (math.isfinite(est) and math.isfinite(se)):
        return math.nan, math.nan
    return float(est - 1.96 * se), float(est + 1.96 * se)


def available_columns(path: Path) -> list[str]:
    """Read Parquet schema columns without loading the full table."""

    if pq is None:
        raise SystemExit(
            "Missing dependency: pyarrow is required to read the Parquet analysis matrix. "
            "Install the public-code environment with `python -m pip install -r requirements.txt`."
        )
    return list(pq.read_schema(path).names)


def compute_missing_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create eligibility and log morphology fields when absent."""

    out = df.copy()
    for col in ["BF", "MBH_m", RESPONSE]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "eligible_hvca_main" not in out.columns:
        out["eligible_hvca_main"] = (
            (out["BF"] >= MAIN_BF_THRESHOLD)
            & (out["MBH_m"] > 0)
            & np.isfinite(pd.to_numeric(out[RESPONSE], errors="coerce"))
        )
    out["eligible_hvca_main"] = out["eligible_hvca_main"].astype(bool)
    if "lnF" not in out.columns:
        out["lnF"] = np.nan
        mask = out["eligible_hvca_main"] & (out["BF"] > 0)
        out.loc[mask, "lnF"] = np.log(out.loc[mask, "BF"].astype(float))
    if "lnH" not in out.columns:
        out["lnH"] = np.nan
        mask = out["eligible_hvca_main"] & (out["MBH_m"] > 0)
        out.loc[mask, "lnH"] = np.log(out.loc[mask, "MBH_m"].astype(float))
    if "lnV" not in out.columns:
        out["lnV"] = np.nan
        mask = out["eligible_hvca_main"] & np.isfinite(out["lnF"]) & np.isfinite(out["lnH"])
        out.loc[mask, "lnV"] = out.loc[mask, "lnF"] + out.loc[mask, "lnH"]
    if "grid_distance_to_city_center_z" not in out.columns and {"grid_col", "grid_row"}.issubset(out.columns):
        for coord in ["grid_col", "grid_row"]:
            vals = pd.to_numeric(out[coord], errors="coerce")
            mean = vals.groupby(out[UID], observed=True).transform("mean")
            sd = vals.groupby(out[UID], observed=True).transform(lambda s: s.std(ddof=0)).replace(0, np.nan)
            out[f"z_{coord}"] = ((vals - mean) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["grid_distance_to_city_center_z"] = np.sqrt(out["z_grid_col"] ** 2 + out["z_grid_row"] ** 2)
    return out.replace([np.inf, -np.inf], np.nan)


def load_analysis_matrix(matrix_path: Path, max_cities: int | None = None) -> pd.DataFrame:
    """Load the analysis-ready matrix with only columns used by this package."""

    if not matrix_path.exists():
        raise FileNotFoundError(f"Analysis matrix not found: {matrix_path}")

    needed = {
        UID,
        "FID",
        "FID_NUM",
        "eligible_hvca_main",
        "fid_shared_by_multiple_uid",
        RESPONSE,
        "BF",
        "MBH_m",
        "lnF",
        "lnH",
        "lnV",
        "terrain_mean_m",
        "slope_mean_deg",
        "relief_p90_p10_m",
        "p_water_1km",
        "p_veg_1km",
        "p_bare_1km",
        "p_crop_1km",
        "p_built_1km",
        "core_veg_30m_1km",
        "ai_built_1km",
        "VF",
        "VVD_m",
        "MVH_m",
        "grid_col",
        "grid_row",
        "grid_distance_to_city_center_z",
        *CONTEXT_FIRST,
    }
    columns = [col for col in available_columns(matrix_path) if col in needed]
    df = pd.read_parquet(matrix_path, columns=columns)
    df = compute_missing_log_features(df)

    if max_cities is not None:
        unique_uids = pd.Series(df[UID].dropna().unique()).sort_values().head(max_cities).tolist()
        df = df.loc[df[UID].isin(unique_uids)].copy()

    string_cols = {
        "FID",
        "Country",
        "Continent",
        "climate_macro",
        "income_group",
        "climate_income_regime",
        "city_size_class",
    }
    for col in df.columns:
        if col in string_cols or col == "eligible_hvca_main":
            continue
        if col == "fid_shared_by_multiple_uid":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required = [UID, RESPONSE, "BF", "MBH_m", "lnF", "lnH", "lnV", "eligible_hvca_main"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Analysis matrix is missing required columns: {missing}")
    return df


def eligible_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows eligible for the primary study models."""

    return df.loc[df["eligible_hvca_main"].astype(bool)].copy()


def city_context(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate city-level context used in summary and moderator models."""

    ctx = df[[UID]].drop_duplicates().copy()
    first_cols = [col for col in CONTEXT_FIRST if col in df.columns]
    if first_cols:
        first = df.groupby(UID, observed=True)[first_cols].first().reset_index()
        ctx = ctx.merge(first, on=UID, how="left")
    mean_cols = [col for col in MEAN_CONTEXT if col in df.columns]
    if mean_cols:
        means = df.groupby(UID, observed=True)[mean_cols].mean().add_prefix("mean_").reset_index()
        ctx = ctx.merge(means, on=UID, how="left")
    counts = df.groupby(UID, observed=True).size().rename("n_eligible").reset_index()
    return ctx.merge(counts, on=UID, how="left")


def zscore_array(values: pd.Series | np.ndarray) -> np.ndarray | None:
    """Return a finite z-score vector or None if the variable is unusable."""

    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return None
    sd = float(arr.std(ddof=0))
    if sd <= 1e-12:
        return None
    return (arr - float(arr.mean())) / sd


def finite_mask(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mask rows with finite response and predictors."""

    return np.isfinite(y) & np.all(np.isfinite(x), axis=1)


def fit_ols_hc3(y: np.ndarray, x_raw: np.ndarray) -> dict[str, Any]:
    """Fit OLS with HC3 robust covariance for city-level models."""

    y = np.asarray(y, dtype=np.float64)
    x_raw = np.asarray(x_raw, dtype=np.float64)
    if x_raw.ndim == 1:
        x_raw = x_raw[:, None]
    mask = finite_mask(y, x_raw)
    y = y[mask]
    x_raw = x_raw[mask]
    n = int(len(y))
    p = int(x_raw.shape[1])
    empty = {
        "n": n,
        "rank": 0,
        "estimable": False,
        "intercept": math.nan,
        "beta": np.full(p, np.nan),
        "se_hc3": np.full(p, np.nan),
        "cov_hc3": np.full((p, p), np.nan),
        "r2": math.nan,
        "rmse": math.nan,
        "df_resid": n - p - 1,
    }
    if n <= p + 1:
        return empty
    x = np.column_stack([np.ones(n), x_raw])
    rank = int(np.linalg.matrix_rank(x))
    if rank < p + 1:
        empty["rank"] = rank
        return empty
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    fitted = x @ beta
    resid = y - fitted
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    leverage = np.clip(leverage, 0.0, 0.999999)
    scaled = (resid / (1.0 - leverage)) ** 2
    meat = x.T @ (x * scaled[:, None])
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": n,
        "rank": rank,
        "estimable": True,
        "intercept": float(beta[0]),
        "beta": beta[1:],
        "se_hc3": se[1:],
        "cov_hc3": cov[1:, 1:],
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan,
        "rmse": float(math.sqrt(ss_res / n)),
        "df_resid": n - p - 1,
    }


def fit_ols_cluster(y: np.ndarray, x_raw: np.ndarray, clusters: np.ndarray) -> dict[str, Any]:
    """Fit OLS with city-clustered covariance for pooled diagnostics."""

    y = np.asarray(y, dtype=np.float64)
    x_raw = np.asarray(x_raw, dtype=np.float64)
    if x_raw.ndim == 1:
        x_raw = x_raw[:, None]
    clusters = np.asarray(clusters)
    mask = finite_mask(y, x_raw) & pd.notna(clusters)
    y = y[mask]
    x_raw = x_raw[mask]
    clusters = clusters[mask]
    n = int(len(y))
    p = int(x_raw.shape[1])
    empty = {
        "n": n,
        "n_cluster": int(pd.Series(clusters).nunique()) if len(clusters) else 0,
        "beta": np.full(p, np.nan),
        "se_cluster": np.full(p, np.nan),
        "cov_cluster": np.full((p, p), np.nan),
        "r2": math.nan,
    }
    if n <= p + 1:
        return empty
    x = np.column_stack([np.ones(n), x_raw])
    if np.linalg.matrix_rank(x) < p + 1:
        return empty
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    meat = np.zeros((p + 1, p + 1), dtype=np.float64)
    unique_clusters = pd.unique(clusters)
    for cluster in unique_clusters:
        idx = clusters == cluster
        score = x[idx].T @ resid[idx]
        meat += np.outer(score, score)
    g = len(unique_clusters)
    k = p + 1
    scale = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else 1.0
    cov = scale * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": n,
        "n_cluster": int(g),
        "beta": beta[1:],
        "se_cluster": se[1:],
        "cov_cluster": cov[1:, 1:],
        "intercept": float(beta[0]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan,
    }


def add_independent_column(
    arrays: list[np.ndarray],
    names: list[str],
    candidate: np.ndarray | None,
    name: str,
    dropped: list[str],
) -> None:
    """Append a predictor only if it is finite and increases matrix rank."""

    if candidate is None or not np.all(np.isfinite(candidate)):
        dropped.append(f"{name}:missing_or_zero_variance")
        return
    n = len(candidate)
    current = np.column_stack([np.ones(n), *arrays]) if arrays else np.ones((n, 1))
    new = np.column_stack([current, candidate])
    if np.linalg.matrix_rank(new) > np.linalg.matrix_rank(current):
        arrays.append(candidate)
        names.append(name)
    else:
        dropped.append(f"{name}:rank_collinear")


def build_control_design(part: pd.DataFrame, tier: ModelTier) -> tuple[np.ndarray, list[str], list[str]]:
    """Create a city-specific z-scored control matrix for a model tier."""

    arrays: list[np.ndarray] = []
    names: list[str] = []
    dropped: list[str] = []
    for control in tier.controls:
        if control in part.columns:
            add_independent_column(arrays, names, zscore_array(part[control]), f"z_{control}", dropped)
        else:
            dropped.append(f"z_{control}:missing_column")
    if tier.spatial_terms:
        zx = zscore_array(part["grid_col"]) if "grid_col" in part.columns else None
        zy = zscore_array(part["grid_row"]) if "grid_row" in part.columns else None
        add_independent_column(arrays, names, zx, "z_grid_col", dropped)
        add_independent_column(arrays, names, zy, "z_grid_row", dropped)
        if zx is not None:
            add_independent_column(arrays, names, zx**2, "z_grid_col_sq", dropped)
        if zy is not None:
            add_independent_column(arrays, names, zy**2, "z_grid_row_sq", dropped)
        if zx is not None and zy is not None:
            add_independent_column(arrays, names, zx * zy, "z_grid_col_x_z_grid_row", dropped)
    if arrays:
        return np.column_stack(arrays), names, dropped
    return np.empty((len(part), 0)), names, dropped


def reduced_rank_design(arrays: list[np.ndarray], names: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a global design matrix while dropping invalid or collinear terms."""

    kept_arrays: list[np.ndarray] = []
    kept_names: list[str] = []
    dropped: list[str] = []
    for arr, name in zip(arrays, names):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 1 or not np.all(np.isfinite(arr)):
            dropped.append(f"{name}:missing_or_nonfinite")
            continue
        current = np.column_stack([np.ones(len(arr)), *kept_arrays]) if kept_arrays else np.ones((len(arr), 1))
        candidate = np.column_stack([current, arr])
        if np.linalg.matrix_rank(candidate) > np.linalg.matrix_rank(current):
            kept_arrays.append(arr)
            kept_names.append(name)
        else:
            dropped.append(f"{name}:rank_collinear")
    if kept_arrays:
        return np.column_stack(kept_arrays), kept_names, dropped
    return np.empty((len(arrays[0]) if arrays else 0, 0)), kept_names, dropped


def within_city_z(df: pd.DataFrame, col: str) -> pd.Series:
    """Return within-city standardized values for pooled models."""

    vals = pd.to_numeric(df[col], errors="coerce")
    means = vals.groupby(df[UID], observed=True).transform("mean")
    stds = vals.groupby(df[UID], observed=True).transform(lambda s: s.std(ddof=0)).replace(0, np.nan)
    z = (vals - means) / stds
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def demean_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Remove city fixed effects from a variable."""

    vals = pd.to_numeric(df[col], errors="coerce")
    return vals - vals.groupby(df[UID], observed=True).transform("mean")


def pooled_design(df: pd.DataFrame, predictors: list[str], tier: ModelTier) -> tuple[np.ndarray, list[str], list[str], pd.Series]:
    """Construct a within-city design for pooled fixed-effect diagnostics."""

    raw_arrays: list[np.ndarray] = []
    raw_names: list[str] = []
    for predictor in predictors:
        raw_arrays.append(demean_series(df, predictor).to_numpy(dtype=np.float64))
        raw_names.append(predictor)
    for control in tier.controls:
        if control in df.columns:
            raw_arrays.append(within_city_z(df, control).to_numpy(dtype=np.float64))
            raw_names.append(f"z_{control}")
    if tier.spatial_terms:
        zx = within_city_z(df, "grid_col")
        zy = within_city_z(df, "grid_row")
        for name, arr in [
            ("z_grid_col", zx),
            ("z_grid_row", zy),
            ("z_grid_col_sq", zx**2),
            ("z_grid_row_sq", zy**2),
            ("z_grid_col_x_z_grid_row", zx * zy),
        ]:
            raw_arrays.append(arr.to_numpy(dtype=np.float64))
            raw_names.append(name)
    x, names, dropped = reduced_rank_design(raw_arrays, raw_names)
    return x, names, dropped, demean_series(df, RESPONSE)


def residualize_vector(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Residualize a vector against controls for contribution accounting."""

    y = np.asarray(values, dtype=np.float64)
    if controls.size == 0 or controls.shape[1] == 0:
        return y - np.nanmean(y)
    fit = fit_ols_hc3(y, controls)
    if not fit["estimable"]:
        return np.full(len(y), np.nan)
    fitted = fit["intercept"] + controls @ fit["beta"]
    return y - fitted


def contribution_decomposition(
    ln_f: np.ndarray,
    ln_h: np.ndarray,
    ln_v: np.ndarray,
    beta_f: float,
    beta_h: float,
    controls: np.ndarray | None = None,
) -> dict[str, float]:
    """Reconstruct the total volume response from footprint and height components."""

    controls_arr = np.empty((len(ln_v), 0)) if controls is None else np.asarray(controls, dtype=np.float64)
    f_resid = residualize_vector(np.asarray(ln_f, dtype=np.float64), controls_arr)
    h_resid = residualize_vector(np.asarray(ln_h, dtype=np.float64), controls_arr)
    v_resid = residualize_vector(np.asarray(ln_v, dtype=np.float64), controls_arr)
    mask = np.isfinite(f_resid) & np.isfinite(h_resid) & np.isfinite(v_resid)
    if int(mask.sum()) < 3:
        return {
            "C_F": math.nan,
            "C_H": math.nan,
            "C_F_10pct": math.nan,
            "C_H_10pct": math.nan,
            "var_lnV_resid": math.nan,
        }
    f = f_resid[mask]
    h = h_resid[mask]
    v = v_resid[mask]
    var_v = float(np.var(v, ddof=0))
    if var_v <= 1e-12:
        return {
            "C_F": math.nan,
            "C_H": math.nan,
            "C_F_10pct": math.nan,
            "C_H_10pct": math.nan,
            "var_lnV_resid": var_v,
        }
    cov_fv = float(np.mean((f - f.mean()) * (v - v.mean())))
    cov_hv = float(np.mean((h - h.mean()) * (v - v.mean())))
    c_f = float(beta_f * cov_fv / var_v)
    c_h = float(beta_h * cov_hv / var_v)
    return {
        "C_F": c_f,
        "C_H": c_h,
        "C_F_10pct": c_f * LOG_10PCT,
        "C_H_10pct": c_h * LOG_10PCT,
        "var_lnV_resid": var_v,
    }


def classify_source_contribution(
    bvr_10pct: float,
    c_f_10pct: float,
    c_h_10pct: float,
    near_zero: float = NEAR_ZERO_C_10PCT,
    dominance_share: float = DOMINANCE_SHARE,
) -> dict[str, Any]:
    """Classify whether total BVR is mainly footprint-, height-, or mixed-driven."""

    if not all(math.isfinite(v) for v in [bvr_10pct, c_f_10pct, c_h_10pct]):
        return {"source_class": "not_estimable", "dominant_share": math.nan, "offset_flag": False}
    abs_sum = abs(c_f_10pct) + abs(c_h_10pct)
    if abs(bvr_10pct) <= near_zero and abs_sum <= 2 * near_zero:
        share = 0.0 if abs_sum == 0 else max(abs(c_f_10pct), abs(c_h_10pct)) / abs_sum
        return {"source_class": "weak_response", "dominant_share": share, "offset_flag": False}
    offset = (c_f_10pct > near_zero and c_h_10pct < -near_zero) or (
        c_h_10pct > near_zero and c_f_10pct < -near_zero
    )
    if offset:
        return {
            "source_class": "offset_footprint_height_signals",
            "dominant_share": max(abs(c_f_10pct), abs(c_h_10pct)) / abs_sum if abs_sum else math.nan,
            "offset_flag": True,
        }
    share_f = abs(c_f_10pct) / abs_sum if abs_sum else math.nan
    share_h = abs(c_h_10pct) / abs_sum if abs_sum else math.nan
    if c_f_10pct > near_zero and share_f >= dominance_share:
        return {"source_class": "footprint_driven_volume_warming", "dominant_share": share_f, "offset_flag": False}
    if c_h_10pct > near_zero and share_h >= dominance_share:
        return {"source_class": "height_driven_volume_warming", "dominant_share": share_h, "offset_flag": False}
    if c_f_10pct > near_zero and c_h_10pct > near_zero:
        return {"source_class": "mixed_source_volume_warming", "dominant_share": max(share_f, share_h), "offset_flag": False}
    return {
        "source_class": "other_or_cooling_response",
        "dominant_share": max(share_f, share_h) if abs_sum else math.nan,
        "offset_flag": False,
    }


def attenuation_diagnostics(base_value: float, target_value: float, near_zero: float = NEAR_ZERO_C_10PCT) -> dict[str, Any]:
    """Quantify attenuation between a baseline tier and an adjusted tier."""

    if not (math.isfinite(base_value) and math.isfinite(target_value)):
        return {"attenuation_ratio": math.nan, "denominator_near_zero": True, "sign_change": False}
    denominator_near_zero = abs(base_value) <= near_zero
    attenuation = math.nan if denominator_near_zero else 1.0 - (target_value / base_value)
    sign_change = (base_value > near_zero and target_value < -near_zero) or (
        base_value < -near_zero and target_value > near_zero
    )
    return {
        "attenuation_ratio": float(attenuation) if math.isfinite(attenuation) else math.nan,
        "denominator_near_zero": bool(denominator_near_zero),
        "sign_change": bool(sign_change),
    }


def summarize_distribution(frame: pd.DataFrame, value_col: str, group_cols: Iterable[str]) -> pd.DataFrame:
    """Summarize a response metric by one or more grouping columns."""

    rows: list[dict[str, Any]] = []
    group_cols = list(group_cols)
    for keys, part in frame.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = pd.to_numeric(part[value_col], errors="coerce").dropna()
        rec = {col: key for col, key in zip(group_cols, keys)}
        rec.update(
            {
                "metric": value_col,
                "n": int(vals.size),
                "mean": float(vals.mean()) if vals.size else math.nan,
                "median": float(vals.median()) if vals.size else math.nan,
                "q25": float(vals.quantile(0.25)) if vals.size else math.nan,
                "q75": float(vals.quantile(0.75)) if vals.size else math.nan,
                "share_positive": float((vals > NEAR_ZERO_C_10PCT).mean()) if vals.size else math.nan,
                "share_negative": float((vals < -NEAR_ZERO_C_10PCT).mean()) if vals.size else math.nan,
                "share_near_zero": float((vals.abs() <= NEAR_ZERO_C_10PCT).mean()) if vals.size else math.nan,
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def add_per1_columns(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add approximate per-1% display columns from per-10% estimates."""

    out = frame.copy()
    for col in cols:
        if col in out.columns:
            out[col.replace("_10pct", "_per1pct_approx")] = pd.to_numeric(out[col], errors="coerce") / 10.0
    return out
