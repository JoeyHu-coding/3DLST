"""Tabular sensitivity checks for temperature products, scale, and view angle.

These functions operate only on supplied tables.  They do not download data,
read rasters, or claim synchronous sensor validation or causal effects.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _pearson(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if len(values_a) < 2:
        return math.nan
    if values_a.std(ddof=0) <= 0 or values_b.std(ddof=0) <= 0:
        return math.nan
    return float(np.corrcoef(values_a, values_b)[0, 1])


def compare_temperature_products(
    landsat: pd.DataFrame,
    gshtd: pd.DataFrame,
    uid_col: str,
    metric_mapping: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    """Compare paired city metrics from Landsat and GSHTD tables.

    ``metric_mapping`` maps each output metric label to
    ``(landsat_column, gshtd_column)``.  Correlations use finite paired cities,
    and leave-one-city-out ranges omit undefined constant-subset correlations.
    """

    _require_columns(landsat, (uid_col,), "landsat")
    _require_columns(gshtd, (uid_col,), "gshtd")
    if landsat[uid_col].duplicated().any() or gshtd[uid_col].duplicated().any():
        raise ValueError("temperature-product inputs must have unique UID rows")
    if not metric_mapping:
        raise ValueError("metric_mapping must contain at least one metric")

    rows: list[dict[str, object]] = []
    for metric, columns in metric_mapping.items():
        if len(columns) != 2:
            raise ValueError(
                f"metric_mapping[{metric!r}] must contain Landsat and GSHTD columns"
            )
        landsat_col, gshtd_col = columns
        _require_columns(landsat, (landsat_col,), "landsat")
        _require_columns(gshtd, (gshtd_col,), "gshtd")
        left = landsat[[uid_col, landsat_col]].rename(
            columns={landsat_col: "landsat_value"}
        )
        right = gshtd[[uid_col, gshtd_col]].rename(
            columns={gshtd_col: "gshtd_value"}
        )
        paired = left.merge(right, on=uid_col, how="inner", validate="one_to_one")
        paired[["landsat_value", "gshtd_value"]] = paired[
            ["landsat_value", "gshtd_value"]
        ].apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(
            paired[["landsat_value", "gshtd_value"]].to_numpy(dtype=float)
        ).all(axis=1)
        paired = paired.loc[finite].reset_index(drop=True)
        landsat_values = paired["landsat_value"].to_numpy(dtype=float)
        gshtd_values = paired["gshtd_value"].to_numpy(dtype=float)
        pearson_r = _pearson(landsat_values, gshtd_values)
        if (
            len(paired) >= 2
            and np.unique(landsat_values).size > 1
            and np.unique(gshtd_values).size > 1
        ):
            spearman_rho = float(spearmanr(landsat_values, gshtd_values).statistic)
        else:
            spearman_rho = math.nan

        leave_one_out: list[float] = []
        for omitted in range(len(paired)):
            keep = np.arange(len(paired)) != omitted
            estimate = _pearson(landsat_values[keep], gshtd_values[keep])
            if math.isfinite(estimate):
                leave_one_out.append(estimate)
        if leave_one_out:
            lower = float(min(leave_one_out))
            upper = float(max(leave_one_out))
            display_range = f"{lower:.3f}\N{EN DASH}{upper:.3f}"
        else:
            lower = upper = math.nan
            display_range = ""
        rows.append(
            {
                "metric": str(metric),
                "n_paired_cities": int(len(paired)),
                "pearson_r": pearson_r,
                "spearman_rho": spearman_rho,
                "leave_one_city_out_pearson_r_min": lower,
                "leave_one_city_out_pearson_r_max": upper,
                "leave_one_city_out_pearson_r_range": display_range,
                "interpretation_scope": (
                    "cross_product_correspondence_not_synchronous_validation_or_causality"
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_multiscale_pathways(
    frame: pd.DataFrame,
    *,
    scale_col: str = "scale_m",
    uid_col: str = "UID",
    footprint_col: str = "BVR_F_10pct",
    height_col: str = "BVR_H_10pct",
    delta_col: str = "Delta_F_minus_H_10pct",
) -> pd.DataFrame:
    """Return equal-city medians of BVR-F, BVR-H, and Delta by scale.

    Repeated rows within a city-scale cell are first reduced to a city median;
    the reported scale summary then gives each city one equal-weight value.
    """

    columns = (scale_col, uid_col, footprint_col, height_col, delta_col)
    _require_columns(frame, columns, "frame")
    work = frame.loc[:, columns].copy()
    for column in (footprint_col, height_col, delta_col):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=list(columns))
    output_columns = [
        scale_col,
        "n_city",
        "equal_city_median_BVR_F",
        "equal_city_median_BVR_H",
        "equal_city_median_Delta",
        "summary_scope",
    ]
    if work.empty:
        return pd.DataFrame(columns=output_columns)

    city_scale = (
        work.groupby([scale_col, uid_col], observed=True, sort=True)[
            [footprint_col, height_col, delta_col]
        ]
        .median()
        .reset_index()
    )
    summary = (
        city_scale.groupby(scale_col, observed=True, sort=True)
        .agg(
            n_city=(uid_col, "nunique"),
            equal_city_median_BVR_F=(footprint_col, "median"),
            equal_city_median_BVR_H=(height_col, "median"),
            equal_city_median_Delta=(delta_col, "median"),
        )
        .reset_index()
    )
    summary["n_city"] = summary["n_city"].astype(int)
    summary["summary_scope"] = "equal_city_descriptive_median_by_scale"
    return summary.loc[:, output_columns]


def fit_viewing_angle_interaction(
    frame: pd.DataFrame,
    outcome: str,
    height: str,
    vza: str,
    controls: Sequence[str] = (),
    city: str = "UID",
    acquisition: str = "acquisition",
) -> pd.DataFrame:
    """Fit a centered height-by-VZA association with fixed effects.

    City indicators and acquisition indicators nested within each city are
    included explicitly.  The returned HC3 interval is an associational
    sensitivity diagnostic, not a causal viewing-angle effect.
    """

    controls = tuple(controls)
    if len(controls) != len(set(controls)):
        raise ValueError("controls must be unique")
    numeric_columns = (outcome, height, vza, *controls)
    required = (*numeric_columns, city, acquisition)
    _require_columns(frame, required, "frame")
    work = frame.loc[:, required].copy()
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    finite = np.isfinite(work.loc[:, numeric_columns].to_numpy(dtype=float)).all(axis=1)
    complete_groups = work[[city, acquisition]].notna().all(axis=1)
    work = work.loc[finite & complete_groups].reset_index(drop=True)
    if work.empty:
        raise ValueError("no complete finite rows are available for the interaction model")

    height_center = float(work[height].mean())
    vza_center = float(work[vza].mean())
    height_values = work[height].to_numpy(dtype=float) - height_center
    vza_values = work[vza].to_numpy(dtype=float) - vza_center
    if height_values.std(ddof=0) <= 0 or vza_values.std(ddof=0) <= 0:
        raise ValueError("height and vza must both vary")
    interaction_term = f"{height}_centered_x_{vza}_centered"
    design: dict[str, np.ndarray] = {
        f"{height}_centered": height_values,
        f"{vza}_centered": vza_values,
        interaction_term: height_values * vza_values,
    }
    for control in controls:
        design[control] = work[control].to_numpy(dtype=float)

    city_values = list(pd.unique(work[city]))
    for city_value in city_values[1:]:
        design[f"FE_city[{city_value}]"] = work[city].eq(city_value).to_numpy(dtype=float)

    acquisition_fe_count = 0
    for city_value in city_values:
        in_city = work[city].eq(city_value)
        acquisition_values = list(pd.unique(work.loc[in_city, acquisition]))
        for acquisition_value in acquisition_values[1:]:
            term = f"FE_acquisition[{city_value}|{acquisition_value}]"
            design[term] = (
                in_city & work[acquisition].eq(acquisition_value)
            ).to_numpy(dtype=float)
            acquisition_fe_count += 1

    design_frame = sm.add_constant(pd.DataFrame(design), has_constant="add")
    design_matrix = design_frame.to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(design_matrix))
    n_rows, n_parameters = design_matrix.shape
    if n_rows <= n_parameters or rank != n_parameters:
        raise ValueError(
            "viewing-angle fixed-effect design is not estimable: "
            f"n={n_rows}, rank={rank}, parameters={n_parameters}"
        )
    fit = sm.OLS(
        work[outcome].to_numpy(dtype=float), design_frame.to_numpy(dtype=float)
    ).fit(cov_type="HC3")
    interaction_index = design_frame.columns.get_loc(interaction_term)
    estimate = float(fit.params[interaction_index])
    standard_error = float(fit.bse[interaction_index])
    ci_low = estimate - 1.96 * standard_error
    ci_high = estimate + 1.96 * standard_error
    return pd.DataFrame(
        [
            {
                "term": interaction_term,
                "estimate": estimate,
                "standard_error": standard_error,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_observations": int(n_rows),
                "n_city": int(work[city].nunique()),
                "n_city_acquisition": int(work[[city, acquisition]].drop_duplicates().shape[0]),
                "height_center": height_center,
                "vza_center": vza_center,
                "design_rank": rank,
                "design_parameter_count": int(n_parameters),
                "city_fixed_effect_count": max(len(city_values) - 1, 0),
                "acquisition_fixed_effect_count": acquisition_fe_count,
                "fixed_effects": "city_and_within_city_acquisition",
                "covariance_estimator": "HC3",
                "interpretation_scope": "association_not_causal_effect",
            }
        ]
    )


__all__ = [
    "compare_temperature_products",
    "fit_viewing_angle_interaction",
    "summarize_multiscale_pathways",
]
