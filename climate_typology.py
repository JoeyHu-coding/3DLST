"""Risk-first pathway typology and North--South summary statistics.

The four types are descriptive screening categories.  They summarize observed
pathway estimates and uncertainty intervals; they are not causal effects or
prescriptive planning classes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


TYPE_ORDER = ("Type I", "Type II", "Type III", "Type IV")
DEFAULT_REGION_ORDER = ("global_south", "global_north")


@dataclass(frozen=True)
class NorthSouthTypologySummary:
    """Tidy tables produced by :func:`summarize_north_south_typology`."""

    composition: pd.DataFrame
    association: pd.DataFrame
    evenness: pd.DataFrame
    bootstrap: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {', '.join(missing)}")


def classify_risk_first_typology(
    frame: pd.DataFrame,
    footprint_col: str = "BVR_F_10pct",
    height_col: str = "BVR_H_10pct",
    ci_low_col: str = "Delta_ci_low",
    ci_high_col: str = "Delta_ci_high",
) -> pd.DataFrame:
    """Assign mutually exclusive risk-first pathway types.

    Type III is evaluated first using ``min(footprint, height) > 0.02``.
    Among remaining finite rows, Type I requires a contrast above 0.01 and a
    lower confidence bound above zero, while Type II requires a contrast below
    -0.01 and an upper confidence bound below zero.  Strict inequalities make
    all threshold-boundary cases Type IV.  Non-finite rows remain unclassified.
    """

    required = (footprint_col, height_col, ci_low_col, ci_high_col)
    _require_columns(frame, required)
    numeric = frame.loc[:, required].apply(pd.to_numeric, errors="coerce")
    footprint = numeric[footprint_col]
    height = numeric[height_col]
    ci_low = numeric[ci_low_col]
    ci_high = numeric[ci_high_col]
    contrast = footprint - height
    lower = pd.concat([footprint, height], axis=1).min(axis=1)
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)

    assigned = pd.Series(pd.NA, index=frame.index, dtype="string")
    type_iii = finite & lower.gt(0.02)
    assigned.loc[type_iii] = "Type III"

    remaining = finite & ~type_iii
    type_i = remaining & contrast.gt(0.01) & ci_low.gt(0.0)
    assigned.loc[type_i] = "Type I"
    type_ii = remaining & contrast.lt(-0.01) & ci_high.lt(0.0)
    assigned.loc[type_ii] = "Type II"
    assigned.loc[remaining & ~(type_i | type_ii)] = "Type IV"

    result = frame.copy()
    result["pathway_contrast"] = contrast
    result["lower_pathway_response"] = lower
    result["risk_first_type"] = pd.Categorical(
        assigned, categories=TYPE_ORDER, ordered=True
    )
    return result


def _normalized_shannon_evenness(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return math.nan
    probabilities = np.asarray(counts, dtype=float) / total
    positive = probabilities[probabilities > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return entropy / math.log(len(TYPE_ORDER))


def summarize_north_south_typology(
    frame: pd.DataFrame,
    region_col: str = "global_region_group",
    type_col: str = "risk_first_type",
    *,
    region_order: Sequence[str] = DEFAULT_REGION_ORDER,
    n_bootstrap: int = 0,
    random_state: int = 20260518,
) -> NorthSouthTypologySummary:
    """Summarize a two-region by four-type table and descriptive association.

    Bootstrap rows, when requested, resample cities independently within each
    region and quantify uncertainty in normalized Shannon evenness.  This small
    generic bootstrap does not encode any project-specific formal result.
    """

    _require_columns(frame, (region_col, type_col))
    regions = tuple(region_order)
    if len(regions) != 2 or len(set(regions)) != 2:
        raise ValueError("region_order must contain two distinct region labels")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be a non-negative integer")

    work = frame[[region_col, type_col]].copy()
    if work.isna().any().any():
        raise ValueError("region and typology values must be complete")
    unknown_regions = sorted(set(work[region_col]) - set(regions))
    unknown_types = sorted(set(work[type_col].astype(str)) - set(TYPE_ORDER))
    if unknown_regions:
        raise ValueError(f"unexpected region labels: {unknown_regions}")
    if unknown_types:
        raise ValueError(f"unexpected typology labels: {unknown_types}")

    index = pd.MultiIndex.from_product(
        [regions, TYPE_ORDER], names=[region_col, type_col]
    )
    counts = (
        work.groupby([region_col, type_col], observed=True)
        .size()
        .reindex(index, fill_value=0)
        .rename("n_city")
        .reset_index()
    )
    denominators = counts.groupby(region_col, observed=True)["n_city"].transform("sum")
    counts["proportion"] = np.where(
        denominators > 0, counts["n_city"] / denominators, np.nan
    )

    matrix = counts.pivot(index=region_col, columns=type_col, values="n_city")
    matrix = matrix.reindex(index=regions, columns=TYPE_ORDER, fill_value=0)
    active = matrix.loc[:, matrix.sum(axis=0).gt(0)]
    if active.shape[1] >= 2 and active.sum(axis=1).gt(0).all():
        chi2, p_value, degrees_of_freedom, _ = chi2_contingency(
            active.to_numpy(dtype=float), correction=False
        )
        n_city = int(active.to_numpy().sum())
        denominator = n_city * min(active.shape[0] - 1, active.shape[1] - 1)
        cramers_v = math.sqrt(float(chi2) / denominator) if denominator > 0 else math.nan
    else:
        chi2 = p_value = cramers_v = math.nan
        degrees_of_freedom = 0
        n_city = int(active.to_numpy().sum())
    association = pd.DataFrame(
        [
            {
                "test": "Pearson chi-square",
                "chi_square": float(chi2),
                "degrees_of_freedom": int(degrees_of_freedom),
                "p_value": float(p_value),
                "cramers_v": float(cramers_v),
                "n_city": n_city,
            }
        ]
    )

    evenness_rows = []
    for region in regions:
        region_counts = matrix.loc[region].to_numpy(dtype=float)
        evenness_rows.append(
            {
                region_col: region,
                "n_city": int(region_counts.sum()),
                "normalized_shannon_evenness": _normalized_shannon_evenness(
                    region_counts
                ),
            }
        )
    evenness = pd.DataFrame(evenness_rows)

    bootstrap_rows: list[dict[str, object]] = []
    if n_bootstrap:
        rng = np.random.default_rng(random_state)
        for replicate in range(1, n_bootstrap + 1):
            for region in regions:
                values = work.loc[work[region_col].eq(region), type_col].astype(str).to_numpy()
                if len(values):
                    sampled = rng.choice(values, size=len(values), replace=True)
                    sampled_counts = np.asarray(
                        [np.sum(sampled == type_name) for type_name in TYPE_ORDER],
                        dtype=float,
                    )
                    estimate = _normalized_shannon_evenness(sampled_counts)
                else:
                    estimate = math.nan
                bootstrap_rows.append(
                    {
                        "replicate": replicate,
                        region_col: region,
                        "normalized_shannon_evenness": estimate,
                    }
                )
    bootstrap = pd.DataFrame(
        bootstrap_rows,
        columns=["replicate", region_col, "normalized_shannon_evenness"],
    )
    return NorthSouthTypologySummary(counts, association, evenness, bootstrap)


__all__ = [
    "NorthSouthTypologySummary",
    "TYPE_ORDER",
    "classify_risk_first_typology",
    "summarize_north_south_typology",
]
