#!/usr/bin/env python3
"""Run the compact city-level nonlinear and typology analyses."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from climate_typology import (
    classify_risk_first_typology,
    summarize_north_south_typology,
)
from lst_common import (
    FINAL_CITY_DRIVER_FEATURES,
    RANDOM_STATE,
    now_utc,
    write_csv,
    write_json,
)
from nonlinear_analysis import (
    accumulated_local_effects,
    grouped_oof_predictions,
    make_hist_gradient_boosting_regressor,
    monte_carlo_shapley_attribution,
)


TARGETS = ("BVR_F_10", "BVR_H_10")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--city-table",
        type=Path,
        required=True,
        help="Path to city_analysis_core.csv from the data release.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "3dlst_city_analysis",
        help="Directory where city-level result tables will be written.",
    )
    parser.add_argument(
        "--shapley-permutations",
        type=int,
        default=12,
        help="Monte Carlo feature orderings for each Shapley evaluation.",
    )
    parser.add_argument(
        "--typology-bootstrap",
        type=int,
        default=0,
        help="Optional within-region typology bootstrap replicates.",
    )
    parser.add_argument(
        "--ale-bins",
        type=int,
        default=10,
        help="Quantile bins for accumulated local effects.",
    )
    parser.add_argument(
        "--min-ale-bin-count",
        type=int,
        default=5,
        help="Minimum city count retained in an ALE bin.",
    )
    return parser.parse_args(argv)


def _fit_fixed_model(
    frame: pd.DataFrame, features: list[str], target: str
) -> object:
    """Fit the documented fixed HGBR model after median imputation."""

    x = (
        frame.loc[:, features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )
    medians = x.median(axis=0, numeric_only=True)
    if not np.isfinite(medians.to_numpy(dtype=float)).all():
        raise ValueError("city table contains a feature with no finite median")
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f"{target} must be complete and finite")
    model = make_hist_gradient_boosting_regressor(random_state=RANDOM_STATE)
    model.fit(x.fillna(medians), y)
    return model


def _validate_release_table(frame: pd.DataFrame, features: list[str]) -> None:
    required = {
        "UID",
        "city_overlap_component",
        "outer_fold",
        "global_region_group",
        "risk_first_type",
        "allocation_contrast_ci_low",
        "allocation_contrast_ci_high",
        *TARGETS,
        *features,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"city table is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("city table must contain at least one row")
    if frame["UID"].isna().any() or frame["UID"].duplicated().any():
        raise ValueError("city table must contain one complete row per UID")

    classified = classify_risk_first_typology(frame)
    observed = frame["risk_first_type"].astype("string")
    reconstructed = classified["risk_first_type"].astype("string")
    mismatched = observed.ne(reconstructed).fillna(True)
    if mismatched.any():
        raise ValueError(
            "risk_first_type does not match the released risk-first thresholds "
            f"for {int(mismatched.sum())} cities"
        )


def main(argv: list[str] | None = None) -> None:
    """Run the fixed nonlinear prediction and risk-first typology summaries."""

    args = parse_args(argv)
    frame = pd.read_csv(args.city_table)
    features = list(FINAL_CITY_DRIVER_FEATURES)
    _validate_release_table(frame, features)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_splits = int(pd.to_numeric(frame["outer_fold"], errors="raise").nunique())
    prediction_tables: list[pd.DataFrame] = []
    performance_tables: list[pd.DataFrame] = []
    shapley_tables: list[pd.DataFrame] = []
    ale_tables: list[pd.DataFrame] = []

    for target in TARGETS:
        predictions, performance = grouped_oof_predictions(
            frame,
            feature_cols=features,
            target_col=target,
            group_col="city_overlap_component",
            fold_col="outer_fold",
            n_splits=n_splits,
            random_state=RANDOM_STATE,
        )
        predictions.insert(0, "target", target)
        predictions.insert(
            1,
            "UID",
            frame.iloc[predictions["row_position"].to_numpy(dtype=int)]["UID"].to_numpy(),
        )
        performance.insert(0, "target", target)
        prediction_tables.append(predictions)
        performance_tables.append(performance)

        model = _fit_fixed_model(frame, features, target)
        shapley = monte_carlo_shapley_attribution(
            model,
            frame,
            features,
            n_permutations=args.shapley_permutations,
            random_state=RANDOM_STATE,
        )
        shapley.insert(0, "target", target)
        shapley.insert(
            1,
            "UID",
            frame.iloc[shapley["row_position"].to_numpy(dtype=int)]["UID"].to_numpy(),
        )
        shapley_tables.append(shapley)

        for feature in features:
            ale = accumulated_local_effects(
                model,
                frame,
                feature_cols=features,
                feature_col=feature,
                n_bins=args.ale_bins,
                min_bin_count=args.min_ale_bin_count,
            )
            ale.insert(0, "target", target)
            ale_tables.append(ale)

    oof_predictions = pd.concat(prediction_tables, ignore_index=True)
    oof_performance = pd.concat(performance_tables, ignore_index=True)
    shapley_values = pd.concat(shapley_tables, ignore_index=True)
    shapley_values["abs_shapley_model_prediction_attribution"] = shapley_values[
        "shapley_model_prediction_attribution"
    ].abs()
    shapley_summary = (
        shapley_values.groupby(["target", "feature"], observed=True, sort=False)
        .agg(
            mean_shapley_model_prediction_attribution=(
                "shapley_model_prediction_attribution",
                "mean",
            ),
            mean_abs_shapley_model_prediction_attribution=(
                "abs_shapley_model_prediction_attribution",
                "mean",
            ),
            n_city=("UID", "nunique"),
        )
        .reset_index()
    )
    shapley_summary["interpretation_scope"] = (
        "model_prediction_attribution_not_causal_effect"
    )
    ale_curves = pd.concat(ale_tables, ignore_index=True)

    typology = summarize_north_south_typology(
        frame,
        region_col="global_region_group",
        type_col="risk_first_type",
        n_bootstrap=args.typology_bootstrap,
        random_state=RANDOM_STATE,
    )

    nonlinear_dir = args.output_dir / "nonlinear"
    typology_dir = args.output_dir / "typology"
    write_csv(oof_predictions, nonlinear_dir / "oof_predictions.csv")
    write_csv(oof_performance, nonlinear_dir / "oof_performance.csv")
    write_csv(shapley_values, nonlinear_dir / "shapley_values.csv")
    write_csv(shapley_summary, nonlinear_dir / "shapley_summary.csv")
    write_csv(ale_curves, nonlinear_dir / "ale_curves.csv")
    write_csv(typology.composition, typology_dir / "composition.csv")
    write_csv(typology.association, typology_dir / "association.csv")
    write_csv(typology.evenness, typology_dir / "evenness.csv")
    if len(typology.bootstrap):
        write_csv(typology.bootstrap, typology_dir / "bootstrap.csv")

    manifest = {
        "created_utc": now_utc(),
        "python": sys.version,
        "platform": platform.platform(),
        "city_table": str(args.city_table),
        "output_dir": str(args.output_dir),
        "n_city": int(len(frame)),
        "n_components": int(frame["city_overlap_component"].nunique()),
        "fold_labels": sorted(
            pd.to_numeric(frame["outer_fold"], errors="raise").astype(int).unique().tolist()
        ),
        "targets": list(TARGETS),
        "features": features,
        "model": "HistGradientBoostingRegressor_fixed_configuration",
        "random_state": RANDOM_STATE,
        "shapley_permutations": int(args.shapley_permutations),
        "ale_bins": int(args.ale_bins),
        "min_ale_bin_count": int(args.min_ale_bin_count),
        "typology_bootstrap": int(args.typology_bootstrap),
        "interpretation_scope": "predictive_and_descriptive_associations_not_causal_effects",
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "manifest": str(args.output_dir / "run_manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
