from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression


TYPE_ORDER = ("Type I", "Type II", "Type III", "Type IV")


def _module(name: str):
    return importlib.import_module(name)


def test_risk_first_typology_is_mutually_exclusive_and_type_iii_has_priority() -> None:
    climate_typology = _module("climate_typology")
    frame = pd.DataFrame(
        {
            "BVR_F_10": [0.06, 0.03, 0.03, 0.00, 0.04, 0.01],
            "BVR_H_10": [0.03, 0.00, 0.06, 0.03, 0.035, 0.005],
            "allocation_contrast_ci_low": [0.01, 0.01, -0.05, -0.05, -0.01, -0.01],
            "allocation_contrast_ci_high": [0.05, 0.05, -0.01, -0.01, 0.02, 0.02],
        },
        index=[10, 11, 12, 13, 14, 15],
    )

    result = climate_typology.classify_risk_first_typology(frame)

    assert result.index.tolist() == frame.index.tolist()
    assert result["risk_first_type"].tolist() == [
        "Type III",
        "Type I",
        "Type III",
        "Type II",
        "Type III",
        "Type IV",
    ]
    assert result["risk_first_type"].notna().all()
    assert len(result) == len(frame)


def test_risk_first_typology_uses_strict_threshold_boundaries() -> None:
    climate_typology = _module("climate_typology")
    frame = pd.DataFrame(
        {
            "BVR_F_10": [0.03, 0.03, 0.00, 0.02],
            "BVR_H_10": [0.02, 0.02, 0.01, 0.021],
            "allocation_contrast_ci_low": [0.001, 0.0, -0.02, -0.02],
            "allocation_contrast_ci_high": [0.02, 0.02, 0.0, 0.02],
        }
    )

    result = climate_typology.classify_risk_first_typology(frame)

    # Delta == 0.01, CI bound == 0, Delta == -0.01, and lower == 0.02
    # all remain outside the strict Type I/II/III inequalities.
    assert result["risk_first_type"].tolist() == ["Type IV"] * 4


def test_north_south_typology_summary_has_complete_composition_and_effect_size() -> None:
    climate_typology = _module("climate_typology")
    frame = pd.DataFrame(
        {
            "global_region_group": ["global_south"] * 8 + ["global_north"] * 8,
            "risk_first_type": [
                "Type I",
                "Type I",
                "Type II",
                "Type II",
                "Type III",
                "Type III",
                "Type IV",
                "Type IV",
                "Type I",
                "Type I",
                "Type I",
                "Type I",
                "Type III",
                "Type III",
                "Type III",
                "Type III",
            ],
        }
    )

    summary = climate_typology.summarize_north_south_typology(frame)

    assert len(summary.composition) == 8
    assert summary.composition.groupby("global_region_group")["n_city"].sum().to_dict() == {
        "global_north": 8,
        "global_south": 8,
    }
    proportions = summary.composition.groupby("global_region_group")["proportion"].sum()
    np.testing.assert_allclose(proportions.to_numpy(), 1.0)
    association = summary.association.iloc[0]
    assert association["test"] == "Pearson chi-square"
    assert association["degrees_of_freedom"] == 3
    assert association["cramers_v"] > 0
    assert set(summary.evenness["global_region_group"]) == {
        "global_south",
        "global_north",
    }
    assert summary.evenness["normalized_shannon_evenness"].between(0, 1).all()
    assert summary.bootstrap.empty


def test_north_south_evenness_bootstrap_is_optional_and_deterministic() -> None:
    climate_typology = _module("climate_typology")
    frame = pd.DataFrame(
        {
            "global_region_group": ["global_south"] * 8 + ["global_north"] * 8,
            "risk_first_type": list(TYPE_ORDER) * 2 + ["Type I"] * 4 + ["Type III"] * 4,
        }
    )

    first = climate_typology.summarize_north_south_typology(
        frame, n_bootstrap=20, random_state=7
    )
    second = climate_typology.summarize_north_south_typology(
        frame, n_bootstrap=20, random_state=7
    )

    assert len(first.bootstrap) == 40
    pd.testing.assert_frame_equal(first.bootstrap, second.bootstrap)


def test_grouped_oof_predictions_cover_rows_without_group_leakage() -> None:
    nonlinear_analysis = _module("nonlinear_analysis")
    rng = np.random.default_rng(19)
    groups = np.repeat(["A", "B", "C", "D", "E", "F"], 12)
    x1 = rng.normal(size=len(groups))
    x2 = rng.normal(size=len(groups))
    x2[::11] = np.nan
    frame = pd.DataFrame(
        {
            "city": groups,
            "x1": x1,
            "x2": x2,
            "response": 1.5 * x1 - 0.5 * np.nan_to_num(x2) + rng.normal(0, 0.05, len(groups)),
        },
        index=np.arange(1000, 1000 + len(groups)),
    )

    predictions, performance = nonlinear_analysis.grouped_oof_predictions(
        frame,
        feature_cols=["x1", "x2"],
        target_col="response",
        group_col="city",
        n_splits=3,
        random_state=23,
    )

    assert len(predictions) == len(frame)
    assert predictions["row_position"].tolist() == list(range(len(frame)))
    assert predictions["source_index"].tolist() == frame.index.tolist()
    assert predictions["model_prediction"].notna().all()
    assert predictions.groupby("city")["fold"].nunique().eq(1).all()
    for fold in predictions["fold"].unique():
        test_groups = set(predictions.loc[predictions["fold"].eq(fold), "city"])
        train_groups = set(predictions.loc[~predictions["fold"].eq(fold), "city"])
        assert test_groups.isdisjoint(train_groups)
    assert performance["splitter"].eq("GroupKFold").all()
    assert performance["imputation"].eq("training_fold_feature_median").all()
    assert set(performance["evaluation_scope"]) == {"fold", "pooled_oof"}
    assert performance["model_prediction_kind"].eq(
        "held_out_group_model_prediction_not_causal_effect"
    ).all()


def test_grouped_oof_predictions_preserve_preassigned_group_folds() -> None:
    nonlinear_analysis = _module("nonlinear_analysis")
    rng = np.random.default_rng(29)
    components = np.repeat([f"OC_{index}" for index in range(10)], 4)
    supplied_folds = np.repeat(np.arange(5), 8)
    x1 = rng.normal(size=len(components))
    x2 = rng.normal(size=len(components))
    frame = pd.DataFrame(
        {
            "city_overlap_component": components,
            "outer_fold": supplied_folds,
            "x1": x1,
            "x2": x2,
            "target": 0.8 * x1 - 0.2 * x2,
        }
    )

    predictions, performance = nonlinear_analysis.grouped_oof_predictions(
        frame,
        feature_cols=["x1", "x2"],
        target_col="target",
        group_col="city_overlap_component",
        fold_col="outer_fold",
        n_splits=5,
    )

    assert predictions["fold"].tolist() == frame["outer_fold"].tolist()
    assert predictions.groupby("city_overlap_component")["fold"].nunique().eq(1).all()
    assert performance["splitter"].eq("preassigned_grouped_folds").all()
    pooled = performance.loc[performance["evaluation_scope"].eq("pooled_oof")]
    assert pooled["n_observations"].iat[0] == len(frame)


def test_monte_carlo_shapley_is_deterministic_and_reconstructs_model_prediction() -> None:
    nonlinear_analysis = _module("nonlinear_analysis")
    features = ["x1", "x2", "x3"]
    frame = pd.DataFrame(
        {
            "x1": [-2.0, -0.5, 1.0, 2.5],
            "x2": [1.0, 2.0, 4.0, 8.0],
            "x3": [0.5, -1.0, 0.0, 2.0],
        }
    )
    y = 2.0 * frame["x1"] - 3.0 * frame["x2"] + 0.25 * frame["x3"]
    model = LinearRegression().fit(frame[features], y)

    first = nonlinear_analysis.monte_carlo_shapley_attribution(
        model,
        frame,
        features,
        n_permutations=24,
        random_state=31,
    )
    second = nonlinear_analysis.monte_carlo_shapley_attribution(
        model,
        frame,
        features,
        n_permutations=24,
        random_state=31,
    )

    pd.testing.assert_frame_equal(first, second)
    totals = first.groupby("row_position", sort=True).agg(
        baseline=("baseline_model_prediction", "first"),
        attribution=("shapley_model_prediction_attribution", "sum"),
        prediction=("model_prediction", "first"),
    )
    np.testing.assert_allclose(
        totals["baseline"] + totals["attribution"],
        totals["prediction"],
        atol=1e-10,
    )
    assert first["interpretation_scope"].eq(
        "model_prediction_attribution_not_causal_effect"
    ).all()


def test_accumulated_local_effects_use_quantile_bins_and_explicit_prediction_fields() -> None:
    nonlinear_analysis = _module("nonlinear_analysis")
    frame = pd.DataFrame(
        {
            "x1": np.linspace(-2.0, 2.0, 100),
            "x2": np.sin(np.linspace(0.0, 4.0, 100)),
        }
    )
    model = LinearRegression().fit(frame, 3.0 * frame["x1"] + 0.5 * frame["x2"])

    ale = nonlinear_analysis.accumulated_local_effects(
        model,
        frame,
        feature_cols=["x1", "x2"],
        feature_col="x1",
        n_bins=5,
        min_bin_count=10,
    )

    assert len(ale) == 5
    assert {
        "feature",
        "bin",
        "bin_left",
        "bin_right",
        "bin_center",
        "n_bin",
        "mean_local_model_prediction_effect",
        "ale_model_prediction_effect",
        "interpretation_scope",
    }.issubset(ale.columns)
    assert ale["n_bin"].ge(10).all()
    assert ale["bin_left"].is_monotonic_increasing
    weighted_mean = np.average(
        ale["ale_model_prediction_effect"], weights=ale["n_bin"]
    )
    assert weighted_mean == pytest.approx(0.0, abs=1e-12)
    assert ale["interpretation_scope"].eq(
        "accumulated_local_effect_on_model_prediction_not_causal_effect"
    ).all()


def test_temperature_product_comparison_reports_correlations_and_loco_range() -> None:
    sensor_validation = _module("sensor_validation")
    landsat = pd.DataFrame(
        {
            "UID": [1, 2, 3, 4, 5],
            "landsat_f": [0.1, 0.2, 0.3, 0.4, 0.5],
            "landsat_h": [0.5, 0.4, 0.3, 0.2, 0.1],
        }
    )
    gshtd = pd.DataFrame(
        {
            "UID": [1, 2, 3, 4, 5],
            "gshtd_f": [0.2, 0.4, 0.6, 0.8, 1.0],
            "gshtd_h": [1.0, 0.8, 0.6, 0.4, 0.2],
        }
    )

    summary = sensor_validation.compare_temperature_products(
        landsat,
        gshtd,
        uid_col="UID",
        metric_mapping={
            "BVR-F": ("landsat_f", "gshtd_f"),
            "BVR-H": ("landsat_h", "gshtd_h"),
        },
    )

    assert summary["metric"].tolist() == ["BVR-F", "BVR-H"]
    assert summary["pearson_r"].tolist() == pytest.approx([1.0, 1.0])
    assert summary["spearman_rho"].tolist() == pytest.approx([1.0, 1.0])
    assert summary["leave_one_city_out_pearson_r_min"].tolist() == pytest.approx([1.0, 1.0])
    assert summary["leave_one_city_out_pearson_r_max"].tolist() == pytest.approx([1.0, 1.0])
    assert summary["interpretation_scope"].eq(
        "cross_product_correspondence_not_synchronous_validation_or_causality"
    ).all()


def test_multiscale_pathways_use_equal_city_medians() -> None:
    sensor_validation = _module("sensor_validation")
    frame = pd.DataFrame(
        {
            "scale_m": [100, 100, 100, 100, 100, 1000, 1000, 1000],
            "UID": [1, 1, 2, 2, 3, 1, 2, 3],
            "BVR_F_10pct": [1.0, 3.0, 10.0, 14.0, 5.0, 2.0, 8.0, 5.0],
            "BVR_H_10pct": [0.0, 2.0, 5.0, 7.0, 4.0, 1.0, 4.0, 3.0],
            "Delta_F_minus_H_10pct": [1.0, 1.0, 5.0, 7.0, 1.0, 1.0, 4.0, 2.0],
        }
    )

    result = sensor_validation.summarize_multiscale_pathways(frame)

    assert result["scale_m"].tolist() == [100, 1000]
    assert result["n_city"].tolist() == [3, 3]
    row_100 = result.loc[result["scale_m"].eq(100)].iloc[0]
    # Equal-city values are medians of UID-level medians: F=(2,12,5),
    # H=(1,6,4), Delta=(1,6,1).
    assert row_100["equal_city_median_BVR_F"] == pytest.approx(5.0)
    assert row_100["equal_city_median_BVR_H"] == pytest.approx(4.0)
    assert row_100["equal_city_median_Delta"] == pytest.approx(1.0)


def test_viewing_angle_interaction_recovers_positive_centered_interaction() -> None:
    sensor_validation = _module("sensor_validation")
    rng = np.random.default_rng(41)
    rows: list[dict[str, object]] = []
    for city_index, city in enumerate(["C1", "C2", "C3", "C4"]):
        for acquisition_index, acquisition in enumerate(["A1", "A2", "A3"]):
            for _ in range(25):
                height = rng.uniform(2.0, 30.0)
                vza = rng.uniform(1.0, 9.0)
                control = rng.normal()
                outcome = (
                    0.7 * height
                    - 0.2 * vza
                    + 0.4 * height * vza
                    + 0.3 * control
                    + 0.8 * city_index
                    + 0.5 * acquisition_index
                    + rng.normal(0.0, 0.03)
                )
                rows.append(
                    {
                        "city": city,
                        "acquisition": acquisition,
                        "temperature": outcome,
                        "height": height,
                        "vza": vza,
                        "control": control,
                    }
                )
    frame = pd.DataFrame(rows)

    result = sensor_validation.fit_viewing_angle_interaction(
        frame,
        outcome="temperature",
        height="height",
        vza="vza",
        controls=["control"],
        city="city",
        acquisition="acquisition",
    )

    row = result.iloc[0]
    assert row["term"] == "height_centered_x_vza_centered"
    assert row["estimate"] == pytest.approx(0.4, abs=0.01)
    assert row["standard_error"] > 0
    assert row["ci_low"] < row["estimate"] < row["ci_high"]
    assert row["fixed_effects"] == "city_and_within_city_acquisition"
    assert row["interpretation_scope"] == "association_not_causal_effect"
