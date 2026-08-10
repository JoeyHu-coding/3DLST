from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import lst_common
import mechanism_analysis
import robustness_analysis


PRIMARY_CONTROLS = (
    "terrain_mean_m",
    "slope_mean_deg",
    "relief_p90_p10_m",
    "p_water_1km",
)


def _primary_tier() -> lst_common.ModelTier:
    return next(tier for tier in lst_common.MODEL_TIERS if tier.tier_id == lst_common.PRIMARY_TIER_ID)


def _city_frame(
    *,
    uid: int,
    n_rows: int,
    beta_f: float = 1.0,
    beta_h: float = 0.2,
    narrow_height: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(10_000 + uid)
    ln_f = np.linspace(math.log(0.05), math.log(0.30), n_rows)
    height_span = 0.06 if narrow_height else 0.85
    ln_h = 2.2 + height_span * np.sin(np.linspace(0.0, 8.0, n_rows)) + rng.normal(0.0, 0.01, n_rows)
    controls = {
        "terrain_mean_m": rng.normal(100.0, 20.0, n_rows),
        "slope_mean_deg": rng.normal(3.0, 0.8, n_rows),
        "relief_p90_p10_m": rng.normal(30.0, 6.0, n_rows),
        "p_water_1km": rng.uniform(0.0, 0.2, n_rows),
    }
    response = beta_f * ln_f + beta_h * ln_h + rng.normal(0.0, 0.002, n_rows)
    return pd.DataFrame(
        {
            lst_common.UID: uid,
            lst_common.RESPONSE: response,
            "BF": np.exp(ln_f),
            "MBH_m": np.exp(ln_h),
            **controls,
        }
    )


def test_pathway_record_requires_both_log_predictor_ranges_for_stability() -> None:
    record = robustness_analysis.pathway_record(
        1,
        _city_frame(uid=1, n_rows=220, narrow_height=True).assign(
            lnF=lambda frame: np.log(frame["BF"]),
            lnH=lambda frame: np.log(frame["MBH_m"]),
        ),
        _primary_tier(),
        min_rows=200,
    )

    assert record["estimable"]
    assert record["n_model"] >= 200
    assert record["lnF_range"] >= 0.2
    assert record["lnH_range"] < 0.2
    assert not record["stable_city"]


def test_sensitivity_configs_is_exact_twelve_cell_factorial_with_explicit_shares() -> None:
    frame = pd.concat(
        [
            _city_frame(uid=1, n_rows=220, beta_f=1.0, beta_h=0.0),
            _city_frame(uid=2, n_rows=220, beta_f=-1.0, beta_h=0.0),
        ],
        ignore_index=True,
    )

    result = robustness_analysis.sensitivity_configs(frame)

    assert len(result) == 12
    assert set(map(tuple, result[["bf", "min_rows"]].itertuples(index=False, name=None))) == {
        (bf, min_rows)
        for bf in (0.005, 0.010, 0.020, 0.050)
        for min_rows in (50, 100, 200)
    }
    assert result["config"].is_unique
    assert "baseline_bf001_min100" in set(result["config"])
    assert (result["n_city_stable"] == 2).all()
    expected_columns = {
        "BVR_F_10pct_median",
        "BVR_H_10pct_median",
        "Delta_10pct_median",
        "share_Delta_gt_zero",
        "share_Delta10_gt_0p01C",
        "share_Delta10_lt_minus_0p01C",
    }
    assert expected_columns.issubset(result.columns)
    assert np.allclose(result["share_Delta_gt_zero"], 0.5)
    assert np.allclose(result["share_Delta10_gt_0p01C"], 0.5)
    assert np.allclose(result["share_Delta10_lt_minus_0p01C"], 0.5)


def test_control_diagnostics_returns_city_vif_and_four_control_support() -> None:
    rng = np.random.default_rng(20260810)
    pieces: list[pd.DataFrame] = []
    for uid in (1, 2):
        n_rows = 180
        control_data = rng.normal(size=(n_rows, 4))
        ln_f = 0.8 * control_data[:, 0] - 0.4 * control_data[:, 1] + rng.normal(0.0, 0.5, n_rows)
        ln_h = rng.normal(size=n_rows)
        pieces.append(
            pd.DataFrame(
                {
                    lst_common.UID: uid,
                    "lnF": ln_f,
                    "lnH": ln_h,
                    **{name: control_data[:, idx] for idx, name in enumerate(PRIMARY_CONTROLS)},
                }
            )
        )
    frame = pd.concat(pieces, ignore_index=True)

    result = robustness_analysis.control_diagnostics(frame)

    assert set(result) == {"predictor_vif", "morphology_support"}
    predictor_vif = result["predictor_vif"]
    morphology_support = result["morphology_support"]
    assert len(predictor_vif) == 2 * (2 + len(PRIMARY_CONTROLS))
    assert set(predictor_vif["predictor"]) == {"lnF", "lnH", *PRIMARY_CONTROLS}
    assert (predictor_vif["vif"] >= 1.0).all()
    assert len(morphology_support) == 4
    assert set(morphology_support["predictor"]) == {"lnF", "lnH"}
    assert {
        "original_variance",
        "residual_variance",
        "residual_variance_fraction",
    }.issubset(morphology_support.columns)
    assert np.allclose(
        morphology_support["residual_variance_fraction"],
        morphology_support["residual_variance"] / morphology_support["original_variance"],
    )
    assert morphology_support["residual_variance_fraction"].between(0.0, 1.0).all()


def test_run_city_moderators_does_not_require_legacy_source_class() -> None:
    total = pd.DataFrame(
        {
            lst_common.UID: [1],
            "model_tier": [lst_common.PRIMARY_TIER_ID],
            "stable_city": [True],
            "BVR_10pct": [0.1],
        }
    )
    pathway = pd.DataFrame(
        {
            lst_common.UID: [1],
            "model_tier": [lst_common.PRIMARY_TIER_ID],
            "stable_city": [True],
            "beta_lnF": [1.0],
            "beta_lnH": [0.2],
            "BVR_F_10pct": [0.1],
            "BVR_H_10pct": [0.02],
            "Delta_F_minus_H_10pct": [0.08],
        }
    )
    decomposition = pd.DataFrame(
        {
            lst_common.UID: [1],
            "model_tier": [lst_common.PRIMARY_TIER_ID],
            "stable_city": [True],
            "C_F_10pct": [0.07],
            "C_H_10pct": [0.01],
        }
    )

    result = mechanism_analysis.run_city_moderators(
        total,
        pathway,
        decomposition,
        pd.DataFrame({lst_common.UID: [1]}),
    )

    assert isinstance(result, pd.DataFrame)


def test_grid_interactions_use_centered_products_and_exact_one_percent_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260811)
    pieces: list[pd.DataFrame] = []
    for uid, f_offset, h_offset, mod_offset in ((1, -8.0, 3.0, 100.0), (2, 12.0, -5.0, -50.0)):
        n_rows = 600
        pieces.append(
            pd.DataFrame(
                {
                    lst_common.UID: uid,
                    "lnF": f_offset + rng.normal(size=n_rows),
                    "lnH": h_offset + rng.normal(size=n_rows),
                    "moderator": mod_offset + rng.normal(size=n_rows),
                }
            )
        )
    frame = pd.concat(pieces, ignore_index=True)
    ln_f_centered = lst_common.demean_series(frame, "lnF")
    ln_h_centered = lst_common.demean_series(frame, "lnH")
    z_mod = lst_common.within_city_z(frame, "moderator")
    interaction_f = (ln_f_centered * z_mod).groupby(frame[lst_common.UID], observed=True).transform(
        lambda values: values - values.mean()
    )
    interaction_h = (ln_h_centered * z_mod).groupby(frame[lst_common.UID], observed=True).transform(
        lambda values: values - values.mean()
    )
    frame[lst_common.RESPONSE] = (
        0.2 * ln_f_centered
        - 0.1 * ln_h_centered
        + 0.3 * z_mod
        + 1.5 * interaction_f
        - 0.7 * interaction_h
    )
    monkeypatch.setattr(mechanism_analysis, "GRID_MODERATORS", [("moderator", "Moderator")])
    monkeypatch.setattr(
        mechanism_analysis,
        "MODEL_TIERS",
        [lst_common.ModelTier(lst_common.PRIMARY_TIER_ID, "Primary", ())],
    )

    result = mechanism_analysis.run_grid_interactions(frame)

    estimates = result.set_index("term")["estimate"]
    assert estimates["theta_F"] == pytest.approx(1.5, abs=1e-10)
    assert estimates["theta_H"] == pytest.approx(-0.7, abs=1e-10)
    assert np.allclose(result["per1pct_estimate"], result["estimate"] * lst_common.LOG_1PCT)
    assert np.allclose(result["per1pct_se"], result["cluster_se"] * lst_common.LOG_1PCT)


def test_surface_context_outputs_canonical_exact_one_percent_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260812)
    pieces: list[pd.DataFrame] = []
    for uid in (1, 2):
        n_rows = 70
        ln_f = rng.normal(size=n_rows)
        ln_h = rng.normal(size=n_rows)
        pieces.append(
            pd.DataFrame(
                {
                    lst_common.UID: uid,
                    "lnF": ln_f,
                    "lnH": ln_h,
                    "context": 0.8 * ln_f - 0.4 * ln_h + rng.normal(0.0, 0.01, n_rows),
                }
            )
        )
    frame = pd.concat(pieces, ignore_index=True)
    monkeypatch.setattr(mechanism_analysis, "SURFACE_CONTEXT_OUTCOMES", [("context", "Context")])
    monkeypatch.setattr(
        mechanism_analysis,
        "MODEL_TIERS",
        [lst_common.ModelTier(lst_common.PRIMARY_TIER_ID, "Primary", ())],
    )

    city, pooled = mechanism_analysis.run_surface_context_models(
        frame,
        pd.DataFrame({lst_common.UID: [1, 2]}),
    )

    assert np.allclose(city["path_F_1pct"], city["beta_lnF"] * lst_common.LOG_1PCT)
    assert np.allclose(city["path_H_1pct"], city["beta_lnH"] * lst_common.LOG_1PCT)
    assert np.allclose(pooled["per1pct_estimate"], pooled["estimate"] * lst_common.LOG_1PCT)
    assert np.allclose(pooled["per1pct_se"], pooled["cluster_se"] * lst_common.LOG_1PCT)


def test_nested_leave_climate_out_is_group_nested_and_audits_train_only_scaling() -> None:
    rng = np.random.default_rng(20260813)
    rows: list[pd.DataFrame] = []
    for group_idx, climate in enumerate(("A", "B", "C", "D")):
        n_rows = 35
        x1 = rng.normal(loc=group_idx * 20.0, scale=2.0, size=n_rows)
        x2 = rng.normal(loc=-group_idx * 5.0, scale=1.0, size=n_rows)
        rows.append(
            pd.DataFrame(
                {
                    "climate_group": climate,
                    "x1": x1,
                    "x2": x2,
                    "target": 1.2 * x1 - 0.6 * x2 + rng.normal(0.0, 0.2, n_rows),
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)

    result = mechanism_analysis.nested_leave_climate_out(
        frame,
        features=["x1", "x2"],
        target="target",
        alphas=(0.01, 1.0, 100.0),
    )

    assert {"predictions", "tuning", "fold_audit"}.issubset(result)
    predictions = result["predictions"]
    tuning = result["tuning"]
    audit = result["fold_audit"]
    assert len(predictions) == len(frame)
    assert np.isfinite(predictions["predicted"]).all()
    assert tuning.groupby("held_out_climate", observed=True)["selected"].sum().eq(1).all()
    assert set(tuning["alpha"]) == {0.01, 1.0, 100.0}

    for row in audit.itertuples(index=False):
        train_groups = set(row.train_climate_groups)
        assert row.held_out_climate not in train_groups
        if row.stage == "inner":
            assert row.validation_climate not in train_groups
            train_mask = ~frame["climate_group"].isin([row.held_out_climate, row.validation_climate])
        else:
            train_mask = frame["climate_group"] != row.held_out_climate
        assert row.scaler_mean_x1 == pytest.approx(frame.loc[train_mask, "x1"].mean())
        assert row.scaler_mean_x2 == pytest.approx(frame.loc[train_mask, "x2"].mean())
