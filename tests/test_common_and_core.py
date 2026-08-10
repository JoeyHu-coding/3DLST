from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import core_analysis
import lst_common


PRIMARY_CONTROLS = (
    "terrain_mean_m",
    "slope_mean_deg",
    "relief_p90_p10_m",
    "p_water_1km",
)


def test_log_coefficient_effect_uses_exact_one_and_ten_percent_increments() -> None:
    assert hasattr(lst_common, "LOG_1PCT")
    assert lst_common.LOG_1PCT == pytest.approx(math.log(1.01))
    assert lst_common.LOG_10PCT == pytest.approx(math.log(1.10))

    converter = getattr(lst_common, "effect_from_log_coefficient", None)
    assert callable(converter)
    beta = -2.75
    assert converter(beta, lst_common.LOG_1PCT) == pytest.approx(beta * math.log(1.01))
    assert converter(beta, lst_common.LOG_10PCT) == pytest.approx(beta * math.log(1.10))


def test_add_per1_columns_emits_only_canonical_exact_values() -> None:
    beta = 3.25
    frame = pd.DataFrame({"BVR_10pct": [beta * lst_common.LOG_10PCT]})

    result = lst_common.add_per1_columns(frame, ["BVR_10pct"])

    assert "BVR_1pct" in result.columns
    assert "BVR_per1pct_approx" not in result.columns
    assert result.loc[0, "BVR_1pct"] == pytest.approx(beta * math.log(1.01))


@pytest.mark.parametrize("missing_column", PRIMARY_CONTROLS)
def test_load_analysis_matrix_rejects_missing_primary_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_column: str,
) -> None:
    matrix_path = tmp_path / "analysis_matrix.parquet"
    matrix_path.touch()
    ln_f = np.log(np.array([0.20, 0.30]))
    ln_h = np.log(np.array([10.0, 12.0]))
    frame = pd.DataFrame(
        {
            lst_common.UID: [1, 1],
            lst_common.RESPONSE: [0.1, 0.2],
            "BF": [0.20, 0.30],
            "MBH_m": [10.0, 12.0],
            "eligible_hvca_main": [True, True],
            "lnF": ln_f,
            "lnH": ln_h,
            "lnV": ln_f + ln_h,
            "terrain_mean_m": [100.0, 110.0],
            "slope_mean_deg": [1.0, 2.0],
            "relief_p90_p10_m": [20.0, 25.0],
            "p_water_1km": [0.05, 0.10],
        }
    )
    schema_columns = [column for column in frame.columns if column != missing_column]
    monkeypatch.setattr(lst_common, "available_columns", lambda _path: schema_columns)
    monkeypatch.setattr(
        lst_common.pd,
        "read_parquet",
        lambda _path, columns: frame.loc[:, columns].copy(),
    )

    with pytest.raises(ValueError) as exc_info:
        lst_common.load_analysis_matrix(matrix_path)

    message = str(exc_info.value)
    assert "primary analysis" in message.lower()
    assert missing_column in message


def test_missing_log_features_preserve_volume_identity() -> None:
    frame = pd.DataFrame(
        {
            lst_common.UID: [1, 1, 1],
            lst_common.RESPONSE: [0.1, 0.2, 0.3],
            "BF": [0.20, 0.25, 0.30],
            "MBH_m": [8.0, 10.0, 12.0],
        }
    )

    result = lst_common.compute_missing_log_features(frame)

    np.testing.assert_allclose(result["lnV"], result["lnF"] + result["lnH"])


def test_load_analysis_matrix_rejects_inconsistent_existing_log_volume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix_path = tmp_path / "analysis_matrix.parquet"
    matrix_path.touch()
    ln_f = np.log(np.array([0.20, 0.30]))
    ln_h = np.log(np.array([10.0, 12.0]))
    frame = pd.DataFrame(
        {
            lst_common.UID: [1, 1],
            lst_common.RESPONSE: [0.1, 0.2],
            "BF": [0.20, 0.30],
            "MBH_m": [10.0, 12.0],
            "eligible_hvca_main": [True, True],
            "lnF": ln_f,
            "lnH": ln_h,
            "lnV": ln_f + ln_h + np.array([0.0, 0.01]),
            "terrain_mean_m": [100.0, 110.0],
            "slope_mean_deg": [1.0, 2.0],
            "relief_p90_p10_m": [20.0, 25.0],
            "p_water_1km": [0.05, 0.10],
        }
    )
    schema_columns = frame.columns.tolist()
    monkeypatch.setattr(lst_common, "available_columns", lambda _path: schema_columns)
    monkeypatch.setattr(
        lst_common.pd,
        "read_parquet",
        lambda _path, columns: frame.loc[:, columns].copy(),
    )

    with pytest.raises(ValueError) as exc_info:
        lst_common.load_analysis_matrix(matrix_path)

    message = str(exc_info.value)
    assert "lnV" in message
    assert "lnF + lnH" in message
    assert "eligible" in message


def test_load_analysis_matrix_completes_nonfinite_logs_for_eligible_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix_path = tmp_path / "analysis_matrix.parquet"
    matrix_path.touch()
    bf = np.array([0.20, 0.30, 0.40])
    mbh = np.array([10.0, 12.0, 14.0])
    expected_ln_f = np.log(bf)
    expected_ln_h = np.log(mbh)
    frame = pd.DataFrame(
        {
            lst_common.UID: [1, 1, 1],
            lst_common.RESPONSE: [0.1, 0.2, 0.3],
            "BF": bf,
            "MBH_m": mbh,
            "eligible_hvca_main": [True, True, True],
            "lnF": [np.nan, expected_ln_f[1], np.inf],
            "lnH": [expected_ln_h[0], np.nan, -np.inf],
            "lnV": [np.nan, np.inf, -np.inf],
            "terrain_mean_m": [100.0, 110.0, 120.0],
            "slope_mean_deg": [1.0, 2.0, 3.0],
            "relief_p90_p10_m": [20.0, 25.0, 30.0],
            "p_water_1km": [0.05, 0.10, 0.15],
        }
    )
    schema_columns = frame.columns.tolist()
    monkeypatch.setattr(lst_common, "available_columns", lambda _path: schema_columns)
    monkeypatch.setattr(
        lst_common.pd,
        "read_parquet",
        lambda _path, columns: frame.loc[:, columns].copy(),
    )

    result = lst_common.load_analysis_matrix(matrix_path)

    np.testing.assert_allclose(result["lnF"], expected_ln_f)
    np.testing.assert_allclose(result["lnH"], expected_ln_h)
    np.testing.assert_allclose(result["lnV"], expected_ln_f + expected_ln_h)


def _fit_primary_tier(monkeypatch: pytest.MonkeyPatch) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260518)
    n_rows = 80
    ln_f = np.linspace(-2.2, -0.2, n_rows)
    ln_h = 0.35 * np.sin(np.linspace(0.0, 5.0, n_rows)) + np.linspace(1.6, 2.5, n_rows)
    frame = pd.DataFrame(
        {
            lst_common.UID: [1] * n_rows,
            lst_common.RESPONSE: 0.8 * ln_f - 0.3 * ln_h + rng.normal(0.0, 0.01, n_rows),
            "lnF": ln_f,
            "lnH": ln_h,
            "lnV": ln_f + ln_h,
            "terrain_mean_m": rng.normal(100.0, 15.0, n_rows),
            "slope_mean_deg": rng.normal(3.0, 0.5, n_rows),
            "relief_p90_p10_m": rng.normal(30.0, 4.0, n_rows),
            "p_water_1km": rng.uniform(0.0, 0.2, n_rows),
        }
    )
    context = pd.DataFrame({lst_common.UID: [1]})
    primary_tier = next(tier for tier in lst_common.MODEL_TIERS if tier.tier_id == lst_common.PRIMARY_TIER_ID)
    monkeypatch.setattr(core_analysis, "MODEL_TIERS", [primary_tier])
    return core_analysis.fit_core_models(frame, context)


def test_core_results_use_exact_scales(monkeypatch: pytest.MonkeyPatch) -> None:
    total, pathway, decomposition = _fit_primary_tier(monkeypatch)

    assert "BVR_1pct" in total.columns
    assert "BVR_F_1pct" in pathway.columns
    assert "BVR_H_1pct" in pathway.columns
    assert "Delta_F_minus_H_1pct" in pathway.columns
    assert "C_F_1pct" in decomposition.columns
    assert "C_H_1pct" in decomposition.columns
    assert total.loc[0, "BVR_10pct"] == pytest.approx(total.loc[0, "beta_lnV"] * math.log(1.10))
    assert total.loc[0, "BVR_1pct"] == pytest.approx(total.loc[0, "beta_lnV"] * math.log(1.01))
    assert pathway.loc[0, "BVR_F_1pct"] == pytest.approx(pathway.loc[0, "beta_lnF"] * math.log(1.01))
    assert pathway.loc[0, "BVR_H_1pct"] == pytest.approx(pathway.loc[0, "beta_lnH"] * math.log(1.01))
    assert decomposition.loc[0, "C_F_1pct"] == pytest.approx(
        decomposition.loc[0, "C_F"] * math.log(1.01)
    )
    assert decomposition.loc[0, "C_H_1pct"] == pytest.approx(
        decomposition.loc[0, "C_H"] * math.log(1.01)
    )


def test_legacy_source_classification_is_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    legacy_api = {"DOMINANCE_SHARE", "classify_source_contribution"}
    assert legacy_api.isdisjoint(vars(lst_common))

    total, pathway, decomposition = _fit_primary_tier(monkeypatch)
    legacy_columns = {"source_class", "dominant_share", "offset_flag"}
    assert legacy_columns.isdisjoint(decomposition.columns)
    assert {"C_F", "C_H", "C_F_10pct", "C_H_10pct"}.issubset(decomposition.columns)

    core_analysis.write_core_outputs(tmp_path, total, pathway, decomposition, pd.DataFrame())
    assert not (tmp_path / "core" / "source_classification_rules.json").exists()
