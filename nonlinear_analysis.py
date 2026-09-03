"""Compact nonlinear predictive diagnostics for tabular 3DLST analyses.

All returned effects describe fitted-model predictions.  They are predictive
associations and must not be interpreted as causal effects.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


RANDOM_STATE = 20260518
MODEL_PREDICTION_KIND = "held_out_group_model_prediction_not_causal_effect"


def make_hist_gradient_boosting_regressor(
    random_state: int = RANDOM_STATE,
) -> HistGradientBoostingRegressor:
    """Return the fixed, reproducible nonlinear prediction model."""

    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=120,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        max_bins=255,
        early_stopping=False,
        tol=1e-7,
        random_state=random_state,
    )


def _validate_feature_columns(
    frame: pd.DataFrame, feature_cols: Sequence[str]
) -> list[str]:
    features = list(feature_cols)
    if not features:
        raise ValueError("feature_cols must contain at least one column")
    if len(features) != len(set(features)):
        raise ValueError("feature_cols must be unique")
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing feature columns: {', '.join(missing)}")
    return features


def _numeric_feature_matrix(
    frame: pd.DataFrame, feature_cols: Sequence[str]
) -> pd.DataFrame:
    return (
        frame.loc[:, list(feature_cols)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )


def _training_medians(x_train: pd.DataFrame) -> pd.Series:
    medians = x_train.median(axis=0, numeric_only=True)
    invalid = medians.index[~np.isfinite(medians.to_numpy(dtype=float))].tolist()
    if invalid:
        raise ValueError(
            "training fold cannot median-impute all-missing features: "
            + ", ".join(invalid)
        )
    return medians


def _prediction_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(observed, predicted))
    r2 = float(r2_score(observed, predicted)) if len(observed) >= 2 else math.nan
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": r2,
    }


def grouped_oof_predictions(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    group_col: str,
    *,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return leakage-controlled grouped out-of-fold model predictions.

    ``GroupKFold`` keeps each group wholly in one test fold.  Missing feature
    values in both train and test partitions are filled only with medians from
    that fold's training partition.
    """

    features = _validate_feature_columns(frame, feature_cols)
    missing = [column for column in (target_col, group_col) if column not in frame]
    if missing:
        raise ValueError(f"frame is missing required columns: {', '.join(missing)}")
    if not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two")
    if frame.empty:
        raise ValueError("frame must contain at least one row")

    x = _numeric_feature_matrix(frame, features)
    y = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)
    groups = frame[group_col]
    if not np.isfinite(y).all():
        raise ValueError("target values must be finite")
    if groups.isna().any():
        raise ValueError("group values must be complete")
    n_groups = int(groups.nunique(dropna=False))
    if n_groups < n_splits:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of groups ({n_groups})"
        )

    predictions = np.full(len(frame), np.nan, dtype=float)
    fold_ids = np.full(len(frame), -1, dtype=int)
    performance_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(x, y, groups=groups), start=1
    ):
        train_groups = set(groups.iloc[train_index])
        test_groups = set(groups.iloc[test_index])
        if not train_groups.isdisjoint(test_groups):
            raise RuntimeError("GroupKFold produced overlapping train and test groups")

        medians = _training_medians(x.iloc[train_index])
        x_train = x.iloc[train_index].fillna(medians)
        x_test = x.iloc[test_index].fillna(medians)
        model = make_hist_gradient_boosting_regressor(random_state=random_state)
        model.fit(x_train, y[train_index])
        fold_prediction = model.predict(x_test).astype(float, copy=False)
        predictions[test_index] = fold_prediction
        fold_ids[test_index] = fold
        performance_rows.append(
            {
                "evaluation_scope": "fold",
                "fold": fold,
                "n_observations": int(len(test_index)),
                "n_train_groups": int(len(train_groups)),
                "n_test_groups": int(len(test_groups)),
                **_prediction_metrics(y[test_index], fold_prediction),
                "model": "HistGradientBoostingRegressor_fixed_configuration",
                "splitter": "GroupKFold",
                "imputation": "training_fold_feature_median",
                "model_prediction_kind": MODEL_PREDICTION_KIND,
            }
        )

    if not np.isfinite(predictions).all() or np.any(fold_ids < 1):
        raise RuntimeError("out-of-fold predictions did not cover every input row")
    performance_rows.append(
        {
            "evaluation_scope": "pooled_oof",
            "fold": pd.NA,
            "n_observations": int(len(frame)),
            "n_train_groups": pd.NA,
            "n_test_groups": n_groups,
            **_prediction_metrics(y, predictions),
            "model": "HistGradientBoostingRegressor_fixed_configuration",
            "splitter": "GroupKFold",
            "imputation": "training_fold_feature_median",
            "model_prediction_kind": MODEL_PREDICTION_KIND,
        }
    )
    prediction_table = pd.DataFrame(
        {
            "row_position": np.arange(len(frame), dtype=int),
            "source_index": frame.index.to_numpy(copy=True),
            group_col: groups.to_numpy(copy=True),
            "fold": fold_ids,
            "observed_target": y,
            "model_prediction": predictions,
            "observed_minus_model_prediction": y - predictions,
            "model_prediction_kind": MODEL_PREDICTION_KIND,
        }
    )
    performance = pd.DataFrame(performance_rows)
    performance["fold"] = pd.array(performance["fold"], dtype="Int64")
    return prediction_table, performance


def _median_imputed_prediction_matrix(
    frame: pd.DataFrame, feature_cols: Sequence[str]
) -> tuple[pd.DataFrame, pd.Series]:
    matrix = _numeric_feature_matrix(frame, feature_cols)
    medians = matrix.median(axis=0, numeric_only=True)
    invalid = medians.index[~np.isfinite(medians.to_numpy(dtype=float))].tolist()
    if invalid:
        raise ValueError(
            "feature medians are undefined for all-missing columns: "
            + ", ".join(invalid)
        )
    return matrix.fillna(medians), medians


def monte_carlo_shapley_attribution(
    model: Any,
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    n_permutations: int = 64,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Approximate feature Shapley attributions for model predictions.

    The reference point is the featurewise median of ``frame``.  Every random
    ordering starts at that fixed reference, so feature attributions sum to the
    fitted model's prediction difference from the reference prediction.
    """

    features = _validate_feature_columns(frame, feature_cols)
    if frame.empty:
        raise ValueError("frame must contain at least one row")
    if not isinstance(n_permutations, (int, np.integer)) or n_permutations <= 0:
        raise ValueError("n_permutations must be a positive integer")
    x, medians = _median_imputed_prediction_matrix(frame, features)
    values = x.to_numpy(dtype=float)
    baseline = medians.loc[features].to_numpy(dtype=float)
    n_rows, n_features = values.shape
    attribution = np.zeros((n_rows, n_features), dtype=float)
    rng = np.random.default_rng(random_state)

    baseline_frame = pd.DataFrame([baseline], columns=features)
    baseline_prediction = float(np.asarray(model.predict(baseline_frame))[0])
    for _ in range(n_permutations):
        order = rng.permutation(n_features)
        current = np.tile(baseline, (n_rows, 1))
        previous = np.asarray(
            model.predict(pd.DataFrame(current, columns=features)), dtype=float
        )
        for feature_index in order:
            current[:, feature_index] = values[:, feature_index]
            updated = np.asarray(
                model.predict(pd.DataFrame(current, columns=features)), dtype=float
            )
            attribution[:, feature_index] += updated - previous
            previous = updated
    attribution /= n_permutations
    model_prediction = np.asarray(model.predict(x), dtype=float)
    if not (
        np.isfinite(attribution).all()
        and np.isfinite(model_prediction).all()
        and math.isfinite(baseline_prediction)
    ):
        raise ValueError("model returned non-finite predictions")

    return pd.DataFrame(
        {
            "row_position": np.repeat(np.arange(n_rows, dtype=int), n_features),
            "source_index": np.repeat(frame.index.to_numpy(copy=True), n_features),
            "feature": np.tile(features, n_rows),
            "feature_value": values.reshape(-1),
            "baseline_feature_median": np.tile(baseline, n_rows),
            "shapley_model_prediction_attribution": attribution.reshape(-1),
            "baseline_model_prediction": np.repeat(baseline_prediction, n_rows * n_features),
            "model_prediction": np.repeat(model_prediction, n_features),
            "n_permutations": n_permutations,
            "random_state": random_state,
            "method": "deterministic_monte_carlo_feature_shapley",
            "interpretation_scope": "model_prediction_attribution_not_causal_effect",
        }
    )


def accumulated_local_effects(
    model: Any,
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    feature_col: str,
    *,
    n_bins: int = 10,
    min_bin_count: int = 5,
) -> pd.DataFrame:
    """Estimate a first-order accumulated local effect on model predictions."""

    features = _validate_feature_columns(frame, feature_cols)
    if feature_col not in features:
        raise ValueError("feature_col must be included in feature_cols")
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least two")
    if not isinstance(min_bin_count, (int, np.integer)) or min_bin_count <= 0:
        raise ValueError("min_bin_count must be a positive integer")
    if frame.empty:
        raise ValueError("frame must contain at least one row")

    x, _ = _median_imputed_prediction_matrix(frame, features)
    values = x[feature_col].to_numpy(dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        raise ValueError("feature_col must have at least two distinct finite values")
    bin_ids = np.searchsorted(edges[1:-1], values, side="right")

    rows: list[dict[str, Any]] = []
    for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        mask = bin_ids == bin_index - 1
        n_bin = int(mask.sum())
        if n_bin < min_bin_count:
            continue
        low = x.loc[mask].copy()
        high = low.copy()
        low[feature_col] = float(left)
        high[feature_col] = float(right)
        local_differences = np.asarray(model.predict(high), dtype=float) - np.asarray(
            model.predict(low), dtype=float
        )
        if not np.isfinite(local_differences).all():
            raise ValueError("model returned non-finite ALE predictions")
        rows.append(
            {
                "feature": feature_col,
                "bin": bin_index,
                "bin_left": float(left),
                "bin_right": float(right),
                "bin_center": float((left + right) / 2.0),
                "n_bin": n_bin,
                "mean_local_model_prediction_effect": float(local_differences.mean()),
            }
        )
    if not rows:
        raise ValueError("no quantile bin meets min_bin_count")

    result = pd.DataFrame(rows)
    accumulated = result[
        "mean_local_model_prediction_effect"
    ].cumsum().to_numpy(copy=True)
    weights = result["n_bin"].to_numpy(dtype=float)
    accumulated -= float(np.average(accumulated, weights=weights))
    result["ale_model_prediction_effect"] = accumulated
    result["model"] = type(model).__name__
    result["quantile_bins_requested"] = n_bins
    result["min_bin_count"] = min_bin_count
    result["interpretation_scope"] = (
        "accumulated_local_effect_on_model_prediction_not_causal_effect"
    )
    return result


__all__ = [
    "RANDOM_STATE",
    "accumulated_local_effects",
    "grouped_oof_predictions",
    "make_hist_gradient_boosting_regressor",
    "monte_carlo_shapley_attribution",
]
