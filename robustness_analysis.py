"""Robustness and resampling checks for 3DLST.

This module contains same-volume contrasts, sample-filter sensitivity checks,
and within-city spatial block bootstrap routines.
"""

from __future__ import annotations

from lst_common import *  # noqa: F403


def ols_beta(y: np.ndarray, x_raw: np.ndarray) -> tuple[bool, np.ndarray, float]:
    """Fast OLS coefficient helper for bootstrap resamples."""

    y = np.asarray(y, dtype=float)
    x_raw = np.asarray(x_raw, dtype=float)
    if x_raw.ndim == 1:
        x_raw = x_raw[:, None]
    mask = np.isfinite(y) & np.all(np.isfinite(x_raw), axis=1)
    y = y[mask]
    x_raw = x_raw[mask]
    if len(y) <= x_raw.shape[1] + 1:
        return False, np.full(x_raw.shape[1], np.nan), math.nan
    x = np.column_stack([np.ones(len(y)), x_raw])
    if np.linalg.matrix_rank(x) < x.shape[1]:
        return False, np.full(x_raw.shape[1], np.nan), math.nan
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan
    return True, beta[1:], r2


def residualize_fast(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Fast residualization without covariance computation."""

    y = np.asarray(values, dtype=float)
    controls = np.asarray(controls, dtype=float)
    out = np.full(len(y), np.nan, dtype=float)
    if controls.size == 0 or controls.shape[1] == 0:
        mask = np.isfinite(y)
        if mask.sum() >= 2:
            out[mask] = y[mask] - float(np.nanmean(y[mask]))
        return out
    mask = np.isfinite(y) & np.all(np.isfinite(controls), axis=1)
    if mask.sum() <= controls.shape[1] + 1:
        return out
    x = np.column_stack([np.ones(int(mask.sum())), controls[mask]])
    if np.linalg.matrix_rank(x) < x.shape[1]:
        return out
    beta = np.linalg.lstsq(x, y[mask], rcond=None)[0]
    out[mask] = y[mask] - x @ beta
    return out


def contribution_decomposition_fast(
    ln_f: np.ndarray,
    ln_h: np.ndarray,
    ln_v: np.ndarray,
    beta_f: float,
    beta_h: float,
    controls: np.ndarray,
) -> dict[str, float]:
    """Fast contribution accounting for bootstrap resamples."""

    f_resid = residualize_fast(ln_f, controls)
    h_resid = residualize_fast(ln_h, controls)
    v_resid = residualize_fast(ln_v, controls)
    mask = np.isfinite(f_resid) & np.isfinite(h_resid) & np.isfinite(v_resid)
    if int(mask.sum()) < 3:
        return {"C_F_10pct": math.nan, "C_H_10pct": math.nan}
    f = f_resid[mask]
    h = h_resid[mask]
    v = v_resid[mask]
    var_v = float(np.var(v, ddof=0))
    if var_v <= 1e-12:
        return {"C_F_10pct": math.nan, "C_H_10pct": math.nan}
    cov_fv = float(np.mean((f - f.mean()) * (v - v.mean())))
    cov_hv = float(np.mean((h - h.mean()) * (v - v.mean())))
    return {
        "C_F_10pct": float(beta_f * cov_fv / var_v * LOG_10PCT),
        "C_H_10pct": float(beta_h * cov_hv / var_v * LOG_10PCT),
    }


def pathway_record(uid: Any, part: pd.DataFrame, tier: ModelTier, min_rows: int = 100) -> dict[str, Any]:
    """Fit the primary pathway model for a city under a sensitivity setting."""

    rec: dict[str, Any] = {UID: uid, "n_rows": int(len(part)), "estimable": False, "stable_city": False}
    if len(part) < max(MIN_ESTIMABLE_ROWS, min_rows):
        return rec
    y = part[RESPONSE].to_numpy(float)
    controls, names, dropped = build_control_design(part, tier)
    x = np.column_stack([part["lnF"].to_numpy(float), part["lnH"].to_numpy(float), controls]) if controls.size else part[
        ["lnF", "lnH"]
    ].to_numpy(float)
    fit = fit_ols_hc3(y, x)
    rec.update({"control_names": ";".join(names), "dropped_controls": ";".join(dropped), "estimable": bool(fit["estimable"])})
    if fit["estimable"]:
        beta_f = float(fit["beta"][0])
        beta_h = float(fit["beta"][1])
        rec.update(
            {
                "stable_city": bool(fit["n"] >= min_rows),
                "beta_lnF": beta_f,
                "beta_lnH": beta_h,
                "BVR_F_10pct": beta_f * LOG_10PCT,
                "BVR_H_10pct": beta_h * LOG_10PCT,
                "Delta_F_minus_H_10pct": (beta_f - beta_h) * LOG_10PCT,
                "r2": fit["r2"],
            }
        )
    return rec


def bootstrap_core_metrics(uid: Any, part: pd.DataFrame, tier: ModelTier, min_rows: int = 50) -> dict[str, Any]:
    """Estimate core metrics on one bootstrap resample."""

    rec: dict[str, Any] = {UID: uid, "estimable": False}
    if len(part) < max(MIN_ESTIMABLE_ROWS, min_rows):
        return rec
    controls, _, _ = build_control_design(part, tier)
    y = part[RESPONSE].to_numpy(float)
    x_path = np.column_stack([part["lnF"].to_numpy(float), part["lnH"].to_numpy(float), controls]) if controls.size else part[
        ["lnF", "lnH"]
    ].to_numpy(float)
    ok_path, beta_path, _ = ols_beta(y, x_path)
    x_total = np.column_stack([part["lnV"].to_numpy(float), controls]) if controls.size else part[["lnV"]].to_numpy(float)
    ok_total, beta_total, _ = ols_beta(y, x_total)
    if not (ok_path and ok_total):
        return rec
    beta_f = float(beta_path[0])
    beta_h = float(beta_path[1])
    beta_v = float(beta_total[0])
    decomp = contribution_decomposition_fast(
        part["lnF"].to_numpy(float),
        part["lnH"].to_numpy(float),
        part["lnV"].to_numpy(float),
        beta_f,
        beta_h,
        controls=controls,
    )
    rec.update(
        {
            "estimable": True,
            "BVR_10pct": beta_v * LOG_10PCT,
            "BVR_F_10pct": beta_f * LOG_10PCT,
            "BVR_H_10pct": beta_h * LOG_10PCT,
            "Delta_F_minus_H_10pct": (beta_f - beta_h) * LOG_10PCT,
            "C_F_10pct": decomp.get("C_F_10pct", math.nan),
            "C_H_10pct": decomp.get("C_H_10pct", math.nan),
        }
    )
    return rec


def quantile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    """Assign quantile bins while preserving the original index."""

    vals = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=vals.index, dtype="Int64")
    valid = vals.dropna()
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if valid.nunique() < 2:
        return out
    ranks = valid.rank(method="first")
    bins = pd.qcut(ranks, q=min(n_bins, int(valid.nunique())), labels=False, duplicates="drop")
    out.loc[valid.index] = pd.Series(bins, index=valid.index).astype("Int64")
    return out


def same_volume_matched_contrast(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare high-horizontal and high-vertical cells within matched volume bins."""

    rows: list[dict[str, Any]] = []
    tier = next(t for t in MODEL_TIERS if t.tier_id == PRIMARY_TIER_ID)
    eligible = eligible_rows(df)
    eligible["allocation_A"] = eligible["lnF"] - eligible["lnH"]
    for uid, part0 in eligible.groupby(UID, sort=False, observed=True):
        part = part0.copy()
        if len(part) < MIN_STABLE_ROWS or part["lnV"].nunique() < 10:
            continue
        part["volume_bin"] = quantile_bins(part["lnV"], 20)
        keep_rows = []
        raw_deltas = []
        for _, bin_part in part.groupby("volume_bin", observed=True):
            if len(bin_part) < 10 or bin_part["allocation_A"].nunique() < 3:
                continue
            lo = bin_part["allocation_A"].quantile(0.30)
            hi = bin_part["allocation_A"].quantile(0.70)
            low = bin_part.loc[bin_part["allocation_A"] <= lo].copy()
            high = bin_part.loc[bin_part["allocation_A"] >= hi].copy()
            if len(low) < 3 or len(high) < 3:
                continue
            low["high_horizontal"] = 0.0
            high["high_horizontal"] = 1.0
            keep_rows.append(pd.concat([low, high], ignore_index=False))
            raw_deltas.append(float(high[RESPONSE].mean() - low[RESPONSE].mean()))
        if not keep_rows:
            continue
        matched = pd.concat(keep_rows, ignore_index=False)
        if len(matched) < 50:
            continue
        controls, names, _ = build_control_design(matched, tier)
        bin_dummies = pd.get_dummies(matched["volume_bin"].astype("Int64").astype(str), prefix="volume_bin", drop_first=True, dtype=float)
        arrays = [matched["high_horizontal"].to_numpy(float)]
        names2 = ["high_horizontal_same_volume"]
        if controls.size:
            for j, name in enumerate(names):
                arrays.append(controls[:, j])
                names2.append(name)
        for col in bin_dummies.columns:
            arrays.append(bin_dummies[col].to_numpy(float))
            names2.append(col)
        fit = fit_ols_hc3(matched[RESPONSE].to_numpy(float), np.column_stack(arrays))
        delta = float(fit["beta"][0]) if fit["estimable"] else math.nan
        se = float(fit["se_hc3"][0]) if fit["estimable"] else math.nan
        rows.append(
            {
                UID: uid,
                "n_matched_rows": int(len(matched)),
                "n_volume_bins_used": int(matched["volume_bin"].nunique()),
                "raw_bin_mean_delta": float(np.nanmean(raw_deltas)) if raw_deltas else math.nan,
                "delta_high_horizontal_vs_vertical_adjusted": delta,
                "se_hc3": se,
                "ci_low": delta - 1.96 * se if math.isfinite(delta) and math.isfinite(se) else math.nan,
                "ci_high": delta + 1.96 * se if math.isfinite(delta) and math.isfinite(se) else math.nan,
                "estimable": bool(fit["estimable"]),
                "r2": fit["r2"],
                "dropped_terms": ";".join([name for name in names2 if name.startswith("missing:")]),
            }
        )
    out = pd.DataFrame(rows)
    summary = {
        "n_city": int(len(out)),
        "median_delta_adjusted": float(out["delta_high_horizontal_vs_vertical_adjusted"].median()) if len(out) else math.nan,
        "share_delta_positive": float((out["delta_high_horizontal_vs_vertical_adjusted"] > 0).mean()) if len(out) else math.nan,
        "median_raw_delta": float(out["raw_bin_mean_delta"].median()) if len(out) else math.nan,
    }
    return out, summary


def sensitivity_configs(df: pd.DataFrame) -> pd.DataFrame:
    """Run sample-filter sensitivity checks for the primary pathway contrast."""

    rows: list[dict[str, Any]] = []
    configs = [
        {"config": "baseline_bf001_min100", "bf": 0.01, "min_rows": 100, "water_max": None, "winsor": False},
        {"config": "bf0005_min100", "bf": 0.005, "min_rows": 100, "water_max": None, "winsor": False},
        {"config": "bf002_min100", "bf": 0.02, "min_rows": 100, "water_max": None, "winsor": False},
        {"config": "bf005_min100", "bf": 0.05, "min_rows": 100, "water_max": None, "winsor": False},
        {"config": "bf001_min50", "bf": 0.01, "min_rows": 50, "water_max": None, "winsor": False},
        {"config": "bf001_min200", "bf": 0.01, "min_rows": 200, "water_max": None, "winsor": False},
        {"config": "bf001_water_le_020", "bf": 0.01, "min_rows": 100, "water_max": 0.20, "winsor": False},
        {"config": "bf001_water_le_010", "bf": 0.01, "min_rows": 100, "water_max": 0.10, "winsor": False},
        {"config": "bf001_lst_winsor_1_99", "bf": 0.01, "min_rows": 100, "water_max": None, "winsor": True},
    ]
    tier = next(t for t in MODEL_TIERS if t.tier_id == PRIMARY_TIER_ID)
    for cfg in configs:
        part = df.loc[(df["BF"] >= cfg["bf"]) & (df["MBH_m"] > 0) & np.isfinite(df[RESPONSE])].copy()
        if cfg["water_max"] is not None and "p_water_1km" in part.columns:
            part = part.loc[part["p_water_1km"] <= cfg["water_max"]]
        part["lnF"] = np.log(part["BF"])
        part["lnH"] = np.log(part["MBH_m"])
        part["lnV"] = part["lnF"] + part["lnH"]
        if cfg["winsor"]:
            qlo = part.groupby(UID, observed=True)[RESPONSE].transform(lambda s: s.quantile(0.01))
            qhi = part.groupby(UID, observed=True)[RESPONSE].transform(lambda s: s.quantile(0.99))
            part[RESPONSE] = part[RESPONSE].clip(qlo, qhi)
        city_rows = [pathway_record(uid, city, tier, min_rows=cfg["min_rows"]) for uid, city in part.groupby(UID, sort=False, observed=True)]
        city = pd.DataFrame(city_rows)
        stable = city.loc[city["stable_city"].astype(bool)] if len(city) else pd.DataFrame()
        rows.append(
            {
                **cfg,
                "n_rows": int(len(part)),
                "n_city_estimable": int(city["estimable"].sum()) if len(city) else 0,
                "n_city_stable": int(len(stable)),
                "Delta_10pct_median": float(stable["Delta_F_minus_H_10pct"].median()) if len(stable) else math.nan,
                "Delta_10pct_q25": float(stable["Delta_F_minus_H_10pct"].quantile(0.25)) if len(stable) else math.nan,
                "Delta_10pct_q75": float(stable["Delta_F_minus_H_10pct"].quantile(0.75)) if len(stable) else math.nan,
                "share_Delta_10pct_positive": float((stable["Delta_F_minus_H_10pct"] > NEAR_ZERO_C_10PCT).mean()) if len(stable) else math.nan,
                "BVR_F_10pct_median": float(stable["BVR_F_10pct"].median()) if len(stable) else math.nan,
                "BVR_H_10pct_median": float(stable["BVR_H_10pct"].median()) if len(stable) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def make_spatial_block_id(df: pd.DataFrame, block_size: int = DEFAULT_BOOT_BLOCK_SIZE) -> pd.Series:
    """Create within-city square block identifiers from grid coordinates."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    col = np.floor(pd.to_numeric(df["grid_col"], errors="coerce") / block_size).astype("Int64")
    row = np.floor(pd.to_numeric(df["grid_row"], errors="coerce") / block_size).astype("Int64")
    return df[UID].astype(str) + "_" + col.astype(str) + "_" + row.astype(str)


def spatial_block_bootstrap(
    df: pd.DataFrame,
    primary_pathway: pd.DataFrame,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    block_size: int = DEFAULT_BOOT_BLOCK_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resample within-city spatial blocks and refit core metrics."""

    rng = np.random.default_rng(random_state)
    tier = next(t for t in MODEL_TIERS if t.tier_id == PRIMARY_TIER_ID)
    eligible = eligible_rows(df)
    eligible["block_id"] = make_spatial_block_id(eligible, block_size=block_size)
    stable_uids = set(
        primary_pathway.loc[
            (primary_pathway["model_tier"] == PRIMARY_TIER_ID) & primary_pathway["stable_city"].astype(bool), UID
        ]
    )
    rows: list[dict[str, Any]] = []
    for uid, part in eligible.groupby(UID, sort=False, observed=True):
        if uid not in stable_uids:
            continue
        block_groups = [idx.to_numpy() for _, idx in part.groupby("block_id", observed=True).groups.items()]
        if len(block_groups) < 5:
            continue
        metric_values = {metric: [] for metric in BOOTSTRAP_METRICS}
        for _ in range(n_bootstrap):
            chosen = rng.integers(0, len(block_groups), size=len(block_groups))
            boot_idx = np.concatenate([block_groups[i] for i in chosen])
            boot = part.loc[boot_idx]
            rec = bootstrap_core_metrics(uid, boot, tier, min_rows=50)
            if rec.get("estimable"):
                for metric in BOOTSTRAP_METRICS:
                    val = rec.get(metric, math.nan)
                    if math.isfinite(val):
                        metric_values[metric].append(float(val))
        if not metric_values["Delta_F_minus_H_10pct"]:
            continue
        row: dict[str, Any] = {
            UID: uid,
            "n_blocks": int(len(block_groups)),
            "n_bootstrap_success": int(len(metric_values["Delta_F_minus_H_10pct"])),
            "bootstrap_replicates_requested": int(n_bootstrap),
            "block_size_grid_cells": int(block_size),
        }
        for metric, vals in metric_values.items():
            if not vals:
                continue
            arr = np.asarray(vals)
            row[f"{metric}_bootstrap_median"] = float(np.median(arr))
            row[f"{metric}_bootstrap_ci_low"] = float(np.quantile(arr, 0.025))
            row[f"{metric}_bootstrap_ci_high"] = float(np.quantile(arr, 0.975))
            row[f"P_{metric}_positive"] = float((arr > 0).mean())
            row[f"P_{metric}_gt_near_zero"] = float((arr > NEAR_ZERO_C_10PCT).mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "n_city": int(len(out)),
        "bootstrap_replicates_requested": int(n_bootstrap),
        "block_size_grid_cells": int(block_size),
    }
    if len(out):
        summary.update(
            {
                "median_P_Delta_positive": float(out["P_Delta_F_minus_H_10pct_positive"].median()),
                "share_P_Delta_positive_gt_095": float((out["P_Delta_F_minus_H_10pct_positive"] > 0.95).mean()),
                "share_ci_excludes_zero_positive": float((out["Delta_F_minus_H_10pct_bootstrap_ci_low"] > 0).mean()),
            }
        )
    return out, summary


def write_robustness_outputs(
    out_dir: Path,
    same_volume: pd.DataFrame,
    same_volume_summary: dict[str, Any],
    sensitivity: pd.DataFrame,
    bootstrap: pd.DataFrame,
    bootstrap_summary: dict[str, Any],
) -> None:
    """Write robustness diagnostic outputs."""

    robust_dir = out_dir / "robustness"
    write_csv(same_volume, robust_dir / "same_volume_matched_contrast.csv")
    write_json(robust_dir / "same_volume_matched_contrast_summary.json", same_volume_summary)
    write_csv(sensitivity, robust_dir / "sensitivity_summary.csv")
    write_csv(bootstrap, robust_dir / "spatial_block_bootstrap.csv")
    write_json(robust_dir / "spatial_block_bootstrap_summary.json", bootstrap_summary)
