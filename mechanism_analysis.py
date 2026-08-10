"""Surface-context and moderator diagnostics for 3DLST.

These routines test how the footprint and height pathways relate to local
surface context and broader city characteristics.
"""

from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from lst_common import *  # noqa: F403


def sequential_adjustment(total: pd.DataFrame, pathway: pd.DataFrame, decomp: pd.DataFrame) -> pd.DataFrame:
    """Compare response metrics across nested adjustment tiers."""

    pieces: list[dict[str, Any]] = []
    for metric, frame, value in [
        ("BVR_10pct", total, "BVR_10pct"),
        ("Delta_F_minus_H_10pct", pathway, "Delta_F_minus_H_10pct"),
        ("C_F_10pct", decomp, "C_F_10pct"),
        ("C_H_10pct", decomp, "C_H_10pct"),
    ]:
        wide = frame.pivot_table(index=UID, columns="model_tier", values=value, aggfunc="first")
        for base_tier, target_tier, label in [
            ("M0_morphology", "M1_terrain_water", "terrain_water_adjustment"),
            ("M1_terrain_water", "M2_vegetation", "vegetation_adjustment"),
            ("M1_terrain_water", "M3_landcover", "landcover_adjustment"),
            ("M1_terrain_water", "M4_spatial", "spatial_adjustment"),
            ("M1_terrain_water", "M5_full", "full_adjustment"),
        ]:
            if base_tier not in wide.columns or target_tier not in wide.columns:
                continue
            for uid, row in wide[[base_tier, target_tier]].dropna(how="all").iterrows():
                pieces.append(
                    {
                        UID: uid,
                        "metric": metric,
                        "comparison": label,
                        "base_tier": base_tier,
                        "target_tier": target_tier,
                        "base_value": row.get(base_tier, math.nan),
                        "target_value": row.get(target_tier, math.nan),
                        **attenuation_diagnostics(float(row.get(base_tier, math.nan)), float(row.get(target_tier, math.nan))),
                    }
                )
    return pd.DataFrame(pieces)


def displacement_base_tier(outcome: str) -> ModelTier:
    """Use a terrain-only base when water cover is the surface-context outcome."""

    if outcome == "p_water_1km":
        return ModelTier("D_terrain_only", "Terrain only", ("terrain_mean_m", "slope_mean_deg", "relief_p90_p10_m"))
    return next(tier for tier in MODEL_TIERS if tier.tier_id == PRIMARY_TIER_ID)


def run_surface_context_models(df: pd.DataFrame, ctx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate how surface-context variables co-vary with footprint and height paths."""

    context_by_uid = ctx.set_index(UID)
    city_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    for outcome, label in SURFACE_CONTEXT_OUTCOMES:
        if outcome not in df.columns:
            continue
        tier = displacement_base_tier(outcome)
        for uid, part0 in df.groupby(UID, sort=False, observed=True):
            part = part0.loc[np.isfinite(pd.to_numeric(part0[outcome], errors="coerce"))].copy()
            if len(part) < MIN_ESTIMABLE_ROWS:
                continue
            y = part[outcome].to_numpy(dtype=np.float64)
            if np.nanmax(y) - np.nanmin(y) <= 1e-12:
                continue
            controls, names, dropped = build_control_design(part, tier)
            x = np.column_stack([part["lnF"].to_numpy(float), part["lnH"].to_numpy(float), controls]) if controls.size else part[
                ["lnF", "lnH"]
            ].to_numpy(float)
            fit = fit_ols_hc3(y, x)
            cctx = context_by_uid.loc[uid].to_dict() if uid in context_by_uid.index else {}
            rec = {
                UID: uid,
                "outcome": outcome,
                "outcome_label": label,
                "control_tier": tier.tier_id,
                "control_names": ";".join(names),
                "dropped_controls": ";".join(dropped),
                "estimable": bool(fit["estimable"]),
                "stable_city": bool(fit["estimable"] and fit["n"] >= MIN_STABLE_ROWS),
                "n_model": int(fit["n"]),
                "r2": fit["r2"],
                "outcome_range": float(np.nanmax(y) - np.nanmin(y)),
                "path_F_1pct": math.nan,
                "path_H_1pct": math.nan,
                "path_F_10pct": math.nan,
                "path_H_10pct": math.nan,
                **cctx,
            }
            if fit["estimable"]:
                rec.update(
                    {
                        "beta_lnF": float(fit["beta"][0]),
                        "beta_lnH": float(fit["beta"][1]),
                        "se_lnF_hc3": float(fit["se_hc3"][0]),
                        "se_lnH_hc3": float(fit["se_hc3"][1]),
                        "path_F_1pct": float(fit["beta"][0] * LOG_1PCT),
                        "path_H_1pct": float(fit["beta"][1] * LOG_1PCT),
                        "path_F_10pct": float(fit["beta"][0] * LOG_10PCT),
                        "path_H_10pct": float(fit["beta"][1] * LOG_10PCT),
                    }
                )
            city_rows.append(rec)

        part = df.loc[np.isfinite(pd.to_numeric(df[outcome], errors="coerce"))].copy()
        if len(part) > 100:
            x, names, dropped, y_fe = pooled_design(part.assign(**{RESPONSE: part[outcome]}), ["lnF", "lnH"], tier)
            fit = fit_ols_cluster(y_fe.to_numpy(float), x, part[UID].to_numpy())
            for idx, term in enumerate(names):
                if term not in {"lnF", "lnH"}:
                    continue
                est = float(fit["beta"][idx])
                se = float(fit["se_cluster"][idx])
                pooled_rows.append(
                    {
                        "outcome": outcome,
                        "outcome_label": label,
                        "term": term,
                        "estimate": est,
                        "cluster_se": se,
                        "per1pct_estimate": est * LOG_1PCT,
                        "per1pct_se": se * LOG_1PCT,
                        "per10pct_estimate": est * LOG_10PCT,
                        "per10pct_se": se * LOG_10PCT,
                        "n_rows": fit["n"],
                        "n_uid": fit["n_cluster"],
                        "r2_within": fit["r2"],
                        "dropped_terms": ";".join(dropped),
                    }
                )
    return pd.DataFrame(city_rows), pd.DataFrame(pooled_rows)


def run_grid_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Fit pooled interactions between pathway terms and grid-level moderators."""

    rows: list[dict[str, Any]] = []
    base_tier = next(tier for tier in MODEL_TIERS if tier.tier_id == PRIMARY_TIER_ID)
    for moderator, label in GRID_MODERATORS:
        if moderator not in df.columns:
            continue
        part = df.loc[np.isfinite(pd.to_numeric(df[moderator], errors="coerce"))].copy()
        if len(part) < 1000 or part[moderator].nunique(dropna=True) < 5:
            continue
        z_mod = within_city_z(part, moderator)
        ln_f_centered = demean_series(part, "lnF")
        ln_h_centered = demean_series(part, "lnH")
        interaction_f = ln_f_centered * z_mod
        interaction_h = ln_h_centered * z_mod
        interaction_f = interaction_f - interaction_f.groupby(part[UID], observed=True).transform("mean")
        interaction_h = interaction_h - interaction_h.groupby(part[UID], observed=True).transform("mean")
        raw_arrays: list[np.ndarray] = [
            ln_f_centered.to_numpy(float),
            ln_h_centered.to_numpy(float),
            z_mod.to_numpy(float),
            interaction_f.to_numpy(float),
            interaction_h.to_numpy(float),
        ]
        raw_names = ["lnF", "lnH", f"z_{moderator}", f"lnF_x_z_{moderator}", f"lnH_x_z_{moderator}"]
        for control in base_tier.controls:
            if control in part.columns:
                raw_arrays.append(within_city_z(part, control).to_numpy(float))
                raw_names.append(f"z_{control}")
        x, names, dropped = reduced_rank_design(raw_arrays, raw_names)
        y = demean_series(part, RESPONSE).to_numpy(float)
        fit = fit_ols_cluster(y, x, part[UID].to_numpy())
        term_map = {name: idx for idx, name in enumerate(names)}
        for term_kind, term_name in [("theta_F", f"lnF_x_z_{moderator}"), ("theta_H", f"lnH_x_z_{moderator}")]:
            idx = term_map.get(term_name)
            est = float(fit["beta"][idx]) if idx is not None else math.nan
            se = float(fit["se_cluster"][idx]) if idx is not None else math.nan
            lo, hi = ci95(est, se)
            rows.append(
                {
                    "moderator": moderator,
                    "moderator_label": label,
                    "term": term_kind,
                    "estimate": est,
                    "cluster_se": se,
                    "ci_low": lo,
                    "ci_high": hi,
                    "per1pct_estimate": est * LOG_1PCT if math.isfinite(est) else math.nan,
                    "per1pct_se": se * LOG_1PCT if math.isfinite(se) else math.nan,
                    "per10pct_estimate": est * LOG_10PCT if math.isfinite(est) else math.nan,
                    "per10pct_se": se * LOG_10PCT if math.isfinite(se) else math.nan,
                    "n_rows": fit["n"],
                    "n_uid": fit["n_cluster"],
                    "r2_within": fit["r2"],
                    "dropped_terms": ";".join(dropped),
                }
            )
    return pd.DataFrame(rows)


def z_city(series: pd.Series) -> np.ndarray | None:
    """Return a city-level z-score vector for moderator models."""

    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(vals)) or vals.std(ddof=0) <= 1e-12:
        return None
    return (vals - vals.mean()) / vals.std(ddof=0)


def run_city_moderators(total: pd.DataFrame, pathway: pd.DataFrame, decomp: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """Fit city-level associations between background variables and response metrics."""

    primary = (
        pathway.loc[
            (pathway["model_tier"] == PRIMARY_TIER_ID) & pathway["stable_city"].astype(bool),
            [UID, "beta_lnF", "beta_lnH", "BVR_F_10pct", "BVR_H_10pct", "Delta_F_minus_H_10pct"],
        ]
        .merge(
            total.loc[
                (total["model_tier"] == PRIMARY_TIER_ID) & total["stable_city"].astype(bool),
                [UID, "BVR_10pct"],
            ],
            on=UID,
            how="inner",
        )
        .merge(
            decomp.loc[
                (decomp["model_tier"] == PRIMARY_TIER_ID) & decomp["stable_city"].astype(bool),
                [UID, "C_F_10pct", "C_H_10pct"],
            ],
            on=UID,
            how="inner",
        )
        .merge(ctx, on=UID, how="left")
    )
    if "GDPpc" in primary.columns:
        primary["log_GDPpc"] = np.where(primary["GDPpc"] > 0, np.log(primary["GDPpc"]), np.nan)
    if "population" in primary.columns:
        primary["log_population"] = np.where(primary["population"] > 0, np.log(primary["population"]), np.nan)

    outcomes = ["beta_lnF", "beta_lnH", "BVR_10pct", "Delta_F_minus_H_10pct", "C_F_10pct", "C_H_10pct"]
    rows: list[dict[str, Any]] = []
    climate_dummies = (
        pd.get_dummies(primary["climate_macro"], prefix="climate", drop_first=True, dtype=float)
        if "climate_macro" in primary.columns
        else pd.DataFrame(index=primary.index)
    )
    for outcome in outcomes:
        for moderator in FINAL_CITY_DRIVER_FEATURES:
            if moderator not in primary.columns:
                continue
            cols = [outcome, moderator, *[c for c in FINAL_CITY_DRIVER_FEATURES if c != moderator and c in primary.columns]]
            work = pd.concat([primary[[c for c in cols if c in primary.columns]], climate_dummies], axis=1).dropna()
            if len(work) < 50:
                continue
            arrays: list[np.ndarray] = []
            names: list[str] = []
            z_mod = z_city(work[moderator])
            if z_mod is None:
                continue
            arrays.append(z_mod)
            names.append(moderator)
            for control in [c for c in FINAL_CITY_DRIVER_FEATURES if c != moderator and c in work.columns]:
                z_control = z_city(work[control])
                if z_control is not None:
                    arrays.append(z_control)
                    names.append(control)
            for dummy in climate_dummies.columns:
                if dummy in work.columns:
                    arrays.append(work[dummy].to_numpy(float))
                    names.append(dummy)
            x, names, dropped = reduced_rank_design(arrays, names)
            fit = fit_ols_hc3(work[outcome].to_numpy(float), x)
            if moderator in names and fit["estimable"]:
                idx = names.index(moderator)
                est = float(fit["beta"][idx])
                se = float(fit["se_hc3"][idx])
                lo, hi = ci95(est, se)
                rows.append(
                    {
                        "outcome": outcome,
                        "moderator": moderator,
                        "estimate_per_1sd": est,
                        "se_hc3": se,
                        "ci_low": lo,
                        "ci_high": hi,
                        "p_norm": normal_pvalue(est / se) if se > 0 else math.nan,
                        "n_city": fit["n"],
                        "r2": fit["r2"],
                        "dropped_terms": ";".join(dropped),
                    }
                )
    return pd.DataFrame(rows)


def nested_leave_climate_out(
    frame: pd.DataFrame,
    features: Iterable[str],
    target: str,
    climate_col: str = "climate_group",
    alphas: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> dict[str, pd.DataFrame]:
    """Evaluate Ridge models with nested leave-one-climate-group-out folds."""

    feature_names = list(features)
    if not feature_names:
        raise ValueError("features must contain at least one column")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("features must be unique")
    alpha_values = tuple(float(alpha) for alpha in alphas)
    if not alpha_values or len(set(alpha_values)) != len(alpha_values):
        raise ValueError("alphas must contain unique values")
    if any(not math.isfinite(alpha) or alpha < 0.0 for alpha in alpha_values):
        raise ValueError("alphas must be finite and non-negative")

    required = [climate_col, *feature_names, target]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Nested climate validation is missing required columns: {missing}")

    work = frame[required].copy()
    work["_row_position"] = np.arange(len(frame), dtype=int)
    work["_row_index"] = frame.index.to_numpy()
    for column in [*feature_names, target]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    finite = np.all(np.isfinite(work[[*feature_names, target]].to_numpy(float)), axis=1)
    work = work.loc[finite & work[climate_col].notna()].copy()
    climate_groups = list(pd.unique(work[climate_col]))
    if len(climate_groups) < 3:
        raise ValueError("nested leave-climate-out validation requires at least three climate groups")

    prediction_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    def scaler_audit(scaler: StandardScaler) -> dict[str, float]:
        audit: dict[str, float] = {}
        for index, feature in enumerate(feature_names):
            audit[f"scaler_mean_{feature}"] = float(scaler.mean_[index])
            audit[f"scaler_scale_{feature}"] = float(scaler.scale_[index])
        return audit

    for held_out in climate_groups:
        outer_test = work.loc[work[climate_col] == held_out]
        outer_train = work.loc[work[climate_col] != held_out]
        inner_groups = list(pd.unique(outer_train[climate_col]))
        squared_errors: dict[float, list[float]] = {alpha: [] for alpha in alpha_values}

        for validation_group in inner_groups:
            inner_validation = outer_train.loc[outer_train[climate_col] == validation_group]
            inner_train = outer_train.loc[outer_train[climate_col] != validation_group]
            scaler = StandardScaler()
            x_train = scaler.fit_transform(inner_train[feature_names].to_numpy(float))
            x_validation = scaler.transform(inner_validation[feature_names].to_numpy(float))
            y_train = inner_train[target].to_numpy(float)
            y_validation = inner_validation[target].to_numpy(float)
            train_groups = tuple(pd.unique(inner_train[climate_col]))
            audit_rows.append(
                {
                    "stage": "inner",
                    "held_out_climate": held_out,
                    "validation_climate": validation_group,
                    "train_climate_groups": train_groups,
                    "n_train": int(len(inner_train)),
                    "n_validation": int(len(inner_validation)),
                    **scaler_audit(scaler),
                }
            )
            for alpha in alpha_values:
                model = Ridge(alpha=alpha)
                model.fit(x_train, y_train)
                predicted = model.predict(x_validation)
                squared_errors[alpha].extend(np.square(y_validation - predicted).tolist())

        inner_mse = {alpha: float(np.mean(squared_errors[alpha])) for alpha in alpha_values}
        selected_alpha = min(alpha_values, key=lambda alpha: (inner_mse[alpha], alpha))
        for alpha in alpha_values:
            tuning_rows.append(
                {
                    "held_out_climate": held_out,
                    "alpha": alpha,
                    "inner_mse": inner_mse[alpha],
                    "inner_rmse": float(math.sqrt(inner_mse[alpha])),
                    "selected": bool(alpha == selected_alpha),
                }
            )

        outer_scaler = StandardScaler()
        x_outer_train = outer_scaler.fit_transform(outer_train[feature_names].to_numpy(float))
        x_outer_test = outer_scaler.transform(outer_test[feature_names].to_numpy(float))
        outer_model = Ridge(alpha=selected_alpha)
        outer_model.fit(x_outer_train, outer_train[target].to_numpy(float))
        outer_predictions = outer_model.predict(x_outer_test)
        audit_rows.append(
            {
                "stage": "outer",
                "held_out_climate": held_out,
                "validation_climate": None,
                "train_climate_groups": tuple(pd.unique(outer_train[climate_col])),
                "n_train": int(len(outer_train)),
                "n_validation": int(len(outer_test)),
                **scaler_audit(outer_scaler),
            }
        )
        for (_, row), predicted in zip(outer_test.iterrows(), outer_predictions):
            prediction_rows.append(
                {
                    "row_position": int(row["_row_position"]),
                    "row_index": row["_row_index"],
                    climate_col: row[climate_col],
                    "held_out_climate": held_out,
                    "observed": float(row[target]),
                    "predicted": float(predicted),
                    "selected_alpha": selected_alpha,
                }
            )

    predictions = pd.DataFrame(prediction_rows).sort_values("row_position").reset_index(drop=True)
    return {
        "predictions": predictions,
        "tuning": pd.DataFrame(tuning_rows),
        "fold_audit": pd.DataFrame(audit_rows),
    }


def write_mechanism_outputs(
    out_dir: Path,
    seq: pd.DataFrame,
    surface_city: pd.DataFrame,
    surface_pooled: pd.DataFrame,
    grid: pd.DataFrame,
    city_mod: pd.DataFrame,
) -> None:
    """Write surface-context and moderator diagnostic tables."""

    mech_dir = out_dir / "mechanisms"
    write_csv(seq, mech_dir / "sequential_adjustment.csv")
    if len(seq):
        summary = (
            seq.groupby(["metric", "comparison"], dropna=False, observed=True)
            .agg(
                n_city=("attenuation_ratio", "count"),
                median_attenuation=("attenuation_ratio", "median"),
                mean_attenuation=("attenuation_ratio", "mean"),
                share_sign_change=("sign_change", "mean"),
                share_denominator_near_zero=("denominator_near_zero", "mean"),
            )
            .reset_index()
        )
        write_csv(summary, mech_dir / "sequential_adjustment_summary.csv")
    write_csv(surface_city, mech_dir / "surface_context_path_models.csv")
    write_csv(surface_pooled, mech_dir / "surface_context_pooled_fixed_effects.csv")
    write_csv(grid, mech_dir / "grid_interactions.csv")
    write_csv(city_mod, mech_dir / "city_moderators.csv")
