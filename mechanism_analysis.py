"""Surface-context and moderator diagnostics for 3DLST.

These routines test how the footprint and height pathways relate to local
surface context and broader city characteristics.
"""

from __future__ import annotations

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
        raw_arrays: list[np.ndarray] = [
            demean_series(part, "lnF").to_numpy(float),
            demean_series(part, "lnH").to_numpy(float),
            z_mod.to_numpy(float),
            (part["lnF"].to_numpy(float) * z_mod.to_numpy(float)),
            (part["lnH"].to_numpy(float) * z_mod.to_numpy(float)),
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
                [UID, "C_F_10pct", "C_H_10pct", "source_class"],
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
