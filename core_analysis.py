"""Core city-level response models for 3DLST.

This module estimates the total building-volume response, the footprint and
height pathway responses, contribution accounting, and pooled fixed-effect
diagnostics.
"""

from __future__ import annotations

from lst_common import *  # noqa: F403


def fit_core_models(df: pd.DataFrame, ctx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit city-level total, pathway, and contribution models."""

    context_by_uid = ctx.set_index(UID)
    total_rows: list[dict[str, Any]] = []
    pathway_rows: list[dict[str, Any]] = []
    decomp_rows: list[dict[str, Any]] = []

    for uid, part in df.groupby(UID, sort=False, observed=True):
        y = part[RESPONSE].to_numpy(dtype=np.float64)
        ln_f = part["lnF"].to_numpy(dtype=np.float64)
        ln_h = part["lnH"].to_numpy(dtype=np.float64)
        ln_v = part["lnV"].to_numpy(dtype=np.float64)
        cctx = context_by_uid.loc[uid].to_dict() if uid in context_by_uid.index else {}
        ranges = {
            "lnF_range": float(np.nanmax(ln_f) - np.nanmin(ln_f)),
            "lnH_range": float(np.nanmax(ln_h) - np.nanmin(ln_h)),
            "lnV_range": float(np.nanmax(ln_v) - np.nanmin(ln_v)),
        }
        corr_fh = (
            float(np.corrcoef(ln_f, ln_h)[0, 1])
            if len(part) > 2 and np.std(ln_f) > 0 and np.std(ln_h) > 0
            else math.nan
        )
        base = {
            UID: uid,
            "n_rows_city": int(len(part)),
            "lnF_lnH_corr": corr_fh,
            **ranges,
            **cctx,
        }
        for tier in MODEL_TIERS:
            controls, control_names, dropped_controls = build_control_design(part, tier)

            total_x = np.column_stack([ln_v, controls]) if controls.size else ln_v[:, None]
            total_fit = fit_ols_hc3(y, total_x)
            total_rec = {
                **base,
                "model_tier": tier.tier_id,
                "tier_label": tier.label,
                "control_names": ";".join(control_names),
                "dropped_controls": ";".join(dropped_controls),
                "estimable": bool(total_fit["estimable"]),
                "stable_city": bool(
                    total_fit["estimable"] and total_fit["n"] >= MIN_STABLE_ROWS and ranges["lnV_range"] >= MIN_LOG_RANGE
                ),
                "n_model": int(total_fit["n"]),
                "rank": int(total_fit["rank"]),
                "r2": total_fit["r2"],
                "rmse": total_fit["rmse"],
                "beta_lnV": math.nan,
                "se_lnV_hc3": math.nan,
                "BVR_10pct": math.nan,
                "se_BVR_10pct": math.nan,
                "BVR_10pct_ci_low": math.nan,
                "BVR_10pct_ci_high": math.nan,
                "p_lnV_norm": math.nan,
            }
            if total_fit["estimable"]:
                beta_v = float(total_fit["beta"][0])
                se_v = float(total_fit["se_hc3"][0])
                total_rec.update(
                    {
                        "beta_lnV": beta_v,
                        "se_lnV_hc3": se_v,
                        "BVR_10pct": beta_v * LOG_10PCT,
                        "se_BVR_10pct": se_v * LOG_10PCT,
                    }
                )
                total_rec["BVR_10pct_ci_low"], total_rec["BVR_10pct_ci_high"] = ci95(
                    total_rec["BVR_10pct"], total_rec["se_BVR_10pct"]
                )
                total_rec["p_lnV_norm"] = normal_pvalue(beta_v / se_v) if se_v > 0 else math.nan
            total_rows.append(total_rec)

            path_x = np.column_stack([ln_f, ln_h, controls]) if controls.size else np.column_stack([ln_f, ln_h])
            path_fit = fit_ols_hc3(y, path_x)
            path_rec = {
                **base,
                "model_tier": tier.tier_id,
                "tier_label": tier.label,
                "control_names": ";".join(control_names),
                "dropped_controls": ";".join(dropped_controls),
                "estimable": bool(path_fit["estimable"]),
                "stable_city": bool(
                    path_fit["estimable"]
                    and path_fit["n"] >= MIN_STABLE_ROWS
                    and ranges["lnF_range"] >= MIN_LOG_RANGE
                    and ranges["lnH_range"] >= MIN_LOG_RANGE
                ),
                "high_collinearity_flag": bool(math.isfinite(corr_fh) and abs(corr_fh) >= 0.9),
                "n_model": int(path_fit["n"]),
                "rank": int(path_fit["rank"]),
                "r2": path_fit["r2"],
                "rmse": path_fit["rmse"],
                "beta_lnF": math.nan,
                "beta_lnH": math.nan,
                "se_lnF_hc3": math.nan,
                "se_lnH_hc3": math.nan,
                "BVR_F_10pct": math.nan,
                "BVR_H_10pct": math.nan,
                "Delta_F_minus_H_10pct": math.nan,
                "se_Delta_F_minus_H_10pct": math.nan,
                "Delta_F_minus_H_10pct_ci_low": math.nan,
                "Delta_F_minus_H_10pct_ci_high": math.nan,
            }
            if path_fit["estimable"]:
                beta_f = float(path_fit["beta"][0])
                beta_h = float(path_fit["beta"][1])
                se_f = float(path_fit["se_hc3"][0])
                se_h = float(path_fit["se_hc3"][1])
                cov_fh = float(path_fit["cov_hc3"][0, 1])
                var_delta = max(se_f**2 + se_h**2 - 2.0 * cov_fh, 0.0)
                path_rec.update(
                    {
                        "beta_lnF": beta_f,
                        "beta_lnH": beta_h,
                        "se_lnF_hc3": se_f,
                        "se_lnH_hc3": se_h,
                        "BVR_F_10pct": beta_f * LOG_10PCT,
                        "BVR_H_10pct": beta_h * LOG_10PCT,
                        "Delta_F_minus_H_10pct": (beta_f - beta_h) * LOG_10PCT,
                        "se_Delta_F_minus_H_10pct": math.sqrt(var_delta) * LOG_10PCT,
                    }
                )
                path_rec["Delta_F_minus_H_10pct_ci_low"], path_rec["Delta_F_minus_H_10pct_ci_high"] = ci95(
                    path_rec["Delta_F_minus_H_10pct"], path_rec["se_Delta_F_minus_H_10pct"]
                )
            pathway_rows.append(path_rec)

            decomp_rec = {
                **base,
                "model_tier": tier.tier_id,
                "tier_label": tier.label,
                "BVR_10pct": total_rec["BVR_10pct"],
                "BVR_F_10pct": path_rec["BVR_F_10pct"],
                "BVR_H_10pct": path_rec["BVR_H_10pct"],
                "Delta_F_minus_H_10pct": path_rec["Delta_F_minus_H_10pct"],
                "C_F": math.nan,
                "C_H": math.nan,
                "C_F_10pct": math.nan,
                "C_H_10pct": math.nan,
                "var_lnV_resid": math.nan,
                "BVR_decomp_sum_10pct": math.nan,
                "BVR_decomp_identity_error_10pct": math.nan,
                "source_class": "not_estimable",
                "dominant_share": math.nan,
                "offset_flag": False,
                "estimable": bool(total_fit["estimable"] and path_fit["estimable"]),
                "stable_city": bool(total_rec["stable_city"] and path_rec["stable_city"]),
            }
            if total_fit["estimable"] and path_fit["estimable"]:
                dec = contribution_decomposition(
                    ln_f,
                    ln_h,
                    ln_v,
                    path_rec["beta_lnF"],
                    path_rec["beta_lnH"],
                    controls,
                )
                class_info = classify_source_contribution(total_rec["BVR_10pct"], dec["C_F_10pct"], dec["C_H_10pct"])
                decomp_rec.update(dec)
                decomp_rec.update(class_info)
                decomp_rec["BVR_decomp_sum_10pct"] = decomp_rec["C_F_10pct"] + decomp_rec["C_H_10pct"]
                decomp_rec["BVR_decomp_identity_error_10pct"] = (
                    decomp_rec["BVR_10pct"] - decomp_rec["BVR_decomp_sum_10pct"]
                )
            decomp_rows.append(decomp_rec)

    total = add_per1_columns(pd.DataFrame(total_rows), ["BVR_10pct"])
    pathway = add_per1_columns(
        pd.DataFrame(pathway_rows),
        ["BVR_F_10pct", "BVR_H_10pct", "Delta_F_minus_H_10pct"],
    )
    decomp = add_per1_columns(
        pd.DataFrame(decomp_rows),
        ["BVR_10pct", "BVR_F_10pct", "BVR_H_10pct", "Delta_F_minus_H_10pct", "C_F_10pct", "C_H_10pct"],
    )
    return total, pathway, decomp


def fit_pooled_models(df: pd.DataFrame) -> pd.DataFrame:
    """Fit pooled within-city fixed-effect diagnostics for all model tiers."""

    rows: list[dict[str, Any]] = []
    clusters = df[UID].to_numpy()
    for tier in MODEL_TIERS:
        for model_name, predictors in [("total_volume", ["lnV"]), ("pathway", ["lnF", "lnH"])]:
            x, names, dropped, y = pooled_design(df, predictors, tier)
            fit = fit_ols_cluster(y.to_numpy(dtype=np.float64), x, clusters)
            for idx, name in enumerate(names):
                est = float(fit["beta"][idx]) if idx < len(fit["beta"]) else math.nan
                se = float(fit["se_cluster"][idx]) if idx < len(fit["se_cluster"]) else math.nan
                lo, hi = ci95(est, se)
                rows.append(
                    {
                        "model": model_name,
                        "model_tier": tier.tier_id,
                        "term": name,
                        "estimate": est,
                        "cluster_se": se,
                        "ci_low": lo,
                        "ci_high": hi,
                        "per10pct_estimate": est * LOG_10PCT if name in {"lnV", "lnF", "lnH"} else math.nan,
                        "per10pct_se": se * LOG_10PCT if name in {"lnV", "lnF", "lnH"} else math.nan,
                        "n_rows": fit["n"],
                        "n_uid": fit["n_cluster"],
                        "r2_within": fit["r2"],
                        "dropped_terms": ";".join(dropped),
                    }
                )
            if model_name == "pathway" and "lnF" in names and "lnH" in names:
                i_f = names.index("lnF")
                i_h = names.index("lnH")
                beta = fit["beta"]
                cov = fit["cov_cluster"]
                d_est = float((beta[i_f] - beta[i_h]) * LOG_10PCT)
                d_se = float(math.sqrt(max(cov[i_f, i_f] + cov[i_h, i_h] - 2.0 * cov[i_f, i_h], 0.0)) * LOG_10PCT)
                lo, hi = ci95(d_est, d_se)
                rows.append(
                    {
                        "model": model_name,
                        "model_tier": tier.tier_id,
                        "term": "Delta_F_minus_H_10pct",
                        "estimate": float(beta[i_f] - beta[i_h]),
                        "cluster_se": math.nan,
                        "ci_low": lo,
                        "ci_high": hi,
                        "per10pct_estimate": d_est,
                        "per10pct_se": d_se,
                        "n_rows": fit["n"],
                        "n_uid": fit["n_cluster"],
                        "r2_within": fit["r2"],
                        "dropped_terms": ";".join(dropped),
                    }
                )
    return pd.DataFrame(rows)


def write_core_outputs(
    out_dir: Path,
    total: pd.DataFrame,
    pathway: pd.DataFrame,
    decomp: pd.DataFrame,
    pooled: pd.DataFrame,
) -> pd.DataFrame:
    """Write core result tables and return a compact summary table."""

    core_dir = out_dir / "core"
    write_csv(total, core_dir / "city_total_bvr.csv")
    write_csv(pathway, core_dir / "city_pathway_contrast.csv")
    write_csv(decomp, core_dir / "city_contribution_decomposition.csv")
    write_csv(pooled, core_dir / "pooled_fixed_effects.csv")

    summaries = []
    for frame, metric in [
        (total.loc[total["stable_city"].astype(bool)], "BVR_10pct"),
        (pathway.loc[pathway["stable_city"].astype(bool)], "BVR_F_10pct"),
        (pathway.loc[pathway["stable_city"].astype(bool)], "BVR_H_10pct"),
        (pathway.loc[pathway["stable_city"].astype(bool)], "Delta_F_minus_H_10pct"),
        (decomp.loc[decomp["stable_city"].astype(bool)], "C_F_10pct"),
        (decomp.loc[decomp["stable_city"].astype(bool)], "C_H_10pct"),
    ]:
        if metric in frame.columns:
            summaries.append(summarize_distribution(frame, metric, ["model_tier"]))
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    write_csv(summary, core_dir / "summary_by_model_tier.csv")
    write_json(
        core_dir / "source_classification_rules.json",
        {
            "near_zero_degC_per_10pct": NEAR_ZERO_C_10PCT,
            "dominance_share": DOMINANCE_SHARE,
            "primary_tier": PRIMARY_TIER_ID,
        },
    )
    return summary
