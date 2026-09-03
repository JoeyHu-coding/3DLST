from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lst_common import FINAL_CITY_DRIVER_FEATURES


def _city_frame() -> pd.DataFrame:
    rng = np.random.default_rng(41)
    n_rows = 60
    components = np.repeat([f"OC_{index}" for index in range(15)], 4)
    folds_by_component = np.repeat(np.arange(5), 3)
    folds = np.repeat(folds_by_component, 4)
    frame = pd.DataFrame(
        {
            "UID": np.arange(1, n_rows + 1),
            "city_overlap_component": components,
            "outer_fold": folds,
            "global_region_group": np.where(
                np.arange(n_rows) % 2, "global_north", "global_south"
            ),
        }
    )
    for feature_index, feature in enumerate(FINAL_CITY_DRIVER_FEATURES, start=1):
        frame[feature] = rng.normal(loc=feature_index / 10, scale=0.5, size=n_rows)
    frame["BVR_F_10"] = (
        0.03 * frame["mean_MVH_m"]
        + 0.02 * frame["mean_BF"]
        + rng.normal(0, 0.01, n_rows)
    )
    frame["BVR_H_10"] = (
        0.02 * frame["mean_MBH_m"]
        - 0.01 * frame["mean_p_bare_1km"]
        + rng.normal(0, 0.01, n_rows)
    )
    contrast = frame["BVR_F_10"] - frame["BVR_H_10"]
    frame["allocation_contrast_ci_low"] = contrast - 0.002
    frame["allocation_contrast_ci_high"] = contrast + 0.002

    lower = frame[["BVR_F_10", "BVR_H_10"]].min(axis=1)
    frame["risk_first_type"] = "Type IV"
    frame.loc[lower.gt(0.02), "risk_first_type"] = "Type III"
    remaining = ~lower.gt(0.02)
    frame.loc[
        remaining
        & contrast.gt(0.01)
        & frame["allocation_contrast_ci_low"].gt(0),
        "risk_first_type",
    ] = "Type I"
    frame.loc[
        remaining
        & contrast.lt(-0.01)
        & frame["allocation_contrast_ci_high"].lt(0),
        "risk_first_type",
    ] = "Type II"
    return frame


def test_city_runner_writes_core_nonlinear_and_typology_outputs(tmp_path: Path) -> None:
    import run_city_analysis

    city_path = tmp_path / "city_analysis_core.csv"
    output_dir = tmp_path / "results"
    _city_frame().to_csv(city_path, index=False)

    run_city_analysis.main(
        [
            "--city-table",
            str(city_path),
            "--output-dir",
            str(output_dir),
            "--shapley-permutations",
            "2",
            "--typology-bootstrap",
            "0",
            "--ale-bins",
            "5",
            "--min-ale-bin-count",
            "5",
        ]
    )

    expected = {
        "nonlinear/oof_predictions.csv",
        "nonlinear/oof_performance.csv",
        "nonlinear/shapley_values.csv",
        "nonlinear/shapley_summary.csv",
        "nonlinear/ale_curves.csv",
        "typology/composition.csv",
        "typology/association.csv",
        "typology/evenness.csv",
        "run_manifest.json",
    }
    written = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    assert written == expected

    performance = pd.read_csv(output_dir / "nonlinear" / "oof_performance.csv")
    pooled = performance.loc[performance["evaluation_scope"].eq("pooled_oof")]
    assert set(pooled["target"]) == {"BVR_F_10", "BVR_H_10"}
    assert pooled["n_observations"].eq(60).all()
    assert performance["splitter"].eq("preassigned_grouped_folds").all()

    composition = pd.read_csv(output_dir / "typology" / "composition.csv")
    assert composition["n_city"].sum() == 60
    assert len(composition) == 8

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_city"] == 60
    assert manifest["targets"] == ["BVR_F_10", "BVR_H_10"]
    assert manifest["features"] == list(FINAL_CITY_DRIVER_FEATURES)
    assert manifest["shapley_permutations"] == 2
