#!/usr/bin/env python3
"""Command-line entry point for the 3DLST reproducible analysis package.

The command starts from an analysis-ready 1 km grid matrix and can run the core
city-level response models, surface-context diagnostics, and robustness checks.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

from core_analysis import fit_core_models, fit_pooled_models, write_core_outputs
from mechanism_analysis import (
    run_city_moderators,
    run_grid_interactions,
    run_surface_context_models,
    sequential_adjustment,
    write_mechanism_outputs,
)
from robustness_analysis import (
    same_volume_matched_contrast,
    sensitivity_configs,
    spatial_block_bootstrap,
    write_robustness_outputs,
)
from lst_common import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOT_BLOCK_SIZE,
    MODEL_TIERS,
    PRIMARY_TIER_ID,
    UID,
    city_context,
    default_matrix_path,
    eligible_rows,
    infer_project_root,
    load_analysis_matrix,
    now_utc,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    project_root = infer_project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=default_matrix_path(project_root),
        help="Path to the analysis-ready grid matrix Parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results" / "3dlst_analysis_results",
        help="Directory where result tables will be written.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=["all", "core", "mechanisms", "robustness"],
        help="Pipeline stages to run.",
    )
    parser.add_argument(
        "--max-cities",
        type=int,
        default=None,
        help="Optional city limit for smoke tests. Do not use for full reproduction.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
        help="Requested within-city spatial block bootstrap replicates.",
    )
    parser.add_argument(
        "--bootstrap-block-size",
        type=int,
        default=DEFAULT_BOOT_BLOCK_SIZE,
        help="Spatial block size in grid cells for the bootstrap.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the selected pipeline stages."""

    args = parse_args(argv)
    selected_steps = {"core", "mechanisms", "robustness"} if "all" in args.steps else set(args.steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_analysis_matrix(args.matrix, max_cities=args.max_cities)
    eligible = eligible_rows(matrix)
    ctx = city_context(eligible)

    total = pathway = decomp = pooled = summary = None
    if selected_steps & {"core", "mechanisms", "robustness"}:
        total, pathway, decomp = fit_core_models(eligible, ctx)
        pooled = fit_pooled_models(eligible)
        summary = write_core_outputs(args.output_dir, total, pathway, decomp, pooled)

    if "mechanisms" in selected_steps:
        assert total is not None and pathway is not None and decomp is not None
        seq = sequential_adjustment(total, pathway, decomp)
        surface_city, surface_pooled = run_surface_context_models(eligible, ctx)
        grid = run_grid_interactions(eligible)
        city_mod = run_city_moderators(total, pathway, decomp, ctx)
        write_mechanism_outputs(args.output_dir, seq, surface_city, surface_pooled, grid, city_mod)

    if "robustness" in selected_steps:
        assert pathway is not None
        same_volume, same_volume_summary = same_volume_matched_contrast(matrix)
        sensitivity = sensitivity_configs(matrix)
        bootstrap, bootstrap_summary = spatial_block_bootstrap(
            matrix,
            pathway,
            n_bootstrap=args.bootstrap_replicates,
            block_size=args.bootstrap_block_size,
        )
        write_robustness_outputs(args.output_dir, same_volume, same_volume_summary, sensitivity, bootstrap, bootstrap_summary)

    manifest = {
        "created_utc": now_utc(),
        "python": sys.version,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "input_matrix": str(args.matrix),
        "output_dir": str(args.output_dir),
        "steps": sorted(selected_steps),
        "max_cities": args.max_cities,
        "n_rows_loaded": int(len(matrix)),
        "n_rows_eligible": int(len(eligible)),
        "n_cities_loaded": int(matrix[UID].nunique()),
        "n_cities_eligible": int(eligible[UID].nunique()),
        "primary_tier": PRIMARY_TIER_ID,
        "model_tiers": [asdict(tier) for tier in MODEL_TIERS],
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_block_size": int(args.bootstrap_block_size),
    }
    if summary is not None and len(summary):
        manifest["summary_rows"] = int(len(summary))
    write_json(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps({"output_dir": str(args.output_dir), "manifest": str(args.output_dir / "run_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
