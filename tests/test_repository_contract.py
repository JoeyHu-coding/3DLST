from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _dependency_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-r "):
            continue
        names.add(line.split("=", 1)[0].split("<", 1)[0].split(">", 1)[0].lower())
    return names


def test_runtime_and_development_dependencies_are_separate() -> None:
    runtime = _dependency_names(ROOT / "requirements.txt")
    development_path = ROOT / "requirements-dev.txt"

    assert "pytest" not in runtime
    assert development_path.exists()
    development_text = development_path.read_text(encoding="utf-8")
    assert "-r requirements.txt" in development_text
    assert "pytest==8.4.2" in development_text


def test_readme_names_the_tested_environment_and_release_inputs() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Python 3.13.9" in readme
    assert "analysis_grid_core.parquet" in readme
    assert "city_analysis_core.csv" in readme
    assert "run_city_analysis.py" in readme
    assert "10.5281/zenodo.20755781" in readme
    assert "all grid-matrix stages" in readme
