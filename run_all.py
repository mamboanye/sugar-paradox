#!/usr/bin/env python3
"""Cross-platform pipeline runner. Works on Windows, macOS, and Linux.

Usage:
    python run_all.py

Requires: uv (https://docs.astral.sh/uv/)
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

SCRIPTS = [
    "01_build_panel.py",
    "02_cross_sectional.py",
    "03_within_country_fe.py",
    "03b_food_groups_twfe.py",
    "04_trend_decomposition.py",
    "05_robustness_failure.py",
    "06_global_scope.py",
    "07_gdp_positive_control.py",
    "08_cross_country_change.py",
    "09_figures.py",
    "10_tables.py",
]

UV_DEPS = ["pandas", "statsmodels", "scipy", "matplotlib"]


def main() -> int:
    print("=" * 60)
    print("Sugar Paradox -- Full Pipeline")
    print("=" * 60)

    uv_run = ["uv", "run"] + [arg for dep in UV_DEPS for arg in ("--with", dep)] + ["python"]

    for script in SCRIPTS:
        path = SCRIPTS_DIR / script
        if not path.exists():
            print(f"SKIP: {script} not found")
            continue
        print(f"\n>>> {script}")
        result = subprocess.run(uv_run + [str(path)], cwd=str(SCRIPTS_DIR))
        if result.returncode != 0:
            print(f"FAILED: {script} (exit code {result.returncode})")
            return result.returncode
        print(f">>> {script} done")

    print("\n" + "=" * 60)
    print("All scripts completed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
