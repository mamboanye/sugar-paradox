#!/usr/bin/env bash
# run_all.sh -- Execute the full Sugar Paradox paper2 pipeline.
#
# Usage:
#   cd workspace/paper2/scripts
#   bash run_all.sh
#
# Each script runs with uv and prints key results to stdout.
# All scripts are deterministic and read from data/raw/ (symlinks).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

UV_RUN="uv run --with pandas --with statsmodels --with scipy --with matplotlib"

echo "============================================================"
echo "Sugar Paradox Paper 2 -- Full Pipeline"
echo "============================================================"
echo ""

for script in \
    01_build_panel.py \
    02_cross_sectional.py \
    03_within_country_fe.py \
    03b_food_groups_twfe.py \
    04_trend_decomposition.py \
    05_robustness_failure.py \
    06_global_scope.py \
    07_gdp_positive_control.py \
    08_cross_country_change.py \
    09_figures.py \
    10_tables.py; do
    echo ""
    echo ">>> Running $script"
    $UV_RUN python "$script"
    echo ">>> $script completed"
    echo ""
done

echo "============================================================"
echo "All scripts completed successfully."
echo "============================================================"
echo "Outputs:"
echo "  data/derived/  -- intermediate panels and result JSONs"
echo "  figures/       -- publication-quality PNGs"
echo "  tables/        -- CSV table source files"
