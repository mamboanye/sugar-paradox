# Sugar Paradox Paper 2 -- Reproducible Workspace

National food supply predicts obesity across countries but not within them.

## Quick Start

```bash
cd scripts
bash run_all.sh
```

Requires `uv` with Python 3.10+. All dependencies are installed inline via `uv run --with`.

## Directory Layout

```
data/
  raw/          Symlinks to source files (FAOSTAT, NCD-RisC, WDI)
  derived/      Intermediate panels and result JSONs (created by scripts)
scripts/
  01_build_panel.py          Rebuild the 37-country analysis panel
  02_cross_sectional.py      Cross-sectional correlations, partial r, LOO
  03_within_country_fe.py    FE family: one-way, TWFE, detrended, FD, LD, CRE
  04_trend_decomposition.py  Trend R2, FE absorption, SE inflation
  05_robustness_failure.py   Vegetable oil case study (co-trending diagnosis)
  06_global_scope.py         Global NCD-RisC trend R2 (all countries)
  07_gdp_positive_control.py GDP: positive cross-sectional, negative detrended
  08_cross_country_change.py Initial conditions vs change regressions
  09_figures.py              6 publication-quality figures
  10_tables.py               4 tables (descriptives, cascade, robustness, change)
  run_all.sh                 Single command to run everything
figures/                     PNG outputs
tables/                      CSV outputs
```

## Data Sources

- FAOSTAT Food Balance Sheets (Africa), downloaded 2024
- NCD-RisC Lancet 2024: BMI and diabetes age-standardised country estimates
- World Development Indicators (pre-downloaded panel)

## Key Results

All standard errors are clustered at the country level. The analysis panel contains 37 SSA countries, 2010-2022, 962 rows (country x year x sex).
