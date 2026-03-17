# The Sugar Paradox

**National food supply predicts obesity across countries but not within them.**

## Quick start

```
git clone https://github.com/mamboanye/sugar-paradox.git
cd sugar-paradox
python run_all.py
```

Requires [uv](https://docs.astral.sh/uv/). All Python dependencies are installed automatically. Works on macOS, Linux, and Windows.

## What this reproduces

Every number in the paper traces to one of 11 scripts in `scripts/`. The pipeline:

1. Validates the 37-country analysis panel against raw NCD-RisC files
2. Computes cross-sectional correlations, partial r, leave-one-out, subregion robustness
3. Runs the within-country specification cascade (FE, TWFE, detrending, FD, LD, CRE)
4. Decomposes variance (entity FE 95.7%, year FE 3.7%, residual 0.6%)
5. Runs the soybean oil case study (8 diagnostic tests, all consistent with co-trending)
6. Tests global scope (98% of 200 countries have linear obesity trends with R2 >= 0.90)
7. GDP positive control (cross-sectional positive, detrended negative at t = -4.12)
8. Cross-country change regression (initial obesity t = 3.45, food supply null)
9. Generates all figures
10. Generates all tables

## Data

All data are included in `data/raw/`:

| File | Source |
|------|--------|
| `FoodBalanceSheets_E_Africa.zip` | [FAOSTAT Food Balance Sheets](https://www.fao.org/faostat/en/#data/FBS) |
| `NCD_RisC_Lancet_2024_BMI_age_standardised_country.csv` | [NCD-RisC](https://www.ncdrisc.org/) |
| `NCD_RisC_Lancet_2024_Diabetes_age_standardised_countries.csv` | [NCD-RisC](https://www.ncdrisc.org/) |
| `wdi_ssa_food_health_panel_2000_2024.csv` | [World Bank WDI](https://databank.worldbank.org/) |
| `ssa_faostat_ncdrisc_analysis_panel_2010_2022.csv` | Merged panel (37 countries, 962 rows) |

## Outputs

- `data/derived/` -- result JSONs (one per script)
- `figures/` -- PNGs (6 figures)
- `tables/` -- CSVs (4 tables)
- `paper/latex/main.pdf` -- manuscript

## Manuscript

LaTeX source in `paper/latex/`. To recompile: `cd paper/latex && bash build.sh`
