"""01_build_panel.py -- Load and validate the 37-country SSA analysis panel.

The panel was built from raw sources by the paper 1 pipeline:
  - FAOSTAT Food Balance Sheets (Africa) -> sugar, oil supply columns
  - NCD-RisC Lancet 2024 BMI -> obesity prevalence
  - NCD-RisC Lancet 2024 Diabetes -> diabetes prevalence
  - World Development Indicators -> GDP, urbanization, population

This script validates the panel against raw NCD-RisC files and copies
it to data/derived/analysis_panel.csv for use by subsequent scripts.

Reads:
  ../../data/derived/ssa_faostat_ncdrisc_analysis_panel_2010_2022.csv
  data/raw/NCD_RisC_Lancet_2024_BMI_age_standardised_country.csv
  data/raw/NCD_RisC_Lancet_2024_Diabetes_age_standardised_countries.csv

Writes:
  data/derived/analysis_panel.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)

SOURCE_PANEL = ROOT.parent / "data" / "derived" / "ssa_faostat_ncdrisc_analysis_panel_2010_2022.csv"
OUTPUT = DERIVED / "analysis_panel.csv"
VALIDATION_OUTPUT = DERIVED / "panel_validation.json"


def validate_ncdrisc_bmi(panel: pd.DataFrame) -> dict:
    """Validate obesity prevalence against raw NCD-RisC BMI file."""
    bmi = pd.read_csv(RAW / "NCD_RisC_Lancet_2024_BMI_age_standardised_country.csv", encoding="utf-8-sig")
    obesity_col = [c for c in bmi.columns if "BMI>=30" in c and "lower" not in c.lower() and "upper" not in c.lower()][0]

    panel_iso3 = set(panel["iso3"].unique())
    panel_years = set(panel["year"].unique())

    bmi_check = bmi[bmi["ISO"].isin(panel_iso3) & bmi["Year"].isin(panel_years)].copy()
    bmi_check = bmi_check[bmi_check["Sex"].isin(["Women", "Men"])].copy()
    bmi_check["obesity_raw"] = pd.to_numeric(bmi_check[obesity_col], errors="coerce")

    mismatches = 0
    max_diff = 0.0
    checked = 0
    for _, bmi_row in bmi_check.iterrows():
        iso3 = bmi_row["ISO"]
        year = int(bmi_row["Year"])
        sex = bmi_row["Sex"]
        raw_val = bmi_row["obesity_raw"]
        if pd.isna(raw_val):
            continue

        panel_row = panel[(panel["iso3"] == iso3) & (panel["year"] == year) & (panel["sex"] == sex)]
        if len(panel_row) == 0:
            continue

        panel_val = float(panel_row["obesity_prevalence_share"].iloc[0])
        diff = abs(raw_val - panel_val)
        max_diff = max(max_diff, diff)
        if diff > 1e-6:
            mismatches += 1
        checked += 1

    return {
        "checked": checked,
        "mismatches_above_1e6": mismatches,
        "max_absolute_diff": round(max_diff, 10),
        "valid": mismatches == 0,
    }


def validate_ncdrisc_diabetes(panel: pd.DataFrame) -> dict:
    """Validate diabetes prevalence against raw NCD-RisC diabetes file."""
    diab = pd.read_csv(RAW / "NCD_RisC_Lancet_2024_Diabetes_age_standardised_countries.csv", encoding="utf-8-sig")
    diab = diab[diab["Sex"].isin(["Women", "Men"])].copy()
    if "Age" in diab.columns:
        diab = diab[diab["Age"] == "Age-standardised"].copy()
    diab["diabetes_raw"] = pd.to_numeric(diab["Prevalence of diabetes (18+ years)"], errors="coerce")

    panel_iso3 = set(panel["iso3"].unique())
    panel_years = set(panel["year"].unique())
    diab_check = diab[diab["ISO"].isin(panel_iso3) & diab["Year"].isin(panel_years)].copy()

    mismatches = 0
    max_diff = 0.0
    checked = 0
    for _, diab_row in diab_check.iterrows():
        iso3 = diab_row["ISO"]
        year = int(diab_row["Year"])
        sex = diab_row["Sex"]
        raw_val = diab_row["diabetes_raw"]
        if pd.isna(raw_val):
            continue

        panel_row = panel[(panel["iso3"] == iso3) & (panel["year"] == year) & (panel["sex"] == sex)]
        if len(panel_row) == 0:
            continue

        panel_val = float(panel_row["diabetes_prevalence_share"].iloc[0])
        diff = abs(raw_val - panel_val)
        max_diff = max(max_diff, diff)
        if diff > 1e-6:
            mismatches += 1
        checked += 1

    return {
        "checked": checked,
        "mismatches_above_1e6": mismatches,
        "max_absolute_diff": round(max_diff, 10),
        "valid": mismatches == 0,
    }


def main() -> None:
    print("=" * 70)
    print("01_build_panel.py: Load and validate the 37-country SSA analysis panel")
    print("=" * 70)

    # Load verified panel
    panel = pd.read_csv(SOURCE_PANEL)

    n_countries = panel["iso3"].nunique()
    n_rows = len(panel)
    years = sorted(panel["year"].unique())
    sexes = sorted(panel["sex"].unique())

    print(f"\nPanel loaded: {SOURCE_PANEL.name}")
    print(f"  Countries: {n_countries}")
    print(f"  Years: {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"  Sexes: {sexes}")
    print(f"  Rows: {n_rows}")

    # Validate
    assert n_countries == 37, f"Expected 37 countries, got {n_countries}"
    assert n_rows == 962, f"Expected 962 rows, got {n_rows}"

    # Cross-validate against raw NCD-RisC files
    print("\n--- Validating against raw NCD-RisC BMI ---")
    bmi_val = validate_ncdrisc_bmi(panel)
    print(f"  Checked {bmi_val['checked']} obesity values")
    print(f"  Mismatches: {bmi_val['mismatches_above_1e6']}")
    print(f"  Max absolute diff: {bmi_val['max_absolute_diff']}")
    print(f"  Valid: {bmi_val['valid']}")

    print("\n--- Validating against raw NCD-RisC Diabetes ---")
    diab_val = validate_ncdrisc_diabetes(panel)
    print(f"  Checked {diab_val['checked']} diabetes values")
    print(f"  Mismatches: {diab_val['mismatches_above_1e6']}")
    print(f"  Max absolute diff: {diab_val['max_absolute_diff']}")
    print(f"  Valid: {diab_val['valid']}")

    # Verify derived columns
    print("\n--- Verifying derived columns ---")
    assert np.allclose(
        panel["obesity_prevalence_pct"],
        panel["obesity_prevalence_share"] * 100,
        atol=1e-4,
    ), "obesity_prevalence_pct mismatch"
    assert np.allclose(
        panel["diabetes_prevalence_pct"],
        panel["diabetes_prevalence_share"] * 100,
        atol=1e-4,
    ), "diabetes_prevalence_pct mismatch"
    assert np.allclose(
        panel["primary_exposure_food_supply_kcal_capita_day"],
        panel["sugar_sweeteners_food_supply_kcal_capita_day"],
        atol=1e-4,
    ), "primary_exposure mismatch"
    print("  All derived columns consistent")

    # Save
    panel.to_csv(OUTPUT, index=False)

    validation = {
        "n_countries": int(n_countries),
        "n_rows": n_rows,
        "years": [int(y) for y in years],
        "bmi_validation": bmi_val,
        "diabetes_validation": diab_val,
        "all_valid": bmi_val["valid"] and diab_val["valid"],
    }
    with open(VALIDATION_OUTPUT, "w") as f:
        json.dump(validation, f, indent=2)

    print(f"\nPanel saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Validation saved: {VALIDATION_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
