"""06_global_scope.py -- Global NCD-RisC trend R2.

Uses the FULL NCD-RisC BMI file (all ~200 countries, Women, 2010-2021).
Tests: "98% of countries globally have trend R2 >= 0.90"

This claim was NOT independently verified in the March 2026 audit.
This script reproduces it from raw data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
OUTPUT = DERIVED / "global_scope_results.json"

YEARS = list(range(2010, 2022))  # 2010-2021


def main() -> None:
    print("=" * 70)
    print("06_global_scope.py: Global NCD-RisC trend R2")
    print("=" * 70)

    # Load full NCD-RisC BMI
    bmi = pd.read_csv(RAW / "NCD_RisC_Lancet_2024_BMI_age_standardised_country.csv", encoding="utf-8-sig")

    # Identify obesity column
    obesity_col = [c for c in bmi.columns if "BMI>=30" in c and "lower" not in c.lower() and "upper" not in c.lower()][0]

    # Filter: Women, 2010-2021, country-level only (exclude regions/world)
    bmi = bmi[bmi["Sex"] == "Women"].copy()
    bmi = bmi[bmi["Year"].isin(YEARS)].copy()
    bmi["obesity"] = pd.to_numeric(bmi[obesity_col], errors="coerce")
    bmi = bmi.dropna(subset=["obesity"])

    # Remove aggregate regions (ISO codes that are empty or non-standard)
    bmi = bmi[bmi["ISO"].str.len() == 3].copy()
    bmi = bmi[~bmi["ISO"].str.startswith("X")].copy()  # exclude aggregate codes

    print(f"Countries with data: {bmi['ISO'].nunique()}")
    print(f"Years: {sorted(bmi['Year'].unique())}")

    # Compute trend R2 per country
    trend_r2 = {}
    for iso3, group in bmi.groupby("ISO"):
        if len(group) < 3:
            continue
        y = group["obesity"].values
        t = group["Year"].values - group["Year"].values.min()
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot < 1e-15:
            trend_r2[iso3] = 1.0
            continue
        slope, intercept = np.polyfit(t, y, 1)
        y_pred = slope * t + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        trend_r2[iso3] = 1 - ss_res / ss_tot

    n_countries = len(trend_r2)
    r2_values = np.array(list(trend_r2.values()))

    pct_above_90 = np.mean(r2_values >= 0.90) * 100
    pct_above_95 = np.mean(r2_values >= 0.95) * 100
    pct_above_99 = np.mean(r2_values >= 0.99) * 100

    print(f"\n--- Global trend R2 (Women, obesity, 2010-2021) ---")
    print(f"  Countries analyzed: {n_countries}")
    print(f"  Mean R2: {np.mean(r2_values):.4f}")
    print(f"  Median R2: {np.median(r2_values):.4f}")
    print(f"  Min R2: {np.min(r2_values):.4f}")
    print(f"  >= 0.90: {pct_above_90:.1f}%")
    print(f"  >= 0.95: {pct_above_95:.1f}%")
    print(f"  >= 0.99: {pct_above_99:.1f}%")

    # Countries below 0.90
    below_90 = {k: round(v, 4) for k, v in sorted(trend_r2.items(), key=lambda x: x[1]) if v < 0.90}
    print(f"\n  Countries below 0.90 ({len(below_90)}):")
    for iso3, r2 in below_90.items():
        print(f"    {iso3}: {r2}")

    # Also compute for diabetes
    print("\n--- Global diabetes trend R2 ---")
    diab = pd.read_csv(RAW / "NCD_RisC_Lancet_2024_Diabetes_age_standardised_countries.csv", encoding="utf-8-sig")
    diab = diab[diab["Sex"] == "Women"].copy()
    diab = diab[diab["Year"].isin(YEARS)].copy()
    if "Age" in diab.columns:
        diab = diab[diab["Age"] == "Age-standardised"].copy()
    diab["diabetes"] = pd.to_numeric(diab["Prevalence of diabetes (18+ years)"], errors="coerce")
    diab = diab.dropna(subset=["diabetes"])
    diab = diab[diab["ISO"].str.len() == 3].copy()

    diab_r2 = {}
    for iso3, group in diab.groupby("ISO"):
        if len(group) < 3:
            continue
        y = group["diabetes"].values
        t = group["Year"].values - group["Year"].values.min()
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot < 1e-15:
            diab_r2[iso3] = 1.0
            continue
        slope, intercept = np.polyfit(t, y, 1)
        y_pred = slope * t + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        diab_r2[iso3] = 1 - ss_res / ss_tot

    diab_r2_values = np.array(list(diab_r2.values()))
    diab_pct_90 = np.mean(diab_r2_values >= 0.90) * 100
    print(f"  Diabetes countries analyzed: {len(diab_r2)}")
    print(f"  Diabetes >= 0.90: {diab_pct_90:.1f}%")

    results = {
        "obesity_women": {
            "n_countries": n_countries,
            "mean_r2": round(float(np.mean(r2_values)), 4),
            "median_r2": round(float(np.median(r2_values)), 4),
            "min_r2": round(float(np.min(r2_values)), 4),
            "max_r2": round(float(np.max(r2_values)), 4),
            "pct_above_90": round(pct_above_90, 1),
            "pct_above_95": round(pct_above_95, 1),
            "pct_above_99": round(pct_above_99, 1),
            "countries_below_90": below_90,
        },
        "diabetes_women": {
            "n_countries": len(diab_r2),
            "mean_r2": round(float(np.mean(diab_r2_values)), 4),
            "pct_above_90": round(diab_pct_90, 1),
        },
        "claim_verification": {
            "claim": "98% of countries globally have trend R2 >= 0.90",
            "observed_pct": round(pct_above_90, 1),
            "verified": bool(abs(pct_above_90 - 98.0) < 2.0),
        },
    }

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
