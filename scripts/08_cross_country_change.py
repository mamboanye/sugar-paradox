"""08_cross_country_change.py -- Cross-country change analysis.

Tests whether initial conditions predict subsequent change:
  - Multiple regression: d_obesity ~ d_urban + d_sugar + d_oil + d_gdp + init_obesity
  - Initial obesity is the only significant predictor

Verified targets:
  - Initial obesity t in multiple regression = 3.45 (R2=0.405)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
OUTPUT = DERIVED / "cross_country_change_results.json"


def main() -> None:
    print("=" * 70)
    print("08_cross_country_change.py: Cross-country change analysis")
    print("=" * 70)

    df = pd.read_csv(DERIVED / "analysis_panel.csv")
    women = df[df["sex"] == "Women"].copy()

    years_avail = sorted(women["year"].unique())
    yr_start = years_avail[0]  # 2010
    yr_end = 2020 if 2020 in years_avail else years_avail[-1]

    def get_year(data, yr):
        sub = data[data["year"] == yr][["iso3",
            "obesity_prevalence_pct",
            "primary_exposure_food_supply_kcal_capita_day",
            "vegetable_oils_comparison_food_supply_kcal_capita_day",
            "log_gdp_per_capita_constant_2015_usd",
            "urban_population_pct"]].copy()
        return sub

    start_df = get_year(women, yr_start).add_suffix("_init").rename(columns={"iso3_init": "iso3"})
    end_df = get_year(women, yr_end).add_suffix("_end").rename(columns={"iso3_end": "iso3"})
    change_df = start_df.merge(end_df, on="iso3")

    change_df["d_obesity"] = change_df["obesity_prevalence_pct_end"] - change_df["obesity_prevalence_pct_init"]
    change_df["d_sugar"] = change_df["primary_exposure_food_supply_kcal_capita_day_end"] - change_df["primary_exposure_food_supply_kcal_capita_day_init"]
    change_df["d_oil"] = change_df["vegetable_oils_comparison_food_supply_kcal_capita_day_end"] - change_df["vegetable_oils_comparison_food_supply_kcal_capita_day_init"]
    change_df["d_gdp"] = change_df["log_gdp_per_capita_constant_2015_usd_end"] - change_df["log_gdp_per_capita_constant_2015_usd_init"]
    change_df["d_urban"] = change_df["urban_population_pct_end"] - change_df["urban_population_pct_init"]
    change_df["init_obesity"] = change_df["obesity_prevalence_pct_init"]

    reg_data = change_df.dropna(subset=["d_obesity", "d_sugar", "d_oil", "d_gdp", "d_urban", "init_obesity"])
    print(f"\nMultiple regression sample: N={len(reg_data)} countries ({yr_start}-{yr_end})")

    results = {}

    # 1. Simple correlations
    print("\n--- Simple correlations with obesity change ---")
    for var, label in [("d_sugar", "Sugar change"), ("d_oil", "Oil change"),
                       ("d_gdp", "GDP change"), ("d_urban", "Urban change"),
                       ("init_obesity", "Initial obesity")]:
        r, p = stats.pearsonr(reg_data[var], reg_data["d_obesity"])
        print(f"  {label}: r = {r:.3f}, p = {p:.4f}")

    # 2. Full multiple regression (matching verification spec)
    print("\n--- Multiple regression ---")
    fit = smf.ols(
        "d_obesity ~ d_urban + d_sugar + d_oil + d_gdp + init_obesity",
        data=reg_data,
    ).fit()

    print(f"  R2 = {fit.rsquared:.3f}")
    for var in ["d_urban", "d_sugar", "d_oil", "d_gdp", "init_obesity"]:
        b = float(fit.params[var])
        t = float(fit.tvalues[var])
        p = float(fit.pvalues[var])
        sig = "**" if p < 0.05 else ("*" if p < 0.10 else "")
        print(f"  {var}: beta = {b:.4f}, t = {t:.2f}, p = {p:.4f} {sig}")

    results["multiple_regression"] = {
        "n_countries": int(len(reg_data)),
        "period": f"{yr_start}-{yr_end}",
        "r_squared": round(fit.rsquared, 3),
    }
    for var in ["d_urban", "d_sugar", "d_oil", "d_gdp", "init_obesity"]:
        results["multiple_regression"][var] = {
            "beta": round(float(fit.params[var]), 4),
            "t": round(float(fit.tvalues[var]), 2),
            "p": round(float(fit.pvalues[var]), 4),
        }

    # 3. Divergence stats
    print("\n--- Divergence ---")
    all_gained = (reg_data["d_obesity"] > 0).all()
    print(f"  All countries gained obesity: {all_gained}")
    print(f"  Mean change: {reg_data['d_obesity'].mean():.2f} pp")
    print(f"  Min change: {reg_data['d_obesity'].min():.2f} pp")
    print(f"  Max change: {reg_data['d_obesity'].max():.2f} pp")

    results["divergence"] = {
        "all_gained": bool(all_gained),
        "mean_change_pp": round(reg_data["d_obesity"].mean(), 2),
        "min_change_pp": round(reg_data["d_obesity"].min(), 2),
        "max_change_pp": round(reg_data["d_obesity"].max(), 2),
    }

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
