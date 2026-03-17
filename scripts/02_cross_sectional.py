"""02_cross_sectional.py -- Cross-sectional correlations, partial r, LOO, subregion.

All analyses use 2018-2020 country averages for consistency with the
primary specification (partial r = 0.496).

Verified targets:
  - Cross-sectional sugar-obesity partial r = 0.496 (p=0.0018)
    (controlling for log GDP, urbanization, and vegetable oil)
  - LOO min partial r = 0.417 (dropping Mauritania)
  - Subregion partial r = 0.632 (controlling for subregion dummies only)
  - Subregion + GDP partial r = 0.462 (controlling for subregion + log GDP)

Note: original claims of subregion r=0.580 and subregion+GDP r=0.385
came from a different specification (kg sugar, latest year, 49 countries).
This pipeline uses kcal sugar, 2018-2020 avg, 37 countries.
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
OUTPUT = DERIVED / "cross_sectional_results.json"

# UN subregion mapping for the 37 panel countries
SUBREGIONS = {
    "AGO": "Middle", "BFA": "Western", "BWA": "Southern",
    "CIV": "Western", "CMR": "Middle", "COD": "Middle",
    "COG": "Middle", "COM": "Eastern", "CPV": "Western",
    "ETH": "Eastern", "GAB": "Middle", "GHA": "Western",
    "GIN": "Western", "GMB": "Western", "GNB": "Western",
    "KEN": "Eastern", "LBR": "Western", "LSO": "Southern",
    "MDG": "Eastern", "MOZ": "Eastern", "MRT": "Western",
    "MUS": "Eastern", "MWI": "Eastern", "NAM": "Southern",
    "NER": "Western", "NGA": "Western", "RWA": "Eastern",
    "SEN": "Western", "SLE": "Western", "STP": "Middle",
    "SWZ": "Southern", "SYC": "Eastern", "TZA": "Eastern",
    "UGA": "Eastern", "ZAF": "Southern", "ZMB": "Eastern",
    "ZWE": "Eastern",
}


def main() -> None:
    print("=" * 70)
    print("02_cross_sectional.py: Cross-sectional analysis")
    print("=" * 70)

    panel = pd.read_csv(DERIVED / "analysis_panel.csv")

    # Women obesity, 2018-2020 average for cross-sectional analysis
    women = panel[panel["sex"] == "Women"].copy()

    # Primary cross-section: 2018-2020 average (matches verified spec)
    cs_years = [2018, 2019, 2020]
    cs = women[women["year"].isin(cs_years)].groupby("iso3").agg(
        obesity_prevalence_pct=("obesity_prevalence_pct", "mean"),
        sugar_kcal=("primary_exposure_food_supply_kcal_capita_day", "mean"),
        oil_kcal=("vegetable_oils_comparison_food_supply_kcal_capita_day", "mean"),
        log_gdp=("log_gdp_per_capita_constant_2015_usd", "mean"),
        urban_pct=("urban_population_pct", "mean"),
    ).dropna()

    n = len(cs)
    print(f"\nCross-sectional sample: {n} countries (avg 2018-2020)")

    # 1. Simple bivariate correlation
    r_raw, p_raw = stats.pearsonr(cs["sugar_kcal"], cs["obesity_prevalence_pct"])
    print(f"\nRaw bivariate r (sugar-obesity): {r_raw:.3f} (p={p_raw:.4f})")

    # 2. Partial r controlling for GDP + urbanization + oil
    controls = "~ log_gdp + urban_pct + oil_kcal"
    res_sugar = smf.ols(f"sugar_kcal {controls}", data=cs).fit().resid
    res_obesity = smf.ols(f"obesity_prevalence_pct {controls}", data=cs).fit().resid
    r_partial, p_partial = stats.pearsonr(res_sugar, res_obesity)
    print(f"Partial r (sugar-obesity | GDP+urban+oil): {r_partial:.3f} (p={p_partial:.4f})")

    # Also report full-panel means for comparison
    cs_all = women.groupby("iso3").agg(
        obesity_prevalence_pct=("obesity_prevalence_pct", "mean"),
        sugar_kcal=("primary_exposure_food_supply_kcal_capita_day", "mean"),
        oil_kcal=("vegetable_oils_comparison_food_supply_kcal_capita_day", "mean"),
        log_gdp=("log_gdp_per_capita_constant_2015_usd", "mean"),
        urban_pct=("urban_population_pct", "mean"),
    ).dropna()
    res_sugar_all = smf.ols(f"sugar_kcal {controls}", data=cs_all).fit().resid
    res_obesity_all = smf.ols(f"obesity_prevalence_pct {controls}", data=cs_all).fit().resid
    r_partial_all, p_partial_all = stats.pearsonr(res_sugar_all, res_obesity_all)
    print(f"Partial r (full-panel means): {r_partial_all:.3f} (p={p_partial_all:.4f})")

    # 3. LOO analysis -- uses 2018-2020 averages (same as primary spec)
    print("\n--- LOO analysis ---")
    loo_results = []
    for drop_iso in cs.index:
        cs_loo = cs.drop(drop_iso)
        r_s = smf.ols(f"sugar_kcal {controls}", data=cs_loo).fit().resid
        r_o = smf.ols(f"obesity_prevalence_pct {controls}", data=cs_loo).fit().resid
        r_loo, p_loo = stats.pearsonr(r_s, r_o)
        loo_results.append({"iso3": drop_iso, "partial_r": r_loo, "p": p_loo})

    loo_df = pd.DataFrame(loo_results).sort_values("partial_r")
    min_loo = loo_df.iloc[0]
    max_loo = loo_df.iloc[-1]
    print(f"LOO range: [{min_loo['partial_r']:.3f}, {max_loo['partial_r']:.3f}]")
    print(f"LOO min: dropping {min_loo['iso3']}, partial r = {min_loo['partial_r']:.3f}")

    # 4. Subregion analysis -- uses 2018-2020 averages for consistency
    print("\n--- Subregion analysis ---")
    cs["subregion"] = cs.index.map(SUBREGIONS)

    res_sugar_sub = smf.ols("sugar_kcal ~ C(subregion)", data=cs).fit().resid
    res_obesity_sub = smf.ols("obesity_prevalence_pct ~ C(subregion)", data=cs).fit().resid
    r_sub, p_sub = stats.pearsonr(res_sugar_sub, res_obesity_sub)
    print(f"Partial r (sugar-obesity | subregion): {r_sub:.3f} (p={p_sub:.4f})")

    # 5. Subregion + GDP
    res_sugar_sg = smf.ols("sugar_kcal ~ C(subregion) + log_gdp", data=cs).fit().resid
    res_obesity_sg = smf.ols("obesity_prevalence_pct ~ C(subregion) + log_gdp", data=cs).fit().resid
    r_sub_gdp, p_sub_gdp = stats.pearsonr(res_sugar_sg, res_obesity_sg)
    print(f"Partial r (sugar-obesity | subregion+GDP): {r_sub_gdp:.3f} (p={p_sub_gdp:.4f})")

    results = {
        "n_countries": int(n),
        "cross_section_years": cs_years,
        "raw_bivariate": {"r": round(r_raw, 3), "p": round(p_raw, 4)},
        "partial_r_gdp_urban_oil": {"r": round(r_partial, 3), "p": round(p_partial, 4)},
        "partial_r_full_panel_means": {"r": round(r_partial_all, 3), "p": round(p_partial_all, 4)},
        "loo": {
            "min_r": round(float(min_loo["partial_r"]), 3),
            "min_country": str(min_loo["iso3"]),
            "max_r": round(float(max_loo["partial_r"]), 3),
            "max_country": str(max_loo["iso3"]),
        },
        "subregion_partial_r": {"r": round(r_sub, 3), "p": round(p_sub, 4)},
        "subregion_gdp_partial_r": {"r": round(r_sub_gdp, 3), "p": round(p_sub_gdp, 4)},
    }

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
