"""04_trend_decomposition.py -- Trend R2, FE absorption, autocorrelation, SE inflation.

Verified targets:
  - Trend R2 (women, SSA, 2010-2021) = 99.68% (pooled)
  - Entity FE absorption = 95.73%
  - Entity + year FE absorption = 99.43%
  - SE inflation range = 1.9x to 3.8x (baseline-dependent)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
OUTPUT = DERIVED / "trend_decomposition_results.json"


def main() -> None:
    print("=" * 70)
    print("04_trend_decomposition.py: Trend decomposition")
    print("=" * 70)

    df = pd.read_csv(DERIVED / "analysis_panel.csv")
    df["sugar_10kcal"] = df["primary_exposure_food_supply_kcal_capita_day"] / 10.0
    df["oil_100kcal"] = df["vegetable_oils_comparison_food_supply_kcal_capita_day"] / 100.0

    women = df[df["sex"] == "Women"].copy()

    results = {}

    # 1. Trend R2: how much of obesity variation within each country is linear trend?
    print("\n--- Country-level trend R2 ---")
    trend_r2s = {}
    for iso3 in women["iso3"].unique():
        sub = women[women["iso3"] == iso3].sort_values("year")
        y = sub["obesity_prevalence_pct"].values
        t = sub["year"].values - sub["year"].values[0]
        if len(y) < 3:
            continue
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            trend_r2s[iso3] = 1.0
            continue
        slope, intercept = np.polyfit(t, y, 1)
        y_pred = slope * t + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        trend_r2s[iso3] = 1 - ss_res / ss_tot

    r2_values = list(trend_r2s.values())
    mean_r2 = np.mean(r2_values)
    min_r2 = np.min(r2_values)
    max_r2 = np.max(r2_values)
    pct_above_90 = np.mean([r >= 0.90 for r in r2_values]) * 100
    pct_above_95 = np.mean([r >= 0.95 for r in r2_values]) * 100

    print(f"  Mean country-level trend R2: {mean_r2:.4f}")
    print(f"  Min: {min_r2:.4f}, Max: {max_r2:.4f}")
    print(f"  Countries with R2 >= 0.90: {pct_above_90:.1f}%")
    print(f"  Countries with R2 >= 0.95: {pct_above_95:.1f}%")

    results["country_trend_r2"] = {
        "mean": round(mean_r2, 4),
        "min": round(min_r2, 4),
        "max": round(max_r2, 4),
        "pct_above_90": round(pct_above_90, 1),
        "pct_above_95": round(pct_above_95, 1),
    }

    # 2. Pooled trend R2 (all SSA women obesity on linear time trend)
    print("\n--- Pooled trend R2 ---")
    women_trend = women.copy()
    women_trend["t"] = women_trend["year"] - 2010
    # Within-country demeaned + linear trend
    fit_pooled = smf.ols("obesity_prevalence_pct ~ C(iso3) + t", data=women_trend).fit()
    print(f"  Pooled R2 (country FE + linear t): {fit_pooled.rsquared:.4f}")
    print(f"  Pooled R2 pct: {fit_pooled.rsquared * 100:.2f}%")
    results["pooled_trend_r2"] = round(fit_pooled.rsquared * 100, 2)

    # 3. FE absorption analysis
    print("\n--- FE absorption ---")
    # Null model (intercept only)
    fit_null = smf.ols("obesity_prevalence_pct ~ 1", data=women).fit()
    ss_tot = np.sum((women["obesity_prevalence_pct"] - women["obesity_prevalence_pct"].mean()) ** 2)

    # Entity FE only
    fit_entity = smf.ols("obesity_prevalence_pct ~ C(iso3)", data=women).fit()
    entity_r2 = fit_entity.rsquared * 100
    print(f"  Entity FE R2: {entity_r2:.2f}%")

    # Entity + year FE
    fit_entity_year = smf.ols("obesity_prevalence_pct ~ C(iso3) + C(year)", data=women).fit()
    entity_year_r2 = fit_entity_year.rsquared * 100
    print(f"  Entity + year FE R2: {entity_year_r2:.2f}%")

    results["fe_absorption"] = {
        "entity_fe_r2_pct": round(entity_r2, 2),
        "entity_year_fe_r2_pct": round(entity_year_r2, 2),
    }

    # 4. SE inflation: compare non-clustered vs clustered SEs across specs
    print("\n--- SE inflation ---")
    se_specs = {
        "one_way_fe": "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal + C(iso3)",
        "twfe": "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal + C(iso3) + C(year)",
        "twfe_bivariate": "obesity_prevalence_pct ~ sugar_10kcal + C(iso3) + C(year)",
    }
    se_ratios = {}
    all_ratios = []
    for spec_name, formula in se_specs.items():
        fit_nc = smf.ols(formula, data=women).fit()
        fit_cl = smf.ols(formula, data=women).fit(
            cov_type="cluster", cov_kwds={"groups": women["iso3"]}
        )
        se_nc = float(fit_nc.bse["sugar_10kcal"])
        se_cl = float(fit_cl.bse["sugar_10kcal"])
        ratio = se_cl / se_nc
        all_ratios.append(ratio)
        se_ratios[spec_name] = {
            "se_non_clustered": round(se_nc, 6),
            "se_clustered": round(se_cl, 6),
            "inflation_ratio": round(ratio, 1),
        }
        print(f"  {spec_name}: SE_nc={se_nc:.6f}, SE_cl={se_cl:.6f}, ratio={ratio:.1f}x")

    print(f"  Range across specifications: {min(all_ratios):.1f}x to {max(all_ratios):.1f}x")
    se_ratios["range"] = {
        "min": round(min(all_ratios), 1),
        "max": round(max(all_ratios), 1),
    }

    results["se_inflation"] = se_ratios

    # 5. Autocorrelation of residuals (using TWFE with oil)
    print("\n--- Residual autocorrelation ---")
    fit_twfe_clust = smf.ols(
        "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal + C(iso3) + C(year)",
        data=women,
    ).fit(cov_type="cluster", cov_kwds={"groups": women["iso3"]})
    resid = fit_twfe_clust.resid
    women_resid = women.copy()
    women_resid["resid"] = resid.values
    ar1_by_country = {}
    for iso3 in women_resid["iso3"].unique():
        sub = women_resid[women_resid["iso3"] == iso3].sort_values("year")
        r = sub["resid"].values
        if len(r) < 3:
            continue
        ar1 = np.corrcoef(r[:-1], r[1:])[0, 1]
        ar1_by_country[iso3] = round(ar1, 3)

    mean_ar1 = np.mean(list(ar1_by_country.values()))
    print(f"  Mean within-country AR(1) of TWFE residuals: {mean_ar1:.3f}")
    results["residual_ar1"] = {
        "mean": round(mean_ar1, 3),
        "min": round(min(ar1_by_country.values()), 3),
        "max": round(max(ar1_by_country.values()), 3),
    }

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
