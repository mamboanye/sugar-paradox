"""10_tables.py -- All paper tables.

Table 1: Sample descriptive statistics
Table 2: FE specification cascade (sugar and oil coefficients)
Table 3: Robustness tests for soybean oil supply (co-trending case study)
Table 4: Cross-country change regressions
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
TABLES = ROOT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def table_1_descriptives(panel: pd.DataFrame) -> None:
    """Sample descriptive statistics."""
    women = panel[panel["sex"] == "Women"]
    men = panel[panel["sex"] == "Men"]

    rows = []
    vars_all = [
        ("Sugar supply (kcal/cap/day)", "primary_exposure_food_supply_kcal_capita_day"),
        ("Sugar raw equiv. (kcal/cap/day)", "sugar_sensitivity_food_supply_kcal_capita_day"),
        ("Vegetable oils (kcal/cap/day)", "vegetable_oils_comparison_food_supply_kcal_capita_day"),
        ("GDP per capita (2015 USD)", "gdp_per_capita_constant_2015_usd"),
        ("Urban population (%)", "urban_population_pct"),
    ]
    vars_sex = [
        ("Obesity prevalence (%)", "obesity_prevalence_pct"),
        ("Diabetes prevalence (%)", "diabetes_prevalence_pct"),
    ]

    for label, col in vars_sex:
        for sex_label, subset in [("Women", women), ("Men", men)]:
            vals = subset[col].dropna()
            rows.append({
                "Variable": f"{label} -- {sex_label}",
                "N": int(len(vals)),
                "Mean": round(vals.mean(), 2),
                "SD": round(vals.std(), 2),
                "Min": round(vals.min(), 2),
                "Max": round(vals.max(), 2),
            })

    for label, col in vars_all:
        vals = panel[col].dropna()
        # Deduplicate for food supply (same for Men and Women in same country-year)
        dedup = panel.drop_duplicates(subset=["iso3", "year"])[col].dropna()
        rows.append({
            "Variable": label,
            "N": int(len(dedup)),
            "Mean": round(dedup.mean(), 2),
            "SD": round(dedup.std(), 2),
            "Min": round(dedup.min(), 2),
            "Max": round(dedup.max(), 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "table1_descriptives.csv", index=False)
    print("  Table 1 saved")


def table_2_fe_cascade(panel: pd.DataFrame) -> None:
    """FE specification cascade."""
    women = panel[panel["sex"] == "Women"].copy()
    women["sugar_10kcal"] = women["primary_exposure_food_supply_kcal_capita_day"] / 10.0
    women["oil_100kcal"] = women["vegetable_oils_comparison_food_supply_kcal_capita_day"] / 100.0

    specs = [
        ("OLS (cross-sectional)", "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal", False),
        ("Country FE", "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal + C(iso3)", True),
        ("TWFE", "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal + C(iso3) + C(year)", True),
    ]

    rows = []
    for label, formula, cluster in specs:
        if cluster:
            fit = smf.ols(formula, data=women).fit(
                cov_type="cluster", cov_kwds={"groups": women["iso3"]}
            )
        else:
            fit = smf.ols(formula, data=women).fit()

        sugar_b = float(fit.params["sugar_10kcal"])
        sugar_se = float(fit.bse["sugar_10kcal"])
        sugar_t = sugar_b / sugar_se
        sugar_p = float(fit.pvalues["sugar_10kcal"])
        oil_b = float(fit.params["oil_100kcal"])
        oil_se = float(fit.bse["oil_100kcal"])
        oil_t = oil_b / oil_se
        oil_p = float(fit.pvalues["oil_100kcal"])

        rows.append({
            "Specification": label,
            "Sugar beta (pp/10kcal)": round(sugar_b, 4),
            "Sugar SE": round(sugar_se, 4),
            "Sugar t": round(sugar_t, 2),
            "Sugar p": round(sugar_p, 4),
            "Oil beta (pp/100kcal)": round(oil_b, 4),
            "Oil SE": round(oil_se, 4),
            "Oil t": round(oil_t, 2),
            "Oil p": round(oil_p, 4),
            "R2": round(fit.rsquared, 4),
            "N": int(fit.nobs),
            "Clustered SE": "Yes" if cluster else "No",
        })

    # Add detrended
    women_dt = women.copy()
    for iso3 in women_dt["iso3"].unique():
        mask = women_dt["iso3"] == iso3
        for col in ["obesity_prevalence_pct", "sugar_10kcal", "oil_100kcal"]:
            vals = women_dt.loc[mask, col].values
            t = np.arange(len(vals), dtype=float)
            slope, intercept = np.polyfit(t, vals, 1)
            women_dt.loc[mask, col] = vals - (slope * t + intercept)

    # No C(iso3) after detrending -- country means are near-zero (see F014)
    fit_dt = smf.ols("obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal", data=women_dt).fit(
        cov_type="cluster", cov_kwds={"groups": women_dt["iso3"]}
    )
    rows.append({
        "Specification": "Detrended",
        "Sugar beta (pp/10kcal)": round(float(fit_dt.params["sugar_10kcal"]), 4),
        "Sugar SE": round(float(fit_dt.bse["sugar_10kcal"]), 4),
        "Sugar t": round(float(fit_dt.params["sugar_10kcal"] / fit_dt.bse["sugar_10kcal"]), 2),
        "Sugar p": round(float(fit_dt.pvalues["sugar_10kcal"]), 4),
        "Oil beta (pp/100kcal)": round(float(fit_dt.params["oil_100kcal"]), 4),
        "Oil SE": round(float(fit_dt.bse["oil_100kcal"]), 4),
        "Oil t": round(float(fit_dt.params["oil_100kcal"] / fit_dt.bse["oil_100kcal"]), 2),
        "Oil p": round(float(fit_dt.pvalues["oil_100kcal"]), 4),
        "R2": round(fit_dt.rsquared, 4),
        "N": int(fit_dt.nobs),
        "Clustered SE": "Yes",
    })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "table2_fe_cascade.csv", index=False)
    print("  Table 2 saved")


def table_3_robustness() -> None:
    """Robustness tests for soybean oil supply (co-trending case study)."""
    results = json.loads((DERIVED / "robustness_failure_results.json").read_text())

    rows = [
        {"Test": "Cross-sectional r", "Statistic": results["cross_sectional"]["r"],
         "Verdict": results["cross_sectional"].get("verdict", "")},
        {"Test": "Country FE t_cl", "Statistic": results["country_fe"]["t_cl"],
         "Verdict": results["country_fe"].get("verdict", "")},
        {"Test": "TWFE t_cl", "Statistic": results["twfe"]["t_cl"],
         "Verdict": results["twfe"].get("verdict", "")},
        {"Test": "Lead-lag ratio", "Statistic": results["lead_lag"]["ratio"],
         "Verdict": results["lead_lag"]["diagnosis"]},
        {"Test": "Detrended t_cl", "Statistic": results["detrended"]["t_cl"],
         "Verdict": results["detrended"].get("verdict", "")},
        {"Test": "First diff r", "Statistic": results["first_differences"]["r"],
         "Verdict": results["first_differences"].get("verdict", "")},
        {"Test": "Granger r", "Statistic": results["granger"]["r"],
         "Verdict": results["granger"].get("verdict", "")},
        {"Test": "Oster delta", "Statistic": results["oster_bounds"]["delta"],
         "Verdict": "misleading (co-trending)"},
    ]

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "table3_robustness_failure.csv", index=False)
    print("  Table 3 saved")


def table_4_change_regressions() -> None:
    """Cross-country change regressions."""
    results = json.loads((DERIVED / "cross_country_change_results.json").read_text())
    reg = results["multiple_regression"]

    rows = []
    for var in ["d_urban", "d_sugar", "d_oil", "d_gdp", "init_obesity"]:
        if var in reg:
            rows.append({
                "Predictor": var,
                "beta": reg[var]["beta"],
                "t": reg[var]["t"],
                "p": reg[var]["p"],
            })

    df = pd.DataFrame(rows)
    df["R2"] = reg["r_squared"]
    df.to_csv(TABLES / "table4_change_regressions.csv", index=False)
    print("  Table 4 saved")


def main() -> None:
    print("=" * 70)
    print("10_tables.py: Generating all tables")
    print("=" * 70)

    panel = pd.read_csv(DERIVED / "analysis_panel.csv")

    table_1_descriptives(panel)
    table_2_fe_cascade(panel)
    table_3_robustness()
    table_4_change_regressions()

    print(f"\nAll tables saved to: {TABLES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
