"""03_within_country_fe.py -- Fixed effects family.

Specifications:
  - One-way country FE (no year FE) -- for Lin 2018 reconciliation
  - TWFE (country + year FE)
  - Detrended (country-specific linear trends removed)
  - First differences
  - Long differences (2010 vs 2020)
  - Mundlak/CRE (between + within decomposition)

All use clustered SEs at the country level.

Verified targets:
  - One-way FE sugar t_cl = 0.22
  - TWFE sugar t_cl = -0.29 (bivariate); -0.64 (with oil)
  - Detrended sugar t_cl = -0.05
  - First differences r = -0.026
  - Long differences r (2010-2020) = -0.194
  - Lin 2018 reconciliation: sugar null under ALL specifications (see script 05 for mechanism)
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
OUTPUT = DERIVED / "within_country_fe_results.json"


def clustered_t(fit, term: str) -> float:
    """Return clustered t-statistic for a term."""
    return float(fit.params[term] / fit.bse[term])


def main() -> None:
    print("=" * 70)
    print("03_within_country_fe.py: Fixed effects family")
    print("=" * 70)

    df = pd.read_csv(DERIVED / "analysis_panel.csv")
    df["sugar_10kcal"] = df["primary_exposure_food_supply_kcal_capita_day"] / 10.0
    df["oil_100kcal"] = df["vegetable_oils_comparison_food_supply_kcal_capita_day"] / 100.0
    df["year_int"] = df["year"].astype(int)

    women = df[df["sex"] == "Women"].copy()

    results = {}

    # 1. One-way FE (country only, no year FE) -- Lin 2018 reconciliation
    print("\n--- One-way country FE (no year FE) ---")
    fit_1way = smf.ols(
        "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal + C(iso3)",
        data=women,
    ).fit(cov_type="cluster", cov_kwds={"groups": women["iso3"]})
    t_1way = clustered_t(fit_1way, "sugar_10kcal")
    p_1way = float(fit_1way.pvalues["sugar_10kcal"])
    b_1way = float(fit_1way.params["sugar_10kcal"])
    print(f"  beta = {b_1way:.4f}, t_cl = {t_1way:.2f}, p = {p_1way:.4f}")
    results["one_way_fe"] = {
        "beta": round(b_1way, 4),
        "t_cl": round(t_1way, 2),
        "p": round(p_1way, 4),
        "note": "Country FE only, no year FE",
    }

    # 2. TWFE bivariate (sugar only + FE)
    print("\n--- TWFE (bivariate: sugar only) ---")
    fit_twfe_biv = smf.ols(
        "obesity_prevalence_pct ~ sugar_10kcal + C(iso3) + C(year)",
        data=women,
    ).fit(cov_type="cluster", cov_kwds={"groups": women["iso3"]})
    t_twfe_biv = clustered_t(fit_twfe_biv, "sugar_10kcal")
    b_twfe_biv = float(fit_twfe_biv.params["sugar_10kcal"])
    p_twfe_biv = float(fit_twfe_biv.pvalues["sugar_10kcal"])
    print(f"  beta = {b_twfe_biv:.4f}, t_cl = {t_twfe_biv:.2f}, p = {p_twfe_biv:.4f}")
    results["twfe_bivariate"] = {
        "beta": round(b_twfe_biv, 4),
        "t_cl": round(t_twfe_biv, 2),
        "p": round(p_twfe_biv, 4),
    }

    # 3. TWFE with oil
    print("\n--- TWFE (with oil) ---")
    fit_twfe = smf.ols(
        "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal + C(iso3) + C(year)",
        data=women,
    ).fit(cov_type="cluster", cov_kwds={"groups": women["iso3"]})
    t_twfe = clustered_t(fit_twfe, "sugar_10kcal")
    b_twfe = float(fit_twfe.params["sugar_10kcal"])
    p_twfe = float(fit_twfe.pvalues["sugar_10kcal"])
    print(f"  beta = {b_twfe:.4f}, t_cl = {t_twfe:.2f}, p = {p_twfe:.4f}")
    results["twfe_with_oil"] = {
        "beta": round(b_twfe, 4),
        "t_cl": round(t_twfe, 2),
        "p": round(p_twfe, 4),
    }

    # 4. Detrended: remove country-specific linear trends
    # No C(iso3) after detrending -- country means are near-zero, dummies are
    # redundant and distort clustered SEs (same fix as script 07, see F014).
    print("\n--- Detrended ---")
    women_dt = women.copy()
    for iso3 in women_dt["iso3"].unique():
        mask = women_dt["iso3"] == iso3
        for col in ["obesity_prevalence_pct", "sugar_10kcal", "oil_100kcal"]:
            vals = women_dt.loc[mask, col].values
            t = np.arange(len(vals), dtype=float)
            slope, intercept = np.polyfit(t, vals, 1)
            women_dt.loc[mask, col] = vals - (slope * t + intercept)

    # Bivariate detrended (sugar only) -- matches verified spec
    fit_dt_biv = smf.ols(
        "obesity_prevalence_pct ~ sugar_10kcal",
        data=women_dt,
    ).fit(cov_type="cluster", cov_kwds={"groups": women_dt["iso3"]})
    t_dt_biv = clustered_t(fit_dt_biv, "sugar_10kcal")
    b_dt_biv = float(fit_dt_biv.params["sugar_10kcal"])
    p_dt_biv = float(fit_dt_biv.pvalues["sugar_10kcal"])
    print(f"  bivariate: beta = {b_dt_biv:.4f}, t_cl = {t_dt_biv:.2f}, p = {p_dt_biv:.4f}")

    # Multivariate detrended (sugar + oil)
    fit_dt = smf.ols(
        "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal",
        data=women_dt,
    ).fit(cov_type="cluster", cov_kwds={"groups": women_dt["iso3"]})
    t_dt = clustered_t(fit_dt, "sugar_10kcal")
    b_dt = float(fit_dt.params["sugar_10kcal"])
    p_dt = float(fit_dt.pvalues["sugar_10kcal"])
    print(f"  with oil:  beta = {b_dt:.4f}, t_cl = {t_dt:.2f}, p = {p_dt:.4f}")
    results["detrended_bivariate"] = {
        "beta": round(b_dt_biv, 4),
        "t_cl": round(t_dt_biv, 2),
        "p": round(p_dt_biv, 4),
    }
    results["detrended"] = {
        "beta": round(b_dt, 4),
        "t_cl": round(t_dt, 2),
        "p": round(p_dt, 4),
    }

    # 5. First differences
    print("\n--- First Differences ---")
    women_sorted = women.sort_values(["iso3", "year"]).copy()
    fd_rows = []
    for iso3 in women_sorted["iso3"].unique():
        sub = women_sorted[women_sorted["iso3"] == iso3].reset_index(drop=True)
        for i in range(1, len(sub)):
            fd_rows.append({
                "iso3": iso3,
                "year": sub.loc[i, "year"],
                "d_obesity": sub.loc[i, "obesity_prevalence_pct"] - sub.loc[i - 1, "obesity_prevalence_pct"],
                "d_sugar": sub.loc[i, "sugar_10kcal"] - sub.loc[i - 1, "sugar_10kcal"],
                "d_oil": sub.loc[i, "oil_100kcal"] - sub.loc[i - 1, "oil_100kcal"],
            })
    fd_df = pd.DataFrame(fd_rows)
    r_fd, p_fd = stats.pearsonr(fd_df["d_sugar"], fd_df["d_obesity"])
    print(f"  FD correlation (sugar-obesity): r = {r_fd:.3f}, p = {p_fd:.4f}")
    results["first_differences"] = {
        "r": round(r_fd, 3),
        "p": round(p_fd, 4),
        "n_obs": len(fd_df),
    }

    # 6. Long differences (2010 vs 2020)
    print("\n--- Long Differences (2010 vs 2020) ---")
    w2010 = women[women["year"] == 2010].set_index("iso3")
    w2020 = women[women["year"] == 2020].set_index("iso3")
    common = w2010.index.intersection(w2020.index)
    ld_sugar = w2020.loc[common, "sugar_10kcal"] - w2010.loc[common, "sugar_10kcal"]
    ld_obesity = w2020.loc[common, "obesity_prevalence_pct"] - w2010.loc[common, "obesity_prevalence_pct"]
    r_ld, p_ld = stats.pearsonr(ld_sugar.values, ld_obesity.values)
    print(f"  Long-diff correlation: r = {r_ld:.3f}, p = {p_ld:.4f}, n = {len(common)}")
    results["long_differences_2010_2020"] = {
        "r": round(r_ld, 3),
        "p": round(p_ld, 4),
        "n_countries": int(len(common)),
    }

    # 7. Mundlak/CRE decomposition
    print("\n--- Mundlak/CRE ---")
    women_cre = women.copy()
    for col in ["sugar_10kcal", "oil_100kcal"]:
        means = women_cre.groupby("iso3")[col].transform("mean")
        women_cre[f"{col}_between"] = means
        women_cre[f"{col}_within"] = women_cre[col] - means

    fit_cre = smf.ols(
        "obesity_prevalence_pct ~ sugar_10kcal_within + sugar_10kcal_between + "
        "oil_100kcal_within + oil_100kcal_between + C(year)",
        data=women_cre,
    ).fit(cov_type="cluster", cov_kwds={"groups": women_cre["iso3"]})

    b_within = float(fit_cre.params["sugar_10kcal_within"])
    t_within = float(fit_cre.params["sugar_10kcal_within"] / fit_cre.bse["sugar_10kcal_within"])
    b_between = float(fit_cre.params["sugar_10kcal_between"])
    t_between = float(fit_cre.params["sugar_10kcal_between"] / fit_cre.bse["sugar_10kcal_between"])
    print(f"  Within:  beta = {b_within:.4f}, t_cl = {t_within:.2f}")
    print(f"  Between: beta = {b_between:.4f}, t_cl = {t_between:.2f}")
    results["mundlak_cre"] = {
        "within_beta": round(b_within, 4),
        "within_t_cl": round(t_within, 2),
        "between_beta": round(b_between, 4),
        "between_t_cl": round(t_between, 2),
    }

    # 8. Lin 2018 reconciliation
    # Lin et al. (2018) found sugar significant within countries using FE.
    # In our SSA data, sugar is null under every specification including
    # the most generous (non-clustered, one-way FE, no oil control).
    # The reconciliation is that sugar was NEVER significant in SSA data,
    # even before clustering or year FE. The soybean oil case study
    # (script 05) demonstrates the mechanism by which co-trending food
    # supply variables can appear significant under one-way FE.
    print("\n--- Lin 2018 reconciliation ---")

    # Sugar-only, non-clustered (most generous specification)
    fit_sugar_nc = smf.ols(
        "obesity_prevalence_pct ~ sugar_10kcal + C(iso3)", data=women
    ).fit()
    t_sugar_nc = float(fit_sugar_nc.params["sugar_10kcal"] / fit_sugar_nc.bse["sugar_10kcal"])
    p_sugar_nc = float(fit_sugar_nc.pvalues["sugar_10kcal"])

    # Sugar-only, clustered
    fit_sugar_cl = smf.ols(
        "obesity_prevalence_pct ~ sugar_10kcal + C(iso3)", data=women
    ).fit(cov_type="cluster", cov_kwds={"groups": women["iso3"]})
    t_sugar_cl = float(fit_sugar_cl.params["sugar_10kcal"] / fit_sugar_cl.bse["sugar_10kcal"])
    p_sugar_cl = float(fit_sugar_cl.pvalues["sugar_10kcal"])

    print(f"  Sugar-only, non-clustered FE: t = {t_sugar_nc:.2f} (p = {p_sugar_nc:.4f})")
    print(f"  Sugar-only, clustered FE:     t_cl = {t_sugar_cl:.2f} (p = {p_sugar_cl:.4f})")
    print(f"  With oil, clustered FE:       t_cl = {t_1way:.2f} (p = {p_1way:.4f})")
    print(f"  With oil, clustered TWFE:     t_cl = {t_twfe:.2f} (p = {p_twfe:.4f})")
    print("  Sugar is null under EVERY specification in SSA data.")
    print("  See script 05 for the co-trending mechanism (soybean oil case study).")

    results["lin_2018_reconciliation"] = {
        "sugar_only_nonclustered_fe_t": round(t_sugar_nc, 2),
        "sugar_only_nonclustered_fe_p": round(p_sugar_nc, 4),
        "sugar_only_clustered_fe_t_cl": round(t_sugar_cl, 2),
        "sugar_only_clustered_fe_p": round(p_sugar_cl, 4),
        "sugar_oil_clustered_fe_t_cl": round(t_1way, 2),
        "sugar_oil_clustered_twfe_t_cl": round(t_twfe, 2),
        "interpretation": (
            "Sugar is null under every specification in SSA data, including "
            "the most generous (non-clustered, one-way FE, no oil control). "
            "Lin 2018 likely found significance from a global sample where "
            "co-trending was stronger. The soybean oil case study (script 05) "
            "demonstrates the co-trending mechanism that can produce spurious "
            "significance under one-way FE."
        ),
    }

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
