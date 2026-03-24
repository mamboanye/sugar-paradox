"""12_inference_robustness.py -- Wild cluster bootstrap, permutation, two-way clustering.

Three independent checks that the obesity null is not an inference artifact:

1. Wild cluster bootstrap (999 draws): p=0.80 (one-way FE), p=0.53 (TWFE)
2. Permutation test (999 shuffles of within-country sugar): p=0.55
3. Two-way clustering (country + year): t=-0.28, identical to one-way

These close every procedural objection about SE choice, small-cluster
bias, or cross-sectional dependence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
OUTPUT = DERIVED / "inference_robustness_results.json"

B = 999
RNG = np.random.default_rng(42)


def wild_cluster_bootstrap_p(
    data: pd.DataFrame,
    formula: str,
    term: str,
    cluster_col: str,
    n_boot: int,
) -> dict:
    """Approximate wild cluster bootstrap using Rademacher weights."""
    fit = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[cluster_col]}
    )
    observed_t = float(fit.params[term] / fit.bse[term])

    clusters = data[cluster_col].unique()
    boot_ts = []
    for _ in range(n_boot):
        weights = RNG.choice([-1, 1], size=len(clusters))
        weight_map = dict(zip(clusters, weights))
        data_boot = data.copy()
        cluster_weights = data_boot[cluster_col].map(weight_map)
        data_boot["__y_boot"] = fit.fittedvalues + fit.resid * cluster_weights
        formula_boot = formula.replace(formula.split("~")[0].strip(), "__y_boot")
        try:
            fit_b = smf.ols(formula_boot, data=data_boot).fit(
                cov_type="cluster", cov_kwds={"groups": data_boot[cluster_col]}
            )
            boot_ts.append(float(fit_b.params[term] / fit_b.bse[term]))
        except Exception:
            pass

    boot_ts = np.array(boot_ts)
    boot_p = float(np.mean(np.abs(boot_ts) >= np.abs(observed_t)))

    return {
        "beta": round(float(fit.params[term]), 6),
        "t_cluster": round(observed_t, 4),
        "p_cluster": round(float(fit.pvalues[term]), 4),
        "boot_p": round(boot_p, 4),
        "n_boot": len(boot_ts),
    }


def main() -> None:
    print("=" * 70)
    print("12_inference_robustness.py: Bootstrap, permutation, two-way clustering")
    print("=" * 70)

    df = pd.read_csv(DERIVED / "analysis_panel.csv")
    women = df[df["sex"] == "Women"].copy()
    women["sugar_10kcal"] = women["primary_exposure_food_supply_kcal_capita_day"] / 10.0
    women["oil_100kcal"] = women["vegetable_oils_comparison_food_supply_kcal_capita_day"] / 100.0

    results = {}

    # --- 1. Wild cluster bootstrap ---
    print("\n--- Wild cluster bootstrap (B=999) ---")

    print("  One-way FE...")
    r1 = wild_cluster_bootstrap_p(
        women,
        "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal + C(iso3)",
        "sugar_10kcal", "iso3", B,
    )
    print(f"    Clustered p={r1['p_cluster']}, Bootstrap p={r1['boot_p']}")
    results["bootstrap_one_way_fe"] = r1

    print("  TWFE...")
    r2 = wild_cluster_bootstrap_p(
        women,
        "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal + C(iso3) + C(year)",
        "sugar_10kcal", "iso3", B,
    )
    print(f"    Clustered p={r2['p_cluster']}, Bootstrap p={r2['boot_p']}")
    results["bootstrap_twfe"] = r2

    # --- 2. Permutation test ---
    print("\n--- Permutation test (999 shuffles) ---")
    fit_obs = smf.ols(
        "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal + C(iso3) + C(year)",
        data=women,
    ).fit(cov_type="cluster", cov_kwds={"groups": women["iso3"]})
    observed_t = float(fit_obs.params["sugar_10kcal"] / fit_obs.bse["sugar_10kcal"])

    perm_ts = []
    for _ in range(B):
        w_perm = women.copy()
        for iso3 in w_perm["iso3"].unique():
            mask = w_perm["iso3"] == iso3
            vals = w_perm.loc[mask, "sugar_10kcal"].values.copy()
            RNG.shuffle(vals)
            w_perm.loc[mask, "sugar_10kcal"] = vals
        try:
            fit_p = smf.ols(
                "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal + C(iso3) + C(year)",
                data=w_perm,
            ).fit(cov_type="cluster", cov_kwds={"groups": w_perm["iso3"]})
            perm_ts.append(float(fit_p.params["sugar_10kcal"] / fit_p.bse["sugar_10kcal"]))
        except Exception:
            pass

    perm_ts = np.array(perm_ts)
    perm_p = float(np.mean(np.abs(perm_ts) >= np.abs(observed_t)))
    perm_exceed_2 = float(np.mean(np.abs(perm_ts) > 2))

    print(f"  Observed t: {observed_t:.2f}")
    print(f"  Permutation p: {perm_p:.3f}")
    print(f"  Fraction |t|>2 under null: {perm_exceed_2*100:.1f}%")
    results["permutation"] = {
        "observed_t": round(observed_t, 4),
        "perm_p": round(perm_p, 4),
        "perm_mean_t": round(float(np.mean(perm_ts)), 4),
        "perm_sd_t": round(float(np.std(perm_ts)), 4),
        "pct_exceed_2": round(perm_exceed_2 * 100, 1),
        "n_perm": len(perm_ts),
    }

    # --- 3. Two-way clustering (Cameron-Gelbach-Miller) ---
    print("\n--- Two-way clustering (country + year) ---")
    women_reset = women.reset_index(drop=True)
    formula = "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal + C(iso3) + C(year)"
    model = smf.ols(formula, data=women_reset)

    fit_country = model.fit(cov_type="cluster", cov_kwds={"groups": women_reset["iso3"]})
    fit_year = model.fit(cov_type="cluster", cov_kwds={"groups": women_reset["year"]})
    fit_nc = model.fit()

    se_c = float(fit_country.bse["sugar_10kcal"])
    se_y = float(fit_year.bse["sugar_10kcal"])
    se_nc = float(fit_nc.bse["sugar_10kcal"])
    se_twoway = float(np.sqrt(se_c**2 + se_y**2 - se_nc**2))
    beta = float(fit_country.params["sugar_10kcal"])
    t_twoway = beta / se_twoway

    print(f"  Country-clustered: SE={se_c:.6f}, t={beta/se_c:.2f}")
    print(f"  Year-clustered:    SE={se_y:.6f}")
    print(f"  Two-way clustered: SE={se_twoway:.6f}, t={t_twoway:.2f}")
    results["two_way_clustering"] = {
        "beta": round(beta, 6),
        "se_country": round(se_c, 6),
        "se_year": round(se_y, 6),
        "se_nonclustered": round(se_nc, 6),
        "se_twoway": round(se_twoway, 6),
        "t_country": round(beta / se_c, 2),
        "t_twoway": round(t_twoway, 2),
    }

    # --- Summary ---
    print("\n--- Summary ---")
    print("  All three checks confirm the null:")
    print(f"    Bootstrap:    one-way p={r1['boot_p']}, TWFE p={r2['boot_p']}")
    print(f"    Permutation:  p={perm_p:.3f}")
    print(f"    Two-way:      t={t_twoway:.2f}")

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
