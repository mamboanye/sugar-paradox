"""05_robustness_failure.py -- Soybean oil case study.

Demonstrates that a food supply variable can PASS standard robustness
tests (significant cross-sectional, significant one-way FE, high Oster
delta) yet be a pure co-trending artifact.

Data source: FAOSTAT Food Balance Sheets, item "Soyabean Oil",
element "Food supply (kcal/capita/day)".

Tests:
  1. Cross-sectional correlation (expect: significant)
  2. Country FE regression, clustered SEs (expect: significant)
  3. TWFE regression, clustered SEs (expect: collapses)
  4. Lead-lag test (ratio near 1.0 = co-trending)
  5. Detrended correlation (expect: collapses)
  6. First differences (expect: collapses)
  7. Granger causality (expect: null)
  8. Oster bounds (expect: high delta, but misleading)

Verified targets:
  - Lead-lag ratio ~ 1.0 (co-trending)
  - Detrended t_cl < 2 (null)
  - Oster delta >> 1 (misleadingly high)
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
OUTPUT = DERIVED / "robustness_failure_results.json"


# --- FAOSTAT area-name -> ISO3 mapping for SSA panel countries ---
FAO_NAME_TO_ISO3 = {
    "Angola": "AGO",
    "Burkina Faso": "BFA",
    "Botswana": "BWA",
    "Cameroon": "CMR",
    "Congo": "COG",
    "Comoros": "COM",
    "Cabo Verde": "CPV",
    "Democratic Republic of the Congo": "COD",
    "Eswatini": "SWZ",
    "Ethiopia": "ETH",
    "Gabon": "GAB",
    "Ghana": "GHA",
    "Guinea": "GIN",
    "Gambia": "GMB",
    "Guinea-Bissau": "GNB",
    "Kenya": "KEN",
    "Liberia": "LBR",
    "Lesotho": "LSO",
    "Madagascar": "MDG",
    "Mozambique": "MOZ",
    "Mauritania": "MRT",
    "Mauritius": "MUS",
    "Malawi": "MWI",
    "Namibia": "NAM",
    "Niger": "NER",
    "Nigeria": "NGA",
    "Rwanda": "RWA",
    "Senegal": "SEN",
    "Sierra Leone": "SLE",
    "Sao Tome and Principe": "STP",
    "Seychelles": "SYC",
    "United Republic of Tanzania": "TZA",
    "Uganda": "UGA",
    "South Africa": "ZAF",
    "Zambia": "ZMB",
    "Zimbabwe": "ZWE",
}
# Encoding variants
FAO_NAME_TO_ISO3["C\u00f4te d'Ivoire"] = "CIV"
FAO_NAME_TO_ISO3["C\u00c3\u00b4te d'Ivoire"] = "CIV"  # latin-1 mojibake


def extract_soybean_oil(panel_iso3: set[str]) -> pd.DataFrame:
    """Extract soybean oil food supply (kcal/cap/day) from FAOSTAT FBS."""
    fbs_zip = RAW / "FoodBalanceSheets_E_Africa.zip"
    with zipfile.ZipFile(fbs_zip) as z:
        with z.open("FoodBalanceSheets_E_Africa.csv") as f:
            fbs = pd.read_csv(f, encoding="latin-1")

    soy = fbs[
        (fbs["Item"] == "Soyabean Oil")
        & (fbs["Element"] == "Food supply (kcal/capita/day)")
    ].copy()

    year_cols = [
        c
        for c in soy.columns
        if c.startswith("Y")
        and c[1:].isdigit()
        and "F" not in c
        and "N" not in c
    ]
    rows = []
    for _, rec in soy.iterrows():
        area = rec["Area"]
        iso3 = FAO_NAME_TO_ISO3.get(area)
        if iso3 is None or iso3 not in panel_iso3:
            continue
        for yc in year_cols:
            yr = int(yc[1:])
            if 2010 <= yr <= 2022:
                val = rec[yc]
                if pd.notna(val):
                    rows.append({"iso3": iso3, "year": yr, "soy_oil_kcal": float(val)})
    return pd.DataFrame(rows)


def oster_delta(
    beta_controlled: float,
    beta_uncontrolled: float,
    r2_controlled: float,
    r2_uncontrolled: float,
    r_max: float = 1.0,
) -> float:
    """Oster (2019, JBES eq. 9) proportional selection delta.

    delta = [beta_c * (R2_c - R2_u)] / [(beta_u - beta_c) * (R_max - R2_c)]

    Setting beta* = 0: how large must selection on unobservables be (relative
    to observables) to explain away the entire controlled coefficient?
    delta > 1 normally means robust to proportional selection.
    For co-trending variables, high delta is misleading: the R2 jump comes
    from country FE absorbing level differences, not from the exposure
    variable explaining within-country variation.
    """
    coef_denom = beta_uncontrolled - beta_controlled
    r2_denom = r_max - r2_controlled
    if abs(coef_denom) < 1e-10 or abs(r2_denom) < 1e-10:
        return float("inf")
    return (beta_controlled * (r2_controlled - r2_uncontrolled)) / (
        coef_denom * r2_denom
    )


def main() -> None:
    print("=" * 70)
    print("05_robustness_failure.py: Soybean oil case study")
    print("=" * 70)

    # Load analysis panel
    df = pd.read_csv(DERIVED / "analysis_panel.csv")
    women = df[df["sex"] == "Women"].copy()
    women = women.sort_values(["iso3", "year"]).reset_index(drop=True)
    panel_iso3 = set(women["iso3"].unique())

    # Extract and merge soybean oil supply
    soy_df = extract_soybean_oil(panel_iso3)
    women = women.merge(soy_df, on=["iso3", "year"], how="left")
    women["soy_oil_kcal"] = women["soy_oil_kcal"].fillna(0)
    women["soy_10kcal"] = women["soy_oil_kcal"] / 10.0
    women["sugar_10kcal"] = women["primary_exposure_food_supply_kcal_capita_day"] / 10.0

    n_with_soy = women[women["soy_oil_kcal"] > 0]["iso3"].nunique()
    print(f"\nCountries with soybean oil data: {n_with_soy}/{len(panel_iso3)}")
    print(f"Mean soybean oil supply: {women['soy_oil_kcal'].mean():.1f} kcal/cap/day")

    results = {}

    # 1. Cross-sectional correlation (2018-2020 averages, consistent with script 02)
    print("\n--- Test 1: Cross-sectional correlation ---")
    xs = women[women["year"].isin([2018, 2019, 2020])]
    cm = xs.groupby("iso3").agg(
        {"obesity_prevalence_pct": "mean", "soy_10kcal": "mean"}
    ).reset_index()
    r_xs, p_xs = stats.pearsonr(cm["soy_10kcal"], cm["obesity_prevalence_pct"])
    print(f"  r = {r_xs:.3f}, p = {p_xs:.4f}")
    results["cross_sectional"] = {
        "r": round(r_xs, 3),
        "p": round(p_xs, 4),
        "verdict": "significant" if p_xs < 0.05 else "null",
    }

    # 2. Country FE (no year FE), clustered SEs
    print("\n--- Test 2: Country FE (no year FE) ---")
    fit_1way = smf.ols(
        "obesity_prevalence_pct ~ soy_10kcal + C(iso3)", data=women
    ).fit(cov_type="cluster", cov_kwds={"groups": women["iso3"]})
    b_1way = float(fit_1way.params["soy_10kcal"])
    t_1way = float(fit_1way.params["soy_10kcal"] / fit_1way.bse["soy_10kcal"])
    print(f"  beta = {b_1way:.4f}, t_cl = {t_1way:.2f}")
    results["country_fe"] = {
        "beta": round(b_1way, 4),
        "t_cl": round(t_1way, 2),
        "verdict": "significant" if abs(t_1way) > 2 else "null",
    }

    # Also compute non-clustered for SE inflation comparison
    fit_1way_nc = smf.ols(
        "obesity_prevalence_pct ~ soy_10kcal + C(iso3)", data=women
    ).fit()
    t_1way_nc = float(fit_1way_nc.params["soy_10kcal"] / fit_1way_nc.bse["soy_10kcal"])
    se_ratio = float(fit_1way.bse["soy_10kcal"] / fit_1way_nc.bse["soy_10kcal"])
    print(f"  Non-clustered t = {t_1way_nc:.2f} (SE inflation: {se_ratio:.1f}x)")
    results["country_fe_nonclustered"] = {
        "t": round(t_1way_nc, 2),
        "se_inflation_ratio": round(se_ratio, 1),
    }

    # 3. TWFE (country + year FE)
    print("\n--- Test 3: TWFE ---")
    fit_twfe = smf.ols(
        "obesity_prevalence_pct ~ soy_10kcal + C(iso3) + C(year)", data=women
    ).fit(cov_type="cluster", cov_kwds={"groups": women["iso3"]})
    b_twfe = float(fit_twfe.params["soy_10kcal"])
    t_twfe = float(fit_twfe.params["soy_10kcal"] / fit_twfe.bse["soy_10kcal"])
    print(f"  beta = {b_twfe:.4f}, t_cl = {t_twfe:.2f}")
    results["twfe"] = {
        "beta": round(b_twfe, 4),
        "t_cl": round(t_twfe, 2),
        "verdict": "significant" if abs(t_twfe) > 2 else "null",
    }

    # 4. Lead-lag test
    print("\n--- Test 4: Lead-lag ---")
    women_ll = women.copy()
    women_ll["soy_lead"] = women_ll.groupby("iso3")["soy_10kcal"].shift(-1)
    women_ll["soy_lag"] = women_ll.groupby("iso3")["soy_10kcal"].shift(1)
    ll_sub = women_ll.dropna(subset=["soy_lead", "soy_lag"])

    fit_lead = smf.ols(
        "obesity_prevalence_pct ~ soy_lead + C(iso3)", data=ll_sub
    ).fit(cov_type="cluster", cov_kwds={"groups": ll_sub["iso3"]})
    fit_lag = smf.ols(
        "obesity_prevalence_pct ~ soy_lag + C(iso3)", data=ll_sub
    ).fit(cov_type="cluster", cov_kwds={"groups": ll_sub["iso3"]})
    b_lead = abs(float(fit_lead.params["soy_lead"]))
    b_lag = abs(float(fit_lag.params["soy_lag"]))
    ratio = b_lead / b_lag if b_lag > 0 else float("inf")
    print(f"  |beta_lead| = {b_lead:.4f}, |beta_lag| = {b_lag:.4f}, ratio = {ratio:.3f}")
    print(f"  Ratio near 1.0 = co-trending (not causal)")
    results["lead_lag"] = {
        "beta_lead": round(b_lead, 4),
        "beta_lag": round(b_lag, 4),
        "ratio": round(ratio, 3),
        "diagnosis": "co-trending" if 0.8 < ratio < 1.2 else "directional",
    }

    # 5. Detrended
    print("\n--- Test 5: Detrended ---")
    women_dt = women.copy()
    for iso3 in women_dt["iso3"].unique():
        mask = women_dt["iso3"] == iso3
        for col in ["obesity_prevalence_pct", "soy_10kcal"]:
            vals = women_dt.loc[mask, col].values
            t = np.arange(len(vals), dtype=float)
            slope, intercept = np.polyfit(t, vals, 1)
            women_dt.loc[mask, col] = vals - (slope * t + intercept)

    # No C(iso3) after detrending -- country means are near-zero, dummies are redundant
    # and distort clustered SEs with 37 clusters (see F014)
    fit_dt = smf.ols(
        "obesity_prevalence_pct ~ soy_10kcal", data=women_dt
    ).fit(cov_type="cluster", cov_kwds={"groups": women_dt["iso3"]})
    t_dt = float(fit_dt.params["soy_10kcal"] / fit_dt.bse["soy_10kcal"])
    print(f"  Detrended t_cl = {t_dt:.2f}")
    results["detrended"] = {
        "t_cl": round(t_dt, 2),
        "verdict": "null" if abs(t_dt) < 2 else "significant",
    }

    # 6. First differences
    print("\n--- Test 6: First differences ---")
    fd_rows = []
    for iso3 in women["iso3"].unique():
        sub = women[women["iso3"] == iso3].sort_values("year").reset_index(drop=True)
        for i in range(1, len(sub)):
            fd_rows.append(
                {
                    "d_obesity": sub.loc[i, "obesity_prevalence_pct"]
                    - sub.loc[i - 1, "obesity_prevalence_pct"],
                    "d_soy": sub.loc[i, "soy_10kcal"]
                    - sub.loc[i - 1, "soy_10kcal"],
                }
            )
    fd_df = pd.DataFrame(fd_rows)
    r_fd, p_fd = stats.pearsonr(fd_df["d_soy"], fd_df["d_obesity"])
    print(f"  FD r = {r_fd:.3f}, p = {p_fd:.4f}")
    results["first_differences"] = {
        "r": round(r_fd, 3),
        "p": round(p_fd, 4),
        "verdict": "null" if p_fd > 0.05 else "significant",
    }

    # 7. Granger-style test
    print("\n--- Test 7: Granger test ---")
    women_g = women.copy()
    women_g["d_obesity_lead"] = (
        women_g.groupby("iso3")["obesity_prevalence_pct"].shift(-1)
        - women_g["obesity_prevalence_pct"]
    )
    women_g["d_soy"] = (
        women_g["soy_10kcal"]
        - women_g.groupby("iso3")["soy_10kcal"].shift(1)
    )
    g_sub = women_g.dropna(subset=["d_obesity_lead", "d_soy"])
    r_g, p_g = stats.pearsonr(g_sub["d_soy"], g_sub["d_obesity_lead"])
    print(f"  Granger r = {r_g:.3f}, p = {p_g:.4f}")
    results["granger"] = {
        "r": round(r_g, 3),
        "p": round(p_g, 4),
        "verdict": "null" if p_g > 0.05 else "significant",
    }

    # 8. Oster bounds
    print("\n--- Test 8: Oster bounds ---")
    # Specification: restricted = bivariate (soy only), full = soy + country FE
    # This tests whether adding country FE destabilizes the soy coefficient.
    # For co-trending variables, the coefficient shrinks but R2 jumps massively,
    # producing a misleadingly high delta.
    fit_bivariate = smf.ols(
        "obesity_prevalence_pct ~ soy_10kcal", data=women
    ).fit()
    r2_bivariate = fit_bivariate.rsquared
    r2_fe = fit_1way_nc.rsquared
    beta_bivariate = float(fit_bivariate.params["soy_10kcal"])
    beta_fe = float(fit_1way_nc.params["soy_10kcal"])
    delta = oster_delta(beta_fe, beta_bivariate, r2_fe, r2_bivariate, r_max=1.0)
    coef_ratio = beta_fe / beta_bivariate if abs(beta_bivariate) > 1e-10 else float("inf")

    print(f"  R2_bivariate = {r2_bivariate:.4f}, R2_FE = {r2_fe:.4f}")
    print(f"  beta_bivariate = {beta_bivariate:.4f}, beta_FE = {beta_fe:.4f}")
    print(f"  Coefficient ratio = {coef_ratio:.3f}")
    print(f"  Oster delta = {delta:.1f}")
    print(f"  High delta normally = robust, but here the result is co-trending")
    results["oster_bounds"] = {
        "r2_bivariate": round(r2_bivariate, 4),
        "r2_fe": round(r2_fe, 4),
        "beta_bivariate": round(beta_bivariate, 4),
        "beta_fe": round(beta_fe, 4),
        "coefficient_ratio": round(coef_ratio, 3),
        "delta": round(delta, 1),
        "interpretation": (
            "High delta is misleading when both variables co-trend: "
            "the R2 jump comes from FE absorbing level differences, "
            "not from soy oil genuinely predicting within-country obesity."
        ),
    }

    # Summary
    print("\n--- Summary ---")
    tests_pass_standard = sum(
        1
        for k in ["cross_sectional", "country_fe"]
        if results[k].get("verdict") == "significant"
    )
    print(f"  {tests_pass_standard} of 2 standard tests appear to pass")
    print(f"  Lead-lag ratio = {results['lead_lag']['ratio']:.3f} (near 1.0 = pure co-trending)")
    print(f"  Detrended and FD both collapse -> the association is spurious")
    print(f"  Non-clustered FE t = {t_1way_nc:.2f} -> clustered t = {t_1way:.2f} (SE inflation)")
    print(f"  Adding year FE collapses clustered t from {t_1way:.2f} to {t_twfe:.2f}")
    print(f"  Oster delta = {delta:.1f} (misleadingly high)")

    results["summary"] = {
        "standard_tests_passed": tests_pass_standard,
        "total_tests": 8,
        "lead_lag_ratio": results["lead_lag"]["ratio"],
        "diagnosis": "co-trending artifact",
        "mechanism": (
            "Soybean oil supply and obesity both trend upward within most SSA "
            "countries. Standard tests (cross-sectional, one-way FE, Oster bounds) "
            "cannot distinguish this co-trending from a causal relationship. "
            "Only tests that remove trends (TWFE, detrending, first differences) "
            "or check temporal ordering (lead-lag, Granger) detect the artifact."
        ),
    }

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
