"""09_figures.py -- All paper figures.

Figure 1: Cross-sectional sugar vs obesity (partial correlation scatter)
Figure 2: FE specification cascade (bar chart of t-statistics)
Figure 3: Trend R2 distribution (global + SSA highlighted)
Figure 4: Soybean oil case study (pass/fail diagnostic panel)
Figure 5: GDP positive control (cross-sectional vs detrended comparison)
Figure 6: Variance decomposition waterfall (entity, year, residual)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Shared styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def figure_1_cross_sectional(panel: pd.DataFrame) -> None:
    """Scatter: 2018-2020 avg sugar supply vs obesity, with partial-r annotation.

    Panel A: raw bivariate correlation (2018-2020 means).
    Panel B: partial correlation residualized on GDP + urbanization + oil.
    Both must match the primary cross-sectional specification (r=0.496).
    """
    import statsmodels.formula.api as smf

    women = panel[panel["sex"] == "Women"]
    cs = women[women["year"].isin([2018, 2019, 2020])].groupby("iso3").agg(
        obesity_prevalence_pct=("obesity_prevalence_pct", "mean"),
        sugar_kcal=("primary_exposure_food_supply_kcal_capita_day", "mean"),
        oil_kcal=("vegetable_oils_comparison_food_supply_kcal_capita_day", "mean"),
        log_gdp=("log_gdp_per_capita_constant_2015_usd", "mean"),
        urban_pct=("urban_population_pct", "mean"),
    ).dropna()

    x = cs["sugar_kcal"].values
    y = cs["obesity_prevalence_pct"].values

    # Residualize both on GDP + urbanization + oil (matching the primary spec)
    controls = "~ log_gdp + urban_pct + oil_kcal"
    x_resid = smf.ols(f"sugar_kcal {controls}", data=cs).fit().resid.values
    y_resid = smf.ols(f"obesity_prevalence_pct {controls}", data=cs).fit().resid.values
    r_partial, p_partial = stats.pearsonr(x_resid, y_resid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Raw scatter
    ax = axes[0]
    ax.scatter(x, y, s=40, alpha=0.7, edgecolors="k", linewidths=0.5, c="#2196F3")
    m, b = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_fit, m * x_fit + b, "k--", alpha=0.5)
    r_raw, p_raw = stats.pearsonr(x, y)
    ax.set_xlabel("Sugar supply (kcal/capita/day)")
    ax.set_ylabel("Obesity prevalence (%)")
    ax.set_title("A. Raw correlation (2018-2020 avg)")
    ax.text(0.05, 0.95, f"r = {r_raw:.3f}\np = {p_raw:.4f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Partial scatter (residualized on GDP + urbanization + oil)
    ax = axes[1]
    ax.scatter(x_resid, y_resid, s=40, alpha=0.7, edgecolors="k", linewidths=0.5, c="#FF9800")
    m2, b2 = np.polyfit(x_resid, y_resid, 1)
    xr_fit = np.linspace(x_resid.min(), x_resid.max(), 100)
    ax.plot(xr_fit, m2 * xr_fit + b2, "k--", alpha=0.5)
    ax.set_xlabel("Sugar supply (residualized)")
    ax.set_ylabel("Obesity prevalence (residualized)")
    ax.set_title("B. Partial correlation (| GDP, urban, oil)")
    ax.text(0.05, 0.95, f"partial r = {r_partial:.3f}\np = {p_partial:.4f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Figure 1: Cross-sectional sugar-obesity relationship (37 SSA countries)", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_cross_sectional.png")
    plt.close(fig)
    print("  Figure 1 saved")


def figure_2_fe_cascade(panel: pd.DataFrame) -> None:
    """Bar chart: t-statistics across FE specifications."""
    import statsmodels.formula.api as smf

    women = panel[panel["sex"] == "Women"].copy()
    women["sugar_10kcal"] = women["primary_exposure_food_supply_kcal_capita_day"] / 10.0
    women["oil_100kcal"] = women["vegetable_oils_comparison_food_supply_kcal_capita_day"] / 100.0

    specs = [
        ("Cross-sectional\n(OLS)", "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal", False, None),
        ("Country FE\n(no year FE)", "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal + C(iso3)", True, "iso3"),
        ("TWFE\n(country + year)", "obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal + C(iso3) + C(year)", True, "iso3"),
    ]

    # Add detrended
    women_dt = women.copy()
    for iso3 in women_dt["iso3"].unique():
        mask = women_dt["iso3"] == iso3
        for col in ["obesity_prevalence_pct", "sugar_10kcal", "oil_100kcal"]:
            vals = women_dt.loc[mask, col].values
            t = np.arange(len(vals), dtype=float)
            slope, intercept = np.polyfit(t, vals, 1)
            women_dt.loc[mask, col] = vals - (slope * t + intercept)

    t_values = []
    labels = []

    for label, formula, use_cluster, cluster_col in specs:
        if use_cluster and cluster_col:
            fit = smf.ols(formula, data=women).fit(
                cov_type="cluster", cov_kwds={"groups": women[cluster_col]}
            )
        else:
            fit = smf.ols(formula, data=women).fit()
        t_val = float(fit.params["sugar_10kcal"] / fit.bse["sugar_10kcal"])
        t_values.append(t_val)
        labels.append(label)

    # Detrended (multivariate, no C(iso3) -- country means are near-zero after detrending)
    fit_dt = smf.ols("obesity_prevalence_pct ~ sugar_10kcal + oil_100kcal", data=women_dt).fit(
        cov_type="cluster", cov_kwds={"groups": women_dt["iso3"]}
    )
    t_values.append(float(fit_dt.params["sugar_10kcal"] / fit_dt.bse["sugar_10kcal"]))
    labels.append("Detrended")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3" if abs(t) > 2 else "#9E9E9E" for t in t_values]
    bars = ax.bar(range(len(t_values)), t_values, color=colors, edgecolor="k", linewidth=0.5)
    ax.axhline(y=2, color="r", linestyle="--", alpha=0.5, label="t = +/-2")
    ax.axhline(y=-2, color="r", linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("t-statistic (clustered SE)")
    ax.set_title("Figure 2: Sugar-obesity t-statistics across specifications")

    for i, t in enumerate(t_values):
        ax.text(i, t + 0.1 * np.sign(t), f"{t:.2f}", ha="center", va="bottom" if t > 0 else "top", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_fe_cascade.png")
    plt.close(fig)
    print("  Figure 2 saved")


def figure_3_global_trend_r2() -> None:
    """Histogram: global trend R2 distribution."""
    results = json.loads((DERIVED / "global_scope_results.json").read_text())
    # We need to recompute for the histogram
    bmi = pd.read_csv(
        ROOT / "data" / "raw" / "NCD_RisC_Lancet_2024_BMI_age_standardised_country.csv",
        encoding="utf-8-sig",
    )
    obesity_col = [c for c in bmi.columns if "BMI>=30" in c and "lower" not in c.lower() and "upper" not in c.lower()][0]
    bmi = bmi[bmi["Sex"] == "Women"].copy()
    bmi = bmi[bmi["Year"].isin(range(2010, 2022))].copy()
    bmi["obesity"] = pd.to_numeric(bmi[obesity_col], errors="coerce")
    bmi = bmi.dropna(subset=["obesity"])
    bmi = bmi[bmi["ISO"].str.len() == 3].copy()

    # SSA countries
    ssa = set([
        "AGO", "BDI", "BEN", "BFA", "BWA", "CAF", "CIV", "CMR", "COD", "COG",
        "ETH", "GAB", "GHA", "GIN", "GMB", "GNB", "KEN", "LBR", "LSO", "MDG",
        "MLI", "MOZ", "MRT", "MUS", "MWI", "NAM", "NER", "NGA", "RWA", "SEN",
        "SLE", "SWZ", "TCD", "TGO", "TZA", "UGA", "ZMB",
    ])

    r2_global = []
    r2_ssa = []
    for iso3, group in bmi.groupby("ISO"):
        if len(group) < 3:
            continue
        y = group["obesity"].values
        t = group["Year"].values - group["Year"].values.min()
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot < 1e-15:
            r2 = 1.0
        else:
            slope, intercept = np.polyfit(t, y, 1)
            y_pred = slope * t + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            r2 = 1 - ss_res / ss_tot
        r2_global.append(r2)
        if iso3 in ssa:
            r2_ssa.append(r2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.2]})

    # Left panel: full histogram
    ax = axes[0]
    ax.hist(r2_global, bins=50, color="#9E9E9E", alpha=0.7, edgecolor="k", linewidth=0.3, label="All countries")
    ax.axvline(x=0.90, color="r", linestyle="--", alpha=0.7, label="R2 = 0.90")
    ax.set_xlabel("Linear trend R2")
    ax.set_ylabel("Number of countries")
    ax.set_title("A. All 200 countries")
    ax.legend(loc="upper left", fontsize=8)
    pct = np.mean(np.array(r2_global) >= 0.90) * 100
    ax.text(0.35, 0.85, f"{pct:.0f}% >= 0.90",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Right panel: zoomed R2 > 0.90, SSA highlighted
    ax = axes[1]
    r2_high = [r for r in r2_global if r >= 0.90]
    r2_ssa_high = [r for r in r2_ssa if r >= 0.90]
    bins_zoom = np.linspace(0.90, 1.0, 25)
    ax.hist(r2_high, bins=bins_zoom, color="#9E9E9E", alpha=0.5, edgecolor="k",
            linewidth=0.3, label="All countries")
    ax.hist(r2_ssa_high, bins=bins_zoom, color="#FF9800", alpha=0.8, edgecolor="k",
            linewidth=0.3, label="SSA countries")
    ax.set_xlabel("Linear trend R2")
    ax.set_ylabel("Number of countries")
    ax.set_title("B. Zoomed: R2 >= 0.90 (SSA highlighted)")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Figure 3: Global obesity trend linearity (Women, 2010-2021)", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_global_trend_r2.png")
    plt.close(fig)
    print("  Figure 3 saved")


def figure_4_soybean_oil_case() -> None:
    """Panel figure: soybean oil passes standard tests but fails trend diagnostics.

    Split into two panels to avoid the Oster delta (17.6) crushing the x-axis
    and making the diagnostic bars invisible.
    """
    results = json.loads((DERIVED / "robustness_failure_results.json").read_text())

    # Panel A: t-statistics and correlations (comparable scale)
    tests_main = [
        ("Cross-sectional r", "significant", results["cross_sectional"]["r"]),
        ("Country FE t", "significant", results["country_fe"]["t_cl"]),
        ("TWFE t", "null", results["twfe"]["t_cl"]),
        ("Lead-lag ratio", "co-trending", results["lead_lag"]["ratio"]),
        ("Detrended t", "null", results["detrended"]["t_cl"]),
        ("First differences r", "null", results["first_differences"]["r"]),
        ("Granger r", "null", results["granger"]["r"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1.2]})

    # Panel A: main diagnostics
    ax = axes[0]
    labels_a = [t[0] for t in tests_main]
    values_a = [t[2] for t in tests_main]
    colors_a = ["#4CAF50" if v == "significant" else "#F44336"
                for _, v, _ in tests_main]

    bars = ax.barh(range(len(tests_main)), values_a, color=colors_a,
                   edgecolor="k", linewidth=0.5, height=0.6)
    ax.axvline(x=0, color="k", linewidth=0.5)
    ax.axvline(x=2, color="k", linestyle=":", alpha=0.3, linewidth=0.8)
    ax.axvline(x=-2, color="k", linestyle=":", alpha=0.3, linewidth=0.8)

    for i, (label, verdict, val) in enumerate(tests_main):
        offset = 0.08
        ha = "left" if val >= 0 else "right"
        x_pos = val + offset if val >= 0 else val - offset
        ax.text(x_pos, i, f"{val:.2f}", va="center", ha=ha, fontsize=9)

    ax.set_yticks(range(len(labels_a)))
    ax.set_yticklabels(labels_a, fontsize=10)
    ax.set_xlabel("Statistic value")
    ax.set_title("A. Diagnostic tests", fontsize=11)
    ax.set_xlim(-0.5, 3.0)
    ax.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#4CAF50", edgecolor="k", label="Passes (standard tests)"),
                       Patch(facecolor="#F44336", edgecolor="k", label="Fails (trend diagnostics)")],
              loc="lower right", fontsize=8, framealpha=0.9)

    # Panel B: Oster delta (separate scale)
    ax2 = axes[1]
    oster_val = results["oster_bounds"]["delta"]
    ax2.barh([0], [oster_val], color="#F44336", edgecolor="k", linewidth=0.5, height=0.6)
    ax2.barh([1], [3.0], color="#9E9E9E", edgecolor="k", linewidth=0.5, height=0.6, alpha=0.4)
    ax2.text(oster_val + 0.3, 0, f"{oster_val:.1f}", va="center", fontsize=10, fontweight="bold")
    ax2.text(3.0 + 0.3, 1, "3.0", va="center", fontsize=9, color="#666")

    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Soybean oil\n(misleading)", "Typical\nthreshold"], fontsize=9)
    ax2.set_xlabel("Oster $\\delta$")
    ax2.set_title("B. Oster bounds", fontsize=11)
    ax2.set_xlim(0, 22)
    ax2.invert_yaxis()

    fig.suptitle("Figure 4: Soybean oil passes standard tests but fails every trend diagnostic", y=1.02, fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig4_soybean_oil_case.png")
    plt.close(fig)
    print("  Figure 4 saved")


def figure_5_gdp_control() -> None:
    """Side-by-side: cross-sectional vs detrended GDP-obesity relationship."""
    panel = pd.read_csv(DERIVED / "analysis_panel.csv")
    panel = panel.dropna(subset=["gdp_per_capita_constant_2015_usd"])
    women = panel[panel["sex"] == "Women"].copy()
    women["log_gdp"] = women["log_gdp_per_capita_constant_2015_usd"]

    # Cross-sectional (2018-2020 averages, consistent with script 02)
    xs = women[women["year"].isin([2018, 2019, 2020])]
    cm = xs.groupby("iso3").agg({"obesity_prevalence_pct": "mean", "log_gdp": "mean"}).reset_index()

    # Detrended
    women_dt = women.copy()
    for iso3 in women_dt["iso3"].unique():
        mask = women_dt["iso3"] == iso3
        for col in ["obesity_prevalence_pct", "log_gdp"]:
            vals = women_dt.loc[mask, col].values
            t = np.arange(len(vals), dtype=float)
            slope, intercept = np.polyfit(t, vals, 1)
            women_dt.loc[mask, col] = vals - (slope * t + intercept)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Cross-sectional
    ax = axes[0]
    ax.scatter(cm["log_gdp"], cm["obesity_prevalence_pct"], s=40, alpha=0.7, c="#4CAF50", edgecolors="k", linewidths=0.5)
    r_xs, _ = stats.pearsonr(cm["log_gdp"], cm["obesity_prevalence_pct"])
    m, b = np.polyfit(cm["log_gdp"], cm["obesity_prevalence_pct"], 1)
    x_fit = np.linspace(cm["log_gdp"].min(), cm["log_gdp"].max(), 100)
    ax.plot(x_fit, m * x_fit + b, "k--", alpha=0.5)
    ax.set_xlabel("Log GDP per capita (country mean)")
    ax.set_ylabel("Obesity prevalence (%)")
    ax.set_title(f"A. Cross-sectional (r = {r_xs:.3f})")

    # Detrended scatter (country-level means of detrended values)
    ax = axes[1]
    dt_cm = women_dt.groupby("iso3").agg({"obesity_prevalence_pct": "mean", "log_gdp": "mean"}).reset_index()
    # For detrended, use pooled observations
    x_dt = women_dt["log_gdp"].values
    y_dt = women_dt["obesity_prevalence_pct"].values
    ax.scatter(x_dt, y_dt, s=8, alpha=0.2, c="#F44336")
    r_dt, _ = stats.pearsonr(x_dt, y_dt)
    # Add regression line so the negative relationship is visible
    m_dt, b_dt = np.polyfit(x_dt, y_dt, 1)
    x_fit_dt = np.linspace(x_dt.min(), x_dt.max(), 100)
    ax.plot(x_fit_dt, m_dt * x_fit_dt + b_dt, "k-", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Log GDP per capita (detrended)")
    ax.set_ylabel("Obesity prevalence (detrended)")
    ax.set_title(f"B. Detrended within-country (r = {r_dt:.3f})")

    fig.suptitle("Figure 5: GDP-obesity relationship reverses after detrending", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig5_gdp_control.png")
    plt.close(fig)
    print("  Figure 5 saved")


def figure_6_variance_decomposition(panel: pd.DataFrame) -> None:
    """Waterfall: how much variance is absorbed by entity FE, year FE, residual."""
    import statsmodels.formula.api as smf

    women = panel[panel["sex"] == "Women"].copy()

    # Compute R2 for nested models
    fit_null = smf.ols("obesity_prevalence_pct ~ 1", data=women).fit()
    fit_entity = smf.ols("obesity_prevalence_pct ~ C(iso3)", data=women).fit()
    fit_entity_year = smf.ols("obesity_prevalence_pct ~ C(iso3) + C(year)", data=women).fit()

    total_var = 100.0
    entity_pct = fit_entity.rsquared * 100
    year_incremental = (fit_entity_year.rsquared - fit_entity.rsquared) * 100
    residual = 100 - fit_entity_year.rsquared * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Entity FE", "Year FE\n(incremental)", "Residual"]
    values = [entity_pct, year_incremental, residual]
    bottoms = [0, entity_pct, entity_pct + year_incremental]
    colors = ["#2196F3", "#FF9800", "#9E9E9E"]

    for i, (cat, val, bot, col) in enumerate(zip(categories, values, bottoms, colors)):
        ax.bar(i, val, bottom=bot, color=col, edgecolor="k", linewidth=0.5, width=0.6)
        ax.text(i, bot + val / 2, f"{val:.1f}%", ha="center", va="center", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylabel("Percentage of total variance")
    ax.set_title("Figure 6: Variance decomposition of obesity prevalence")
    ax.set_ylim(0, 105)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig6_variance_decomposition.png")
    plt.close(fig)
    print("  Figure 6 saved")


def main() -> None:
    print("=" * 70)
    print("09_figures.py: Generating all figures")
    print("=" * 70)

    panel = pd.read_csv(DERIVED / "analysis_panel.csv")

    figure_1_cross_sectional(panel)
    figure_2_fe_cascade(panel)
    figure_3_global_trend_r2()
    figure_4_soybean_oil_case()
    figure_5_gdp_control()
    figure_6_variance_decomposition(panel)

    print(f"\nAll figures saved to: {FIGURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
