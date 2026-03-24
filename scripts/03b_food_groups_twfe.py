"""03b_food_groups_twfe.py -- Full cascade for all 10 FAOSTAT food groups.

Tests each food group under four specifications to show the collapse pattern:
  - Cross-sectional correlation (2018-2020 country means)
  - One-way country FE (clustered SEs)
  - Two-way FE (clustered SEs)
  - First differences

Key result:
  6/10 significant cross-sectionally
  2/10 significant under one-way FE
  0/10 significant under TWFE
  0/10 significant under first differences

Writes:
  data/derived/food_groups_twfe_results.json
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
OUTPUT = DERIVED / "food_groups_twfe_results.json"

FOOD_GROUPS = [
    "Cereals - Excluding Beer",
    "Sugar & Sweeteners",
    "Vegetable Oils",
    "Meat",
    "Milk - Excluding Butter",
    "Fruits - Excluding Wine",
    "Vegetables",
    "Starchy Roots",
    "Pulses",
    "Animal fats",
]


def load_faostat_food_groups() -> pd.DataFrame:
    """Extract kcal/capita/day for each food group from FAOSTAT FBS."""
    fbs_zip = RAW / "FoodBalanceSheets_E_Africa.zip"
    with zipfile.ZipFile(fbs_zip) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        fbs = pd.read_csv(z.open(csv_name), encoding="latin-1")

    kcal = fbs[
        (fbs["Item"].isin(FOOD_GROUPS))
        & (fbs["Element"] == "Food supply (kcal/capita/day)")
    ].copy()

    year_cols = [c for c in kcal.columns if c.startswith("Y") and c[1:].isdigit()]
    id_cols = ["Area Code (M49)", "Area", "Item"]

    long = kcal.melt(
        id_vars=id_cols, value_vars=year_cols, var_name="year_str", value_name="kcal"
    )
    long["year"] = long["year_str"].str[1:].astype(int)
    long["kcal"] = pd.to_numeric(long["kcal"], errors="coerce")

    wide = long.pivot_table(
        index=["Area Code (M49)", "Area", "year"],
        columns="Item",
        values="kcal",
    ).reset_index()

    wide.columns.name = None
    return wide


def main() -> None:
    print("=" * 70)
    print("03b_food_groups_twfe.py: Full cascade for all 10 FAOSTAT food groups")
    print("=" * 70)

    panel = pd.read_csv(DERIVED / "analysis_panel.csv")
    women = panel[panel["sex"] == "Women"].copy()

    # Load food group kcal data from FAOSTAT
    fg = load_faostat_food_groups()

    # Map FAOSTAT area names to panel country names
    iso_map = women[["iso3", "country"]].drop_duplicates()
    fg_countries = fg["Area"].unique()
    panel_countries = women["country"].unique()

    name_map = {}
    for pc in panel_countries:
        for fc in fg_countries:
            pc_norm = pc.lower().replace("\u2019", "'").replace(",", "").strip()
            fc_norm = fc.lower().replace("\u2019", "'").replace(",", "").strip()
            if pc_norm == fc_norm:
                name_map[fc] = pc
                break
            if "ivoire" in fc_norm and "ivoire" in pc_norm:
                name_map[fc] = pc
                break
            if "tanzania" in fc_norm and "tanzania" in pc_norm:
                name_map[fc] = pc
                break
            if "congo" in fc_norm and "congo" in pc_norm and "dem" in fc_norm and "dem" in pc_norm:
                name_map[fc] = pc
                break
            if "congo" in fc_norm and "congo" in pc_norm and "dem" not in fc_norm and "dem" not in pc_norm:
                name_map[fc] = pc
                break

    fg["country"] = fg["Area"].map(name_map)
    fg = fg.dropna(subset=["country"])
    merged = women.merge(fg, on=["country", "year"], how="left")

    print(f"\n  Panel: {len(merged)} obs, {merged['iso3'].nunique()} countries")
    print(f"  Food groups matched: {len(name_map)}/{len(panel_countries)}\n")

    all_results = {}
    counts = {"cross_sectional_sig": 0, "one_way_fe_sig": 0, "twfe_sig": 0, "first_diff_sig": 0}

    for group in FOOD_GROUPS:
        if group not in merged.columns:
            print(f"  {group}: NOT FOUND")
            all_results[group] = {"status": "missing"}
            continue

        safe_name = group.replace(" ", "_").replace("-", "_").replace("&", "and").lower()
        col = f"fg_{safe_name}"
        merged[col] = merged[group] / 10.0

        valid = merged.dropna(subset=[col, "obesity_prevalence_pct"])
        if len(valid) < 50:
            all_results[group] = {"status": "insufficient_data"}
            continue

        res = {}

        # 1. Cross-sectional (2018-2020 means)
        cs = valid[valid["year"].isin([2018, 2019, 2020])].groupby("iso3").agg(
            obesity=("obesity_prevalence_pct", "mean"),
            fg_val=(col, "mean"),
        ).dropna()
        r_xs, p_xs = stats.pearsonr(cs["fg_val"], cs["obesity"])
        res["cross_sectional_r"] = round(r_xs, 3)
        res["cross_sectional_p"] = round(p_xs, 4)
        if p_xs < 0.05:
            counts["cross_sectional_sig"] += 1

        # 2. One-way country FE
        fit_1 = smf.ols(
            f"obesity_prevalence_pct ~ {col} + C(iso3)", data=valid
        ).fit(cov_type="cluster", cov_kwds={"groups": valid["iso3"]})
        t_1 = float(fit_1.tvalues[col])
        p_1 = float(fit_1.pvalues[col])
        res["one_way_fe_t"] = round(t_1, 2)
        res["one_way_fe_p"] = round(p_1, 4)
        if p_1 < 0.05:
            counts["one_way_fe_sig"] += 1

        # 3. TWFE
        fit_2 = smf.ols(
            f"obesity_prevalence_pct ~ {col} + C(iso3) + C(year)", data=valid
        ).fit(cov_type="cluster", cov_kwds={"groups": valid["iso3"]})
        t_2 = float(fit_2.tvalues[col])
        p_2 = float(fit_2.pvalues[col])
        res["twfe_t"] = round(t_2, 2)
        res["twfe_p"] = round(p_2, 4)
        if p_2 < 0.05:
            counts["twfe_sig"] += 1

        # 4. First differences
        valid_sorted = valid.sort_values(["iso3", "year"])
        fd_rows = []
        for iso3 in valid_sorted["iso3"].unique():
            sub = valid_sorted[valid_sorted["iso3"] == iso3].reset_index(drop=True)
            for i in range(1, len(sub)):
                fd_rows.append({
                    "d_obesity": sub.loc[i, "obesity_prevalence_pct"] - sub.loc[i - 1, "obesity_prevalence_pct"],
                    "d_fg": sub.loc[i, col] - sub.loc[i - 1, col],
                })
        fd_df = pd.DataFrame(fd_rows)
        r_fd, p_fd = stats.pearsonr(fd_df["d_fg"], fd_df["d_obesity"])
        res["first_diff_r"] = round(r_fd, 3)
        res["first_diff_p"] = round(p_fd, 4)
        if p_fd < 0.05:
            counts["first_diff_sig"] += 1

        all_results[group] = res
        print(f"  {group:30s}  xs_r={r_xs:+.3f}  1way_t={t_1:+5.2f}  twfe_t={t_2:+5.2f}  fd_r={r_fd:+.3f}")

    # Summary
    print(f"\n--- Cascade summary ---")
    print(f"  Cross-sectional significant: {counts['cross_sectional_sig']}/10")
    print(f"  One-way FE significant:      {counts['one_way_fe_sig']}/10")
    print(f"  TWFE significant:            {counts['twfe_sig']}/10")
    print(f"  First differences significant: {counts['first_diff_sig']}/10")

    output = {
        "cascade_counts": counts,
        "per_group": all_results,
    }

    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results written to {OUTPUT}")


if __name__ == "__main__":
    main()
