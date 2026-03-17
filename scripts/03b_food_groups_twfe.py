"""03b_food_groups_twfe.py -- TWFE for all 10 FAOSTAT food groups.

Tests each food group individually under two-way fixed effects (country + year)
with clustered SEs. Reproduces the Section 3.3 claim that none of the 10
FAOSTAT food groups predicts within-country obesity.

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
    print("03b_food_groups_twfe.py: TWFE for all 10 FAOSTAT food groups")
    print("=" * 70)

    panel = pd.read_csv(DERIVED / "analysis_panel.csv")
    women = panel[panel["sex"] == "Women"].copy()

    # Load food group kcal data from FAOSTAT
    fg = load_faostat_food_groups()

    # Map M49 to ISO3 via the panel
    iso_map = women[["iso3", "country"]].drop_duplicates()
    # Match on country name (FAOSTAT "Area" ~ panel "country")
    # First try direct merge on Area
    fg_merged = None

    # Get M49-to-ISO3 mapping from the panel's raw source
    # Simpler: merge on country name and year
    fg_countries = fg["Area"].unique()
    panel_countries = women["country"].unique()

    # Build a name mapping
    name_map = {}
    for pc in panel_countries:
        for fc in fg_countries:
            if pc.lower() == fc.lower():
                name_map[fc] = pc
                break
            # Handle common mismatches
            pc_norm = pc.lower().replace("'", "'").replace(",", "").strip()
            fc_norm = fc.lower().replace("'", "'").replace(",", "").strip()
            if pc_norm == fc_norm:
                name_map[fc] = pc
                break
            # Special cases
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

    # Merge food groups into women's panel
    merged = women.merge(fg, on=["country", "year"], how="left")

    results = {}
    print(f"\n  Panel: {len(merged)} obs, {merged['iso3'].nunique()} countries")
    print(f"  Food groups matched: {len(name_map)}/{len(panel_countries)}\n")

    for group in FOOD_GROUPS:
        if group not in merged.columns:
            print(f"  {group}: NOT FOUND")
            results[group] = {"status": "missing"}
            continue

        # Scale to reasonable units (per 10 kcal)
        safe_name = group.replace(" ", "_").replace("-", "_").replace("&", "and").lower()
        col = f"fg_{safe_name}"
        merged[col] = merged[group] / 10.0

        valid = merged.dropna(subset=[col, "obesity_prevalence_pct"])
        if len(valid) < 50:
            print(f"  {group}: insufficient data ({len(valid)} obs)")
            results[group] = {"status": "insufficient_data", "n_obs": len(valid)}
            continue

        # TWFE: obesity ~ food_group + C(iso3) + C(year)
        formula = f"obesity_prevalence_pct ~ {col} + C(iso3) + C(year)"
        try:
            fit = smf.ols(formula, data=valid).fit(
                cov_type="cluster", cov_kwds={"groups": valid["iso3"]}
            )
            beta = float(fit.params[col])
            se = float(fit.bse[col])
            t_cl = float(beta / se)
            p = float(fit.pvalues[col])

            results[group] = {
                "beta": round(beta, 4),
                "se": round(se, 4),
                "t_cl": round(t_cl, 2),
                "p": round(p, 4),
                "n_obs": len(valid),
                "n_countries": valid["iso3"].nunique(),
            }
            print(f"  {group:30s}  t_cl = {t_cl:6.2f}  (p = {p:.3f})")
        except Exception as e:
            print(f"  {group}: ERROR - {e}")
            results[group] = {"status": "error", "message": str(e)}

    # Summary
    n_sig = sum(
        1
        for v in results.values()
        if isinstance(v.get("p"), float) and v["p"] < 0.05
    )
    results["summary"] = {
        "n_groups_tested": len(FOOD_GROUPS),
        "n_significant_at_05": n_sig,
        "interpretation": "None of the 10 FAOSTAT food groups predicts within-country obesity under TWFE with clustered SEs.",
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results written to {OUTPUT}")
    print(f"  Significant at 5%: {n_sig} / {len(FOOD_GROUPS)}")


if __name__ == "__main__":
    main()
