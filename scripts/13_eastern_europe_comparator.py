"""13_eastern_europe_comparator.py -- total-energy/macronutrient comparator.

Targeted adversarial check from the literature closure. A 2025 Eastern
Europe FAOSTAT/WHO paper reports significant TWFE energy/fiber associations with
obesity. This script asks whether grand-total energy, fat, or protein supply
recreates that pattern in the SSA panel.

Writes:
  data/derived/eastern_europe_comparator_results.json
  tables/table5_eastern_europe_comparator.csv
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from math import erf, sqrt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
TABLES = ROOT / "tables"
OUT_JSON = DERIVED / "eastern_europe_comparator_results.json"
OUT_CSV = TABLES / "table5_eastern_europe_comparator.csv"

FBS_FEATURES = [
    {
        "name": "total_energy_100kcal",
        "item": "Grand Total",
        "element": "Food supply (kcal/capita/day)",
        "scale": 100.0,
        "label": "Total energy supply (+100 kcal/cap/day)",
    },
    {
        "name": "total_fat_10g",
        "item": "Grand Total",
        "element": "Fat supply quantity (g/capita/day)",
        "scale": 10.0,
        "label": "Total fat supply (+10 g/cap/day)",
    },
    {
        "name": "total_protein_10g",
        "item": "Grand Total",
        "element": "Protein supply quantity (g/capita/day)",
        "scale": 10.0,
        "label": "Total protein supply (+10 g/cap/day)",
    },
]


def norm_name(x: str) -> str:
    return str(x).lower().replace(",", "").replace("'", "").strip()


def load_fbs_features(panel_countries: pd.Series) -> pd.DataFrame:
    fbs_zip = RAW / "FoodBalanceSheets_E_Africa.zip"
    with zipfile.ZipFile(fbs_zip) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        fbs = pd.read_csv(z.open(csv_name), encoding="latin-1")

    keep = pd.concat(
        [
            fbs[(fbs["Item"] == feat["item"]) & (fbs["Element"] == feat["element"])].assign(
                feature_name=feat["name"], scale=feat["scale"], feature_label=feat["label"]
            )
            for feat in FBS_FEATURES
        ],
        ignore_index=True,
    )
    year_cols = [c for c in keep.columns if c.startswith("Y") and c[1:].isdigit()]
    long = keep.melt(
        id_vars=["Area", "feature_name", "scale", "feature_label"],
        value_vars=year_cols,
        var_name="year_str",
        value_name="raw_value",
    )
    long["year"] = long["year_str"].str[1:].astype(int)
    long["raw_value"] = pd.to_numeric(long["raw_value"], errors="coerce")

    # country-name mapping against the paper panel
    fbs_countries = long["Area"].dropna().unique()
    panel_country_values = list(pd.Series(panel_countries).dropna().unique())
    name_map = {}
    for pc in panel_country_values:
        pc_norm = norm_name(pc)
        for fc in fbs_countries:
            fc_norm = norm_name(fc)
            if pc_norm == fc_norm:
                name_map[fc] = pc
                break
            if "ivoire" in fc_norm and "ivoire" in pc_norm:
                name_map[fc] = pc
                break
            if "tanzania" in fc_norm and "tanzania" in pc_norm:
                name_map[fc] = pc
                break
            if "congo" in fc_norm and "congo" in pc_norm and ("dem" in fc_norm) == ("dem" in pc_norm):
                name_map[fc] = pc
                break
    long["country"] = long["Area"].map(name_map)
    long = long.dropna(subset=["country"])
    long["value_scaled"] = long["raw_value"] / long["scale"]
    wide = long.pivot_table(index=["country", "year"], columns="feature_name", values="value_scaled").reset_index()
    wide.columns.name = None
    return wide


def normal_pvalue(t: float) -> float:
    if not np.isfinite(t):
        return float("nan")
    phi = 0.5 * (1.0 + erf(abs(t) / sqrt(2.0)))
    return float(2.0 * (1.0 - phi))


def ols_clustered(data: pd.DataFrame, y_col: str, x_col: str, controls: list[str] | None = None, fe_cols: list[str] | None = None) -> dict:
    """OLS with country-clustered sandwich SE using numpy only.

    This avoids remote statsmodels dependency for small DGX comparator runs.
    """
    controls = controls or []
    fe_cols = fe_cols or []
    cols = [y_col, x_col, "iso3"] + controls + fe_cols
    d = data.dropna(subset=cols).copy()
    parts = [pd.Series(1.0, index=d.index, name="intercept"), d[x_col].astype(float).rename(x_col)]
    for c in controls:
        parts.append(d[c].astype(float).rename(c))
    for fe in fe_cols:
        cats = pd.get_dummies(d[fe].astype(str), prefix=fe, drop_first=True, dtype=float)
        parts.append(cats)
    Xdf = pd.concat(parts, axis=1)
    X = Xdf.to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k), dtype=float)
    clusters = d["iso3"].to_numpy()
    unique_clusters = np.unique(clusters)
    for g in unique_clusters:
        idx = clusters == g
        Xg = X[idx, :]
        ug = resid[idx][:, None]
        xu = Xg.T @ ug
        meat += xu @ xu.T
    G = len(unique_clusters)
    correction = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
    vcov = correction * xtx_inv @ meat @ xtx_inv
    term_idx = list(Xdf.columns).index(x_col)
    se = float(np.sqrt(max(vcov[term_idx, term_idx], 0.0)))
    b = float(beta[term_idx])
    t = b / se if se else float("nan")
    ssr = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "beta": b,
        "se": se,
        "t_cl": t,
        "p": normal_pvalue(t),
        "n_obs": int(n),
        "n_countries": int(G),
        "r2": float(1.0 - ssr / tss) if tss else float("nan"),
    }


def detrend_by_country(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols + ["obesity_prevalence_pct"]:
        resid = []
        for _, g in out.groupby("iso3", sort=False):
            valid = g[["year", col]].dropna()
            if len(valid) >= 3 and valid["year"].nunique() >= 3:
                x = g["year"].to_numpy(dtype=float)
                y = g[col].to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() >= 3:
                    slope, intercept = np.polyfit(x[ok], y[ok], 1)
                    r = pd.Series(y - (slope * x + intercept), index=g.index)
                else:
                    r = pd.Series(np.nan, index=g.index)
            else:
                r = pd.Series(np.nan, index=g.index)
            resid.append(r)
        out[f"{col}_detrended"] = pd.concat(resid).sort_index()
    return out


def first_diff_by_country(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.sort_values(["iso3", "year"]).copy()
    for col in cols + ["obesity_prevalence_pct"]:
        out[f"d_{col}"] = out.groupby("iso3")[col].diff()
    return out


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(DERIVED / "analysis_panel.csv")
    women = panel[panel["sex"] == "Women"].copy()
    features = load_fbs_features(women["country"])
    merged = women.merge(features, on=["country", "year"], how="left")
    terms = [feat["name"] for feat in FBS_FEATURES]

    rows = []
    results = {
        "question": "Do grand-total energy/fat/protein FAOSTAT features recreate the Eastern Europe TWFE comparator in the SSA panel?",
        "features": FBS_FEATURES,
        "panel": {
            "rows": int(len(merged)),
            "countries": int(merged["iso3"].nunique()),
            "years": [int(merged["year"].min()), int(merged["year"].max())],
        },
        "models": {},
    }

    # Cross-section: 2018-2020 country means, partial with GDP/urban/oil.
    xs = merged[merged["year"].between(2018, 2020)].groupby(["iso3", "country"], as_index=False).mean(numeric_only=True)
    for term in terms:
        valid = xs.dropna(subset=[term, "obesity_prevalence_pct", "log_gdp_per_capita_constant_2015_usd", "urban_population_10pct"])
        res = ols_clustered(valid, "obesity_prevalence_pct", term, controls=["log_gdp_per_capita_constant_2015_usd", "urban_population_10pct"])
        results["models"][("cross_section_controls", term)] = res
        rows.append({"model": "cross_section_controls", "feature": term, **res})

    # Panel models.
    for term in terms:
        valid = merged.dropna(subset=[term, "obesity_prevalence_pct"])
        res = ols_clustered(valid, "obesity_prevalence_pct", term, fe_cols=["iso3", "year"])
        results["models"][("twfe", term)] = res
        rows.append({"model": "twfe", "feature": term, **res})

        valid2 = merged.dropna(subset=[term, "obesity_prevalence_pct", "log_gdp_per_capita_constant_2015_usd", "urban_population_10pct"])
        res2 = ols_clustered(valid2, "obesity_prevalence_pct", term, controls=["log_gdp_per_capita_constant_2015_usd", "urban_population_10pct"], fe_cols=["iso3", "year"])
        results["models"][("twfe_gdp_urban", term)] = res2
        rows.append({"model": "twfe_gdp_urban", "feature": term, **res2})

    # Detrended and first-difference simple correlations/regressions.
    detr = detrend_by_country(merged, terms)
    fd = first_diff_by_country(merged, terms)
    for term in terms:
        dterm = f"{term}_detrended"
        dout = "obesity_prevalence_pct_detrended"
        valid = detr.dropna(subset=[dterm, dout, "iso3"])[["iso3", dterm, dout]].copy()
        valid = valid.rename(columns={dterm: term, dout: "obesity_prevalence_pct"})
        res = ols_clustered(valid, "obesity_prevalence_pct", term)
        results["models"][("country_detrended", term)] = res
        rows.append({"model": "country_detrended", "feature": term, **res})

        fdterm = f"d_{term}"
        fdout = "d_obesity_prevalence_pct"
        validfd = fd.dropna(subset=[fdterm, fdout, "iso3"])[["iso3", fdterm, fdout]].copy()
        validfd = validfd.rename(columns={fdterm: term, fdout: "obesity_prevalence_pct"})
        resfd = ols_clustered(validfd, "obesity_prevalence_pct", term)
        results["models"][("first_difference", term)] = resfd
        rows.append({"model": "first_difference", "feature": term, **resfd})

    table = pd.DataFrame(rows)
    table["feature_label"] = table["feature"].map({feat["name"]: feat["label"] for feat in FBS_FEATURES})
    table = table[["model", "feature", "feature_label", "beta", "se", "t_cl", "p", "n_obs", "n_countries", "r2"]]
    table.to_csv(OUT_CSV, index=False)

    # JSON keys cannot be tuples.
    results["models"] = {f"{k[0]}::{k[1]}": v for k, v in results["models"].items()}
    results["summary"] = {
        "n_tests": int(len(table)),
        "n_p_lt_0_05": int((table["p"] < 0.05).sum()),
        "twfe_energy": table[(table["model"] == "twfe") & (table["feature"] == "total_energy_100kcal")].to_dict("records"),
        "interpretation_guardrail": "This is an SSA comparator against a regional Eastern Europe result; it tests aggregate energy/macronutrients, not individual diet or fiber intake.",
    }
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(json.dumps(results["summary"], indent=2))
    print(f"wrote {OUT_JSON} and {OUT_CSV}")


if __name__ == "__main__":
    main()
