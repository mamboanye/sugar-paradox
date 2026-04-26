"""14_energy_surprise_probe.py -- attack the detrended-energy surprise.

The Eastern-Europe comparator found total energy null in TWFE and first
differences but positive after simple country-linear detrending. This script
checks whether that isolated cell survives stricter adversarial tests.

Writes:
  data/derived/energy_surprise_probe_results.json
  tables/table6_energy_surprise_probe.csv
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
TABLES = ROOT / "tables"
OUT_JSON = DERIVED / "energy_surprise_probe_results.json"
OUT_CSV = TABLES / "table6_energy_surprise_probe.csv"
HELPER_PATH = ROOT / "scripts" / "13_eastern_europe_comparator.py"

spec = importlib.util.spec_from_file_location("ee_comparator", HELPER_PATH)
ee = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ee)


def make_panel() -> pd.DataFrame:
    panel = pd.read_csv(DERIVED / "analysis_panel.csv")
    women = panel[panel["sex"] == "Women"].copy()
    features = ee.load_fbs_features(women["country"])
    merged = women.merge(features, on=["country", "year"], how="left")
    cols_to_detrend = [
        "total_energy_100kcal",
        "log_gdp_per_capita_constant_2015_usd",
        "urban_population_10pct",
    ]
    detr = ee.detrend_by_country(merged, cols_to_detrend)
    # compact columns with short names
    detr["y"] = detr["obesity_prevalence_pct_detrended"]
    detr["energy"] = detr["total_energy_100kcal_detrended"]
    detr["gdp"] = detr["log_gdp_per_capita_constant_2015_usd_detrended"]
    detr["urban"] = detr["urban_population_10pct_detrended"]
    detr = detr.sort_values(["iso3", "year"])
    detr["energy_lag1"] = detr.groupby("iso3")["energy"].shift(1)
    detr["energy_lead1"] = detr.groupby("iso3")["energy"].shift(-1)
    return detr


def run_model(label: str, data: pd.DataFrame, x: str = "energy", controls: list[str] | None = None, fe_cols: list[str] | None = None) -> dict:
    cols = ["iso3", "y", x] + (controls or []) + (fe_cols or [])
    valid = data.dropna(subset=cols).copy()
    res = ee.ols_clustered(valid.rename(columns={x: "x"}), "y", "x", controls=controls, fe_cols=fe_cols)
    res.update({"model": label, "x": x})
    return res


def permutation_test(data: pd.DataFrame, n_perm: int = 999, seed: int = 20260426) -> dict:
    rng = np.random.default_rng(seed)
    base_valid = data.dropna(subset=["iso3", "y", "energy"])[["iso3", "year", "y", "energy"]].copy()
    observed = ee.ols_clustered(base_valid.rename(columns={"energy": "x"}), "y", "x")
    obs_t = abs(observed["t_cl"])
    perm_t = []
    for _ in range(n_perm):
        tmp = base_valid.copy()
        tmp["x"] = tmp.groupby("iso3")["energy"].transform(lambda s: rng.permutation(s.to_numpy()))
        res = ee.ols_clustered(tmp[["iso3", "y", "x"]], "y", "x")
        perm_t.append(abs(res["t_cl"]))
    perm_t = np.asarray(perm_t)
    return {
        "observed_beta": observed["beta"],
        "observed_t": observed["t_cl"],
        "n_perm": n_perm,
        "empirical_p_abs_t": float((np.sum(perm_t >= obs_t) + 1) / (n_perm + 1)),
        "perm_abs_t_p95": float(np.quantile(perm_t, 0.95)),
        "perm_abs_t_p99": float(np.quantile(perm_t, 0.99)),
    }


def leave_one_country(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iso in sorted(data["iso3"].dropna().unique()):
        d = data[data["iso3"] != iso]
        res = run_model(f"loo_drop_{iso}", d)
        rows.append({"dropped_iso3": iso, **res})
    return pd.DataFrame(rows)


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    data = make_panel()

    tests = []
    tests.append(run_model("detrended_energy_baseline", data))
    tests.append(run_model("detrended_energy_plus_year_fe", data, fe_cols=["year"]))
    tests.append(run_model("detrended_energy_plus_controls", data, controls=["gdp", "urban"]))
    tests.append(run_model("detrended_energy_plus_controls_year_fe", data, controls=["gdp", "urban"], fe_cols=["year"]))
    tests.append(run_model("detrended_energy_lag1", data, x="energy_lag1"))
    tests.append(run_model("detrended_energy_lead1", data, x="energy_lead1"))
    table = pd.DataFrame(tests)

    loo = leave_one_country(data)
    perm = permutation_test(data)

    summary = {
        "baseline": table[table["model"] == "detrended_energy_baseline"].iloc[0].to_dict(),
        "year_fe": table[table["model"] == "detrended_energy_plus_year_fe"].iloc[0].to_dict(),
        "controls_year_fe": table[table["model"] == "detrended_energy_plus_controls_year_fe"].iloc[0].to_dict(),
        "lead1": table[table["model"] == "detrended_energy_lead1"].iloc[0].to_dict(),
        "lag1": table[table["model"] == "detrended_energy_lag1"].iloc[0].to_dict(),
        "loo_min_t": float(loo["t_cl"].min()),
        "loo_max_t": float(loo["t_cl"].max()),
        "loo_positive_p_lt_0_05": int(((loo["beta"] > 0) & (loo["p"] < 0.05)).sum()),
        "loo_n": int(len(loo)),
        "permutation": perm,
    }
    verdict = {
        "collapses_under_year_fe": bool(abs(summary["year_fe"]["t_cl"]) < 1.96),
        "lead_lag_symmetric_or_ambiguous": bool(abs(summary["lead1"]["t_cl"]) >= abs(summary["lag1"]["t_cl"]) * 0.75),
        "permutation_extreme": bool(perm["empirical_p_abs_t"] < 0.05),
    }
    out = {
        "question": "Is the simple country-detrended total-energy association robust or an artifact?",
        "panel": {"rows": int(len(data)), "countries": int(data["iso3"].nunique()), "years": [int(data["year"].min()), int(data["year"].max())]},
        "tests": table.to_dict("records"),
        "leave_one_country": loo.to_dict("records"),
        "summary": summary,
        "verdict_flags": verdict,
        "interpretation_guardrail": "This probes one isolated detrended total-energy cell; it does not estimate individual diet effects.",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    table.to_csv(OUT_CSV, index=False)
    print(json.dumps({"summary": summary, "verdict_flags": verdict}, indent=2))
    print(f"wrote {OUT_JSON} and {OUT_CSV}")


if __name__ == "__main__":
    main()
