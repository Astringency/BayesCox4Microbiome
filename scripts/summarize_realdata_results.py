#!/usr/bin/env python3
"""Summarize real-data method comparisons without non-stdlib dependencies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REALDATA_SUMMARY_ROOT = ROOT / "results" / "summary" / "realdata"


def latest_result_dir(prefix: str) -> Path:
    matches = sorted(p for p in REALDATA_SUMMARY_ROOT.glob(f"{prefix}*") if p.is_dir())
    return matches[-1] if matches else REALDATA_SUMMARY_ROOT / prefix


def default_bayes_root() -> Path:
    candidates = [
        ROOT / "results" / "realdata" / "soft_center",
        ROOT / "results" / "realdata_soft_center",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


FLORAL_DIR = latest_result_dir("realdata_FLORAL_")
CODA_DIR = latest_result_dir("realdata_coda4microbiome_")
BAYES_ROOT = default_bayes_root()
DEFAULT_OUTDIR = ROOT / "results" / "summary" / "realdata" / "soft_center_method_comparison"
R_PLOT_SCRIPT = ROOT / "scripts" / "plot_realdata_results.R"

PRIOR_ORDER = ["hs", "rhs", "new-rhs", "cox-hs", "cox-rhs", "lasso", "sslasso"]
PRIOR_COLORS = {
    "hs": "#276FBF",
    "rhs": "#F28E2B",
    "new-rhs": "#59A14F",
    "cox-hs": "#AF7AA1",
    "cox-rhs": "#E15759",
    "lasso": "#4E79A7",
    "sslasso": "#76B7B2",
}
METHOD_COLORS = {
    "FLORAL": "#4E79A7",
    "coda4microbiome": "#F28E2B",
    "Bayes-soft-center": "#59A14F",
}
BASELINE_METHOD_COLORS = {
    "FLORAL_min": "#1F77B4",
    "FLORAL_1se": "#6BAED6",
    "coda4microbiome_lambda.min": "#D95F02",
    "coda4microbiome_lambda.1se": "#FDB863",
}
BASELINE_METHOD_ORDER = [
    "FLORAL_min",
    "FLORAL_1se",
    "coda4microbiome_lambda.min",
    "coda4microbiome_lambda.1se",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_output_dir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for child in outdir.iterdir():
        if child.is_dir() and child.name == "figures":
            clean_figures_dir(outdir)
        elif child.is_dir() and child.name == "by_dataset":
            shutil.rmtree(child)
        elif child.is_file() and child.suffix.lower() in {".csv", ".md", ".svg", ".png", ".pdf", ".eps"}:
            child.unlink()


def clean_figures_dir(outdir: Path) -> None:
    figures = outdir / "figures"
    if figures.exists():
        for child in figures.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except OSError:
                pass
    figures.mkdir(parents=True, exist_ok=True)


def can_plot_from_existing_tables(outdir: Path) -> bool:
    required = [
        "best_per_prior_and_baselines.csv",
        "all_coefficients_for_forest.csv",
        "bayes_rhat_by_parameter.csv",
    ]
    return all((outdir / name).is_file() for name in required)


def run_r_plotting(outdir: Path, r_env: str) -> None:
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError("conda was not found; cannot run R plotting environment.")
    cmd = [
        conda,
        "run",
        "-n",
        r_env,
        "Rscript",
        str(R_PLOT_SCRIPT),
        "--outdir",
        str(outdir),
    ]
    subprocess.run(cmd, check=True)


def to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        x = float(value)
    else:
        text = str(value).strip().strip('"')
        if text in {"", "NA", "NaN", "nan", "missing", "nothing", "Inf", "-Inf"}:
            return None
        try:
            x = float(text)
        except ValueError:
            return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def to_int(value: object) -> int | None:
    x = to_float(value)
    if x is None:
        return None
    return int(round(x))


def fmt_num(value: object, digits: int = 3) -> str:
    x = to_float(value)
    if x is None:
        return ""
    return f"{x:.{digits}f}"


def bool_text(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return "true"
    if text in {"false", "0", "no", "n"}:
        return "false"
    return ""


def canonical_dataset(value: str | None) -> str:
    if not value:
        return ""
    name = str(value).strip()
    if name.endswith(".csv"):
        name = name[:-4]
    if name.endswith("_CIBERSORT_OS"):
        name = name[: -len("_CIBERSORT_OS")]
    if name.endswith("_CIBERSORT_survival"):
        name = name[: -len("_CIBERSORT_survival")]
    return name


def dataset_group(dataset_id: str) -> str:
    if dataset_id.startswith("TCGA_"):
        return "TCGA"
    if dataset_id.startswith("HCT"):
        return "HCT"
    return dataset_id


def split_features(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).split(";") if x.strip()]


def selected_flag(row: dict[str, str]) -> bool:
    return str(row.get("selected", "")).strip().lower() == "true"


def numeric_rank_key(value: object, reverse: bool = False) -> float:
    x = to_float(value)
    if x is None:
        return -math.inf if reverse else math.inf
    return x


def median(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return statistics.median(vals)


def percentile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def svg_text(x: float, y: float, text: str, size: int = 11, anchor: str = "start", weight: str = "400", color: str = "#222") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{escape(text)}</text>'
    )


def truncate_label(text: str, max_len: int = 48) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def load_baseline_metrics() -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, float]], dict[tuple[str, str], list[str]]]:
    performance: list[dict[str, object]] = []
    selected_coefficients: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    selected_methods: dict[tuple[str, str], list[str]] = defaultdict(list)

    for row in read_csv(FLORAL_DIR / "metrics.csv"):
        dataset_id = canonical_dataset(row.get("dataset_id"))
        method = row.get("method", "")
        features = split_features(row.get("selected_features"))
        performance.append(
            {
                "dataset_id": dataset_id,
                "dataset_group": row.get("dataset_group") or dataset_group(dataset_id),
                "method_family": "FLORAL",
                "method": method,
                "fit_success": bool_text(row.get("fit_success")),
                "n": row.get("n_used", ""),
                "p": row.get("p", ""),
                "events": row.get("n_events", ""),
                "censoring_rate": row.get("censoring_rate", ""),
                "cv_cindex": row.get("floral_cv_cindex", ""),
                "cv_cindex_se": row.get("floral_cv_cindex_se", ""),
                "cv_cindex_sd": "",
                "apparent_cindex": row.get("cindex", ""),
                "apparent_cindex_se": row.get("cindex_se", ""),
                "n_selected": row.get("n_selected", ""),
                "selected_features": ";".join(features),
                "waic_pw1": "",
                "waic_pw2": "",
                "lpml": "",
                "elpd_loo": "",
                "elpd_loo_se": "",
                "looic": "",
                "looic_se": "",
                "pareto_k_max": "",
                "pareto_k_bad": "",
                "source": str(FLORAL_DIR / "metrics.csv"),
            }
        )

    for row in read_csv(CODA_DIR / "metrics.csv"):
        dataset_id = canonical_dataset(row.get("dataset_id"))
        method = row.get("method", "")
        features = split_features(row.get("selected_features"))
        performance.append(
            {
                "dataset_id": dataset_id,
                "dataset_group": row.get("dataset_group") or dataset_group(dataset_id),
                "method_family": "coda4microbiome",
                "method": method,
                "fit_success": bool_text(row.get("fit_success")),
                "n": row.get("n_used", ""),
                "p": row.get("p", ""),
                "events": row.get("n_events", ""),
                "censoring_rate": row.get("censoring_rate", ""),
                "cv_cindex": row.get("mean_cv_cindex_package", ""),
                "cv_cindex_se": "",
                "cv_cindex_sd": row.get("sd_cv_cindex_package", ""),
                "apparent_cindex": row.get("apparent_cindex_package") or row.get("cindex", ""),
                "apparent_cindex_se": row.get("cindex_se", ""),
                "n_selected": row.get("n_selected", ""),
                "selected_features": ";".join(features),
                "waic_pw1": "",
                "waic_pw2": "",
                "lpml": "",
                "elpd_loo": "",
                "elpd_loo_se": "",
                "looic": "",
                "looic_se": "",
                "pareto_k_max": "",
                "pareto_k_bad": "",
                "source": str(CODA_DIR / "metrics.csv"),
            }
        )

    for source_dir in [FLORAL_DIR, CODA_DIR]:
        for row in read_csv(source_dir / "selected_features.csv"):
            dataset_id = canonical_dataset(row.get("dataset_id"))
            method = row.get("method", "")
            feature = row.get("feature", "")
            coef = to_float(row.get("coefficient"))
            if dataset_id and feature and coef is not None:
                selected_coefficients[(dataset_id, method)][feature] = coef
                selected_methods[(dataset_id, feature)].append(method)

    return performance, selected_coefficients, selected_methods


def load_bayes_cv() -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    tuning_rows: list[dict[str, object]] = []
    best_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for dataset_dir in sorted(BAYES_ROOT.iterdir()):
        path = dataset_dir / "tuning" / "prior_tuning_summary.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            dataset_id = canonical_dataset(row.get("dataset_id") or row.get("dataset_name") or dataset_dir.name)
            out = dict(row)
            out["source_dataset_id"] = row.get("dataset_id", "")
            out["dataset_id"] = dataset_id
            out["dataset_group"] = dataset_group(dataset_id)
            out["original_dataset_dir"] = dataset_dir.name
            out["method_family"] = "Bayes-soft-center"
            out["method"] = f"Bayes_{row.get('prior', '')}"
            out["row_type"] = "bayes_tuning"
            out["source"] = str(path)
            tuning_rows.append(out)
            if row.get("status") != "success":
                continue
            prior = row.get("prior", "")
            key = (dataset_id, prior)
            score = to_float(row.get("CV_Cindex_mean"))
            if score is None:
                continue
            old = best_by_key.get(key)
            if old is None or score > (to_float(old.get("CV_Cindex_mean")) or -math.inf):
                best_by_key[key] = out
    return tuning_rows, best_by_key


def load_bayes_coefficients() -> dict[tuple[str, str], list[dict[str, object]]]:
    coef_by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
    for dataset_dir in sorted(BAYES_ROOT.iterdir()):
        tables = dataset_dir / "tables"
        if not tables.exists():
            continue
        dataset_id = canonical_dataset(dataset_dir.name)
        for path in sorted(tables.glob("coef_*.csv")):
            prior = path.stem.replace("coef_", "")
            rows: list[dict[str, object]] = []
            for idx, row in enumerate(read_csv(path), start=1):
                out = {
                    "dataset_id": dataset_id,
                    "original_dataset_dir": dataset_dir.name,
                    "prior": prior,
                    "parameter": f"beta[{idx}]",
                    "variable": row.get("variable", ""),
                    "mean": row.get("mean", ""),
                    "median": row.get("median", ""),
                    "map": row.get("map", ""),
                    "lower": row.get("lower", ""),
                    "upper": row.get("upper", ""),
                    "selected": bool_text(row.get("selected")),
                }
                rows.append(out)
            coef_by_key[(dataset_id, prior)] = rows
    return coef_by_key


def load_bayes_metrics(
    best_cv_by_key: dict[tuple[str, str], dict[str, object]],
    coef_by_key: dict[tuple[str, str], list[dict[str, object]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    bayes_metrics: list[dict[str, object]] = []
    performance: list[dict[str, object]] = []
    for dataset_dir in sorted(BAYES_ROOT.iterdir()):
        path = dataset_dir / "tables" / "model_metrics.csv"
        if not path.exists():
            continue
        dataset_id = canonical_dataset(dataset_dir.name)
        for row in read_csv(path):
            if row.get("estimate") != "mean":
                continue
            prior = row.get("prior", "")
            cv = best_cv_by_key.get((dataset_id, prior), {})
            selected_count = sum(1 for r in coef_by_key.get((dataset_id, prior), []) if r.get("selected") == "true")
            metric = {
                "dataset_id": dataset_id,
                "original_dataset_dir": dataset_dir.name,
                "dataset_group": dataset_group(dataset_id),
                "prior": prior,
                "estimate": row.get("estimate", ""),
                "loglik": row.get("loglik", ""),
                "LPML": row.get("LPML", ""),
                "AIC": row.get("AIC", ""),
                "BIC": row.get("BIC", ""),
                "WAIC_pw1": row.get("WAIC_pw1", ""),
                "WAIC_pw2": row.get("WAIC_pw2", ""),
                "DIC": row.get("DIC", ""),
                "ELPD_LOO": row.get("ELPD_LOO", ""),
                "ELPD_LOO_SE": row.get("ELPD_LOO_SE", ""),
                "PLOO": row.get("PLOO", ""),
                "PLOO_SE": row.get("PLOO_SE", ""),
                "LOOIC": row.get("LOOIC", ""),
                "LOOIC_SE": row.get("LOOIC_SE", ""),
                "ParetoK_max": row.get("ParetoK_max", ""),
                "ParetoK_bad": row.get("ParetoK_bad", ""),
                "PSIS_LOO": row.get("PSIS_LOO", ""),
                "Cindex": row.get("Cindex", ""),
                "n_selected": selected_count,
                "sample_seconds": row.get("sample_seconds", ""),
                "total_seconds": row.get("total_seconds", ""),
                "cv_cindex": cv.get("CV_Cindex_mean", ""),
                "cv_cindex_se": cv.get("CV_Cindex_se", ""),
                "cv_cindex_sd": cv.get("CV_Cindex_sd", ""),
                "cv_config_index": cv.get("config_index", ""),
                "cv_prior_config_index": cv.get("prior_config_index", ""),
                "source": str(path),
            }
            bayes_metrics.append(metric)
            performance.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_group": dataset_group(dataset_id),
                    "method_family": "Bayes-soft-center",
                    "method": f"Bayes_{prior}",
                    "fit_success": "true",
                    "n": cv.get("n", ""),
                    "p": cv.get("p", ""),
                    "events": cv.get("events", ""),
                    "censoring_rate": cv.get("censoring_rate", ""),
                    "cv_cindex": metric["cv_cindex"],
                    "cv_cindex_se": metric["cv_cindex_se"],
                    "cv_cindex_sd": metric["cv_cindex_sd"],
                    "apparent_cindex": metric["Cindex"],
                    "apparent_cindex_se": "",
                    "n_selected": selected_count,
                    "selected_features": ";".join(
                        str(r.get("variable", "")) for r in coef_by_key.get((dataset_id, prior), []) if r.get("selected") == "true"
                    ),
                    "waic_pw1": metric["WAIC_pw1"],
                    "waic_pw2": metric["WAIC_pw2"],
                    "lpml": metric["LPML"],
                    "elpd_loo": metric["ELPD_LOO"],
                    "elpd_loo_se": metric["ELPD_LOO_SE"],
                    "looic": metric["LOOIC"],
                    "looic_se": metric["LOOIC_SE"],
                    "pareto_k_max": metric["ParetoK_max"],
                    "pareto_k_bad": metric["ParetoK_bad"],
                    "source": str(path),
                }
            )
    return bayes_metrics, performance


def bayes_tuning_standard_rows(tuning_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in tuning_rows:
        prior = row.get("prior", "")
        out = dict(row)
        out.update(
            {
                "row_type": "bayes_tuning",
                "dataset_id": row.get("dataset_id", ""),
                "dataset_group": row.get("dataset_group", dataset_group(str(row.get("dataset_id", "")))),
                "method_family": "Bayes-soft-center",
                "method": row.get("method") or f"Bayes_{prior}",
                "prior": prior,
                "selection": "",
                "fit_success": "true" if row.get("status") == "success" else "false",
                "cv_cindex": row.get("CV_Cindex_mean", ""),
                "cv_cindex_se": row.get("CV_Cindex_se", ""),
                "cv_cindex_sd": row.get("CV_Cindex_sd", ""),
                "cv_cindex_map": row.get("CV_Cindex_MAP_mean", ""),
                "cv_cindex_map_se": row.get("CV_Cindex_MAP_se", ""),
                "cv_cindex_median": row.get("CV_Cindex_median_mean", ""),
                "cv_cindex_median_se": row.get("CV_Cindex_median_se", ""),
                "apparent_cindex": row.get("Cindex_mean", ""),
                "waic_pw1": row.get("WAIC_pw1_mean", ""),
                "waic_pw2": row.get("WAIC_pw2_mean", ""),
                "lpml": row.get("LPML_mean", ""),
                "elpd_loo": row.get("ELPD_LOO_mean", ""),
                "looic": row.get("LOOIC_mean", ""),
                "pareto_k_max": row.get("ParetoK_max_mean", ""),
                "n_successful_folds": row.get("n_successful_folds", ""),
                "source": row.get("source", ""),
            }
        )
        rows.append(out)
    return rows


def baseline_standard_rows(performance: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in performance:
        if row.get("method_family") == "Bayes-soft-center":
            continue
        method = str(row.get("method", ""))
        selection = ""
        if method.endswith("_min") or method.endswith("lambda.min"):
            selection = "min"
        elif method.endswith("_1se") or method.endswith("lambda.1se"):
            selection = "1se"
        out = dict(row)
        out.update(
            {
                "row_type": "baseline_summary",
                "prior": "",
                "selection": selection,
                "config_index": "",
                "prior_config_index": "",
                "status": "success" if row.get("fit_success") == "true" else "",
                "cv_cindex_map": "",
                "cv_cindex_map_se": "",
                "cv_cindex_median": "",
                "cv_cindex_median_se": "",
                "n_successful_folds": "",
            }
        )
        rows.append(out)
    return rows


def comparison_table_rows(
    tuning_rows: list[dict[str, object]],
    performance: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows = bayes_tuning_standard_rows(tuning_rows) + baseline_standard_rows(performance)
    best_bayes: dict[tuple[str, str], dict[str, object]] = {}
    for row in bayes_tuning_standard_rows(tuning_rows):
        if row.get("status") != "success":
            continue
        score = to_float(row.get("cv_cindex"))
        if score is None:
            continue
        key = (str(row.get("dataset_id", "")), str(row.get("prior", "")))
        old = best_bayes.get(key)
        if old is None or score > (to_float(old.get("cv_cindex")) or -math.inf):
            best_bayes[key] = row
    best_rows = sorted(
        best_bayes.values(),
        key=lambda r: (str(r.get("dataset_id", "")), PRIOR_ORDER.index(str(r.get("prior"))) if str(r.get("prior")) in PRIOR_ORDER else 999),
    )
    best_rows.extend(baseline_standard_rows(performance))
    best_rows.sort(
        key=lambda r: (
            str(r.get("dataset_id", "")),
            0 if r.get("method_family") == "Bayes-soft-center" else 1,
            PRIOR_ORDER.index(str(r.get("prior"))) if str(r.get("prior")) in PRIOR_ORDER else 999,
            str(r.get("method", "")),
        )
    )
    return all_rows, best_rows


def write_dataset_split_tables(
    outdir: Path,
    all_rows: list[dict[str, object]],
    best_rows: list[dict[str, object]],
    coefficient_rows: list[dict[str, object]],
) -> None:
    by_dataset_dir = outdir / "by_dataset"
    datasets = sorted(
        set(str(r.get("dataset_id", "")) for r in all_rows)
        | set(str(r.get("dataset_id", "")) for r in best_rows)
        | set(str(r.get("dataset_id", "")) for r in coefficient_rows)
    )
    for dataset_id in datasets:
        if not dataset_id:
            continue
        dataset_dir = by_dataset_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        write_csv(dataset_dir / "all_tuning_and_baselines.csv", [r for r in all_rows if r.get("dataset_id") == dataset_id])
        write_csv(dataset_dir / "best_per_prior_and_baselines.csv", [r for r in best_rows if r.get("dataset_id") == dataset_id])
        write_csv(dataset_dir / "all_coefficients_for_forest.csv", [r for r in coefficient_rows if r.get("dataset_id") == dataset_id])


def loo_summary_rows(bayes_metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in bayes_metrics:
        by_dataset[str(row["dataset_id"])].append(row)
    for dataset_id, rows in by_dataset.items():
        ranked = sorted(rows, key=lambda r: numeric_rank_key(r.get("ELPD_LOO"), reverse=True), reverse=True)
        best_elpd = to_float(ranked[0].get("ELPD_LOO")) if ranked else None
        for rank, row in enumerate(ranked, start=1):
            elpd = to_float(row.get("ELPD_LOO"))
            delta = "" if best_elpd is None or elpd is None else elpd - best_elpd
            out.append(
                {
                    "dataset_id": dataset_id,
                    "prior": row.get("prior", ""),
                    "loo_rank": rank,
                    "delta_elpd_from_best": delta,
                    "ELPD_LOO": row.get("ELPD_LOO", ""),
                    "ELPD_LOO_SE": row.get("ELPD_LOO_SE", ""),
                    "LOOIC": row.get("LOOIC", ""),
                    "LOOIC_SE": row.get("LOOIC_SE", ""),
                    "PLOO": row.get("PLOO", ""),
                    "PLOO_SE": row.get("PLOO_SE", ""),
                    "ParetoK_max": row.get("ParetoK_max", ""),
                    "ParetoK_bad": row.get("ParetoK_bad", ""),
                    "PSIS_LOO": row.get("PSIS_LOO", ""),
                    "LPML": row.get("LPML", ""),
                    "WAIC_pw1": row.get("WAIC_pw1", ""),
                    "WAIC_pw2": row.get("WAIC_pw2", ""),
                    "Cindex": row.get("Cindex", ""),
                    "cv_cindex": row.get("cv_cindex", ""),
                    "n_selected": row.get("n_selected", ""),
                }
            )
    return out


def best_bayes_by_dataset(bayes_metrics: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in bayes_metrics:
        by_dataset[str(row["dataset_id"])].append(row)
    best: dict[str, dict[str, object]] = {}
    for dataset_id, rows in by_dataset.items():
        rows_with_elpd = [r for r in rows if to_float(r.get("ELPD_LOO")) is not None]
        if rows_with_elpd:
            chosen = max(rows_with_elpd, key=lambda r: to_float(r.get("ELPD_LOO")) or -math.inf)
        else:
            chosen = min(rows, key=lambda r: to_float(r.get("LOOIC")) or math.inf)
        best[dataset_id] = chosen
    return best


def run_prior_from_dir(run_dir: Path) -> str:
    name = run_dir.name
    for prefix in ["realdata_soft_center-", "realdata_prox-", "realdata_est-", "realdata_"]:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def compute_rhat_for_file(
    sample_path: Path,
    dataset_id: str,
    original_dataset_dir: str,
    prior: str,
    variable_names: list[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    with sample_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        chain_idx = header.index("chain")
        beta_indices: list[tuple[int, int, str]] = []
        for idx, name in enumerate(header):
            m = re.fullmatch(r"beta\[(\d+)\]", name)
            if m:
                beta_num = int(m.group(1))
                variable = variable_names[beta_num - 1] if 0 <= beta_num - 1 < len(variable_names) else name
                beta_indices.append((idx, beta_num, variable))

        sums: dict[str, list[float]] = {}
        sums2: dict[str, list[float]] = {}
        counts: dict[str, list[int]] = {}
        for row in reader:
            chain = row[chain_idx]
            if chain not in sums:
                k = len(beta_indices)
                sums[chain] = [0.0] * k
                sums2[chain] = [0.0] * k
                counts[chain] = [0] * k
            s = sums[chain]
            ss = sums2[chain]
            n = counts[chain]
            for j, (col_idx, _beta_num, _variable) in enumerate(beta_indices):
                try:
                    x = float(row[col_idx])
                except (ValueError, IndexError):
                    continue
                if not math.isfinite(x):
                    continue
                s[j] += x
                ss[j] += x * x
                n[j] += 1

    chains = sorted(sums)
    rows: list[dict[str, object]] = []
    finite_rhats: list[float] = []
    for j, (_col_idx, beta_num, variable) in enumerate(beta_indices):
        means: list[float] = []
        variances: list[float] = []
        ns: list[int] = []
        for chain in chains:
            n = counts[chain][j]
            if n < 2:
                continue
            total = sums[chain][j]
            mean = total / n
            var = (sums2[chain][j] - total * total / n) / (n - 1)
            if var < 0 and var > -1e-12:
                var = 0.0
            if var < 0:
                continue
            means.append(mean)
            variances.append(var)
            ns.append(n)
        rhat = ""
        if len(means) >= 2 and min(ns) >= 2:
            n_eff = min(ns)
            m = len(means)
            mean_of_means = sum(means) / m
            b = n_eff * sum((x - mean_of_means) ** 2 for x in means) / (m - 1)
            w = sum(variances) / m
            if w > 0:
                var_hat = ((n_eff - 1) / n_eff) * w + b / n_eff
                if var_hat >= 0:
                    rhat_value = math.sqrt(var_hat / w)
                    if math.isfinite(rhat_value):
                        rhat = rhat_value
                        finite_rhats.append(rhat_value)
        rows.append(
            {
                "dataset_id": dataset_id,
                "original_dataset_dir": original_dataset_dir,
                "prior": prior,
                "parameter": f"beta[{beta_num}]",
                "variable": variable,
                "rhat": rhat,
                "n_chains": len(chains),
                "n_per_chain_min": min(ns) if ns else "",
                "n_per_chain_max": max(ns) if ns else "",
                "sample_file": str(sample_path),
            }
        )

    summary = {
        "dataset_id": dataset_id,
        "original_dataset_dir": original_dataset_dir,
        "prior": prior,
        "n_parameters": len(rows),
        "n_rhat_finite": len(finite_rhats),
        "rhat_median": median(finite_rhats) or "",
        "rhat_p95": percentile(finite_rhats, 0.95) or "",
        "rhat_max": max(finite_rhats) if finite_rhats else "",
        "n_rhat_gt_1_01": sum(1 for x in finite_rhats if x > 1.01),
        "n_rhat_gt_1_05": sum(1 for x in finite_rhats if x > 1.05),
        "n_rhat_gt_1_10": sum(1 for x in finite_rhats if x > 1.10),
        "sample_file": str(sample_path),
    }
    return rows, summary


def compute_all_rhat(coef_by_key: dict[tuple[str, str], list[dict[str, object]]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    sample_files = sorted(BAYES_ROOT.glob("*/runs/*/samples_wide.csv"))
    for idx, sample_path in enumerate(sample_files, start=1):
        dataset_dir = sample_path.parents[2]
        dataset_id = canonical_dataset(dataset_dir.name)
        prior = run_prior_from_dir(sample_path.parent)
        variable_names = [str(r.get("variable", "")) for r in coef_by_key.get((dataset_id, prior), [])]
        print(f"[rhat] {idx}/{len(sample_files)} {dataset_id} {prior}", flush=True)
        rows, summary = compute_rhat_for_file(sample_path, dataset_id, dataset_dir.name, prior, variable_names)
        all_rows.extend(rows)
        summaries.append(summary)
    return all_rows, summaries


def cell_type_label(variable: str) -> str:
    base = variable
    if base.endswith("_cibersort"):
        base = base[: -len("_cibersort")]
    labels = {
        "b_cell_naive": "naive B cells",
        "b_cell_memory": "memory B cells",
        "b_cell_plasma": "plasma cells",
        "plasma_cell": "plasma cells",
        "t_cell_cd8": "CD8 T cells",
        "t_cell_cd4_naive": "naive CD4 T cells",
        "t_cell_cd4_memory_resting": "resting memory CD4 T cells",
        "t_cell_cd4_memory_activated": "activated memory CD4 T cells",
        "t_cell_follicular_helper": "follicular helper T cells",
        "t_cell_regulatory_tregs": "regulatory T cells",
        "t_cell_gamma_delta": "gamma delta T cells",
        "nk_cell_resting": "resting NK cells",
        "nk_cell_activated": "activated NK cells",
        "monocyte": "monocytes",
        "macrophage_m0": "M0 macrophages",
        "macrophage_m1": "M1 macrophages",
        "macrophage_m2": "M2 macrophages",
        "dendritic_cell_resting": "resting dendritic cells",
        "dendritic_cell_activated": "activated dendritic cells",
        "myeloid_dendritic_cell_resting": "resting myeloid dendritic cells",
        "myeloid_dendritic_cell_activated": "activated myeloid dendritic cells",
        "mast_cell_resting": "resting mast cells",
        "mast_cell_activated": "activated mast cells",
        "eosinophil": "eosinophils",
        "neutrophil": "neutrophils",
    }
    return labels.get(base, base.replace("_", " "))


def variable_kind(dataset_id: str, variable: str) -> str:
    if dataset_id.startswith("TCGA_") or variable.endswith("_cibersort"):
        return "immune_cell_fraction"
    if dataset_id.startswith("T1D") and re.fullmatch(r"X\d+", variable):
        return "microbiome_feature_id"
    return "microbiome_taxon"


def interpret_variable(dataset_id: str, variable: str, direction: str) -> str:
    kind = variable_kind(dataset_id, variable)
    if kind == "immune_cell_fraction":
        subject = f"CIBERSORT immune-cell fraction: {cell_type_label(variable)}"
    elif kind == "microbiome_feature_id":
        subject = f"Microbiome feature ID {variable}; no taxonomy annotation is present in these summary files"
    else:
        subject = f"Microbiome taxon: {variable}"

    if direction == "positive":
        direction_text = "positive coefficient: higher relative abundance/proportion is associated with higher hazard"
    elif direction == "negative":
        direction_text = "negative coefficient: higher relative abundance/proportion is associated with lower hazard"
    else:
        direction_text = "direction is not stable from the available coefficient estimates"
    return f"{subject}; {direction_text} under the compositional/log-contrast constraint. Association only, not causal."


def build_key_variables(
    coef_by_key: dict[tuple[str, str], list[dict[str, object]]],
    bayes_metrics: list[dict[str, object]],
    baseline_selected_methods: dict[tuple[str, str], list[str]],
    baseline_selected_coefficients: dict[tuple[str, str], dict[str, float]],
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    best_bayes = best_bayes_by_dataset(bayes_metrics)
    selected_count: dict[tuple[str, str], int] = defaultdict(int)
    selected_priors: dict[tuple[str, str], list[str]] = defaultdict(list)
    selected_coef_rows: dict[tuple[str, str, str], dict[str, object]] = {}
    for (dataset_id, prior), rows in coef_by_key.items():
        for row in rows:
            variable = str(row.get("variable", ""))
            if row.get("selected") == "true":
                selected_count[(dataset_id, variable)] += 1
                selected_priors[(dataset_id, variable)].append(prior)
                selected_coef_rows[(dataset_id, variable, prior)] = row

    variables_by_dataset: dict[str, set[str]] = defaultdict(set)
    for dataset_id, variable in selected_count:
        variables_by_dataset[dataset_id].add(variable)
    for dataset_id, variable in baseline_selected_methods:
        variables_by_dataset[dataset_id].add(variable)

    rows_out: list[dict[str, object]] = []
    forest_by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    datasets = sorted(set(list(variables_by_dataset) + [str(r["dataset_id"]) for r in bayes_metrics]))
    for dataset_id in datasets:
        best_prior = str(best_bayes.get(dataset_id, {}).get("prior", ""))
        variables = set(variables_by_dataset.get(dataset_id, set()))
        # If no method selected anything, keep the largest absolute estimates from the best LOO Bayes prior for the forest.
        fallback_rows: list[dict[str, object]] = []
        if best_prior:
            fallback_rows = sorted(
                coef_by_key.get((dataset_id, best_prior), []),
                key=lambda r: abs(to_float(r.get("mean")) or 0.0),
                reverse=True,
            )[:10]
            if not variables:
                variables.update(str(r.get("variable", "")) for r in fallback_rows)

        for variable in sorted(variables):
            priors = selected_priors.get((dataset_id, variable), [])
            chosen_prior = ""
            chosen_coef: dict[str, object] | None = None
            if priors:
                candidates = []
                for prior in priors:
                    metric = next((m for m in bayes_metrics if m["dataset_id"] == dataset_id and m["prior"] == prior), None)
                    candidates.append((to_float(metric.get("ELPD_LOO")) if metric else None, prior))
                candidates.sort(key=lambda item: item[0] if item[0] is not None else -math.inf, reverse=True)
                chosen_prior = candidates[0][1]
                chosen_coef = selected_coef_rows.get((dataset_id, variable, chosen_prior))
            elif best_prior:
                chosen_prior = best_prior
                chosen_coef = next((r for r in coef_by_key.get((dataset_id, best_prior), []) if r.get("variable") == variable), None)

            mean_value = to_float(chosen_coef.get("mean")) if chosen_coef else None
            lower = to_float(chosen_coef.get("lower")) if chosen_coef else None
            upper = to_float(chosen_coef.get("upper")) if chosen_coef else None
            if mean_value is None:
                baseline_coefs = []
                for (d, method), coefs in baseline_selected_coefficients.items():
                    if d == dataset_id and variable in coefs:
                        baseline_coefs.append(coefs[variable])
                if baseline_coefs:
                    mean_value = sum(baseline_coefs) / len(baseline_coefs)
            direction = "positive" if mean_value is not None and mean_value > 0 else "negative" if mean_value is not None and mean_value < 0 else "unstable"
            floral_methods = [m for m in baseline_selected_methods.get((dataset_id, variable), []) if m.startswith("FLORAL")]
            coda_methods = [m for m in baseline_selected_methods.get((dataset_id, variable), []) if m.startswith("coda4microbiome")]
            support = []
            if priors:
                support.append(f"Bayes:{','.join(priors)}")
            if floral_methods:
                support.append(f"FLORAL:{','.join(floral_methods)}")
            if coda_methods:
                support.append(f"coda4microbiome:{','.join(coda_methods)}")

            row_out = {
                "dataset_id": dataset_id,
                "variable": variable,
                "variable_kind": variable_kind(dataset_id, variable),
                "direction": direction,
                "bayes_selected_count": selected_count.get((dataset_id, variable), 0),
                "bayes_selected_priors": ";".join(priors),
                "chosen_bayes_prior_for_estimate": chosen_prior,
                "bayes_mean": "" if mean_value is None else mean_value,
                "bayes_lower": "" if lower is None else lower,
                "bayes_upper": "" if upper is None else upper,
                "selected_by_FLORAL": ";".join(floral_methods),
                "selected_by_coda4microbiome": ";".join(coda_methods),
                "support_methods": " | ".join(support),
                "interpretation": interpret_variable(dataset_id, variable, direction),
            }
            rows_out.append(row_out)
            if chosen_coef is not None and mean_value is not None and lower is not None and upper is not None:
                forest_by_dataset[dataset_id].append(row_out)

        # Keep fallback forest entries when selected variables lack Bayes intervals.
        if not forest_by_dataset.get(dataset_id):
            for coef in fallback_rows:
                mean_value = to_float(coef.get("mean"))
                lower = to_float(coef.get("lower"))
                upper = to_float(coef.get("upper"))
                if mean_value is None or lower is None or upper is None:
                    continue
                variable = str(coef.get("variable", ""))
                direction = "positive" if mean_value > 0 else "negative" if mean_value < 0 else "unstable"
                forest_by_dataset[dataset_id].append(
                    {
                        "dataset_id": dataset_id,
                        "variable": variable,
                        "direction": direction,
                        "bayes_selected_count": 0,
                        "bayes_selected_priors": "",
                        "chosen_bayes_prior_for_estimate": best_prior,
                        "bayes_mean": mean_value,
                        "bayes_lower": lower,
                        "bayes_upper": upper,
                        "support_methods": "top_abs_best_LOO_prior",
                        "interpretation": interpret_variable(dataset_id, variable, direction),
                    }
                )

    rows_out.sort(key=lambda r: (str(r["dataset_id"]), -int(r.get("bayes_selected_count") or 0), str(r["variable"])))
    for dataset_id in list(forest_by_dataset):
        forest_by_dataset[dataset_id].sort(
            key=lambda r: (-(int(r.get("bayes_selected_count") or 0)), -abs(to_float(r.get("bayes_mean")) or 0.0), str(r.get("variable", "")))
        )
        forest_by_dataset[dataset_id] = forest_by_dataset[dataset_id][:30]
    return rows_out, forest_by_dataset


def build_all_coefficient_rows(
    coef_by_key: dict[tuple[str, str], list[dict[str, object]]],
    baseline_selected_coefficients: dict[tuple[str, str], dict[str, float]],
    performance: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    variables_by_dataset: dict[str, list[str]] = {}
    baseline_methods_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in performance:
        if row.get("method_family") != "Bayes-soft-center":
            baseline_methods_by_dataset[str(row.get("dataset_id", ""))].add(str(row.get("method", "")))
    for (dataset_id, prior), coef_rows in coef_by_key.items():
        if dataset_id not in variables_by_dataset:
            variables_by_dataset[dataset_id] = [str(r.get("variable", "")) for r in coef_rows]
        for row in coef_rows:
            variable = str(row.get("variable", ""))
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_group": dataset_group(dataset_id),
                    "method_family": "Bayes-soft-center",
                    "method": f"Bayes_{prior}",
                    "prior": prior,
                    "variable": variable,
                    "variable_kind": variable_kind(dataset_id, variable),
                    "estimate": row.get("mean", ""),
                    "lower": row.get("lower", ""),
                    "upper": row.get("upper", ""),
                    "selected": row.get("selected", ""),
                    "interval_available": "true",
                    "source": "bayes_coef",
                }
            )
    for dataset_id, variables in variables_by_dataset.items():
        for method in BASELINE_METHOD_ORDER:
            coefs = baseline_selected_coefficients.get((dataset_id, method), {})
            if method not in baseline_methods_by_dataset.get(dataset_id, set()):
                continue
            family = "FLORAL" if method.startswith("FLORAL") else "coda4microbiome"
            for variable in variables:
                value = coefs.get(variable, 0.0)
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "dataset_group": dataset_group(dataset_id),
                        "method_family": family,
                        "method": method,
                        "prior": "",
                        "variable": variable,
                        "variable_kind": variable_kind(dataset_id, variable),
                        "estimate": value,
                        "lower": "",
                        "upper": "",
                        "selected": "true" if variable in coefs else "false",
                        "interval_available": "false",
                        "source": "baseline_selected_features",
                    }
                )
    rows.sort(
        key=lambda r: (
            str(r.get("dataset_id", "")),
            str(r.get("variable", "")),
            0 if r.get("method_family") == "Bayes-soft-center" else 1,
            PRIOR_ORDER.index(str(r.get("prior"))) if str(r.get("prior")) in PRIOR_ORDER else 999,
            BASELINE_METHOD_ORDER.index(str(r.get("method"))) if str(r.get("method")) in BASELINE_METHOD_ORDER else 999,
        )
    )
    return rows


def best_methods(performance: list[dict[str, object]], bayes_metrics: list[dict[str, object]], rhat_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in performance:
        if to_float(row.get("cv_cindex")) is not None:
            by_dataset[str(row["dataset_id"])].append(row)
    best_bayes = best_bayes_by_dataset(bayes_metrics)
    rhat_by_key = {(str(r["dataset_id"]), str(r["prior"])): r for r in rhat_summary}
    rows: list[dict[str, object]] = []
    for dataset_id in sorted(set(list(by_dataset) + list(best_bayes))):
        cv_rows = by_dataset.get(dataset_id, [])
        best_cv = max(cv_rows, key=lambda r: to_float(r.get("cv_cindex")) or -math.inf) if cv_rows else {}
        bayes = best_bayes.get(dataset_id, {})
        rhat = rhat_by_key.get((dataset_id, str(bayes.get("prior", ""))), {})
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_group": dataset_group(dataset_id),
                "best_cv_method_family": best_cv.get("method_family", ""),
                "best_cv_method": best_cv.get("method", ""),
                "best_cv_cindex": best_cv.get("cv_cindex", ""),
                "best_cv_cindex_se": best_cv.get("cv_cindex_se", ""),
                "best_cv_cindex_sd": best_cv.get("cv_cindex_sd", ""),
                "best_bayes_loo_prior": bayes.get("prior", ""),
                "best_bayes_elpd_loo": bayes.get("ELPD_LOO", ""),
                "best_bayes_looic": bayes.get("LOOIC", ""),
                "best_bayes_lpml": bayes.get("LPML", ""),
                "best_bayes_waic_pw1": bayes.get("WAIC_pw1", ""),
                "best_bayes_waic_pw2": bayes.get("WAIC_pw2", ""),
                "best_bayes_cindex": bayes.get("Cindex", ""),
                "best_bayes_cv_cindex": bayes.get("cv_cindex", ""),
                "best_bayes_pareto_k_bad": bayes.get("ParetoK_bad", ""),
                "best_bayes_rhat_median": rhat.get("rhat_median", ""),
                "best_bayes_rhat_max": rhat.get("rhat_max", ""),
                "best_bayes_n_rhat_gt_1_05": rhat.get("n_rhat_gt_1_05", ""),
                "best_bayes_n_selected": bayes.get("n_selected", ""),
            }
        )
    return rows


def axis_scale(values: list[float], low_pad: float = 0.05, high_pad: float = 0.05) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return 0.0, 1.0
    lo = min(vals)
    hi = max(vals)
    if lo == hi:
        return lo - 1.0, hi + 1.0
    span = hi - lo
    return lo - span * low_pad, hi + span * high_pad


def write_rhat_svg(rhat_rows: list[dict[str, object]], path: Path) -> None:
    rows = [r for r in rhat_rows if to_float(r.get("rhat")) is not None]
    datasets = sorted({str(r["dataset_id"]) for r in rows})
    priors = [p for p in PRIOR_ORDER if any(r.get("prior") == p for r in rows)]
    if not rows or not datasets:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"800\" height=\"300\"></svg>\n", encoding="utf-8")
        return
    width = max(1100, 92 * len(datasets) + 180)
    height = 520
    left, right, top, bottom = 80, 40, 55, 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [to_float(r["rhat"]) for r in rows if to_float(r.get("rhat")) is not None]
    y_max_raw = max(values) if values else 1.1
    y_max = max(1.1, min(y_max_raw, 1.25))
    y_min = min(0.995, min(values) if values else 1.0)
    if y_max <= y_min:
        y_max = y_min + 0.1

    def x_pos(dataset: str, prior: str) -> float:
        i = datasets.index(dataset)
        cat_w = plot_w / len(datasets)
        p_i = priors.index(prior) if prior in priors else 0
        offset = 0.0
        if len(priors) > 1:
            offset = (p_i - (len(priors) - 1) / 2) * (cat_w * 0.095)
        return left + cat_w * (i + 0.5) + offset

    def y_pos(value: float) -> float:
        clipped = min(value, y_max)
        return top + (y_max - clipped) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(left, 25, "Bayesian beta Rhat convergence diagnostics", 16, weight="700"),
        svg_text(left, 43, "Each point is one beta coefficient; values above the axis cap are drawn at the top.", 11, color="#555"),
        f'<line x1="{left}" x2="{width-right}" y1="{top+plot_h}" y2="{top+plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h}" stroke="#333" stroke-width="1"/>',
    ]
    for tick in [1.0, 1.01, 1.05, 1.10]:
        if y_min <= tick <= y_max:
            y = y_pos(tick)
            color = "#999" if tick == 1.0 else "#d95f02"
            dash = "" if tick == 1.0 else ' stroke-dasharray="4 4"'
            parts.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="1"{dash}/>')
            parts.append(svg_text(left - 8, y + 4, f"{tick:.2f}", 10, anchor="end", color="#333"))
    if y_max_raw > y_max:
        parts.append(svg_text(width - right, top - 10, f"max={y_max_raw:.3f}", 10, anchor="end", color="#9b2226"))
    for r in rows:
        value = to_float(r.get("rhat"))
        if value is None:
            continue
        prior = str(r.get("prior", ""))
        x = x_pos(str(r["dataset_id"]), prior)
        y = y_pos(value)
        color = PRIOR_COLORS.get(prior, "#666")
        radius = 2.3 if value <= 1.05 else 3.4
        stroke = "#9b2226" if value > y_max else "none"
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.45" stroke="{stroke}" stroke-width="0.8">'
            f'<title>{escape(str(r["dataset_id"]))} {escape(prior)} {escape(str(r.get("variable","")))} Rhat={value:.4f}</title></circle>'
        )
    cat_w = plot_w / len(datasets)
    for i, dataset in enumerate(datasets):
        x = left + cat_w * (i + 0.5)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top+plot_h}" y2="{top+plot_h+5}" stroke="#333"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+16}" font-family="Arial, sans-serif" font-size="10" text-anchor="end" transform="rotate(-35 {x:.1f} {top+plot_h+16})">{escape(dataset)}</text>')
    legend_x = left
    legend_y = height - 30
    for i, prior in enumerate(priors):
        x = legend_x + i * 115
        parts.append(f'<circle cx="{x:.1f}" cy="{legend_y:.1f}" r="4" fill="{PRIOR_COLORS.get(prior, "#666")}"/>')
        parts.append(svg_text(x + 8, legend_y + 4, prior, 11))
    parts.append(svg_text(20, top + plot_h / 2, "Rhat", 12, anchor="middle"))
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_cv_svg(performance: list[dict[str, object]], path: Path) -> None:
    rows = [r for r in performance if to_float(r.get("cv_cindex")) is not None]
    datasets = sorted({str(r["dataset_id"]) for r in rows})
    if not rows or not datasets:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"800\" height=\"300\"></svg>\n", encoding="utf-8")
        return
    width = max(1100, 92 * len(datasets) + 180)
    height = 500
    left, right, top, bottom = 80, 40, 45, 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [to_float(r.get("cv_cindex")) for r in rows if to_float(r.get("cv_cindex")) is not None]
    y_min, y_max = axis_scale(values, 0.08, 0.08)
    y_min = min(0.4, y_min)
    y_max = max(0.75, y_max)

    categories = ["FLORAL", "coda4microbiome"] + [p for p in PRIOR_ORDER if any(r.get("method") == f"Bayes_{p}" for r in rows)]

    def row_category(row: dict[str, object]) -> str:
        if row.get("method_family") == "Bayes-soft-center":
            method = str(row.get("method", ""))
            return method.replace("Bayes_", "", 1) if method.startswith("Bayes_") else str(row.get("prior", ""))
        return str(row.get("method_family", ""))

    def category_color(category: str) -> str:
        return PRIOR_COLORS.get(category, METHOD_COLORS.get(category, "#666"))

    def x_pos(dataset: str, category: str) -> float:
        i = datasets.index(dataset)
        cat_w = plot_w / len(datasets)
        c_i = categories.index(category) if category in categories else 0
        if len(categories) == 1:
            return left + cat_w * (i + 0.5)
        offset = (c_i - (len(categories) - 1) / 2) * (cat_w * 0.055)
        return left + cat_w * (i + 0.5) + offset

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(left, 25, "CV-Cindex comparison", 16, weight="700"),
        f'<line x1="{left}" x2="{width-right}" y1="{top+plot_h}" y2="{top+plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h}" stroke="#333" stroke-width="1"/>',
    ]
    for tick in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        if y_min <= tick <= y_max:
            y = y_pos(tick)
            parts.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>')
            parts.append(svg_text(left - 8, y + 4, f"{tick:.2f}", 10, anchor="end"))
    for r in rows:
        cv = to_float(r.get("cv_cindex"))
        if cv is None:
            continue
        method = str(r.get("method", ""))
        category = row_category(r)
        x = x_pos(str(r["dataset_id"]), category)
        y = y_pos(cv)
        color = category_color(category)
        radius = 4 if r.get("method_family") == "Bayes-soft-center" else 4.8
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.72">'
            f'<title>{escape(str(r["dataset_id"]))} {escape(method)} CV-Cindex={cv:.4f}</title></circle>'
        )
    cat_w = plot_w / len(datasets)
    for i, dataset in enumerate(datasets):
        x = left + cat_w * (i + 0.5)
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+16}" font-family="Arial, sans-serif" font-size="10" text-anchor="end" transform="rotate(-35 {x:.1f} {top+plot_h+16})">{escape(dataset)}</text>')
    legend_x = left
    legend_y = height - 34
    x = legend_x
    for category in categories:
        if x > width - right - 110:
            x = legend_x
            legend_y += 16
        parts.append(f'<circle cx="{x:.1f}" cy="{legend_y:.1f}" r="4.5" fill="{category_color(category)}"/>')
        parts.append(svg_text(x + 9, legend_y + 4, category, 10))
        x += max(86, len(category) * 7 + 26)
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_forest_svg(dataset_id: str, rows: list[dict[str, object]], path: Path) -> None:
    rows = [r for r in rows if to_float(r.get("estimate")) is not None]
    if not rows:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"800\" height=\"220\"></svg>\n", encoding="utf-8")
        return
    variables = []
    seen_vars = set()
    for row in rows:
        variable = str(row.get("variable", ""))
        if variable and variable not in seen_vars:
            seen_vars.add(variable)
            variables.append(variable)
    methods = [f"Bayes_{p}" for p in PRIOR_ORDER if any(r.get("method") == f"Bayes_{p}" for r in rows)]
    methods.extend([m for m in BASELINE_METHOD_ORDER if any(r.get("method") == m for r in rows)])
    width = 1280
    row_h = 22 if len(variables) > 50 else 26
    height = max(280, 120 + row_h * len(variables))
    left, right, top, bottom = 390, 55, 72, 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = []
    for r in rows:
        est = to_float(r.get("estimate"))
        if est is None:
            continue
        values.append(est)
        lower = to_float(r.get("lower"))
        upper = to_float(r.get("upper"))
        if lower is not None:
            values.append(lower)
        if upper is not None:
            values.append(upper)
        values.append(0.0)
    x_min, x_max = axis_scale(values, 0.10, 0.10)
    if x_min >= 0:
        x_min = -0.01
    if x_max <= 0:
        x_max = 0.01

    def x_pos(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def y_pos(variable: str, method: str) -> float:
        i = variables.index(variable)
        base = top + i * row_h + row_h * 0.56
        if not methods:
            return base
        m_i = methods.index(method) if method in methods else 0
        spread = min(row_h * 0.72, 13.0)
        offset = 0.0 if len(methods) == 1 else (m_i - (len(methods) - 1) / 2) * (spread / max(1, len(methods) - 1))
        return base + offset

    def method_color(row: dict[str, object]) -> str:
        method = str(row.get("method", ""))
        if row.get("method_family") == "Bayes-soft-center":
            return PRIOR_COLORS.get(str(row.get("prior", "")), "#666")
        return BASELINE_METHOD_COLORS.get(method, "#666")

    def draw_baseline_marker(x: float, y: float, color: str, selected: bool, title: str) -> str:
        opacity = "0.90" if selected else "0.18"
        size = 4.2 if selected else 2.5
        return (
            f'<rect x="{x-size:.1f}" y="{y-size:.1f}" width="{2*size:.1f}" height="{2*size:.1f}" '
            f'fill="{color}" fill-opacity="{opacity}" transform="rotate(45 {x:.1f} {y:.1f})">'
            f'<title>{escape(title)}</title></rect>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(20, 25, f"{dataset_id}: all coefficient estimates by prior and method", 16, weight="700"),
        svg_text(20, 43, "Bayes priors show posterior means and 95% intervals; FLORAL/coda points show selected coefficients, with unselected baseline coefficients drawn at zero in low opacity.", 11, color="#555"),
        f'<line x1="{left}" x2="{width-right}" y1="{top+plot_h}" y2="{top+plot_h}" stroke="#333" stroke-width="1"/>',
    ]
    zero_x = x_pos(0.0)
    parts.append(f'<line x1="{zero_x:.1f}" x2="{zero_x:.1f}" y1="{top-8}" y2="{top+plot_h}" stroke="#555" stroke-width="1.1" stroke-dasharray="4 4"/>')
    tick_candidates = [x_min, 0.0, x_max]
    if x_min < x_max:
        tick_candidates.insert(1, x_min + (x_max - x_min) * 0.25)
        tick_candidates.insert(-1, x_min + (x_max - x_min) * 0.75)
    for tick in tick_candidates:
        x = x_pos(tick)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top+plot_h}" stroke="#eee" stroke-width="1"/>')
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top+plot_h}" y2="{top+plot_h+5}" stroke="#333"/>')
        parts.append(svg_text(x, top + plot_h + 18, f"{tick:.3g}", 10, anchor="middle"))
    for i, variable in enumerate(variables):
        y_mid = top + i * row_h + row_h * 0.56
        if i % 2 == 0:
            parts.append(f'<rect x="0" y="{top+i*row_h:.1f}" width="{width}" height="{row_h:.1f}" fill="#f8f8f8"/>')
        parts.append(svg_text(20, y_mid + 4, truncate_label(variable, 57), 10 if len(variables) > 50 else 11))
    parts.append(f'<line x1="{zero_x:.1f}" x2="{zero_x:.1f}" y1="{top-8}" y2="{top+plot_h}" stroke="#555" stroke-width="1.1" stroke-dasharray="4 4"/>')
    for r in rows:
        variable = str(r.get("variable", ""))
        method = str(r.get("method", ""))
        estimate = to_float(r.get("estimate"))
        if variable not in seen_vars or estimate is None:
            continue
        color = method_color(r)
        y = y_pos(variable, method)
        x = x_pos(estimate)
        selected = str(r.get("selected", "")).lower() == "true"
        title = f"{dataset_id} {method} {variable}: estimate={estimate:.5g}"
        if r.get("method_family") == "Bayes-soft-center":
            lower = to_float(r.get("lower"))
            upper = to_float(r.get("upper"))
            opacity = "0.88" if selected else "0.38"
            radius = 3.4 if selected else 2.4
            if lower is not None and upper is not None:
                title += f", 95% [{lower:.5g}, {upper:.5g}]"
                parts.append(f'<line x1="{x_pos(lower):.1f}" x2="{x_pos(upper):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}" stroke-opacity="{opacity}" stroke-width="1.3"/>')
            stroke = "#111" if selected else "none"
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="0.7"><title>{escape(title)}</title></circle>')
        else:
            parts.append(draw_baseline_marker(x, y, color, selected, title))
    legend_x = left
    legend_y = 57
    for method in methods:
        if legend_x > width - right - 140:
            legend_x = left
            legend_y += 17
        if method.startswith("Bayes_"):
            prior = method.replace("Bayes_", "", 1)
            color = PRIOR_COLORS.get(prior, "#666")
            parts.append(f'<circle cx="{legend_x:.1f}" cy="{legend_y:.1f}" r="4" fill="{color}" fill-opacity="0.75"/>')
            label = prior
        else:
            color = BASELINE_METHOD_COLORS.get(method, "#666")
            parts.append(f'<rect x="{legend_x-4:.1f}" y="{legend_y-4:.1f}" width="8" height="8" fill="{color}" transform="rotate(45 {legend_x:.1f} {legend_y:.1f})"/>')
            label = method
        parts.append(svg_text(legend_x + 9, legend_y + 4, label, 10))
        legend_x += max(78, len(label) * 6 + 28)
    parts.append(svg_text(left + plot_w / 2, height - 8, "coefficient", 11, anchor="middle"))
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(
    outdir: Path,
    performance: list[dict[str, object]],
    bayes_metrics: list[dict[str, object]],
    loo_rows: list[dict[str, object]],
    rhat_summary: list[dict[str, object]],
    best_rows: list[dict[str, object]],
    key_vars: list[dict[str, object]],
) -> None:
    n_datasets = len({str(r["dataset_id"]) for r in performance})
    n_bayes = len(bayes_metrics)
    rhat_bad = [r for r in rhat_summary if to_float(r.get("rhat_max")) is not None and (to_float(r.get("rhat_max")) or 0.0) > 1.05]
    best_cv_lines = []
    for row in best_rows:
        best_cv_lines.append(
            f"| {row['dataset_id']} | {row['best_cv_method']} | {fmt_num(row['best_cv_cindex'])} | "
            f"{row['best_bayes_loo_prior']} | {fmt_num(row['best_bayes_cv_cindex'])} | "
            f"{fmt_num(row['best_bayes_waic_pw2'])} | {fmt_num(row['best_bayes_lpml'])} | "
            f"{fmt_num(row['best_bayes_elpd_loo'])} | {fmt_num(row['best_bayes_looic'])} | "
            f"{row['best_bayes_pareto_k_bad']} | {fmt_num(row['best_bayes_rhat_max'])} |"
        )
    key_preview = []
    for dataset_id in sorted({str(r["dataset_id"]) for r in key_vars}):
        rows = [r for r in key_vars if r["dataset_id"] == dataset_id]
        rows = sorted(rows, key=lambda r: (-int(r.get("bayes_selected_count") or 0), str(r.get("variable", ""))))[:8]
        vars_text = "; ".join(str(r["variable"]) for r in rows) if rows else "none"
        key_preview.append(f"- **{dataset_id}**: {vars_text}")

    report = [
        "# Real-Data Analysis Summary: FLORAL, coda4microbiome, Bayes Soft-Center",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Sources",
        "",
        f"- FLORAL metrics: `{FLORAL_DIR / 'metrics.csv'}`",
        f"- coda4microbiome metrics: `{CODA_DIR / 'metrics.csv'}`",
        f"- Bayes soft-center results: `{BAYES_ROOT}`",
        "",
        "## Main Outputs",
        "",
        "- `all_tuning_and_baselines.csv`: all Bayesian tuning rows plus FLORAL min/1se and coda4microbiome lambda.min/lambda.1se.",
        "- `best_per_prior_and_baselines.csv`: best CV-Cindex tuning row within each Bayesian prior, plus FLORAL min/1se and coda4microbiome lambda.min/lambda.1se.",
        "- `by_dataset/<dataset>/`: dataset-specific copies of the two comparison tables and the coefficient long table.",
        "- `method_performance.csv`: unified CV-Cindex table for all methods, plus WAIC/LPML/LOO fields for Bayesian fits.",
        "- `bayes_model_metrics.csv`: Bayesian mean-estimate metrics by dataset and prior.",
        "- `bayes_loo_summary.csv`: Bayesian LOO ranking by dataset.",
        "- `bayes_rhat_by_parameter.csv` and `bayes_rhat_summary.csv`: beta-parameter convergence diagnostics.",
        "- `all_coefficients_for_forest.csv`: all coefficient rows used for the forest plots.",
        "- `key_variables.csv`: nonzero/key variables, method support, coefficient direction, and interpretation.",
        "- `figures/rhat_scatter.svg`, `figures/cv_cindex_comparison.svg`, and `figures/forest_*.svg`.",
        "",
        "## Overview",
        "",
        f"- Unified performance rows cover {n_datasets} canonical datasets.",
        f"- Bayesian soft-center summary contains {n_bayes} dataset-prior fits.",
        f"- Rhat was recomputed from `samples_wide.csv` beta draws. {len(rhat_bad)} dataset-prior fits have max Rhat > 1.05.",
        "- CV-Cindex is the cross-validated discrimination metric used to compare FLORAL, coda4microbiome, and tuned Bayesian priors. Bayesian WAIC/LPML/LOO are only comparable among Bayesian priors within the same dataset.",
        "",
        "## Dataset-Level Summary",
        "",
        "| Dataset | Best CV Method | Best CV-Cindex | Best Bayes LOO Prior | Bayes CV-Cindex | WAIC_pw2 | LPML | ELPD_LOO | LOOIC | ParetoK_bad | Best Bayes max Rhat |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *best_cv_lines,
        "",
        "## Key Variables",
        "",
        "Variables below are the union of Bayesian nonzero selections and FLORAL/coda4microbiome selected features. Coefficients are interpreted under the compositional/log-contrast constraint: positive values indicate higher relative abundance/proportion associated with higher hazard, while negative values indicate lower hazard. These are observational associations.",
        "",
        *key_preview,
        "",
        "## Diagnostics and Figures",
        "",
        "- Rhat scatter: `figures/rhat_scatter.svg`.",
        "- CV-Cindex comparison: `figures/cv_cindex_comparison.svg`.",
        "- Coefficient forest plots: `figures/forest_<dataset>.svg`; each dataset plot now overlays all Bayesian priors and available FLORAL/coda coefficients.",
        "",
        "## Notes",
        "",
        "- FLORAL and coda4microbiome provide cross-validated C-index and selected variables, but not WAIC/LPML/LOO.",
        "- Bayesian LOO quality should be checked with `ParetoK_bad`; nonzero counts indicate observations where PSIS-LOO may be less reliable.",
        "- T1D feature IDs such as `X681370` do not have taxonomy annotations in the current summary files, so their biological labels are not inferred.",
    ]
    (outdir / "realdata_analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    global BAYES_ROOT, FLORAL_DIR, CODA_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--bayes-root", type=Path, default=BAYES_ROOT)
    parser.add_argument("--floral-dir", type=Path, default=FLORAL_DIR)
    parser.add_argument("--coda-dir", type=Path, default=CODA_DIR)
    parser.add_argument("--skip-rhat", action="store_true", help="Do not recompute Rhat; useful for quick table-only runs.")
    parser.add_argument("--reuse-rhat", action="store_true", help="Reuse existing Rhat CSV files if present.")
    parser.add_argument("--r-env", default="zhangxf_r", help="Conda environment used for R/ggplot2 figure generation.")
    args = parser.parse_args(argv)

    BAYES_ROOT = args.bayes_root if args.bayes_root.is_absolute() else ROOT / args.bayes_root
    FLORAL_DIR = args.floral_dir if args.floral_dir.is_absolute() else ROOT / args.floral_dir
    CODA_DIR = args.coda_dir if args.coda_dir.is_absolute() else ROOT / args.coda_dir

    outdir = args.outdir
    figures = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    if not BAYES_ROOT.is_dir():
        if can_plot_from_existing_tables(outdir):
            print(f"Raw Bayes result root is missing: {BAYES_ROOT}. Replotting from existing summary CSVs.", flush=True)
            clean_figures_dir(outdir)
            run_r_plotting(outdir, args.r_env)
            print(f"Wrote summary to {outdir}")
            return 0
        raise FileNotFoundError(f"Raw Bayes result root is missing and existing summary CSVs are incomplete: {BAYES_ROOT}")

    baseline_performance, baseline_selected_coefficients, baseline_selected_methods = load_baseline_metrics()
    tuning_rows, best_cv_by_key = load_bayes_cv()
    coef_by_key = load_bayes_coefficients()
    bayes_metrics, bayes_performance = load_bayes_metrics(best_cv_by_key, coef_by_key)
    performance = baseline_performance + bayes_performance

    rhat_path = outdir / "bayes_rhat_by_parameter.csv"
    rhat_summary_path = outdir / "bayes_rhat_summary.csv"
    if args.skip_rhat:
        rhat_rows: list[dict[str, object]] = []
        rhat_summary: list[dict[str, object]] = []
    elif args.reuse_rhat and rhat_path.exists() and rhat_summary_path.exists():
        rhat_rows = read_csv(rhat_path)
        rhat_summary = read_csv(rhat_summary_path)
    else:
        rhat_rows, rhat_summary = compute_all_rhat(coef_by_key)

    coefficient_rows = build_all_coefficient_rows(coef_by_key, baseline_selected_coefficients, performance)
    all_comparison_rows, best_per_prior_rows = comparison_table_rows(tuning_rows, performance)

    clean_output_dir(outdir)
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "all_tuning_and_baselines.csv", all_comparison_rows)
    write_csv(outdir / "best_per_prior_and_baselines.csv", best_per_prior_rows)
    write_csv(outdir / "all_coefficients_for_forest.csv", coefficient_rows)
    if rhat_rows:
        write_csv(rhat_path, rhat_rows)
        write_csv(rhat_summary_path, rhat_summary)
    write_dataset_split_tables(outdir, all_comparison_rows, best_per_prior_rows, coefficient_rows)

    run_r_plotting(outdir, args.r_env)

    print(f"Wrote summary to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
