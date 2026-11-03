"""Figures for the report, built from the pipeline artifacts.

Outputs (PNG, 150 dpi):
  fig1_scenario_f1.png        per-scenario F1: model vs detector-prior baselines vs always-attack
  fig2_threshold_curve.png    pooled threshold sweep: alert reduction vs attack recall
  fig3_pr_curves.png          precision-recall curves per LOS scenario (model)
  fig4_top_detectors.png      largest detector-level false-positive / false-negative contributors
  fig5_phase_coverage.png     model recall by attack phase (top phases by volume)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib  # type: ignore[import-not-found]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]
from sklearn.metrics import average_precision_score, precision_recall_curve  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
FIGURE_DIR = ROOT / "report" / "figures"
CONFIG = "lgb_structured"
PHASE_TOP = 14


def log(message: str) -> None:
    print(message, flush=True)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot read {path}: {error}") from error


def read_predictions(config: str) -> Any:
    path = ARTIFACT_DIR / f"predictions_los_{config}.parquet"
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot read {path}: {error}") from error


def save_figure(fig: Any, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {path}")


def fig1_scenario_f1(analysis: Any) -> None:
    folds = analysis["folds"]
    labels = [str(item["test_scenario"]) for item in folds]
    model = [float(item["model"]["f1"]) for item in folds]
    prior_fine = [float(item["prior_fine"]["f1"]) for item in folds]
    majority = [float(item["majority"]["f1"]) for item in folds]
    x = np.arange(len(labels), dtype=float)
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.bar(x - width / 2, model, width, label="model (LOS)", color="#1f77b4")
    ax.bar(x + width / 2, prior_fine, width, label="detector-prior baseline", color="#ff7f0e")
    ax.axhline(max(majority), color="#8c564b", linestyle="--", linewidth=1.2, label="always-attack baseline")
    ax.axhline(
        float(analysis["pooled"]["model"]["f1"]),
        color="#2ca02c",
        linestyle=":",
        linewidth=1.2,
        label=f"model pooled F1 = {analysis['pooled']['model']['f1']:.3f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("F1 (attack class)")
    ax.set_title(f"Leave-one-scenario-out F1 per scenario ({CONFIG})")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, "fig1_scenario_f1.png")


def fig2_threshold_curve(analysis: Any) -> None:
    sweep = analysis["sweep_threshold"]
    recall = [float(item["recall"]) for item in sweep]
    arr = [float(item["alert_reduction_rate"]) for item in sweep]
    threshold = [float(item["threshold"]) for item in sweep]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(recall, arr, color="#1f77b4", lw=2, label="alert reduction rate")
    ax.set_xlabel("attack recall (pooled out-of-fold)")
    ax.set_ylabel("alert reduction rate (ARR)")
    ax.set_title(f"Operating points by decision threshold ({CONFIG}, pooled LOS)")
    for fraction in (0.05, 0.25, 0.5, 0.75, 0.95):
        index = int(np.argmin(np.abs(np.asarray(threshold) - fraction)))
        ax.annotate(
            f"t={threshold[index]:.2f}",
            (recall[index], arr[index]),
            fontsize=8,
            xytext=(4, -10),
            textcoords="offset points",
        )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save_figure(fig, "fig2_threshold_curve.png")


def fig3_pr_curves() -> None:
    frame = read_predictions(CONFIG)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for scenario in sorted(set(str(item) for item in frame["scenario"])):
        rows = frame[frame["scenario"] == scenario]
        y = np.asarray(rows["y"].to_numpy(), dtype=int)
        if len(set(y)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y, np.asarray(rows["prob_model"].to_numpy(), dtype=float))
        score = average_precision_score(y, np.asarray(rows["prob_model"].to_numpy(), dtype=float))
        ax.plot(recall, precision, lw=1.4, label=f"{scenario} (PR-AUC {score:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0.55, 1.01)
    ax.set_title(f"Precision-recall per LOS scenario, out-of-fold scores ({CONFIG})")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(alpha=0.3)
    save_figure(fig, "fig3_pr_curves.png")


def fig4_top_detectors(analysis: Any) -> None:
    table = analysis["detector_worst_fp"][:14]
    names = []
    for item in table:
        key = str(item["fine_key"])
        parts = key.split("|")
        label = parts[0]
        text = " ".join(parts[2:]) if len(parts) > 2 else key
        names.append(f"{label}: {text[:44]}")
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    indices = np.arange(len(names), dtype=float)
    fp = [int(item["fp"]) for item in table]
    fn = [int(item["fn"]) for item in table]
    ax.barh(indices - 0.2, fp, height=0.38, color="#d62728", label="false positives")
    ax.barh(indices + 0.2, fn, height=0.38, color="#1f77b4", label="false negatives")
    ax.set_yticks(indices)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("alerts (log scale)")
    ax.set_title(f"Top detector-level error contributors, pooled LOS ({CONFIG})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    save_figure(fig, "fig4_top_detectors.png")


def fig5_phase_coverage(analysis: Any) -> None:
    table = analysis["phase_table"]
    rows = []
    for key, record in table.items():
        true_attack = int(record.get("attack_true", 0))
        predicted = int(record.get("predicted_attack", 0))
        if true_attack < 500:
            continue
        true_positive = max(predicted - int(record.get("fp", 0)), 0)
        rows.append((key, true_attack, true_positive / true_attack))
    rows.sort(key=lambda item: item[1], reverse=True)
    rows = rows[:PHASE_TOP]
    names = [item[0] for item in rows]
    values = [item[2] for item in rows]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.barh(np.arange(len(names), dtype=float), values, color="#1f77b4")
    ax.set_yticks(np.arange(len(names), dtype=float))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("recall of the attack phase (model, pooled LOS)")
    ax.set_title(f"Model recall by attack phase, phases with >=500 attack alerts ({CONFIG})")
    ax.grid(axis="x", alpha=0.3)
    save_figure(fig, "fig5_phase_coverage.png")


def main() -> None:
    analysis = load_json(ARTIFACT_DIR / "analysis_los.json")
    fig1_scenario_f1(analysis)
    fig2_threshold_curve(analysis)
    fig3_pr_curves()
    fig4_top_detectors(analysis)
    fig5_phase_coverage(analysis)


if __name__ == "__main__":
    main()
