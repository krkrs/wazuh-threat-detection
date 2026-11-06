"""Finalize the deployment package for delivery.

Steps (idempotent):
  1. Unpack the chosen bundle (model/model.zip) into model/ so the inference
     script, README, requirements and metadata sit next to the archive.
  2. Write model/metadata.json and model/thresholds.json in the flat shape the
     report consumes (zip hash/size, deployment threshold, holdout metrics, FP budgets).
  3. Rebuild artifacts/train_summary.json from the per-protocol metrics files and
     the bundle manifests, so a single small JSON summarizes the whole pipeline.
  4. Emit SHA-256 of every deliverable into artifacts/checksums.sha256.

Usage: .venv/bin/python -u research/finalize_bundle.py [--config lgb_structured]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ROOT / "model"
FEATURE_DIR = ROOT / "data" / "features"
CONFIGS = ("lgb_structured", "lgb_text")
PHASE_MIN_ROWS = 1000
FEATURE_FILES = ("build_features.py", "ml_common.py", "train.py", "report_stats.py", "analyze_predictions.py")


def log(message: str) -> None:
    print(message, flush=True)


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        log(f"skip unreadable {path.name}: {error}")
        return {}


def bundle_name(config: str) -> str:
    path = MODEL_DIR / f"model_{config}.zip"
    return path.name if path.exists() else "model.zip"


def unpack(config: str) -> dict[str, Any]:
    name = bundle_name(config)
    archive = MODEL_DIR / name
    if not archive.exists():
        raise SystemExit(f"missing bundle: {archive}")
    MODEL_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        # Flatten the "model/" prefix so artifacts sit directly under model/.
        for info in bundle.infolist():
            relative = info.filename
            if relative.startswith("model/"):
                relative = relative[len("model/") :]
            if not relative:
                continue
            target = MODEL_DIR / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(info))
        try:
            metadata = json.loads(bundle.read("model/metadata.json").decode("utf-8"))
        except (KeyError, ValueError, UnicodeDecodeError) as error:
            raise SystemExit(f"bundle {name} has no readable metadata.json: {error}") from error
    shutil.rmtree(MODEL_DIR / "model", ignore_errors=True)
    log(f"unpacked {name} into {MODEL_DIR} (flattened)")
    flat = dict(metadata)
    flat["zip_path"] = f"model/{name}"
    flat["zip_bytes"] = archive.stat().st_size
    flat["sha256"] = sha256(archive)
    metrics = metadata.get("deployment_metrics", {})
    if not metrics:
        metrics = metadata.get("holdout_metrics_same_distribution", {})
    flat["deployment_metrics"] = metrics
    flat["deployment_threshold"] = metadata.get("deployment_threshold", 0.0)
    flat["train_rows"] = metadata.get("n_rows_trained", 0)
    flat["cal_rows"] = metadata.get("n_rows_holdout", 0)
    flat["n_rows_trained"] = metadata.get("n_rows_trained", 0)
    flat["n_rows_calibration"] = metadata.get("n_rows_holdout", 0)
    flat["best_iteration_calibration"] = metadata.get("best_iteration", 0)
    flat["best_iteration_full_data"] = metadata.get("best_iteration_full_data", 0)
    flat["n_rows_training_mode"] = metadata.get("n_rows_training_mode", 0)
    flat["deploy_seconds"] = metadata.get("deploy_seconds", 0.0)
    flat["best_iteration"] = metadata.get("best_iteration", metadata.get("best_iteration_full_data", 0))
    flat["interim_fit_seconds"] = metadata.get("interim_fit_seconds", 0.0)
    flat["final_fit_seconds"] = metadata.get("final_fit_seconds", 0.0)
    flat["budgets"] = metadata.get("budgets", {})
    (MODEL_DIR / "thresholds.json").write_text(
        json.dumps(
            {
                "deployment": metadata.get("deployment_threshold", 0.0),
                "budgets": metadata.get("budgets", {}),
                "per_fold": metadata.get("thresholds", {}),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (MODEL_DIR / "metadata.json").write_text(json.dumps(flat, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {MODEL_DIR / 'metadata.json'} and {MODEL_DIR / 'thresholds.json'}")
    return flat


def phase_summary() -> dict[str, Any]:
    counts: dict[str, int] = {}
    for path in sorted(FEATURE_DIR.glob("*.features.parquet")):
        try:
            frame = pd.read_parquet(path, columns=["label", "phase"])
        except (OSError, ValueError) as error:
            log(f"skip {path.name}: {error}")
            continue
        labels = frame["label"].to_numpy()
        phases = frame["phase"].to_numpy()
        for index, phase in enumerate(phases):
            if labels[index] == "attack":
                key = str(phase)
                counts[key] = counts.get(key, 0) + 1
    kept = {key: value for key, value in counts.items() if value >= PHASE_MIN_ROWS}
    return {
        "attack_phase_rows_min_1000": len(kept),
        "attack_phase_rows_total": int(sum(kept.values())),
        "top_phases": sorted(kept.items(), key=lambda item: item[1], reverse=True)[:10],
    }


def build_summary(config: str) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for protocol, source in (("los", "aggregate_model"), ("chrono", "aggregate_model")):
        for cfg in CONFIGS:
            metrics = load_json(ARTIFACT_DIR / f"metrics_{protocol}_{cfg}.json")
            block = metrics.get(source)
            if not block:
                continue
            runs.append(
                {
                    "protocol": protocol,
                    "config": cfg,
                    "kind": "model",
                    "f1": block.get("f1", 0.0),
                    "pr_auc": block.get("pr_auc", 0.0),
                    "alert_reduction_rate": block.get("alert_reduction_rate", 0.0),
                    "background_filtered_pct": block.get("background_filtered_pct", 0.0),
                    "false_positives": block.get("fp", 0),
                    "false_negatives": block.get("fn", 0),
                    "fp_per_million_alerts": block.get("fp_per_million_alerts", 0.0),
                }
            )
            prior = metrics.get("aggregate_prior_baseline")
            if protocol == "los" and prior:
                runs.append(
                    {
                        "protocol": protocol,
                        "config": cfg,
                        "kind": "baseline_detector_prior",
                        "f1": prior.get("f1", 0.0),
                        "pr_auc": prior.get("pr_auc", 0.0),
                        "alert_reduction_rate": prior.get("alert_reduction_rate", 0.0),
                        "background_filtered_pct": prior.get("background_filtered_pct", 0.0),
                        "false_positives": prior.get("fp", 0),
                        "false_negatives": prior.get("fn", 0),
                        "fp_per_million_alerts": prior.get("fp_per_million_alerts", 0.0),
                    }
                )
    deployed: list[dict[str, Any]] = []
    for cfg in CONFIGS:
        manifest = load_json(ARTIFACT_DIR / f"model_manifest_{cfg}.json")
        if not manifest:
            continue
        deployed.append(
            {
                "config": cfg,
                "zip": manifest.get("zip_path", ""),
                "sha256": manifest.get("sha256", ""),
                "zip_bytes": manifest.get("zip_bytes", 0),
                "threshold": manifest.get("deployment_threshold", 0.0),
                "train_rows": manifest.get("n_rows_trained", 0),
                "holdout_rows": manifest.get("n_rows_holdout", 0),
                "deploy_seconds": manifest.get("deploy_seconds", 0.0),
                "primary": bool(manifest.get("primary")),
            }
        )
    plan = load_json(ARTIFACT_DIR / "feature_plan.json")
    summary = {
        "created_utc": utc_now(),
        "deploy_config": config,
        "runs": runs,
        "deployed": deployed,
        "total_seconds": float(sum(float(item.get("deploy_seconds", 0.0)) for item in deployed)),
        "feature_plan_summary": {
            "numeric": len(plan.get("numeric", [])),
            "categorical": len(plan.get("categorical", [])),
            "dropped": len(plan.get("dropped", [])),
        },
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    (ARTIFACT_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(f"wrote artifacts/train_summary.json (runs={len(runs)}, deployed={len(deployed)})")
    return summary


def checksums(config: str) -> None:
    targets = [
        MODEL_DIR / bundle_name(config),
        MODEL_DIR / "deployed_model.joblib",
        MODEL_DIR / "metadata.json",
        MODEL_DIR / "thresholds.json",
        MODEL_DIR / "predict.py",
        MODEL_DIR / "README.md",
        MODEL_DIR / "requirements.txt",
        ARTIFACT_DIR / "feature_plan.json",
        ARTIFACT_DIR / "train_summary.json",
        ARTIFACT_DIR / "analysis_los.json",
        ARTIFACT_DIR / "table_stats.json",
        ROOT / "report" / "sprawozdanie.md",
        ROOT / "research" / "RESEARCH.md",
    ]
    targets += [ROOT / "research" / name for name in FEATURE_FILES]
    targets += sorted(FEATURE_DIR.glob("*.features.parquet"))
    lines = []
    for path in targets:
        if not path.exists():
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(ROOT)}")
    (ARTIFACT_DIR / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote artifacts/checksums.sha256 ({len(lines)} entries)")


def main() -> None:
    parser = argparse.ArgumentParser(description="finalize the deployment package")
    parser.add_argument("--config", default="lgb_structured", choices=list(CONFIGS))
    args = parser.parse_args()

    unpack(args.config)
    build_summary(args.config)
    phases = phase_summary()
    (ARTIFACT_DIR / "phase_summary.json").write_text(json.dumps(phases, indent=2) + "\n", encoding="utf-8")
    log(f"phases >=1000 rows: {phases['attack_phase_rows_min_1000']} kinds")
    checksums(args.config)


if __name__ == "__main__":
    main()
