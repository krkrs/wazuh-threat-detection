"""Post-hoc analysis of the leave-one-scenario-out predictions.

Adds the analyses that a single training fold cannot compute:

* detector-identity granularity ablation (coarse detector key vs fine detector key),
* performance on detectors that never occur in the training scenarios (novel detectors),
* threshold sweeps (alert reduction versus recall) used for the figures,
* detector-level and attack-phase-level confusion counts.

Predictions are joined back to the feature tables on (scenario, ts_id).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]

import ml_common as mc  # type: ignore[import-not-found]
import train as train_module  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
KEEP_COLUMNS = ["scenario", "ts_id", "fine_key", "sig_key", "label", "phase", "y"]
KEY_COLUMNS = ["scenario", "ts_id", "fine_key"]
NOVEL_LIMIT = 30


def log(message: str) -> None:
    print(message, flush=True)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot read {path}: {error}") from error


def fine_detector_key(frame: Any) -> list[str]:
    sources = [str(item) for item in frame["source"].to_numpy()]
    sig_ids = [str(item) for item in frame["sig_id"].to_numpy()]
    descs = [str(item) for item in frame["rule_desc"].to_numpy()]
    return [f"{source}|{sig_id}|{desc}" for source, sig_id, desc in zip(sources, sig_ids, descs, strict=True)]


def load_features_with_keys() -> tuple[Any, Any]:
    """Slim feature table plus the (scenario, ts_id) -> fine detector key map.

    Uses the same loader as training, so the (scenario, ts_id) numbering matches the
    stored out-of-fold predictions exactly.
    """
    frame = mc.load_features()
    frame["fine_key"] = fine_detector_key(frame)
    slim = frame[KEEP_COLUMNS].copy()
    return slim, slim[KEY_COLUMNS].copy()


def merge_predictions(predictions: Any, features: Any) -> Any:
    try:
        return predictions.merge(features, on=["scenario", "ts_id"], how="left", validate="one_to_one")
    except (pd.errors.MergeError, ValueError) as error:
        raise SystemExit(f"prediction/feature join failed: {error}") from error


def prior_lookup(train_frame: Any, key_column: str) -> tuple[dict[str, float], float]:
    keys = [str(item) for item in train_frame[key_column].to_numpy()]
    y = np.asarray(train_frame["y"].to_numpy(), dtype=np.float64)
    totals: dict[str, float] = defaultdict(float)
    hits: dict[str, float] = defaultdict(float)
    for index, key in enumerate(keys):
        totals[key] += 1.0
        hits[key] += float(y[index])
    lookup = {key: hits[key] / totals[key] for key in totals}
    return lookup, float(y.mean())


def apply_prior(lookup: dict[str, float], global_prior: float, keys: list[str]) -> np.ndarray:
    return np.asarray([lookup.get(key, global_prior) for key in keys], dtype=np.float64)


def confusion_counts(y: Any, pred: Any) -> dict[str, int]:
    y_array = np.asarray(y)
    pred_array = np.asarray(pred)
    tp = int(((y_array == 1) & (pred_array == 1)).sum())
    fp = int(((y_array == 0) & (pred_array == 1)).sum())
    fn = int(((y_array == 1) & (pred_array == 0)).sum())
    tn = int(((y_array == 0) & (pred_array == 0)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def rates(counts: dict[str, int]) -> dict[str, Any]:
    tn = int(counts.get("tn", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))
    tp = int(counts.get("tp", 0))
    total = tn + fp + fn + tp
    positive = tp + fn
    negative = tn + fp
    predicted = tp + fp
    return {
        "n": total,
        "attack_true": positive,
        "background_true": negative,
        "predicted_attack": predicted,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": round(tp / max(predicted, 1), 4),
        "recall": round(tp / max(positive, 1), 4),
        "f1": round(2.0 * tp / max(2.0 * tp + fp + fn, 1), 4),
        "background_filtered_pct": round(100.0 * tn / max(negative, 1), 2),
        "alert_reduction_rate": round(1.0 - (fp + fn) / max(total, 1), 4),
        "fp_per_million_alerts": round(fp / max(total, 1) * 1_000_000, 1),
    }


def novel_metrics(
    y: Any,
    pred_model: Any,
    pred_coarse: Any,
    keys_fine: list[str],
    keys_coarse: list[str],
    seen_fine: set[str],
    seen_coarse: set[str],
    majority: int,
) -> tuple[Any, Any, dict[str, dict[str, int]]]:
    """Metrics restricted to detectors that never occur in the training scenarios."""
    y_array = np.asarray(y)
    mask_fine = np.asarray([key not in seen_fine for key in keys_fine], dtype=bool)
    mask_coarse = np.asarray([key not in seen_coarse for key in keys_coarse], dtype=bool)
    blocks = {
        "novel_model": confusion_counts(y_array[mask_fine], np.asarray(pred_model)[mask_fine]),
        "novel_prior_fine_majority": confusion_counts(
            y_array[mask_fine], np.full(int(mask_fine.sum()), majority)
        ),
        "novel_prior_coarse": confusion_counts(y_array[mask_coarse], np.asarray(pred_coarse)[mask_coarse]),
    }
    return mask_fine, mask_coarse, blocks


def main() -> None:
    metrics_path = ARTIFACT_DIR / "metrics_los_lgb_structured.json"
    predictions_path = ARTIFACT_DIR / "predictions_los_lgb_structured.parquet"
    for path in (metrics_path, predictions_path):
        if not path.exists():
            log(f"missing artifact: {path}")
            return
    metrics = load_json(metrics_path)
    try:
        predictions = pd.read_parquet(predictions_path)
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot read {predictions_path}: {error}") from error
    slim, key_table = load_features_with_keys()
    merged = merge_predictions(predictions, key_table)
    log(f"merged rows={len(merged)} missing_join={int(merged['fine_key'].isna().sum())}")
    rotation = train_module.rotation_map()

    names = [
        "model",
        "prior_coarse",
        "prior_fine",
        "majority",
        "novel_model",
        "novel_prior_coarse",
        "novel_prior_fine_majority",
    ]
    pooled_counts: dict[str, dict[str, int]] = {name: defaultdict(int) for name in names}
    fold_reports: list[dict[str, Any]] = []
    novel_detector_rows: list[dict[str, Any]] = []
    detector_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    phase_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_y: list[Any] = []
    all_prob: list[Any] = []

    for fold in metrics["folds"]:
        test_scenario = str(fold["fold"]).replace("los_", "")
        cal_scenario = rotation[test_scenario]
        train_scenarios = [item for item in mc.SCENARIOS if item not in (test_scenario, cal_scenario)]
        test_rows = merged[merged["scenario"] == test_scenario]
        cal_rows = slim[slim["scenario"] == cal_scenario]
        train_rows = slim[slim["scenario"].isin(train_scenarios)]

        coarse_lookup, coarse_global = prior_lookup(train_rows, "sig_key")
        fine_lookup, fine_global = prior_lookup(train_rows, "fine_key")
        seen_coarse = set(coarse_lookup)
        seen_fine = set(fine_lookup)
        majority = 1 if fine_global >= 0.5 else 0

        test_coarse = [str(item) for item in test_rows["sig_key"].to_numpy()]
        test_fine = [str(item) for item in test_rows["fine_key"].to_numpy()]
        prior_coarse_test = apply_prior(coarse_lookup, coarse_global, test_coarse)
        prior_fine_test = apply_prior(fine_lookup, fine_global, test_fine)
        cal_coarse = apply_prior(coarse_lookup, coarse_global, [str(item) for item in cal_rows["sig_key"].to_numpy()])
        cal_fine = apply_prior(fine_lookup, fine_global, [str(item) for item in cal_rows["fine_key"].to_numpy()])
        y_cal = np.asarray(cal_rows["y"].to_numpy())
        y_test = np.asarray(test_rows["y"].to_numpy())
        threshold_coarse = mc.best_f1_threshold(y_cal, cal_coarse)
        threshold_fine = mc.best_f1_threshold(y_cal, cal_fine)

        prob_model = np.asarray(test_rows["prob_model"].to_numpy())
        pred_model = np.asarray(test_rows["pred_model"].to_numpy())
        pred_coarse = np.where(prior_coarse_test >= threshold_coarse, 1, 0)
        pred_fine = np.where(prior_fine_test >= threshold_fine, 1, 0)

        blocks = {
            "model": confusion_counts(y_test, pred_model),
            "prior_coarse": confusion_counts(y_test, pred_coarse),
            "prior_fine": confusion_counts(y_test, pred_fine),
            "majority": confusion_counts(y_test, np.full(len(y_test), majority)),
        }
        for name, counts in blocks.items():
            for field, value in counts.items():
                pooled_counts[name][field] += value
        novel_mask_fine, novel_mask_coarse, novel_blocks = novel_metrics(
            y_test, pred_model, pred_coarse, test_fine, test_coarse, seen_fine, seen_coarse, majority
        )
        for name, counts in novel_blocks.items():
            for field, value in counts.items():
                pooled_counts[name][field] += value

        novel_by_key: dict[str, list[int]] = defaultdict(list)
        novel_seen: set[str] = set()
        for index, key in enumerate(test_fine):
            if key in seen_fine:
                continue
            novel_seen.add(key)
            row = novel_by_key[key]
            row.append(1)
            row.append(int(y_test[index]))
        for key, values in novel_by_key.items():
            novel_detector_rows.append(
                {
                    "fine_key": key,
                    "test_scenario": test_scenario,
                    "rows": len(values) // 2,
                    "attack_rows": int(sum(values[1::2])),
                }
            )

        for index, key in enumerate(test_fine):
            record = detector_counts[key]
            record["n"] += 1
            record["attack_true"] += int(y_test[index])
            record["predicted_attack"] += int(pred_model[index])
            if int(pred_model[index]) == 1 and int(y_test[index]) == 0:
                record["fp"] += 1
            if int(pred_model[index]) == 0 and int(y_test[index]) == 1:
                record["fn"] += 1
        labels = [str(item) for item in test_rows["label"].to_numpy()]
        phases = [str(item) for item in test_rows["phase"].to_numpy()]
        for index, phase_label in enumerate(f"{label}|{phase}" for label, phase in zip(labels, phases, strict=True)):
            record = phase_counts[phase_label]
            record["n"] += 1
            record["attack_true"] += int(y_test[index])
            record["predicted_attack"] += int(pred_model[index])
            if int(pred_model[index]) == 1 and int(y_test[index]) == 0:
                record["fp"] += 1
            if int(pred_model[index]) == 0 and int(y_test[index]) == 1:
                record["fn"] += 1

        fold_reports.append(
            {
                "fold": fold["fold"],
                "test_scenario": test_scenario,
                "cal_scenario": cal_scenario,
                "threshold_model": fold["threshold_model"],
                "threshold_prior_coarse": round(threshold_coarse, 6),
                "threshold_prior_fine": round(threshold_fine, 6),
                "model": rates(blocks["model"]),
                "prior_coarse": rates(blocks["prior_coarse"]),
                "prior_fine": rates(blocks["prior_fine"]),
                "majority": rates(blocks["majority"]),
                "novel_fine_rows": int(novel_mask_fine.sum()),
                "novel_coarse_rows": int(novel_mask_coarse.sum()),
                "novel_model": rates(novel_blocks["novel_model"]),
                "novel_prior_coarse": rates(novel_blocks["novel_prior_coarse"]),
                "novel_majority_prior": rates(novel_blocks["novel_prior_fine_majority"]),
            }
        )
        all_y.append(y_test)
        all_prob.append(prob_model)
        log(
            f"fold={test_scenario} rows={len(y_test)} novel_fine_rows={int(novel_mask_fine.sum())} "
            f"model_f1={rates(blocks['model'])['f1']} "
            f"prior_fine_f1={rates(blocks['prior_fine'])['f1']} prior_coarse_f1={rates(blocks['prior_coarse'])['f1']}"
        )

    y_all = np.concatenate(all_y)
    prob_all = np.concatenate(all_prob)
    sweep_rows = [
        {
            "threshold": round(float(threshold), 3),
            **rates(confusion_counts(y_all, np.where(prob_all >= threshold, 1, 0))),
        }
        for threshold in np.linspace(0.02, 0.98, 49)
    ]

    novel_detector_rows.sort(key=lambda item: item["rows"], reverse=True)
    detector_table: list[dict[str, Any]] = []
    for key, record in detector_counts.items():
        predicted = record.get("predicted_attack", 0)
        true_positive = predicted - record.get("fp", 0)
        detector_table.append(
            {
                "fine_key": key,
                "n": record.get("n", 0),
                "attack_true": record.get("attack_true", 0),
                "predicted_attack": predicted,
                "true_positive": true_positive,
                "fp": record.get("fp", 0),
                "fn": record.get("fn", 0),
                "precision": round(true_positive / predicted, 4) if predicted else 0.0,
                "recall": round(true_positive / record["attack_true"], 4) if record.get("attack_true") else 0.0,
            }
        )
    detector_table.sort(key=lambda item: item["fp"], reverse=True)
    payload = {
        "pooled": {name: rates(dict(counts)) for name, counts in pooled_counts.items()},
        "folds": fold_reports,
        "sweep_threshold": sweep_rows,
        "novel_detectors_top": novel_detector_rows[:NOVEL_LIMIT],
        "novel_detector_kinds": len({item["fine_key"] for item in novel_detector_rows}),
        "novel_detector_rows_total": sum(item["rows"] for item in novel_detector_rows),
        "detector_worst_fp": detector_table[:NOVEL_LIMIT],
        "detector_table": detector_table,
        "phase_table": {key: dict(value) for key, value in phase_counts.items()},
    }
    (ARTIFACT_DIR / "analysis_los.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name in names:
        log(f"POOLED {name}: {json.dumps(payload['pooled'][name])}")
    log(f"novel detector kinds across folds: {payload['novel_detector_kinds']}")


if __name__ == "__main__":
    main()
