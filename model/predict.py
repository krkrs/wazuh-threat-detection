"""Score Wazuh alert feature rows with the trained bundle.

Self-contained inference script: it reads only artifacts shipped in the ZIP
(deployed_model.joblib, feature_plan.json, metadata.json). Run it from the
unpacked bundle directory as `python predict.py` - no project code required.

Input: a Parquet feature table produced by research/build_features.py (one row per
alert; one row per scenario is enough for live scoring a window).

Fusion rule: alarm if the LightGBM model probability is above the deployment threshold,
OR the novelty detector score is above its calibrated threshold. Novelty is calibrated on
operational background rows only at NOVELTY_FP_RATE, so its alarm rate is bounded.

Usage:
    python predict.py --features data/features/shaw.features.parquet --out scores.parquet
    python predict.py --features window.parquet --out scores.parquet --evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]
from scipy.special import expit  # type: ignore[import-not-found]
from sklearn.metrics import precision_recall_curve  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parent


def positive_probabilities(model: Any, frame: Any) -> np.ndarray:
    raw: Any = model.predict(frame)
    table = np.asarray(raw, dtype=np.float64).reshape(-1)
    if float(np.min(table)) < 0.0 or float(np.max(table)) > 1.0:
        table = expit(table)
    return table


def build_matrix(frame: Any, plan: dict[str, Any], model: Any, meta: dict[str, Any]) -> Any:
    """Rebuild the feature matrix for scoring; mirrors ml_common.make_matrix.

    Text encoder is applied first so that it keeps the filtered frame's index, exactly as
    ml_common.make_matrix expects (concat is positional).
    """
    blocks: list[Any] = []
    numeric = list(plan["numeric_cols"])
    if numeric:
        blocks.append(frame[numeric].astype("float32").reset_index(drop=True))
    categorical = list(plan["categorical_cols"])
    if categorical:
        blocks.append(frame[categorical].astype("category").reset_index(drop=True))
    text_model = payload.get("text_model")
    if isinstance(text_model, dict) and text_model.get("vectorizer") is not None:
        vectorizer = text_model["vectorizer"]
        reducer = text_model["reducer"]
        matrix = vectorizer.transform([str(value) for value in frame["rule_desc"].to_numpy()])
        dense = reducer.transform(matrix).astype("float32")
        columns = {f"text_svd_{index}": dense[:, index] for index in range(dense.shape[1])}
        blocks.append(pd.DataFrame(columns, index=frame.index).reset_index(drop=True))
    return pd.concat(blocks, axis=1)


def best_f1_threshold(y_true: Any, prob: Any) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if len(thresholds) == 0:
        return 0.5
    denominator = np.maximum(precision[:-1] + recall[:-1], 1e-12)
    f1_values = 2.0 * precision[:-1] * recall[:-1] / denominator
    return float(thresholds[int(np.argmax(f1_values))])


parser = argparse.ArgumentParser(description="score alert rows with the bundled model")
parser.add_argument("--features", required=True, help="Parquet feature table (one row per alert)")
parser.add_argument("--out", required=True, help="output Parquet with per-alert scores")
parser.add_argument("--evaluate", action="store_true", help="label column present: report metrics")
parser.add_argument("--threshold", default=None, type=float, help="override deployment threshold")
parser.add_argument("--novelty-threshold", default=None, type=float, help="override novelty threshold")
args = parser.parse_args()

payload: Any = joblib.load(ROOT / "deployed_model.joblib")
if not isinstance(payload, dict) or "model" not in payload:
    print("bad artifact: deployed_model.joblib must hold a dict with a model key", file=sys.stderr)
    raise SystemExit(1)

plan_path = ROOT / "feature_plan.json"
meta_path = ROOT / "metadata.json"
if not plan_path.exists() or not meta_path.exists():
    print("missing artifacts: feature_plan.json and metadata.json must sit next to predict.py", file=sys.stderr)
    raise SystemExit(1)

plan: dict[str, Any] = {}
meta: dict[str, Any] = {}
try:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
except (OSError, ValueError) as error:
    print(f"cannot read artifacts: {error}", file=sys.stderr)
    raise SystemExit(1) from error

model = payload["model"]
frame = pd.read_parquet(args.features)
used_columns = list(plan["numeric_cols"]) + list(plan["categorical_cols"])
missing = [column for column in used_columns if column not in frame.columns]
if missing:
    print(f"missing feature columns in input: {missing[:10]}", file=sys.stderr)
    raise SystemExit(1)

matrix = build_matrix(frame, plan, model, meta)
prob = positive_probabilities(model, matrix)

novelty_forest = payload.get("novelty_forest")
novelty_cols = list(payload.get("novelty_cols") or [])
novelty_raw = meta.get("novelty")
novelty_info: dict[str, Any] = novelty_raw if isinstance(novelty_raw, dict) else {}
novelty_threshold = float(
    args.novelty_threshold if args.novelty_threshold is not None else novelty_info.get("threshold", float("inf"))
)
if novelty_forest is not None and novelty_cols:
    numeric = [column for column in novelty_cols if column in frame.columns]
    score_novelty = -np.asarray(
        novelty_forest.score_samples(frame[numeric].fillna(0).to_numpy(dtype=np.float32)),
        dtype=np.float64,
    )
else:
    score_novelty = np.zeros(len(frame), dtype=np.float64)
pred_novelty = (score_novelty >= novelty_threshold).astype(np.int8)

threshold = float(args.threshold if args.threshold is not None else meta["deployment_threshold"])
pred_model = np.where(prob >= threshold, 1, 0)
pred_fused = np.maximum(pred_model, pred_novelty)
fused_prob = np.where(pred_fused.astype(bool), np.maximum(prob, 0.999), prob)

out = frame[["scenario", "ts", "ts_id", "label", "phase", "sig_key"]].copy()
out["prob_model"] = prob
out["prob_fused"] = fused_prob
out["pred_model"] = pred_model
out["pred_novelty"] = pred_novelty
out["pred_fused"] = pred_fused
out["score_novelty"] = score_novelty
out["threshold_model"] = threshold
out["threshold_novelty"] = novelty_threshold
out.to_parquet(args.out, index=False)

total = int(len(frame))
n_attack = int(pred_fused.sum())
print(f"rows={total} threshold={threshold:.6f} novelty_threshold={novelty_threshold:.6f}")
print(f"escalated_attack={n_attack} escalated_share={round(100.0 * n_attack / max(total, 1), 2)}%")

if args.evaluate:
    if "label" not in frame.columns:
        print("cannot evaluate: label column missing", file=sys.stderr)
        raise SystemExit(1)
    y = np.asarray(frame["label"].astype("object").to_numpy() == "attack", dtype=np.int8)
    tp = int(np.sum((pred_fused == 1) & (y == 1)))
    fp = int(np.sum((pred_fused == 1) & (y == 0)))
    fn = int(np.sum((pred_fused == 0) & (y == 1)))
    tn = int(np.sum((pred_fused == 0) & (y == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    kept = int(pred_fused.sum())
    novelty_fp = int(np.sum((pred_novelty == 1) & (y == 0)))
    print(
        f"eval rows={total} TP={tp} FP={fp} TN={tn} FN={fn} "
        f"precision={round(precision, 4)} recall={round(recall, 4)} F1={round(f1, 4)}"
    )
    print(
        f"novelty arm: alarms={int(pred_novelty.sum())} novelty_FP={novelty_fp} "
        f"model_only_FP={int(np.sum((pred_model == 1) & (y == 0)))}"
    )
    print(
        f"kept_for_review={kept} share_kept={round(100.0 * kept / max(total, 1), 2)}% "
        f"FP_per_million_alerts={round(fp / max(total, 1) * 1_000_000, 1)}"
    )
