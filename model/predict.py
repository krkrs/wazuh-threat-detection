"""Score alert feature rows with the model bundled in this ZIP.

Input: a feature table produced by `research/build_features.py` (Parquet), i.e. one row
per alert with the 152 feature-table columns. The script mirrors the training-time
feature matrix construction (`ml_common.make_matrix`) and writes per-alert scores.

    python predict.py --features data/features/fox.features.parquet --out scores.csv
    python predict.py --features data/features/fox.features.parquet --evaluate

Columns of the output: the identifier columns of the input plus `prob_attack` and
`pred_attack` (1 = escalate to the analyst, 0 = filter out as operational background).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]

KEEP_OUTPUT = ["scenario", "ts", "source", "sig_key", "rule_desc", "label", "phase"]


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot read {path}: {error}") from error


def read_features(path: Path) -> Any:
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot read feature table {path}: {error}") from error


def positive_probabilities(model: Any, matrix: Any) -> np.ndarray:
    raw = model.predict_proba(matrix)
    array = np.asarray(raw)
    if array.ndim == 1:
        return array.astype(float)
    return array[:, 1].astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="feature table (Parquet) to score")
    parser.add_argument("--out", default="scores.csv", help="output CSV/Parquet path")
    parser.add_argument("--bundle", default=str(Path(__file__).resolve().parent), help="unzipped bundle root")
    parser.add_argument("--threshold", type=float, default=None, help="override the recorded decision threshold")
    parser.add_argument("--evaluate", action="store_true", help="report metrics if `label` is present")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N rows (smoke test)")
    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()
    inner = bundle / "model"
    plan = load_json(inner / "feature_plan.json")
    meta = load_json(inner / "metadata.json")
    payload = joblib.load(inner / "deployed_model.joblib")
    model = payload["model"]
    text_model = payload.get("text_model")
    uses_text = bool(meta.get("uses_text", text_model is not None))

    frame = read_features(Path(args.features))
    if args.limit > 0:
        frame = frame.head(args.limit)
    numeric_cols = [str(item) for item in plan["numeric_cols"]]
    categorical_cols = [str(item) for item in plan["categorical_cols"]]
    missing = [column for column in numeric_cols + categorical_cols if column not in frame.columns]
    if missing:
        raise SystemExit(f"feature table is missing {len(missing)} model columns, e.g. {missing[:8]}")

    blocks: list[Any] = [frame[numeric_cols].astype("float32").reset_index(drop=True)]
    blocks.append(frame[categorical_cols].astype("category").reset_index(drop=True))
    if uses_text:
        if not text_model:
            raise SystemExit("model expects text features but the bundle has none")
        vectorizer = text_model["vectorizer"]
        reducer = text_model["reducer"]
        matrix = vectorizer.transform([str(item) for item in frame["rule_desc"].to_numpy()])
        dense = reducer.transform(matrix).astype("float32")
        columns = {f"text_svd_{index}": dense[:, index] for index in range(dense.shape[1])}
        blocks.append(pd.DataFrame(columns).reset_index(drop=True))
    matrix = pd.concat(blocks, axis=1)

    probabilities = positive_probabilities(model, matrix)
    threshold = float(args.threshold) if args.threshold is not None else float(meta["deployment_threshold"])
    predictions = np.where(probabilities >= threshold, 1, 0)

    identifier = [column for column in KEEP_OUTPUT if column in frame.columns]
    out = frame[identifier].copy().reset_index(drop=True)
    out["prob_attack"] = probabilities
    out["pred_attack"] = predictions
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if out_path.suffix == ".parquet":
            out.to_parquet(out_path, compression="zstd")
        else:
            out.to_csv(out_path, index=False)
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot write {out_path}: {error}") from error

    total = int(len(out))
    escalated = int(predictions.sum())
    print(
        f"rows={total} threshold={threshold:.6f} escalated_attack={escalated} "
        f"({100.0 * escalated / max(total, 1):.2f} %) mean_prob={float(probabilities.mean()):.4f} "
        f"config={meta.get('config')} wrote={out_path}",
        flush=True,
    )

    if args.evaluate and "label" in frame.columns:
        y = np.asarray([1 if str(item) == "attack" else 0 for item in frame["label"].to_numpy()])
        tp = int(((y == 1) & (predictions == 1)).sum())
        fp = int(((y == 0) & (predictions == 1)).sum())
        fn = int(((y == 1) & (predictions == 0)).sum())
        tn = int(((y == 0) & (predictions == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
        arr = 1.0 - (fp + fn) / max(total, 1)
        print(
            f"EVAL n={total} tp={tp} fp={fp} fn={fn} tn={tn} precision={precision:.4f} recall={recall:.4f} "
            f"f1={f1:.4f} ARR={arr:.4f} background_filtered={100.0 * tn / max(tn + fp, 1):.2f} %",
            flush=True,
        )


if __name__ == "__main__":
    main()
