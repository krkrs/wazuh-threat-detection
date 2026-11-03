from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]
from sklearn.decomposition import TruncatedSVD  # type: ignore[import-not-found]
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]
from sklearn.metrics import (  # type: ignore[import-not-found]
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "features"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ROOT / "model"
SCENARIOS = ["fox", "harrison", "russellmitchell", "santos", "shaw", "wardbeck", "wheeler", "wilson"]
SEED = 42

CATEGORICAL_CANDIDATES = [
    "sig_key",
    "rule_desc",
    "rule_groups",
    "decoder",
    "decoder_parent",
    "location",
    "agent_name",
    "program_name",
    "sig_category",
    "sig_id",
    "sig_action",
    "sid_rev",
    "mitre_tactic",
    "mitre_technique",
    "http_method",
    "status_code",
    "dst_port",
    "src_ip_ctx",
    "dst_ip_ctx",
    "srcuser",
    "dstuser",
    "db_name",
    "audit_type",
    "syscall",
    "rule_md5",
    "log_resource_class",
    "ac_type",
    "ac_id",
    "ac_message",
    "persistence",
    "miner_id_ctx",
    "url_ext",
    "hostname_ctx",
    "agent_id",
]
NUMERIC_EXTRA = [
    "rule_level",
    "sig_severity",
    "src_port",
    "firedtimes",
    "log_lines_count",
    "is_training_mode",
    "attr_value_count",
    "attr_value_max_len",
    "attr_value_max_entropy",
    "attr_path_depth",
    "missing_char_count",
    "missing_char_special",
    "anomaly_score_max",
    "anomaly_score_mean",
    "critical_value",
    "probability_threshold",
    "count_confidence",
    "count_confidence_factor",
    "freq_actual",
    "freq_expected",
    "freq_window",
    "freq_range_lo",
    "freq_range_hi",
    "freq_confidence_factor",
    "range_lo",
    "range_hi",
    "range_width",
    "url_len",
    "url_parts",
    "url_query",
    "url_tokens",
    "url_encoded_chars",
    "url_semicolon",
    "url_traversal",
    "url_entropy",
    "url_max_token_len",
    "dns_len",
    "dns_labels",
    "dns_max_label_len",
    "dns_avg_label_len",
    "dns_digits",
    "dns_letters",
    "dns_digit_ratio",
    "dns_entropy",
    "hour",
    "sig_count_prev_60s",
    "sig_count_prev_600s",
    "group_ids",
    "group_suricata",
    "group_web",
    "group_accesslog",
    "group_syslog",
    "group_audit",
    "group_mitre",
    "is_aggregated",
    "sha256_present",
    "id_attr_numeric",
]
META_KEEP = ["scenario", "ts", "ts_id", "label", "phase", "sig_key", "source", "rule_desc", "y"]
CARDINALITY_MAX = 400
MISSING_SHARE_MAX = 0.999

LGB_BASE: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": SEED,
    "deterministic": True,
    "force_row_sample": True,
    "force_col_sample": True,
}
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "lgb_structured": {"kind": "lgb", "use_text": False, "params": dict(LGB_BASE)},
    "lgb_text": {"kind": "lgb", "use_text": True, "params": dict(LGB_BASE)},
}
TEXT_PARAMS: dict[str, Any] = {"max_features": 4000, "ngram_range": (1, 2), "min_df": 10, "sublinear_tf": True}
TEXT_SVD_DIM = 24
EARLY_STOP_ROUNDS = 40
FP_BUDGETS = [5_000, 20_000, 50_000]
QUANTS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
CHRONO_STEPS = [
    {"name": "walk50", "train_q": 0.5, "cal_q": 0.6, "test_q": 0.7},
    {"name": "walk70", "train_q": 0.7, "cal_q": 0.8, "test_q": 0.9},
    {"name": "walk90", "train_q": 0.9, "cal_q": 0.95, "test_q": None},
]


def log(message: str) -> None:
    print(message, flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return "unreadable"
    return digest.hexdigest()


def flag_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if re.match(r"^(urlflag_|ua_)", str(column))]


def load_features() -> Any:
    frames = [pd.read_parquet(FEATURE_DIR / f"{scenario}.features.parquet") for scenario in SCENARIOS]
    df: Any = pd.concat(frames, ignore_index=True)
    ts_values = np.asarray(df["ts"].to_numpy(), dtype=np.float64)
    scenario_values = np.asarray(df["scenario"].to_numpy())
    ts_id = np.zeros(len(df), dtype=np.int64)
    for scenario in SCENARIOS:
        positions = np.flatnonzero(scenario_values == scenario)
        order = np.argsort(ts_values[positions], kind="stable")
        ts_id[positions[order]] = np.arange(1, len(positions) + 1, dtype=np.int64)
    df["ts_id"] = ts_id
    df["y"] = np.where(np.asarray(df["label"].to_numpy()) == "attack", 1, 0).astype("int64")
    keep = META_KEEP + NUMERIC_EXTRA + CATEGORICAL_CANDIDATES + flag_columns(df)
    columns = [column for column in dict.fromkeys(keep) if column in df.columns]
    return df.loc[:, columns]


def prepare_features(df: Any) -> dict[str, Any]:
    numeric_cols: list[str] = []
    categorical_cols: dict[str, int] = {}
    dropped: dict[str, str] = {}
    for column in NUMERIC_EXTRA + flag_columns(df):
        if column not in df.columns:
            continue
        missing_value: Any = df[column].isna().mean()
        nunique_value: Any = df[column].nunique(dropna=True)
        missing = float(missing_value)
        nunique = int(nunique_value)
        if nunique <= 1 or (missing >= MISSING_SHARE_MAX and nunique <= 2):
            dropped[column] = f"numeric weak: nunique={nunique} missing={missing:.4f}"
            continue
        numeric_cols.append(column)
    for column in CATEGORICAL_CANDIDATES:
        if column not in df.columns:
            continue
        nunique_cat: Any = df[column].nunique(dropna=False)
        nunique = int(nunique_cat)
        if nunique <= 1:
            dropped[column] = "constant category"
            continue
        if nunique > CARDINALITY_MAX:
            dropped[column] = f"high cardinality: {nunique} > {CARDINALITY_MAX}"
            continue
        categorical_cols[column] = nunique
    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "dropped_features": dropped,
        "text_params": dict(TEXT_PARAMS),
        "text_svd_dim": TEXT_SVD_DIM,
    }


def make_matrix(df: Any, prepared: dict[str, Any], text: Any) -> Any:
    """Join numeric, categorical and text blocks on a positional (0..n-1) index.

    Index alignment matters: text blocks are built from a fresh range index, while
    frame-derived blocks keep the original filtered index, so both are reset here.
    """
    blocks: list[Any] = []
    numeric_cols = list(prepared["numeric_cols"])
    if numeric_cols:
        blocks.append(df[numeric_cols].astype("float32").reset_index(drop=True))
    categorical_cols = list(prepared["categorical_cols"])
    if categorical_cols:
        blocks.append(df[categorical_cols].astype("category").reset_index(drop=True))
    if text is not None:
        blocks.append(text.reset_index(drop=True))
    return pd.concat(blocks, axis=1)


def fit_text_transform(
    train_df: Any,
    targets: list[Any],
) -> tuple[list[Any], Any, Any]:
    vectorizer = TfidfVectorizer(
        max_features=int(TEXT_PARAMS["max_features"]),
        ngram_range=tuple(TEXT_PARAMS["ngram_range"]),
        min_df=int(TEXT_PARAMS["min_df"]),
        sublinear_tf=bool(TEXT_PARAMS["sublinear_tf"]),
        dtype=np.float64,
    )
    train_matrix = vectorizer.fit_transform([str(item) for item in train_df["rule_desc"].to_numpy()])
    reducer = TruncatedSVD(n_components=TEXT_SVD_DIM, random_state=SEED)
    reducer.fit(train_matrix)
    outputs: list[pd.DataFrame] = []
    for frame in targets:
        matrix = vectorizer.transform([str(item) for item in frame["rule_desc"].to_numpy()])
        dense = reducer.transform(matrix).astype("float32")
        columns = {f"text_svd_{index}": dense[:, index] for index in range(dense.shape[1])}
        outputs.append(pd.DataFrame(columns, index=frame.index))
    return outputs, vectorizer, reducer


def best_f1_threshold(y_true: Any, prob: Any) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if len(thresholds) == 0:
        return 0.5
    denominator = np.maximum(precision[:-1] + recall[:-1], 1e-12)
    f1_values = 2.0 * precision[:-1] * recall[:-1] / denominator
    index = int(np.argmax(f1_values))
    return float(thresholds[index])


def positive_probabilities(model: Any, frame: Any) -> np.ndarray:
    raw: Any = model.predict_proba(frame)
    table = np.asarray(raw, dtype=np.float64)
    if table.ndim == 1:
        return table.reshape(-1)
    return table[:, 1].reshape(-1)


def metrics_under_budget(y_true: Any, prob: Any, budget_per_million: int) -> dict[str, Any]:
    labels = np.asarray(y_true)
    scores = np.asarray(prob, dtype=np.float64)
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    fp_cum = np.cumsum(np.where(sorted_labels == 0, 1, 0))
    limit = budget_per_million * len(labels) / 1_000_000
    valid = np.flatnonzero(fp_cum <= limit)
    if len(valid) == 0:
        return {"recall_at_budget": 0.0, "threshold": 1.0, "false_positives": 0, "true_positives": 0}
    chosen = int(valid[-1])
    applied = max(0.0, float(scores[order][chosen]) - 1e-9)
    predictions = np.where(scores >= applied, 1, 0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    positive = int(tp + fn)
    return {
        "recall_at_budget": round(float(tp) / max(positive, 1), 4),
        "threshold": round(applied, 6),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "predicted_positives": int(tp + fp),
    }


def metrics_block(y_true: Any, y_pred: Any, prob: Any, threshold: float) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    total = int(len(y_true))
    negative = tn + fp
    positive = tp + fn
    predicted = tp + fp
    out: dict[str, Any] = {
        "n": total,
        "n_attack": positive,
        "n_background": negative,
        "n_predicted_attack": predicted,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": round(float(threshold), 6),
        "accuracy": round((tn + tp) / max(total, 1), 4),
        "base_rate_attack": round(positive / max(total, 1), 4),
        "predicted_attack_share": round(predicted / max(total, 1), 4),
        "precision_attack": round(tp / max(predicted, 1), 4),
        "recall_attack": round(tp / max(positive, 1), 4),
        "f1_attack": round(2.0 * tp / max(2.0 * tp + fp + fn, 1), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "background_filtered_share": round(tn / max(negative, 1), 4),
        "background_filtered_pct": round(100.0 * tn / max(negative, 1), 2),
        "false_positive_rate": round(fp / max(negative, 1), 4),
        "false_positives_per_million_alerts": round(fp / max(total, 1) * 1_000_000, 1),
        "false_positives_per_million_background": round(fp / max(negative, 1) * 1_000_000, 1),
        "alert_reduction_rate": round(1.0 - (fp + fn) / max(total, 1), 4),
        "noise_reduction_rate": round(tn / max(negative, 1), 4),
        "alerts_kept_for_review": predicted,
    }
    if len(np.unique(np.asarray(y_true))) == 2:
        out["pr_auc_attack"] = round(float(average_precision_score(y_true, prob)), 4)
        out["roc_auc"] = round(float(roc_auc_score(y_true, prob)), 4)
    for budget in FP_BUDGETS:
        out[f"budget_fp_per_m_{budget}"] = metrics_under_budget(y_true, prob, budget)
    return out


def detector_prior_probabilities(train_df: Any, eval_df: Any) -> np.ndarray:
    train_keys = np.asarray(train_df["sig_key"].to_numpy())
    train_y = np.asarray(train_df["y"].to_numpy(), dtype=np.float64)
    unique_keys, inverse = np.unique(train_keys, return_inverse=True)
    sums = np.bincount(inverse, weights=train_y, minlength=len(unique_keys))
    counts = np.bincount(inverse, minlength=len(unique_keys))
    share = sums / np.maximum(counts, 1)
    lookup = dict(zip([str(key) for key in unique_keys], [float(value) for value in share], strict=True))
    global_prior = float(np.mean(train_y))
    eval_keys = [str(key) for key in eval_df["sig_key"].to_numpy()]
    return np.asarray([lookup.get(key, global_prior) for key in eval_keys], dtype=np.float64)


def detector_error_table(test_df: Any, y_pred: Any, limit: int = 40) -> list[dict[str, Any]]:
    keys = [str(key) for key in test_df["sig_key"].to_numpy()]
    y = np.asarray(test_df["y"].to_numpy())
    pred = np.asarray(y_pred)
    unique_keys, inverse = np.unique(np.asarray(keys), return_inverse=True)
    size = len(unique_keys)
    tp = np.bincount(inverse, weights=np.where((y == 1) & (pred == 1), 1.0, 0.0), minlength=size)
    fp = np.bincount(inverse, weights=np.where((y == 0) & (pred == 1), 1.0, 0.0), minlength=size)
    fn = np.bincount(inverse, weights=np.where((y == 1) & (pred == 0), 1.0, 0.0), minlength=size)
    total = np.bincount(inverse, minlength=size)
    predicted = np.bincount(inverse, weights=np.where(pred == 1, 1.0, 0.0), minlength=size)
    descriptions: dict[str, dict[str, str]] = {}
    for key, source, desc in zip(keys, test_df["source"].to_numpy(), test_df["rule_desc"].to_numpy(), strict=True):
        if key not in descriptions:
            descriptions[key] = {"source": str(source), "desc": str(desc)}
    order = np.argsort(-fp)
    records: list[dict[str, Any]] = []
    for index in order[:limit]:
        key = str(unique_keys[index])
        attack_true = int(tp[index] + fn[index])
        records.append(
            {
                "sig_key": key,
                "source": descriptions[key]["source"],
                "detector": descriptions[key]["desc"],
                "n": int(total[index]),
                "attack_true": attack_true,
                "predicted_attack": int(predicted[index]),
                "tp": int(tp[index]),
                "fp": int(fp[index]),
                "fn": int(fn[index]),
                "precision": round(float(tp[index]) / max(int(float(predicted[index])), 1), 4),
                "recall": round(float(tp[index]) / max(attack_true, 1), 4),
                "true_background": int(total[index] - attack_true),
            }
        )
    return records


def phase_error_table(test_df: Any, y_pred: Any) -> dict[str, dict[str, int]]:
    labels = [
        f"{label}|{phase}"
        for label, phase in zip(test_df["label"].to_numpy(), test_df["phase"].to_numpy(), strict=True)
    ]
    y_all = np.asarray(test_df["y"].to_numpy())
    pred = np.asarray(y_pred)
    unique_labels, inverse = np.unique(np.asarray(labels), return_inverse=True)
    size = len(unique_labels)
    total = np.bincount(inverse, minlength=size)
    predicted = np.bincount(inverse, weights=np.where(pred == 1, 1.0, 0.0), minlength=size)
    out: dict[str, dict[str, int]] = {}
    for index, label in enumerate(unique_labels):
        members = inverse == index
        out[str(label)] = {
            "n": int(total[index]),
            "kept_as_attack": int(predicted[index]),
            "filtered_out": int(total[index] - predicted[index]),
            "attack_true": int(y_all[members].sum()),
        }
    return out


def evaluate_fold(
    model_cfg: dict[str, Any], prepared: dict[str, Any], train_df: Any, cal_df: Any, test_df: Any, fold_name: str
) -> dict[str, Any]:
    y_train = np.asarray(train_df["y"].to_numpy())
    y_cal = np.asarray(cal_df["y"].to_numpy())
    y_test = np.asarray(test_df["y"].to_numpy())
    if model_cfg.get("use_text"):
        text_all, _, _ = fit_text_transform(train_df, [train_df, cal_df, test_df])
        text_train, text_cal, text_test = text_all[0], text_all[1], text_all[2]
    else:
        text_train = None
        text_cal = None
        text_test = None
    x_train = make_matrix(train_df, prepared, text_train)
    x_cal = make_matrix(cal_df, prepared, text_cal)
    x_test = make_matrix(test_df, prepared, text_test)
    params = dict(model_cfg["params"])
    model = lgb.LGBMClassifier(**params)
    began = time.time()
    model.fit(
        x_train,
        y_train,
        eval_X=x_cal,
        eval_y=y_cal,
        eval_metric=["binary_logloss", "binary_error"],
        callbacks=[lgb.callback.early_stopping(EARLY_STOP_ROUNDS, verbose=False)],  # type: ignore[attr-defined]
    )
    fit_seconds = round(time.time() - began, 2)
    best_iteration_value: Any = getattr(model, "best_iteration_", None)
    best_iteration = int(best_iteration_value) if best_iteration_value else int(params["n_estimators"])
    prob_cal = positive_probabilities(model, x_cal)
    prob_test = positive_probabilities(model, x_test)
    threshold = best_f1_threshold(y_cal, prob_cal)
    pred_test = np.where(prob_test >= threshold, 1, 0)

    prob_prior_cal = detector_prior_probabilities(train_df, cal_df)
    prior_threshold = best_f1_threshold(y_cal, prob_prior_cal)
    prob_prior_test = detector_prior_probabilities(train_df, test_df)
    prior_pred_test = np.where(prob_prior_test >= prior_threshold, 1, 0)
    const_prob = np.ones_like(y_test, dtype=np.float64)

    booster = model.booster_
    feature_names = list(booster.feature_name())
    gain_pairs = sorted(
        zip(feature_names, [float(v) for v in booster.feature_importance("gain")], strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    split_pairs = sorted(
        zip(feature_names, [int(v) for v in booster.feature_importance("split")], strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    importance = {
        "gain": [{"feature": name, "importance": round(value, 2)} for name, value in gain_pairs[:40]],
        "split": [{"feature": name, "importance": value} for name, value in split_pairs[:40]],
    }
    test_predictions = test_df[["ts", "ts_id", "scenario", "phase", "sig_key", "label", "y"]].copy()
    test_predictions["prob_model"] = prob_test
    test_predictions["prob_prior_baseline"] = prob_prior_test
    test_predictions["pred_model"] = pred_test
    test_predictions["pred_prior_baseline"] = prior_pred_test
    test_predictions["threshold_model"] = threshold
    cal_predictions = cal_df[["ts", "ts_id", "scenario", "label", "y"]].copy()
    cal_predictions["prob_model"] = prob_cal
    cal_predictions["prob_prior_baseline"] = prob_prior_cal
    cal_predictions["pred_model"] = np.where(prob_cal >= threshold, 1, 0)
    cal_predictions["pred_prior_baseline"] = np.where(prob_prior_cal >= prior_threshold, 1, 0)
    cal_predictions["threshold_model"] = threshold
    return {
        "cal_predictions": cal_predictions,
        "fold": fold_name,
        "train_rows": int(len(train_df)),
        "cal_rows": int(len(cal_df)),
        "test_rows": int(len(test_df)),
        "best_iteration": best_iteration,
        "fit_seconds": fit_seconds,
        "threshold_model": round(threshold, 6),
        "threshold_prior_baseline": round(prior_threshold, 6),
        "model_metrics": metrics_block(y_test, pred_test, prob_test, threshold),
        "model_metrics_cal": metrics_block(y_cal, np.where(prob_cal >= threshold, 1, 0), prob_cal, threshold),
        "baseline_detector_prior": metrics_block(y_test, prior_pred_test, prob_prior_test, prior_threshold),
        "baseline_const_attack": metrics_block(y_test, np.ones_like(y_test), const_prob, 1.0),
        "detector_top_errors": detector_error_table(test_df, pred_test),
        "phase_errors": phase_error_table(test_df, pred_test),
        "feature_importance": importance,
        "predictions": test_predictions,
        "model": model,
    }
