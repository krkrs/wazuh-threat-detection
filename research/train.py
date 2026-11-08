from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
import zipfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-not-found]
import lightgbm as lgb  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]

import ml_common as mc  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]

fold_type = dict[str, Any]
result_type = dict[str, Any]


def log(message: str) -> None:
    print(message, flush=True)


def rotation_map() -> dict[str, str]:
    rng = np.random.default_rng(mc.SEED)
    order = list(mc.SCENARIOS)
    rng.shuffle(order)
    rotation: dict[str, str] = {}
    for index, scenario in enumerate(order):
        rotation[scenario] = order[(index + 1) % len(order)]
    return rotation


def run_los(df: Any, prepared: dict[str, Any], cfg_name: str) -> result_type:
    cfg = mc.MODEL_CONFIGS[cfg_name]
    rotation = rotation_map()
    scenario_values = np.asarray(df["scenario"].to_numpy())
    fold_reports: list[fold_type] = []
    test_chunks: list[pd.DataFrame] = []
    cal_chunks: list[pd.DataFrame] = []
    started = time.time()
    for test_scenario in mc.SCENARIOS:
        cal_scenario = rotation[test_scenario]
        test_mask = scenario_values == test_scenario
        cal_mask = scenario_values == cal_scenario
        train_mask = ~(test_mask | cal_mask)
        fold = mc.evaluate_fold(
            cfg,
            prepared,
            df.loc[train_mask].copy(),
            df.loc[cal_mask].copy(),
            df.loc[test_mask].copy(),
            f"los_{test_scenario}",
        )
        test_chunks.append(fold.pop("predictions"))
        cal_chunks.append(fold.pop("cal_predictions"))
        fold.pop("model")
        fold_reports.append(fold)
        log(
            f"fold={test_scenario} n_test={fold['test_rows']} best_iter={fold['best_iteration']} "
            f"f1={fold['model_metrics']['f1_attack']:.4f} arr={fold['model_metrics']['alert_reduction_rate']:.4f} "
            f"prior_f1={fold['baseline_detector_prior']['f1_attack']:.4f} prior_arr="
            f"{fold['baseline_detector_prior']['alert_reduction_rate']:.4f} "
            f"novelty_f1={fold['novelty_metrics']['f1_attack']:.4f}"
        )
    fit_seconds = round(time.time() - started, 1)
    predictions = pd.concat(test_chunks, ignore_index=True)
    cal_predictions = pd.concat(cal_chunks, ignore_index=True)
    per_fold_thresholds = {fold["fold"]: fold["threshold_model"] for fold in fold_reports}
    per_fold_novelty = {fold["fold"]: fold.get("threshold_novelty") for fold in fold_reports}
    test_share = len(predictions) / max(len(df), 1)
    pooled_all = mc.pooled_stream_metrics(predictions, cal_predictions, test_share)
    deployment_threshold = pooled_all["thresholds"]["model"]
    prior_threshold = pooled_all["thresholds"]["prior"]
    novelty_threshold = pooled_all["thresholds"]["novelty"]
    pooled = pooled_all["blocks"]["model"]
    pooled_prior = pooled_all["blocks"]["prior_baseline"]
    pooled_lgb = pooled_all["blocks"]["model_lgb"]
    pooled_novelty = pooled_all["blocks"]["novelty"]
    pooled_budgets = pooled_all["budgets"]
    return {
        "protocol": "leave_one_scenario_out",
        "config": cfg_name,
        "seed": mc.SEED,
        "fit_total_seconds": fit_seconds,
        "deployment_threshold": round(deployment_threshold, 6),
        "prior_baseline_threshold": round(prior_threshold, 6),
        "per_fold_thresholds": per_fold_thresholds,
        "per_fold_novelty_thresholds": per_fold_novelty,
        "novelty_threshold": round(novelty_threshold, 6),
        "aggregate_model": pooled,
        "aggregate_model_budgets": pooled_budgets,
        "aggregate_prior_baseline": pooled_prior,
        "aggregate_model_lgb": pooled_lgb,
        "aggregate_novelty": pooled_novelty,
        "folds": fold_reports,
        "_predictions": predictions,
    }


def chronologic_masks(df: Any) -> dict[str, dict[str, np.ndarray]]:
    ts = np.asarray(df["ts"].to_numpy(), dtype=np.float64)
    scenario_values = np.asarray(df["scenario"].to_numpy())
    bounds: dict[str, np.ndarray] = {}
    for scenario in mc.SCENARIOS:
        values = np.sort(ts[scenario_values == scenario])
        bounds[scenario] = np.quantile(values, mc.QUANTS)
    out: dict[str, dict[str, np.ndarray]] = {}
    for step in mc.CHRONO_STEPS:
        train_mask = np.zeros(len(df), dtype=bool)
        cal_mask = np.zeros(len(df), dtype=bool)
        test_mask = np.zeros(len(df), dtype=bool)
        train_hi = mc.QUANTS.index(step["train_q"])
        cal_hi = mc.QUANTS.index(step["cal_q"])
        for scenario in mc.SCENARIOS:
            positions = np.flatnonzero(scenario_values == scenario)
            values = ts[positions]
            upper_train = bounds[scenario][train_hi]
            upper_cal = bounds[scenario][cal_hi]
            if step["test_q"] is None:
                upper_test = np.inf
            else:
                upper_test = bounds[scenario][mc.QUANTS.index(step["test_q"])]
            train_mask[positions[values < upper_train]] = True
            cal_mask[positions[(values >= upper_train) & (values < upper_cal)]] = True
            test_mask[positions[(values >= upper_cal) & (values < upper_test)]] = True
        out[step["name"]] = {"train": train_mask, "cal": cal_mask, "test": test_mask}
    return out


def run_chrono(df: Any, prepared: dict[str, Any], cfg_name: str) -> result_type:
    cfg = mc.MODEL_CONFIGS[cfg_name]
    masks = chronologic_masks(df)
    steps: list[fold_type] = []
    cal_chunks: list[pd.DataFrame] = []
    test_chunks: list[pd.DataFrame] = []
    started = time.time()
    for step in mc.CHRONO_STEPS:
        mask = masks[step["name"]]
        fold = mc.evaluate_fold(
            cfg,
            prepared,
            df.loc[mask["train"]].copy(),
            df.loc[mask["cal"]].copy(),
            df.loc[mask["test"]].copy(),
            f"chrono_{step['name']}",
        )
        test_chunks.append(fold.pop("predictions"))
        cal_chunks.append(fold.pop("cal_predictions"))
        fold.pop("model")
        steps.append(fold)
        log(
            f"step={step['name']} n_test={fold['test_rows']} best_iter={fold['best_iteration']} "
            f"f1={fold['model_metrics']['f1_attack']:.4f} prauc={fold['model_metrics'].get('pr_auc_attack', 0.0):.4f} "
            f"prior_f1={fold['baseline_detector_prior']['f1_attack']:.4f}"
        )
    fit_seconds = round(time.time() - started, 1)
    predictions = pd.concat(test_chunks, ignore_index=True)
    cal_predictions = pd.concat(cal_chunks, ignore_index=True)
    test_share = len(predictions) / max(len(df), 1)
    pooled_all = mc.pooled_stream_metrics(predictions, cal_predictions, test_share)
    threshold = pooled_all["thresholds"]["model"]
    per_fold_thresholds = {step["fold"]: step["threshold_model"] for step in steps}
    per_fold_novelty = {step["fold"]: step.get("threshold_novelty") for step in steps}
    return {
        "protocol": "chronologic_expanding_window",
        "config": cfg_name,
        "seed": mc.SEED,
        "fit_total_seconds": fit_seconds,
        "deployment_threshold": round(threshold, 6),
        "prior_baseline_threshold": round(pooled_all["thresholds"]["prior"], 6),
        "novelty_threshold": round(pooled_all["thresholds"]["novelty"], 6),
        "per_fold_thresholds": per_fold_thresholds,
        "per_fold_novelty_thresholds": per_fold_novelty,
        "aggregate_model": pooled_all["blocks"]["model"],
        "aggregate_model_budgets": pooled_all["budgets"],
        "aggregate_model_lgb": pooled_all["blocks"]["model_lgb"],
        "aggregate_prior_baseline": pooled_all["blocks"]["prior_baseline"],
        "aggregate_novelty": pooled_all["blocks"]["novelty"],
        "steps": steps,
        "_predictions": predictions,
    }


def deploy_model(df: Any, prepared: dict[str, Any], cfg_name: str, threshold: float) -> result_type:
    cfg = dict(mc.MODEL_CONFIGS[cfg_name])
    cfg["params"] = dict(cfg["params"])
    rng = np.random.default_rng(mc.SEED)
    order = rng.permutation(len(df))
    holdout_size = int(0.1 * len(df))
    holdout_positions = order[:holdout_size]
    fit_positions = order[holdout_size:]
    holdout_ids = np.arange(len(df), dtype=np.int64)[holdout_positions]
    fit_ids = np.arange(len(df), dtype=np.int64)[fit_positions]
    holdout_df = df.iloc[holdout_ids].copy()
    fit_df = df.iloc[fit_ids].copy()

    if cfg["use_text"]:
        text_frames, vectorizer, reducer = mc.fit_text_transform(fit_df, [fit_df, holdout_df])
        text_fit_frame: Any = text_frames[0]
        text_hold_frame: Any = text_frames[1]
        text_model: dict[str, Any] | None = {"vectorizer": vectorizer, "reducer": reducer}
    else:
        text_fit_frame = None
        text_hold_frame = None
        text_model = None
    x_fit = mc.make_matrix(fit_df, prepared, text_fit_frame)
    x_hold = mc.make_matrix(holdout_df, prepared, text_hold_frame)
    y_fit = np.asarray(fit_df["y"].to_numpy())
    y_hold = np.asarray(holdout_df["y"].to_numpy())

    interim = lgb.LGBMClassifier(**cfg["params"])
    began_interim = time.time()
    interim.fit(
        x_fit, y_fit,
        eval_X=x_hold, eval_y=y_hold,
        eval_metric=["auc", "binary_error"], 
        callbacks=[
            lgb.callback.early_stopping(
                stopping_rounds=mc.EARLY_STOP_ROUNDS,
                first_metric_only=False,
                verbose=False,
            )
        ],
    )
    interim_seconds = time.time() - began_interim
    best_iteration_value: Any = getattr(interim, "best_iteration_", None)
    best_iteration = int(best_iteration_value) if best_iteration_value else int(cfg["params"]["n_estimators"])
    novelty_cols = [column for column in prepared["numeric_cols"] if column in fit_df.columns]
    novelty_forest = mc.fit_novelty_forest(fit_df[y_fit == 0], novelty_cols)
    novelty_scores_fit = mc.novelty_scores(novelty_forest, fit_df, novelty_cols)
    novelty_threshold = mc.calibrate_novelty_threshold(novelty_scores_fit[y_fit == 0])
    hold_novelty = mc.novelty_scores(novelty_forest, holdout_df, novelty_cols)
    pred_hold_novelty = (hold_novelty >= novelty_threshold).astype(np.int8)
    novelty_metrics = mc.metrics_block(y_hold, pred_hold_novelty, hold_novelty, novelty_threshold)
    hold_prob = mc.positive_probabilities(interim, x_hold)
    pred_hold_lgb = np.where(hold_prob >= threshold, 1, 0)
    pred_hold_fused = pred_hold_lgb  # <--- ZMIENIONE (tylko LGBM)
    fused_prob = hold_prob           # <--- ZMIENIONE (tylko LGBM)
    hold_metrics = mc.metrics_block(y_hold, pred_hold_fused, fused_prob, threshold)
    hold_metrics["background_filtered_2nd"] = mc.two_pass_background_share(y_hold, fused_prob, threshold, 0.10)
    hold_metrics["fp_per_million_alerts"] = round(
        int(hold_metrics["fp"]) / max(len(y_hold), 1) * 1_000_000, 1
    )
    novelty_metrics = mc.metrics_block(y_hold, pred_hold_novelty, hold_novelty, novelty_threshold)
    final_params = dict(cfg["params"])
    final_params["n_estimators"] = max(50, best_iteration)
    final_model = lgb.LGBMClassifier(**final_params)
    began_final = time.time()
    final_model.fit(x_fit, y_fit)
    final_seconds = time.time() - began_final
    return {
        "config": cfg_name,
        "best_iteration_full_data": best_iteration,
        "holdout_metrics": hold_metrics,
        "final_model": final_model,
        "text_model": text_model,
        "n_rows_fit_full": int(len(fit_df)),
        "n_rows_holdout": int(len(holdout_df)),
        "interim_fit_seconds": round(interim_seconds, 2),
        "final_fit_seconds": round(final_seconds, 2),
        "threshold_deploy": round(threshold, 6),
        "novelty_threshold": round(novelty_threshold, 6),
        "novelty_cols": novelty_cols,
        "novelty_forest": novelty_forest,
        "holdout_metrics_lgb": mc.metrics_block(y_hold, pred_hold_lgb, hold_prob, threshold),
        "holdout_metrics_novelty": novelty_metrics,
    }


def save_bundle(
    cfg_name: str,
    model: Any,
    text_model: dict[str, Any] | None,
    manifest: dict[str, Any],
    primary: bool,
    novelty: dict[str, Any] | None = None,
) -> None:
    model_dir = mc.MODEL_DIR
    artifact_dir = mc.ARTIFACT_DIR
    model_dir.mkdir(exist_ok=True)
    artifact_dir.mkdir(exist_ok=True)
    model_path = artifact_dir / f"deployed_model_{cfg_name}.joblib"
    payload: dict[str, Any] = {"model": model, "text_model": text_model}
    if novelty is not None:
        payload["novelty_forest"] = novelty["forest"]
        payload["novelty_threshold"] = novelty["threshold"]
        payload["novelty_cols"] = novelty["columns"]
    joblib.dump(payload, model_path)
    manifest_path = artifact_dir / f"model_manifest_{cfg_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plan_path = artifact_dir / "feature_plan.json"
    assets = ROOT / "model_assets"
    bundle = model_dir / f"model_{cfg_name}.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(model_path, "model/deployed_model.joblib")
        archive.write(plan_path, "model/feature_plan.json")
        archive.write(manifest_path, "model/metadata.json")
        for asset in sorted(assets.glob("*")):
            if asset.is_file():
                archive.write(asset, f"model/{asset.name}")
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    published = dict(manifest)
    published["sha256"] = digest
    published["zip_bytes"] = int(bundle.stat().st_size)
    published["zip_path"] = str(bundle.relative_to(ROOT))
    published["primary"] = primary
    manifest_path.write_text(json.dumps(published, indent=2), encoding="utf-8")
    if primary:
        shutil.copy2(bundle, model_dir / "model.zip")
        (artifact_dir / "model_manifest.json").write_text(json.dumps(published, indent=2), encoding="utf-8")
    log(f"saved bundle: {bundle} size={bundle.stat().st_size} sha256={digest[:16]} primary={primary}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocols", default="los,chrono")
    parser.add_argument("--configs", default="lgb_structured,lgb_text")
    parser.add_argument("--feature-plan", default="")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--deploy-primary", default="lgb_structured", help="config written to model/model.zip")
    args = parser.parse_args()

    began = time.time()
    log("loading features")
    df = mc.load_features()
    log(f"rows={len(df)} columns={len(df.columns)} seconds={round(time.time() - began, 1)}")
    if args.feature_plan:
        try:
            plan = json.loads(Path(args.feature_plan).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise SystemExit(f"cannot read feature plan {args.feature_plan}: {error}") from error
        numeric_cols = [column for column in plan["numeric_cols"] if column in df.columns]
        categorical_cols = {key: value for key, value in plan["categorical_cols"].items() if key in df.columns}
        prepared: dict[str, Any] = {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "dropped_features": plan["dropped_features"],
            "text_params": dict(mc.TEXT_PARAMS),
            "text_svd_dim": mc.TEXT_SVD_DIM,
        }
    else:
        started = time.time()
        prepared = mc.prepare_features(df)
        log(f"feature selection seconds={round(time.time() - started, 1)}")
        plan_payload = {
            "numeric_cols": prepared["numeric_cols"],
            "categorical_cols": prepared["categorical_cols"],
            "dropped_features": prepared["dropped_features"],
            "text_params": prepared["text_params"],
            "text_svd_dim": prepared["text_svd_dim"],
        }
        mc.ARTIFACT_DIR.mkdir(exist_ok=True)
        (mc.ARTIFACT_DIR / "feature_plan.json").write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")
    log(
        f"features numeric={len(prepared['numeric_cols'])} categorical={len(prepared['categorical_cols'])} "
        f"dropped={len(prepared['dropped_features'])}"
    )

    results: dict[str, Any] = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": []}
    best_candidate: dict[str, Any] | None = None
    for protocol in [item for item in args.protocols.split(",") if item]:
        for cfg_name in [item for item in args.configs.split(",") if item]:
            log(f"running protocol={protocol} config={cfg_name}")
            if protocol == "los":
                result = run_los(df, prepared, cfg_name)
            elif protocol == "chrono":
                result = run_chrono(df, prepared, cfg_name)
            else:
                raise ValueError(f"unknown protocol: {protocol}")
            predictions = result.pop("_predictions")
            predictions.to_parquet(mc.ARTIFACT_DIR / f"predictions_{protocol}_{cfg_name}.parquet", compression="zstd")
            (mc.ARTIFACT_DIR / f"metrics_{protocol}_{cfg_name}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            threshold_rows: list[dict[str, Any]] = []
            for fold in result["folds"] if "folds" in result else result["steps"]:
                threshold_rows.append(
                    {
                        "fold": fold["fold"],
                        "threshold_model": fold["threshold_model"],
                        "threshold_prior_baseline": fold["threshold_prior_baseline"],
                        "threshold_novelty": fold.get("threshold_novelty"),
                        "best_iteration": fold["best_iteration"],
                    }
                )
            threshold_rows.append(
                {
                    "fold": f"deployment_{protocol}_{cfg_name}",
                    "threshold_model": result["deployment_threshold"],
                    "threshold_prior_baseline": result.get("prior_baseline_threshold"),
                    "threshold_novelty": result.get("novelty_threshold"),
                    "best_iteration": None,
                }
            )
            (mc.ARTIFACT_DIR / f"thresholds_{protocol}_{cfg_name}.json").write_text(
                json.dumps({"thresholds": threshold_rows}, indent=2), encoding="utf-8"
            )
            aggregate = result["aggregate_model"]
            log(
                f"  AGGREGATE {protocol} {cfg_name}: f1={aggregate['f1_attack']:.4f} "
                f"prauc={aggregate.get('pr_auc_attack', 0.0):.4f} arr={aggregate['alert_reduction_rate']:.4f} "
                f"bg_filtered={aggregate['background_filtered_pct']:.2f}% "
                f"bg_2nd={aggregate.get('background_filtered_2nd', 0.0):.4f} "
                f"novelty_f1={result['aggregate_novelty']['f1_attack']:.4f} "
                f"prior_f1={result['aggregate_prior_baseline']['f1_attack']:.4f} "
                f"fp_per_m={aggregate['false_positives_per_million_alerts']:.0f} "
                f"bg_filtered_2nd={aggregate['background_filtered_share']:.4f}"
            )
            results["runs"].append(
                {
                    "protocol": protocol,
                    "config": cfg_name,
                    "deployment_threshold": result["deployment_threshold"],
                    "aggregate_model": aggregate,
                    "aggregate_prior_baseline": result.get("aggregate_prior_baseline"),
                    "aggregate_novelty": result.get("aggregate_novelty"),
                    "novelty_threshold": result.get("novelty_threshold"),
                    "prior_baseline_threshold": result.get("prior_baseline_threshold"),
                    "fit_total_seconds": result["fit_total_seconds"],
                }
            )
            if protocol == "los" and (
                best_candidate is None
                or aggregate["alert_reduction_rate"] > best_candidate["arr"]
            ):
                best_candidate = {
                    "config": cfg_name,
                    "arr": aggregate["alert_reduction_rate"],
                    "threshold": result["deployment_threshold"],
                }
    if best_candidate:
        log(f"best by LOS ARR: {json.dumps(best_candidate)}")
        results["best_by_los_arr"] = best_candidate
    if args.deploy:
        order = [args.deploy_primary] + [item for item in mc.MODEL_CONFIGS if item != args.deploy_primary]
        deployed: list[dict[str, Any]] = []
        for cfg_name in order:
            metrics_path = mc.ARTIFACT_DIR / f"metrics_los_{cfg_name}.json"
            if not metrics_path.exists():
                log(f"deploy skipped: {cfg_name} (no LOS threshold reference)")
                continue
            try:
                reference = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                log(f"deploy skipped: {cfg_name} ({error})")
                continue
            threshold_value = float(reference["deployment_threshold"])
            log(f"deploying model config={cfg_name}")
            began_deploy = time.time()
            deploy_result = deploy_model(df, prepared, cfg_name, threshold_value)
            holdout = deploy_result["holdout_metrics"]
            bundle_thresholds: dict[str, Any] = {"los_deployment_threshold": threshold_value}
            for protocol in ("los", "chrono"):
                path = mc.ARTIFACT_DIR / f"thresholds_{protocol}_{cfg_name}.json"
                if path.exists():
                    try:
                        bundle_thresholds[protocol] = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, ValueError) as error:
                        log(f"cannot read {path}: {error}")
            manifest = {
                "task": "alert triage attack vs background (AIT-ADS)",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "config": cfg_name,
                "uses_text": bool(mc.MODEL_CONFIGS[cfg_name]["use_text"]),
                "trained_on": "all 8 AIT-ADS scenarios",
                "dataset": "AIT Alert Data Set, DOI 10.5281/zenodo.8263181",
                "split": (
                    "random 90/10 row split (seed 42); early stopping evaluated on the same 10% holdout, "
                    "so the holdout metrics below are optimistic; unbiased estimates are the LOS/chrono "
                    "metrics in artifacts/"
                ),
                "n_rows_trained": deploy_result["n_rows_fit_full"],
                "n_rows_holdout": deploy_result["n_rows_holdout"],
                "deployment_threshold": round(threshold_value, 6),
                "thresholds": bundle_thresholds,
                "novelty": {
                    "threshold": deploy_result["novelty_threshold"],
                    "features": len(deploy_result["novelty_cols"]),
                    "target_fp_rate": mc.NOVELTY_FP_RATE,
                    "fusion_rule": "model alarm OR novelty score above threshold",
                },
                "best_iteration_full_data": deploy_result["best_iteration_full_data"],
                "interim_fit_seconds": deploy_result["interim_fit_seconds"],
                "final_fit_seconds": deploy_result["final_fit_seconds"],
                "deploy_seconds": round(time.time() - began_deploy, 1),
                "features": {
                    "numeric": len(prepared["numeric_cols"]),
                    "categorical": len(prepared["categorical_cols"]),
                    "text_svd": mc.TEXT_SVD_DIM if mc.MODEL_CONFIGS[cfg_name]["use_text"] else 0,
                },
                "feature_columns": {
                    "numeric": prepared["numeric_cols"],
                    "categorical": list(prepared["categorical_cols"]),
                },
                "holdout_metrics_same_distribution": holdout,
                "holdout_metrics_lgb": deploy_result["holdout_metrics_lgb"],
                "holdout_metrics_novelty": deploy_result["holdout_metrics_novelty"],
                "budgets": {
                    key: value for key, value in holdout.items() if str(key).startswith("budget_")
                },
                "environment": {
                    "python": platform.python_version(),
                    "lightgbm": lgb.__version__,
                    "scikit_learn": version("scikit-learn"),
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                    "joblib": joblib.__version__,
                },
                "seed": mc.SEED,
            }
            save_bundle(
                cfg_name,
                deploy_result["final_model"].booster_,
                deploy_result["text_model"],
                manifest,
                primary=(cfg_name == args.deploy_primary),
                novelty={
                "forest": deploy_result["novelty_forest"],
                "threshold": deploy_result["novelty_threshold"],
                "columns": deploy_result["novelty_cols"],
            },
            )
            deployed.append(
                {
                    "config": cfg_name,
                    "sha256": manifest.get("sha256", ""),
                    "threshold": threshold_value,
                    "novelty_threshold": deploy_result["novelty_threshold"],
                    "fp_per_million_alerts": holdout.get("fp_per_million_alerts"),
                    "background_filtered_2nd": holdout.get("background_filtered_2nd"),
                }
            )
            results["deployed"] = deployed
    results["total_seconds"] = round(time.time() - began, 1)
    results["feature_plan_summary"] = {
        "numeric": len(prepared["numeric_cols"]),
        "categorical": len(prepared["categorical_cols"]),
        "dropped": len(prepared["dropped_features"]),
    }
    (mc.ARTIFACT_DIR / "train_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"done seconds={results['total_seconds']}")


if __name__ == "__main__":
    main()