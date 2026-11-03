"""Aggregate report tables directly from the feature Parquet tables.

The feature tables are the single source of truth for the ML section of the report:
row counts, label/phase distributions, detector statistics and the count of detectors
that appear in only one capture scenario (the genuinely novel detectors).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "features"
ARTIFACT_DIR = ROOT / "artifacts"
SCENARIOS = ["fox", "harrison", "russellmitchell", "santos", "shaw", "wardbeck", "wheeler", "wilson"]
COLUMNS = ["label", "phase", "source", "sig_key", "rule_desc", "rule_level", "is_training_mode"]


def log(message: str) -> None:
    print(message, flush=True)


def load_table(scenario: str) -> Any:
    try:
        frame: Any = pd.read_parquet(FEATURE_DIR / f"{scenario}.features.parquet", columns=COLUMNS)
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot read features for {scenario}: {error}") from error
    return frame


def main() -> None:
    detector_total: dict[str, int] = defaultdict(int)
    detector_attack: dict[str, int] = defaultdict(int)
    detector_meta: dict[str, dict[str, Any]] = {}
    detector_scenarios: dict[str, set[str]] = defaultdict(set)
    detector_sources: dict[str, set[str]] = defaultdict(set)
    training_mode_total = 0
    training_mode_attack = 0
    scenarios: dict[str, dict[str, Any]] = {}
    global_phases: dict[str, int] = defaultdict(int)
    global_rows = 0
    global_attack = 0

    for scenario in SCENARIOS:
        frame = load_table(scenario)
        labels = [str(item) for item in frame["label"].to_numpy()]
        phases = [str(item) for item in frame["phase"].to_numpy()]
        sources = [str(item) for item in frame["source"].to_numpy()]
        sig_keys = [str(item) for item in frame["sig_key"].to_numpy()]
        rule_descs = [str(item) for item in frame["rule_desc"].to_numpy()]
        rule_levels = [int(item) if pd.notna(item) else -1 for item in frame["rule_level"].to_numpy()]
        training_flags = [int(item) if pd.notna(item) else 0 for item in frame["is_training_mode"].to_numpy()]
        y_flags = [1 if label == "attack" else 0 for label in labels]

        label_counts: dict[str, int] = defaultdict(int)
        phase_counts: dict[str, int] = defaultdict(int)
        source_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        detector_counts: dict[str, int] = defaultdict(int)
        detector_hits: dict[str, int] = defaultdict(int)
        scenario_training = 0
        for index, label in enumerate(labels):
            label_counts[label] += 1
            phase_key = f"{label}|{phases[index]}"
            phase_counts[phase_key] += 1
            global_phases[phase_key] += 1
            source_counts[sources[index]][label] += 1
            key = sig_keys[index]
            attack_flag = y_flags[index]
            detector_counts[key] += 1
            detector_hits[key] += attack_flag
            detector_total[key] += 1
            detector_attack[key] += attack_flag
            detector_scenarios[key].add(scenario)
            detector_sources[key].add(sources[index])
            if key not in detector_meta:
                detector_meta[key] = {
                    "detector": rule_descs[index],
                    "rule_level": rule_levels[index],
                    "signature_id": key.split("|", maxsplit=1)[1],
                }
            if training_flags[index] == 1:
                training_mode_total += 1
                training_mode_attack += attack_flag
                scenario_training += 1

        novel = [key for key in detector_counts if len(detector_scenarios[key]) == 1]
        novel_rows = sum(detector_counts[key] for key in novel)
        novel_attack = sum(detector_hits[key] for key in novel)
        scenarios[scenario] = {
            "rows": len(labels),
            "attack": int(label_counts.get("attack", 0)),
            "background": int(label_counts.get("background", 0)),
            "attack_share": round(label_counts.get("attack", 0) / max(len(labels), 1), 4),
            "sources": {key: dict(value) for key, value in source_counts.items()},
            "phases": dict(phase_counts),
            "detectors": len(detector_counts),
            "novel_detectors": len(novel),
            "novel_rows": novel_rows,
            "novel_attack_rows": novel_attack,
            "training_mode_rows": scenario_training,
        }
        global_rows += len(labels)
        global_attack += sum(y_flags)
        log(
            f"{scenario}: rows={len(labels)} attack={label_counts.get('attack', 0)} "
            f"detectors={len(detector_counts)} novel={len(novel)} novel_rows={novel_rows}"
        )

    detectors: list[dict[str, Any]] = []
    for key, total_value in detector_total.items():
        total = int(total_value)
        attack = int(detector_attack[key])
        meta = detector_meta[key]
        detectors.append(
            {
                "sig_key": key,
                "signature_id": meta["signature_id"],
                "detector": meta["detector"],
                "rule_level": meta["rule_level"],
                "sources": sorted(detector_sources[key]),
                "scenarios": sorted(detector_scenarios[key]),
                "n_scenarios": len(detector_scenarios[key]),
                "total": total,
                "attack": attack,
                "background": total - attack,
                "precision": round(attack / max(total, 1), 4),
            }
        )
    detectors.sort(key=lambda item: item["total"], reverse=True)
    novel_total = sum(1 for item in detectors if item["n_scenarios"] == 1)
    payload = {
        "total_rows": global_rows,
        "total_attack": global_attack,
        "total_background": global_rows - global_attack,
        "scenarios": scenarios,
        "phase_totals": dict(global_phases),
        "training_mode_rows": training_mode_total,
        "training_mode_attack_rows": training_mode_attack,
        "detectors": detectors,
        "detector_count": len(detectors),
        "novel_detectors_total": novel_total,
        "novel_rows_total": sum(item["total"] for item in detectors if item["n_scenarios"] == 1),
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    (ARTIFACT_DIR / "table_stats.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"total rows={global_rows} attack={global_attack} detectors={len(detectors)}")
    log(f"novel detectors (exactly one scenario): {novel_total} rows={payload['novel_rows_total']}")
    log(f"training mode rows: {training_mode_total} (attack {training_mode_attack})")


if __name__ == "__main__":
    main()
