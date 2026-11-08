#!/usr/bin/env python3
"""
Diagnostyka wszystkich scenariuszy Wazuh i labels.csv.

Przykład:
  python diagnose_all_scenarios.py --data-dir ./data --labels ./data/labels.csv

Skrypt:
- szuka rekurencyjnie plików *_wazuh.json (JSONL: jeden alert JSON na linię),
- dopasowuje plik do labels.csv na podstawie nazwy scenariusza,
- przypisuje alerty do faz na podstawie @timestamp i przedziałów start/end,
- raportuje liczniki, nakładanie przedziałów, błędne timestampy,
  strukturę alertów oraz najczęstsze rule.id / dekodery / grupy,
- nie zapisuje surowych alertów, adresów IP ani pełnych logów.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_time(value: Any) -> float | None:
    """Parse Unix seconds or ISO-8601 timestamp into Unix seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        # Numeric timestamps may be stored as strings.
        return float(text)
    except ValueError:
        pass
    try:
        normalized = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, OverflowError):
        return None


def norm_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def safe_counter_values(counter: Counter, limit: int = 20) -> list[dict[str, Any]]:
    return [{"value": str(k), "count": v} for k, v in counter.most_common(limit)]


def load_labels(path: Path):
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_scenarios = set()
    required = {"scenario", "attack", "start", "end"}

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(
                f"labels.csv nie ma wymaganych kolumn: {', '.join(sorted(missing))}. "
                f"Znalezione kolumny: {', '.join(reader.fieldnames or [])}"
            )

        for row_num, row in enumerate(reader, start=2):
            scenario = (row.get("scenario") or "").strip()
            attack = (row.get("attack") or "").strip()
            start = parse_time(row.get("start"))
            end = parse_time(row.get("end"))
            if not scenario:
                continue
            raw_scenarios.add(scenario)
            by_scenario[norm_name(scenario)].append({
                "scenario": scenario,
                "attack": attack or "UNKNOWN",
                "start": start,
                "end": end,
                "row": row_num,
            })
    return by_scenario, raw_scenarios


def find_scenario_for_file(file_path: Path, labels_by_scenario):
    stem = file_path.name
    stem = re.sub(r"_wazuh\.json$", "", stem, flags=re.IGNORECASE)
    key = norm_name(stem)
    if key in labels_by_scenario:
        return key, stem

    # Fallback: match scenario name as a normalized substring of the filename.
    candidates = [
        k for k in labels_by_scenario
        if k and (k in norm_name(file_path.name) or norm_name(file_path.name).startswith(k))
    ]
    if len(candidates) == 1:
        return candidates[0], stem
    return None, stem


def interval_stats(intervals):
    invalid = sum(1 for x in intervals if x["start"] is None or x["end"] is None)
    valid = [x for x in intervals if x["start"] is not None and x["end"] is not None]
    reversed_intervals = sum(1 for x in valid if x["end"] < x["start"])
    ordered = sorted(
        [x for x in valid if x["end"] >= x["start"]],
        key=lambda x: (x["start"], x["end"])
    )
    overlaps = []
    max_end = None
    max_item = None
    for item in ordered:
        if max_end is not None and item["start"] < max_end:
            overlaps.append({
                "first_attack": max_item["attack"] if max_item else "?",
                "second_attack": item["attack"],
            })
        if max_end is None or item["end"] > max_end:
            max_end = item["end"]
            max_item = item
    return {
        "interval_count": len(intervals),
        "invalid_intervals": invalid,
        "reversed_intervals": reversed_intervals,
        "overlap_count": len(overlaps),
        "overlap_examples": overlaps[:10],
    }


def diagnose_file(file_path: Path, scenario_key: str, intervals: list[dict[str, Any]]):
    counts = Counter()
    top_keys = Counter()
    rule_ids = Counter()
    decoders = Counter()
    groups = Counter()
    descriptions = Counter()
    timestamp_invalid = 0
    json_invalid = 0
    total = 0
    missing_rule = 0
    missing_decoder = 0

    valid_intervals = [
        x for x in intervals
        if x["start"] is not None and x["end"] is not None and x["end"] >= x["start"]
    ]

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                alert = json.loads(line)
                if not isinstance(alert, dict):
                    json_invalid += 1
                    continue
            except json.JSONDecodeError:
                json_invalid += 1
                continue

            total += 1
            top_keys.update(str(k) for k in alert.keys())

            ts = parse_time(alert.get("@timestamp"))
            if ts is None:
                timestamp_invalid += 1
                counts["invalid_timestamp"] += 1
                continue

            matching = [
                x for x in valid_intervals
                if x["start"] <= ts <= x["end"]
            ]
            if len(matching) == 1:
                counts[matching[0]["attack"]] += 1
            elif len(matching) > 1:
                counts["overlapping_label_intervals"] += 1
            else:
                counts["background"] += 1

            rule = alert.get("rule") if isinstance(alert.get("rule"), dict) else {}
            decoder = alert.get("decoder") if isinstance(alert.get("decoder"), dict) else {}

            rid = rule.get("id")
            if rid is None:
                missing_rule += 1
            else:
                rule_ids[str(rid)] += 1

            desc = rule.get("description")
            if desc:
                # Truncate descriptions to keep the report compact.
                descriptions[str(desc)[:120]] += 1

            dec_name = decoder.get("name")
            if dec_name is None:
                missing_decoder += 1
            else:
                decoders[str(dec_name)] += 1

            rule_groups = rule.get("groups", [])
            if isinstance(rule_groups, list):
                groups.update(str(x) for x in rule_groups if x is not None)
            elif rule_groups:
                groups.update([str(rule_groups)])

    return {
        "file": str(file_path),
        "scenario_key": scenario_key,
        "total_alerts": total,
        "invalid_json_lines": json_invalid,
        "invalid_timestamps": timestamp_invalid,
        "alerts_by_time_label": dict(counts),
        "top_level_keys": safe_counter_values(top_keys, 100),
        "top_rule_ids": safe_counter_values(rule_ids, 30),
        "top_decoders": safe_counter_values(decoders, 20),
        "top_rule_groups": safe_counter_values(groups, 30),
        "top_rule_descriptions": safe_counter_values(descriptions, 20),
        "alerts_without_rule_id": missing_rule,
        "alerts_without_decoder_name": missing_decoder,
    }


def main():
    parser = argparse.ArgumentParser(description="Raport diagnostyczny Wazuh + labels.csv")
    parser.add_argument("--data-dir", required=True, help="Katalog z plikami *_wazuh.json")
    parser.add_argument("--labels", required=True, help="Ścieżka do labels.csv")
    parser.add_argument("--out", default="diagnostic_report.json", help="Plik raportu JSON")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    labels_path = Path(args.labels)
    if not data_dir.exists():
        raise SystemExit(f"Nie znaleziono katalogu: {data_dir}")
    if not labels_path.exists():
        raise SystemExit(f"Nie znaleziono labels.csv: {labels_path}")

    labels_by_scenario, raw_scenarios = load_labels(labels_path)
    files = sorted(
        p for p in data_dir.rglob("*")
        if p.is_file() and p.name.lower().endswith("_wazuh.json")
    )

    matched_keys = set()
    reports = []
    unmatched_files = []

    for path in files:
        scenario_key, inferred_name = find_scenario_for_file(path, labels_by_scenario)
        if scenario_key is None:
            unmatched_files.append(str(path))
            continue
        matched_keys.add(scenario_key)
        reports.append(diagnose_file(path, scenario_key, labels_by_scenario[scenario_key]))

    missing_files = sorted(
        labels_by_scenario[k][0]["scenario"]
        for k in labels_by_scenario
        if k not in matched_keys
    )

    interval_report = {}
    for key, intervals in labels_by_scenario.items():
        scenario_name = intervals[0]["scenario"] if intervals else key
        interval_report[scenario_name] = interval_stats(intervals)

    report = {
        "labels_file": str(labels_path),
        "data_directory": str(data_dir),
        "files_found": len(files),
        "files_matched_to_labels": len(reports),
        "label_scenarios": sorted(raw_scenarios),
        "missing_wazuh_files_for_label_scenarios": missing_files,
        "unmatched_wazuh_files": unmatched_files,
        "label_interval_diagnostics": interval_report,
        "scenario_reports": reports,
        "notes": [
            "Etykiety są przypisane wyłącznie na podstawie czasu (@timestamp) i labels.csv.",
            "To są etykiety czasowe (time_label/weak labels), a nie potwierdzone event_label.",
            "Alert poza przedziałami ataku otrzymuje etykietę background.",
            "Alert w nakładających się przedziałach nie jest arbitralnie przypisywany do jednej fazy.",
            "Raport nie zawiera adresów IP ani pełnych logów."
        ]
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n========== DATASET DIAGNOSTICS ==========")
    print(f"Pliki Wazuh znalezione:          {len(files)}")
    print(f"Pliki dopasowane do labels.csv:  {len(reports)}")
    print(f"Scenariusze w labels.csv:        {len(raw_scenarios)}")
    print(f"Brakujące pliki Wazuh:           {len(missing_files)}")
    print(f"Pliki bez dopasowania:           {len(unmatched_files)}")

    if missing_files:
        print("\nScenariusze z labels.csv bez pliku Wazuh:")
        for x in missing_files:
            print(f"  - {x}")
    if unmatched_files:
        print("\nPliki Wazuh bez dopasowania do labels.csv:")
        for x in unmatched_files:
            print(f"  - {x}")

    for item in reports:
        print(f"\n--- {Path(item['file']).name} ---")
        print(f"Alerty łącznie: {item['total_alerts']}")
        print(f"Niepoprawne JSON: {item['invalid_json_lines']}")
        print(f"Niepoprawne @timestamp: {item['invalid_timestamps']}")
        print("Alerty według etykiety czasowej:")
        for label, count in sorted(item["alerts_by_time_label"].items()):
            print(f"  {label:28s} {count:>10,}")
        print("Najczęstsze rule.id:")
        for x in item["top_rule_ids"][:10]:
            print(f"  {x['value']:20s} {x['count']:>10,}")
        print("Najczęstsze dekodery:")
        for x in item["top_decoders"][:10]:
            print(f"  {x['value']:20s} {x['count']:>10,}")

    print(f"\nRaport JSON zapisano w: {out_path.resolve()}")
    print("Uwaga: raport opisuje etykietowanie czasowe, nie potwierdza semantycznie każdego alertu.")


if __name__ == "__main__":
    main()
