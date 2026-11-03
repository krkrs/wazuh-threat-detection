"""Statystyki benchmarku AIT-ADS: liczby alertow, udzial klas, statystyki detectorow.

Uzycie: python3 research/compute_stats.py [katalog_danych]  (domylnie data/ait_ads)
Wynik: JSON na stdout, np. python3 research/compute_stats.py data/ait_ads > research/ads_stats.json
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCENARIOS = ["fox", "harrison", "russellmitchell", "santos", "shaw", "wardbeck", "wheeler", "wilson"]

PhaseList = list[tuple[str, float, float]]

TIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%m/%d/%Y-%H:%M:%S.%f",
    "%b %d %H:%M:%S",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Rows of a CSV file; empty list when the file is missing or unreadable."""
    rows: list[dict[str, str]] = []
    with suppress(OSError, UnicodeDecodeError, csv.Error), path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_phases(path: Path) -> dict[str, PhaseList]:
    """Attack-phase intervals per scenario from labels.csv."""
    phases: dict[str, PhaseList] = defaultdict(list)
    for row in read_rows(path):
        try:
            scenario = row["scenario"]
            attack = row["attack"]
            start = to_float(row["start"])
            end = to_float(row["end"])
        except KeyError:
            continue
        if start is None or end is None:
            continue
        phases[scenario].append((attack, start, end))
    return phases


def parse_time(raw: Any) -> float | None:
    """Epoch seconds from the timestamp formats used by Wazuh/Suricata alert fields."""
    if not isinstance(raw, str):
        return None
    for fmt in TIME_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        if dt.year < 1970:
            dt = dt.replace(year=2022)
        return dt.replace(tzinfo=UTC if dt.tzinfo is None else dt.tzinfo).timestamp()
    return None


def alert_time_wazuh(obj: dict[str, Any]) -> float | None:
    """Event time of a Wazuh/Suricata alert (the timestamp field depends on the rule family)."""
    ts = parse_time(obj.get("@timestamp"))
    if ts is None:
        for key in ("predecoder", "data"):
            value = obj.get(key)
            if isinstance(value, dict) and "timestamp" in value:
                ts = parse_time(value["timestamp"])
                if ts is not None:
                    break
    return ts


def phase_of(phases: dict[str, PhaseList], scenario: str, ts: float) -> str:
    """Label name when the timestamp falls inside an attack interval, else 'background'."""
    for name, start, end in phases.get(scenario, []):
        if start <= ts < end:
            return name
    return "background"


def iter_json_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Parsed JSON objects of a JSONL file; nothing is yielded when the file is unreadable."""
    with suppress(OSError), path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def main() -> None:
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/ait_ads")
    phases = load_phases(data_dir / "labels.csv")
    detector_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"attack": 0, "background": 0, "total": 0})
    totals: Counter = Counter()
    out: dict[str, Any] = {"scenarios": {}, "totals": dict(totals), "top_detectors": []}

    for scenario in SCENARIOS:
        s: dict[str, int] = {"wazuh": 0, "suricata": 0, "aminer": 0, "attack_alerts": 0, "background_alerts": 0}
        phase_counts: Counter = Counter()
        det: Counter = Counter()
        for obj in iter_json_lines(data_dir / f"{scenario}_wazuh.json"):
            try:
                desc = obj["rule"]["description"]
            except (KeyError, TypeError):
                continue
            src = "suricata" if desc.startswith("Suricata:") else "wazuh"
            ts = alert_time_wazuh(obj)
            if ts is None:
                continue
            s[src] += 1
            label = phase_of(phases, scenario, ts)
            binary = "background" if label == "background" else "attack"
            s["attack_alerts" if binary == "attack" else "background_alerts"] += 1
            phase_counts[label] += 1
            det[(src, desc, binary)] += 1
        for obj in iter_json_lines(data_dir / f"{scenario}_aminer.json"):
            try:
                desc = obj["AnalysisComponent"]["AnalysisComponentName"]
            except (KeyError, TypeError):
                continue
            stamps = obj.get("LogData", {}).get("DetectionTimestamp") or [None]
            ts = stamps[-1]
            s["aminer"] += 1
            label = phase_of(phases, scenario, ts) if isinstance(ts, (int, float)) else "background"
            binary = "background" if label == "background" else "attack"
            s["attack_alerts" if binary == "attack" else "background_alerts"] += 1
            phase_counts[label] += 1
            det[("aminer", desc, binary)] += 1
        for (src, desc, binary), count in det.items():
            entry = detector_stats[f"{src}|{desc}"]
            entry[binary] += count
            entry["total"] += count
        out["scenarios"][scenario] = {**s, "phase_counts": dict(phase_counts)}
        totals.update(s)

    out["totals"] = dict(totals)
    ranked = sorted(detector_stats.items(), key=lambda kv: -kv[1]["total"])
    out["top_detectors"] = [{"detector": name, **vals} for name, vals in ranked[:60]]
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
