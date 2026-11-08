from __future__ import annotations

import csv
import heapq
import json
import math
import re
import sys
import time
from collections import Counter, deque
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
import pandas as pd
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

SCENARIOS = ["fox", "harrison", "russellmitchell", "santos", "shaw", "wardbeck", "wheeler", "wilson"]

UA_PATTERNS = {
    "ua_dirb": "dirb",
    "ua_wpscan": "wpscan",
    "ua_nmap": "nmap",
    "ua_ncat": "ncat",
    "ua_wget": "wget",
    "ua_headless": "HeadlessChrome",
}
URL_FLAGS = {
    "urlflag_wpadmin": "wp-admin",
    "urlflag_wpcontent": "wp-content",
    "urlflag_wpjson": "wp-json",
    "urlflag_wplogin": "wp-login",
    "urlflag_xmlrpc": "xmlrpc",
    "urlflag_wpconfig": "wp-config",
    "urlflag_phpmyadmin": "phpmyadmin",
    "urlflag_backup": "backup",
    "urlflag_dump": "dump",
    "urlflag_config": "config",
    "urlflag_git": ".git",
    "urlflag_sql": "sql",
    "urlflag_readme": "readme",
    "urlflag_license": "license",
    "urlflag_test": "test",
    "urlflag_dev": "dev",
    "urlflag_secret": "secret",
    "urlflag_admin": "admin",
    "urlflag_pass": "pass",
    "urlflag_user": "user",
    "urlflag_include": "include",
    "urlflag_proc": "proc",
    "urlflag_ocs": "ocs/",
    "urlflag_remote": "remote",
    "urlflag_shell": "shell",
    "urlflag_cmd": "cmd",
    "urlflag_php": ".php",
    "urlflag_txt": ".txt",
    "urlflag_zip": ".zip",
}
EXT_RX = re.compile(r"\.([A-Za-z0-9]{1,5})(?:\?|$)")
TOKEN_RX = re.compile(r"[A-Za-z0-9+/=%_.-]{8,}")
REQ_RX = re.compile(r'"([A-Z]+)\s+(\S+)\s+HTTP/[0-9.]"\s+(\d{3})')
AUDIT_TYPE_RX = re.compile(r"type=([A-Z_]+)")
RANGE_LO = "range_lo"

BATCH_ROWS = 100_000
LABEL_ATTACK = "attack"
LABEL_BACKGROUND = "background"


def load_phases(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    intervals: dict[str, list[tuple[float, float, str]]] = {scenario: [] for scenario in SCENARIOS}
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                scenario = row["scenario"].strip()
                if scenario not in intervals:
                    continue
                start = parse_boundary(row["start"])
                end = parse_boundary(row["end"])
                if start is None or end is None:
                    continue
                intervals[scenario].append((start, end, row["attack"].strip()))
    except OSError as exc:
        print(f"warn: missing label file {path}: {exc}", file=sys.stderr)
    for scenario in intervals:
        intervals[scenario].sort(key=lambda item: item[0])
    return intervals


def parse_boundary(value: str) -> float | None:
    text = value.strip()
    epoch = to_num(text)
    if epoch is not None:
        return epoch
    return parse_iso(text)


def phase_of(intervals: list[tuple[float, float, str]], ts: float) -> str:
    for start, end, name in intervals:
        if start > ts:
            break
        if ts <= end:
            return name
    return "background"


def parse_iso(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()
    return None


def to_num(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    number: float | None = None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            number = None
    if number is None or math.isnan(number):
        return None
    return number


def as_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return ""


def dict_of(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def list_of(value: object) -> list:
    return value if isinstance(value, list) else []


def iter_rows(path: Path, scenario: str) -> Iterator[dict[str, object]]:
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                row = wazuh_row(obj, scenario)
                if row is not None:
                    yield row
    except OSError as exc:
        print(f"warn: cannot read {path}: {exc}", file=sys.stderr)


WINDOW_CAP = 50_000


def row_ts(row: dict[str, object]) -> float:
    value = row["ts"]
    return float(value) if isinstance(value, (int, float)) else 0.0


def merged_rows(data_dir: Path, scenario: str) -> Iterator[dict[str, object]]:
    heap: list[tuple[float, int, dict[str, object], Iterator[dict[str, object]]]] = []
    for idx, (path) in enumerate((data_dir / f"{scenario}_wazuh.json",)):
        stream = iter(iter_rows(path, scenario))
        head = next(stream, None)
        if head is not None:
            heapq.heappush(heap, (row_ts(head), idx, head, stream))
    while heap:
        _ts, idx, row, stream = heapq.heappop(heap)
        yield row
        nxt = next(stream, None)
        if nxt is not None:
            heapq.heappush(heap, (row_ts(nxt), idx, nxt, stream))


def norm_path(value: object) -> str:
    text = as_str(value).split("?")[0]
    parts = [part for part in text.split("/") if part]
    if parts and parts[0] in ("var", "opt", "home", "usr"):
        parts = parts[1:]
    return "/".join(parts[:3])


def entropy_of(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    value = 0.0
    for count in counts.values():
        probability = count / total
        value -= probability * math.log2(probability)
    return round(value, 4)


def url_features(url: object, scan_text: str) -> dict[str, float]:
    text = as_str(url)
    lowered = text.lower()
    out: dict[str, float] = {
        "url_len": float(len(text)),
        "url_parts": float(len([part for part in lowered.split("/") if part])),
        "url_query": float("?" in text),
        "url_tokens": float(len(re.findall(r"[A-Za-z0-9]+", lowered))),
        "url_encoded_chars": float(lowered.count("%")),
        "url_entropy": entropy_of(text),
        "url_max_token_len": float(max((len(t) for t in TOKEN_RX.findall(text)), default=0)),
    }
    for flag, needle in URL_FLAGS.items():
        out[flag] = float(needle in lowered)
    low_scan = (scan_text or "").lower()
    for flag, needle in UA_PATTERNS.items():
        out[flag] = float(needle.lower() in low_scan)
    return out


def scan_text_for_url(raw: str) -> tuple[str, str, str]:
    match = REQ_RX.search(raw)
    if match:
        return match.group(2), match.group(1), match.group(3)
    return "", "", ""


def log_resource_class(resources: list) -> str:
    text = " ".join(as_str(r) for r in resources).lower()
    if "access" in text or "apache" in text:
        return "apache"
    if "metricbeat" in text or "filebeat" in text:
        return "metrics"
    if "syslog" in text or "auth" in text:
        return "syslog"
    return "other"


def flatten(values: object) -> list[str]:
    out: list[str] = []
    if isinstance(values, list):
        for value in values:
            out.extend(flatten(value))
    elif isinstance(values, str):
        out.append(values)
    elif values is not None:
        out.append(str(values))
    return out


def wazuh_row(obj: dict, scenario: str) -> dict | None:
    rule = dict_of(obj.get("rule"))
    description = as_str(rule.get("description"))
    groups = sorted({as_str(item) for item in list_of(rule.get("groups"))})
    if "suricata" in groups or description.startswith("Suricata:"):
        return None

    ts = parse_iso(obj.get("@timestamp"))
    if ts is None:
        return None

    data = dict_of(obj.get("data"))
    http_block = dict_of(data.get("http"))
    decoder = dict_of(obj.get("decoder"))

    full_log = as_str(obj.get("full_log"))
    raw_id = data.get("id")
    status_raw = raw_id if raw_id is not None else http_block.get("status")
    src_port = to_num(data.get("src_port") if data.get("src_port") is not None else data.get("srcport"))
    user_agent = as_str(http_block.get("http_user_agent"))
    url = data.get("url") if data.get("url") is not None else http_block.get("url")
    auto_url, auto_method, auto_status = scan_text_for_url(full_log)
    if not url:
        url = auto_url
    status_code = as_str(to_num(status_raw) or status_raw)
    if not status_code:
        status_code = auto_status
    http_method = as_str(data.get("protocol") or http_block.get("http_method")) or auto_method

    row: dict[str, object] = {
        "source": "wazuh",
        "scenario": scenario,
        "ts": ts,
        "rule_id": as_str(to_num(rule.get("id")) or rule.get("id")),
        "rule_level": to_num(rule.get("level")),
        "rule_desc": description,
        "rule_groups": ",".join(groups),
        "group_ids": float("ids" in groups),
        "group_web": float("web" in groups),
        "group_accesslog": float("accesslog" in groups),
        "group_syslog": float("syslog" in groups),
        "group_audit": float("audit" in groups or "attack_log" in groups),
        "group_mitre": float("attack" in groups or "mitre_information" in groups),
        "is_aggregated": float(description.startswith("Multiple ")),
        "decoder": as_str(decoder.get("name")),
        "decoder_parent": as_str(decoder.get("parent")),
        "location": norm_path(obj.get("location")),
        "http_method": http_method,
        "status_code": status_code,
        "src_port": src_port,
        "src_ip_ctx": as_str(data.get("srcip") or data.get("src_ip")),
        "dst_ip_ctx": as_str(data.get("dstip") or data.get("dest_ip")),
        "srcuser": as_str(data.get("srcuser")),
        "dstuser": as_str(data.get("dstuser")),
        "log_resource_class": log_resource_class([obj.get("location")]),
        "log_head": full_log[:400],
        "firedtimes": to_num(rule.get("firedtimes")),
    }
    row.update(url_features(url, user_agent or full_log))
    return row


COLS_STR = [
    "scenario",
    "source",
    "label",
    "phase",
    "sig_key",
    "rule_id",
    "rule_desc",
    "rule_groups",
    "decoder",
    "decoder_parent",
    "location",
    "http_method",
    "status_code",
    "src_ip_ctx",
    "dst_ip_ctx",
    "srcuser",
    "dstuser",
    "log_resource_class",
    "log_head",
]
COLS_NUM = (
    [
        "ts",
        "time_since_last_event",
        "rule_level",
        "src_port",
        "firedtimes",
        "url_len",
        "url_parts",
        "url_query",
        "url_tokens",
        "url_encoded_chars",
        "url_entropy",
        "url_max_token_len",
        "hour",
        "weekday",
        "seconds_from_start",
        "sig_count_prev_60s",
        "sig_count_prev_600s",
        "group_ids",
        "group_web",
        "group_accesslog",
        "group_syslog",
        "group_audit",
        "group_mitre",
        "is_aggregated",
        "alert_count_in_window",  # <--- Dodana kolumna z zagregowanym ruchem
    ]
    + list(URL_FLAGS)
    + list(UA_PATTERNS)
)
NUMERIC_SET = set(COLS_NUM)


def make_schema():
    fields = [pa.field(name, pa.string()) for name in COLS_STR]
    fields += [pa.field(name, pa.float64()) for name in COLS_NUM]
    return pa.schema(fields)


def batch_from_rows(rows: list[dict], schema) -> object:
    columns = {}
    for field in schema:
        if field.name in NUMERIC_SET:
            columns[field.name] = pa.array([to_num(row.get(field.name)) for row in rows], type=pa.float64())
        else:
            columns[field.name] = pa.array([as_str(row.get(field.name)) for row in rows], type=pa.string())
    return pa.RecordBatch.from_pydict(columns, schema=schema)


def scenario_start(data_dir: Path, scenario: str) -> float:
    found: list[float] = []
    for name in (f"{scenario}_wazuh.json", f"{scenario}_aminer.json"):
        try:
            with open(data_dir / name, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    ts = parse_iso(obj.get("@timestamp"))
                    if ts is None:
                        stamps = dict_of(obj.get("LogData")).get("DetectionTimestamp")
                        ts = to_num(stamps[-1]) if isinstance(stamps, list) and stamps else None
                    if ts is not None:
                        found.append(ts)
                    break
        except OSError:
            continue
    return min(found) if found else 0.0


def aggregate_into_windows(df: pd.DataFrame, window_seconds: int = 30) -> pd.DataFrame:
    """Grupuje pojedyncze logi w okna czasowe (meta-alerty), redukując szum i RAM."""
    if df.empty:
        return df
        
    df['time_window'] = df['ts'] // window_seconds
    
    aggs = {
        'ts': 'min',
        'label': lambda x: 'attack' if (x == 'attack').any() else 'background',
        'phase': lambda x: x[x != 'background'].iloc[0] if (x != 'background').any() else 'background'
    }
    
    # Automatyczne budowanie logiki agregacji (max dla liczb, pierwszy element dla stringów)
    for col in df.columns:
        if col in aggs or col in ['time_window']:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            aggs[col] = 'max'
        else:
            aggs[col] = 'first'
            
    window_sizes = df.groupby('time_window').size().rename('alert_count_in_window')
    grouped = df.groupby('time_window').agg(aggs)
    
    if isinstance(grouped.columns, pd.MultiIndex):
        grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns]
        
    grouped = grouped.rename(columns={'label_<lambda>': 'label', 'phase_<lambda>': 'phase'})
    grouped = grouped.join(window_sizes).reset_index(drop=True)
    
    return grouped


def enrich(row: dict, scenario: str, phases: dict, windows: dict, start_ts: float) -> None:
    ts = row["ts"]
    phase = phase_of(phases[scenario], ts)
    row["label"] = LABEL_BACKGROUND if phase == "background" else LABEL_ATTACK
    row["phase"] = phase
    key = f"{row['rule_desc']}"
    row["sig_key"] = key
    entry = windows.get(key)
    if entry is None:
        entry = (deque(maxlen=WINDOW_CAP), deque(maxlen=WINDOW_CAP))
        windows[key] = entry
    recent60, recent600 = entry
    while recent60 and ts - recent60[0] > 60.0:
        recent60.popleft()
    while recent600 and ts - recent600[0] > 600.0:
        recent600.popleft()
    row["sig_count_prev_60s"] = float(len(recent60))
    row["sig_count_prev_600s"] = float(len(recent600))
    recent60.append(ts)
    recent600.append(ts)
    stamp = datetime.fromtimestamp(ts, tz=UTC)
    row["hour"] = float(stamp.hour)
    row["weekday"] = float(stamp.weekday())
    row["seconds_from_start"] = ts - start_ts


def build_scenario(data_dir: Path, out_dir: Path, phases: dict, scenario: str) -> int:
    schema = make_schema()
    windows: dict[str, tuple[deque, deque]] = {}
    buffer: list[dict] = []
    start_ts = 0.0
    last_ts = 0.0
    
    # Zbieramy wszystkie wiersze dla danego scenariusza
    for row in merged_rows(data_dir, scenario):
        if not buffer and start_ts == 0.0:
            start_ts = row_ts(row)
        enrich(row, scenario, phases, windows, start_ts)
        current_ts = row_ts(row)
        row["time_since_last_event"] = 0.0 if last_ts == 0.0 else max(0.0, current_ts - float(last_ts))
        last_ts = current_ts
        buffer.append(row)
        
    if not buffer:
        return 0
        
    # Zamieniamy wszystkie pojedyncze alerty w Meta-Alerty (okna 120s)
    df = pd.DataFrame(buffer)
    df = aggregate_into_windows(df, window_seconds=30)
    
    # Zapisujemy odchudzone dane
    records = df.to_dict('records')
    writer = pq.ParquetWriter(out_dir / f"{scenario}.features.parquet", schema, compression="zstd")
    try:
        writer.write_batch(batch_from_rows(records, schema))
    finally:
        writer.close()
        
    return len(records)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "ait_ads"
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    phases = load_phases(data_dir / "labels.csv")
    total = 0
    for scenario in SCENARIOS:
        began = time.time()
        count = build_scenario(data_dir, out_dir, phases, scenario)
        total += count
        print(
            f"{scenario}: zgrupowane alerty={count} cols={len(COLS_STR) + len(COLS_NUM)} seconds={time.time() - began:.1f}",
            flush=True,
        )
    print(f"TOTAL okien (meta-alertów)={total} cols={len(COLS_STR) + len(COLS_NUM)}", flush=True)


if __name__ == "__main__":
    main()