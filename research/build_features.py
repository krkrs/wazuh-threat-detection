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

import pyarrow as pa
import pyarrow.parquet as pq

SCENARIOS = ["fox", "harrison", "russellmitchell", "santos", "shaw", "wardbeck", "wheeler", "wilson"]

UA_PATTERNS = {
    "ua_dirb": "dirb",
    "ua_wpscan": "wpscan",
    "ua_nmap": "nmap",
    "ua_nikto": "nikto",
    "ua_sqlmap": "sqlmap",
    "ua_ncat": "ncat",
    "ua_curl": "curl/",
    "ua_python": "python-requests",
    "ua_wget": "wget",
    "ua_golang": "Go-http-client",
    "ua_apt": "APT-HTTP",
    "ua_headless": "HeadlessChrome",
    "ua_java": "Java/",
    "ua_openvas": "OpenVAS",
    "ua_masscan": "masscan",
    "ua_zmap": "zmap",
    "ua_scrapy": "Scrapy",
    "ua_apachebench": "ApacheBench",
    "ua_persistent": "persistent",
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
    "urlflag_env": ".env",
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
    "urlflag_json": ".json",
}
EXT_RX = re.compile(r"\.([A-Za-z0-9]{1,5})(?:\?|$)")
TOKEN_RX = re.compile(r"[A-Za-z0-9+/=%_.-]{8,}")
REQ_RX = re.compile(r'"([A-Z]+)\s+(\S+)\s+HTTP/[0-9.]"\s+(\d{3})')
DNS_QUERY_RX = re.compile(r"query\[[A-Za-z]+\]\s+(\S+)")
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


def iter_rows(path: Path, kind: str, scenario: str) -> Iterator[dict[str, object]]:
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
                row = wazuh_row(obj, scenario) if kind == "wazuh" else aminer_row(obj, scenario)
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
    for idx, (path, kind) in enumerate(
        (
            (data_dir / f"{scenario}_wazuh.json", "wazuh")
#            (data_dir / f"{scenario}_aminer.json", "aminer"), # commented by user
        )
    ):
        stream = iter(iter_rows(path, kind, scenario))
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
        "url_semicolon": float(lowered.count(";")),
        "url_traversal": float("../" in lowered or "%2e%2e" in lowered),
        "url_entropy": entropy_of(text),
        "url_max_token_len": float(max((len(t) for t in TOKEN_RX.findall(text)), default=0)),
    }
    for flag, needle in URL_FLAGS.items():
        out[flag] = float(needle in lowered)
    low_scan = (scan_text or "").lower()
    for flag, needle in UA_PATTERNS.items():
        out[flag] = float(needle in low_scan)
    return out


def dns_features(text: str) -> dict[str, float]:
    low = text.lower()
    labels = [label for label in low.split(".") if label]
    total_len = sum(len(label) for label in labels)
    out = {
        "dns_len": float(len(low)),
        "dns_labels": float(len(labels)),
        "dns_max_label_len": float(max((len(label) for label in labels), default=0)),
        "dns_avg_label_len": float(round(total_len / len(labels), 2)) if labels else 0.0,
        "dns_digits": float(sum(ch.isdigit() for ch in low)),
        "dns_letters": float(sum(ch.isalpha() for ch in low)),
        "dns_entropy": entropy_of(low),
    }
    out["dns_digit_ratio"] = round(out["dns_digits"] / out["dns_len"], 3) if out["dns_len"] else 0.0
    return out


def scan_text_for_url(raw: str) -> tuple[str, str, str]:
    match = REQ_RX.search(raw)
    if match:
        return match.group(2), match.group(1), match.group(3)
    return "", "", ""


def log_resource_class(resources: list) -> str:
    text = " ".join(as_str(r) for r in resources).lower()
    if "audit" in text:
        return "audit"
    if "dns" in text:
        return "dns"
    if "access" in text or "apache" in text:
        return "apache"
    if "metricbeat" in text or "filebeat" in text:
        return "metrics"
    if "syslog" in text or "auth" in text:
        return "syslog"
    if "mysql" in text:
        return "mysql"
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
    ts = parse_iso(obj.get("@timestamp"))
    if ts is None:
        return None
    rule = dict_of(obj.get("rule"))
    data = dict_of(obj.get("data"))
    alert_block = dict_of(data.get("alert"))
    http_block = dict_of(data.get("http"))
    decoder = dict_of(obj.get("decoder"))
    agent = dict_of(obj.get("agent"))
    predecoder = dict_of(obj.get("predecoder"))
    description = as_str(rule.get("description"))
    groups = sorted({as_str(item) for item in list_of(rule.get("groups"))})
    is_suricata = "suricata" in groups or description.startswith("Suricata:")
    full_log = as_str(obj.get("full_log"))
    raw_id = data.get("id")
    status_raw = raw_id if raw_id is not None else http_block.get("status")
    port_raw = data.get("dstport") if data.get("dstport") is not None else data.get("dest_port")
    port_value = to_num(port_raw)
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
    dns_block = dict_of(data.get("dns"))
    dns_query = as_str(dns_block.get("query") or dns_block.get("question") or data.get("dns_query"))
    audit_type = as_str(data.get("audit_type") or data.get("action"))
    audit_match = AUDIT_TYPE_RX.search(full_log)
    if audit_match:
        audit_type = audit_type or audit_match.group(1)

    row: dict[str, object] = {
        "scenario": scenario,
        "source": "suricata" if is_suricata else "wazuh",
        "ts": ts,
        "rule_id": as_str(to_num(rule.get("id")) or rule.get("id")),
        "rule_level": to_num(rule.get("level")),
        "rule_desc": description.removeprefix("Suricata: Alert - "),
        "rule_groups": ",".join(groups),
        "group_ids": float("ids" in groups),
        "group_suricata": float(is_suricata),
        "group_web": float("web" in groups),
        "group_accesslog": float("accesslog" in groups),
        "group_syslog": float("syslog" in groups),
        "group_audit": float("audit" in groups or "attack_log" in groups),
        "group_mitre": float("attack" in groups or "mitre_information" in groups),
        "is_aggregated": float(description.startswith("Multiple ")),
        "decoder": as_str(decoder.get("name")),
        "decoder_parent": as_str(decoder.get("parent")),
        "location": norm_path(obj.get("location")),
        "agent_id": as_str(agent.get("id")),
        "agent_name": as_str(agent.get("name")),
        "agent_ip": as_str(agent.get("ip")),
        "program_name": as_str(predecoder.get("program_name")),
        "hostname_ctx": as_str(predecoder.get("hostname")),
        "sig_category": as_str(alert_block.get("category")),
        "sig_id": as_str(to_num(alert_block.get("signature_id")) or alert_block.get("signature_id")),
        "sig_severity": to_num(alert_block.get("severity")),
        "sig_action": as_str(alert_block.get("action")),
        "sid_rev": as_str(alert_block.get("rev")),
        "mitre_tactic": as_str(rule.get("mitre_tactic")),
        "mitre_technique": as_str(rule.get("mitre_technique_id")),
        "http_method": http_method,
        "status_code": status_code,
        "dst_port": as_str(port_value if port_value is not None else ""),
        "src_port": src_port,
        "src_ip_ctx": as_str(data.get("srcip") or data.get("src_ip")),
        "dst_ip_ctx": as_str(data.get("dstip") or data.get("dest_ip")),
        "srcuser": as_str(data.get("srcuser")),
        "dstuser": as_str(data.get("dstuser")),
        "db_name": as_str(data.get("database_name") or data.get("schema")),
        "audit_type": audit_type,
        "syscall": as_str(data.get("syscall")),
        "rule_md5": as_str(data.get("rule_md5")),
        "sha256_present": float(bool(data.get("sha256") or data.get("md5"))),
        "user_agent": user_agent,
        "log_resource_class": log_resource_class([obj.get("location")]),
        "log_head": full_log[:400],
        "firedtimes": to_num(rule.get("firedtimes")),
        "is_training_mode": 0.0,
        "ac_type": "",
        "ac_id": "",
        "ac_message": "",
        "persistence": "",
        "log_lines_count": None,
        "attr_value_count": 0.0,
        "attr_value_max_len": 0.0,
        "attr_value_max_entropy": 0.0,
        "attr_path_depth": 0.0,
        "missing_char_count": 0.0,
        "missing_char_special": 0.0,
        "anomaly_score_max": None,
        "anomaly_score_mean": None,
        "critical_value": None,
        "probability_threshold": None,
        "count_confidence": None,
        "count_confidence_factor": None,
        "freq_actual": None,
        "freq_expected": None,
        "freq_window": None,
        "freq_confidence_factor": None,
        "miner_id_ctx": "",
    }
    row["range_lo"] = None
    row["range_hi"] = None
    row["range_width"] = None
    row["freq_range_lo"] = None
    row["freq_range_hi"] = None
    row.update(url_features(url, user_agent or full_log))
    row.update(dns_features(dns_query or auto_dns(full_log)))
    row["id_attr_numeric"] = 0.0
    return row


def auto_dns(full_log: str) -> str:
    match = DNS_QUERY_RX.search(full_log)
    return match.group(1) if match else ""


def aminer_row(obj: dict, scenario: str) -> dict | None:
    log_data = dict_of(obj.get("LogData"))
    component = dict_of(obj.get("AnalysisComponent"))
    stamps = log_data.get("DetectionTimestamp")
    ts = to_num(stamps[-1]) if isinstance(stamps, list) and stamps else None
    if ts is None:
        return None
    name = as_str(component.get("AnalysisComponentName"))
    paths = [as_str(p) for p in list_of(component.get("AffectedLogAtomPaths"))]
    values = flatten(component.get("AffectedLogAtomValues"))
    resources = [as_str(r) for r in list_of(log_data.get("LogResources"))]
    raw_lines = [as_str(r) for r in list_of(log_data.get("RawLogData"))]
    raw_head = " | ".join(raw_lines)
    frequency = dict_of(obj.get("FrequencyData"))
    count_data = dict_of(obj.get("CountData"))
    ranges = list_of(component.get("Range"))
    anomaly_scores = [to_num(v) for v in list_of(component.get("AnomalyScores"))]
    missing_chars = flatten(component.get("MissingCharacters"))
    path_segments = [p for p in paths if p]
    value_lengths = [len(v) for v in values]
    dns_candidate = ""
    for value in values:
        if "." in value and " " not in value:
            dns_candidate = value
            break
    if not dns_candidate:
        dns_candidate = auto_dns(raw_head)
    url_candidate, method, status = scan_text_for_url(raw_head)
    audit_match = AUDIT_TYPE_RX.search(raw_head)
    freq_range = list_of(frequency.get("ExpectedLogAtomValuesFrequencyRange"))
    freq_lo = to_num(freq_range[0]) if len(freq_range) > 0 else None
    freq_hi = to_num(freq_range[1]) if len(freq_range) > 1 else None

    row: dict[str, object] = {
        "scenario": scenario,
        "source": "aminer",
        "ts": ts,
        "rule_id": name.replace("AMiner: ", "A-"),
        "rule_level": None,
        "rule_desc": name.replace("AMiner: ", ""),
        "rule_groups": "aminer," + as_str(component.get("AnalysisComponentType")),
        "group_ids": 0.0,
        "group_suricata": 0.0,
        "group_web": 0.0,
        "group_accesslog": 0.0,
        "group_syslog": 0.0,
        "group_audit": 0.0,
        "group_mitre": 0.0,
        "is_aggregated": 0.0,
        "decoder": "aminer",
        "decoder_parent": "",
        "location": norm_path(resources[0]) if resources else "",
        "agent_id": "",
        "agent_name": "",
        "agent_ip": "",
        "program_name": "",
        "hostname_ctx": "",
        "sig_category": "",
        "sig_id": as_str(component.get("AnalysisComponentIdentifier")),
        "sig_severity": None,
        "sig_action": "",
        "sid_rev": "",
        "mitre_tactic": "",
        "mitre_technique": "",
        "http_method": method,
        "status_code": status,
        "dst_port": "",
        "src_port": None,
        "src_ip_ctx": "",
        "dst_ip_ctx": "",
        "srcuser": "",
        "dstuser": "",
        "db_name": "",
        "audit_type": audit_match.group(1) if audit_match else "",
        "syscall": "",
        "rule_md5": "",
        "sha256_present": 0.0,
        "user_agent": "",
        "log_resource_class": log_resource_class(resources),
        "log_head": raw_head[:400],
        "firedtimes": None,
        "is_training_mode": float(bool(component.get("TrainingMode"))),
        "ac_type": as_str(component.get("AnalysisComponentType")),
        "ac_id": as_str(component.get("AnalysisComponentIdentifier")),
        "ac_message": as_str(component.get("Message")),
        "persistence": as_str(component.get("PersistenceFileName")),
        "log_lines_count": to_num(log_data.get("LogLinesCount")),
        "attr_value_count": float(len(values)),
        "attr_value_max_len": float(max(value_lengths, default=0)),
        "attr_value_max_entropy": float(max((entropy_of(v) for v in values), default=0.0)),
        "attr_path_depth": float(max((len(p.split("/")) for p in path_segments), default=0)),
        "missing_char_count": float(len(missing_chars)),
        "missing_char_special": float(sum(1 for c in "".join(missing_chars) if not c.isalnum())),
        "anomaly_score_max": max([s for s in anomaly_scores if s is not None], default=None),
        "anomaly_score_mean": None,
        "critical_value": to_num(component.get("CriticalValue")),
        "probability_threshold": to_num(component.get("ProbabilityThreshold")),
        "count_confidence": to_num(count_data.get("Confidence")),
        "count_confidence_factor": to_num(count_data.get("ConfidenceFactor")),
        "freq_actual": to_num(frequency.get("LogAtomValuesFrequency")),
        "freq_expected": to_num(frequency.get("ExpectedLogAtomValuesFrequency")),
        "freq_window": to_num(frequency.get("WindowSize")),
        "freq_confidence_factor": to_num(frequency.get("ConfidenceFactor")),
        "freq_range_lo": freq_lo,
        "freq_range_hi": freq_hi,
        "miner_id_ctx": as_str(dict_of(obj.get("AMiner")).get("ID")),
    }
    scores = [s for s in anomaly_scores if s is not None]
    if scores:
        row["anomaly_score_mean"] = sum(scores) / len(scores)
    if len(ranges) >= 2:
        lo_value = to_num(ranges[0])
        hi_value = to_num(ranges[1])
        row["range_lo"], row["range_hi"] = lo_value, hi_value
        if lo_value is not None and hi_value is not None:
            row["range_width"] = hi_value - lo_value
    row.update(url_features(url_candidate, raw_head))
    row.update(dns_features(dns_candidate))
    row["id_attr_numeric"] = float(any(re.fullmatch(r"-?[0-9]+", v) for v in values))
    return row


COLS_STR = [
    "scenario", "source", "label", "phase", "sig_key", "rule_id", "rule_desc", "rule_groups", "decoder",
    "decoder_parent", "location", "agent_id", "agent_name", "agent_ip", "program_name", "hostname_ctx",
    "sig_category", "sig_id", "sig_action", "sid_rev", "mitre_tactic", "mitre_technique", "http_method",
    "status_code", "dst_port", "src_ip_ctx", "dst_ip_ctx", "srcuser", "dstuser", "db_name", "audit_type",
    "syscall", "rule_md5", "user_agent", "log_resource_class", "log_head", "ac_type", "ac_id", "ac_message",
    "persistence", "miner_id_ctx", "url_ext",
]
COLS_NUM = [
    "ts", "rule_level", "sig_severity", "src_port", "firedtimes", "log_lines_count", "is_training_mode",
    "attr_value_count", "attr_value_max_len", "attr_value_max_entropy", "attr_path_depth", "missing_char_count",
    "missing_char_special", "anomaly_score_max", "anomaly_score_mean", "critical_value", "probability_threshold",
    "count_confidence", "count_confidence_factor", "freq_actual", "freq_expected", "freq_window",
    "freq_range_lo", "freq_range_hi",
    "freq_confidence_factor", "range_lo", "range_hi", "range_width", "url_len", "url_parts", "url_query",
    "url_tokens", "url_encoded_chars", "url_semicolon", "url_traversal", "url_entropy", "url_max_token_len",
    "dns_len", "dns_labels", "dns_max_label_len", "dns_avg_label_len", "dns_digits", "dns_letters",
    "dns_digit_ratio", "dns_entropy", "hour", "weekday", "seconds_from_start", "sig_count_prev_60s",
    "sig_count_prev_600s", "group_ids", "group_suricata", "group_web", "group_accesslog", "group_syslog",
    "group_audit", "group_mitre", "is_aggregated", "sha256_present", "id_attr_numeric",
] + list(URL_FLAGS) + list(UA_PATTERNS)
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


def enrich(row: dict, scenario: str, phases: dict, windows: dict, start_ts: float) -> None:
    ts = row["ts"]
    phase = phase_of(phases[scenario], ts)
    row["label"] = LABEL_BACKGROUND if phase == "background" else LABEL_ATTACK
    row["phase"] = phase
    key = f"{row['source']}|{row['rule_desc']}"
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
    writer = pq.ParquetWriter(out_dir / f"{scenario}.features.parquet", schema, compression="zstd")
    windows: dict[str, tuple[deque, deque]] = {}
    buffer: list[dict] = []
    total = 0
    start_ts = 0.0
    try:
        for row in merged_rows(data_dir, scenario):
            if total == 0 and start_ts == 0.0:
                start_ts = row_ts(row)
            enrich(row, scenario, phases, windows, start_ts)
            buffer.append(row)
            if len(buffer) >= BATCH_ROWS:
                writer.write_batch(batch_from_rows(buffer, schema))
                total += len(buffer)
                buffer = []
        if buffer:
            writer.write_batch(batch_from_rows(buffer, schema))
            total += len(buffer)
    finally:
        writer.close()
    return total


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
            f"{scenario}: rows={count} cols={len(COLS_STR) + len(COLS_NUM)} seconds={time.time() - began:.1f}",
            flush=True,
        )
    print(f"TOTAL rows={total} cols={len(COLS_STR) + len(COLS_NUM)}", flush=True)


if __name__ == "__main__":
    main()
