
import json
import csv
import re
import random
from pathlib import Path
from datetime import datetime

# DOSTOSUJ ŚCIEŻKI DO SWOJEGO PROJEKTU
JSON_PATH = Path("data/ait_ads/fox_wazuh.json")
CSV_PATH = Path("data/ait_ads/labels.csv")

OUTPUT_PATH = Path("artifacts/sample_alerts.json")

SAMPLES_PER_PHASE = 10
BACKGROUND_SAMPLES = 20
SEED = 42

rng = random.Random(SEED)

def parse_timestamp(value):
    """Konwertuje ISO timestamp na Unix timestamp."""
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        ).timestamp()
    except (ValueError, TypeError):
        return None


def anonymize(obj):
    """Usuwa adresy IP i identyfikatory agentów."""

    if isinstance(obj, dict):
        result = {}

        for key, value in obj.items():
            key_lower = key.lower()

            if key_lower in {
                "ip", "agent.id", "agent_id",
                "srcip", "dstip", "src_ip", "dst_ip"
            }:
                result[key] = "REDACTED"
            elif key_lower == "id":
                result[key] = "REDACTED"
            elif key_lower == "name" and "agent" in str(obj).lower():
                result[key] = "AGENT"
            else:
                result[key] = anonymize(value)

        return result

    if isinstance(obj, list):
        return [anonymize(x) for x in obj]

    if isinstance(obj, str):
        # Anonimizacja IPv4 również wewnątrz full_log
        return re.sub(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "IP_REDACTED",
            obj
        )

    return obj


# Scenariusz odczytujemy z nazwy pliku.
scenario = JSON_PATH.stem.replace("_wazuh", "")

# Wczytanie przedziałów ataków
phases = {}

with CSV_PATH.open(encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["scenario"] != scenario:
            continue

        attack = row["attack"]

        phases.setdefault(attack, []).append((
            float(row["start"]),
            float(row["end"])
        ))

if not phases:
    raise ValueError(
        f"Nie znaleziono scenariusza {scenario} w labels.csv"
    )

# Liczniki i próbki
counts = {phase: 0 for phase in phases}
samples = {phase: [] for phase in phases}

counts["background"] = 0
samples["background"] = []

invalid_timestamps = 0
overlaps = 0

# Przetwarzanie strumieniowe - bez ładowania 500 tys. alertów do RAM
with JSON_PATH.open(encoding="utf-8") as f:

    for line_number, line in enumerate(f, 1):
        try:
            alert = json.loads(line)
        except json.JSONDecodeError:
            continue

        ts = parse_timestamp(alert.get("@timestamp"))

        if ts is None:
            invalid_timestamps += 1
            continue

        matching = [
            phase
            for phase, intervals in phases.items()
            if any(start <= ts < end for start, end in intervals)
        ]

        if len(matching) > 1:
            overlaps += 1

        # Zachowujemy wyłącznie jednoznaczne etykiety czasowe.
        if len(matching) == 1:
            label = matching[0]
        elif len(matching) == 0:
            label = "background"
        else:
            continue

        counts[label] += 1

        # Reservoir sampling - reprezentatywna próbka
        limit = (
            BACKGROUND_SAMPLES
            if label == "background"
            else SAMPLES_PER_PHASE
        )

        bucket = samples[label]
        anonymized_alert = anonymize(alert)

        if len(bucket) < limit:
            bucket.append(anonymized_alert)
        else:
            index = rng.randrange(counts[label])

            if index < limit:
                bucket[index] = anonymized_alert

# Raport
print("\n========== ALERT COUNTS ==========")
print("Scenario:", scenario)

for label, count in counts.items():
    print(f"{label:25} {count:10}")

print("\nInvalid timestamps:", invalid_timestamps)
print("Overlapping intervals:", overlaps)

# Przygotowanie pliku do analizy
output = []

for label, records in samples.items():
    for record in records:
        output.append({
            "time_label": label,
            "alert": record
        })

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\nSamples saved:", OUTPUT_PATH)
print("Total selected:", len(output))
