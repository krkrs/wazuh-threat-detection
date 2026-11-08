
import json
import csv

JSON_PATH = "data/ait_ads/fox_wazuh.json"
CSV_PATH = "data/ait_ads/labels.csv"

print("\n========== JSON STRUCTURE ==========")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    for i in range(3):
        line = f.readline()
        if not line:
            break

        try:
            alert = json.loads(line)
            print(f"\nRECORD {i + 1}")
            print("Top-level keys:", list(alert.keys()))
            print("Timestamp:", alert.get("timestamp"))
            print("Rule:", alert.get("rule"))
            print("Agent:", alert.get("agent"))
            print("Data keys:",
                  list(alert.get("data", {}).keys())
                  if isinstance(alert.get("data"), dict)
                  else None)
        except json.JSONDecodeError:
            print("JSON is not in JSONL format.")
            print("First characters:", line[:300])
            break

print("\n========== CSV STRUCTURE ==========")

with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    print("CSV columns:", reader.fieldnames)

    for i, row in enumerate(reader):
        if i == 5:
            break
        print(row)
