#!/usr/bin/env bash
# Caly pipeline AIT-ADS od surowych JSONL do sprawozdania Markdown.
# Uzycie: bash run_all.sh (calosc w logs/run_all.log)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

mkdir -p logs artifacts model report/figures data/features
LOG="logs/run_all.log"
: > "$LOG"

run() {
  echo "==> $*" | tee -a "$LOG"
  start=$(date +%s)
  "$@" 2>&1 | tee -a "$LOG"
  echo "    elapsed $(( $(date +%s) - start ))s" | tee -a "$LOG"
}

run "$PY" -u research/build_features.py data/features
run "$PY" -u research/report_stats.py
run "$PY" -u research/train.py --protocols los,chrono --configs lgb_structured,lgb_text --deploy
run "$PY" -u research/analyze_predictions.py
run "$PY" -u research/figures.py
run "$PY" -u research/finalize_bundle.py
run "$PY" -u report/build_report.py

echo "done" | tee -a "$LOG"
echo "sprawozdanie: report/sprawozdanie.md"
echo "model: model/model.zip"
