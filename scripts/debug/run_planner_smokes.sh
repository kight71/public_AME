#!/usr/bin/env bash
# Run planner smokes in separate Isaac processes (avoids second gym.make hang).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-.AME/bin/python}"
EXTRA=("$@")

run() {
  echo ""
  echo "=== $1 ==="
  "$PY" "$ROOT/$1" "${EXTRA[@]}"
}

run smoke_planner_legacy_repro.py
run smoke_planner_v2_play.py
run smoke_planner_v2_legacy_play.py
echo ""
echo "Smoke OK (all three passed)"
