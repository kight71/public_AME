#!/usr/bin/env bash
# Phase 0 acceptance helper (PLAN.md Task 0.6 / 0.7).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.AME/bin/python}"
OUT="${OUT:-scripts/debug/output/debug_plan.npz}"
TASK="${TASK:-AME-G1-29DOF-DTC-Play-v0}"
STEPS="${STEPS:-500}"

echo "== Phase 0 offline dump =="
echo "task=$TASK num_envs=1 steps=$STEPS"
"$PY" scripts/debug_footstep_plan.py --headless --num_envs 1 --num_steps "$STEPS" \
  --task "$TASK" --out "$OUT"

echo ""
echo "== Phase 0 offline plot + sanity =="
"$PY" scripts/debug/plot_footstep_plan.py "$OUT"

echo ""
echo "== Phase 0 GUI eyeball (run manually, no headless) =="
cat <<EOF

Open a terminal WITH display and run:

  $PY scripts/rsl_rl/play.py \\
      --task AME-G1-29DOF-Play-v0 \\
      --checkpoint pretrained/ame1.pt \\
      --num_envs 1

What to verify in the viewport:
  • Blue sphere  = planned left foot target  (target_w[:,0])
  • Red sphere   = planned right foot target (target_w[:,1])
  • Gray spheres = last touchdown per foot   (last_contact_w)
  • Yellow (DTC) = phantom pelvis when use_phantom=True

Frame / marker checks:
  1. Markers sit ON the terrain mesh (z snapped via height_scanner), not floating
     or buried by >5 cm on flat patches.
  2. On hollow stairs / step edges, targets shift to safer cells (cost selection
     on DTC) or nearest-ray z (default Play-v0).
  3. Markers stay fixed in WORLD frame between swing crossovers; only jump when
     the swing foot swaps (~every t_step/2).
  4. Blue/red alternate which foot is closer to the gray last-contact marker for
     the current swing leg.

Optional rough-terrain DTC play:

  $PY scripts/rsl_rl/play.py \\
      --task AME-G1-29DOF-DTC-Play-v0 \\
      --checkpoint <your_dtc_ckpt.pt> \\
      --num_envs 1

EOF
