"""Deprecated wrapper — run smokes in **separate processes** (one Isaac env each).

Isaac Sim often hangs when ``gym.make`` is called twice in one Kit session.
Use these instead:

.. code-block:: bash

    .AME/bin/python scripts/debug/smoke_planner_legacy_repro.py --headless --enable_cameras --steps 30
    .AME/bin/python scripts/debug/smoke_planner_v2_play.py --headless --enable_cameras --steps 30
    .AME/bin/python scripts/debug/smoke_planner_v2_legacy_play.py --headless --enable_cameras --steps 30

Or run all three in fresh processes:

.. code-block:: bash

    bash scripts/debug/run_planner_smokes.sh --headless --enable_cameras --steps 30
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "smoke_planner_legacy_repro.py",
    "smoke_planner_v2_play.py",
    "smoke_planner_v2_legacy_play.py",
]


def main() -> None:
    root = Path(__file__).resolve().parent
    py = sys.executable
    extra = [a for a in sys.argv[1:] if not a.startswith("-h")]
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        return
    for name in SCRIPTS:
        path = root / name
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([py, str(path), *extra], check=True)
    print("\nSmoke OK (all subprocesses passed)", flush=True)


if __name__ == "__main__":
    main()
