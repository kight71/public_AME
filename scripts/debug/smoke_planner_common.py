"""Shared helpers for planner smoke scripts (heartbeat, crash diagnostics)."""

from __future__ import annotations

import faulthandler
import sys
import traceback
from typing import Any

import torch


def enable_crash_diagnostics() -> None:
    """Print Python tracebacks on crash; dump stacks on SIGSEGV/SIGABRT when possible."""
    faulthandler.enable(all_threads=True, file=sys.stderr)


def run_smoke(name: str, main_fn) -> None:
    """Run ``main_fn``; always exit non-zero on failure with a visible message."""
    enable_crash_diagnostics()
    exit_code = 0
    try:
        main_fn()
        print(f"[{name}] main() returned normally", flush=True)
    except SystemExit as exc:
        exit_code = int(exc.code) if exc.code is not None else 0
        if exit_code != 0:
            print(f"[{name}] SystemExit({exit_code})", flush=True)
    except BaseException:
        print(f"[{name}] FAIL — uncaught exception:", flush=True)
        traceback.print_exc()
        exit_code = 1
    finally:
        if exit_code == 0:
            print(f"[{name}] exiting 0", flush=True)
        else:
            print(f"[{name}] exiting {exit_code}", flush=True)
        sys.exit(exit_code)


def zero_actions(env) -> torch.Tensor:
    return torch.zeros(env.action_space.shape, device=env.unwrapped.device)


def format_cmd_buffers(cmd, env_idx: int = 0) -> str:
    scores = cmd.foothold_score_buffer[env_idx].tolist()
    fallback = cmd.used_fallback_buffer[env_idx].tolist()
    valid = cmd.valid_count_buffer[env_idx].tolist()
    return f"foothold_score={scores} fallback={fallback} valid_count={valid}"


def step_loop(
    env,
    steps: int,
    *,
    prefix: str,
    cmd: Any | None = None,
    heartbeat: int = 5,
) -> None:
    """Step env with periodic heartbeats; re-raise any exception with step index."""
    actions = zero_actions(env)
    print(f"[{prefix}] starting {steps} zero-action steps (heartbeat every {heartbeat})", flush=True)
    for step_idx in range(steps):
        try:
            env.step(actions)
        except Exception as exc:
            print(f"[{prefix}] FAIL at step {step_idx}/{steps}: {type(exc).__name__}: {exc}", flush=True)
            raise
        if cmd is not None and (step_idx == 0 or (step_idx + 1) % heartbeat == 0):
            print(f"[{prefix}] step {step_idx + 1}/{steps}  {format_cmd_buffers(cmd)}", flush=True)
        elif cmd is None and (step_idx == 0 or (step_idx + 1) % heartbeat == 0):
            print(f"[{prefix}] step {step_idx + 1}/{steps} ok", flush=True)
    print(f"[{prefix}] finished {steps} steps", flush=True)
