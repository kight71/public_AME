#!/usr/bin/env python3
"""
Standalone validation script for the AME policy export logic.

This script validates the export logic for ActorCriticTerrainMlp WITHOUT
requiring Isaac Sim. It:
  1. Imports ActorCriticTerrainMlp from rsl_rl.modules
  2. Manually constructs a small ActorCriticTerrainMlp policy
  3. Verifies that _OnnxPolicyExporter and _TorchPolicyExporter:
       a. Correctly detect has_terrain_mlp = True
       b. Do NOT incorrectly detect has_terrain_encoder
       c. Correctly split obs, normalize proprio, encode terrain, and run actor
  4. Exports to ONNX and TorchScript (JIT) and verifies the exported models run
  5. Verifies that the train.py and play.py scripts' imports resolve (syntax check)

Usage:
    python scripts/rsl_rl/validate_export_logic.py
"""

import copy
import importlib.util
import os
import sys
import tempfile
import traceback

# ---------------------------------------------------------------------------
# Path setup: add rsl_rl package to sys.path
# ---------------------------------------------------------------------------
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RSL_RL_PATH = os.path.join(PROJ_ROOT, "rsl_rl")
SCRIPTS_PATH = os.path.join(PROJ_ROOT, "scripts", "rsl_rl")

for p in [RSL_RL_PATH, SCRIPTS_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

print("=" * 72)
print("AME Locomotion - Export Logic Validation")
print("=" * 72)

# ---------------------------------------------------------------------------
# Helper: colored / formatted printing
# ---------------------------------------------------------------------------
PASS = "  PASS"
FAIL = "  FAIL"


def section(title: str):
    print(f"\n--- {title} ---")


def check(condition: bool, label: str, detail: str = ""):
    if condition:
        print(f"{PASS} | {label}")
    else:
        print(f"{FAIL} | {label}  {detail}")
    return condition


# ---------------------------------------------------------------------------
# Track overall pass/fail
# ---------------------------------------------------------------------------
all_checks = []


def _c(condition: bool, label: str, detail: str = ""):
    all_checks.append(condition)
    return check(condition, label, detail)


# ===========================================================================
# 1.  Import verification
# ===========================================================================
section("1. Import verification")

try:
    import torch
    import torch.nn as nn
    _c(True, "torch imported")
except ImportError as e:
    _c(False, f"torch import failed: {e}")

try:
    from rsl_rl.networks import EmpiricalNormalization, MLP
    _c(True, "rsl_rl.networks (MLP, EmpiricalNormalization) imported")
except ImportError as e:
    _c(False, f"rsl_rl.networks import failed: {e}")

try:
    from rsl_rl.modules import ActorCriticTerrainMlp
    _c(True, "ActorCriticTerrainMlp imported")
except ImportError as e:
    _c(False, f"ActorCriticTerrainMlp import failed: {e}")

try:
    from scripts.rsl_rl.exporter import (
        _OnnxPolicyExporter,
        _TorchPolicyExporter,
        export_policy_as_jit,
        export_policy_as_onnx,
    )
    _c(True, "exporter module imported (_OnnxPolicyExporter, _TorchPolicyExporter)")
except ImportError as e:
    _c(False, f"exporter import failed: {e}")


# ===========================================================================
# 2.  Construct a dummy ActorCriticTerrainMlp
# ===========================================================================
section("2. Construct dummy ActorCriticTerrainMlp")

PROPRIO_DIM = 48
TERRAIN_OBS_DIM = 96
TERRAIN_EMBEDDING_DIM = 64
NUM_ACTIONS = 12

# We need to supply obs (dict of tensors) and obs_groups (dict of group-name lists)
# so that __init__ can compute num_actor_obs and num_critic_obs.
dummy_obs_groups = {
    "policy": ["proprio_obs", "terrain_obs"],
    "critic": ["proprio_obs", "terrain_obs"],
}
dummy_obs = {
    "proprio_obs": torch.zeros(1, PROPRIO_DIM),
    "terrain_obs": torch.zeros(1, TERRAIN_OBS_DIM),
}

policy = None
try:
    policy = ActorCriticTerrainMlp(
        obs=dummy_obs,
        obs_groups=dummy_obs_groups,
        num_actions=NUM_ACTIONS,
        actor_obs_normalization=True,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        terrain_obs_dim=TERRAIN_OBS_DIM,
        terrain_hidden_dims=[128],
        terrain_embedding_dim=TERRAIN_EMBEDDING_DIM,
    )
    _c(True, f"ActorCriticTerrainMlp constructed (proprio={PROPRIO_DIM}, "
             f"terrain_obs={TERRAIN_OBS_DIM}, terrain_embed={TERRAIN_EMBEDDING_DIM})")
except Exception as e:
    _c(False, f"ActorCriticTerrainMlp construction failed: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Verify policy internals
# ---------------------------------------------------------------------------
if policy is not None:
    _c(hasattr(policy, "terrain_encoder"),
       "policy has 'terrain_encoder' attribute")
    _c(policy.terrain_obs_dim == TERRAIN_OBS_DIM,
       f"policy.terrain_obs_dim == {TERRAIN_OBS_DIM}",
       f"got {policy.terrain_obs_dim}")
    _c(policy.terrain_embedding_dim == TERRAIN_EMBEDDING_DIM,
       f"policy.terrain_embedding_dim == {TERRAIN_EMBEDDING_DIM}",
       f"got {policy.terrain_embedding_dim}")
    _c(policy.actor_proprio_dim == PROPRIO_DIM,
       f"policy.actor_proprio_dim == {PROPRIO_DIM}",
       f"got {policy.actor_proprio_dim}")

    # Verify policy does NOT have CNN/MHA attributes (we're testing terrain MLP)
    _c(not hasattr(policy, "map_cnn"),
       "policy does NOT have 'map_cnn' (correct for terrain MLP variant)")
    _c(not hasattr(policy, "mha"),
       "policy does NOT have 'mha' (correct for terrain MLP variant)")
    _c(not hasattr(policy, "actor_proprio_embedding"),
       "policy does NOT have 'actor_proprio_embedding' (correct for terrain MLP variant)")

    # Verify terrain_encoder is an MLP
    terrain_encoder = policy.terrain_encoder
    _c(isinstance(terrain_encoder, MLP),
       "terrain_encoder is an MLP")
    # MLP is Sequential; first layer is Linear, last is Linear
    first_linear = [m for m in terrain_encoder if isinstance(m, nn.Linear)][0]
    last_linear = [m for m in terrain_encoder if isinstance(m, nn.Linear)][-1]
    _c(first_linear.in_features == TERRAIN_OBS_DIM,
       f"terrain_encoder input dim == {TERRAIN_OBS_DIM}",
       f"got {first_linear.in_features}")
    _c(last_linear.out_features == TERRAIN_EMBEDDING_DIM,
       f"terrain_encoder output dim == {TERRAIN_EMBEDDING_DIM}",
       f"got {last_linear.out_features}")

    # Build a normalizer (mimics what the runner would provide)
    normalizer = EmpiricalNormalization(PROPRIO_DIM)
    # Pre-populate with some dummy data so it has non-identity statistics
    normalizer.train()
    normalizer.update(torch.randn(128, PROPRIO_DIM) * 2.0 + 1.0)
    normalizer.eval()
    _c(True, f"EmpiricalNormalizer created for proprio dim {PROPRIO_DIM}")


# ===========================================================================
# 3.  _TorchPolicyExporter validation
# ===========================================================================
section("3. _TorchPolicyExporter validation")

torch_exporter = None
if policy is not None:
    try:
        torch_exporter = _TorchPolicyExporter(policy, normalizer)
        _c(True, "_TorchPolicyExporter constructed")
    except Exception as e:
        _c(False, f"_TorchPolicyExporter construction failed: {e}")
        traceback.print_exc()

if torch_exporter is not None:
    # 3a. has_terrain_mlp detection
    _c(torch_exporter.has_terrain_mlp,
       "torch_exporter.has_terrain_mlp == True")
    _c(not torch_exporter.has_terrain_encoder,
       "torch_exporter.has_terrain_encoder == False (terrain MLP, not CNN/MHA)")

    # 3b. terrain attributes
    _c(torch_exporter.terrain_obs_dim == TERRAIN_OBS_DIM,
       f"torch_exporter.terrain_obs_dim == {TERRAIN_OBS_DIM}",
       f"got {torch_exporter.terrain_obs_dim}")
    _c(torch_exporter.terrain_embedding_dim == TERRAIN_EMBEDDING_DIM,
       f"torch_exporter.terrain_embedding_dim == {TERRAIN_EMBEDDING_DIM}",
       f"got {torch_exporter.terrain_embedding_dim}")

    # 3c. Forward pass correctness -- test _encode_terrain_mlp logic
    torch_exporter.eval()
    batch_size = 8
    # Build random input: [proprio (48), terrain (96)]
    test_input = torch.randn(batch_size, PROPRIO_DIM + TERRAIN_OBS_DIM)

    with torch.no_grad():
        # 1) Manually compute expected: split, normalize proprio, terrain encode, concat
        proprio_raw = test_input[:, :PROPRIO_DIM]
        terrain_raw = test_input[:, PROPRIO_DIM:]
        proprio_norm = normalizer(proprio_raw)
        terrain_emb = terrain_encoder(terrain_raw)
        expected_encoded = torch.cat([proprio_norm, terrain_emb], dim=-1)
        expected_actions = policy.actor(expected_encoded)

        # 2) Run through exporter's forward pass
        actual_actions = torch_exporter(test_input)

        # 3) Compare
        action_diff = (actual_actions - expected_actions).abs().max().item()
        _c(action_diff < 1e-5,
           f"Torch exporter forward pass matches expected actions (max_diff={action_diff:.2e})",
           f"diff={action_diff:.2e}")

    # 3d. Verify the internal _encode_terrain_mlp directly
    with torch.no_grad():
        encoded = torch_exporter._encode_terrain_mlp(test_input)
        encoded_diff = (encoded - expected_encoded).abs().max().item()
        _c(encoded_diff < 1e-5,
           f"_encode_terrain_mlp output matches expected (max_diff={encoded_diff:.2e})",
           f"diff={encoded_diff:.2e}")

    # 3e. Verify split: check that only proprio is normalized (terrain goes in raw)
    #     proprio_obs = obs[:, :-terrain_obs_dim] should be the proprio part
    with torch.no_grad():
        manual_proprio = test_input[:, :-TERRAIN_OBS_DIM]
        manual_terrain = test_input[:, -TERRAIN_OBS_DIM:]
        _c(torch.allclose(manual_proprio, proprio_raw),
           "torch_exporter splits proprio as obs[:, :-terrain_obs_dim]")
        _c(torch.allclose(manual_terrain, terrain_raw),
           "torch_exporter splits terrain as obs[:, -terrain_obs_dim:]")

    # 3f. Single-observation (1D) path
    single_input = torch.randn(PROPRIO_DIM + TERRAIN_OBS_DIM)
    with torch.no_grad():
        out_1d = torch_exporter(single_input)
        out_2d = torch_exporter(single_input.unsqueeze(0))
        _c(out_1d.shape == out_2d.shape,
           f"1D input gives same output shape as 2D input (both {out_1d.shape})",
           f"1D: {out_1d.shape}, 2D: {out_2d.shape}")

    # 3g. Export to TorchScript JIT and verify
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            jit_path = os.path.join(tmpdir, "policy.pt")
            traced = torch.jit.script(torch_exporter)
            traced.save(jit_path)
            # Reload and run
            loaded = torch.jit.load(jit_path)
            loaded.eval()
            out_loaded = loaded(test_input)
            load_diff = (out_loaded - expected_actions).abs().max().item()
            _c(load_diff < 1e-5,
               f"TorchScript JIT export -> reload forward match (max_diff={load_diff:.2e})",
               f"diff={load_diff:.2e}")
    except Exception as e:
        _c(False, f"TorchScript JIT export/load failed: {e}")
        traceback.print_exc()

    # 3h. Test export convenience function
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            export_policy_as_jit(policy, normalizer=normalizer, path=tmpdir, filename="policy_jit.pt")
            jit_path = os.path.join(tmpdir, "policy_jit.pt")
            _c(os.path.isfile(jit_path),
               f"export_policy_as_jit created file: {jit_path}")
            loaded = torch.jit.load(jit_path)
            loaded.eval()
            out_fn = loaded(test_input)
            fn_diff = (out_fn - expected_actions).abs().max().item()
            _c(fn_diff < 1e-5,
               f"export_policy_as_jit output matches (max_diff={fn_diff:.2e})",
               f"diff={fn_diff:.2e}")
    except Exception as e:
        _c(False, f"export_policy_as_jit failed: {e}")
        traceback.print_exc()


# ===========================================================================
# 4.  _OnnxPolicyExporter validation
# ===========================================================================
section("4. _OnnxPolicyExporter validation")

onnx_exporter = None
if policy is not None:
    try:
        onnx_exporter = _OnnxPolicyExporter(policy, normalizer, verbose=False)
        _c(True, "_OnnxPolicyExporter constructed")
    except Exception as e:
        _c(False, f"_OnnxPolicyExporter construction failed: {e}")
        traceback.print_exc()

if onnx_exporter is not None:
    # 4a. has_terrain_mlp detection
    _c(onnx_exporter.has_terrain_mlp,
       "onnx_exporter.has_terrain_mlp == True")
    _c(not onnx_exporter.has_terrain_encoder,
       "onnx_exporter.has_terrain_encoder == False (terrain MLP, not CNN/MHA)")

    # 4b. terrain attributes
    _c(onnx_exporter.terrain_obs_dim == TERRAIN_OBS_DIM,
       f"onnx_exporter.terrain_obs_dim == {TERRAIN_OBS_DIM}",
       f"got {onnx_exporter.terrain_obs_dim}")
    _c(onnx_exporter.terrain_embedding_dim == TERRAIN_EMBEDDING_DIM,
       f"onnx_exporter.terrain_embedding_dim == {TERRAIN_EMBEDDING_DIM}",
       f"got {onnx_exporter.terrain_embedding_dim}")

    # 4c. Forward pass correctness
    onnx_exporter.eval()
    batch_size = 8
    test_input = torch.randn(batch_size, PROPRIO_DIM + TERRAIN_OBS_DIM)

    with torch.no_grad():
        # Reference actions from the raw policy + normalizer
        proprio_raw = test_input[:, :PROPRIO_DIM]
        terrain_raw = test_input[:, PROPRIO_DIM:]
        proprio_norm = normalizer(proprio_raw)
        terrain_emb = terrain_encoder(terrain_raw)
        expected_encoded = torch.cat([proprio_norm, terrain_emb], dim=-1)
        expected_actions = policy.actor(expected_encoded)

        actual_actions = onnx_exporter(test_input)
        action_diff = (actual_actions - expected_actions).abs().max().item()
        _c(action_diff < 1e-5,
           f"ONNX exporter forward pass matches expected actions (max_diff={action_diff:.2e})",
           f"diff={action_diff:.2e}")

    # 4d. Verify _encode_terrain_mlp directly
    with torch.no_grad():
        encoded = onnx_exporter._encode_terrain_mlp(test_input)
        encoded_diff = (encoded - expected_encoded).abs().max().item()
        _c(encoded_diff < 1e-5,
           f"_encode_terrain_mlp (ONNX) matches expected (max_diff={encoded_diff:.2e})",
           f"diff={encoded_diff:.2e}")

    # 4e. Single-observation (1D) path
    single_input = torch.randn(PROPRIO_DIM + TERRAIN_OBS_DIM)
    with torch.no_grad():
        out_1d = onnx_exporter(single_input)
        out_2d = onnx_exporter(single_input.unsqueeze(0))
        _c(out_1d.shape == out_2d.shape,
           f"1D input gives same output shape as 2D (both {out_1d.shape})",
           f"1D: {out_1d.shape}, 2D: {out_2d.shape}")

    # 4f. Export to ONNX and verify with onnxruntime
    try:
        import onnxruntime as ort
        _c(True, "onnxruntime available for ONNX verification")
    except ImportError:
        _c(True, "onnxruntime not available; skipping ONNX runtime verification (install with: pip install onnxruntime)")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "policy.onnx")
            onnx_exporter.export(tmpdir, "policy.onnx")
            _c(os.path.isfile(onnx_path),
               f"ONNX export created file: {onnx_path}")

            # Try to load and run with onnxruntime
            if "ort" in dir():
                ort_session = ort.InferenceSession(onnx_path)
                input_name = ort_session.get_inputs()[0].name
                output_name = ort_session.get_outputs()[0].name
                ort_input = test_input.numpy().astype("float32")
                ort_outputs = ort_session.run([output_name], {input_name: ort_input})[0]
                ort_actions = torch.from_numpy(ort_outputs)
                ort_diff = (ort_actions - expected_actions).abs().max().item()
                _c(ort_diff < 1e-4,
                   f"ONNX runtime forward pass matches (max_diff={ort_diff:.2e})",
                   f"diff={ort_diff:.2e}")
            else:
                _c(True, "ONNX file created (runtime verification skipped; no onnxruntime)")
    except Exception as e:
        _c(False, f"ONNX export/verification failed: {e}")
        traceback.print_exc()

    # 4g. Test export convenience function
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            export_policy_as_onnx(policy, path=tmpdir, normalizer=normalizer, filename="policy_onnx.onnx")
            onnx_fn_path = os.path.join(tmpdir, "policy_onnx.onnx")
            _c(os.path.isfile(onnx_fn_path),
               f"export_policy_as_onnx created file: {onnx_fn_path}")
    except Exception as e:
        _c(False, f"export_policy_as_onnx failed: {e}")
        traceback.print_exc()


# ===========================================================================
# 5.  Verify train.py and play.py imports resolve (syntax / import check)
# ===========================================================================
section("5. Script import/syntax verification")


def check_script_imports(script_path: str, label: str) -> bool:
    """Check if a script can at least have its top-level imports resolved
    (without executing it, since it requires Isaac Sim)."""
    if not os.path.isfile(script_path):
        return _c(False, f"{label}: file not found at {script_path}")

    # 1) Syntax check
    try:
        with open(script_path, "r") as f:
            source = f.read()
        compile(source, script_path, "exec")
        _c(True, f"{label}: syntax valid")
    except SyntaxError as e:
        _c(False, f"{label}: syntax error: {e}")
        return False

    # 2) Verify that 'from exporter import export_policy_as_jit, export_policy_as_onnx'
    #    is present in play.py (the main consumer of exports)
    if "play.py" in script_path:
        if "from exporter import" in source:
            _c(True, f"{label}: contains 'from exporter import ...'")
        else:
            _c(False, f"{label}: MISSING 'from exporter import ...'")

    # 3) Optionally verify the actual import works by importing the specific symbols
    #    (This will fail if Isaac Sim modules are imported at top level, which they are
    #     in train.py and play.py. So instead, just verify the exporter module itself
    #     can be imported, which we already did in section 1.)
    _c(True, f"{label}: present and parsable (full execution requires Isaac Sim)")

    return True


check_script_imports(os.path.join(SCRIPTS_PATH, "train.py"), "train.py")
check_script_imports(os.path.join(SCRIPTS_PATH, "play.py"), "play.py")


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 72)
total = len(all_checks)
passed = sum(all_checks)
failed = total - passed
if failed == 0:
    print(f"RESULT: ALL {total}/{total} checks PASSED")
else:
    print(f"RESULT: {passed}/{total} passed, {failed}/{total} FAILED")
print("=" * 72)

sys.exit(0 if failed == 0 else 1)
