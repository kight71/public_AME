"""Unit tests for mdp/foothold_selection.py."""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

_MDP_DIR = (
    Path(__file__).resolve().parents[1]
    / "source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp"
)
_PKG = "foothold_selection_test_mdp"


def _load_package_module(module_name: str):
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(_MDP_DIR)]
        sys.modules[_PKG] = pkg

    full_name = f"{_PKG}.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    path = _MDP_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(full_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def _load():
    _load_package_module("foot_geometry_constants")
    _load_package_module("planner")
    _load_package_module("gridmap_utils")
    _load_package_module("foothold_candidates")
    _load_package_module("foothold_geometry")
    _load_package_module("foothold_costs")
    return _load_package_module("foothold_selection")


@pytest.fixture(scope="module")
def fs():
    return _load()


def _grid_rays(center_xy, half_size=0.5, res=0.05, z_fn=lambda x, y: 0.0):
    """Build a (1, K, 3) ray_hits_w tensor for one env over a (half_size*2)^2 patch."""
    n = int(2 * half_size / res) + 1
    xs = torch.linspace(center_xy[0] - half_size, center_xy[0] + half_size, n)
    ys = torch.linspace(center_xy[1] - half_size, center_xy[1] + half_size, n)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    z = z_fn(gx, gy)
    if not isinstance(z, torch.Tensor):
        z = torch.full_like(gx, float(z))
    return torch.stack([gx, gy, z], dim=-1).reshape(1, -1, 3)


def test_flat_terrain_returns_raibert_center(fs):
    raibert = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]]])  # (1, 2, 3)
    rays = _grid_rays((0.0, 0.1))
    sel, cost = fs.select_foothold_by_cost(
        raibert_xy_w=raibert, ray_hits_w=rays,
        window_m=0.10, alpha=2.0, beta=1.0, gamma=5.0, delta=1.0,
        obstacle_height_threshold=0.15,
    )
    # On flat terrain the cost minimum should sit on (delta only) → at the
    # cell closest to the Raibert xy.
    assert torch.allclose(sel[..., :2], raibert[..., :2], atol=0.05)


def test_step_edge_penalty_pushes_off_edge(fs):
    # Step edge at x=0.0: z=0 for x<0, z=0.15 for x>=0.
    def step(x, y):
        return (x >= 0.0).float() * 0.15
    raibert = torch.tensor([[[0.0, 0.0, 0.0]]])  # right on the edge
    rays = _grid_rays((0.0, 0.0), z_fn=step)
    sel, _ = fs.select_foothold_by_cost(
        raibert_xy_w=raibert, ray_hits_w=rays,
        window_m=0.10, alpha=2.0, beta=1.0, gamma=5.0, delta=1.0,
        obstacle_height_threshold=0.15,
    )
    # Roughness (max-min in window) is high on the edge → selector should
    # pick a cell away from the discontinuity.
    assert abs(sel[0, 0, 0].item()) > 0.02


def test_output_z_matches_grid_z(fs):
    def slope(x, y):
        return 0.1 * x
    raibert = torch.tensor([[[0.3, 0.0, 0.0]]])
    rays = _grid_rays((0.3, 0.0), z_fn=slope)
    sel, _ = fs.select_foothold_by_cost(
        raibert_xy_w=raibert, ray_hits_w=rays,
        window_m=0.05, alpha=0.0, beta=0.0, gamma=0.0, delta=1.0,
        obstacle_height_threshold=0.15,
    )
    # With only delta active and a tight window, selector picks the cell
    # closest to Raibert; its z must match slope(x≈0.3) ≈ 0.03.
    assert sel[0, 0, 2].item() == pytest.approx(0.03, abs=0.01)


# ---------------------------------------------------------------------------
# Grid-path tests (max_pool2d fast path)
# ---------------------------------------------------------------------------

def _grid_rays_regular(B: int, H: int, W: int, res: float, z_fn=None):
    """Build a (B, H*W, 3) ray_hits_w tensor using row-major (H, W) layout.

    Matches IsaacLab grid_pattern ordering="xy":
      outer loop y (rows, H), inner loop x (cols, W).
    Grid is centred at origin. z defaults to 0.
    """
    x = torch.linspace(-res * (W - 1) / 2, res * (W - 1) / 2, W)
    y = torch.linspace(-res * (H - 1) / 2, res * (H - 1) / 2, H)
    # indexing="xy": grid_x.shape=(H, W), outer=y, inner=x
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    if z_fn is not None:
        z = z_fn(grid_x, grid_y)
        if not isinstance(z, torch.Tensor):
            z = torch.full_like(grid_x, float(z))
    else:
        z = torch.zeros_like(grid_x)
    flat = torch.stack([grid_x, grid_y, z], dim=-1).reshape(1, H * W, 3)
    return flat.expand(B, -1, -1).contiguous()


def test_grid_path_matches_brute_force_on_flat(fs):
    """Grid path and brute-force path must choose the same foothold on flat terrain."""
    B, H, W = 2, 11, 11
    res = 0.05
    rays = _grid_rays_regular(B, H, W, res)
    # Two arbitrary Raibert targets inside the grid
    raibert = torch.zeros(B, 2, 3)
    raibert[0, 0, :2] = torch.tensor([0.0, 0.0])
    raibert[0, 1, :2] = torch.tensor([0.1, 0.05])
    raibert[1, 0, :2] = torch.tensor([-0.05, 0.0])
    raibert[1, 1, :2] = torch.tensor([0.0, 0.1])

    common_kwargs = dict(
        window_m=0.10, alpha=2.0, beta=1.0, gamma=5.0, delta=1.0,
        obstacle_height_threshold=0.15,
    )
    sel_brute, cost_brute = fs.select_foothold_by_cost(
        raibert_xy_w=raibert, ray_hits_w=rays, **common_kwargs
    )
    sel_grid, cost_grid = fs.select_foothold_by_cost(
        raibert_xy_w=raibert, ray_hits_w=rays,
        grid_shape=(H, W), grid_resolution=res,
        **common_kwargs,
    )
    # On flat terrain roughness = 0 everywhere; both paths should pick cells at
    # the same (x, y) — the cell nearest the Raibert prior.
    assert torch.allclose(sel_brute[..., :2], sel_grid[..., :2], atol=1e-5), (
        f"brute={sel_brute[..., :2]}\ngrid={sel_grid[..., :2]}"
    )


def test_grid_path_step_edge(fs):
    """Grid path should avoid a step edge just like brute-force does."""
    B, H, W = 1, 11, 11
    res = 0.05

    def step(x, y):
        return (x >= 0.0).float() * 0.15

    rays = _grid_rays_regular(B, H, W, res, z_fn=step)
    raibert = torch.tensor([[[0.0, 0.0, 0.0]]])  # right on the edge

    sel_brute, _ = fs.select_foothold_by_cost(
        raibert_xy_w=raibert, ray_hits_w=rays,
        window_m=0.10, alpha=2.0, beta=1.0, gamma=5.0, delta=1.0,
        obstacle_height_threshold=0.15,
    )
    sel_grid, _ = fs.select_foothold_by_cost(
        raibert_xy_w=raibert, ray_hits_w=rays,
        window_m=0.10, alpha=2.0, beta=1.0, gamma=5.0, delta=1.0,
        obstacle_height_threshold=0.15,
        grid_shape=(H, W), grid_resolution=res,
    )
    # Both paths should push the selection away from the discontinuity.
    assert abs(sel_brute[0, 0, 0].item()) > 0.02, (
        f"brute-force failed to avoid edge: x={sel_brute[0, 0, 0].item()}"
    )
    assert abs(sel_grid[0, 0, 0].item()) > 0.02, (
        f"grid path failed to avoid edge: x={sel_grid[0, 0, 0].item()}"
    )


def test_grid_path_memory_smoke(fs):
    """Smoke test: real grid shape (B=512, H=21, W=33) completes without OOM."""
    B, H, W = 512, 21, 33
    res = 0.05
    rays = _grid_rays_regular(B, H, W, res)
    raibert = torch.zeros(B, 2, 3)

    sel, cost = fs.select_foothold_by_cost(
        raibert_xy_w=raibert, ray_hits_w=rays,
        window_m=0.10, alpha=2.0, beta=1.0, gamma=5.0, delta=1.0,
        obstacle_height_threshold=0.15,
        grid_shape=(H, W), grid_resolution=res,
    )
    assert sel.shape == (B, 2, 3), f"Unexpected shape: {sel.shape}"
    assert cost.shape == (B, 2), f"Unexpected cost shape: {cost.shape}"
    # All selected z should be 0 (flat terrain)
    assert torch.all(sel[..., 2] == 0.0)


# ---------------------------------------------------------------------------
# Task 2: hard-filter masks
# ---------------------------------------------------------------------------

SOLE_Z_OFFSET = -0.035409145057201385


def test_reachability_zero_yaw(fs):
    body_pos = torch.tensor([[0.0, 0.0, 0.80]])
    body_yaw = torch.tensor([0.0])
    foot_body_z = 0.0 - SOLE_Z_OFFSET
    candidates = torch.tensor([[[[0.2, 0.15, foot_body_z]]]])  # (1, 1, 1, 3) -> expand F,K
    candidates = candidates.expand(1, 2, 1, 3).clone()

    mask = fs.reachability_mask(candidates, body_pos, body_yaw)
    assert mask.shape == (1, 2, 1)
    assert mask[0, 0, 0].item() is True
    assert mask[0, 1, 0].item() is False


def test_reachability_yawed_robot(fs):
    body_pos = torch.tensor([[0.0, 0.0, 0.80]])
    body_yaw = torch.tensor([torch.pi / 2])
    foot_body_z = 0.0 - SOLE_Z_OFFSET
    # Body +x is world +y. Place left-foot candidate at body (0.2, 0.15, z).
    candidates = torch.tensor([[[[ -0.15, 0.2, foot_body_z]]]])
    candidates = candidates.expand(1, 2, 1, 3).clone()

    mask = fs.reachability_mask(candidates, body_pos, body_yaw)
    assert mask[0, 0, 0].item() is True
    assert mask[0, 1, 0].item() is False


def test_reachability_ignores_extreme_body_z(fs):
    """Reachability is xy-only; extreme foot-body z must not reject a valid xy cell."""
    body_pos = torch.tensor([[0.0, 0.0, 0.80]])
    body_yaw = torch.tensor([0.0])
    candidates = torch.tensor([[[[0.15, 0.15, -1.20]]]])
    candidates = candidates.expand(1, 2, 1, 3).clone()
    mask = fs.reachability_mask(candidates, body_pos, body_yaw)
    assert mask[0, 0, 0].item() is True


def test_reachability_body_frame_negative(fs):
    body_pos = torch.tensor([[0.0, 0.0, 0.80]])
    foot_body_z = 0.02
    candidates = torch.tensor([[[[0.20, 0.12, foot_body_z]]]])
    candidates = candidates.expand(1, 2, 1, 3).clone()
    assert fs.reachability_mask(candidates, body_pos, torch.tensor([0.0]))[0, 0, 0].item() is True
    assert fs.reachability_mask(candidates, body_pos, torch.tensor([math.pi]))[0, 0, 0].item() is False


def test_reachability_landing_frame_vs_current_pelvis(fs):
    """Long-horizon Raibert priors must be checked in the landing pelvis frame."""
    foot_body_z = 0.0 - SOLE_Z_OFFSET
    candidates = torch.zeros(1, 2, 1, 3)
    # Right-foot candidate under hip at landing x=0.6 (world).
    candidates[0, 1, 0] = torch.tensor([0.6, -0.12, foot_body_z])

    current_pelvis = torch.tensor([[0.0, 0.0, 0.80]])
    landing_pelvis = torch.tensor([[[0.0, 0.0, 0.80], [0.6, 0.0, 0.80]]])

    mask_current = fs.reachability_mask(candidates, current_pelvis, torch.zeros(1))
    mask_landing = fs.reachability_mask(candidates, landing_pelvis, torch.zeros(1, 2))
    assert mask_current[0, 1, 0].item() is False
    assert mask_landing[0, 1, 0].item() is True


def test_step_height_flat_stance(fs):
    stance_z = torch.zeros(1, 1)
    foot_z_values = [-0.3, -0.1, 0.0, 0.1, 0.3]
    terrain_z = torch.tensor(foot_z_values)
    foot_body_z = terrain_z - SOLE_Z_OFFSET
    candidates = torch.stack(
        [torch.tensor([0.0, 0.0, z]) for z in foot_body_z],
        dim=0,
    ).view(1, 1, 5, 3)

    mask = fs.step_height_mask(candidates, stance_z, max_dz=0.20)
    expected = torch.tensor([False, True, True, True, False])
    assert torch.equal(mask[0, 0], expected)


def test_step_height_foot_body_at_flat_stance(fs):
    """Foot-body z = -sole_z_offset ↔ terrain z = 0 at stance → valid."""
    stance_z = torch.zeros(1, 1)
    foot_body_z = -SOLE_Z_OFFSET
    candidates = torch.tensor([[[[0.0, 0.0, foot_body_z]]]])
    mask = fs.step_height_mask(candidates, stance_z)
    assert mask[0, 0, 0].item() is True


def test_roughness_cap(fs):
    dz_omega = torch.tensor([[[0.02, 0.08, 0.15]]])
    mask = fs.roughness_cap_mask(dz_omega, max_dz_omega=0.10)
    expected = torch.tensor([[[True, True, False]]])
    assert torch.equal(mask, expected)


def test_masks_composable_shape(fs):
    B, F, K = 2, 2, 13
    candidates = torch.randn(B, F, K, 3)
    body_pos = torch.randn(B, 3)
    body_yaw = torch.randn(B)
    dz_omega = torch.rand(B, F, K)
    stance_z = torch.randn(B, F)

    m_reach = fs.reachability_mask(candidates, body_pos, body_yaw)
    m_step = fs.step_height_mask(candidates, stance_z)
    m_rough = fs.roughness_cap_mask(dz_omega)

    assert m_reach.shape == (B, F, K)
    assert m_step.shape == (B, F, K)
    assert m_rough.shape == (B, F, K)
    combined = m_reach & m_step & m_rough
    assert combined.shape == (B, F, K)


def test_empty_window_falls_back_to_nearest_ray(fs):
    """Raibert outside scanner range must not silently pick flat index 0."""
    rays = _grid_rays((0.0, 0.0), half_size=0.5, res=0.05)
    raibert = torch.zeros(1, 2, 3)
    raibert[..., 0] = 10.0
    raibert[..., 1] = 10.0

    sel, _ = fs.select_foothold_by_cost(
        raibert_xy_w=raibert,
        ray_hits_w=rays,
        window_m=0.10,
        alpha=2.0,
        beta=1.0,
        gamma=5.0,
        delta=1.0,
        obstacle_height_threshold=0.15,
    )
    diff = rays[0, :, :2] - torch.tensor([10.0, 10.0])
    nearest = diff.pow(2).sum(dim=-1).argmin()
    assert nearest.item() != 0
    assert torch.allclose(sel[0, 0, :2], rays[0, nearest, :2], atol=1e-5)
    assert torch.allclose(sel[0, 1, :2], rays[0, nearest, :2], atol=1e-5)


# ---------------------------------------------------------------------------
# Task 4: select_foothold_v2
# ---------------------------------------------------------------------------


def _v2_rays(h: int = 11, w: int = 11, res: float = 0.05, z_fn=None, *, yaw: float = 0.0):
    """Row-major scanner hits; when ``yaw != 0`` cells are laid out in the rotated grid frame."""
    center_row = (h - 1) * 0.5
    center_col = (w - 1) * 0.5
    rows = torch.arange(h, dtype=torch.float32) - center_row
    cols = torch.arange(w, dtype=torch.float32) - center_col
    drow, dcol = torch.meshgrid(rows, cols, indexing="ij")
    local_x = dcol * res
    local_y = drow * res
    c, s = math.cos(yaw), math.sin(yaw)
    gx = c * local_x - s * local_y
    gy = s * local_x + c * local_y
    if z_fn is None:
        z = torch.zeros_like(gx)
    else:
        z = z_fn(gx, gy)
        if not isinstance(z, torch.Tensor):
            z = torch.full_like(gx, float(z))
    flat = torch.stack([gx, gy, z], dim=-1).reshape(1, h * w, 3)
    return flat, (h, w)


def _v2_defaults(grid_shape, resolution=0.05):
    return dict(
        body_pos_w=torch.tensor([[0.0, 0.0, 0.80]]),
        body_yaw=torch.zeros(1),
        target_yaw=torch.zeros(1, 2),
        stance_terrain_z_w=torch.zeros(1, 2),
        grid_shape=grid_shape,
        grid_resolution=resolution,
        grid_center_w=torch.zeros(1, 2),
        grid_yaw=torch.zeros(1),
    )


def test_v2_selected_z_applies_sole_offset(fs):
    rays, shp = _v2_rays()
    raibert = torch.tensor([[[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]]])
    out = fs.select_foothold_v2(raibert, rays, **_v2_defaults(shp))
    assert out["selected_xyz_w"][0, 0, 2].item() == pytest.approx(-SOLE_Z_OFFSET, abs=1e-4)
    assert not out["used_fallback"].any()


def test_v2_score_candidates_matches_select(fs):
    fc = _load_package_module("foothold_candidates")
    rays, shp = _v2_rays(h=21, w=21)
    raibert = torch.tensor([[[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]]])
    kw = _v2_defaults(shp)
    candidates, mask = fc.generate_candidates(
        raibert,
        rays,
        kw["grid_center_w"],
        kw["grid_yaw"],
        grid_shape=kw["grid_shape"],
        grid_resolution=kw["grid_resolution"],
        half_width_cells=2,
    )
    out_score = fs.score_foothold_candidates_v2(
        raibert,
        candidates,
        mask,
        rays,
        return_mask_debug=True,
        **kw,
    )
    out_select = fs.select_foothold_v2(raibert, rays, return_mask_debug=True, **kw)

    for key in ("selected_xyz_w", "foothold_score", "per_cost_debug"):
        assert torch.allclose(out_score[key], out_select[key])
    for key in ("valid_count", "used_fallback"):
        assert torch.equal(out_score[key], out_select[key])
    for key in out_select["mask_debug"]:
        assert torch.equal(out_score["mask_debug"][key], out_select["mask_debug"][key])


def test_v2_terrain_window_flat_no_fallback(fs):
    fc = _load_package_module("foothold_candidates")
    rays, shp = _v2_rays(h=21, w=21)
    raibert = torch.tensor([[[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]]])
    kw = _v2_defaults(shp)
    candidates, mask = fc.generate_terrain_window_candidates(
        raibert,
        rays,
        kw["grid_center_w"],
        kw["grid_yaw"],
        grid_shape=kw["grid_shape"],
        grid_resolution=kw["grid_resolution"],
    )
    out = fs.score_foothold_candidates_v2(raibert, candidates, mask, rays, **kw)
    assert candidates.shape == (1, 2, 81, 3)
    assert mask.all()
    assert not out["used_fallback"].any()
    assert out["foothold_score"].min().item() > 0.9


def test_v2_terrain_window_step_edge_selects_tread_not_edge(fs):
    fc = _load_package_module("foothold_candidates")

    def step(x, y):
        return (x >= 0.0).float() * 0.15

    rays, shp = _v2_rays(h=21, w=21, z_fn=step)
    raibert = torch.tensor([[[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]]])
    kw = _v2_defaults(shp)
    candidates, mask = fc.generate_terrain_window_candidates(
        raibert,
        rays,
        kw["grid_center_w"],
        kw["grid_yaw"],
        grid_shape=kw["grid_shape"],
        grid_resolution=kw["grid_resolution"],
    )
    out = fs.score_foothold_candidates_v2(
        raibert,
        candidates,
        mask,
        rays,
        w_terrain=2.0,
        w_nominal=0.0,
        w_reach=1.0,
        w_height=5.0,
        w_edge=1.0,
        w_slope=0.0,
        reach_x_range=(-0.25, 0.35),
        **kw,
    )
    assert not out["used_fallback"][0, 0].item()
    assert out["selected_xyz_w"][0, 0, 0].item() < -0.06
    assert out["selected_xyz_w"][0, 0, 2].item() == pytest.approx(-SOLE_Z_OFFSET, abs=1e-4)


def test_v2_score_zero_on_fallback(fs):
    rays, shp = _v2_rays()
    raibert = torch.tensor([[[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]]])
    out = fs.select_foothold_v2(
        raibert,
        rays,
        reach_x_range=(10.0, 20.0),
        reach_y_range_left=(10.0, 20.0),
        reach_y_range_right=(10.0, 20.0),
        **_v2_defaults(shp),
    )
    assert out["used_fallback"].all()
    assert torch.all(out["foothold_score"] == 0.0)
    assert torch.all(out["per_cost_debug"] == 0.0)


def test_v2_prefers_flat_over_edge(fs):
    def step(x, y):
        # Step at x=0.05 so flat-side candidates sit fully on z=0, raised side on z=0.15.
        return (x >= 0.05).float() * 0.15

    rays, shp = _v2_rays(h=21, w=21, z_fn=step)
    raibert = torch.tensor([[[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]]])
    out = fs.select_foothold_v2(
        raibert,
        rays,
        w_terrain=2.0,
        w_nominal=0.0,
        w_reach=0.0,
        w_height=5.0,
        w_edge=1.0,
        w_slope=0.0,
        **_v2_defaults(shp),
    )
    sel_x = out["selected_xyz_w"][0, 0, 0].item()
    assert sel_x < -0.02, f"expected flat-side cell (x<0), got x={sel_x:.3f}"
    assert not out["used_fallback"][0, 0].item()


def test_v2_yaw_rotation_invariance(fs):
    """Selected foothold body-frame xy must match when yaw and prior rotate together."""
    def body_xy_to_world(xy_b, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return c * xy_b[0] - s * xy_b[1], s * xy_b[0] + c * xy_b[1]

    def world_xy_to_body(xy_w, yaw):
        c, s = math.cos(-yaw), math.sin(-yaw)
        return c * xy_w[0] - s * xy_w[1], s * xy_w[0] + c * xy_w[1]

    prior_left_b = (0.20, 0.12)
    prior_right_b = (0.20, -0.12)
    results_b = []
    for yaw in (0.0, math.pi / 2):
        rays, shp = _v2_rays(h=21, w=21, yaw=yaw)
        lx, ly = body_xy_to_world(prior_left_b, yaw)
        rx, ry = body_xy_to_world(prior_right_b, yaw)
        raibert = torch.tensor([[[lx, ly, 0.0], [rx, ry, 0.0]]])
        kw = _v2_defaults(shp)
        kw["body_yaw"] = torch.tensor([yaw])
        kw["grid_yaw"] = torch.tensor([yaw])
        out = fs.select_foothold_v2(raibert, rays, **kw)
        sel_w = out["selected_xyz_w"][0, 0, :2].tolist()
        results_b.append(world_xy_to_body(sel_w, yaw))

    assert results_b[0][0] == pytest.approx(results_b[1][0], abs=0.06)
    assert results_b[0][1] == pytest.approx(results_b[1][1], abs=0.06)


def test_v2_hard_filter_rejects_cross_leg(fs):
    rays, shp = _v2_rays()
    # Raibert prior on the right side of body — left foot 5×5 window is all cross-leg.
    raibert = torch.tensor([[[0.20, -0.15, 0.0], [0.20, -0.15, 0.0]]])
    out = fs.select_foothold_v2(raibert, rays, **_v2_defaults(shp))
    assert out["used_fallback"][0, 0].item() is True
    assert out["valid_count"][0, 0].item() == 0


def test_v2_grid_contract_required(fs):
    rays, shp = _v2_rays()
    raibert = torch.zeros(1, 2, 3)
    kw = _v2_defaults(shp)
    kw["grid_yaw"] = None
    with pytest.raises(ValueError, match="grid_center_w and grid_yaw"):
        fs.select_foothold_v2(raibert, rays, **kw)


def test_v2_output_shapes(fs):
    rays, shp = _v2_rays()
    raibert = torch.zeros(4, 2, 3)
    rays = rays.expand(4, -1, -1).contiguous()
    kw = _v2_defaults(shp)
    kw["body_pos_w"] = kw["body_pos_w"].expand(4, -1)
    kw["body_yaw"] = kw["body_yaw"].expand(4)
    kw["target_yaw"] = kw["target_yaw"].expand(4, -1)
    kw["stance_terrain_z_w"] = kw["stance_terrain_z_w"].expand(4, -1)
    kw["grid_center_w"] = kw["grid_center_w"].expand(4, -1)
    kw["grid_yaw"] = kw["grid_yaw"].expand(4)
    out = fs.select_foothold_v2(raibert, rays, **kw)
    assert out["selected_xyz_w"].shape == (4, 2, 3)
    assert out["foothold_score"].shape == (4, 2)
    assert out["valid_count"].shape == (4, 2)
    assert out["used_fallback"].shape == (4, 2)
    assert out["per_cost_debug"].shape == (4, 2, 6)


def test_v2_mask_debug_returned_when_requested(fs):
    rays, shp = _v2_rays()
    raibert = torch.zeros(1, 2, 3)
    out = fs.select_foothold_v2(raibert, rays, return_mask_debug=True, **_v2_defaults(shp))
    md = out["mask_debug"]
    assert set(md.keys()) == {
        "reach_valid_count",
        "step_height_valid_count",
        "roughness_valid_count",
        "in_bounds_valid_count",
        "combined_valid_count",
    }


def test_v2_flat_mask_counts_match_five_by_five_reach_band(fs):
    """5×5 grid corners exceed y-band → reach < 25; longer horizon → more in_bounds loss."""
    rays, shp = _v2_rays(h=21, w=33, res=0.05)
    landing_pos = torch.tensor([[[0.30, 0.0, 0.80], [0.60, 0.0, 0.80]]])
    landing_yaw = torch.zeros(1, 2)
    raibert = torch.tensor([[[0.30, 0.12, 0.0], [0.60, -0.12, 0.0]]])
    kw = _v2_defaults(shp, 0.05)
    kw["body_pos_w"] = landing_pos
    kw["body_yaw"] = landing_yaw
    kw["target_yaw"] = landing_yaw
    kw["stance_terrain_z_w"] = torch.zeros(1, 2)
    out = fs.select_foothold_v2(raibert, rays, return_mask_debug=True, **kw)
    md = out["mask_debug"]
    # Left hip y≈+0.12: bottom grid row drops below y_range_left min 0.06 → 15/25 reach.
    assert md["reach_valid_count"][0, 0].item() == 15
    # Both feet lose the same 10 cells to y-band clipping; combined equals reach on flat z=0.
    assert md["reach_valid_count"][0, 1].item() == 15
    assert md["in_bounds_valid_count"][0].tolist() == [25, 25]
    assert out["valid_count"][0].tolist() == [15, 15]
    assert out["used_fallback"][0].tolist() == [False, False]


def test_foothold_quality_at_centers_flat(fs):
    rays, shp = _v2_rays(h=21, w=21)
    centers = torch.zeros(1, 2, 3)
    centers[..., 2] = -SOLE_Z_OFFSET
    yaws = torch.zeros(1, 2)
    scores = fs.foothold_quality_at_centers(
        centers,
        yaws,
        rays,
        grid_shape=shp,
        grid_resolution=0.05,
        grid_center_w=torch.zeros(1, 2),
        grid_yaw=torch.zeros(1),
    )
    assert scores.shape == (1, 2)
    assert scores.min().item() > 0.9
