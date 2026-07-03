# Planner V2 Geometry Contract

This document is the source of truth for the footstep planner's grid-map,
height, and foot-polygon conventions. The implementation plan should point
here instead of duplicating these rules.

## Height Grid Frame

The local height grid is centered on the robot/scanner frame. The grid's local
origin is the center cell, not cell `(0, 0)`.

- `grid_center_w: (B, 2)` is the world xy of the scanner/grid center.
- `grid_yaw: (B,)` is the yaw of the height grid. For the current
  `height_scanner` this is the real robot torso/root yaw, because
  `ray_alignment="yaw"`.
- `grid_shape = (H, W)` where `H` is the number of y rows and `W` is the
  number of x columns.
- `grid_resolution` is the cell size in metres.
- `row` indexes y, `col` indexes x, and `flat = row * W + col`.
- For `size=[1.6, 1.0]` and `resolution=0.05`, `W=33`, `H=21`,
  `center_col=16`, and `center_row=10`.

World xy to grid index:

```python
xy_local = R(-grid_yaw) @ (xy_w - grid_center_w)
col = round(xy_local[..., 0] / grid_resolution + (W - 1) / 2)
row = round(xy_local[..., 1] / grid_resolution + (H - 1) / 2)
flat = row * W + col
```

Candidate-window ordering is row-major:

```python
for drow in range(-half_width_cells, half_width_cells + 1):
    for dcol in range(-half_width_cells, half_width_cells + 1):
        ...
```

## Yaw Roles

Do not reuse one variable named `root_yaw` for every frame. Planner V2 has
three different yaw roles:

- `grid_yaw`: yaw of the real height grid, used only for grid lookup.
- `body_yaw`: yaw of the body frame used for reachability and comfort costs.
- `target_yaw`: yaw of the candidate foot polygon used by `foot_patch_stats`.

When `use_phantom=True`, Raibert targets may be generated from a phantom body,
but height lookup still uses the real scanner grid. Therefore `grid_yaw` and
planner/phantom yaw must remain separate.

## Z Semantics

Keep terrain, sole, and foot-body z explicit in names.

| Name | Meaning |
|---|---|
| `terrain_z_w` | Ground surface height from the height grid |
| `sole_z_w` | Physical touchdown/contact height of the foot sole |
| `foot_body_z_w` | Foot body link target height used by DTC tracking |
| `sole_z_offset` | Signed offset such that `sole_z_w = foot_body_z_w + sole_z_offset` |

For the G1 ankle roll link, `sole_z_offset = -0.035409145057201385`.

Planner buffers store foot-body targets:

```python
candidate_foot_body_z_w = candidate_terrain_z_w - sole_z_offset
candidate_terrain_z_w = candidate_foot_body_z_w + sole_z_offset
```

Reachability and comfort costs operate on foot-body targets in body frame:

```python
candidate_foot_body_pos_b = R(-body_yaw) @ (candidate_foot_body_xyz_w - body_pos_w)
```

Step-height continuity operates on terrain/sole heights, never foot-body
heights:

```python
stance_terrain_z_w = last_contact_foot_body_z_w + sole_z_offset
```

## Foot-Polygon Sampling

The planner and BeamDojo-style reward use the same G1 foot geometry constants:

- `foot_length = 0.18`
- `foot_width = 0.065`
- `n_long = 4`
- `n_lat = 3`
- `sole_z_offset = -0.035409145057201385`

`foot_patch_stats` samples the foot polygon around a candidate center using
`target_yaw`. If any footprint samples fall outside the height grid, the
function must expose that fact instead of silently treating clamped boundary
samples as valid support:

- `sample_in_bounds: (B, M, S)`
- `footprint_in_bounds_ratio: (B, M)`

Out-of-bounds samples may be clamped for safe tensor gather, but they must be
counted as unsupported and should be available to hard filters such as
`min_footprint_in_bounds_ratio`.

## Planner/Reward Parity

Parity tests should compare the terrain samples and derived terrain statistics
that both sides genuinely share, such as `terrain_z`, `z_median`, and
`dz_omega`.

Do not blindly compare support ratios unless the test explicitly makes the
reward-side sole plane equal to the planner-side support reference. The reward
checks `abs(terrain_z - foot_sole_z)`, while the planner may check support
relative to patch statistics such as `z_median` or `z_q70`.
