"""Functions to generate height fields for different terrains."""

from __future__ import annotations

import numpy as np
import scipy.interpolate as interpolate
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import loco_hf_terrains_cfg

from random import randint


def _make_box(size: tuple[float, float, float], pos: tuple[float, float, float]) -> trimesh.Trimesh:
    return trimesh.creation.box(size, trimesh.transformations.translation_matrix(pos))


def single_step_terrain(
    difficulty: float, cfg: loco_hf_terrains_cfg.MeshSingleStepTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a single up-step or down-step ledge with a fixed spawn origin."""

    del difficulty
    step_height = abs(float(cfg.step_height))
    step_x = float(np.clip(cfg.step_x, 0.5, cfg.size[0] - 0.5))
    center_y = 0.5 * cfg.size[1]
    floor_thickness = 0.04
    meshes_list = []

    if cfg.direction == "up":
        lower_length = step_x
        upper_length = cfg.size[0] - step_x
        meshes_list.append(
            _make_box(
                (lower_length, cfg.size[1], floor_thickness),
                (0.5 * lower_length, center_y, -0.5 * floor_thickness),
            )
        )
        meshes_list.append(
            _make_box(
                (upper_length, cfg.size[1], step_height + floor_thickness),
                (step_x + 0.5 * upper_length, center_y, 0.5 * (step_height - floor_thickness)),
            )
        )
        spawn_z = 0.0
    elif cfg.direction == "down":
        upper_length = step_x
        lower_length = cfg.size[0] - step_x
        meshes_list.append(
            _make_box(
                (upper_length, cfg.size[1], step_height + floor_thickness),
                (0.5 * upper_length, center_y, 0.5 * (step_height - floor_thickness)),
            )
        )
        meshes_list.append(
            _make_box(
                (lower_length, cfg.size[1], floor_thickness),
                (step_x + 0.5 * lower_length, center_y, -0.5 * floor_thickness),
            )
        )
        spawn_z = step_height
    else:
        raise ValueError(f"Unsupported single-step direction: {cfg.direction}")

    origin = np.array([float(cfg.spawn_x), center_y, spawn_z])
    return meshes_list, origin


def random_blocks_terrain(
    difficulty: float, cfg: loco_hf_terrains_cfg.MeshRandomBlocksTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a deterministic field of small square blocks with +/- height offsets."""

    difficulty_key = int(round(float(difficulty) * 1_000_000.0))
    if cfg.seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng((int(cfg.seed) * 1_000_003 + difficulty_key) % (2**32))

    block_size = max(float(cfg.block_size), 0.05)
    min_height, max_height = cfg.height_range
    bottom_z = min(-0.08, float(min_height) - 0.04)
    center_y = 0.5 * cfg.size[1]
    meshes_list = []

    num_x = int(np.ceil(cfg.size[0] / block_size))
    num_y = int(np.ceil(cfg.size[1] / block_size))
    height_values = np.arange(float(min_height), float(max_height) + 0.0001, 0.01)

    for ix in range(num_x):
        x0 = ix * block_size
        x1 = min(cfg.size[0], x0 + block_size)
        if x1 <= x0:
            continue
        for iy in range(num_y):
            y0 = iy * block_size
            y1 = min(cfg.size[1], y0 + block_size)
            if y1 <= y0:
                continue

            top_z = 0.0 if x1 <= cfg.flat_start_length else float(rng.choice(height_values))
            size = (x1 - x0, y1 - y0, top_z - bottom_z)
            pos = (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (top_z + bottom_z))
            meshes_list.append(_make_box(size, pos))

    origin = np.array([float(cfg.spawn_x), center_y, 0.0])
    return meshes_list, origin


def solid_pyramid_stairs_terrain(
    difficulty: float, cfg: loco_hf_terrains_cfg.MeshSolidPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate solid pyramid stairs with a flat approach or top-platform spawn."""

    del difficulty
    center_x = 0.5 * cfg.size[0]
    center_y = 0.5 * cfg.size[1]
    terrain_size = (
        cfg.size[0] - 2.0 * cfg.border_width,
        cfg.size[1] - 2.0 * cfg.border_width,
    )
    platform_width = min(cfg.platform_width, terrain_size[0], terrain_size[1])
    step_width = float(cfg.step_width)
    step_height = float(cfg.step_height)
    num_steps_x = (terrain_size[0] - platform_width) // (2.0 * step_width)
    num_steps_y = (terrain_size[1] - platform_width) // (2.0 * step_width)
    num_steps = max(1, int(min(num_steps_x, num_steps_y)))

    meshes_list = [make_plane(cfg.size, 0.0, center_zero=False)]

    for step_id in range(num_steps):
        outer_x = terrain_size[0] - 2.0 * step_id * step_width
        outer_y = terrain_size[1] - 2.0 * step_id * step_width
        inner_x = max(platform_width, outer_x - 2.0 * step_width)
        inner_y = max(platform_width, outer_y - 2.0 * step_width)
        top_z = (step_id + 1) * step_height
        box_z = 0.5 * top_z
        y_offset = 0.5 * (outer_y - step_width)
        x_offset = 0.5 * (outer_x - step_width)

        meshes_list.append(
            _make_box((outer_x, step_width, top_z), (center_x, center_y + y_offset, box_z))
        )
        meshes_list.append(
            _make_box((outer_x, step_width, top_z), (center_x, center_y - y_offset, box_z))
        )
        meshes_list.append(
            _make_box((step_width, inner_y, top_z), (center_x + x_offset, center_y, box_z))
        )
        meshes_list.append(
            _make_box((step_width, inner_y, top_z), (center_x - x_offset, center_y, box_z))
        )

        if inner_x <= platform_width and inner_y <= platform_width:
            break

    top_z = (num_steps + 1) * step_height
    meshes_list.append(
        _make_box((platform_width, platform_width, top_z), (center_x, center_y, 0.5 * top_z))
    )

    if cfg.origin_mode == "front_base":
        origin = np.array([max(0.5, 0.5 * cfg.border_width), center_y, 0.0])
    elif cfg.origin_mode == "top_platform":
        origin = np.array([center_x, center_y, top_z])
    else:
        raise ValueError(f"Unsupported pyramid origin_mode: {cfg.origin_mode}")

    return meshes_list, origin


def hollow_stairs_terrain(
    difficulty: float, cfg: loco_hf_terrains_cfg.MeshHollowStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a hollow pyramid-stair terrain from thin treads and corner posts."""

    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    tread_thickness = min(cfg.tread_thickness, step_height)

    center_x = 0.5 * cfg.size[0]
    center_y = 0.5 * cfg.size[1]
    terrain_size = (
        cfg.size[0] - 2.0 * cfg.border_width,
        cfg.size[1] - 2.0 * cfg.border_width,
    )
    platform_width = min(cfg.platform_width, terrain_size[0], terrain_size[1])
    step_width = cfg.step_width
    if cfg.num_steps is None:
        num_steps_x = (terrain_size[0] - platform_width) // (2.0 * step_width)
        num_steps_y = (terrain_size[1] - platform_width) // (2.0 * step_width)
        num_steps = max(1, int(min(num_steps_x, num_steps_y)))
    else:
        num_steps = max(1, cfg.num_steps)
    support_post_width = min(cfg.support_post_width, step_width, 0.5 * platform_width)
    support_beam_width = min(cfg.support_beam_width, step_width, 0.5 * platform_width)

    meshes_list = [make_plane(cfg.size, 0.0, center_zero=False)]

    def add_plate_with_posts(
        pos: tuple[float, float, float],
        size_xy: tuple[float, float],
        top_z: float,
        extra_beam_z: float | None = None,
    ) -> None:
        x_size, y_size = size_xy
        if x_size <= 0.0 or y_size <= 0.0:
            return
        tread_z = top_z - 0.5 * tread_thickness
        meshes_list.append(_make_box((x_size, y_size, tread_thickness), (pos[0], pos[1], tread_z)))

        post_height = max(top_z - tread_thickness, tread_thickness)
        post_z = 0.5 * post_height
        beam_center_z = min(cfg.support_beam_height_above_previous_step, max(0.5 * support_beam_width, post_height - 0.5 * support_beam_width))
        post_x_offset = 0.5 * x_size - 0.5 * support_post_width
        post_y_offset = 0.5 * y_size - 0.5 * support_post_width
        for x_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                meshes_list.append(
                    _make_box(
                        (support_post_width, support_post_width, post_height),
                        (
                            pos[0] + x_sign * post_x_offset,
                            pos[1] + y_sign * post_y_offset,
                            post_z,
                        ),
                    )
                )

        beam_x_length = max(support_beam_width, x_size - support_post_width)
        beam_y_length = max(support_beam_width, y_size - support_post_width)
        for y_sign in (-1.0, 1.0):
            meshes_list.append(
                _make_box(
                    (beam_x_length, support_beam_width, support_beam_width),
                    (pos[0], pos[1] + y_sign * post_y_offset, beam_center_z),
                )
            )
        for x_sign in (-1.0, 1.0):
            meshes_list.append(
                _make_box(
                    (support_beam_width, beam_y_length, support_beam_width),
                    (pos[0] + x_sign * post_x_offset, pos[1], beam_center_z),
                )
            )

        # Add an extra ring of beams, bottom 4.5cm above the previous step's top surface
        if extra_beam_z is not None:
            extra_beam_z = min(extra_beam_z, post_height - 0.5 * support_beam_width)
            for y_sign in (-1.0, 1.0):
                meshes_list.append(
                    _make_box(
                        (beam_x_length, support_beam_width, support_beam_width),
                        (pos[0], pos[1] + y_sign * post_y_offset, extra_beam_z),
                    )
                )
            for x_sign in (-1.0, 1.0):
                meshes_list.append(
                    _make_box(
                        (support_beam_width, beam_y_length, support_beam_width),
                        (pos[0] + x_sign * post_x_offset, pos[1], extra_beam_z),
                    )
                )

    for step_id in range(num_steps):
        outer_x = terrain_size[0] - 2.0 * step_id * step_width
        outer_y = terrain_size[1] - 2.0 * step_id * step_width
        inner_x = max(platform_width, outer_x - 2.0 * step_width)
        inner_y = max(platform_width, outer_y - 2.0 * step_width)
        top_z = (step_id + 1) * step_height
        # extra beam: lowest point 4.5cm above the previous step's top surface
        extra_beam_z = step_id * step_height + 0.045 + 0.5 * support_beam_width
        y_offset = 0.5 * (outer_y - step_width)
        x_offset = 0.5 * (outer_x - step_width)

        add_plate_with_posts(
            (center_x, center_y + y_offset, top_z),
            (outer_x, step_width),
            top_z,
            extra_beam_z,
        )
        add_plate_with_posts(
            (center_x, center_y - y_offset, top_z),
            (outer_x, step_width),
            top_z,
            extra_beam_z,
        )
        add_plate_with_posts(
            (center_x + x_offset, center_y, top_z),
            (step_width, inner_y),
            top_z,
            extra_beam_z,
        )
        add_plate_with_posts(
            (center_x - x_offset, center_y, top_z),
            (step_width, inner_y),
            top_z,
            extra_beam_z,
        )

        if inner_x <= platform_width and inner_y <= platform_width:
            break

    add_plate_with_posts(
        (center_x, center_y, (num_steps + 1) * step_height),
        (platform_width, platform_width),
        (num_steps + 1) * step_height,
        extra_beam_z=num_steps * step_height + 0.045 + 0.5 * support_beam_width,
    )

    if cfg.origin_mode == "front_base":
        origin_x = center_x - 0.5 * terrain_size[0] - 0.5 * cfg.border_width
        origin = np.array([max(0.5, origin_x), center_y, 0.0])
    else:
        origin = np.array([center_x, center_y, (num_steps + 1) * step_height])
    return meshes_list, origin

@height_field_to_mesh
def stones_bridge_terrain(difficulty: float, cfg: loco_hf_terrains_cfg.HfStonesBridgeTerrainCfg) -> np.array:
    """Generate a terrain with stones bridge pattern.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_length = cfg.stone_length_range[1] - difficulty * (cfg.stone_length_range[1] - cfg.stone_length_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
            cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )
    stone_lateral_distance = cfg.stone_lateral_distance_range[0] + difficulty * (
        cfg.stone_lateral_distance_range[1] - cfg.stone_lateral_distance_range[0]
    )

    # switch parameters to discrete units
    # --terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # --stones
    stone_distance = int(stone_distance / cfg.horizontal_scale)
    stone_lateral_distance = int(stone_lateral_distance / cfg.horizontal_scale)
    stone_width = int(stone_width / cfg.horizontal_scale)
    stone_length = int(stone_length / cfg.horizontal_scale)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
    # --holes
    holes_depth = int(cfg.holes_depth / cfg.vertical_scale)
    # -- platform
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    # create range of heights
    stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)

    # create a terrain with a flat platform at one side
    hf_raw = np.full((width_pixels, length_pixels), holes_depth)

    # add the stones
    start_x = stone_distance
    while start_x < width_pixels:
        # ensure that stones stops along x-axis
        stop_x = min(width_pixels, start_x + stone_width)
        # randomly sample x-position
        start_y = (length_pixels - stone_length) // 2 + np.random.choice([-stone_lateral_distance, stone_lateral_distance])
        stop_y = start_y + stone_length
        hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
        # update y-position
        start_x = stop_x + stone_distance
    start_y = stone_distance
    while start_y < length_pixels:
        # ensure that stones stops along y-axis
        stop_y = min(length_pixels, start_y + stone_width)
        # randomly sample x-position
        start_x = (width_pixels - stone_length) // 2 + np.random.choice([-stone_lateral_distance, stone_lateral_distance])
        stop_x = start_x + stone_length
        hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
        # update y-position
        start_y = stop_y + stone_distance

    # add the platform in the center
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def double_column_stakes_terrain(
    difficulty: float, cfg: loco_hf_terrains_cfg.HfDoubleColumnStakesTerrainCfg
) -> np.ndarray:
    """Generate a double-column stake heightfield extending along x/y directions."""

    # Interpolate parameters by difficulty
    stake_side = cfg.stake_side_range[1] - difficulty * (
        cfg.stake_side_range[1] - cfg.stake_side_range[0]
    )
    stake_gap = cfg.stake_gap_range[0] + difficulty * (
        cfg.stake_gap_range[1] - cfg.stake_gap_range[0]
    )
    column_gap = cfg.column_gap_range[0] + difficulty * (
        cfg.column_gap_range[1] - cfg.column_gap_range[0]
    )

    # Discretized grid parameters
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    stake_side_px = max(1, int(stake_side / cfg.horizontal_scale))
    stake_gap_px = max(0, int(stake_gap / cfg.horizontal_scale))
    column_gap_px = max(0, int(column_gap / cfg.horizontal_scale))
    column_jitter_px = max(0, int(cfg.column_jitter / cfg.horizontal_scale))

    stake_height_max_px = max(0, int(cfg.stake_height_max / cfg.vertical_scale))
    holes_depth_px = int(cfg.holes_depth / cfg.vertical_scale)

    platform_width_px = max(1, int(cfg.platform_width / cfg.horizontal_scale))

    hf_raw = np.full((width_pixels, length_pixels), holes_depth_px, dtype=float)
    half_lower = stake_side_px // 2
    half_upper = stake_side_px - half_lower
    center_offset_px = stake_side_px + column_gap_px

    center_x = width_pixels // 2
    center_y = length_pixels // 2

    rng = np.random.default_rng()
    stake_height_values = (
        np.arange(-stake_height_max_px, stake_height_max_px + 1)
        if stake_height_max_px > 0
        else np.array([0], dtype=int)
    )

    def paint_square(cx: int, cy: int, value: int) -> None:
        if cx < 0 or cx >= width_pixels or cy < 0 or cy >= length_pixels:
            return
        x1 = max(0, cx - half_lower)
        x2 = min(width_pixels, cx + half_upper)
        y1 = max(0, cy - half_lower)
        y2 = min(length_pixels, cy + half_upper)
        hf_raw[x1:x2, y1:y2] = value

    def place_column_pair(primary_pos: int, along_x: bool) -> None:
        if along_x:
            axis_limit_low = half_lower
            axis_limit_high = length_pixels - half_upper
            base_offset = max(center_offset_px // 2, half_lower)
            for sign in (-1, 1):
                jitter = (
                    rng.integers(-column_jitter_px, column_jitter_px + 1)
                    if column_jitter_px > 0
                    else 0
                )
                cy = int(np.clip(center_y + sign * base_offset + jitter, axis_limit_low, axis_limit_high))
                height_value = int(rng.choice(stake_height_values))
                paint_square(primary_pos, cy, height_value)
        else:
            axis_limit_low = half_lower
            axis_limit_high = width_pixels - half_upper
            base_offset = max(center_offset_px // 2, half_lower)
            for sign in (-1, 1):
                jitter = (
                    rng.integers(-column_jitter_px, column_jitter_px + 1)
                    if column_jitter_px > 0
                    else 0
                )
                cx = int(np.clip(center_x + sign * base_offset + jitter, axis_limit_low, axis_limit_high))
                height_value = int(rng.choice(stake_height_values))
                paint_square(cx, primary_pos, height_value)

    def extend_from_center(along_x: bool, direction: int) -> None:
        if along_x:
            start = center_x + direction * (half_upper + stake_gap_px + stake_side_px)
            step = (stake_gap_px + stake_side_px) * direction
            while 0 <= start < width_pixels:
                if not (half_lower <= start <= width_pixels - half_upper):
                    break
                place_column_pair(int(start), along_x=True)
                start += step
        else:
            start = center_y + direction * (half_upper + stake_gap_px + stake_side_px)
            step = (stake_gap_px + stake_side_px) * direction
            while 0 <= start < length_pixels:
                if not (half_lower <= start <= length_pixels - half_upper):
                    break
                place_column_pair(int(start), along_x=False)
                start += step

    def extend_from_edge(along_x: bool) -> None:
        start = 0
        step = stake_gap_px + stake_side_px
        while 0 <= start < width_pixels:
            place_column_pair(int(start), along_x)
            start += step


    # Extend along +x/-x
    # extend_from_center(along_x=True, direction=1)
    # extend_from_center(along_x=True, direction=-1)
    extend_from_edge(along_x=True)

    # Extend along +y/-y
    # extend_from_center(along_x=False, direction=1)
    # extend_from_center(along_x=False, direction=-1)
    extend_from_edge(along_x=False)

    # add the platform in the center
    x1 = (width_pixels - platform_width_px) // 2
    x2 = (width_pixels + platform_width_px) // 2
    y1 = (length_pixels - platform_width_px) // 2
    y2 = (length_pixels + platform_width_px) // 2
    hf_raw[x1:x2, y1:y2] = 0

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def concentric_gap_terrain(difficulty: float, cfg: loco_hf_terrains_cfg.HfConcentricGapTerrainCfg) -> np.ndarray:
    """
    Generate concentric gap terrain with a center platform.
    Gap width is difficulty-dependent and gap depth is fixed.
    """
    # Gap depth in pixels
    gap_depth = int(2.0 / cfg.vertical_scale)
    # Gap width varies with difficulty
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    gap_width = int(gap_width / cfg.horizontal_scale)
    # Ground width varies with difficulty (narrower for harder terrains)
    ground_width = cfg.ground_width_range[0] + (1.0 - difficulty) * (cfg.ground_width_range[1] - cfg.ground_width_range[0])
    ground_width = int(ground_width / cfg.horizontal_scale)
    # Ground height
    ground_height_max = int(cfg.ground_height_max / cfg.vertical_scale)
    # Terrain dimensions
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # Platform width
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    hf_raw = np.zeros((width_pixels, length_pixels))
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    is_gap = True
    while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
        if is_gap:
            # Fill gap ring
            hf_raw[start_x:stop_x, start_y:stop_y] = -gap_depth
            start_x += gap_width
            stop_x -= gap_width
            start_y += gap_width
            stop_y -= gap_width
        else:
            # Fill ground ring
            hf_raw[start_x:stop_x, start_y:stop_y] = randint(-ground_height_max, ground_height_max)
            start_x += ground_width
            stop_x -= ground_width
            start_y += ground_width
            stop_y -= ground_width
        is_gap = not is_gap
    # add the platform in the center
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def alternate_column_stakes_terrain(
    difficulty: float, cfg: loco_hf_terrains_cfg.HfDoubleColumnStakesTerrainCfg
) -> np.ndarray:
    """Generate alternating double-column stake terrain along x/y directions."""

    # Interpolate parameters by difficulty
    stake_side = cfg.stake_side_range[1] - difficulty * (
        cfg.stake_side_range[1] - cfg.stake_side_range[0]
    )
    stake_gap = cfg.stake_gap_range[0] + difficulty * (
        cfg.stake_gap_range[1] - cfg.stake_gap_range[0]
    )
    column_gap = cfg.column_gap_range[1] - difficulty * (
        cfg.column_gap_range[1] - cfg.column_gap_range[0]
    )

    # Discretized grid parameters
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    stake_side_px = max(1, int(stake_side / cfg.horizontal_scale))
    stake_gap_px = max(0, int(stake_gap / cfg.horizontal_scale))
    column_gap_px = max(0, int(column_gap / cfg.horizontal_scale))
    column_jitter_px = max(0, int(cfg.column_jitter / cfg.horizontal_scale))

    stake_height_max_px = max(0, int(cfg.stake_height_max / cfg.vertical_scale))
    holes_depth_px = int(cfg.holes_depth / cfg.vertical_scale)

    platform_width_px = max(1, int(cfg.platform_width / cfg.horizontal_scale))

    hf_raw = np.full((width_pixels, length_pixels), holes_depth_px, dtype=float)
    half_lower = stake_side_px // 2
    half_upper = stake_side_px - half_lower

    # Build a deterministic RNG for this sub-terrain when cfg.seed is provided.
    # We mix in quantized difficulty so each tile can still look different while
    # remaining reproducible across runs.
    if getattr(cfg, "seed", None) is not None:
        difficulty_key = int(round(float(difficulty) * 1_000_000.0))
        local_seed = (int(cfg.seed) * 1_000_003 + difficulty_key) % (2**32)
        rng = np.random.default_rng(local_seed)
    else:
        rng = np.random.default_rng()
    stake_height_values = (
        np.arange(-stake_height_max_px, stake_height_max_px + 1)
        if stake_height_max_px > 0
        else np.array([0], dtype=int)
    )

    def paint_square(cx: int, cy: int, value: int) -> None:
        if cx < 0 or cx >= width_pixels or cy < 0 or cy >= length_pixels:
            return
        x1 = max(0, cx - half_lower)
        x2 = min(width_pixels, cx + half_upper)
        y1 = max(0, cy - half_lower)
        y2 = min(length_pixels, cy + half_upper)
        hf_raw[x1:x2, y1:y2] = value

    def place_alternate_columns(start_pos: int, along_x: bool) -> None:
        offset = column_gap_px // 2  # Alternating offset
        step = stake_gap_px + stake_side_px
        while start_pos < (width_pixels if along_x else length_pixels):
            jitter = (
                rng.integers(-column_jitter_px, column_jitter_px + 1)
                if column_jitter_px > 0
                else 0
            )
            height_value = int(rng.choice(stake_height_values))

            if along_x:
                cx = start_pos
                cy = (length_pixels // 2) + offset + jitter
                paint_square(cx, cy, height_value)
            else:
                cy = start_pos
                cx = (width_pixels // 2) + offset + jitter
                paint_square(cx, cy, height_value)

            # Flip offset for alternating pattern
            offset = -offset
            start_pos += step

    # Place alternating columns along x and y
    place_alternate_columns(0, along_x=True)
    place_alternate_columns(0, along_x=False)

    # add the platform in the center
    x1 = (width_pixels - platform_width_px) // 2
    x2 = (width_pixels + platform_width_px) // 2
    y1 = (length_pixels - platform_width_px) // 2
    y2 = (length_pixels + platform_width_px) // 2
    hf_raw[x1:x2, y1:y2] = 0

    return np.rint(hf_raw).astype(np.int16)
