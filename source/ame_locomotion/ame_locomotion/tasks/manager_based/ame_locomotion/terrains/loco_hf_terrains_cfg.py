from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from . import loco_hf_terrains


@configclass
class MeshHollowStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a hollow pyramid stair mesh terrain."""

    function = loco_hf_terrains.hollow_stairs_terrain

    border_width: float = 0.0
    """Unused ground border kept for API symmetry with mesh stair terrains."""

    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of each step (in m)."""

    step_width: float = MISSING
    """The width of each pyramid stair ring (in m)."""

    tread_depth: float = MISSING
    """Deprecated. Kept for compatibility with older configs."""

    tread_thickness: float = 0.03
    """The thickness of each tread plate (in m)."""

    stair_width: float = 0.6
    """The lateral width of the stair tread (in m)."""

    support_post_width: float = 0.05
    """The side length of each corner support post (in m)."""

    support_beam_width: float = 0.05
    """The square side length of the low horizontal support beams (in m)."""

    support_beam_height_above_previous_step: float = 0.05
    """The center height of support beams above the previous step top surface (in m)."""

    platform_width: float = 2.0
    """The center platform width (in m)."""

    top_platform_width: float = 0.6
    """Deprecated. Kept for compatibility with older configs."""

    num_steps: int | None = None
    """Optional number of hollow steps. Defaults to the official pyramid-stair step-count rule."""

    origin_mode: str = "center_platform"
    """Spawn origin mode: ``center_platform`` starts on the top platform, ``front_base`` starts on flat ground."""


@configclass
class MeshSingleStepTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a fixed single ledge used by screen-recorded play experiments."""

    function = loco_hf_terrains.single_step_terrain

    step_height: float = MISSING
    """Positive step/drop height in meters."""

    direction: str = "up"
    """``up`` starts on the lower side; ``down`` starts on the raised side."""

    step_x: float = 3.2
    """X coordinate of the vertical step edge in terrain-local coordinates."""

    spawn_x: float = 1.2
    """Robot spawn x coordinate in terrain-local coordinates."""


@configclass
class MeshRandomBlocksTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a field of fixed-size square blocks with small height offsets."""

    function = loco_hf_terrains.random_blocks_terrain

    block_size: float = 0.35
    """Side length of each square block in meters."""

    height_range: tuple[float, float] = (-0.05, 0.05)
    """Minimum and maximum block top height in meters."""

    flat_start_length: float = 1.2
    """Length of the flat approach patch before random blocks start."""

    spawn_x: float = 0.8
    """Robot spawn x coordinate in terrain-local coordinates."""

    seed: int | None = 7
    """Optional deterministic seed for repeatable play screenshots."""


@configclass
class MeshSolidPyramidStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a solid pyramid-stair terrain with configurable spawn origin."""

    function = loco_hf_terrains.solid_pyramid_stairs_terrain

    border_width: float = 1.0
    """Flat border around the pyramid in meters."""

    step_height: float = 0.10
    """Height of each stair ring in meters."""

    step_width: float = 0.30
    """Width of each stair ring in meters."""

    platform_width: float = 3.0
    """Width of the center top platform in meters."""

    origin_mode: str = "front_base"
    """``front_base`` starts before the first step; ``top_platform`` starts on the center platform."""


@configclass
class HfStonesBridgeTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a stones bridge height field terrain."""

    function = loco_hf_terrains.stones_bridge_terrain

    stone_height_max: float = MISSING
    """The maximum height of the stones (in m)."""
    stone_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the stones (in m)."""
    stone_length_range: tuple[float, float] = MISSING
    """The minimum and maximum length of the stones (in m)."""
    stone_distance_range: tuple[float, float] = MISSING
    """The minimum and maximum distance between stones (in m)."""
    stone_lateral_distance_range: tuple[float, float] = MISSING
    """The minimum and maximum lateral distance between stones (in m)."""
    holes_depth: float = -10.0
    """The depth of the holes (negative obstacles). Defaults to -10.0."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class HfConcentricGapTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a concentric gaps height field terrain."""

    function = loco_hf_terrains.concentric_gap_terrain

    gap_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the gaps (in m)."""
    ground_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the ground (in m)."""
    ground_height_max: float = MISSING
    """The maximum height of the ground (in m).""" 
    gap_depth: float = -2.0
    """The depth of the gaps (negative obstacles). Defaults to -2.0."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class HfDoubleColumnStakesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a two-column plum-blossom stakes height field terrain."""

    function = loco_hf_terrains.double_column_stakes_terrain

    stake_height_max: float = MISSING
    """The maximum height variation of the stakes (in m)."""
    stake_side_range: tuple[float, float] = MISSING
    """The minimum and maximum side length of the square stakes (in m)."""
    stake_gap_range: tuple[float, float] = MISSING
    """The minimum and maximum clear gap between successive stakes along the extension axis (in m)."""
    column_gap_range: tuple[float, float] = MISSING
    """The minimum and maximum lateral clear gap between the two stake columns (in m)."""
    column_jitter: float = 0.0
    """Maximum lateral jitter applied to each stake center (in m). Defaults to 0.0."""
    holes_depth: float = -2.0
    """The base depth around the stakes (negative obstacles). Defaults to -2.0."""
    platform_width: float = 1.0
    """Width of the central platform patch (in m). Defaults to 1.0."""


@configclass
class HfAlternateColumnStakesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a two-column plum-blossom stakes height field terrain."""

    function = loco_hf_terrains.alternate_column_stakes_terrain

    stake_height_max: float = MISSING
    """The maximum height variation of the stakes (in m)."""
    stake_side_range: tuple[float, float] = MISSING
    """The minimum and maximum side length of the square stakes (in m)."""
    stake_gap_range: tuple[float, float] = MISSING
    """The minimum and maximum clear gap between successive stakes along the extension axis (in m)."""
    column_gap_range: tuple[float, float] = MISSING
    """The minimum and maximum lateral clear gap between the two stake columns (in m)."""
    column_jitter: float = 0.0
    """Maximum lateral jitter applied to each stake center (in m). Defaults to 0.0."""
    holes_depth: float = -2.0
    """The base depth around the stakes (negative obstacles). Defaults to -2.0."""
    platform_width: float = 1.0
    """Width of the central platform patch (in m). Defaults to 1.0."""
