from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def ray_hits_s(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Get the 3D coordinates of scan points in the sensor's local coordinate 
    frame.

    This function converts the ray hit points from world coordinates to
    sensor-local coordinates. The returned coordinates represent the relative
    positions of scan points w.r.t. the sensor frame.

    Args:
        env: The environment containing the sensor.
        sensor_cfg: The SceneEntity configuration for the sensor.

    Returns:
        A flattened tensor containing the 3D coordinates of scan points in 
        sensor frame. The original shape (N, B, 3) is flattened to (N*B*3,) 
        for compatibility with other observation terms.
    """
    
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    relative_pos_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    sensor_quat = sensor.data.quat_w  # (N, 4)
    N, B, _ = relative_pos_w.shape
   
    # Handle different sensor alignments
    alignment = getattr(sensor.cfg, "ray_alignment", "base")
    if alignment == "yaw":
        sensor_quat = yaw_quat(sensor_quat)

    sensor_quat_expanded = (sensor_quat.unsqueeze(1).expand(N, B, 4).reshape(N*B, 4)).to(torch.float)
    relative_pos_w_reshaped = relative_pos_w.reshape(N*B, 3)
    sensor_coords = quat_apply_inverse(sensor_quat_expanded, relative_pos_w_reshaped)
    sensor_coords = sensor_coords.reshape(N, B, 3)

    # print("ray_hits_s:", torch.round(sensor_coords * 100) / 100)
    # print(sensor_coords.reshape(N, B*3))

    if torch.isnan(sensor_coords).any() or torch.isinf(sensor_coords).any():
        print(f"Warning: ray_hits_s contains NaN or Inf: {sensor_coords}")
        sensor_coords = torch.nan_to_num(sensor_coords)
        
    return sensor_coords.reshape(N, B*3)


def elevation_map(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, noise: bool = False) -> torch.Tensor:
    
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    relative_pos_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    sensor_quat = sensor.data.quat_w  # (N, 4)
    N, B, _ = relative_pos_w.shape
   
    # Handle different sensor alignments
    alignment = getattr(sensor.cfg, "ray_alignment", "base")
    if alignment == "yaw":
        sensor_quat = yaw_quat(sensor_quat)

    sensor_quat_expanded = (sensor_quat.unsqueeze(1).expand(N, B, 4).reshape(N*B, 4)).to(torch.float)
    relative_pos_w_reshaped = relative_pos_w.reshape(N*B, 3)
    sensor_coords = quat_apply_inverse(sensor_quat_expanded, relative_pos_w_reshaped)
    sensor_coords = sensor_coords.reshape(N, B, 3)

    if torch.isnan(sensor_coords).any() or torch.isinf(sensor_coords).any():
        # print(f"Warning: elevation_map contains NaN or Inf: {sensor_coords}")
        sensor_coords = torch.nan_to_num(sensor_coords)

    if noise:
        # Initialize buffers for shift and delayed observation
        # if getattr(env, "_elevation_map_shift", None) is None:
        #     env._elevation_map_shift = torch.zeros((env.num_envs, 2), device=env.device)
        if getattr(env, "_elevation_map_offset", None) is None or env._elevation_map_offset.shape != (N, 1):
            env._elevation_map_offset = torch.zeros((N, 1), device=env.device)
        # if getattr(env, "_last_elevation_map", None) is None or env._last_elevation_map.shape != (N, B * 3):
        #     env._last_elevation_map = torch.zeros((N, B * 3), device=env.device)

        # Resample x and y shift for reset environments (-3cm to 3cm)
        if hasattr(env, "reset_buf"):
            reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                # env._elevation_map_shift[reset_env_ids] = torch.rand((len(reset_env_ids), 2), device=env.device) * 0.06 - 0.03
                env._elevation_map_offset[reset_env_ids] = torch.rand((len(reset_env_ids), 1), device=env.device) * 0.1 - 0.05

        # Apply x, y shift
        # sensor_coords[..., :2] += env._elevation_map_shift.unsqueeze(1)

        # Add Gaussian noise to each height value
        height_noise = torch.randn_like(sensor_coords[..., 2]) * 0.03 # std dev of 3 cm
        sensor_coords[..., 2] += height_noise  
        # Add a global offset noise to simulate sensor initialization error
        offset_noise = env._elevation_map_offset
        sensor_coords[..., 2] += offset_noise
        
    # Clip height values
    sensor_coords[..., 2] = torch.clamp(sensor_coords[..., 2], min=-1.2, max=0.0)

    current_map = sensor_coords.reshape(N, B * 3)

    return current_map
    
    # if noise:
    #     # Apply data delay: update map only every 5 steps or on reset (10Hz instead of 50Hz)
    #     update_mask = (env.episode_length_buf % 5 == 0)
    #     if hasattr(env, "reset_buf"):
    #         update_mask = update_mask | env.reset_buf
    #     update_env_ids = update_mask.nonzero(as_tuple=False).squeeze(-1)
        
    #     if len(update_env_ids) > 0:
    #         env._last_elevation_map[update_env_ids] = current_map[update_env_ids]

    #     return env._last_elevation_map.clone()
    # else:
    #     return current_map


def height_samples(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, noise: bool = False) -> torch.Tensor:
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    relative_pos_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    sensor_quat = sensor.data.quat_w  # (N, 4)
    N, B, _ = relative_pos_w.shape

    # Handle different sensor alignments
    alignment = getattr(sensor.cfg, "ray_alignment", "base")
    if alignment == "yaw":
        sensor_quat = yaw_quat(sensor_quat)

    sensor_quat_expanded = (sensor_quat.unsqueeze(1).expand(N, B, 4).reshape(N*B, 4)).to(torch.float)
    relative_pos_w_reshaped = relative_pos_w.reshape(N*B, 3)
    sensor_coords = quat_apply_inverse(sensor_quat_expanded, relative_pos_w_reshaped)
    sensor_coords = sensor_coords.reshape(N, B, 3)

    if torch.isnan(sensor_coords).any() or torch.isinf(sensor_coords).any():
        sensor_coords = torch.nan_to_num(sensor_coords)

    if noise:
        if getattr(env, "_height_samples_offset", None) is None or env._height_samples_offset.shape != (N, 1):
            env._height_samples_offset = torch.zeros((N, 1), device=env.device)

        if hasattr(env, "reset_buf"):
            reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                env._height_samples_offset[reset_env_ids] = torch.rand((len(reset_env_ids), 1), device=env.device) * 0.1 - 0.05

        height_noise = torch.randn_like(sensor_coords[..., 2]) * 0.03
        sensor_coords[..., 2] += height_noise
        sensor_coords[..., 2] += env._height_samples_offset

    sensor_coords[..., 2] = torch.clamp(sensor_coords[..., 2], min=-1.2, max=0.0)

    return sensor_coords[..., 2].reshape(N, B)


def _world_to_body_yaw(points_w: torch.Tensor, robot) -> torch.Tensor:
    """Transform world-frame xyz points into the robot's body frame using yaw only.

    Translation removed (subtract root xy), rotation applied with -yaw. The z
    component is expressed as ``z_world - root_z_world`` (height of the point
    relative to the root) so the planner observation is translation-invariant.

    Args:
        points_w: ``(B, ..., 3)`` world-frame points (leading dim is batch).
        robot: Articulation with ``data.root_pos_w`` ``(B, 3)`` and
            ``data.heading_w`` ``(B,)`` (radians, world frame).
    """
    root_pos = robot.data.root_pos_w
    yaw = robot.data.heading_w
    B = points_w.shape[0]

    # broadcast helpers to align with arbitrary middle dims
    trailing_shape = points_w.shape[1:-1]                                # e.g. (N, 2)
    view_shape = (B,) + (1,) * len(trailing_shape) + (3,)
    root_view = root_pos.view(view_shape)

    rel = points_w - root_view  # (B, ..., 3)
    c = torch.cos(yaw).view((B,) + (1,) * len(trailing_shape))
    s = torch.sin(yaw).view((B,) + (1,) * len(trailing_shape))
    x_b = c * rel[..., 0] + s * rel[..., 1]
    y_b = -s * rel[..., 0] + c * rel[..., 1]
    z_b = rel[..., 2]
    return torch.stack([x_b, y_b, z_b], dim=-1)


def footstep_plan(env: ManagerBasedRLEnv, command_name: str = "footstep_plan") -> torch.Tensor:
    """Body-frame footstep plan observation.

    Reads the (world-frame) plan buffer from a :class:`FootstepPlanCommand`
    term and returns its body-frame view, flattened to ``(B, N*2*3)``.
    Body-frame transform is yaw-only and z is expressed relative to the root.
    """
    cmd = env.command_manager.get_term(command_name)
    plan_w = cmd.plan_buffer  # (B, N, 2, 3)
    plan_b = _world_to_body_yaw(plan_w, cmd.robot)
    return plan_b.flatten(start_dim=1)


def footstep_phase_info(env: ManagerBasedRLEnv, command_name: str = "footstep_plan") -> torch.Tensor:
    """Body-agnostic phase metadata per planned step: (time_left, contact_mask).

    Returns ``(B, N*2*2)`` flattened in order ``(k, foot, [time_left, contact])``.
    ``contact`` is the current phase mask: 0 for the active swing foot and 1
    for the stance foot. Use alongside :func:`footstep_plan` to give the
    policy the DTC "what / where / when" triplet for each future step.
    """
    cmd = env.command_manager.get_term(command_name)
    # (B, N, 2, 2)
    info = torch.stack([cmd.time_left_buffer, cmd.contact_target_buffer], dim=-1)
    return info.flatten(start_dim=1)


def footstep_foothold_score(
    env: ManagerBasedRLEnv, command_name: str = "footstep_plan"
) -> torch.Tensor:
    """Per planned foothold quality score. Shape ``(B, N*2)``. Values in ``[0, 1]``.

    Zero indicates the v2 selector used fallback (no valid candidate).
    """
    cmd = env.command_manager.get_term(command_name)
    return cmd.foothold_score_buffer.flatten(start_dim=1)


def footstep_swing_side(
    env: ManagerBasedRLEnv, command_name: str = "footstep_plan"
) -> torch.Tensor:
    """One-hot encoding of the current swing foot. Shape ``(B, 2)`` — ``[1,0]`` left, ``[0,1]`` right."""
    cmd = env.command_manager.get_term(command_name)
    return torch.nn.functional.one_hot(cmd.swing_foot, num_classes=2).float()


def footstep_local_heightscan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "footstep_plan",
    half_size_m: float = 0.10,
    n_per_axis: int = 5,
) -> torch.Tensor:
    """Per-planned-foothold local height patches (relative-z), shape ``(B, N*2*n²)``.

    For each (k, foot) in the planner's world-frame plan buffer, samples an
    ``n_per_axis × n_per_axis`` height grid around the planned xy and returns
    ``z_terrain - z_planned``. This is the sim/real-invariant terrain feature
    the policy actually sees — small patches drawn from the same elevation
    grid that ``grid_map_builder`` would provide on hardware.
    """
    from . import planner as planner_ops  # local import: keeps planner Isaac-Sim-free
    cmd = env.command_manager.get_term(command_name)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    ray_hits_w = sensor.data.ray_hits_w  # (B, K, 3)

    plan_w = cmd.plan_buffer  # (B, N, 2, 3)
    B, N, _, _ = plan_w.shape
    centers = plan_w.reshape(B, N * 2, 3)
    patches = planner_ops.local_heightscan_around(
        centers_w=centers,
        ray_hits_w=ray_hits_w,
        half_size_m=half_size_m,
        n_per_axis=n_per_axis,
    )  # (B, N*2, n*n)
    return patches.flatten(start_dim=1)
