from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import math

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.warp import raycast_mesh

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
    ) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def foot_clearance_reward_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    min_command_speed: float = 0.1,
) -> torch.Tensor:
    """Positive swing-foot clearance reward with command, airborne, and velocity gates."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    is_airborne = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0.0

    per_foot_reward = torch.exp(-foot_z_target_error / std) * foot_velocity_tanh * is_airborne.float()
    reward = torch.sum(per_foot_reward, dim=1)

    command_gate = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > min_command_speed
    upright_gate = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward * command_gate * upright_gate


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


def joint_coordination_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, coord_joints: list[list[str]], coord_signs: list[list[float]] = None) -> torch.Tensor:
    """Reward coordinated motion between joint pairs using relative positions and signs.

    Args:
        coord_joints: List of joint pairs to coordinate.
        coord_signs: Sign per joint in each pair, e.g. [[1.0, -1.0]].
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    if not hasattr(env, "joint_coord_joints_cache") or env.joint_coord_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_coord_joints_cache = [
            [asset.find_joints(joint_name)[0] for joint_name in joint_pair] for joint_pair in coord_joints
        ]
    
    # Default signs (positive correlation)
    if coord_signs is None:
        coord_signs = [[1.0, 1.0]] * len(coord_joints)
    
    reward = torch.zeros(env.num_envs, device=env.device)
    
    # Iterate over all joint pairs
    for i, joint_indices in enumerate(env.joint_coord_joints_cache):
        # Get joint indices for the pair
        joint1_idx = joint_indices[0][0]  # First joint's first index
        joint2_idx = joint_indices[1][0]  # Second joint's first index
        
        # Get relative joint positions (relative to default)
        joint1_rel = asset.data.joint_pos[:, joint1_idx] - asset.data.default_joint_pos[:, joint1_idx]
        joint2_rel = asset.data.joint_pos[:, joint2_idx] - asset.data.default_joint_pos[:, joint2_idx]
        
        # Apply coordination signs
        joint1_signed = joint1_rel * coord_signs[i][0]
        joint2_signed = joint2_rel * coord_signs[i][1]
        
        # Calculate coordination error
        coord_error = torch.square(joint1_signed - joint2_signed)
        reward += coord_error
    
    reward *= 1 / len(coord_joints) if len(coord_joints) > 0 else 0
    return reward


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def time_limited_position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, activation_time: float
) -> torch.Tensor:
    """Reward position tracking with tanh kernel, activated when time-to-go is below a threshold."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    time_left = command[:, 4]
    return (1 - torch.tanh(distance / std)) * (time_left < activation_time).float()


def velocity_towards_target_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fine_grained_std: float = 0.2,
    threshold_ratio: float = 0.5,
) -> torch.Tensor:
    """Reward alignment between base velocity and target direction.
    
    The reward is the cosine similarity between the base velocity vector and the vector pointing to the target.
    This reward is disabled when the fine-grained position tracking reward (assumed to be based on
    time_limited_position_command_error_tanh) exceeds a certain threshold ratio of its maximum value (1.0).
    
    Args:
        fine_grained_std: The std parameter used in the fine-grained position tracking reward.
        fine_grained_activation_time: The activation_time parameter used in the fine-grained position tracking reward.
        threshold_ratio: The ratio of the max fine-grained reward at which to disable this guidance reward.
    """
    # Get commands
    command = env.command_manager.get_command(command_name)
    target_pos_b = command[:, :3]  # (N, 3)
    time_left = command[:, 4]      # (N,)
    
    # Target direction in 2D (base frame)
    target_vec_2d = target_pos_b[:, :2]
    dist = torch.norm(target_vec_2d, dim=1)
    target_dir_norm = target_vec_2d / (dist.unsqueeze(1) + 1e-6)

    # Robot Velocity in base frame
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_b = asset.data.root_lin_vel_b[:, :2] # (N, 2)
    speed = torch.norm(vel_b, dim=1)
    vel_dir_norm = vel_b / (speed.unsqueeze(1) + 1e-6)

    # Cosine Similarity: dot product of normalized vectors
    alignment = torch.sum(target_dir_norm * vel_dir_norm, dim=1)
    
    # Reward only when moving somewhat towards target (positive alignment)
    # And scale by movement status (if not moving, direction is noise/irrelevant, reward 0)
    reward = torch.clamp(alignment, min=0.0) * (speed > 0.1).float()

    # Termination Condition check
    # Calculate what the fine-grained reward value would be
    # Formula: (1 - tanh(dist/std)) * (time < activation)
    fine_grained_val = (1 - torch.tanh(dist / fine_grained_std))
    
    # Disable if fine-grained reward is high enough (indicating we are in the "fine-tuning" zone)
    mask = fine_grained_val < threshold_ratio
    
    return reward * mask.float()


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()


# =========================================================================
# Footstep planner rewards (Phase 1 — see PLAN.md)
# =========================================================================


def footstep_swing_tracking(
    env: ManagerBasedRLEnv,
    command_name: str = "footstep_plan",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    force_threshold: float = 1.0,
    std: float = 0.08,
    apex: float = 0.10,
    vel_gate_std: float = 1.0,
    vel_gate_command_name: str = "base_velocity",
) -> torch.Tensor:
    """Per-step dense reward for tracking the desired swing-foot trajectory.

    Builds the reference foot position by interpolating between the latest
    ``last_contact_w`` and the committed footstep ``target_w`` along the
    foot's swing phase (cubic xy, parabolic z apex). Only the foot in the
    swing half of the stride contributes to the reward, and only after it is
    actually airborne; the stance foot contributes 0 so we don't double-count
    with `feet_slide`.

    reward = sum over feet of ``exp(-||foot_actual - foot_ref||² / std²) *
             is_swing_mask * is_airborne * vel_gate``  →  shape ``(B,)`` in ``[0, 2]``.

    The ``vel_gate`` is a soft ``sech²`` mask on the body-frame xy velocity
    tracking error ``||v_root_xy - v_cmd_xy||`` (world frame, magnitude only).
    It removes the "marching-in-place" exploit where a non-translating robot
    can still collect ~40% of the swing-tracking reward by lifting and
    setting the foot down near the current root position. With the gate:

      * standing still (|v|≈0 while cmd≠0) → gate≈0 → swing_tracking≈0,
        so the only forward signal left is `track_lin_vel_xy_exp`, which is
        what we want (break the saddle).
      * walking with the command → gate≈1 → full swing-tracking reward
        applies, refining the gait as before.

    Math: ``gate = sech²(|v_err| / std) = 1 - tanh²(|v_err| / std)``, which
    is the standard RBF-form membership function. Two properties that
    distinguish it from ``1 - tanh(|x|)`` were the reason for this choice:

      (1) **Zero gradient at the optimum.** ``sech²(x)`` has
          ``d/dx sech²(0) = 0``, so once the robot's xy velocity matches the
          command the velocity-gate stops contributing gradient and lets
          ``swing_err`` (the foot-tracking error) own the gradient budget —
          the policy can then refine the gait instead of being perpetually
          yanked by the velocity term. ``1 - tanh(x)`` has -1 gradient at x=0,
          which would compete with swing_err even at perfect tracking.
      (2) **Softer transition.** At ``|v_err| = std``, ``sech²=0.42`` vs
          ``1-tanh=0.24``. The wider shoulder keeps a partial-bonus signal
          alive through "half-walking" states, so a partial-commits gait
          earns a reduced but non-zero signal and the gradient always points
          toward "more forward velocity" before "more precise foot
          placement". This avoids the scaffold-too-narrow failure of
          tightening ``std`` on the exponential directly (signal vanishes
          outright for random early policies).

    Default ``vel_gate_std=1.0`` keeps the gate ≥ 0.05 for tracking errors
    up to ~0.6 m/s, matching the env's commanded-speed scale; tighten to
    ~0.5 once the base gait is reliable.

    The other anti-slide cues (``footstep_contact_phase`` AND-product and
    ``feet_air_time`` requiring actual airborne phases) already block the
    "slide both feet forward while connected to ground" failure mode, so
    we do not gate on z-height or air time here.
    """
    from . import planner as planner_ops  # local import: keeps planner Isaac-Sim-free

    cmd = env.command_manager.get_term(command_name)
    foot_w = cmd.robot.data.body_pos_w[:, cmd.foot_ids]  # (B, 2, 3)
    phase = cmd.phase                                     # (B,)
    f = cmd.cfg.t_swing_fraction
    one_minus_f = 1.0 - f

    # Per-foot normalized swing phase in [0, 1]; outside its swing window,
    # value is clamped (and the mask zeroes the contribution anyway).
    swing_phase_left = (phase / f).clamp(0.0, 1.0)
    swing_phase_right = ((phase - f) / one_minus_f).clamp(0.0, 1.0)
    swing_phase = torch.stack([swing_phase_left, swing_phase_right], dim=-1)  # (B, 2)

    is_swing_left = (phase < f).float()
    is_swing_right = (phase >= f).float()
    is_swing = torch.stack([is_swing_left, is_swing_right], dim=-1)           # (B, 2)

    ref = planner_ops.swing_trajectory(
        start_w=cmd.last_contact_w,
        end_w=cmd.target_w,
        phase=swing_phase,
        apex=apex,
    )                                                                          # (B, 2, 3)

    err2 = ((foot_w - ref) ** 2).sum(dim=-1)                                   # (B, 2)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history.norm(dim=-1)             # (B, T, K)
    foot_forces = forces.max(dim=1).values[:, sensor_cfg.body_ids]             # (B, 2)
    if foot_forces.shape[1] != 2:
        raise RuntimeError(
            "footstep_swing_tracking expects sensor_cfg to resolve exactly two foot bodies; "
            f"got {foot_forces.shape[1]} bodies."
        )
    is_airborne = (foot_forces <= force_threshold).float()

    # Velocity-tracking gate: sech² on world-frame xy speed error, applied
    # identically to both feet (broadcast over F dim). This makes
    # marching-in-place (v≈0 while cmd≠0) unable to farm the swing-tracking
    # reward via in-place foot oscillation; only a body that translates with
    # its commanded speed unlocks the bonus. Using world-frame |v| avoids
    # sign-cancellation when the robot walks at the commanded speed in any
    # direction.
    v_cmd_b = env.command_manager.get_command(vel_gate_command_name)[:, :3]
    root_yaw = cmd.robot.data.heading_w
    c = torch.cos(root_yaw)
    s = torch.sin(root_yaw)
    v_cmd_w_x = c * v_cmd_b[:, 0] - s * v_cmd_b[:, 1]
    v_cmd_w_y = s * v_cmd_b[:, 0] + c * v_cmd_b[:, 1]
    v_root_xy = cmd.robot.data.root_lin_vel_w[:, :2]
    v_err = ((v_root_xy[:, 0] - v_cmd_w_x) ** 2
             + (v_root_xy[:, 1] - v_cmd_w_y) ** 2)
    vel_gate = 1.0 - torch.tanh(torch.sqrt(v_err + 1e-6) / vel_gate_std) ** 2

    per_foot = torch.exp(-err2 / (std * std)) * is_swing * is_airborne
    per_foot = per_foot * vel_gate.unsqueeze(-1)
    return per_foot.sum(dim=-1)


def footstep_contact_phase(
    env: ManagerBasedRLEnv,
    command_name: str = "footstep_plan",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Per-step reward for matching the planned contact schedule.

    Expected contact (1 = on ground, 0 = airborne):
        left foot  on ground iff  phase >= t_swing_fraction
        right foot on ground iff  phase <  t_swing_fraction

    reward = **product** over feet of ``1 - |expected - actual|``  →
    ``(B,)`` in ``{0, 1}``. Product (AND) rather than mean (OR) is
    important: with mean, a "both feet always in contact" policy could
    free-load 0.5 reward (one foot always matched). Product gives 0
    unless BOTH feet match the planned schedule simultaneously — the
    only state earning the reward is correct single-stance.
    """
    cmd = env.command_manager.get_term(command_name)
    phase = cmd.phase                                     # (B,)
    f = cmd.cfg.t_swing_fraction

    expected_left = (phase >= f).float()
    expected_right = (phase < f).float()
    expected = torch.stack([expected_left, expected_right], dim=-1)            # (B, 2)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history.norm(dim=-1)             # (B, T, K)
    foot_forces = forces.max(dim=1).values[:, sensor_cfg.body_ids]             # (B, 2)
    actual = (foot_forces > force_threshold).float()

    match = 1.0 - (expected - actual).abs()                                    # (B, 2)
    return match[:, 0] * match[:, 1]                                            # (B,) AND


def footstep_swing_tracking_log(
    env: ManagerBasedRLEnv,
    command_name: str = "footstep_plan",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    force_threshold: float = 1.0,
    eps: float = 1e-3,
    apex: float = 0.10,
) -> torch.Tensor:
    """DTC-style log-distance reward for swing-foot tracking.

    Equivalent to :func:`footstep_swing_tracking` but uses ``-log(d² + eps)``
    instead of ``exp(-d²/std²)``. The log form gives a much steeper gradient
    near the target than exp, which the DTC paper credits for fast convergence
    on planner-tracking tasks. The gating (swing-only, airborne-only) is
    unchanged.
    """
    from . import planner as planner_ops

    cmd = env.command_manager.get_term(command_name)
    foot_w = cmd.robot.data.body_pos_w[:, cmd.foot_ids]  # (B, 2, 3)
    phase = cmd.phase
    f = cmd.cfg.t_swing_fraction
    one_minus_f = 1.0 - f

    swing_phase_left = (phase / f).clamp(0.0, 1.0)
    swing_phase_right = ((phase - f) / one_minus_f).clamp(0.0, 1.0)
    swing_phase = torch.stack([swing_phase_left, swing_phase_right], dim=-1)

    is_swing_left = (phase < f).float()
    is_swing_right = (phase >= f).float()
    is_swing = torch.stack([is_swing_left, is_swing_right], dim=-1)

    ref = planner_ops.swing_trajectory(
        start_w=cmd.last_contact_w,
        end_w=cmd.target_w,
        phase=swing_phase,
        apex=apex,
    )

    d2 = ((foot_w - ref) ** 2).sum(dim=-1)  # (B, 2)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history.norm(dim=-1)
    foot_forces = forces.max(dim=1).values[:, sensor_cfg.body_ids]
    is_airborne = (foot_forces <= force_threshold).float()

    # -log(d² + eps): bounded above by -log(eps), goes to -inf as d→∞ (clamp).
    log_term = -torch.log(d2 + eps)
    # Subtract the at-target maximum so a perfect track yields 0, mistracks negative.
    log_term = log_term - (-math.log(eps))
    per_foot = log_term * is_swing * is_airborne
    # Cap downside so a single bad foot can't dominate.
    per_foot = per_foot.clamp(min=-10.0)
    return per_foot.sum(dim=-1)


def planner_consistency(env: ManagerBasedRLEnv, command_name: str = "footstep_plan") -> torch.Tensor:
    """Squared distance between successive plan_buffers, summed over (k, foot).

    Returns a positive value; assign a negative weight in the env config.
    Used to discourage the cost-based planner from flip-flopping between
    candidate footholds frame-over-frame — the DTC paper's "consistency"
    cost weighted at 20.
    """
    cmd = env.command_manager.get_term(command_name)
    diff = cmd.plan_buffer - cmd.prev_plan_buffer  # (B, N, 2, 3)
    return (diff ** 2).sum(dim=(1, 2, 3))


G1_ANKLE_ROLL_MESH_MIN_Z = -0.035409145057201385
_FOOTHOLD_FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize second-order action differences (BeamDojo Table VII smoothness term)."""
    action_manager = env.action_manager
    if not hasattr(env, "_beamdojo_prev_prev_action"):
        env._beamdojo_prev_prev_action = torch.zeros_like(action_manager.action)
    diff = action_manager.action - 2.0 * action_manager.prev_action + env._beamdojo_prev_prev_action
    penalty = torch.sum(torch.square(diff), dim=1)
    env._beamdojo_prev_prev_action[:] = action_manager.prev_action
    return penalty


def joint_power(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Normalized joint power penalty (BeamDojo Table VII)."""
    asset: Articulation = env.scene[asset_cfg.name]
    power = torch.sum(torch.abs(asset.data.applied_torque) * torch.abs(asset.data.joint_vel), dim=-1)
    lin_vel = torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=-1)
    ang_vel = torch.sum(torch.square(asset.data.root_ang_vel_w), dim=-1)
    return power / (lin_vel + 0.2 * ang_vel + 1e-6)


def feet_distance_y(
    env: ManagerBasedRLEnv,
    min_distance: float = 0.18,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
) -> torch.Tensor:
    """Reward lateral foot separation: ReLU(|y_left - y_right| - d_min)."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_body_ids, _ = asset.find_bodies(
        ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    )
    feet_pos = asset.data.body_pos_w[:, foot_body_ids, :]
    y_sep = torch.abs(feet_pos[:, 0, 1] - feet_pos[:, 1, 1])
    return torch.relu(y_sep - min_distance)


def feet_ground_parallel(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    ),
    height_scanner_name: str = "height_scanner",
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    ),
    foot_length: float = 0.18,
    foot_width: float = 0.065,
    n_long: int = 4,
    n_lat: int = 3,
    force_threshold: float = 1.0,
    sole_z_offset: float = G1_ANKLE_ROLL_MESH_MIN_Z,
) -> torch.Tensor:
    """Penalize per-foot variance of sampled terrain z under stance feet."""
    terrain_z, in_contact, _ = _foothold_sample_geometry(
        env,
        sensor_cfg=sensor_cfg,
        height_scanner_name=height_scanner_name,
        asset_cfg=asset_cfg,
        foot_length=foot_length,
        foot_width=foot_width,
        n_long=n_long,
        n_lat=n_lat,
        force_threshold=force_threshold,
        sole_z_offset=sole_z_offset,
    )
    # Var over sample points per foot; mask to stance feet only.
    mean_z = terrain_z.mean(dim=-1, keepdim=True)
    var_z = torch.mean(torch.square(terrain_z - mean_z), dim=-1)
    return torch.sum(var_z * in_contact, dim=1)


# =========================================================================
# BeamDojo-style rewards (Wang et al. 2025, arXiv:2502.10363).
#
# Design (per BeamDojo paper, "sampling-based foothold reward for polygonal
# feet"): when a foot is in stance, sample N points on the rectangular
# outline of the sole and check each against the height scanner's ray hits.
# A sample is "supported" if the ray hit nearest in xy is within
# `support_threshold` of the actual foot sole z. The per-step reward is the
# fraction of supported samples; sparse reward − only fires when the foot
# is actually in contact (no reward for flying feet).
#
# This is feedback-style: evaluates actual foot placement quality rather
# than asking the policy to track a planner-supplied target. We use a
# rectangular sampling grid. The sole offset comes from Unitree G1
# ankle_roll_link STL bounds:
#   /home/tan/unitree_mujoco/unitree_robots/g1/meshes/*_ankle_roll_link.STL
#   bounds min_z = -0.035409145 m, x extent = 0.2082 m, y extent = 0.0756 m.
# =========================================================================


def _foothold_sample_geometry(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    height_scanner_name: str,
    asset_cfg: SceneEntityCfg,
    foot_length: float,
    foot_width: float,
    n_long: int,
    n_lat: int,
    force_threshold: float,
    sole_z_offset: float,
    support_threshold: float = 0.03,
):
    """Return terrain z at sole samples, contact mask, and support mask."""
    from isaaclab.sensors import RayCaster

    asset: Articulation = env.scene[asset_cfg.name]
    foot_body_ids, resolved_asset_names = asset.find_bodies(_FOOTHOLD_FOOT_NAMES, preserve_order=True)
    foot_pos_w = asset.data.body_pos_w[:, foot_body_ids]

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    sensor_body_ids, resolved_sensor_names = contact_sensor.find_bodies(_FOOTHOLD_FOOT_NAMES, preserve_order=True)

    if not getattr(env, "_foothold_dbg_printed", False):
        print(
            "[foothold_sampling DBG] "
            f"asset_ids={foot_body_ids} asset_names={resolved_asset_names} "
            f"sensor_ids={sensor_body_ids} sensor_names={resolved_sensor_names} "
            f"foot_pos_w.shape={tuple(foot_pos_w.shape)}"
        )
        env._foothold_dbg_printed = True
    if foot_pos_w.shape[1] != 2:
        raise RuntimeError(
            f"foothold sampling expects exactly two foot bodies; got "
            f"foot_body_ids={foot_body_ids} (n={foot_pos_w.shape[1]}). "
            f"Resolved asset names: {resolved_asset_names}."
        )
    if len(sensor_body_ids) != 2:
        raise RuntimeError(
            "foothold sampling expects the contact sensor to resolve exactly two foot bodies; "
            f"got sensor_body_ids={sensor_body_ids}, names={resolved_sensor_names}."
        )

    foot_rot_w = asset.data.body_quat_w[:, foot_body_ids]
    qw = foot_rot_w[..., 0]
    qx = foot_rot_w[..., 1]
    qy = foot_rot_w[..., 2]
    qz = foot_rot_w[..., 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    device = foot_pos_w.device
    xs = torch.linspace(-foot_length * 0.5, foot_length * 0.5, n_long, device=device)
    ys = torch.linspace(-foot_width * 0.5, foot_width * 0.5, n_lat, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    grid_b = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    n_samples = grid_b.shape[0]

    grid_b_exp = grid_b.view(1, 1, n_samples, 2)
    rot_c = c.unsqueeze(-1)
    rot_s = s.unsqueeze(-1)
    x_w = rot_c * grid_b_exp[..., 0] - rot_s * grid_b_exp[..., 1]
    y_w = rot_s * grid_b_exp[..., 0] + rot_c * grid_b_exp[..., 1]
    sample_xy = torch.stack([x_w, y_w], dim=-1) + foot_pos_w[..., :2].unsqueeze(2)

    height_scanner: RayCaster = env.scene.sensors[height_scanner_name]
    ray_hits_w = height_scanner.data.ray_hits_w
    diff = sample_xy[..., None, :] - ray_hits_w[:, None, None, :, :2]
    d2 = (diff * diff).sum(dim=-1)
    nearest = d2.argmin(dim=-1)
    b_idx = torch.arange(sample_xy.shape[0], device=device).view(-1, 1, 1).expand(-1, 2, n_samples)
    terrain_z = ray_hits_w[b_idx, nearest, 2]

    foot_sole_z = foot_pos_w[..., 2:3] + sole_z_offset
    foot_force_hist = contact_sensor.data.net_forces_w_history[:, :, sensor_body_ids, :]
    foot_forces = foot_force_hist.norm(dim=-1).amax(dim=1)
    if foot_forces.ndim != 2 or foot_forces.shape[1] != 2:
        raise RuntimeError(
            "foothold sampling expects contact forces with shape (num_envs, 2); "
            f"got {tuple(foot_forces.shape)}."
        )
    in_contact = (foot_forces > force_threshold).float()
    supported = (torch.abs(terrain_z - foot_sole_z) <= support_threshold).float()
    return terrain_z, in_contact, supported


def foothold_sampling(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    ),
    height_scanner_name: str = "height_scanner",
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    ),
    foot_length: float = 0.18,
    foot_width: float = 0.065,
    n_long: int = 4,
    n_lat: int = 3,
    support_threshold: float = 0.03,
    force_threshold: float = 1.0,
    sole_z_offset: float = G1_ANKLE_ROLL_MESH_MIN_Z,
) -> torch.Tensor:
    """Positive support-fraction reward in [0, 1] (legacy / ablation helper)."""
    _, in_contact, supported = _foothold_sample_geometry(
        env,
        sensor_cfg=sensor_cfg,
        height_scanner_name=height_scanner_name,
        asset_cfg=asset_cfg,
        foot_length=foot_length,
        foot_width=foot_width,
        n_long=n_long,
        n_lat=n_lat,
        force_threshold=force_threshold,
        sole_z_offset=sole_z_offset,
        support_threshold=support_threshold,
    )
    support_frac = supported.mean(dim=-1)
    per_foot = support_frac * in_contact
    if not getattr(env, "_foothold_geom_dbg_printed", False):
        print(
            "[foothold_sampling GEOM] "
            f"sole_z_offset={sole_z_offset:.6f} "
            f"support_frac_mean={support_frac.mean().item():.4f} "
            f"contact_ratio={in_contact.mean().item():.4f}"
        )
        env._foothold_geom_dbg_printed = True
    return per_foot.mean(dim=1)


def foothold_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    ),
    height_scanner_name: str = "height_scanner",
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    ),
    foot_length: float = 0.18,
    foot_width: float = 0.065,
    n_long: int = 4,
    n_lat: int = 3,
    height_epsilon: float = -0.1,
    force_threshold: float = 1.0,
    sole_z_offset: float = G1_ANKLE_ROLL_MESH_MIN_Z,
) -> torch.Tensor:
    """BeamDojo sparse foothold penalty (Table VII, weight -1.0).

    Returns ``sum_i C_i * sum_j 1{d_ij < epsilon}`` — positive magnitude to
    multiply by a negative reward weight.
    """
    terrain_z, in_contact, _ = _foothold_sample_geometry(
        env,
        sensor_cfg=sensor_cfg,
        height_scanner_name=height_scanner_name,
        asset_cfg=asset_cfg,
        foot_length=foot_length,
        foot_width=foot_width,
        n_long=n_long,
        n_lat=n_lat,
        force_threshold=force_threshold,
        sole_z_offset=sole_z_offset,
    )
    bad_samples = (terrain_z < height_epsilon).float()
    per_foot = bad_samples.sum(dim=-1) * in_contact
    return per_foot.sum(dim=1)
