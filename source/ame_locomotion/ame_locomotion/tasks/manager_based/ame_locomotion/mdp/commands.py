from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.mdp import TerrainBasedPose2dCommandCfg, TerrainBasedPose2dCommand
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils import configclass

from . import planner as planner_ops


class TimeLimitedTerrainBasedPose2dCommand(TerrainBasedPose2dCommand):
    """Command generator that generates pose commands containing a 3-D position, heading, and time-to-go.

    The command generator samples uniform 2D positions around the environment origin. It sets
    the height of the position command to the default root height of the robot. The heading
    command is either set to point towards the target or is sampled uniformly.
    This can be configured through the :attr:`Pose2dCommandCfg.simple_heading` parameter in
    the configuration.

    The command tensor includes the time remaining to reach the target as the last element.
    Shape: (num_envs, 5) -> [x, y, z, heading, time_to_go]
    """

    cfg: TimeLimitedTerrainBasedPose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TimeLimitedTerrainBasedPose2dCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose and time-to-go in base frame. Shape is (num_envs, 5).

        Indices:
            0-2: Position command (x, y, z) in base frame.
            3: Heading command in base frame.
            4: Time to go.
        """
        return torch.cat([
            self.pos_command_b,
            self.heading_command_b.unsqueeze(1),
            self.time_left.unsqueeze(1)
        ], dim=1)


@configclass
class TimeLimitedTerrainBasedPose2dCommandCfg(TerrainBasedPose2dCommandCfg):

    class_type = TimeLimitedTerrainBasedPose2dCommand


# =========================================================================
# Footstep planner (Phase 0 — Raibert heuristic, see PLAN.md)
# =========================================================================


_DEFAULT_FOOTSTEP_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/footstep_plan",
    markers={
        "target_left": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 1.0)),
        ),
        "target_right": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.3)),
        ),
        "last_contact": sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
        "phantom": sim_utils.SphereCfg(
            radius=0.06,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        ),
    },
)


class FootstepPlanCommand(CommandTerm):
    """Raibert-style heuristic footstep planner, exposed as a ``CommandTerm``.

    Per environment we maintain a phase clock ``phase in [0, 1)`` over a full
    stride of duration ``t_step``. The first ``t_swing_fraction`` of each
    stride swings the left foot; the remainder swings the right foot
    (no explicit double-support phase in Phase 0).

    At every step we project the next ``n_future_steps`` footsteps using:
        1. forward-extrapolating root pose by the velocity command,
        2. computing a Raibert target around each predicted hip,
        3. snapping the target z to the nearest ray-cast hit from the height
           scanner — this is privileged (clean) height info.

    The flattened plan is exposed via :pyattr:`command` for both
    ``mdp.generated_commands`` observations and reward terms.
    """

    cfg: "FootstepPlanCommandCfg"

    def __init__(self, cfg: "FootstepPlanCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # ---- assets ------------------------------------------------------
        self.robot: Articulation = env.scene[cfg.asset_name]
        foot_ids, _ = self.robot.find_bodies(list(cfg.foot_body_names), preserve_order=True)
        if len(foot_ids) != 2:
            raise ValueError(
                f"FootstepPlanCommand expects exactly 2 foot bodies matching "
                f"{cfg.foot_body_names}, got ids={foot_ids}."
            )
        self.foot_ids = torch.tensor(foot_ids, device=self.device, dtype=torch.long)

        self.height_scanner: RayCaster = env.scene.sensors[cfg.height_scanner_name]

        # ---- grid shape for fast foothold selection ---------------------
        # Derive (H, W) and resolution from the scanner's pattern_cfg when it
        # is a GridPatternCfg; otherwise fall back to None (brute-force path).
        self._cost_grid_shape: tuple[int, int] | None = None
        self._cost_grid_resolution: float | None = None
        try:
            from isaaclab.sensors.ray_caster.patterns.patterns_cfg import GridPatternCfg as _GridPatternCfg
            pcfg = self.height_scanner.cfg.pattern_cfg
            if isinstance(pcfg, _GridPatternCfg):
                import math as _math
                W = round(pcfg.size[0] / pcfg.resolution) + 1  # x-direction cells
                H = round(pcfg.size[1] / pcfg.resolution) + 1  # y-direction cells
                self._cost_grid_shape = (H, W)
                self._cost_grid_resolution = pcfg.resolution
        except Exception:
            pass  # Any import/attribute error → keep None (brute-force fallback)

        # ---- hip offsets (body frame) -----------------------------------
        # Initial guess from cfg; will be overwritten on first _resample_command
        # using the robot's actual standing pose (foot pos relative to root,
        # rotated into body frame). Hardcoded values produced a systematic
        # forward-bias of ~20cm against AME pretrained, see PLAN.md notes.
        self.hip_offset_b = torch.tensor(
            [[0.0, +cfg.hip_y, -cfg.leg_length],
             [0.0, -cfg.hip_y, -cfg.leg_length]],
            device=self.device,
            dtype=torch.float,
        )
        self._hip_offset_calibrated = False

        # ---- per-env buffers --------------------------------------------
        B = self.num_envs
        N = cfg.n_future_steps
        self.phase = torch.zeros(B, device=self.device)
        self.swing_foot = torch.zeros(B, device=self.device, dtype=torch.long)  # 0=L, 1=R
        self.target_w = torch.zeros(B, 2, 3, device=self.device)
        self.last_contact_w = torch.zeros(B, 2, 3, device=self.device)
        self.plan_buffer = torch.zeros(B, N, 2, 3, device=self.device)
        self.prev_plan_buffer = torch.zeros(B, N, 2, 3, device=self.device)
        self.time_left_buffer = torch.zeros(B, N, 2, device=self.device)
        # Static between commits; (1 = on ground at landing time, 0 = airborne)
        self.contact_target_buffer = torch.zeros(B, N, 2, device=self.device)

        # ---- phantom (virtual reference body) ---------------------------
        # Phantom is a virtual pelvis that advances by v_cmd every dt with a
        # leash bound to robot. When use_phantom=True, _commit_plan uses
        # phantom pose instead of robot pose, so foot targets stay ahead of
        # the robot even when the robot stalls.
        self.phantom_pos_w = torch.zeros(B, 3, device=self.device)
        self.phantom_yaw_w = torch.zeros(B, device=self.device)

        # cached timing
        self._t_swing = cfg.t_step * cfg.t_swing_fraction

        # metrics
        self.metrics["plan_step_mean_dx"] = torch.zeros(B, device=self.device)

    # ---------------------------------------------------------------- API

    @property
    def command(self) -> torch.Tensor:
        """Flattened plan, shape ``(B, n_future_steps * 2 * 3)``."""
        return self.plan_buffer.flatten(start_dim=1)

    # -------------------------------------------------- CommandTerm hooks

    def _update_metrics(self):
        # crude diagnostic: signed forward-component of the immediate target
        # relative to current root xy, projected on root-x.
        root_pos = self.robot.data.root_pos_w
        root_yaw = self.robot.data.heading_w
        c = torch.cos(root_yaw)
        s = torch.sin(root_yaw)
        dx = self.target_w[:, :, 0] - root_pos[:, 0:1]
        dy = self.target_w[:, :, 1] - root_pos[:, 1:2]
        # dot with body-x (cos, sin)
        forward = (c.unsqueeze(-1) * dx + s.unsqueeze(-1) * dy).mean(dim=-1)
        self.metrics["plan_step_mean_dx"][:] = forward

    def _resample_command(self, env_ids: Sequence[int]):
        # Episode-level reset: zero phase clock, latch current foot poses,
        # and commit an initial plan from current state.
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.phase[env_ids] = 0.0
        self.swing_foot[env_ids] = 0
        foot_w = self.robot.data.body_pos_w[env_ids][:, self.foot_ids]  # (E, 2, 3)
        self.last_contact_w[env_ids] = foot_w

        # First-time auto-calibrate hip offsets from actual standing geometry.
        # We expect the very first reset to happen with the robot in its
        # default (near-standing) pose, so the body-frame foot offset is a
        # good proxy for hip offset (z absorbs leg length).
        if not self._hip_offset_calibrated and env_ids.numel() >= 1:
            self._calibrate_hip_offset(env_ids[:1])

        # Phantom: anchor to robot's current pose at episode/resample boundary.
        # From here phantom drifts forward by v_cmd each dt (with leash).
        self.phantom_pos_w[env_ids] = self.robot.data.root_pos_w[env_ids]
        self.phantom_yaw_w[env_ids] = self.robot.data.heading_w[env_ids]

        self._commit_plan(env_ids)
        self.target_w[env_ids] = self.plan_buffer[env_ids, 0]
        # Sync prev_plan_buffer to the freshly committed plan so the very next
        # planner_consistency reward computation sees zero diff. Without this,
        # _commit_plan's internal snapshot captured the zero-initialized
        # plan_buffer, and the diff against the first real commit produces a
        # ~70 m² spike per env → ~-1700 reward at iter 0 (smoke 2026-06-29).
        self.prev_plan_buffer[env_ids] = self.plan_buffer[env_ids].clone()

    def _calibrate_hip_offset(self, env_ids: torch.Tensor):
        """Measure body-frame foot offset (= effective hip_offset_b) at the
        robot's default standing pose. Overwrites the cfg-supplied initial
        guess. Called once on the very first reset."""
        env_id = env_ids[0:1]
        root_pos = self.robot.data.root_pos_w[env_id]   # (1, 3)
        yaw = self.robot.data.heading_w[env_id]         # (1,)
        foot_w = self.robot.data.body_pos_w[env_id][:, self.foot_ids]  # (1, 2, 3)
        rel = foot_w - root_pos.unsqueeze(1)            # (1, 2, 3)
        c = torch.cos(-yaw)
        s = torch.sin(-yaw)
        # rotate world rel into body frame using -yaw
        rel_b_x = c * rel[..., 0] - s * rel[..., 1]
        rel_b_y = s * rel[..., 0] + c * rel[..., 1]
        rel_b_z = rel[..., 2]
        self.hip_offset_b = torch.stack([rel_b_x[0], rel_b_y[0], rel_b_z[0]], dim=-1)  # (2, 3)
        self._hip_offset_calibrated = True
        print(
            f"[FootstepPlanCommand] hip_offset_b auto-calibrated from standing pose:\n"
            f"  left  = {self.hip_offset_b[0].cpu().tolist()}\n"
            f"  right = {self.hip_offset_b[1].cpu().tolist()}"
        )

    def _update_phantom(self):
        """Advance phantom pelvis by v_cmd*dt, then leash to robot.

        Phantom is a virtual reference body that follows the command perfectly,
        independent of whether the real robot tracks it. The leash bounds the
        ahead-of-robot offset to ``max_leash`` so targets remain reachable.
        """
        dt = self._env.step_dt
        vel_cmd_b = self._env.command_manager.get_command(self.cfg.velocity_command_name)[:, :3]
        # Option B: phantom shares robot yaw — phantom is always "in front of"
        # the robot in body frame, regardless of yaw drift.
        # Option A (default): phantom yaw evolves only via commanded wz.
        if self.cfg.phantom_yaw_track_robot:
            self.phantom_yaw_w = self.robot.data.heading_w.clone()
        c = torch.cos(self.phantom_yaw_w)
        s = torch.sin(self.phantom_yaw_w)
        vel_w_x = c * vel_cmd_b[:, 0] - s * vel_cmd_b[:, 1]
        vel_w_y = s * vel_cmd_b[:, 0] + c * vel_cmd_b[:, 1]
        self.phantom_pos_w[:, 0] = self.phantom_pos_w[:, 0] + dt * vel_w_x
        self.phantom_pos_w[:, 1] = self.phantom_pos_w[:, 1] + dt * vel_w_y
        if not self.cfg.phantom_yaw_track_robot:
            self.phantom_yaw_w = self.phantom_yaw_w + dt * vel_cmd_b[:, 2]

        # Leash: clip phantom-to-robot xy offset to max_leash.
        robot_xy = self.robot.data.root_pos_w[:, :2]
        offset = self.phantom_pos_w[:, :2] - robot_xy
        norm = torch.linalg.norm(offset, dim=-1)
        over = norm > self.cfg.max_leash
        leash_active = over[0].item() if over.numel() > 0 else False
        if over.any():
            scale = self.cfg.max_leash / norm[over].clamp_min(1e-6)
            self.phantom_pos_w[over, :2] = robot_xy[over] + offset[over] * scale.unsqueeze(-1)

        # Keep phantom z aligned with robot z (so target z snapping uses
        # roughly the right scanner cell). Phantom does not model height.
        self.phantom_pos_w[:, 2] = self.robot.data.root_pos_w[:, 2]

        # ---- Debug print (env 0, once per second) -----------------------
        if not hasattr(self, "_dbg_step"):
            self._dbg_step = 0
        self._dbg_step += 1
        if self.cfg.debug_print_period > 0 and (self._dbg_step % self.cfg.debug_print_period == 0):
            i = 0
            rpos = self.robot.data.root_pos_w[i].cpu().tolist()
            ppos = self.phantom_pos_w[i].cpu().tolist()
            ryaw = self.robot.data.heading_w[i].item()
            pyaw = self.phantom_yaw_w[i].item()
            vcmd = vel_cmd_b[i].cpu().tolist()
            off_xy = (
                self.phantom_pos_w[i, 0].item() - rpos[0],
                self.phantom_pos_w[i, 1].item() - rpos[1],
            )
            # Body-frame phantom offset (forward, lateral) — what we expect
            # for forward-only: forward > 0, lateral ≈ 0.
            cr = math.cos(-ryaw)
            sr = math.sin(-ryaw)
            fwd = cr * off_xy[0] - sr * off_xy[1]
            lat = sr * off_xy[0] + cr * off_xy[1]
            tgt_L = self.target_w[i, 0].cpu().tolist()
            tgt_R = self.target_w[i, 1].cpu().tolist()
            tL_off = (tgt_L[0] - ppos[0], tgt_L[1] - ppos[1])
            tR_off = (tgt_R[0] - ppos[0], tgt_R[1] - ppos[1])
            print(
                f"[phantom dbg step={self._dbg_step}] "
                f"vcmd=({vcmd[0]:+.2f},{vcmd[1]:+.2f},wz={vcmd[2]:+.2f})  "
                f"yaw(r={ryaw:+.2f} p={pyaw:+.2f})  "
                f"phantom-robot xy=({off_xy[0]:+.3f},{off_xy[1]:+.3f}) "
                f"body=(fwd={fwd:+.3f},lat={lat:+.3f})  "
                f"leash={'ON' if leash_active else 'off'}  "
                f"tgtL-phantom=({tL_off[0]:+.2f},{tL_off[1]:+.2f}) "
                f"tgtR-phantom=({tR_off[0]:+.2f},{tR_off[1]:+.2f})"
            )

    def _update_command(self):
        if self.cfg.use_phantom:
            self._update_phantom()
        dt = self._env.step_dt
        prev_phase = self.phase.clone()
        self.phase = (self.phase + dt / self.cfg.t_step) % 1.0

        # time_left_in_phase(k, foot) = horizon(k, foot) - elapsed_in_stride.
        # horizon mirrors _commit_plan's per-foot schedule:
        #   swing foot: k*t_step + t_swing
        #   other foot: (k+1)*t_step
        N = self.cfg.n_future_steps
        elapsed = self.phase * self.cfg.t_step  # (B,)
        is_swing_left = (self.swing_foot == 0).float()  # (B,)
        k_idx = torch.arange(N, device=self.device).view(1, N, 1).float()
        # horizon for foot 0 (left)
        h_swing = k_idx * self.cfg.t_step + self._t_swing
        h_other = (k_idx + 1) * self.cfg.t_step
        # left-foot horizon: h_swing if left is swing, else h_other
        h_left = is_swing_left.view(-1, 1, 1) * h_swing + (1.0 - is_swing_left.view(-1, 1, 1)) * h_other
        h_right = (1.0 - is_swing_left.view(-1, 1, 1)) * h_swing + is_swing_left.view(-1, 1, 1) * h_other
        horizon = torch.cat([h_left, h_right], dim=-1)  # (B, N, 2)
        self.time_left_buffer = (horizon - elapsed.view(-1, 1, 1)).clamp(min=0.0)

        # swing-foot crossover: phase crossed t_swing_fraction (left→right)
        # or wrapped back through 0 (right→left).
        crossed_half = (prev_phase < self.cfg.t_swing_fraction) & (
            self.phase >= self.cfg.t_swing_fraction
        )
        wrapped = self.phase < prev_phase  # passed through 1.0 → 0.0
        crossover_mask = crossed_half | wrapped
        if crossover_mask.any():
            ids = crossover_mask.nonzero(as_tuple=False).squeeze(-1)
            swing_foot = self.swing_foot[ids]
            # the foot that *was* swinging just touched down → latch its pos
            row = torch.arange(ids.numel(), device=self.device)
            foot_pos_w = self.robot.data.body_pos_w[ids][row, self.foot_ids[swing_foot]]
            self.last_contact_w[ids, swing_foot] = foot_pos_w
            # flip swing foot
            self.swing_foot[ids] = 1 - swing_foot
            # Commit a new plan only on crossover — targets stay fixed in world
            # frame between crossovers (DTC-style "plan once, then track").
            self._commit_plan(ids)
            self.target_w[ids] = self.plan_buffer[ids, 0]

    # -------------------------------------------------------- planning

    def _commit_plan(self, env_ids: torch.Tensor):
        """Commit the next ``n_future_steps`` footsteps based on **current**
        root state, for the given environments. Plan stays fixed (world frame)
        until the next swing-onset commit.

        Each foot has its **own** landing schedule (the two feet leap-frog):
            - current swing foot: k-th landing at  ``t = k*t_step + t_swing``
            - other foot:         k-th landing at  ``t = (k+1)*t_step``

        Both targets are extrapolated from the **current** root state along
        the velocity command, including yaw-rate (``wz``) compensation:

        * For the position extrapolation we use **mid-path yaw** (root_yaw +
          0.5·horizon·wz), so the straight-line approximation samples the
          rotating body velocity at the path midpoint — accurate to second
          order for moderate wz.
        * For the Raibert hip-offset and forward-push terms we use
          **landing yaw** (root_yaw + horizon·wz), so the foot is placed in
          the body frame that the robot will *actually have* at touchdown.

        Assumes ``t_swing_fraction = 0.5`` (no double-support).
        """
        N = self.cfg.n_future_steps
        if self.cfg.use_phantom:
            # Targets anchored to phantom (idealized command-following body)
            # so they keep advancing even when the real robot stalls.
            root_pos = self.phantom_pos_w[env_ids]
            root_yaw = self.phantom_yaw_w[env_ids]
        else:
            root_pos = self.robot.data.root_pos_w[env_ids]
            root_yaw = self.robot.data.heading_w[env_ids]
        root_vel = self.robot.data.root_lin_vel_w[env_ids]
        vel_cmd_b = self._env.command_manager.get_command(self.cfg.velocity_command_name)[env_ids, :3]
        wz = vel_cmd_b[:, 2]
        swing_foot = self.swing_foot[env_ids]  # (E,) int in {0=L, 1=R}

        ray_hits_w = self.height_scanner.data.ray_hits_w[env_ids]

        # Snapshot previous plan for consistency reward.
        self.prev_plan_buffer[env_ids] = self.plan_buffer[env_ids].clone()

        for k in range(N):
            for foot_idx in range(2):
                is_swing = (swing_foot == foot_idx).float()  # (E,)
                h_swing = k * self.cfg.t_step + self._t_swing
                h_other = (k + 1) * self.cfg.t_step
                horizon = is_swing * h_swing + (1.0 - is_swing) * h_other  # (E,)

                # Per-foot yaw extrapolation (constant-wz approximation)
                mid_yaw = root_yaw + 0.5 * horizon * wz
                landing_yaw = root_yaw + horizon * wz

                # Velocity in world at the path midpoint orientation
                c_mid = torch.cos(mid_yaw)
                s_mid = torch.sin(mid_yaw)
                vel_w_xy_mid = torch.stack(
                    [c_mid * vel_cmd_b[:, 0] - s_mid * vel_cmd_b[:, 1],
                     s_mid * vel_cmd_b[:, 0] + c_mid * vel_cmd_b[:, 1]],
                    dim=-1,
                )

                root_pos_kf = root_pos.clone()
                root_pos_kf[:, :2] = root_pos_kf[:, :2] + horizon.unsqueeze(-1) * vel_w_xy_mid

                hip_offset_single = self.hip_offset_b[foot_idx : foot_idx + 1]  # (1, 3)
                tgt = planner_ops.raibert_target(
                    root_pos_w=root_pos_kf,
                    root_lin_vel_w=root_vel,
                    root_yaw=landing_yaw,  # rotate hip_offset & raibert delta by future body yaw
                    vel_cmd_b=vel_cmd_b,
                    hip_offset_b=hip_offset_single,
                    t_swing=self._t_swing,
                    k_fb=self.cfg.raibert_k,
                    raibert_factor=self.cfg.raibert_factor,
                )  # (E, 1, 3)
                if self.cfg.use_cost_selection:
                    from . import foothold_selection
                    tgt, _ = foothold_selection.select_foothold_by_cost(
                        raibert_xy_w=tgt,
                        ray_hits_w=ray_hits_w,
                        window_m=self.cfg.cost_window_m,
                        alpha=self.cfg.cost_alpha,
                        beta=self.cfg.cost_beta,
                        gamma=self.cfg.cost_gamma,
                        delta=self.cfg.cost_delta,
                        obstacle_height_threshold=self.cfg.cost_obstacle_threshold,
                        grid_shape=self._cost_grid_shape,
                        grid_resolution=self._cost_grid_resolution,
                    )
                else:
                    tgt = self._snap_z_via_nearest_ray(tgt, ray_hits_w)
                self.plan_buffer[env_ids, k, foot_idx] = tgt[:, 0]
                # contact_target: 1 (on ground at touchdown), but for swing foot
                # at k=0 it's "still airborne now". For DTC observation purposes
                # we use the *target* contact state at landing time = always 1.
                self.contact_target_buffer[env_ids, k, foot_idx] = 1.0

    def _snap_z_via_nearest_ray(
        self, target_w: torch.Tensor, ray_hits_w: torch.Tensor
    ) -> torch.Tensor:
        """Set target.z to the height of the ray-cast hit nearest in xy.

        The height scanner is yaw-aligned and may have drift, so a regular
        grid lookup is fragile. Nearest-ray is O(B*F*K) but K ~ 700 and F=2,
        which is fine for Phase 0 inner loop.
        """
        B, F, _ = target_w.shape
        diff = target_w[:, :, None, :2] - ray_hits_w[:, None, :, :2]
        d2 = (diff * diff).sum(dim=-1)
        nearest = d2.argmin(dim=-1)  # (B, F)
        b_idx = torch.arange(B, device=self.device).view(B, 1).expand(B, F)
        z = ray_hits_w[b_idx, nearest, 2]
        out = target_w.clone()
        out[..., 2] = z
        return out

    # ----------------------------------------------------------- debug viz

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_marker"):
                self._marker = VisualizationMarkers(self.cfg.marker_cfg)
            self._marker.set_visibility(True)
        elif hasattr(self, "_marker"):
            self._marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        B = self.num_envs
        pos_left = self.target_w[:, 0]
        pos_right = self.target_w[:, 1]
        pos_last = self.last_contact_w.reshape(B * 2, 3)
        if self.cfg.use_phantom:
            positions = torch.cat([pos_left, pos_right, pos_last, self.phantom_pos_w], dim=0)
            marker_indices = torch.cat(
                [
                    torch.zeros(B, dtype=torch.long, device=self.device),
                    torch.ones(B, dtype=torch.long, device=self.device),
                    torch.full((2 * B,), 2, dtype=torch.long, device=self.device),
                    torch.full((B,), 3, dtype=torch.long, device=self.device),
                ],
                dim=0,
            )
        else:
            positions = torch.cat([pos_left, pos_right, pos_last], dim=0)
            marker_indices = torch.cat(
                [
                    torch.zeros(B, dtype=torch.long, device=self.device),
                    torch.ones(B, dtype=torch.long, device=self.device),
                    torch.full((2 * B,), 2, dtype=torch.long, device=self.device),
                ],
                dim=0,
            )
        self._marker.visualize(translations=positions, marker_indices=marker_indices)


@configclass
class FootstepPlanCommandCfg(CommandTermCfg):
    """Configuration for :class:`FootstepPlanCommand`."""

    class_type: type = FootstepPlanCommand

    asset_name: str = MISSING
    """Articulation to read root state and foot positions from."""

    foot_body_names: tuple[str, str] = MISSING
    """(left, right) body name patterns. Must resolve to exactly 2 bodies."""

    height_scanner_name: str = "height_scanner"
    """Name of the RayCaster sensor used for the privileged elevation map."""

    velocity_command_name: str = "base_velocity"
    """Name of the velocity command term to read forward-prediction signal from."""

    t_step: float = 0.6
    """Full stride duration, seconds (one left-swing + one right-swing)."""

    t_swing_fraction: float = 0.5
    """Fraction of ``t_step`` spent in single-support per foot."""

    raibert_k: float = 0.05
    """Velocity-tracking feedback gain in the Raibert heuristic."""

    raibert_factor: float = 0.3
    """Scales the forward-push term: ``factor * t_swing * v_des``.

    - 0.5 = classic Raibert (foot lands at predicted mid-swing hip position).
    - 0.0 ≈ matches the pretrained AME policy's "marching" gait (foot lands
      directly under hip; we measured an effective factor of ~0 in the
      residual eval).
    - 0.3 (default) = compromise: a soft pull toward Raibert that leaves the
      policy room to adapt without forcing dramatic gait change."""

    n_future_steps: int = 2
    """How many future footsteps to expose in the plan buffer."""

    use_phantom: bool = False
    """If True, anchor footstep targets to an idealized virtual pelvis
    ("phantom") that advances by the velocity command each dt, instead of
    the actual robot pose. The phantom is leashed to within ``max_leash``
    meters of the robot so targets remain reachable. This eliminates the
    static-target exploit where a non-translating robot can satisfy
    foot-tracking rewards via in-place foot oscillation."""

    max_leash: float = 0.5
    """Maximum xy distance phantom may lead the robot, in meters. When the
    robot lags, phantom is dragged back to this radius so targets do not
    wander out of reach. Only used when ``use_phantom=True``."""

    debug_print_period: int = 0
    """If > 0, print phantom/robot/target diagnostics for env 0 every N
    update_phantom calls. 0 disables. Use 50 for ~1 print/sec at dt=0.02."""

    phantom_yaw_track_robot: bool = False
    """If True, phantom_yaw is synced to robot.heading_w every dt (Option B —
    decouples translation enforcement from yaw control; phantom always points
    along the robot's current heading, so targets are always "in front of"
    the robot regardless of yaw drift). If False (Option A, default), phantom
    yaw evolves only via the commanded wz, so a yaw-drifting policy will see
    its phantom appear sideways, implicitly penalizing yaw drift via foot
    target misalignment. Use B for visual validation of phantom mechanics
    against drifting policies; use A for training (stronger signal)."""

    use_cost_selection: bool = False
    """When True, replace the snap-to-nearest-ray z lookup with a full
    cost-based foothold selection over a window around the Raibert prior.
    See ``foothold_selection.select_foothold_by_cost`` for the cost function.

    Default False — keeps the existing DTC env behavior bit-identical."""

    cost_window_m: float = 0.10
    """Half-side of the candidate window for cost-based selection, meters."""

    cost_alpha: float = 2.0
    """Cost weight on local roughness (max-min over window)."""

    cost_beta: float = 1.0
    """Cost weight on local slope (currently == roughness in the 1-ring approx)."""

    cost_gamma: float = 5.0
    """Cost weight on the obstacle flag (roughness > threshold)."""

    cost_delta: float = 1.0
    """Cost weight on distance to the Raibert prior (keeps planner close to
    velocity-pushed nominal)."""

    cost_obstacle_threshold: float = 0.15
    """Roughness threshold (m) above which a cell is flagged as obstacle.
    0.15 m corresponds to a typical stair tread height — anything higher than
    that in a 10cm window is treated as an unstep-onable edge."""

    hip_y: float = 0.10
    """Hardcoded |y| hip offset in body frame (Phase 0 approximation)."""

    leg_length: float = 0.78
    """Hardcoded body-frame z offset from base to neutral foot."""

    marker_cfg: VisualizationMarkersCfg = _DEFAULT_FOOTSTEP_MARKER_CFG
    """Debug visualization markers for planned and last-contact footsteps."""
