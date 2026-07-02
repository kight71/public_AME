# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ame_locomotion.tasks.manager_based.ame_locomotion import mdp
# import isaaclab.terrains as terrain_gen
import ame_locomotion.tasks.manager_based.ame_locomotion.terrains as terrain_gen

FINETUNE = False

##
# Pre-defined configs
##
from ame_locomotion.tasks.manager_based.ame_locomotion.terrains.terrain_cfg import ROUGH_TERRAINS_CFG  # isort: skip
from ame_locomotion.tasks.manager_based.ame_locomotion.terrains.finetune_terrain_cfg import FINETUNE_ROUGH_TERRAINS_CFG
from ame_locomotion.tasks.manager_based.ame_locomotion.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from ame_locomotion.tasks.manager_based.ame_locomotion.assets.robots.unitree import UNITREE_G1_29DOF_USD_FOOT_STL_CFG as ROBOT_USD_FOOT_STL_CFG
from ame_locomotion.tasks.manager_based.ame_locomotion.assets.robots.unitree import UNITREE_G1_29DOF_URDF_CFG as ROBOT_URDF_CFG
##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FINETUNE_ROUGH_TERRAINS_CFG if FINETUNE else ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0]),  # 0.05m resolution, 1.6m x 1.0m, grid 33x21
        # pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),   # 0.1m resolution, 1.6m x 1.0m, grid 17x11
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

    # Phase 0 footstep planner — output is observed by the policy but does
    # NOT yet drive any reward term (see PLAN.md).
    #
    # IMPORTANT: raibert_factor=0.0 does NOT decouple the target from v_cmd.
    # The foothold XY is always forward-extrapolated by the commanded velocity
    # in `_commit_plan` (commands.py: `root_pos_kf += horizon * vel_w_xy_mid`),
    # i.e. the predicted landing hip is already moving at v_cmd. `raibert_factor`
    # only adds an *extra* push beyond that predicted hip:
    #
    #     target = predicted_hip(t_touchdown) + R(yaw)·hip_offset
    #              + raibert_factor · t_swing · v_cmd_w
    #
    #   factor=0  -> foot lands at predicted hip's foot-frame origin (directly
    #                under hip; stable "marching" gait, no active push-off)
    #   factor=0.5 -> foot lands ahead of hip by half a swing's worth of v_cmd
    #                (classic Raibert, produces push-off / braking couples)
    #
    # So factor=0 still keeps the plan chasing the velocity command; it just
    # stops the planner from *adding* push-off cues on top. We start at 0 to
    # isolate "does forcing the policy toward this signal speed up training?"
    # (see PLAN.md) before layering more aggressive Raibert factors (0.3 / 0.5)
    # that would push the policy toward dynamic push-off. If you raise it, you
    # are increasing the fore-aft foot displacement relative to hip, NOT
    # enabling velocity coupling.
    footstep_plan = mdp.FootstepPlanCommandCfg(
        asset_name="robot",
        foot_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        height_scanner_name="height_scanner",
        velocity_command_name="base_velocity",
        t_step=0.6,
        t_swing_fraction=0.5,
        raibert_k=0.05,
        raibert_factor=0.0,
        n_future_steps=2,
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-2.0, n_max=2.0))    # 1.5
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": True},
        )
        # NOTE (Phase 0): footstep_plan command is computed and visualized but
        # NOT added to policy observations here — adding a new obs term would
        # change the actor input shape and break loading `pretrained/ame1.pt`.
        # The planner observation triplet (footstep_plan_xy +
        # footstep_phase_info + footstep_local_heightscan) is wired in
        # `_FootstepPolicyCfg` / `_FootstepCriticCfg` (see G1RoughEnvCfg_Footstep
        # below), where they are inserted BEFORE height_scan so the AME encoder's
        # tail-slice (obs[:, -L*W*coord_dim:]) still picks up the terrain map.

        def __post_init__(self):
            self.enable_corruption = True   # Enable observation noise (default False)
            self.concatenate_terms = True   # Concatenate observation terms (default True)

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
        )

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),    # 0.8
            "dynamic_friction_range": (0.3, 1.0),   # 0.6
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),  # "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.25}
    )

    # -- penalties
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2, 
        weight=-0.05
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]), # All body parts except feet
        },
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        weight=-1.5e-7,
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-0.001
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
    )
    dof_torques_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.01,
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

    # -- style
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            # threshold matches the planner's per-foot swing duration
            # (t_step * t_swing_fraction = 0.6 * 0.5 = 0.3). The previous 0.6s
            # threshold expected double the swing time of the planner schedule
            # and biased the policy away from contact_phase compliance.
            "threshold": 0.3,
        },
    )
    feet_air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )    
    
    feet_too_near = RewTerm(
        func=mdp.feet_too_near,
        weight=-1.0,
        params={
            "threshold": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"), 
        },
    )
    # -- coordination (cross-body coordination)
    joint_coordination = RewTerm(
        func=mdp.joint_coordination_rel,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "coord_joints": [
                # Cross-side coordination: left leg forward swing with right arm forward swing
                ["left_hip_pitch_joint", "right_shoulder_pitch_joint"],
                ["right_hip_pitch_joint", "left_shoulder_pitch_joint"],
            ],
            "coord_signs": [
                [1.0, 1.0],  # Left hip and right shoulder move in the same direction
                [1.0, 1.0],  # Right hip and left shoulder move in the same direction
            ],
        },
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )

    # ---- Phase 1: footstep-plan tracking (see PLAN.md) -------------------
    # Dense reward on swing-foot trajectory vs. the planner's reference.
    # Off by default (weight=0.0); set to ~1.0 to enable the experiment.
    footstep_swing_tracking = RewTerm(
        func=mdp.footstep_swing_tracking,
        weight=0.0,
        params={
            "command_name": "footstep_plan",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "force_threshold": 1.0,
            "std": 0.08,
            "apex": 0.10,
        },
    )
    # Contact-phase consistency — prevents "skip the swing" gaming.
    footstep_contact_phase = RewTerm(
        func=mdp.footstep_contact_phase,
        weight=0.0,
        params={
            "command_name": "footstep_plan",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "force_threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", 
                body_names=[
                    "torso_link",
                    ".*_shoulder_.*_link",
                    ".*_hip_.*_link",
                    ".*_knee_link",
                    ".*_elbow_link",
                    "waist_.*_link",
                    "pelvis",
                ]
                # body_names=["torso_link"]
            ), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class MyViewerCfg(ViewerCfg):
    """Configuration for the viewer."""
    eye = (0.0, 0.0, 10.0)
    lookat = (0.0, 0.0, 0.0)

##
# Environment configuration
##


@configclass
class G1RoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # viewer: MyViewerCfg = MyViewerCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        if FINETUNE:
            self.commands.base_velocity.rel_heading_envs = 0.5
            self.events.reset_base.params= {
                "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                }
            }
            self.commands.base_velocity.ranges.heading = (0.0, 0.0)
            
            self.rewards.termination_penalty.weight = -200.0
            self.rewards.track_lin_vel_xy_exp.weight = 2.0
            self.rewards.track_ang_vel_z_exp.weight = 3.0
            self.rewards.ang_vel_xy_l2.weight = -0.05
            self.rewards.undesired_contacts.weight = -1.0
            self.rewards.dof_torques_l2.weight = -1.5e-7
            self.rewards.dof_acc_l2.weight = -1.25e-7
            self.rewards.dof_vel_l2.weight = -0.001
            self.rewards.dof_pos_limits.weight = -1.0
            self.rewards.dof_torques_limits.weight = -0.05
            self.rewards.action_rate_l2.weight = -0.05
            self.rewards.flat_orientation_l2.weight = -5.0
            self.rewards.feet_air_time.weight = 0.5
            self.rewards.feet_air_time_variance.weight = -2.0
            self.rewards.feet_slide.weight = -0.3
            self.rewards.feet_stumble.weight = -5.0
            self.rewards.feet_too_near.weight = -5.0
            self.rewards.joint_coordination.weight = -0.5
            self.rewards.joint_deviation_hip.weight = -0.1
            self.rewards.joint_deviation_arms.weight = -0.3
            self.rewards.joint_deviation_waists.weight = -1.0
        else:
            # Randomization
            self.events.push_robot = None
            self.events.add_base_mass = None
            self.events.base_com = None
            # Observations
            self.observations.policy.base_ang_vel.noise=None
            self.observations.policy.projected_gravity.noise=None
            self.observations.policy.velocity_commands.noise=None
            self.observations.policy.joint_pos.noise=None
            self.observations.policy.joint_vel.noise=None
            self.observations.policy.actions.noise=None
            self.observations.policy.height_scan.params["noise"]=False
            # Reward Weights
            self.rewards.termination_penalty.weight = -200
            self.rewards.track_lin_vel_xy_exp.weight = 2.0
            self.rewards.track_ang_vel_z_exp.weight = 3.0
            self.rewards.ang_vel_xy_l2.weight = -0.05
            self.rewards.undesired_contacts.weight = -1.0
            self.rewards.dof_torques_l2.weight = -1.5e-7
            self.rewards.dof_acc_l2.weight = -1.25e-7
            self.rewards.dof_vel_l2.weight = -0.001
            self.rewards.dof_pos_limits.weight = -1.0
            self.rewards.dof_torques_limits.weight = -0.01
            self.rewards.action_rate_l2.weight = -0.01
            self.rewards.flat_orientation_l2.weight = -2.0
            self.rewards.feet_air_time.weight = 0.25
            self.rewards.feet_air_time_variance.weight = -0.7
            self.rewards.feet_slide.weight = -0.1
            self.rewards.feet_stumble.weight = -2.0
            self.rewards.feet_too_near.weight = -1.0
            self.rewards.joint_coordination.weight = -0.2
            self.rewards.joint_deviation_hip.weight = -0.1
            self.rewards.joint_deviation_arms.weight = -0.3
            self.rewards.joint_deviation_waists.weight = -1.0


@configclass
class _FootstepPolicyCfg(ObsGroup):
    """Policy observation group for the Phase 1 footstep experiment.

    Same proprioceptive terms as ``ObservationsCfg.PolicyCfg`` but with the
    planner observation triplet (footstep_plan_xy + footstep_phase_info +
    footstep_local_heightscan) inserted BETWEEN proprio and height_scan.

    Why the reordering: ``ActorCriticEncoder._encode_terrain`` (see
    ``rsl_rl/rsl_rl/modules/actor_critic_encoder.py:161``) slices the
    terrain map from the TAIL of the observation
    (``obs[:, -L*W*coord_dim:]``). Appending the planner terms after
    ``height_scan`` (the default ``ObservationGroupCfg`` behavior) would
    push the map off the tail and corrupt the AME encoder's CNN input.

    Noise: all proprio terms run noise-free. This mirrors the Stage-1
    regime in ``G1RoughEnvCfg.__post_init__`` (the FINETUNE=False branch
    clears every noise param on the policy group), since the Footstep arm
    retrains from scratch on Stage-1 rough terrain.
    """

    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
    actions = ObsTerm(func=mdp.last_action)
    # ---- planner observation triplet (body-frame, dense) ----
    footstep_plan_xy = ObsTerm(
        func=mdp.footstep_plan,
        params={"command_name": "footstep_plan"},
    )
    footstep_phase_info = ObsTerm(
        func=mdp.footstep_phase_info,
        params={"command_name": "footstep_plan"},
    )
    footstep_local_heightscan = ObsTerm(
        func=mdp.footstep_local_heightscan,
        params={
            "command_name": "footstep_plan",
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "half_size_m": 0.10,
            "n_per_axis": 5,
        },
    )
    # ---- terrain map (MUST stay last for AME encoder tail-slice) ----
    height_scan = ObsTerm(
        func=mdp.elevation_map,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class _FootstepCriticCfg(ObsGroup):
    """Critic observation group for the Phase 1 footstep experiment.

    Mirrors ``_FootstepPolicyCfg`` but adds the privileged ``base_lin_vel``
    term (matching the base ``CriticCfg``), keeps all terms noise-free
    (critic obs is privileged), and keeps ``height_scan`` last for the AME
    encoder tail-slice.
    """

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
    actions = ObsTerm(func=mdp.last_action)
    footstep_plan_xy = ObsTerm(
        func=mdp.footstep_plan,
        params={"command_name": "footstep_plan"},
    )
    footstep_phase_info = ObsTerm(
        func=mdp.footstep_phase_info,
        params={"command_name": "footstep_plan"},
    )
    footstep_local_heightscan = ObsTerm(
        func=mdp.footstep_local_heightscan,
        params={
            "command_name": "footstep_plan",
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "half_size_m": 0.10,
            "n_per_axis": 5,
        },
    )
    height_scan = ObsTerm(
        func=mdp.elevation_map,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
    )


@configclass
class G1RoughEnvCfg_Footstep(G1RoughEnvCfg):
    """Phase 1 footstep experiment — planner BOTH observed AND rewarded.

    Differs from ``G1RoughEnvCfg`` in two ways:

    1. **Observations** — ``policy``/``critic`` are replaced with
       ``_FootstepPolicyCfg`` / ``_FootstepCriticCfg`` so the policy now
       sees the planner's "what / where / when" triplet (footstep_plan_xy,
       footstep_phase_info, footstep_local_heightscan). This breaks loading
       ``pretrained/ame1.pt`` (input shape changed) — the Footstep arm
       retrains from scratch. Note: ``raibert_factor=0.0`` is kept (target
       still forwards-extrapolates with v_cmd in ``_commit_plan``; factor
       only adds *extra* push-off beyond the predicted hip — see the
       comment on ``CommandsCfg.footstep_plan``).

    2. **Rewards** — ``footstep_swing_tracking`` (exp-form, vel-gated) and
       ``footstep_contact_phase`` are enabled with the same weights as the
       previous Footstep config, so this is a clean continuation of the
       A/B vs. ``AME-G1-29DOF-v0``:

        train.py --task AME-G1-29DOF-v0          --max_iterations 2000  (A)
        train.py --task AME-G1-29DOF-Footstep-v0 --max_iterations 2000  (B)
    """

    def __post_init__(self):
        super().__post_init__()
        # Swap in obs groups with the planner triplet inserted before
        # height_scan (see comment on _FootstepPolicyCfg). Done AFTER
        # super().__post_init__() so the base's Stage-1 noise-clearing on
        # the original groups does not collide.
        self.observations.policy = _FootstepPolicyCfg()
        self.observations.critic = _FootstepCriticCfg()
        # Phase 1 footstep reward weights (unchanged vs previous config).
        self.rewards.footstep_swing_tracking.weight = 1.0
        self.rewards.footstep_contact_phase.weight = 0.5


@configclass
class G1RoughEnvCfg_Unitree(G1RoughEnvCfg):
    """unitree_rl_lab-style reward shaping (dense positive shaping).

    Baseline (``G1RoughEnvCfg``) lacks dense per-step rewards for the actual
    gait shape — only ``feet_air_time_positive_biped`` (cap 0.075/step,
    gated on single-stance) provides positive signal for foot lift, while
    penalties accumulate to ~-1.0/step. Result: PPO finds a saddle where
    "stand still" dominates "try to walk".

    unitree_rl_lab's velocity_env_cfg uses heavier penalties but
    compensates with ~1.65/step of dense positive shaping that does NOT
    depend on already-walking: ``alive`` baseline + ``foot_clearance``
    (target 10cm) + ``feet_gait`` (phase-locked contact pattern). This
    config ports those three terms plus aligns ``joint_deviation_arms``
    to their tested weight, while keeping our terrain encoder + obs
    unchanged.

    Changes vs. ``G1RoughEnvCfg``:
      - ``joint_deviation_arms`` -0.3 → -0.1  (matches unitree_rl_lab)
      - ``track_ang_vel_z_exp`` 3.0 → 0.5  (avoid rotate-saddle on 3k-iter horizon)
      - ``feet_air_time`` threshold 0.3 → 0.6  (revert to original; no footstep_swing here)
      - + ``alive`` +0.15  (dense survival baseline)
      - + ``foot_clearance`` +1.0  (target 0.1m, std 0.05, tanh_mult 2.0)
      - ``termination_penalty`` -200 KEPT  (complements alive — hard cliff for
        catastrophic terminate events; pure alive only is too weak a
        per-step signal to prevent early random-policy face-plants).

    Note on consistency: we intentionally do NOT add ``feet_gait`` from
    unitree_rl_lab even though that function exists in our mdp. It would
    conflict with our planner-driven ``footstep_contact_phase`` (same goal
    — match prescribed contact rhythm — different clocks: fixed 0.8s vs
    planner-driven). When later stacking footstep rewards on top of this
    cfg, the two would fight per the consistency > density > functionality
    rule. ``footstep_contact_phase`` is the more flexible choice and is
    already wired (weight 0 here, enabled in ``G1RoughEnvCfg_Footstep``).
    """

    def __post_init__(self):
        super().__post_init__()
        # NOTE: previously zeroed termination_penalty here under the theory
        # that alive (+0.15) is a strictly better replacement. Empirically
        # (v3 run 2026-06-19) that broke training — episodes collapsed to
        # ~100 steps because the random early policy lost the hard "don't
        # fall" cliff and never accumulated enough alive reward to learn
        # standing. The two are COMPLEMENTARY (hard cliff at terminate
        # events + soft slope per step), not redundant. Keep both.
        # Arms get -0.3 in our base cfg; unitree_rl_lab uses -0.1 and walks fine.
        self.rewards.joint_deviation_arms.weight = -0.1
        # Base cfg's `track_ang_vel_z_exp` weight = 3.0 is the original AME
        # repo value (confirmed against upstream 2026-06-19), but on a short
        # 3k-iter horizon it dominates ~3/step — easily beating the ~1.2/step
        # max foot-lift reward stack (alive + foot_clearance + feet_air_time)
        # and trapping PPO in a "stand and rotate" saddle. The original
        # train recipe is 15k iter, which presumably escapes the saddle
        # eventually; we trade some yaw-tracking final performance for
        # faster walk acquisition by aligning to unitree_rl_lab's 0.5.
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        # Original AME repo uses feet_air_time threshold = 0.6. We had
        # lowered it to 0.3 in a prior session to align with the
        # footstep planner's t_swing schedule (t_step 0.6 × frac 0.5).
        # In Unitree cfg the footstep terms have weight 0 so that
        # consistency concern doesn't apply — revert to the original 0.6
        # threshold which encourages larger swing amplitude (the larger
        # threshold caps single-stance reward higher, rewarding longer
        # commits to a step rather than micro-shuffles).
        self.rewards.feet_air_time.params["threshold"] = 0.6
        # +0.15/step survival baseline — strong continuous gradient toward "don't fall".
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.15)
        # Dense foot-clearance shaping: rewards lifting foot to 10cm only when moving.
        # Standing still (xy_vel ≈ 0) keeps tanh ≈ 0 so this term ≈ 1 regardless of z.
        # Note: target_height 0.1m must stay aligned with planner apex if
        # footstep_swing_tracking is ever enabled on top of this cfg.
        self.rewards.foot_clearance = RewTerm(
            func=mdp.foot_clearance_reward,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
                "target_height": 0.1,
                "std": 0.05,
                "tanh_mult": 2.0,
            },
        )


@configclass
class G1RoughEnvCfg_DTC(G1RoughEnvCfg_Unitree):
    """DTC-style: footstep planner provides the only foot-shaping signal.

    Inherits ``G1RoughEnvCfg_Unitree``'s base-reward fixes (alive,
    termination_penalty -200, yaw 0.5, joint_deviation_arms -0.1) but
    hands the foot-shaping job entirely to the planner-driven terms:

      - ``footstep_swing_tracking`` stays as the dense planner guide:
        weight 2.0, std 0.15. At std=0.15
        a 10cm tracking error still gives 0.64 reward (vs. 0.21 at
        std=0.08), so the policy retains meaningful gradient even
        when off-target. The reward is gated by actual airborne state,
        so a swing foot that remains in contact cannot free-load near the
        phase endpoints.
      - ``footstep_contact_phase`` stays as a soft companion (weight 0.5):
        correct single-stance is rewarded, while mismatched contact state
        simply receives no contact-phase reward. This avoids discouraging
        early swing exploration with negative contact penalties.
      - ``foot_clearance`` and ``feet_air_time`` are zeroed: their
        responsibilities (z-height shaping, rhythm) are now fully
        covered by the planner-driven terms. Keeping them would
        dilute the swing_tracking signal and (for foot_clearance)
        clash on rough terrain (see PLAN.md Risks).

    Motivation: keep the footstep planner as dense guidance, but make its
    reward conditional on physically valid swing/contact semantics instead
    of relying on a very large unconditional tracking weight.
    """

    def __post_init__(self):
        super().__post_init__()
        # DTC focuses on forward stepping first; avoid mixing in backward
        # locomotion while debugging planner-driven foot rewards.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        # Make planned footsteps follow the desired velocity more explicitly:
        # at vx=1.0 m/s and t_swing=0.3s this adds ~9cm forward placement.
        self.commands.footstep_plan.raibert_factor = 0.3
        # The planner can track a turning body too well; keep yaw tracking
        # strong enough that "walk in circles" is not an easy local optimum.
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        # Dense planner guide, with contact gating inside the reward.
        self.rewards.footstep_swing_tracking.weight = 2.0
        self.rewards.footstep_swing_tracking.params["std"] = 0.15
        # Soft companion: reward correct phase, but don't penalize early exploration.
        self.rewards.footstep_contact_phase.weight = 0.5
        # Remove redundant shaping that would dilute swing_tracking's signal.
        self.rewards.foot_clearance.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0


@configclass
class G1RoughEnvCfg_DTC_FORWARD(G1RoughEnvCfg_DTC):
    """Forward-only DTC ablation for isolating straight-line footstep guidance.

    This removes backward, lateral, and yaw/heading commands so we can test
    whether planner-driven swing rewards can learn clean forward walking before
    adding turning back into the command distribution.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # Keep the same command-driven step push as dtc_v4; this isolates
        # whether raibert_factor=0.3 works for straight-line walking.
        self.commands.footstep_plan.raibert_factor = 0.3
        # Phantom: idealized command-following pelvis with 0.5m leash.
        # Solves the static-target exploit (foot oscillating in place
        # because the planner re-extrapolates from a non-moving root).
        self.commands.footstep_plan.use_phantom = True
        self.commands.footstep_plan.max_leash = 0.5
        # One-step plan during phantom-mode debugging (clean signal).
        self.commands.footstep_plan.n_future_steps = 1
        # B mode: phantom_yaw tracks robot_yaw each dt → foot targets always
        # in front of robot in body frame, physically reachable. Trade-off:
        # phantom no longer implicitly penalizes yaw drift (that was A mode's
        # role), so yaw control is fully delegated to track_ang_vel_z_exp,
        # which we bump in weight below to compensate.
        self.commands.footstep_plan.phantom_yaw_track_robot = True
        # Forward-only ablation: widen yaw-rate tracking so large early yaw
        # errors still provide gradient instead of saturating to zero.
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5
        # B mode delegates ALL yaw control to this term, so it must dominate
        # over track_lin_vel_xy_exp (weight 2.0) and footstep_swing_tracking
        # (weight 2.0). Otherwise PPO would learn to trade yaw for lin+swing
        # (the circle-walking failure mode of linyawstd05). Weight 3.0 puts
        # yaw at ~1.5x the per-term weight of competitors.
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        # Same issue on forward velocity: keep gradients alive when the early
        # policy is far from the commanded speed.
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        # Make correct left/right phase switching more valuable without adding
        # negative mismatch penalties that can suppress exploration.
        self.rewards.footstep_contact_phase.weight = 1.0


@configclass
class G1RoughEnvCfg_DTCLite(G1RoughEnvCfg_Unitree):
    """DTC-lite: cost-based foothold planner + pure-MLP policy (no AME/CNN+MHA).

    Differences from ``G1RoughEnvCfg_DTC``:
    - Planner uses cost-based foothold selection (Task 1) instead of nearest-z snap.
    - Policy observations drop the global ``height_scan`` term and add three
      planner-derived terms (foothold xyz, phase info, per-foothold local
      heightscan). Critic keeps a global ``height_scan`` for privileged value
      estimation.
    - Reward terms swap exp-form swing tracking for log-form (weight 6) and add
      ``planner_consistency`` (weight -20). ``track_lin_vel_xy_exp`` drops to 1.0
      so foothold tracking is the dominant task signal.
    - Runner cfg in the gym registration uses ``ActorCriticDTC`` (pure MLP).
    """

    def __post_init__(self):
        super().__post_init__()

        # ---- Planner: enable cost-based selection -----------------------
        self.commands.footstep_plan.use_cost_selection = True
        self.commands.footstep_plan.cost_window_m = 0.10
        self.commands.footstep_plan.cost_alpha = 2.0
        self.commands.footstep_plan.cost_beta = 1.0
        self.commands.footstep_plan.cost_gamma = 5.0
        self.commands.footstep_plan.cost_delta = 1.0
        self.commands.footstep_plan.cost_obstacle_threshold = 0.15
        self.commands.footstep_plan.n_future_steps = 2
        self.commands.footstep_plan.raibert_factor = 0.3
        # No phantom: cost selection already filters out unreachable cells,
        # and we want the policy to see real-robot foothold geometry.
        self.commands.footstep_plan.use_phantom = False

        # ---- Policy obs: drop global heightmap, add planner triplet -----
        self.observations.policy.height_scan = None  # remove from policy
        self.observations.policy.footstep_plan_xy = ObsTerm(
            func=mdp.footstep_plan,
            params={"command_name": "footstep_plan"},
        )
        self.observations.policy.footstep_phase_info = ObsTerm(
            func=mdp.footstep_phase_info,
            params={"command_name": "footstep_plan"},
        )
        self.observations.policy.footstep_local_heightscan = ObsTerm(
            func=mdp.footstep_local_heightscan,
            params={
                "command_name": "footstep_plan",
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "half_size_m": 0.10,
                "n_per_axis": 5,
            },
        )
        # Critic keeps global height_scan (privileged) plus the same planner terms.
        self.observations.critic.footstep_plan_xy = ObsTerm(
            func=mdp.footstep_plan,
            params={"command_name": "footstep_plan"},
        )
        self.observations.critic.footstep_phase_info = ObsTerm(
            func=mdp.footstep_phase_info,
            params={"command_name": "footstep_plan"},
        )
        self.observations.critic.footstep_local_heightscan = ObsTerm(
            func=mdp.footstep_local_heightscan,
            params={
                "command_name": "footstep_plan",
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "half_size_m": 0.10,
                "n_per_axis": 5,
            },
        )

        # ---- Rewards: DTC paper defaults --------------------------------
        # Foothold tracking is the dominant task signal.
        self.rewards.footstep_swing_tracking.weight = 0.0  # legacy exp-form off
        self.rewards.footstep_contact_phase.weight = 0.5    # soft companion
        self.rewards.footstep_swing_tracking_log = RewTerm(
            func=mdp.footstep_swing_tracking_log,
            weight=6.0,
            params={
                "command_name": "footstep_plan",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "force_threshold": 1.0,
                "eps": 1e-3,
                "apex": 0.10,
            },
        )
        # TODO(2026-06-29): DTC uses weight=20. Our cost selector should give
        # more stable plans than TAMOLS, so this may over-suppress. Watch the
        # planner_consistency channel in tensorboard — if it stays near 0 the
        # whole run, this is fine; if the planner is locked even when terrain
        # changes, drop to ~5.
        self.rewards.planner_consistency = RewTerm(
            func=mdp.planner_consistency,
            weight=-20.0,
            params={"command_name": "footstep_plan"},
        )
        # Drop velocity tracking weight so foothold is the primary objective.
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        # Keep yaw tracking strong (the planner has no yaw command of its own).
        self.rewards.track_ang_vel_z_exp.weight = 2.0


@configclass
class G1RoughEnvCfg_DTCLite_PLAY(G1RoughEnvCfg_DTCLite):
    """Play configuration for DTC-lite checkpoints, fixed forward command."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.scene.visualize_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/visualize_cam",
            update_period=0.1, height=480, width=640, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0,
                horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 3.0), rot=(0.707, 0.0, 0.707, 0.0), convention="world"),
        )

        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False

        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        }

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        
        # add visualization camera only for play
        self.scene.visualize_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/visualize_cam",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 3.), rot=(0.707, 0.0, 0.707, 0.0), convention="world"),
        )
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        self.events.reset_base.params= {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        }

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.size = (8.0, 8.0)
            self.scene.terrain.terrain_generator.sub_terrains =  {
                "hollow_stairs": terrain_gen.MeshHollowStairsTerrainCfg(
                    proportion=1.0,
                    border_width=1.0,
                    step_height_range=(0.10, 0.30),
                    step_width=0.3,
                    tread_depth=0.35,
                    tread_thickness=0.03,
                    stair_width=0.6,
                    support_post_width=0.05,
                    support_beam_width=0.05,
                    support_beam_height_above_previous_step=0.05,
                    platform_width=3.0,
                    top_platform_width=0.8,
                    num_steps=None,
                    origin_mode="front_base",
                ),
            }

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"]=False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1RoughEnvCfg_DTC_PLAY(G1RoughEnvCfg_DTC):
    """Play configuration for DTC checkpoints with a fixed forward command."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.scene.visualize_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/visualize_cam",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 3.0), rot=(0.707, 0.0, 0.707, 0.0), convention="world"),
        )

        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.size = (8.0, 8.0)
            self.scene.terrain.terrain_generator.sub_terrains = {
                "hollow_stairs": terrain_gen.MeshHollowStairsTerrainCfg(
                    proportion=1.0,
                    border_width=1.0,
                    step_height_range=(0.10, 0.30),
                    step_width=0.3,
                    tread_depth=0.35,
                    tread_thickness=0.03,
                    stair_width=0.6,
                    support_post_width=0.05,
                    support_beam_width=0.05,
                    support_beam_height_above_previous_step=0.05,
                    platform_width=3.0,
                    top_platform_width=0.8,
                    num_steps=None,
                    origin_mode="front_base",
                ),
            }

        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"] = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1RoughEnvCfg_DTC_FORWARD_PLAY(G1RoughEnvCfg_DTC_PLAY):
    """Play configuration for the forward-only DTC ablation."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.footstep_plan.raibert_factor = 0.3
        # Mirror the train-side DTC-Forward phantom settings so play uses the
        # same planner behavior. Inheritance chain is DTC_FORWARD_PLAY ->
        # DTC_PLAY -> DTC (NOT through DTC_FORWARD), so these must be set here
        # explicitly.
        self.commands.footstep_plan.use_phantom = True
        # Validated 2026-06-22: phantom mechanics (yaw sync, forward direction,
        # leash, target geometry) all confirmed correct via Option B + tight
        # leash debug pass. Reverting to train-side settings.
        self.commands.footstep_plan.max_leash = 0.5
        self.commands.footstep_plan.n_future_steps = 1
        # Disabled: debug print and Option B are validation-only.
        self.commands.footstep_plan.debug_print_period = 0
        self.commands.footstep_plan.phantom_yaw_track_robot = False


def _configure_single_env_experiment_play(env_cfg: G1RoughEnvCfg_PLAY, terrain_cfg, forward_speed: float = 0.6):
    """Shared deterministic play settings for screen-recorded terrain experiments."""

    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    env_cfg.episode_length_s = 60.0
    env_cfg.viewer = MyViewerCfg()
    env_cfg.viewer.eye = (3.6, -4.2, 2.2)
    env_cfg.viewer.lookat = (3.4, 0.0, 0.45)

    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 1
        env_cfg.scene.terrain.terrain_generator.num_cols = 1
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.scene.terrain.terrain_generator.size = (10.0, 4.0)
        env_cfg.scene.terrain.terrain_generator.border_width = 2.0
        env_cfg.scene.terrain.terrain_generator.use_cache = False
        env_cfg.scene.terrain.terrain_generator.sub_terrains = {"experiment_terrain": terrain_cfg}

    env_cfg.events.reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.events.physics_material = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None

    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (forward_speed, forward_speed)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.policy.height_scan.params["noise"] = False


@configclass
class G1ExperimentStepUp10PlayEnvCfg(G1RoughEnvCfg_PLAY):
    """Single 10 cm up-step play scene for screen recording."""

    def __post_init__(self):
        super().__post_init__()
        _configure_single_env_experiment_play(
            self,
            terrain_gen.MeshSingleStepTerrainCfg(
                proportion=1.0,
                step_height=0.10,
                direction="up",
                step_x=3.2,
                spawn_x=1.2,
            ),
            forward_speed=0.6,
        )


@configclass
class G1ExperimentStepUp20PlayEnvCfg(G1RoughEnvCfg_PLAY):
    """Single 20 cm up-step play scene for screen recording."""

    def __post_init__(self):
        super().__post_init__()
        _configure_single_env_experiment_play(
            self,
            terrain_gen.MeshSingleStepTerrainCfg(
                proportion=1.0,
                step_height=0.20,
                direction="up",
                step_x=3.2,
                spawn_x=1.2,
            ),
            forward_speed=0.55,
        )


@configclass
class G1ExperimentStepDown10PlayEnvCfg(G1RoughEnvCfg_PLAY):
    """Single 10 cm down-step play scene for screen recording."""

    def __post_init__(self):
        super().__post_init__()
        _configure_single_env_experiment_play(
            self,
            terrain_gen.MeshSingleStepTerrainCfg(
                proportion=1.0,
                step_height=0.10,
                direction="down",
                step_x=3.2,
                spawn_x=1.2,
            ),
            forward_speed=0.55,
        )


@configclass
class G1ExperimentStepDown20PlayEnvCfg(G1RoughEnvCfg_PLAY):
    """Single 20 cm down-step play scene for screen recording."""

    def __post_init__(self):
        super().__post_init__()
        _configure_single_env_experiment_play(
            self,
            terrain_gen.MeshSingleStepTerrainCfg(
                proportion=1.0,
                step_height=0.20,
                direction="down",
                step_x=3.2,
                spawn_x=1.2,
            ),
            forward_speed=0.5,
        )


@configclass
class G1ExperimentRandomBlocksPlayEnvCfg(G1RoughEnvCfg_PLAY):
    """Small +/-5 cm discrete-block field play scene for screen recording."""

    def __post_init__(self):
        super().__post_init__()
        _configure_single_env_experiment_play(
            self,
            terrain_gen.MeshRandomBlocksTerrainCfg(
                proportion=1.0,
                block_size=0.35,
                height_range=(-0.05, 0.05),
                flat_start_length=1.2,
                spawn_x=0.8,
                seed=23,
            ),
            forward_speed=0.55,
        )


@configclass
class G1ExperimentPyramidUpPlayEnvCfg(G1RoughEnvCfg_PLAY):
    """Training-style solid pyramid stairs, starting before the first stair ring."""

    def __post_init__(self):
        super().__post_init__()
        _configure_single_env_experiment_play(
            self,
            terrain_gen.MeshSolidPyramidStairsTerrainCfg(
                proportion=1.0,
                border_width=1.0,
                step_height=0.10,
                step_width=0.30,
                platform_width=3.0,
                origin_mode="front_base",
            ),
            forward_speed=0.55,
        )
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.size = (8.0, 8.0)
        self.viewer.eye = (3.3, -5.0, 2.4)
        self.viewer.lookat = (3.0, 0.0, 0.55)


@configclass
class G1ExperimentPyramidDownPlayEnvCfg(G1RoughEnvCfg_PLAY):
    """Training-style solid pyramid stairs, starting on the center top platform."""

    def __post_init__(self):
        super().__post_init__()
        _configure_single_env_experiment_play(
            self,
            terrain_gen.MeshSolidPyramidStairsTerrainCfg(
                proportion=1.0,
                border_width=1.0,
                step_height=0.10,
                step_width=0.30,
                platform_width=3.0,
                origin_mode="top_platform",
            ),
            forward_speed=0.45,
        )
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.size = (8.0, 8.0)
        self.viewer.eye = (5.0, -5.0, 2.8)
        self.viewer.lookat = (4.3, 0.0, 0.45)


def _configure_height_mlp_env(env_cfg: G1RoughEnvCfg):
    env_cfg.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.15, size=[1.65, 1.05])
    env_cfg.observations.policy.height_scan.func = mdp.height_samples
    env_cfg.observations.critic.height_scan.func = mdp.height_samples


def _configure_flat_velgate_diagnostic(env_cfg: G1RoughEnvCfg, lin_vel_x: tuple[float, float]):
    """Make a deterministic flat-ground reward/command diagnostic env."""

    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.curriculum.terrain_levels = None

    env_cfg.commands.footstep_plan = None
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = lin_vel_x
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

    env_cfg.events.physics_material = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.events.reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.policy.height_scan.params["noise"] = False
    env_cfg.observations.critic.height_scan.params["noise"] = False


@configclass
class G1HeightMlpEnvCfg(G1RoughEnvCfg):
    """Configuration for the height-only MLP locomotion baseline."""

    def __post_init__(self):
        super().__post_init__()
        _configure_height_mlp_env(self)


@configclass
class G1HeightMlpEnvCfg_PLAY(G1RoughEnvCfg_PLAY):
    """Play configuration for the height-only MLP locomotion baseline."""

    def __post_init__(self):
        super().__post_init__()
        _configure_height_mlp_env(self)


@configclass
class G1HeightMlpFlatVelGateEnvCfg(G1HeightMlpEnvCfg):
    """Flat-ground HeightMLP reward diagnostic with the initial slow command band."""

    def __post_init__(self):
        super().__post_init__()
        _configure_flat_velgate_diagnostic(self, lin_vel_x=(0.0, 0.3))


@configclass
class G1HeightMlpFlatVelGateFastEnvCfg(G1HeightMlpEnvCfg):
    """Second-stage flat-ground HeightMLP diagnostic after tracking reaches the threshold."""

    def __post_init__(self):
        super().__post_init__()
        _configure_flat_velgate_diagnostic(self, lin_vel_x=(0.3, 0.6))


@configclass
class G1HeightMlpZeroCmdEnvCfg_PLAY(G1HeightMlpEnvCfg_PLAY):
    """Play configuration for checking zero-command standing behavior."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Keep play deterministic: no observation corruption, no pushes.
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"] = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1UsdFootStlHeightMlpEnvCfg(G1HeightMlpEnvCfg):
    """HeightMLP training config using the stock USD robot with STL foot collision patch."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ROBOT_USD_FOOT_STL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class G1UsdFootStlForwardHeightMlpEnvCfg(G1UsdFootStlHeightMlpEnvCfg):
    """Comparable stage-1 config: stock USD + STL feet with the old forward-only command distribution."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class G1UsdFootStlHeightMlpEnvCfg_PLAY(G1HeightMlpEnvCfg_PLAY):
    """HeightMLP play config using the stock USD robot with STL foot collision patch."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ROBOT_USD_FOOT_STL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class G1UsdFootStlForwardHeightMlpEnvCfg_PLAY(G1UsdFootStlHeightMlpEnvCfg_PLAY):
    """Play config for the comparable old forward-only command distribution."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


# URDF-based environment configs with improved foot collision geometry.
# These replace the default USD robot with a URDF import that uses box
# collision (0.18m x 0.08m x 0.03m) instead of tiny spheres (r=5mm) for the feet.


@configclass
class G1UrdfRoughEnvCfg(G1RoughEnvCfg):
    """Training config using URDF-based robot with improved foot collision."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ROBOT_URDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class G1UrdfRoughEnvCfg_PLAY(G1RoughEnvCfg_PLAY):
    """Play config using URDF-based robot with improved foot collision."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ROBOT_URDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class G1UrdfHeightMlpEnvCfg(G1HeightMlpEnvCfg):
    """HeightMLP training config using URDF-based robot with improved foot collision."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ROBOT_URDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class G1UrdfHeightMlpEnvCfg_PLAY(G1HeightMlpEnvCfg_PLAY):
    """HeightMLP play config using URDF-based robot with improved foot collision."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ROBOT_URDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# =========================================================================
# BeamDojo env (Wang et al. 2025, arXiv:2502.10363, RSS 2025).
#
# Core deviation from the DTC / Footstep line: there is NO model-based
# footstep planner. The policy chooses footholds itself; the environment
# evaluates actual foot placement quality per stance step via a
# sampling-based foothold penalty (see mdp.rewards.foothold_penalty).
# Does NOT use the Raibert / LIP target / phantom / vel_gate apparatus
# assembled in the Footstep arm — those were the source of the
# swing-tracking-vs-velocity-tracking saddle we kept hitting. BeamDojo
# sidesteps that conflict by removing the reference target entirely.
#
# Configuration follows BeamDojo paper Table VII (Appendix VI-A):
#   - dense locomotion group + sparse foothold penalty (weight -1.0)
#   - no Unitree foot_clearance extra; no model-based planner
#   - alive (+0.15) + termination_penalty (-200) kept for early stability
# Double critic and two-stage soft/hard terrain remain follow-ups.
# =========================================================================

_BEAMDOJO_FOOTHOLD_PARAMS = {
    "sensor_cfg": SceneEntityCfg(
        "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    ),
    "height_scanner_name": "height_scanner",
    "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
    "foot_length": 0.18,
    "foot_width": 0.065,
    "n_long": 4,
    "n_lat": 3,
    "force_threshold": 1.0,
    "sole_z_offset": -0.035409145057201385,
}


def _enable_low_risk_randomization(env_cfg: G1RoughEnvCfg):
    env_cfg.events.add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    env_cfg.events.base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )


def _configure_forward_only_beamdojo(env_cfg: G1RoughEnvCfg):
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

    # Keep robustness randomization, but leave push recovery for a later stage.
    _enable_low_risk_randomization(env_cfg)
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None


def _configure_mlp_beamdojo_v2_rewards(env_cfg: G1RoughEnvCfg):
    r = env_cfg.rewards
    ankle_sensor = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
    ankle_bodies = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")

    r.track_lin_vel_xy_exp.weight = 2.0
    r.track_lin_vel_xy_exp.params = {"command_name": "base_velocity", "std": 0.5}
    r.base_height.weight = -1.0
    r.flat_orientation_l2.weight = -0.5
    r.feet_distance_y.weight = 0.0
    r.feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": ankle_sensor,
            "threshold": 0.3,
        },
    )
    r.feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward_gated,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": ankle_bodies,
            "sensor_cfg": ankle_sensor,
            "target_height": 0.08,
            "std": 0.05,
            "tanh_mult": 2.0,
        },
    )
    r.termination_penalty.weight = -200.0
    r.alive.weight = 0.0


@configclass
class G1RoughEnvCfg_BeamDojo(G1RoughEnvCfg):
    """BeamDojo env: paper Table VII rewards, no planner, sparse foothold penalty."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.footstep_plan = None
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["torso_link", "pelvis"],
        )

        r = self.rewards
        ankle_sensor = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        ankle_bodies = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")

        # --- Group 1: locomotion (BeamDojo Table VII) ---
        r.track_lin_vel_xy_exp.weight = 1.0
        r.track_lin_vel_xy_exp.params = {"command_name": "base_velocity", "std": 0.25}
        r.track_ang_vel_z_exp.weight = 1.0
        r.track_ang_vel_z_exp.params = {"command_name": "base_velocity", "std": 0.25}
        r.base_height = RewTerm(
            func=mdp.base_height_l2,
            weight=-10.0,
            params={
                "target_height": 0.725,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        r.flat_orientation_l2.weight = -2.0
        r.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
        r.ang_vel_xy_l2.weight = -0.05
        r.action_rate_l2.weight = -0.01
        r.action_smoothness = RewTerm(func=mdp.action_smoothness_l2, weight=-1.0e-3)
        r.stand_still = RewTerm(func=mdp.stand_still, weight=-0.05)
        r.dof_vel_l2.weight = -1.0e-4
        r.dof_acc_l2.weight = -2.5e-8
        r.dof_pos_limits.weight = -5.0
        r.dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-1.0e-3, params={"soft_ratio": 1.0})
        r.joint_power = RewTerm(func=mdp.joint_power, weight=-2.0e-5)
        r.feet_ground_parallel = RewTerm(
            func=mdp.feet_ground_parallel,
            weight=-0.02,
            params=dict(_BEAMDOJO_FOOTHOLD_PARAMS),
        )
        r.feet_distance_y = RewTerm(func=mdp.feet_distance_y, weight=0.5, params={"min_distance": 0.18})
        r.feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": ankle_sensor,
                "threshold": 0.5,
            },
        )
        r.feet_clearance = RewTerm(
            func=mdp.feet_height_body,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": ankle_bodies,
                "target_height": 0.1,
                "tanh_mult": 2.0,
            },
        )

        # --- Group 2: sparse foothold ---
        r.foothold = RewTerm(
            func=mdp.foothold_penalty,
            weight=-1.0,
            params={**_BEAMDOJO_FOOTHOLD_PARAMS, "height_epsilon": -0.1},
        )

        # --- Early stability (complements paper rewards; reduces face-plant churn) ---
        r.termination_penalty.weight = 0.0
        r.alive = RewTerm(func=mdp.is_alive, weight=0.15)

        # --- Disable non-paper / legacy terms ---
        r.undesired_contacts.weight = 0.0
        r.dof_torques_l2.weight = 0.0
        r.dof_torques_limits.weight = 0.0
        r.feet_air_time_variance.weight = 0.0
        r.feet_slide.weight = 0.0
        r.feet_stumble.weight = 0.0
        r.feet_too_near.weight = 0.0
        r.joint_coordination.weight = 0.0
        r.joint_deviation_hip.weight = 0.0
        r.joint_deviation_arms.weight = 0.0
        r.joint_deviation_waists.weight = 0.0
        r.footstep_swing_tracking.weight = 0.0
        r.footstep_contact_phase.weight = 0.0


@configclass
class G1MlpBeamDojoEnvCfg(G1RoughEnvCfg_BeamDojo):
    """Mainline HeightMLP + BeamDojo reward config on rough terrain curriculum."""

    def __post_init__(self):
        super().__post_init__()

        _configure_height_mlp_env(self)
        _configure_forward_only_beamdojo(self)


@configclass
class G1MlpBeamDojoFlatEnvCfg(G1MlpBeamDojoEnvCfg):
    """Flat-ground first-stage config for the HeightMLP + BeamDojo mainline."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None

        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


@configclass
class G1MlpBeamDojoV2EnvCfg(G1MlpBeamDojoEnvCfg):
    """V2 mainline: make stepping positive and remove standing-reward shortcuts."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        _configure_mlp_beamdojo_v2_rewards(self)


@configclass
class G1MlpBeamDojoFlatV2EnvCfg(G1MlpBeamDojoFlatEnvCfg):
    """Flat-ground V2 for learning forward stepping before rough-terrain curriculum."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.8)
        _configure_mlp_beamdojo_v2_rewards(self)


@configclass
class G1RoughEnvCfg_BeamDojo_FLAT(G1RoughEnvCfg_BeamDojo):
    """Flat-ground BeamDojo diagnostic config.

    This keeps the BeamDojo reward stack but removes terrain difficulty and
    early reset randomness so we can tell whether the policy/reward setup can
    learn basic locomotion before adding foothold terrain.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None

        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"] = False


@configclass
class G1RoughEnvCfg_BeamDojo_SMOKE(G1RoughEnvCfg_BeamDojo):
    """Low-memory BeamDojo config for random-agent and short smoke tests.

    Isaac/PhysX allocates several GPU contact-pair buffers up front. The
    training cfg keeps generous buffers, but smoke tests only need a tiny
    scene to verify env construction and reward shapes.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10.0

        # Keep the smoke terrain small so PhysX does not preallocate contact
        # buffers sized for a full curriculum field.
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.size = (4.0, 4.0)

        # Low-memory PhysX GPU buffers for smoke only. These are intentionally
        # below the training defaults; use AME-G1-29DOF-BeamDojo-v0 for real
        # training runs.
        self.sim.physx.gpu_max_rigid_contact_count = 2**20
        self.sim.physx.gpu_max_rigid_patch_count = 2**14
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**18
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**21
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
        self.sim.physx.gpu_collision_stack_size = 2**23
        self.sim.physx.gpu_heap_capacity = 2**23
        self.sim.physx.gpu_temp_buffer_capacity = 2**22


@configclass
class G1RoughEnvCfg_BeamDojo_PLAY(G1RoughEnvCfg_BeamDojo):
    """Play configuration for BeamDojo checkpoints (forward-only)."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.scene.visualize_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/visualize_cam",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 3.), rot=(0.707, 0.0, 0.707, 0.0), convention="world"),
        )

        self.scene.terrain.max_init_terrain_level = None

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }

        # Reduce terrain to a single patch for visual play.
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.size = (8.0, 8.0)

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Disable observation corruption and external perturbations.
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"] = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
