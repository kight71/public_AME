import gymnasium as gym
from ame_locomotion.tasks.manager_based.ame_locomotion import agents

gym.register(
    id="AME-G1-29DOF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="AME-G1-29DOF-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

# Phase 1 footstep-reward experiment (see PLAN.md): same env as
# AME-G1-29DOF-v0 but with footstep_swing_tracking + footstep_contact_phase
# reward weights turned on.
gym.register(
    id="AME-G1-29DOF-Footstep-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_Footstep",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

# unitree_rl_lab-style dense positive shaping: adds alive + foot_clearance,
# drops sparse termination_penalty, aligns joint_deviation_arms to their
# tested weight. feet_gait deliberately omitted to avoid clashing with
# our planner-driven footstep_contact_phase when stacking experiments.
# See G1RoughEnvCfg_Unitree docstring for the full motivation.
gym.register(
    id="AME-G1-29DOF-Unitree-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_Unitree",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

# DTC-style: Unitree base + footstep planner as the foot-shaping driver.
# See G1RoughEnvCfg_DTC docstring for the full design.
gym.register(
    id="AME-G1-29DOF-DTC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-Forward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_FORWARD",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-Forward-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_FORWARD_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-PlaceRew-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_PlaceRew",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-StepUp15-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_StepUp15_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-PyramidUp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_PyramidUp_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-HeadingPyramid-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_HeadingPyramid_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-PlaceRew-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_PlaceRew_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-LIPM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_LIPM",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-LIPM-PlaceRew-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_LIPM_PlaceRew",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-LIPM-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_LIPM_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-LIPM-PlaceRew-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_LIPM_PlaceRew_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-LIPM-PaperMLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_LIPM_PaperMLP",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-LIPM-PaperMLP-TerrainWindow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_LIPM_PaperMLP_TERRAIN_WINDOW",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-LIPM-PaperMLP-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_LIPM_PaperMLP_FLAT",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-LIPM-PaperMLP-FlatOmni-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_LIPM_PaperMLP_FLAT_OMNI",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-LIPM-PaperMLP-FlatXY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_LIPM_PaperMLP_FLAT_XY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-LIPM-PaperMLP-FlatYaw-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_LIPM_PaperMLP_FLAT_YAW",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-LIPM-PaperMLP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_LIPM_PaperMLP_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-LIPM-PaperMLP-FlatOmni-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_LIPM_PaperMLP_FLAT_OMNI_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-Legacy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_LEGACY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTC-PlannerV2-Legacy-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTC_PlannerV2_LEGACY_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-Exp-StepUp10-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1ExperimentStepUp10PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-Exp-StepUp20-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1ExperimentStepUp20PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-Exp-StepDown10-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1ExperimentStepDown10PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-Exp-StepDown20-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1ExperimentStepDown20PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-Exp-RandomBlocks-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1ExperimentRandomBlocksPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-Exp-PyramidUp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1ExperimentPyramidUpPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-Exp-PyramidDown-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1ExperimentPyramidDownPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-HeightMLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1HeightMlpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-HeightMLP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1HeightMlpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-HeightMLP-FlatVelGate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1HeightMlpFlatVelGateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-HeightMLP-FlatVelGateFast-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1HeightMlpFlatVelGateFastEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-HeightMLP-ZeroCmd-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1HeightMlpZeroCmdEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

# USD-based environments with only the foot collision replaced by STL mesh collision.
gym.register(
    id="AME-G1-29DOF-USD-FootSTL-HeightMLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1UsdFootStlHeightMlpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-USD-FootSTL-Forward-HeightMLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1UsdFootStlForwardHeightMlpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-USD-FootSTL-HeightMLP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1UsdFootStlHeightMlpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-USD-FootSTL-Forward-HeightMLP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1UsdFootStlForwardHeightMlpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

# URDF-based environments with improved foot collision geometry
gym.register(
    id="AME-G1-29DOF-URDF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1UrdfRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-URDF-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1UrdfRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-URDF-HeightMLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1UrdfHeightMlpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-URDF-HeightMLP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1UrdfHeightMlpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

# DTC-lite: cost-based planner + pure-MLP policy.
# See G1RoughEnvCfg_DTCLite docstring for the full design.
gym.register(
    id="AME-G1-29DOF-DTCLite-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTCLite",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1DTCLitePPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-DTCLite-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_DTCLite_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1DTCLitePPORunnerCfg",
    },
)

# BeamDojo: model-free foothold policy + sampling-based support reward
# (Wang et al. 2025, arXiv:2502.10363). No model-based planner; the
# policy chooses footholds and the reward scores actual stance quality.
# TODO: switch to G1BeamDojoPPORunnerCfg once ActorCriticDoubleCritic is wired.
gym.register(
    id="AME-G1-29DOF-MLP-BeamDojo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1MlpBeamDojoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-MLP-BeamDojo-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1MlpBeamDojoFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-MLP-BeamDojo-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1MlpBeamDojoV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-MLP-BeamDojo-Flat-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1MlpBeamDojoFlatV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-MLP-BeamDojo-Flat-Omni-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1MlpBeamDojoFlatOmniV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-MLP-BeamDojo-Flat-1mps-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1MlpBeamDojoFlat1mpsPlayV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-MLP-BeamDojo-Omni-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1MlpBeamDojoOmniV3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-MLP-BeamDojo-Flat-Omni-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1MlpBeamDojoFlatOmniV3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-MLP-BeamDojo-TerrainCurriculum-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1MlpBeamDojoTerrainCurriculumV3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-MLP-BeamDojo-Omni-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1MlpBeamDojoOmniV4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-MLP-BeamDojo-Flat-Omni-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1MlpBeamDojoFlatOmniV4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1TerrainMlpPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-BeamDojo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_BeamDojo",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-BeamDojo-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_BeamDojo_FLAT",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-BeamDojo-Smoke-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_BeamDojo_SMOKE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-BeamDojo-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_BeamDojo_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)
