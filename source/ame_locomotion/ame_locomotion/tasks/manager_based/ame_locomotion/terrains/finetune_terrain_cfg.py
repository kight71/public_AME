# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

# import isaaclab.terrains as terrain_gen
import ame_locomotion.tasks.manager_based.ame_locomotion.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .loco_hf_terrains_cfg import *

FINETUNE_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=50.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hollow_stairs": terrain_gen.MeshHollowStairsTerrainCfg(
            proportion=0.25,
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
        ),
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.25, rail_height_range=(0.25, 0.05), rail_thickness_range=(0.1, 0.3), platform_width=2.0
        ),
    },
)
"""Rough terrains configuration."""
