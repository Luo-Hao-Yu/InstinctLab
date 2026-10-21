"""Parkour for the standard G1 23-DoF articulation (unmodified official feet)."""

import copy
from pathlib import Path

from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from instinctlab.assets.unitree_g1_23dof import G1_23DOF_ACTION_SCALE, G1_23DOF_CFG, G1_23DOF_LINKS, g1_23dof_symmetry
from instinctlab.motion_reference import MotionReferenceManager
from instinctlab.sensors import get_link_prim_targets
from instinctlab.tasks.parkour import mdp

from .g1_parkour_target_amp_cfg import G1ParkourRoughEnvCfg, G1ParkourRoughEnvCfg_PLAY, motion_reference_cfg


class G1_23DofMotionReferenceManager(MotionReferenceManager):
    def _initialize_data(self):
        super()._initialize_data()
        mapping, signs = g1_23dof_symmetry(list(self.isaac_joint_names))
        self.cfg.symmetric_augmentation_joint_mapping = mapping
        self.cfg.symmetric_augmentation_joint_reverse_buf = signs


def _configure_robot(cfg):
    cfg.scene.robot = G1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actions.joint_pos.scale = copy.deepcopy(G1_23DOF_ACTION_SCALE)
    cfg.scene.camera.mesh_prim_paths = ["/World/ground", *get_link_prim_targets(G1_23DOF_LINKS)]
    reference = copy.deepcopy(motion_reference_cfg)
    reference.class_type = G1_23DofMotionReferenceManager
    reference.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
    reference.reference_prim_path = "/World/envs/env_.*/RobotReference/pelvis"
    reference.robot_model_path = G1_23DOF_CFG.spawn.asset_path
    reference.symmetric_augmentation_joint_mapping = None  # resolved from the live articulation
    reference.symmetric_augmentation_joint_reverse_buf = None
    reference.link_of_interests = [
        name.replace("wrist_yaw_link", "wrist_roll_rubber_hand") for name in reference.link_of_interests
    ]
    motion = reference.motion_buffers["run_walk"]
    motion.path = str(Path(__file__).resolve().parents[7] / "datasets/parkour/g1_23dof_motion_reference")
    motion.filtered_motion_selection_filepath = str(Path(motion.path) / "parkour_motion_without_run.yaml")
    motion.clip_joint_ref_to_robot_limits = True
    cfg.scene.motion_reference = reference


@configclass
class G1Parkour23DofEnvCfg(G1ParkourRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_robot(self)
        self.scene.num_envs = 512


@configclass
class G1Parkour23DofEnvCfg_PLAY(G1ParkourRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _configure_robot(self)
        # Evaluate crossings with the same horizon and fall rules used in training.
        self.episode_length_s = 60.0
        self.terminations.root_height = TerminationTermCfg(
            func=mdp.root_height_below_env_origin_minimum, params={"minimum_height": 0.5}
        )
        self.scene.terrain.terrain_generator.num_rows = 10
        self.scene.terrain.max_init_terrain_level = 9
