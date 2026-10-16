"""Standard G1 23-DoF model, with native pelvis root and SDK joint signs."""

import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .unitree_g1 import (
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_delayed_actuators,
)

G1_23DOF_URDF = Path(__file__).parent / "resources/unitree_g1/g1_23dof.urdf"
_model = ET.parse(G1_23DOF_URDF).getroot()
_joints = [joint for joint in _model.findall("joint") if joint.get("type") == "revolute"]
# Official URDF declaration order is SDK order with the six absent slots omitted.
G1_23DOF_JOINT_NAMES = [joint.get("name") for joint in _joints]
G1_23DOF_SDK_INDICES = [*range(13), *range(15, 20), *range(22, 27)]
G1_23DOF_LINKS = ["pelvis", *(joint.find("child").get("link") for joint in _joints)]
assert len(G1_23DOF_JOINT_NAMES) == len(set(G1_23DOF_JOINT_NAMES)) == 23


def g1_23dof_symmetry(joint_names):
    """Resolve the mirror permutation in the actual PhysX order, never SDK order."""
    if len(joint_names) != 23 or set(joint_names) != set(G1_23DOF_JOINT_NAMES):
        raise ValueError(f"Unexpected G1 23-DoF articulation joints: {joint_names}")
    mapping, signs = [], []
    for name in joint_names:
        mirror = name.replace("left_", "right_", 1) if name.startswith("left_") else name.replace("right_", "left_", 1)
        mapping.append(joint_names.index(mirror))
        signs.append(-1 if "_roll_" in name or "_yaw_" in name else 1)
    return mapping, signs


def _actuators():
    """Reuse Parkour gains/delays, remove absent motors and use official motor limits."""
    actuators = {}
    for name, original in beyondmimic_g1_29dof_delayed_actuators.items():
        cfg = copy.deepcopy(original)
        cfg.joint_names_expr = [
            expr for expr in cfg.joint_names_expr if any(re.fullmatch(expr, joint) for joint in G1_23DOF_JOINT_NAMES)
        ]
        if not cfg.joint_names_expr:
            continue
        for field in ("stiffness", "damping", "armature"):
            value = getattr(cfg, field)
            if isinstance(value, dict):
                setattr(
                    cfg,
                    field,
                    {
                        expr: v
                        for expr, v in value.items()
                        if any(re.fullmatch(expr, joint) for joint in G1_23DOF_JOINT_NAMES)
                    },
                )
        selected = [
            joint for joint in _joints if any(re.fullmatch(expr, joint.get("name")) for expr in cfg.joint_names_expr)
        ]
        cfg.effort_limit_sim = {joint.get("name"): float(joint.find("limit").get("effort")) for joint in selected}
        cfg.velocity_limit_sim = {joint.get("name"): float(joint.find("limit").get("velocity")) for joint in selected}
        actuators[name] = cfg
    return actuators


G1_23DOF_CFG = copy.deepcopy(G1_29DOF_TORSOBASE_POPSICLE_CFG)
G1_23DOF_CFG.spawn.asset_path = str(G1_23DOF_URDF)
G1_23DOF_CFG.spawn.root_link_name = "pelvis"
G1_23DOF_CFG.spawn.merge_fixed_joints = True
G1_23DOF_CFG.spawn.self_collision = True
G1_23DOF_CFG.init_state.pos = (0.0, 0.0, 0.8)
G1_23DOF_CFG.actuators = _actuators()
G1_23DOF_ACTION_SCALE = {
    expr: value
    for expr, value in beyondmimic_action_scale.items()
    if any(re.fullmatch(expr, joint) for joint in G1_23DOF_JOINT_NAMES)
}
