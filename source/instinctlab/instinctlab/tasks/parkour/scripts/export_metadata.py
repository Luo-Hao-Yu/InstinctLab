"""Write the G1 23-DoF policy interface alongside its ONNX files."""

import hashlib
import json
import torch
from pathlib import Path


def export_g1_23dof_metadata(env, model_dir, checkpoint):
    from instinctlab.assets.unitree_g1_23dof import G1_23DOF_JOINT_NAMES, G1_23DOF_SDK_INDICES, G1_23DOF_URDF

    robot = env.unwrapped.scene["robot"]
    if robot.num_joints != 23 or set(robot.joint_names) != set(G1_23DOF_JOINT_NAMES):
        return
    action = env.unwrapped.action_manager.get_term("joint_pos")
    descriptor = action.IO_descriptor
    joint_names = list(descriptor.joint_names)
    robot_ids = [robot.joint_names.index(name) for name in joint_names]

    def values(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.ndim > 1:
                value = value[0]
            return value.tolist()
        return value

    def joint_values(value):
        return value[0, robot_ids].detach().cpu().tolist()

    obs_format = {
        group: {name: [int(size) for size in shape] for name, shape in segments.items()}
        for group, segments in env.get_obs_format().items()
    }
    meta = {
        "robot": "Unitree G1 23dof",
        "task": env.unwrapped.spec.id,
        "checkpoint": str(checkpoint),
        "urdf_sha256": hashlib.sha256(G1_23DOF_URDF.read_bytes()).hexdigest(),
        "root_body": "pelvis",
        "policy_dt_s": env.unwrapped.step_dt,
        "observation_joint_names": list(robot.joint_names),
        "observation_segments": obs_format["policy"],
        "observation_formats": obs_format,
        "observation_joint_pos_offset_rad": values(robot.data.default_joint_pos),
        "observation_base_ang_vel_scale": env.unwrapped.cfg.observations.policy.base_ang_vel.scale,
        "observation_joint_vel_scale": env.unwrapped.cfg.observations.policy.joint_vel.scale,
        "action_joint_names": joint_names,
        "action_sdk_motor_indices": [G1_23DOF_SDK_INDICES[G1_23DOF_JOINT_NAMES.index(name)] for name in joint_names],
        "action_formula": "joint_position_target_rad = action * action_scale + action_offset_rad",
        "action_scale": values(descriptor.scale),
        "action_offset_rad": values(descriptor.offset),
        "stiffness_nm_per_rad": joint_values(robot.data.default_joint_stiffness),
        "damping_nm_s_per_rad": joint_values(robot.data.default_joint_damping),
        "effort_limits_nm": joint_values(robot.data.joint_effort_limits),
        "velocity_limits_rad_s": joint_values(robot.data.joint_vel_limits),
        "joint_position_limits_rad": robot.data.soft_joint_pos_limits[0, robot_ids].detach().cpu().tolist(),
        "depth_encoder_input_shape": [1, *obs_format["policy"]["depth_image"]],
        "depth_encoder_output_shape": [1, 128],
        "actor_output_shape": [1, 23],
        "note": (
            "Observation/action order is PhysX order, not ascending SDK motor index. "
            "Depth input is the configured cropped, normalized image history, not a raw camera frame."
        ),
    }
    path = Path(model_dir) / "policy_metadata.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Exported G1 23-DoF policy metadata to {path}")
