"""Project the bundled torso-root G1 29-DoF reference onto the official pelvis-root G1 23-DoF model.

Run with Isaac Sim's Python, but no SimulationApp or GPU is needed. Root poses
are transformed with the source URDF; cached link poses/velocities are not reused.
This is a kinematic projection, not an optimization of the original human motion.
"""

import argparse
import hashlib
import json
import numpy as np
import torch
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path

import pytorch_kinematics as pk
from pytorch_kinematics.transforms import matrix_to_quaternion, quaternion_to_matrix

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "source/instinctlab/instinctlab/assets/resources/unitree_g1"
DELETED_JOINTS = {
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform_points(position, rotation, local):
    return position + torch.bmm(rotation, local.unsqueeze(-1)).squeeze(-1)


def convert(input_dir, output_dir, overwrite=False):
    source_urdf = ASSETS / "urdf/g1_29dof_torsobase_popsicle.urdf"
    target_urdf = ASSETS / "g1_23dof.urdf"
    source_chain = pk.build_chain_from_urdf(source_urdf.read_text()).to(dtype=torch.float64, device="cpu")
    target_chain = pk.build_chain_from_urdf(target_urdf.read_text()).to(dtype=torch.float64, device="cpu")
    source_names = source_chain.get_joint_parameter_names()
    target_names = [
        joint.get("name")
        for joint in ET.parse(target_urdf).getroot().findall("joint")
        if joint.get("type") == "revolute"
    ]
    if set(source_names) - set(target_names) != DELETED_JOINTS or len(target_names) != 23:
        raise ValueError("The source/target models do not describe the expected standard G1 29-to-23 conversion")
    selection_path = input_dir / "parkour_motion_without_run.yaml"
    selection = yaml.safe_load(selection_path.read_text())
    outputs = [output_dir / relative for relative in selection["selected_files"]]
    for path in outputs:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to replace {path}; pass --overwrite to regenerate it")
        if not path.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError(f"Motion path must remain inside the output directory: {path}")

    records = []
    for relative, output in zip(selection["selected_files"], outputs):
        source = input_dir / relative
        if not source.resolve().is_relative_to(input_dir.resolve()):
            raise ValueError(f"Motion path must remain inside the input directory: {source}")
        with np.load(source, allow_pickle=False) as raw:
            names = raw["joint_names"].tolist()
            positions = np.asarray(raw["joint_pos"], dtype=np.float64)
            base_pos = torch.as_tensor(np.array(raw["base_pos_w"]), dtype=torch.float64)
            base_quat = torch.as_tensor(np.array(raw["base_quat_w"]), dtype=torch.float64)
            framerate = float(raw["framerate"])
        if positions.ndim != 2 or positions.shape[1] != len(names) or len(set(names)) != len(names):
            raise ValueError(f"Invalid joint names/positions in {source}")
        count = positions.shape[0]
        if base_pos.shape != (count, 3) or base_quat.shape != (count, 4) or count < 2 or framerate <= 0:
            raise ValueError(f"Invalid root pose or framerate in {source}")
        if not all(np.isfinite(value).all() for value in (positions, base_pos.numpy(), base_quat.numpy())):
            raise ValueError(f"Nonfinite motion values in {source}")
        if not torch.allclose(base_quat.norm(dim=-1), torch.ones(count, dtype=torch.float64), atol=1e-3, rtol=0):
            raise ValueError(f"Root quaternions are not normalized in {source}")
        source_q = torch.as_tensor(positions[:, [names.index(name) for name in source_names]])
        source_fk = source_chain.forward_kinematics(source_q)
        pelvis_local = source_fk["pelvis"].get_matrix()
        old_rotation = quaternion_to_matrix(base_quat)
        new_pos = transform_points(base_pos, old_rotation, pelvis_local[:, :3, 3])
        new_rotation = old_rotation @ pelvis_local[:, :3, :3]
        new_quat = matrix_to_quaternion(new_rotation)
        new_quat /= new_quat.norm(dim=-1, keepdim=True)
        target_q = positions[:, [names.index(name) for name in target_names]].copy()
        # The original torso-root URDF reverses the waist edge but keeps its +Z
        # axis. The official pelvis-root joint uses the opposite angle convention.
        target_q[:, target_names.index("waist_yaw_joint")] *= -1.0

        # Both models have the same leg geometry: projection must preserve foot
        # trajectories in world space despite the changed root frame.
        target_order = target_chain.get_joint_parameter_names()
        target_fk = target_chain.forward_kinematics(
            torch.as_tensor(target_q[:, [target_names.index(name) for name in target_order]])
        )
        foot_errors = {}
        for foot in ("left_ankle_roll_link", "right_ankle_roll_link"):
            original = transform_points(base_pos, old_rotation, source_fk[foot].get_matrix()[:, :3, 3])
            projected = transform_points(new_pos, new_rotation, target_fk[foot].get_matrix()[:, :3, 3])
            error = (original - projected).norm(dim=-1).max().item()
            if error > 1e-5:
                raise ValueError(f"World foot trajectory changed unexpectedly: {foot}: {error} m")
            foot_errors[foot] = error
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            framerate=np.float32(framerate),
            joint_names=np.asarray(target_names),
            joint_pos=target_q.astype(np.float32),
            base_pos_w=new_pos.numpy().astype(np.float32),
            base_quat_w=new_quat.numpy().astype(np.float32),
        )
        records.append(
            {
                "source": str(source),
                "source_sha256": sha256(source),
                "output": str(output),
                "sha256": sha256(output),
                "frames": count,
                "joints": 23,
                "framerate": framerate,
                "max_world_foot_position_error_m": foot_errors,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / selection_path.name).write_text(yaml.safe_dump(selection, sort_keys=False))
    report = {
        "source_model_sha256": sha256(source_urdf),
        "target_model_sha256": sha256(target_urdf),
        "source_root": "torso_link",
        "target_root": "pelvis",
        "quaternion_order": "wxyz",
        "removed_joints": sorted(DELETED_JOINTS),
        "negated_joints": ["waist_yaw_joint"],
        "method": "project joints, transform root with source FK; recompute target FK/velocities at load",
        "joint_names": target_names,
        "motions": records,
    }
    (output_dir / "CONVERSION.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "datasets/parkour/parkour_motion_reference")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets/parkour/g1_23dof_motion_reference")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("Input and output directories must be different")
    torch.set_num_threads(2)
    with torch.inference_mode():
        convert(args.input_dir, args.output_dir, args.overwrite)
