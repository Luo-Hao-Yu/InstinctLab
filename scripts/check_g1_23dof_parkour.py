"""Validate G1 23-DoF joints, mirror FK, AMP observations and ONNX simulation steps."""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--steps", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1:
    parser.error("--steps must be positive")
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import numpy as np
    import torch

    from instinct_rl.runners import OnPolicyRunner
    from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size

    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    import instinctlab.tasks  # noqa: F401
    from instinctlab.assets.unitree_g1_23dof import G1_23DOF_JOINT_NAMES, G1_23DOF_SDK_INDICES
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab/tasks/parkour/scripts")
    )
    from onnxer import load_parkour_onnx_model

    train_task = "Instinct-Parkour-Target-Amp-G1-23dof-v0"
    play_task = "Instinct-Parkour-Target-Amp-G1-23dof-Play-v0"
    old_cfg = parse_env_cfg("Instinct-Parkour-Target-Amp-G1-v0")
    assert old_cfg.scene.num_envs == 4096
    assert "g1_29dof" in old_cfg.scene.robot.spawn.asset_path
    assert "g1_23dof_motion_reference" not in old_cfg.scene.motion_reference.motion_buffers["run_walk"].path
    train_cfg = parse_env_cfg(train_task)
    assert train_cfg.scene.num_envs == 512
    agent_cfg = load_cfg_from_registry(play_task, "instinct_rl_cfg_entry_point")
    assert agent_cfg.max_iterations == 30000 and agent_cfg.experiment_name == "g1_23dof_parkour"
    cfg = parse_env_cfg(play_task, num_envs=1)
    env = InstinctRlVecEnvWrapper(gym.make(play_task, cfg=cfg))
    robot = env.unwrapped.scene["robot"]
    reference = env.unwrapped.scene["motion_reference"]
    assert robot.num_joints == 23 and set(robot.joint_names) == set(G1_23DOF_JOINT_NAMES)
    assert robot.body_names[0] == "pelvis"
    assert list(reference.isaac_joint_names) == list(robot.joint_names)
    fmt = env.get_obs_format()
    assert int(get_subobs_size(fmt["amp_policy"])) == int(get_subobs_size(fmt["amp_reference"])) == 550
    assert int(get_subobs_size(fmt["policy"])) == 5232

    torch.manual_seed(42)
    q = robot.data.default_joint_pos[0:1].repeat(8, 1) + 0.1 * torch.randn(8, 23, device=env.device)
    mirrored = reference._symmetric_augment_joint_buffer(q.clone())
    torch.testing.assert_close(reference._symmetric_augment_joint_buffer(mirrored.clone()), q)
    original_fk = reference.target_link_pose_forward_kinematics(q)[..., :3]
    mirrored_fk = reference.target_link_pose_forward_kinematics(mirrored)[..., :3]
    expected_fk = original_fk[:, reference.cfg.symmetric_augmentation_link_mapping] * q.new_tensor([1, -1, 1])
    mirror_error = (mirrored_fk - expected_fk).abs().max().item()
    # The official left/right shoulder origins differ by 10 micrometers.
    assert mirror_error < 2e-5, mirror_error

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=env.device)
    segments = env.get_obs_segments()
    proprio_size = get_subobs_size(segments, [name for name in segments if name != "depth_image"])
    model_dir = args.checkpoint.parent / "exported"
    metadata = json.loads((model_dir / "policy_metadata.json").read_text())
    assert metadata["action_joint_names"] == list(robot.joint_names)
    assert metadata["action_sdk_motor_indices"] == [
        G1_23DOF_SDK_INDICES[G1_23DOF_JOINT_NAMES.index(name)] for name in robot.joint_names
    ]
    assert np.all(np.asarray(metadata["stiffness_nm_per_rad"]) > 0)
    onnx_policy = load_parkour_onnx_model(
        str(model_dir),
        lambda obs: get_subobs_by_components(obs, ["depth_image"], segments, temporal=True),
        segments["depth_image"],
        slice(0, proprio_size),
    )
    obs, _ = env.get_observations()
    max_error = 0.0
    with torch.inference_mode():
        for _ in range(args.steps):
            torch_actions = policy(obs)
            onnx_actions = onnx_policy(obs)
            torch.testing.assert_close(onnx_actions, torch_actions, atol=1e-4, rtol=1e-4)
            max_error = max(max_error, (onnx_actions - torch_actions).abs().max().item())
            obs, reward, _, _ = env.step(onnx_actions)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            assert torch.isfinite(reference.data.joint_pos).all()
    report = {
        "status": "PASSED",
        "joints": list(robot.joint_names),
        "num_actions": env.num_actions,
        "amp_dimensions": 550,
        "policy_observation_size": 5232,
        "simulation_steps": args.steps,
        "mirror_fk_max_error_m": mirror_error,
        "onnx_max_action_error": max_error,
        "sdk_motor_indices": metadata["action_sdk_motor_indices"],
        "29dof_config_unchanged": True,
    }
    print("G1_23DOF_PARKOUR_CHECK " + json.dumps(report), flush=True)
    env.close()
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)

sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
