"""Check ten-platform dimensions, ordered progress, collision and a short G1 rollout.

Run with scripts/env.sh and --headless --livestream=0. Writes only review artifacts.
"""

import argparse
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path(".environment/platform-course-check"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import numpy as np
    import torch

    from instinct_rl.modules import build_actor_critic

    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    import instinctlab.tasks  # noqa: F401
    from instinctlab.tasks.parkour.mdp.commands.gap_crossing_command import GapCrossingCommand
    from instinctlab.tasks.parkour.mdp.curriculums import gap_crossing_levels
    from instinctlab.terrains.platform_course import platform_course_bounds
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

    args.output.mkdir(parents=True, exist_ok=True)
    task = "Instinct-Parkour-Target-Amp-G1-23dof-v0"
    cfg = parse_env_cfg(task, num_envs=20)
    cfg.sim.enable_scene_query_support = True
    terrain_cfg = cfg.scene.terrain.terrain_generator
    expected = {"perlin_rough", "perlin_rough_stand", "square_gaps"}
    assert set(terrain_cfg.sub_terrains) == set(cfg.commands.base_velocity.velocity_ranges) == expected
    assert cfg.commands.base_velocity.resampling_time_range[0] > cfg.episode_length_s
    assert cfg.commands.base_velocity.rel_standing_envs == 0
    assert cfg.commands.base_velocity.ranges.ang_vel_z == (0.0, 0.0)
    play = parse_env_cfg("Instinct-Parkour-Target-Amp-G1-23dof-Play-v0")
    assert play.episode_length_s == cfg.episode_length_s and play.terminations.root_height is not None

    measurements = []
    previews = []
    for seed in (0, 1, 2):
        for difficulty in (0.0, 0.5, 1.0):
            gap = terrain_cfg.sub_terrains["square_gaps"].copy()
            for field in ("size", "horizontal_scale", "vertical_scale", "slope_threshold"):
                setattr(gap, field, getattr(terrain_cfg, field))
            gap.seed = seed
            np.random.seed(seed)
            bounds = platform_course_bounds(difficulty, gap, seed)
            meshes, _ = gap.function(difficulty, gap)
            mesh = meshes[0]
            warp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device="cuda")
            # 1 mm rays away from the exact grid coordinates; select every platform top.
            x = torch.arange(-6.4995, 6.5, 0.001, device="cuda")
            rays = torch.stack((x + gap.size[0] / 2, torch.full_like(x, gap.size[1] / 2), torch.ones_like(x)), -1)
            directions = torch.zeros_like(rays)
            directions[:, 2] = -1
            heights = raycast_mesh(rays, directions, warp_mesh)[0][:, 2]
            assert torch.isfinite(heights).all()
            mask = (heights > -0.1).cpu().numpy()
            transitions = np.diff(np.concatenate(([False], mask, [False])).astype(int))
            starts = np.where(transitions == 1)[0]
            stops = np.where(transitions == -1)[0]
            assert len(starts) == 10, (seed, difficulty, len(starts))
            lengths = (stops - starts) * 0.001
            gaps = (starts[1:] - stops[:-1]) * 0.001
            assert np.allclose(lengths, 0.9, atol=0.002), lengths
            assert np.allclose(gaps, bounds[1:, 0] - bounds[:-1, 1], atol=0.002), gaps
            assert np.allclose(x[starts].cpu().numpy(), bounds[:, 0], atol=0.002)
            assert float(heights[heights > -0.1].std()) > 0.0001
            measured_widths = []
            for center in bounds.mean(axis=-1):
                y = torch.arange(-0.7995, 0.8, 0.001, device="cuda")
                rays = torch.stack(
                    (torch.full_like(y, float(center + gap.size[0] / 2)), y + gap.size[1] / 2, torch.ones_like(y)), -1
                )
                directions = torch.zeros_like(rays)
                directions[:, 2] = -1
                hits = raycast_mesh(rays, directions, warp_mesh)[0][:, 2]
                width = float((hits > -0.1).sum()) * 0.001
                assert abs(width - 0.6) < 0.002, width
                measured_widths.append(width)
            measurements.append(
                {
                    "seed": seed,
                    "difficulty": difficulty,
                    "lengths_m": lengths.tolist(),
                    "widths_m": measured_widths,
                    "gaps_m": gaps.tolist(),
                }
            )
            if seed == 0 and difficulty == 1.0:
                np.savez(
                    args.output / "course_profile.npz", x=x.cpu().numpy(), height=heights.cpu().numpy(), bounds=bounds
                )
    print("PLATFORM_COURSE_GEOMETRY_PASS " + json.dumps(measurements), flush=True)

    # Exercise the actual metric with synthetic states, independently of policy quality.
    metric = object.__new__(GapCrossingCommand)
    metric._env = SimpleNamespace(step_dt=0.02, max_episode_length=3000, num_envs=1, device="cpu")
    metric.cfg = cfg.commands.base_velocity
    metric.robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[1.9, 0.0, 0.8]]),
            root_lin_vel_w=torch.zeros(1, 3),
            root_lin_vel_b=torch.zeros(1, 3),
            root_ang_vel_b=torch.zeros(1, 3),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            body_pos_w=torch.tensor([[[1.8, 0.1, 0.0], [1.8, -0.1, 0.0]]]),
        )
    )
    metric.pos_command_w = torch.tensor([[1.9, 0.0, 0.0]])
    metric.vel_command_b = torch.zeros(1, 3)
    metric.terrain = SimpleNamespace(
        env_origins=torch.zeros(1, 3),
        terrain_levels=torch.zeros(1, dtype=torch.long),
        terrain_types=torch.zeros(1, dtype=torch.long),
    )
    metric.feet_ids = metric.contact_ids = [0, 1]
    metric.contact = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=torch.tensor([[[0.0, 0.0, 50.0], [0.0, 0.0, 50.0]]]))
    )
    metric.platform_half_width = 0.3
    metric.platform_count = 10
    metric.course_bounds = torch.tensor(bounds, dtype=torch.float32)[None, None]
    metric.gaps_crossed = torch.zeros(1, dtype=torch.long)
    metric.is_gap_env = torch.tensor([True])
    metric.goal_hold_time = torch.zeros(1)
    metric.metrics = {
        key: torch.zeros(1)
        for key in (
            "error_vel_xy",
            "error_vel_yaw",
            "tracking_exp_vel_xy",
            "tracking_exp_vel_yaw",
            "crossing_success",
            "gaps_crossed",
        )
    }

    def place_on(index):
        center = float(bounds[index].mean())
        metric.robot.data.root_pos_w[:, 0] = center
        metric.robot.data.body_pos_w[:, :, 0] = center

    metric.pos_command_w[:, 0] = float(bounds[-1].mean())
    place_on(9)
    for _ in range(60):
        metric._update_metrics()
    assert (
        metric.gaps_crossed.item() == 0 and metric.metrics["crossing_success"].item() == 0
    ), "Skipping platforms must not count"
    place_on(1)
    metric.contact.data.net_forces_w.zero_()
    for _ in range(60):
        metric._update_metrics()
    assert metric.gaps_crossed.item() == 0, "Airborne arrival must not count"
    metric.contact.data.net_forces_w[:, :, 2] = 50
    metric.robot.data.body_pos_w[0, 0, 0] = float(bounds[0].mean())
    metric._update_metrics()
    assert metric.gaps_crossed.item() == 0, "Both feet must pass the gap"
    for index in range(1, 10):
        place_on(index)
        metric._update_metrics()
        assert metric.gaps_crossed.item() == index
    assert metric.metrics["crossing_success"].item() == 0, "Final arrival needs a one-second hold"
    for _ in range(60):
        metric._update_metrics()
    assert metric.metrics["crossing_success"].item() == 1
    metric.is_gap_env[:] = False
    metric.metrics["crossing_success"].zero_()
    for _ in range(60):
        metric._update_metrics()
    assert metric.metrics["crossing_success"].item() == 0, "Standing terrain must not count"

    moves = []
    fake_command = SimpleNamespace(
        is_gap_env=torch.tensor([True, True, True, False]),
        metrics={
            "crossing_success": torch.tensor([0.0, 1.0, 1.0, 0.0]),
            "tracking_exp_vel_xy": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        },
    )
    fake_env = SimpleNamespace(
        scene=SimpleNamespace(
            terrain=SimpleNamespace(
                terrain_levels=torch.zeros(4),
                update_env_origins=lambda ids, up, down: moves.append((up.tolist(), down.tolist())),
            )
        ),
        command_manager=SimpleNamespace(get_term=lambda name: fake_command),
        termination_manager=SimpleNamespace(terminated=torch.tensor([False, False, True, False])),
        episode_length_buf=torch.ones(4),
    )
    gap_crossing_levels(fake_env, torch.arange(4))
    assert moves == [([False, True, False, True], [True, False, True, False])]

    # Keep the full training terrain so patch sampling covers every difficulty.
    cfg.curriculum.terrain_levels = None
    env = InstinctRlVecEnvWrapper(gym.make(task, cfg=cfg))
    base = env.unwrapped
    terrain = base.scene.terrain
    command = base.command_manager.get_term("base_velocity")
    import omni.physx

    query = omni.physx.get_physx_scene_query_interface()
    collision_checks = 0
    for index, origin in enumerate(terrain.env_origins.cpu().tolist()):
        if not command.is_gap_env[index]:
            continue
        course = command.course_bounds[terrain.terrain_levels[index], terrain.terrain_types[index]].cpu().numpy()
        for center in course.mean(axis=-1):
            hit = query.raycast_closest((origin[0] + float(center), origin[1] + 0.26, 0.1), (0.0, 0.0, -1.0), 2.0)
            assert hit["hit"] and "/World/ground/" in str(hit["collision"]) and hit["position"][2] > -0.05, hit
            collision_checks += 1
        for center in (course[:-1, 1] + course[1:, 0]) / 2:
            hit = query.raycast_closest((origin[0] + float(center), origin[1] + 0.26, 0.1), (0.0, 0.0, -1.0), 2.0)
            assert hit["hit"] and hit["position"][2] < -0.3, hit
            collision_checks += 1
    all_bounds = command.course_bounds
    gap_columns = all_bounds[:, :, -1, 1] > 0
    for patches_name, platform in (("init_pos", 0), ("target", -1)):
        positions = terrain.flat_patches[patches_name] - terrain.terrain_origins[:, :, None, :]
        centers = all_bounds[:, :, platform].mean(dim=-1)
        assert torch.all((positions[..., 0] - centers[:, :, None])[gap_columns].abs() < 0.026)
        assert torch.all(positions[..., 1][gap_columns].abs() < 0.026)
    with torch.inference_mode():
        base.reset(seed=42)
        obs, _ = env.get_observations()
        starts = base.scene["robot"].data.root_pos_w[:, 0] - terrain.env_origins[:, 0]
        goals = command.pos_command_w[:, 0] - terrain.env_origins[:, 0]
        current_bounds = command.course_bounds[terrain.terrain_levels, terrain.terrain_types]
        assert torch.all((starts - current_bounds[:, 0].mean(-1))[command.is_gap_env].abs() < 0.026)
        assert torch.all((goals - current_bounds[:, -1].mean(-1))[command.is_gap_env].abs() < 0.026)
        assert not command.is_standing_env.any()
        assert int(command.is_gap_env.sum()) == 14
        assert torch.all(command.command[command.random_velocity_indices] == 0)
        model_cfg = load_cfg_from_registry(task, "instinct_rl_cfg_entry_point").policy.to_dict()
        model_class = model_cfg.pop("class_name")
        model = build_actor_critic(model_class, model_cfg, env.get_obs_format(), env.num_actions, env.num_rewards)
        model.to(env.device).eval()
        state = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
        model.load_state_dict(state["model_state_dict"], strict=True)
        initial_goals = command.pos_command_w.clone()
        reset_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for _ in range(100):
            actions = model.act_inference(obs)
            obs, reward, done, _ = env.step(actions)
            reset_seen |= done.bool()
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all() and torch.isfinite(actions).all()
            torch.testing.assert_close(command.pos_command_w[~reset_seen], initial_goals[~reset_seen])
        base.reset()
        assert (
            not command.metrics["crossing_success"].any()
            and not command.goal_hold_time.any()
            and not command.gaps_crossed.any()
        )
    report = {
        "status": "PASSED",
        "terrain_names": sorted(expected),
        "course_geometry": measurements,
        "num_envs": env.num_envs,
        "physx_collision_checks": collision_checks,
        "rollout_steps": 100,
        "checkpoint": str(args.checkpoint),
        "note": "Configuration/geometry/smoke validation only; this is not a trained crossing success rate.",
    }
    (args.output / "validation.json").write_text(json.dumps(report, indent=2))
    print("PARKOUR_GAPS_CHECK " + json.dumps(report), flush=True)
    env.close()
except Exception:
    traceback.print_exc()
    os._exit(1)
finally:
    app.close()
