"""A continuous ten-platform route with ordered progress and stable final arrival."""

import numpy as np
import torch

from .pose_velocity_command import PoseVelocityCommand


class GapCrossingCommand(PoseVelocityCommand):
    """Count success after visiting every platform and holding the final target for one second.

    A success is latched for the episode; the curriculum also rejects subsequent failures.
    This metric is limited to ``square_gaps`` and does not count rough-ground standing.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        terrain_cfg = self.terrain.cfg.terrain_generator
        names = list(terrain_cfg.sub_terrains)
        proportions = np.array([c.proportion for c in terrain_cfg.sub_terrains.values()])
        columns = np.searchsorted(
            np.cumsum(proportions / proportions.sum()), np.arange(terrain_cfg.num_cols) / terrain_cfg.num_cols + 0.001
        )
        gap_columns = torch.tensor(columns == names.index("square_gaps"), device=self.device)
        self.is_gap_env = gap_columns[self.terrain.terrain_types]
        course_cfg = terrain_cfg.sub_terrains["square_gaps"]
        self.platform_half_width = course_cfg.platform_width / 2
        origin_x = self.terrain.terrain_origins[:, :, None, 0]
        self.course_bounds = torch.stack(
            (
                self.terrain.flat_patches["platform_starts"][..., 0] - origin_x,
                self.terrain.flat_patches["platform_ends"][..., 0] - origin_x,
            ),
            dim=-1,
        )
        self.platform_count = course_cfg.platform_count
        self.gaps_crossed = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.feet_ids, _ = self.robot.find_bodies(".*_ankle_roll_link")
        self.contact = env.scene["contact_forces"]
        self.contact_ids, _ = self.contact.find_bodies(".*_ankle_roll_link")
        if len(self.feet_ids) != 2 or len(self.contact_ids) != 2:
            raise ValueError("Gap crossing requires two ankle-roll foot bodies and contact sensors.")
        self.goal_hold_time = torch.zeros(self.num_envs, device=self.device)
        self.metrics["crossing_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gaps_crossed"] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.goal_hold_time[env_ids] = 0.0
        self.metrics["crossing_success"][env_ids] = 0.0
        self.gaps_crossed[env_ids] = 0
        self.metrics["gaps_crossed"][env_ids] = 0.0

    def _update_metrics(self):
        super()._update_metrics()
        distance = torch.linalg.vector_norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=-1)
        bounds = self.course_bounds[self.terrain.terrain_levels, self.terrain.terrain_types]
        feet = self.robot.data.body_pos_w[:, self.feet_ids] - self.terrain.env_origins[:, None, :]
        contact = self.contact.data.net_forces_w[:, self.contact_ids, 2] > 1.0
        next_index = (self.gaps_crossed + 1).clamp(max=self.platform_count - 1)
        next_bounds = bounds[torch.arange(self.num_envs, device=self.device), next_index]
        within_width = (feet[:, :, 1].abs() < self.platform_half_width - 0.06).all(dim=-1)
        upright = self.robot.data.projected_gravity_b[:, 2] < -0.9
        elevated = (self.robot.data.root_pos_w[:, 2] - self.terrain.env_origins[:, 2]) > 0.5
        # Intermediate platforms need no stop. Both feet must have passed the gap,
        # with at least one foot supporting the robot on that platform, in order.
        reached_next = (
            self.is_gap_env
            & (self.gaps_crossed < self.platform_count - 1)
            & (feet[:, :, 0] > next_bounds[:, 0:1] + 0.05).all(dim=-1)
            & (feet[:, :, 0] < next_bounds[:, 1:2] - 0.05).all(dim=-1)
            & within_width
            & contact.any(dim=-1)
            & upright
            & elevated
        )
        self.gaps_crossed += reached_next.long()
        self.metrics["gaps_crossed"][:] = self.gaps_crossed.float()
        stable = (
            self.is_gap_env
            & (self.gaps_crossed == self.platform_count - 1)
            & (distance < self.cfg.target_dis_threshold)
            & (feet[:, :, 0] > bounds[:, -1, 0:1] + 0.15).all(dim=-1)
            & (feet[:, :, 0] < bounds[:, -1, 1:2] - 0.15).all(dim=-1)
            & within_width
            & contact.all(dim=-1)
            & upright
            & elevated
            & (torch.linalg.vector_norm(self.robot.data.root_lin_vel_w[:, :2], dim=-1) < 0.25)
        )
        self.goal_hold_time[:] = torch.where(stable, self.goal_hold_time + self._env.step_dt, 0.0)
        self.metrics["crossing_success"][:] = torch.maximum(
            self.metrics["crossing_success"], (self.goal_hold_time >= 1.0).float()
        )
