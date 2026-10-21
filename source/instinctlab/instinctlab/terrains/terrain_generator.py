from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGenerator

from .platform_course import platform_course_bounds

if TYPE_CHECKING:
    from .terrain_generator_cfg import FiledTerrainGeneratorCfg


class PlatformCourseTerrainGenerator(TerrainGenerator):
    """Sample spawn/goal patches on the actual first/last platform of each generated tile."""

    def __init__(self, cfg, device="cpu"):
        count = cfg.sub_terrains["square_gaps"].platform_count
        self.platform_bounds = np.zeros((cfg.num_rows, cfg.num_cols, count, 2), dtype=np.float32)
        super().__init__(cfg, device)

    def _get_terrain_mesh(self, difficulty, cfg):
        self._current_difficulty = difficulty
        return super()._get_terrain_mesh(difficulty, cfg)

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        if getattr(sub_terrain_cfg, "layout", None) == "platform_course":
            sub_terrain_cfg = sub_terrain_cfg.copy()
            bounds = platform_course_bounds(self._current_difficulty, sub_terrain_cfg, self.cfg.seed)
            self.platform_bounds[row, col] = bounds
            # IsaacLab retains flat_patches after releasing the generator. Store the
            # geometric endpoints there as positions, using its world-offset handling.
            for key, edge in (("platform_starts", 0), ("platform_ends", 1)):
                if key not in self.flat_patches:
                    self.flat_patches[key] = torch.zeros(
                        self.cfg.num_rows, self.cfg.num_cols, len(bounds), 3, device=self.device
                    )
                self.flat_patches[key][row, col, :, 0] = torch.as_tensor(bounds[:, edge], device=self.device)
            for key, index in (("init_pos", 0), ("target", -1)):
                center = float(bounds[index].mean())
                patch = sub_terrain_cfg.flat_patch_sampling[key]
                patch.x_range = (center - 0.025, center + 0.025)
                patch.y_range = (-0.025, 0.025)
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)


class FiledTerrainGenerator(TerrainGenerator):
    """A terrain generator that uses the filed generator."""

    def __init__(self, cfg: FiledTerrainGeneratorCfg, device: str = "cpu"):

        # Access the i-th row, j-th column subterrain config by
        # self._subterrain_specific_cfgs[i*num_cols + j]
        self._subterrain_specific_cfgs: list[SubTerrainBaseCfg] = []
        super().__init__(cfg, device)

    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg):
        """This function intercept the terrain mesh generation process and records the specific config
        for each subterrain.
        """
        mesh, origin = super()._get_terrain_mesh(difficulty, cfg)
        # >>> NOTE: This code snippet is copied from the super implementation because they copied the cfg
        # but we need to store the modified cfg for each subterrain.
        cfg = cfg.copy()
        # add other parameters to the sub-terrain configuration
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # <<< NOTE
        self._subterrain_specific_cfgs.append(cfg)  # since in super function, cfg is a copy of the original config.

        return mesh, origin

    @property
    def subterrain_specific_cfgs(self) -> list[SubTerrainBaseCfg]:
        """Get the specific configurations for all subterrains."""
        return self._subterrain_specific_cfgs.copy()  # Return a copy to avoid external modification.

    def get_subterrain_cfg(
        self, row_ids: int | torch.Tensor, col_ids: int | torch.Tensor
    ) -> list[SubTerrainBaseCfg] | SubTerrainBaseCfg | None:
        """Get the specific configuration for a subterrain by its row and column index."""
        num_cols = self.cfg.num_cols
        idx = row_ids * num_cols + col_ids
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy().tolist()  # Convert to list if it's a tensor.
            return [
                self._subterrain_specific_cfgs[i] if 0 <= i < len(self._subterrain_specific_cfgs) else None for i in idx
            ]
        if isinstance(idx, int):
            return self._subterrain_specific_cfgs[idx] if 0 <= idx < len(self._subterrain_specific_cfgs) else None
