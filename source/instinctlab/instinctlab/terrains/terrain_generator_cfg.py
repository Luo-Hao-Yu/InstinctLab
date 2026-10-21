from isaaclab.terrains import TerrainGeneratorCfg as TerrainGeneratorCfgBase
from isaaclab.utils import configclass

from .terrain_generator import FiledTerrainGenerator, PlatformCourseTerrainGenerator


@configclass
class FiledTerrainGeneratorCfg(TerrainGeneratorCfgBase):
    class_type: type = FiledTerrainGenerator


@configclass
class PlatformCourseTerrainGeneratorCfg(TerrainGeneratorCfgBase):
    class_type: type = PlatformCourseTerrainGenerator
