"""Lazy CuRobo private-type import boundary.

This module contains no planning logic or compatibility API.  It exists only
because Physics-schema discovery is exercised without CuRobo installed.
"""

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.types import Cuboid, Mesh, SceneCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.usd_scene_parser import UsdSceneParser

__all__ = ["Cuboid", "Mesh", "RobotCfg", "SceneCfg", "ToolPoseCriteria", "UsdSceneParser"]
