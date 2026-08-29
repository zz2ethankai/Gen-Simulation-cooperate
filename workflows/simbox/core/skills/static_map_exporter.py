"""Isaac Sim static-map exporter for the ROS-free local navigator."""

from __future__ import annotations

import math
import logging

import numpy as np
import omni.usd  # type: ignore[import-not-found]
from pxr import Gf, Usd, UsdGeom, UsdPhysics

try:
    from .navigation_geometry import StaticMap
except ImportError:
    from workflows.simbox.core.skills.navigation_geometry import StaticMap


LOGGER = logging.getLogger(__name__)


class IsaacStaticMapExporter:
    """Export Isaac scene collision geometry into a local occupancy map."""

    def __init__(self, workflow, robot, base_cfg: dict, map_cfg: dict | None = None):
        self.workflow = workflow
        self.robot = robot
        self.base_cfg = base_cfg

        local_navigation_cfg = self.base_cfg.get("local_navigation", {})
        if not isinstance(local_navigation_cfg, dict):
            raise TypeError("base_cfg['local_navigation'] must be a dict")
        configured_map_cfg = local_navigation_cfg.get("map", {})
        if not isinstance(configured_map_cfg, dict):
            raise TypeError("base_cfg['local_navigation']['map'] must be a dict")
        self.localization_cfg = dict(configured_map_cfg)
        if map_cfg is not None:
            if not isinstance(map_cfg, dict):
                raise TypeError("map_cfg must be a dict")
            self.localization_cfg.update(map_cfg)

        self._resolution = float(self.localization_cfg.get("map_resolution", 0.02))
        self._z_min = float(self.localization_cfg.get("map_z_min", 0.0))
        self._z_max = float(self.localization_cfg.get("map_z_max", 1.50))
        self._padding = float(self.localization_cfg.get("map_bounds_padding_m", 0.75))
        self._border_obstacle_thickness = float(self.localization_cfg.get("map_border_obstacle_thickness_m", 0.15))
        self._min_obstacle_height = float(self.localization_cfg.get("map_min_obstacle_height_m", 0.04))
        self.last_export_debug: dict[str, object] = {}
        if self._resolution <= 0.0:
            raise ValueError("localization.map_resolution must be positive")
        if self._z_max <= self._z_min:
            raise ValueError("localization.map_z_max must be greater than map_z_min")

    def export_map(self) -> StaticMap:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is unavailable, cannot export localization map")
        robot_path = self._require_robot_prim_path(stage)

        min_x, min_y, max_x, max_y = self._compute_bounds_xy(stage)
        min_x = math.floor(min_x / self._resolution) * self._resolution
        min_y = math.floor(min_y / self._resolution) * self._resolution
        max_x = math.ceil(max_x / self._resolution) * self._resolution
        max_y = math.ceil(max_y / self._resolution) * self._resolution

        width = int(round((max_x - min_x) / self._resolution)) + 1
        height = int(round((max_y - min_y) / self._resolution)) + 1
        if width <= 0 or height <= 0:
            raise RuntimeError("Occupancy map generator returned empty dimensions")

        occupancy = np.zeros((height, width), dtype=np.uint8)
        self._paint_map_border(occupancy=occupancy)
        usd_prim_count, _ = self._rasterize_static_colliders(
            stage=stage,
            occupancy=occupancy,
            min_x=min_x,
            min_y=min_y,
            robot_path=robot_path,
        )
        occupied_cell_count = int(np.count_nonzero(occupancy == 1))
        self.last_export_debug = {
            "shape": [int(height), int(width)],
            "resolution": float(self._resolution),
            "origin": [float(min_x), float(min_y), 0.0],
            "usd_collision_prim_count": int(usd_prim_count),
            "occupied_cell_count": int(occupied_cell_count),
            "robot_prim_path": robot_path,
        }
        LOGGER.info(
            "[local-navigation] static map shape=%s resolution=%.6f origin=%s "
            "usd_collision_prims=%d occupied_cells=%d robot=%s",
            self.last_export_debug["shape"],
            self._resolution,
            self.last_export_debug["origin"],
            usd_prim_count,
            occupied_cell_count,
            robot_path,
        )
        return StaticMap(
            occupancy=occupancy,
            resolution=self._resolution,
            origin=(float(min_x), float(min_y), 0.0),
        )

    def _compute_bounds_xy(self, stage) -> tuple[float, float, float, float]:
        root_path = str(getattr(self.workflow.task, "root_prim_path", "/World"))
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            raise RuntimeError(f"Task root prim '{root_path}' is invalid, cannot compute localization map bounds")

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox_cache.Clear()
        bounds = bbox_cache.ComputeWorldBound(root_prim)
        aligned = Gf.BBox3d(bounds.ComputeAlignedRange()).GetBox()
        min_point = aligned.GetMin()
        max_point = aligned.GetMax()

        min_x = float(min_point[0]) - self._padding
        min_y = float(min_point[1]) - self._padding
        max_x = float(max_point[0]) + self._padding
        max_y = float(max_point[1]) + self._padding
        if not all(math.isfinite(v) for v in (min_x, min_y, max_x, max_y)):
            raise RuntimeError("Localization map bounds are not finite")
        if max_x <= min_x or max_y <= min_y:
            raise RuntimeError("Localization map bounds are degenerate")
        return min_x, min_y, max_x, max_y

    def _paint_map_border(self, occupancy: np.ndarray):
        border_cells = max(1, int(math.ceil(self._border_obstacle_thickness / self._resolution)))
        occupancy[:border_cells, :] = 1
        occupancy[-border_cells:, :] = 1
        occupancy[:, :border_cells] = 1
        occupancy[:, -border_cells:] = 1

    def _require_robot_prim_path(self, stage) -> str:
        robot_path = str(getattr(self.robot, "robot_prim_path", "") or "").strip().rstrip("/")
        if not robot_path:
            raise RuntimeError(
                "Robot must expose a non-empty robot_prim_path for static-map self-collision exclusion"
            )
        robot_prim = stage.GetPrimAtPath(robot_path)
        if not robot_prim.IsValid():
            raise RuntimeError(
                f"Robot prim path '{robot_path}' is invalid; cannot exclude robot from static map"
            )
        return robot_path

    def _rasterize_static_colliders(
        self,
        stage,
        occupancy: np.ndarray,
        min_x: float,
        min_y: float,
        robot_path: str,
    ) -> tuple[int, int]:
        root_path = str(getattr(self.workflow.task, "root_prim_path", "/World"))
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            raise RuntimeError(f"Task root prim '{root_path}' is invalid, cannot rasterize static colliders")

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox_cache.Clear()

        usd_prim_count = 0
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsValid():
                continue

            prim_path = str(prim.GetPath())
            if prim_path == robot_path or prim_path.startswith(f"{robot_path}/"):
                continue
            has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
            if not has_collision:
                continue

            if has_collision:
                collision_enabled_attr = prim.GetAttribute("physics:collisionEnabled")
                if collision_enabled_attr.IsValid():
                    collision_enabled = collision_enabled_attr.Get()
                    if collision_enabled is False:
                        continue

            bounds = bbox_cache.ComputeWorldBound(prim)
            aligned = Gf.BBox3d(bounds.ComputeAlignedRange()).GetBox()
            min_point = aligned.GetMin()
            max_point = aligned.GetMax()
            bbox_min = (float(min_point[0]), float(min_point[1]), float(min_point[2]))
            bbox_max = (float(max_point[0]), float(max_point[1]), float(max_point[2]))
            if not self._should_rasterize_collider(bbox_min, bbox_max):
                continue

            self._paint_world_rect(
                occupancy=occupancy,
                min_x=min_x,
                min_y=min_y,
                rect_min_xy=(bbox_min[0], bbox_min[1]),
                rect_max_xy=(bbox_max[0], bbox_max[1]),
            )
            usd_prim_count += 1

        return usd_prim_count, int(np.count_nonzero(occupancy == 1))

    def _should_rasterize_collider(self, bbox_min: tuple[float, float, float], bbox_max: tuple[float, float, float]) -> bool:
        if not all(math.isfinite(v) for v in (*bbox_min, *bbox_max)):
            return False

        size_x = float(bbox_max[0] - bbox_min[0])
        size_y = float(bbox_max[1] - bbox_min[1])
        size_z = float(bbox_max[2] - bbox_min[2])
        if size_x <= 1.0e-4 or size_y <= 1.0e-4 or size_z <= 1.0e-4:
            return False
        if bbox_max[2] < self._z_min or bbox_min[2] > self._z_max:
            return False
        if size_z < self._min_obstacle_height:
            return False
        return True

    def _paint_world_rect(
        self,
        occupancy: np.ndarray,
        min_x: float,
        min_y: float,
        rect_min_xy: tuple[float, float],
        rect_max_xy: tuple[float, float],
    ):
        height, width = occupancy.shape[:2]
        min_col = int(math.floor((float(rect_min_xy[0]) - min_x) / self._resolution))
        max_col = int(math.ceil((float(rect_max_xy[0]) - min_x) / self._resolution))
        min_row = int(math.floor((float(rect_min_xy[1]) - min_y) / self._resolution))
        max_row = int(math.ceil((float(rect_max_xy[1]) - min_y) / self._resolution))

        min_col = max(0, min(width - 1, min_col))
        max_col = max(0, min(width - 1, max_col))
        min_row = max(0, min(height - 1, min_row))
        max_row = max(0, min(height - 1, max_row))
        if min_col > max_col or min_row > max_row:
            return

        occupancy[height - 1 - max_row : height - min_row, min_col : max_col + 1] = 1
