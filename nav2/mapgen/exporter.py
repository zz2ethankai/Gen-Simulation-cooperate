"""Isaac Sim static-map exporter for Nav2."""

from __future__ import annotations

import math
import os
import warnings

import cv2
import numpy as np
import omni.usd  # type: ignore[import-not-found]
from pxr import Gf, Usd, UsdGeom, UsdPhysics


class IsaacStaticMapExporter:
    """Export Isaac scene collision geometry into a Nav2-compatible static map."""

    def __init__(self, workflow, robot, base_cfg: dict, scene_name: str = "scene"):
        self.workflow = workflow
        self.robot = robot
        self.base_cfg = base_cfg
        self.scene_name = str(scene_name or "scene")

        self.ros_cfg = self.base_cfg.get("ros", {})
        self.localization_cfg = self.ros_cfg.get("localization", {})
        if not isinstance(self.localization_cfg, dict):
            raise TypeError("base_cfg['ros']['localization'] must be a dict when present")

        self._resolution = float(self.localization_cfg.get("map_resolution", 0.02))
        self._z_min = float(self.localization_cfg.get("map_z_min", 0.0))
        self._z_max = float(self.localization_cfg.get("map_z_max", 1.50))
        self._padding = float(self.localization_cfg.get("map_bounds_padding_m", 0.75))
        self._robot_clear_radius = float(self.localization_cfg.get("robot_clear_radius_m", 0.70))
        self._robot_clear_footprint_points = self._resolve_robot_clear_footprint_points()
        self._border_obstacle_thickness = float(self.localization_cfg.get("map_border_obstacle_thickness_m", 0.15))
        self._min_obstacle_height = float(self.localization_cfg.get("map_min_obstacle_height_m", 0.04))
        if self._resolution <= 0.0:
            raise ValueError("localization.map_resolution must be positive")
        if self._z_max <= self._z_min:
            raise ValueError("localization.map_z_max must be greater than map_z_min")

    def export_map(self, output_dir: str, clear_center_xy=None) -> dict:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is unavailable, cannot export localization map")

        min_x, min_y, max_x, max_y = self._compute_bounds_xy(stage)
        min_x = math.floor(min_x / self._resolution) * self._resolution
        min_y = math.floor(min_y / self._resolution) * self._resolution
        max_x = math.ceil(max_x / self._resolution) * self._resolution
        max_y = math.ceil(max_y / self._resolution) * self._resolution

        width = int(round((max_x - min_x) / self._resolution)) + 1
        height = int(round((max_y - min_y) / self._resolution)) + 1
        if width <= 0 or height <= 0:
            raise RuntimeError("Occupancy map generator returned empty dimensions")

        image = np.full((height, width), 254, dtype=np.uint8)
        self._paint_map_border(image=image)
        box_object_paths = self._configured_box_object_paths()
        occupied_cell_count = self._rasterize_static_colliders(
            stage=stage,
            image=image,
            min_x=min_x,
            min_y=min_y,
            excluded_prim_paths=box_object_paths,
        )
        if occupied_cell_count <= 0:
            occupied_cell_count = self._rasterize_configured_obstacles(
                image=image,
                min_x=min_x,
                min_y=min_y,
            )

        if clear_center_xy is not None:
            self._clear_robot_footprint(
                image=image,
                min_x=min_x,
                min_y=min_y,
                clear_center_xy=clear_center_xy,
            )

        map_dir = os.path.join(output_dir, self.scene_name)
        os.makedirs(map_dir, exist_ok=True)
        pgm_path = os.path.join(map_dir, "map.pgm")
        yaml_path = os.path.join(map_dir, "map.yaml")

        if not cv2.imwrite(pgm_path, image):
            raise RuntimeError(f"Failed to write localization map image to {pgm_path}")

        yaml_lines = [
            f"image: {os.path.basename(pgm_path)}",
            "mode: trinary",
            f"resolution: {self._resolution:.6f}",
            f"origin: [{float(min_x):.6f}, {float(min_y):.6f}, 0.0]",
            "negate: 0",
            "occupied_thresh: 0.65",
            "free_thresh: 0.25",
            "",
        ]
        with open(yaml_path, "w", encoding="utf-8") as file:
            file.write("\n".join(yaml_lines))

        return {
            "yaml_path": yaml_path,
            "pgm_path": pgm_path,
            "resolution": float(self._resolution),
            "origin": [float(min_x), float(min_y), 0.0],
            "width": int(width),
            "height": int(height),
            "bounds_xy": {
                "min_x": float(min_x),
                "min_y": float(min_y),
                "max_x": float(max_x),
                "max_y": float(max_y),
            },
            "z_bounds": {
                "min_z": float(self._z_min),
                "max_z": float(self._z_max),
            },
            "robot_clear_radius_m": float(self._robot_clear_radius),
            "robot_clear_footprint_points": [
                [float(x), float(y)] for x, y in self._robot_clear_footprint_points
            ],
            "occupied_cell_count": int(occupied_cell_count),
        }

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

    def _paint_map_border(self, image: np.ndarray):
        border_cells = max(1, int(math.ceil(self._border_obstacle_thickness / self._resolution)))
        image[:border_cells, :] = 0
        image[-border_cells:, :] = 0
        image[:, :border_cells] = 0
        image[:, -border_cells:] = 0

    def _configured_box_object_paths(self) -> set[str]:
        task_cfg = getattr(self.workflow.task, "cfg", {})
        if not isinstance(task_cfg, dict):
            return set()

        root_path = str(getattr(self.workflow.task, "root_prim_path", "/World")).rstrip("/")
        objects = task_cfg.get("objects", [])
        if not isinstance(objects, list):
            return set()

        prim_paths: set[str] = set()
        for obj_cfg in objects:
            if not isinstance(obj_cfg, dict):
                continue
            if str(obj_cfg.get("target_class", "")) != "BoxObject":
                continue
            name = str(obj_cfg.get("name", "")).strip()
            if not name:
                continue
            prim_paths.add(f"{root_path}/{name}")
        return prim_paths

    def _rasterize_configured_obstacles(self, image: np.ndarray, min_x: float, min_y: float) -> int:
        warnings.warn(
            "BoxObject-based static map rasterization is deprecated; prefer USD collision geometry instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        task_cfg = getattr(self.workflow.task, "cfg", {})
        if not isinstance(task_cfg, dict):
            return 0

        occupied_before = int(np.count_nonzero(image == 0))
        objects = task_cfg.get("objects", [])
        if not isinstance(objects, list):
            objects = []

        for obj_cfg in objects:
            if not isinstance(obj_cfg, dict):
                continue
            if str(obj_cfg.get("target_class", "")) != "BoxObject":
                continue
            if not bool(obj_cfg.get("collision_enabled", True)):
                continue

            translation = obj_cfg.get("translation", [0.0, 0.0, 0.0])
            scale = obj_cfg.get("scale", [1.0, 1.0, 1.0])
            if not (
                isinstance(translation, (list, tuple))
                and isinstance(scale, (list, tuple))
                and len(translation) == 3
                and len(scale) == 3
            ):
                continue

            half_x = 0.5 * float(scale[0])
            half_y = 0.5 * float(scale[1])
            half_z = 0.5 * float(scale[2])
            bbox_min = (
                float(translation[0]) - half_x,
                float(translation[1]) - half_y,
                float(translation[2]) - half_z,
            )
            bbox_max = (
                float(translation[0]) + half_x,
                float(translation[1]) + half_y,
                float(translation[2]) + half_z,
            )
            if not self._should_rasterize_collider(bbox_min, bbox_max):
                continue

            self._paint_world_rect(
                image=image,
                min_x=min_x,
                min_y=min_y,
                rect_min_xy=(bbox_min[0], bbox_min[1]),
                rect_max_xy=(bbox_max[0], bbox_max[1]),
            )

        return int(np.count_nonzero(image == 0) - occupied_before)

    def _rasterize_static_colliders(
        self,
        stage,
        image: np.ndarray,
        min_x: float,
        min_y: float,
        excluded_prim_paths: set[str] | None = None,
    ) -> int:
        root_path = str(getattr(self.workflow.task, "root_prim_path", "/World"))
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            raise RuntimeError(f"Task root prim '{root_path}' is invalid, cannot rasterize static colliders")

        robot_path = str(getattr(self.robot, "prim_path", "") or "")
        excluded_prim_paths = set(excluded_prim_paths or ())
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox_cache.Clear()

        occupied_before = int(np.count_nonzero(image == 0))
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsValid():
                continue

            prim_path = str(prim.GetPath())
            if robot_path and (prim_path == robot_path or prim_path.startswith(f"{robot_path}/")):
                continue
            if any(prim_path == path or prim_path.startswith(f"{path}/") for path in excluded_prim_paths):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue

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
                image=image,
                min_x=min_x,
                min_y=min_y,
                rect_min_xy=(bbox_min[0], bbox_min[1]),
                rect_max_xy=(bbox_max[0], bbox_max[1]),
            )

        return int(np.count_nonzero(image == 0) - occupied_before)

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
        image: np.ndarray,
        min_x: float,
        min_y: float,
        rect_min_xy: tuple[float, float],
        rect_max_xy: tuple[float, float],
    ):
        height, width = image.shape[:2]
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

        image[height - 1 - max_row : height - min_row, min_col : max_col + 1] = 0

    def _clear_robot_footprint(self, image: np.ndarray, min_x: float, min_y: float, clear_center_xy):
        height, width = image.shape[:2]
        center_x = float(clear_center_xy[0])
        center_y = float(clear_center_xy[1])
        center_yaw = float(clear_center_xy[2]) if len(clear_center_xy) >= 3 else 0.0

        if self._robot_clear_footprint_points:
            cos_yaw = math.cos(center_yaw)
            sin_yaw = math.sin(center_yaw)
            pixel_points = []
            for point_x, point_y in self._robot_clear_footprint_points:
                world_x = center_x + point_x * cos_yaw - point_y * sin_yaw
                world_y = center_y + point_x * sin_yaw + point_y * cos_yaw
                col = int(round((world_x - min_x) / self._resolution))
                row = int(round((world_y - min_y) / self._resolution))
                col = max(0, min(width - 1, col))
                row = max(0, min(height - 1, row))
                pixel_points.append([col, height - 1 - row])
            if len(pixel_points) >= 3:
                cv2.fillPoly(image, [np.asarray(pixel_points, dtype=np.int32)], 254)
                return

        center_col = int(round((center_x - min_x) / self._resolution))
        center_row = int(round((center_y - min_y) / self._resolution))
        radius_cells = max(1, int(math.ceil(self._robot_clear_radius / self._resolution)))

        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > radius_cells * radius_cells:
                    continue
                col = center_col + dx
                row = center_row + dy
                if 0 <= col < width and 0 <= row < height:
                    image[height - 1 - row, col] = 254

    def _resolve_robot_clear_footprint_points(self) -> list[tuple[float, float]]:
        localization_points = self.localization_cfg.get("clear_footprint_points")
        if localization_points is not None:
            return self._normalize_footprint_points(localization_points)

        nav2_skill_cfg = self.base_cfg.get("nav2_skill", {})
        if isinstance(nav2_skill_cfg, dict):
            return self._normalize_footprint_points(nav2_skill_cfg.get("footprint_points"))
        return []

    @staticmethod
    def _normalize_footprint_points(points) -> list[tuple[float, float]]:
        if not isinstance(points, (list, tuple)):
            return []
        normalized = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            normalized.append((float(point[0]), float(point[1])))
        if len(normalized) < 3:
            return []
        return normalized
