"""USD-only visualization of the CuRobo trajectory selected for execution."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

try:
    from omni.isaac.core.utils.prims import get_prim_at_path
    from omni.isaac.core.utils.transformations import get_relative_transform
except ImportError:  # Allows USD/pure-math unit tests outside the Isaac Sim runtime.
    get_prim_at_path = None
    get_relative_transform = None

from .trajectory_math import (
    distance_sample_indices,
    transform_points,
    uniform_sample_indices,
    valid_sphere_arrays,
)


LOGGER = logging.getLogger("de_logger")
DEBUG_ROOT_NAME = "__debug_curobo_trajectory__"


class CuroboTrajectoryExportSnapshot:
    """Immutable anonymous-layer snapshot safe to use from an async save worker."""

    def __init__(self, layer, export_enabled: bool):
        self.layer = layer
        self.export_enabled = export_enabled

    def export(self, episode_dir: str | Path) -> Path | None:
        if not self.export_enabled:
            return None
        output_path = Path(episode_dir) / "trajectory_debug.usda"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.layer.Export(str(output_path))
        return output_path

    def close(self):
        pass


def create_curobo_trajectory_visualizer(stage, task_root_path: str, task_config: dict):
    config = task_config.get("visualization", {}).get("curobo_trajectory", {})
    if not config.get("enabled", False):
        return None
    return CuroboTrajectoryVisualizer(stage, task_root_path, config)


def _prim_name(value: object) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
    return name if name and not name[0].isdigit() else f"_{name}"


class CuroboTrajectoryVisualizer:
    """Own one anonymous, non-physical USD layer shared by all arm controllers."""

    def __init__(self, stage, task_root_path: str, config: dict):
        self.stage = stage
        self.task_root_path = str(task_root_path).rstrip("/")
        self.root_path = f"{self.task_root_path}/{DEBUG_ROOT_NAME}"
        self.config = dict(config)
        self.export_enabled = bool(self.config.get("export_usd", True))
        self.accumulate = bool(self.config.get("accumulate_within_episode", True))
        self.show_ee_path = bool(self.config.get("show_ee_path", True))
        self.show_robot_spheres = bool(
            self.config.get("show_robot_spheres", True)
        )
        self.ee_sample_count = int(self.config.get("ee_sample_count", 64))
        self.robot_pose_sample_count = int(
            self.config.get("robot_pose_sample_count", 8)
        )
        self.ee_radius_m = float(self.config.get("ee_radius_m", 0.015))
        self.ee_min_center_spacing_m = float(
            self.config.get("ee_min_center_spacing_m", 0.0)
        )
        self.ee_color = tuple(
            float(v) for v in self.config.get("ee_color", [1.0, 0.35, 0.0, 1.0])
        )
        self.robot_color = tuple(
            float(v) for v in self.config.get("robot_color", [0.1, 0.85, 0.25, 0.28])
        )
        self.robot_radius_scale = float(self.config.get("robot_radius_scale", 1.0))
        if self.ee_sample_count <= 0 or self.robot_pose_sample_count <= 0:
            raise ValueError("trajectory sample counts must be positive")
        if self.ee_radius_m <= 0.0 or self.robot_radius_scale <= 0.0:
            raise ValueError("trajectory sphere radii/scales must be positive")
        if self.ee_min_center_spacing_m < 0.0:
            raise ValueError("ee_min_center_spacing_m must be non-negative")
        if len(self.ee_color) != 4 or len(self.robot_color) != 4:
            raise ValueError("trajectory colors must be RGBA lists")

        self.layer = Sdf.Layer.CreateAnonymous("trajectory_debug.usda")
        self.stage.GetRootLayer().subLayerPaths.append(self.layer.identifier)
        self.plan_count = 0
        self._closed = False
        self._define_episode_root()

    def _edit(self):
        return Usd.EditContext(self.stage, self.layer)

    def _define_episode_root(self):
        with self._edit():
            root = UsdGeom.Xform.Define(self.stage, self.root_path).GetPrim()
            root.SetCustomData({"plan_count": int(self.plan_count)})
            if self.show_ee_path:
                self._define_material("ee", self.ee_color)
            if self.show_robot_spheres:
                self._define_material("robot", self.robot_color)

    def _standalone_layer_copy(self):
        layer = Sdf.Layer.CreateAnonymous("trajectory_debug_snapshot.usda")
        layer.TransferContent(self.layer)
        # The live layer is composed below an existing task stage, so its
        # ancestors are `over` specs. Promote only the detached export copy;
        # changing the live task-root spec would invalidate Isaac objects.
        path = Sdf.Path(self.task_root_path)
        while path != Sdf.Path.absoluteRootPath:
            spec = Sdf.CreatePrimInLayer(layer, path)
            spec.specifier = Sdf.SpecifierDef
            if not spec.typeName:
                spec.typeName = "Xform"
            path = path.GetParentPath()
        return layer

    def _define_material(self, name: str, color: tuple[float, ...]):
        material_path = f"{self.root_path}/Materials/{_prim_name(name)}"
        material = UsdShade.Material.Define(self.stage, material_path)
        shader = UsdShade.Shader.Define(self.stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color[:3])
        )
        if name == "ee":
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*color[:3])
            )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(color[3]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        return material

    def clear(self):
        """Start a fresh episode while retaining the same anonymous sublayer."""
        if self._closed:
            return
        with self._edit():
            self.stage.RemovePrim(self.root_path)
        self.plan_count = 0
        self._define_episode_root()

    def close(self):
        if self._closed:
            return
        sublayers = self.stage.GetRootLayer().subLayerPaths
        if self.layer.identifier in sublayers:
            sublayers.remove(self.layer.identifier)
        self._closed = True

    def clone_for_save(self):
        return CuroboTrajectoryExportSnapshot(
            self._standalone_layer_copy(), self.export_enabled
        )

    def _add_instancer(
        self,
        path: str,
        positions: np.ndarray,
        radii: np.ndarray,
        material_name: str,
        color: tuple[float, ...],
    ):
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        radii = np.asarray(radii, dtype=np.float64).reshape(-1)
        if positions.shape[0] != radii.shape[0]:
            raise ValueError("PointInstancer positions/radii length mismatch")
        instancer = UsdGeom.PointInstancer.Define(self.stage, path)
        prototype = UsdGeom.Sphere.Define(self.stage, f"{path}/Prototype")
        prototype.CreateRadiusAttr(1.0)
        prototype.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color[:3])]))
        prototype.CreateDisplayOpacityAttr(Vt.FloatArray([float(color[3])]))
        material = UsdShade.Material.Get(
            self.stage, f"{self.root_path}/Materials/{material_name}"
        )
        UsdShade.MaterialBindingAPI.Apply(prototype.GetPrim()).Bind(material)
        instancer.CreatePrototypesRel().SetTargets([prototype.GetPath()])
        instancer.CreateProtoIndicesAttr(Vt.IntArray([0] * len(positions)))
        instancer.CreatePositionsAttr(
            Vt.Vec3fArray([Gf.Vec3f(*point) for point in positions])
        )
        instancer.CreateScalesAttr(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(float(radius), float(radius), float(radius))
                    for radius in radii
                ]
            )
        )

    def record_plan(self, controller, cmd_plan, command: str):
        """Render the final selected JointState trajectory for one controller plan."""
        if self._closed:
            return
        if get_prim_at_path is None or get_relative_transform is None:
            raise RuntimeError("record_plan requires the Isaac Sim runtime")
        positions = cmd_plan.position
        trajectory_length = int(positions.shape[0])
        if trajectory_length <= 0:
            return
        if not self.accumulate:
            self.clear()

        kinematics = controller.motion_gen.kinematics
        ee_indices = np.empty((0,), dtype=np.int64)
        ee_points_base = np.empty((0, 3), dtype=np.float64)
        if self.show_ee_path:
            all_ee_state = kinematics.get_state(positions)
            all_ee_points_base = all_ee_state.ee_position.detach().cpu().numpy()
            ee_indices = distance_sample_indices(
                all_ee_points_base,
                self.ee_min_center_spacing_m,
                self.ee_sample_count,
            )
            ee_points_base = all_ee_points_base[ee_indices]

        robot_indices = np.empty((0,), dtype=np.int64)
        frame_centers: list[np.ndarray] = []
        frame_radii: list[np.ndarray] = []
        if self.show_robot_spheres:
            robot_indices = uniform_sample_indices(
                trajectory_length, self.robot_pose_sample_count
            )
            robot_q = positions[robot_indices.tolist()]
            sphere_frames = kinematics.get_robot_as_spheres(
                robot_q, filter_valid=True
            )
            for spheres in sphere_frames:
                centers, radii = valid_sphere_arrays(spheres)
                if len(radii):
                    frame_centers.append(centers)
                    frame_radii.append(radii * self.robot_radius_scale)

        task_from_arm_base = get_relative_transform(
            get_prim_at_path(controller.robot_base_path),
            get_prim_at_path(self.task_root_path),
        )
        ee_points_task = transform_points(ee_points_base, task_from_arm_base)
        robot_points_task = (
            transform_points(np.concatenate(frame_centers, axis=0), task_from_arm_base)
            if frame_centers
            else np.empty((0, 3), dtype=np.float64)
        )
        robot_radii = (
            np.concatenate(frame_radii, axis=0) if frame_radii else np.empty((0,))
        )

        robot_name = _prim_name(controller.name)
        arm_name = _prim_name(controller.lr_name or "unknown")
        plan_name = f"plan_{self.plan_count:03d}"
        plan_path = f"{self.root_path}/{robot_name}/{arm_name}/{plan_name}"
        with self._edit():
            plan_prim = UsdGeom.Xform.Define(self.stage, plan_path).GetPrim()
            plan_prim.SetCustomData(
                {
                    "arm": str(controller.lr_name or "unknown"),
                    "command": str(command),
                    "trajectory_length": trajectory_length,
                    "ee_sample_indices": Vt.IntArray(ee_indices.tolist()),
                    "robot_pose_sample_indices": Vt.IntArray(robot_indices.tolist()),
                }
            )
            if self.show_ee_path and len(ee_points_task):
                self._add_instancer(
                    f"{plan_path}/ee_path",
                    ee_points_task,
                    np.full((len(ee_points_task),), self.ee_radius_m),
                    "ee",
                    self.ee_color,
                )
            if self.show_robot_spheres and len(robot_radii):
                self._add_instancer(
                    f"{plan_path}/robot_spheres",
                    robot_points_task,
                    robot_radii,
                    "robot",
                    self.robot_color,
                )
            self.plan_count += 1
            self.stage.GetPrimAtPath(self.root_path).SetCustomDataByKey(
                "plan_count", self.plan_count
            )
        LOGGER.info(
            "[TrajectoryDebug] recorded robot=%s arm=%s command=%s trajectory_length=%d ee_samples=%d "
            "robot_pose_samples=%d robot_spheres=%d",
            controller.name,
            controller.lr_name,
            command,
            trajectory_length,
            len(ee_indices),
            len(robot_indices),
            len(robot_radii),
        )

    def export(self, episode_dir: str | Path) -> Path | None:
        if self._closed or not self.export_enabled:
            return None
        output_path = Path(episode_dir) / "trajectory_debug.usda"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._standalone_layer_copy().Export(str(output_path))
        return output_path
