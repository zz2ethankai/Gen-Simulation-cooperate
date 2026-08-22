"""Non-physical USD visualization of selected Pick and Place targets."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

try:
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.transformations import get_relative_transform
except ImportError:  # Allows offline USD tests outside Isaac Sim.
    get_prim_at_path = None
    get_relative_transform = None

from .skill_target_math import (
    dashed_line_curves,
    gripper_line_curves,
    plane_from_region_points,
    pose_matrix,
    transform_points,
)


LOGGER = logging.getLogger("de_logger")
DEBUG_ROOT_NAME = "__debug_skill_targets__"


@dataclass(frozen=True)
class SkillTargetReferenceFrame:
    """Narrow robot inputs required to draw one skill target.

    Skills expose this information through ``SkillRuntimePort``.  Keeping the
    frame and robot geometry together avoids making the visualization layer
    depend on the controller façade (which is an Isaac lifecycle object).
    ``robot`` is intentionally the only robot value needed by the renderer:
    gripper keypoints and maximum width.
    """

    robot: Any
    arm_name: str
    robot_base_path: str


def _skill_reference_frame(skill) -> SkillTargetReferenceFrame:
    """Build a render frame from the explicit Skill runtime port."""

    runtime = skill.skill_runtime
    if runtime is None:
        raise RuntimeError(
            "skill target visualization requires a bound SkillRuntimePort"
        )
    base_path = str(runtime.robot_base_path or "").strip()
    if not base_path:
        raise RuntimeError("skill target visualization requires robot_base_path")
    arm_name = str(runtime.arm_name or "").strip()
    if arm_name not in {"left", "right"}:
        raise RuntimeError(
            "skill target visualization requires a left/right SkillRuntimePort arm"
        )
    return SkillTargetReferenceFrame(
        robot=runtime.robot,
        arm_name=arm_name,
        robot_base_path=base_path,
    )


def _prim_name(value: object) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
    return name if name and not name[0].isdigit() else f"_{name}"


def create_skill_target_visualizer(stage, task_root_path: str, task_config: dict):
    config = task_config.get("visualization", {}).get("skill_targets", {})
    if not config.get("enabled", False):
        return None
    return SkillTargetVisualizer(stage, task_root_path, config)


class SkillTargetExportSnapshot:
    def __init__(self, layer, export_enabled: bool):
        self.layer = layer
        self.export_enabled = export_enabled

    def export(self, episode_dir: str | Path) -> Path | None:
        if not self.export_enabled:
            return None
        output_path = Path(episode_dir) / "skill_targets_debug.usda"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.layer.Export(str(output_path))
        return output_path

    def close(self):
        pass


class SkillTargetVisualizer:
    """Own one anonymous USD layer containing episode-local Skill intent markers."""

    def __init__(self, stage, task_root_path: str, config: dict):
        self.stage = stage
        self.task_root_path = str(task_root_path).rstrip("/")
        self.root_path = f"{self.task_root_path}/{DEBUG_ROOT_NAME}"
        self.config = dict(config)
        self.export_enabled = bool(self.config.get("export_usd", True))
        self.retain_completed = bool(self.config.get("retain_completed", True))
        self.completed_opacity_scale = float(
            self.config.get("completed_opacity_scale", 0.25)
        )
        if not 0.0 <= self.completed_opacity_scale <= 1.0:
            raise ValueError("completed_opacity_scale must be in [0, 1]")

        self.pick_config = dict(self.config.get("pick", {}))
        self.place_config = dict(self.config.get("place", {}))
        self.pick_enabled = bool(self.pick_config.get("enabled", True))
        self.place_enabled = bool(self.place_config.get("enabled", True))
        self.pick_line_width = float(self.pick_config.get("line_width_m", 0.006))
        self.place_line_width = float(self.place_config.get("line_width_m", 0.006))
        self.show_pregrasp = bool(self.pick_config.get("show_pregrasp", True))
        self.show_preplace = bool(self.place_config.get("show_preplace", True))
        self.plane_normal_offset = float(
            self.place_config.get("plane_normal_offset_m", 0.003)
        )
        self.min_display_extent = float(
            self.place_config.get("min_display_extent_m", 0.08)
        )
        if min(self.pick_line_width, self.place_line_width, self.min_display_extent) <= 0:
            raise ValueError("Skill target line widths and display extents must be positive")

        self.colors = {
            "pick": self._color(
                self.pick_config.get("grasp_color", [0.0, 0.8, 1.0, 1.0])
            ),
            "pick_pre": self._color(
                self.pick_config.get("pregrasp_color", [0.35, 0.75, 1.0, 0.45])
            ),
            "place_plane": self._color(
                self.place_config.get("plane_color", [0.75, 0.15, 1.0, 0.22])
            ),
            "place_outline": self._color(
                self.place_config.get("outline_color", [0.85, 0.35, 1.0, 0.9])
            ),
            "place_selected": self._color(
                self.place_config.get("selected_color", [0.15, 1.0, 0.35, 1.0])
            ),
            "failed": (1.0, 0.08, 0.08, 0.9),
        }

        self.layer = Sdf.Layer.CreateAnonymous("skill_targets_debug.usda")
        self.stage.GetRootLayer().subLayerPaths.append(self.layer.identifier)
        self.target_count = 0
        self.target_paths: dict[str, str] = {}
        self._closed = False
        self._define_episode_root()

    @staticmethod
    def _color(value) -> tuple[float, float, float, float]:
        color = tuple(float(component) for component in value)
        if len(color) != 4:
            raise ValueError("Skill target colors must be RGBA lists")
        return color

    def _edit(self):
        return Usd.EditContext(self.stage, self.layer)

    def _material_path(self, role: str, completed: bool = False) -> str:
        suffix = "_completed" if completed else ""
        return f"{self.root_path}/Materials/{_prim_name(role + suffix)}"

    def _define_material(self, role: str, color: tuple[float, ...], completed=False):
        material_path = self._material_path(role, completed)
        material = UsdShade.Material.Define(self.stage, material_path)
        shader = UsdShade.Shader.Define(self.stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color[:3])
        )
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color[:3])
        )
        opacity = float(color[3]) * (
            self.completed_opacity_scale if completed else 1.0
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )

    def _define_episode_root(self):
        with self._edit():
            root = UsdGeom.Xform.Define(self.stage, self.root_path).GetPrim()
            root.SetCustomData({"target_count": int(self.target_count)})
            for role, color in self.colors.items():
                self._define_material(role, color, completed=False)
                self._define_material(role, color, completed=True)

    def _standalone_layer_copy(self):
        layer = Sdf.Layer.CreateAnonymous("skill_targets_debug_snapshot.usda")
        layer.TransferContent(self.layer)
        path = Sdf.Path(self.task_root_path)
        while path != Sdf.Path.absoluteRootPath:
            spec = Sdf.CreatePrimInLayer(layer, path)
            spec.specifier = Sdf.SpecifierDef
            if not spec.typeName:
                spec.typeName = "Xform"
            path = path.GetParentPath()
        return layer

    def clear(self):
        if self._closed:
            return
        with self._edit():
            self.stage.RemovePrim(self.root_path)
        self.target_count = 0
        self.target_paths.clear()
        self._define_episode_root()

    def close(self):
        if self._closed:
            return
        sublayers = self.stage.GetRootLayer().subLayerPaths
        if self.layer.identifier in sublayers:
            sublayers.remove(self.layer.identifier)
        self._closed = True

    def clone_for_save(self):
        return SkillTargetExportSnapshot(
            self._standalone_layer_copy(), self.export_enabled
        )

    def export(self, episode_dir: str | Path) -> Path | None:
        if self._closed or not self.export_enabled:
            return None
        output_path = Path(episode_dir) / "skill_targets_debug.usda"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._standalone_layer_copy().Export(str(output_path))
        return output_path

    def _bind_material(self, prim, role: str, completed=False):
        material = UsdShade.Material.Get(
            self.stage, self._material_path(role, completed)
        )
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        prim.SetCustomDataByKey("material_role", role)

    def _add_curves(self, path: str, curves, width: float, role: str):
        curves = [np.asarray(curve, dtype=np.float64).reshape(-1, 3) for curve in curves]
        curves = [curve for curve in curves if len(curve) >= 2]
        if not curves:
            return None
        basis = UsdGeom.BasisCurves.Define(self.stage, path)
        basis.CreateTypeAttr(UsdGeom.Tokens.linear)
        basis.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        basis.CreateCurveVertexCountsAttr(Vt.IntArray([len(curve) for curve in curves]))
        basis.CreatePointsAttr(
            Vt.Vec3fArray(
                [Gf.Vec3f(*point) for curve in curves for point in curve]
            )
        )
        basis.CreateWidthsAttr(Vt.FloatArray([float(width)]))
        basis.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        self._bind_material(basis.GetPrim(), role)
        return basis

    def _add_plane(self, path: str, corners: np.ndarray, role: str):
        corners = np.asarray(corners, dtype=np.float64).reshape(4, 3)
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in corners]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3, 3]))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 0, 2, 3]))
        mesh.CreateDoubleSidedAttr(True)
        self._bind_material(mesh.GetPrim(), role)
        return mesh

    @staticmethod
    def _pose_values(position, orientation) -> Vt.DoubleArray:
        return Vt.DoubleArray(
            np.concatenate(
                [
                    np.asarray(position, dtype=np.float64).reshape(3),
                    np.asarray(orientation, dtype=np.float64).reshape(4),
                ]
            ).tolist()
        )

    @staticmethod
    def _array_values(value) -> Vt.DoubleArray:
        return Vt.DoubleArray(np.asarray(value, dtype=np.float64).reshape(-1).tolist())

    def _relative_transform(self, source_path: str) -> np.ndarray:
        if get_prim_at_path is None or get_relative_transform is None:
            raise RuntimeError("Skill target visualization requires the Isaac Sim runtime")
        return np.asarray(
            get_relative_transform(
                get_prim_at_path(source_path), get_prim_at_path(self.task_root_path)
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _skill_context(skill) -> dict:
        frame = _skill_reference_frame(skill)
        context = dict(skill._target_visualization_context or {})
        context.setdefault("robot", frame.robot.name)
        context.setdefault("arm", frame.arm_name)
        context.setdefault("skill", skill.__class__.__name__.lower())
        return context

    @staticmethod
    def _gripper_parameters(frame: SkillTargetReferenceFrame):
        arm = frame.arm_name
        keypoints = (
            frame.robot.fr_gripper_keypoints
            if arm == "right"
            else frame.robot.fl_gripper_keypoints
        )
        return (
            np.asarray(keypoints["tool_head"], dtype=np.float64),
            np.asarray(keypoints["tool_tail"], dtype=np.float64),
            np.asarray(keypoints["tool_side"], dtype=np.float64),
            float(frame.robot.gripper_max_width),
        )

    def record_target(self, skill, descriptor: dict) -> str | None:
        if self._closed:
            return None
        kind = str(descriptor.get("kind", "")).lower()
        if (kind == "pick" and not self.pick_enabled) or (
            kind == "place" and not self.place_enabled
        ):
            return None
        if kind not in {"pick", "place"}:
            return None
        frame = _skill_reference_frame(skill)
        context = self._skill_context(skill)
        handle = f"skill_{self.target_count:03d}_{_prim_name(kind)}"
        path = f"{self.root_path}/Skills/{handle}"
        with self._edit():
            prim = UsdGeom.Xform.Define(self.stage, path).GetPrim()
            prim.SetCustomData(
                {
                    "arm": str(context.get("arm", "unknown")),
                    "kind": kind,
                    "robot": str(context.get("robot", "unknown")),
                    "skill": str(context.get("skill", kind)),
                    "skill_index": int(context.get("skill_index", -1)),
                    "status": "active",
                }
            )
            constraints = descriptor.get("constraints")
            if constraints is not None:
                prim.SetCustomDataByKey(
                    "constraints_json",
                    json.dumps(constraints, sort_keys=True, default=lambda value: np.asarray(value).tolist()),
                )
            for object_name in descriptor.get("objects", []):
                prim.SetCustomDataByKey(
                    f"object_{len([k for k in prim.GetCustomData() if k.startswith('object_')])}",
                    str(object_name),
                )
            if not descriptor.get("has_target", True):
                prim.SetCustomDataByKey("has_target", False)
                prim.SetCustomDataByKey("status", "failed")
                failure_reason = descriptor.get("failure_reason")
                if failure_reason:
                    prim.SetCustomDataByKey("failure_reason", str(failure_reason))
                candidate_count = descriptor.get("candidate_count")
                if candidate_count is not None:
                    prim.SetCustomDataByKey("candidate_count", int(candidate_count))
            elif kind == "pick":
                prim.SetCustomDataByKey("has_target", True)
                self._record_pick(frame, prim, path, descriptor)
            else:
                prim.SetCustomDataByKey("has_target", True)
                self._record_place(frame, prim, path, descriptor)
            self.target_count += 1
            self.target_paths[handle] = path
            self.stage.GetPrimAtPath(self.root_path).SetCustomDataByKey(
                "target_count", self.target_count
            )
        return handle

    def _record_pick(
        self, frame: SkillTargetReferenceFrame, prim, path: str, descriptor: dict
    ):
        selected_index = int(descriptor["selected_index"])
        prim.SetCustomDataByKey("selected_index", selected_index)
        score = descriptor.get("selected_score")
        if score is not None:
            prim.SetCustomDataByKey("selected_score", float(score))
        grasp_position = np.asarray(descriptor["grasp_position"], dtype=np.float64)
        grasp_orientation = np.asarray(descriptor["grasp_orientation"], dtype=np.float64)
        prim.SetCustomDataByKey(
            "grasp_pose_arm_base", self._pose_values(grasp_position, grasp_orientation)
        )
        task_from_base = self._relative_transform(frame.robot_base_path)
        grasp_transform = task_from_base @ pose_matrix(grasp_position, grasp_orientation)
        tool_head, tool_tail, tool_side, max_width = self._gripper_parameters(frame)
        self._add_curves(
            f"{path}/grasp",
            gripper_line_curves(
                grasp_transform, tool_head, tool_tail, tool_side, max_width
            ),
            self.pick_line_width,
            "pick",
        )
        if self.show_pregrasp and descriptor.get("pregrasp_position") is not None:
            pre_position = np.asarray(descriptor["pregrasp_position"], dtype=np.float64)
            pre_orientation = np.asarray(descriptor["pregrasp_orientation"], dtype=np.float64)
            prim.SetCustomDataByKey(
                "pregrasp_pose_arm_base", self._pose_values(pre_position, pre_orientation)
            )
            pre_transform = task_from_base @ pose_matrix(pre_position, pre_orientation)
            self._add_curves(
                f"{path}/pregrasp",
                gripper_line_curves(
                    pre_transform, tool_head, tool_tail, tool_side, max_width
                ),
                self.pick_line_width * 0.75,
                "pick_pre",
            )
            pre_head = transform_points(tool_head[:3][None], pre_transform)[0]
            grasp_head = transform_points(tool_head[:3][None], grasp_transform)[0]
            self._add_curves(
                f"{path}/approach",
                dashed_line_curves(pre_head, grasp_head),
                self.pick_line_width * 0.55,
                "pick_pre",
            )

    def _record_place(
        self, frame: SkillTargetReferenceFrame, prim, path: str, descriptor: dict
    ):
        direction = str(descriptor.get("place_direction", "vertical"))
        constraint = str(descriptor.get("position_constraint", "gripper"))
        prim.SetCustomDataByKey("place_direction", direction)
        prim.SetCustomDataByKey("position_constraint", constraint)
        prim.SetCustomDataByKey("selected_index", int(descriptor["selected_index"]))
        prim.SetCustomDataByKey(
            "bbox_world", self._array_values(descriptor["bbox_world"])
        )
        prim.SetCustomDataByKey(
            "ratio_ranges", self._array_values(descriptor["ratio_ranges"])
        )

        task_from_world = self._relative_transform("/World")
        region_world = np.asarray(descriptor["region_points_world"], dtype=np.float64)
        region_task = transform_points(region_world, task_from_world)
        normal_world = np.asarray(descriptor["region_normal_world"], dtype=np.float64)
        normal_task = task_from_world[:3, :3] @ normal_world
        hint_world = np.asarray(descriptor["region_tangent_hint_world"], dtype=np.float64)
        hint_task = task_from_world[:3, :3] @ hint_world
        plane = plane_from_region_points(
            region_task,
            normal_task,
            self.min_display_extent,
            self.plane_normal_offset,
            tangent_hint=hint_task,
        )
        prim.SetCustomDataByKey(
            "true_region_points_world", self._array_values(region_world)
        )
        prim.SetCustomDataByKey(
            "true_plane_extents", self._array_values(plane["true_extents"])
        )
        prim.SetCustomDataByKey("display_padded", bool(plane["display_padded"]))
        self._add_plane(f"{path}/target_region", plane["corners"], "place_plane")
        self._add_curves(
            f"{path}/target_region_outline",
            [np.vstack([plane["corners"], plane["corners"][0]])],
            self.place_line_width,
            "place_outline",
        )

        selected_reference_world = np.asarray(
            descriptor["selected_reference_world"], dtype=np.float64
        )
        selected_reference_task = transform_points(
            selected_reference_world[None], task_from_world
        )[0]
        prim.SetCustomDataByKey(
            "selected_reference_world", self._array_values(selected_reference_world)
        )
        marker_size = max(self.min_display_extent * 0.22, 0.018)
        u_axis = np.asarray(plane["tangent_u"])
        v_axis = np.asarray(plane["tangent_v"])
        self._add_curves(
            f"{path}/selected_reference",
            [
                np.stack(
                    [
                        selected_reference_task - u_axis * marker_size,
                        selected_reference_task + u_axis * marker_size,
                    ]
                ),
                np.stack(
                    [
                        selected_reference_task - v_axis * marker_size,
                        selected_reference_task + v_axis * marker_size,
                    ]
                ),
            ],
            self.place_line_width * 1.25,
            "place_selected",
        )

        task_from_base = self._relative_transform(frame.robot_base_path)
        place_position = np.asarray(descriptor["place_position"], dtype=np.float64)
        place_orientation = np.asarray(descriptor["place_orientation"], dtype=np.float64)
        prim.SetCustomDataByKey(
            "place_pose_arm_base", self._pose_values(place_position, place_orientation)
        )
        place_transform = task_from_base @ pose_matrix(place_position, place_orientation)
        tool_head, tool_tail, tool_side, max_width = self._gripper_parameters(frame)
        self._add_curves(
            f"{path}/place_gripper",
            gripper_line_curves(
                place_transform, tool_head, tool_tail, tool_side, max_width
            ),
            self.place_line_width,
            "place_selected",
        )
        if self.show_preplace and descriptor.get("preplace_position") is not None:
            pre_position = np.asarray(descriptor["preplace_position"], dtype=np.float64)
            pre_orientation = np.asarray(descriptor["preplace_orientation"], dtype=np.float64)
            prim.SetCustomDataByKey(
                "preplace_pose_arm_base", self._pose_values(pre_position, pre_orientation)
            )
            pre_transform = task_from_base @ pose_matrix(pre_position, pre_orientation)
            self._add_curves(
                f"{path}/preplace_gripper",
                gripper_line_curves(
                    pre_transform, tool_head, tool_tail, tool_side, max_width
                ),
                self.place_line_width * 0.75,
                "place_outline",
            )

    def finish_target(self, handle: str | None, success: bool, reason: str = ""):
        if self._closed or handle is None:
            return
        path = self.target_paths.get(handle)
        if path is None:
            return
        with self._edit():
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                return
            status = "completed" if success else "failed"
            prim.SetCustomDataByKey("status", status)
            if reason:
                prim.SetCustomDataByKey("failure_reason", str(reason))
            if not self.retain_completed:
                self.stage.RemovePrim(path)
                return
            for descendant in Usd.PrimRange(prim):
                role = descendant.GetCustomDataByKey("material_role")
                if not role:
                    continue
                if success:
                    self._bind_material(descendant, str(role), completed=True)
                else:
                    self._bind_material(descendant, "failed", completed=False)

    def abort_active(self, reason: str):
        for handle, path in list(self.target_paths.items()):
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid() and prim.GetCustomDataByKey("status") == "active":
                self.finish_target(handle, False, reason=reason)


__all__ = [
    "SkillTargetExportSnapshot",
    "SkillTargetReferenceFrame",
    "SkillTargetVisualizer",
    "create_skill_target_visualizer",
]
