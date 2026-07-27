"""InterndataEngine sparse-grasp exporter and robot profile loader.

GraspGenX predicts a canonical gripper-base pose conditioned on a gripper
description.  InterndataEngine stores a robot-independent GraspNet-style TCP
pose and converts it to the selected robot end-effector frame at runtime.  This
module bridges those contracts while validating the project CuRobo/URDF chain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import re
import xml.etree.ElementTree as ET

import numpy as np
import yaml


# Columns are the GraspNet basis vectors expressed in the GraspGenX canonical
# grasp basis.  GraspNet uses x=approach, y=closing, z=height.  GraspGenX uses
# x=closing, z=approach, leaving y as the right-handed height direction.
R_GRASPGENX_FROM_GRASPNET = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


# Gripper selection is an adapter concern, not a task/pick-skill parameter.
# These are official GraspGenX descriptors whose geometry best matches the
# corresponding project robot.  An optional robot-YAML `graspgenx.gripper`
# entry can override this table for future embodiments.
ROBOT_GRIPPER_MAP = {
    "pandaomron": "franka_panda",
    "pandaomronvirtual": "franka_panda",
    "fr3": "franka_panda",
    "tracer2franka": "franka_panda",
    "splitaloha": "piper_hand",
    "splitalohaactual": "piper_hand",
    "lift2": "arx_x5",
    "genie1": "galaxea_g1",
    "frankarobotiq85": "robotiq_2f_85",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        marker = candidate / "workflows/simbox/core/configs/robots"
        if marker.is_dir():
            return candidate
    raise ValueError(
        f"Cannot locate InterndataEngine root from {start}; pass project_root explicitly"
    )


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _resolve_urdf_path(curobo_config: Path, urdf_ref: str) -> Path:
    path = Path(urdf_ref)
    if path.is_absolute():
        return path

    content_dir = next(
        (parent for parent in curobo_config.parents if parent.name == "content"),
        None,
    )
    candidates: list[Path] = []
    if content_dir is not None:
        candidates.append(content_dir / "assets" / path)
    candidates.extend(
        [curobo_config.parent / path, curobo_config.parent.parent / path]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else path.resolve()


def _load_urdf_links(urdf_path: Path) -> tuple[set[str], str]:
    try:
        root = ET.parse(urdf_path).getroot()
        return (
            {node.attrib.get("name", "") for node in root.findall("link")},
            "xml",
        )
    except ET.ParseError:
        # A few project URDFs contain nested comments.  CuRobo still consumes
        # their links, so recover link names without modifying user assets.
        text = urdf_path.read_text(encoding="utf-8", errors="replace")
        links = set(re.findall(r"<link\s+name\s*=\s*['\"]([^'\"]+)['\"]", text))
        if not links:
            raise
        return links, "recovered_link_names"


@dataclass(frozen=True)
class ArmUrdf:
    curobo_config: str
    urdf_path: str
    base_link: str
    ee_link: str
    validation: str


@dataclass(frozen=True)
class RobotProfile:
    """Robot parameters used by InterndataEngine's runtime pose conversion."""

    name: str
    robot_config: str
    gripper_name: str
    gripper_selection: str
    gripper_min_width: float
    gripper_max_width: float
    tcp_offset: float
    ee_axis: str
    r_ee_graspnet: tuple[tuple[float, float, float], ...]
    arms: tuple[ArmUrdf, ...]

    @classmethod
    def from_project_config(
        cls,
        robot_config: str | Path,
        *,
        project_root: str | Path | None = None,
        validate_urdf: bool = True,
    ) -> "RobotProfile":
        config_path = Path(robot_config).expanduser()
        if project_root is None:
            probe = config_path.resolve() if config_path.is_absolute() else Path.cwd()
            root = _find_project_root(probe)
        else:
            root = Path(project_root).expanduser().resolve()
        config_path = _resolve_project_path(root, str(config_path)).resolve()
        config = _load_yaml(config_path)

        required = (
            "target_class",
            "robot_file",
            "gripper_min_width",
            "gripper_max_width",
            "tcp_offset",
            "R_ee_graspnet",
            "ee_axis",
        )
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"Robot config {config_path} is missing: {missing}")

        target_class = str(config["target_class"])
        adapter_cfg = config.get("graspgenx", {})
        if adapter_cfg is None:
            adapter_cfg = {}
        if not isinstance(adapter_cfg, dict):
            raise ValueError(f"graspgenx must be a mapping in {config_path}")
        configured_gripper = adapter_cfg.get("gripper")
        if configured_gripper is None:
            try:
                gripper_name = ROBOT_GRIPPER_MAP[target_class.lower()]
            except KeyError as exc:
                raise ValueError(
                    f"No GraspGenX gripper mapping for target_class={target_class!r}; "
                    "add it to ROBOT_GRIPPER_MAP or set graspgenx.gripper in the robot YAML"
                ) from exc
            gripper_selection = "target_class_map"
        else:
            gripper_name = str(configured_gripper).strip()
            if not gripper_name:
                raise ValueError(f"graspgenx.gripper is empty in {config_path}")
            gripper_selection = "robot_config_override"

        r_ee = np.asarray(config["R_ee_graspnet"], dtype=np.float64)
        if r_ee.shape != (3, 3):
            raise ValueError(f"R_ee_graspnet must be 3x3 in {config_path}")
        if not np.allclose(r_ee.T @ r_ee, np.eye(3), atol=1e-6) or not np.isclose(
            np.linalg.det(r_ee), 1.0, atol=1e-6
        ):
            raise ValueError(f"R_ee_graspnet is not a proper rotation: {config_path}")

        ee_axis = str(config["ee_axis"])
        if ee_axis not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported ee_axis={ee_axis!r} in {config_path}")

        robot_files = config["robot_file"]
        if isinstance(robot_files, str):
            robot_files = [robot_files]
        if not isinstance(robot_files, list) or not robot_files:
            raise ValueError(f"robot_file must contain CuRobo config paths: {config_path}")

        arms: list[ArmUrdf] = []
        for robot_file in robot_files:
            curobo_path = _resolve_project_path(root, str(robot_file)).resolve()
            if not curobo_path.is_file():
                raise FileNotFoundError(f"CuRobo config does not exist: {curobo_path}")
            curobo = _load_yaml(curobo_path)
            try:
                kinematics = curobo["robot_cfg"]["kinematics"]
                urdf_ref = str(kinematics["urdf_path"])
                base_link = str(kinematics["base_link"])
                ee_link = str(kinematics["ee_link"])
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Invalid CuRobo kinematics config: {curobo_path}") from exc

            urdf_path = _resolve_urdf_path(curobo_path, urdf_ref)
            if not urdf_path.is_file():
                raise FileNotFoundError(
                    f"URDF referenced by {curobo_path} does not exist: {urdf_path}"
                )
            validation = "skipped"
            if validate_urdf:
                links, validation = _load_urdf_links(urdf_path)
                absent = [link for link in (base_link, ee_link) if link not in links]
                if absent:
                    raise ValueError(
                        f"URDF {urdf_path} does not contain configured links: {absent}"
                    )
            arms.append(
                ArmUrdf(
                    curobo_config=str(curobo_path),
                    urdf_path=str(urdf_path),
                    base_link=base_link,
                    ee_link=ee_link,
                    validation=validation,
                )
            )

        min_width = float(config["gripper_min_width"])
        max_width = float(config["gripper_max_width"])
        if not 0.0 <= min_width < max_width:
            raise ValueError(
                f"Invalid gripper width range [{min_width}, {max_width}] in {config_path}"
            )

        return cls(
            name=target_class,
            robot_config=str(config_path),
            gripper_name=gripper_name,
            gripper_selection=gripper_selection,
            gripper_min_width=min_width,
            gripper_max_width=max_width,
            tcp_offset=float(config["tcp_offset"]),
            ee_axis=ee_axis,
            r_ee_graspnet=tuple(
                tuple(float(value) for value in row) for row in r_ee
            ),
            arms=tuple(arms),
        )

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)


def resolve_gripper_name(profile: RobotProfile, requested: str = "auto") -> str:
    """Resolve the inference gripper without requiring a pick-task parameter."""
    value = str(requested).strip()
    return profile.gripper_name if value in {"", "auto"} else value


def _validate_predictions(grasps: np.ndarray, confidences: np.ndarray) -> None:
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
        raise ValueError(f"Expected GraspGenX poses [N,4,4], got {grasps.shape}")
    if confidences.shape != (len(grasps),):
        raise ValueError(
            f"Expected one confidence per pose, got {confidences.shape} for {len(grasps)} poses"
        )
    if not np.isfinite(grasps).all() or not np.isfinite(confidences).all():
        raise ValueError("GraspGenX predictions contain non-finite values")
    rotations = grasps[:, :3, :3]
    should_be_identity = np.swapaxes(rotations, 1, 2) @ rotations
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-4):
        raise ValueError("GraspGenX predictions contain non-orthonormal rotations")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-4):
        raise ValueError("GraspGenX predictions contain improper rotations")


def export_interndata_grasps(
    grasps: np.ndarray,
    confidences: np.ndarray,
    profile: RobotProfile,
    *,
    tool_tcp_transform: np.ndarray,
    count: int | None = None,
    height: float = 0.02,
    insertion_depth: float = 0.0,
    object_id: int = -1,
) -> np.ndarray:
    """Convert canonical GraspGenX poses to InterndataEngine ``N x 17``.

    The TCP center uses the descriptor's full canonical-grasp-to-tool transform,
    including lateral translation.  The stored orientation remains the generic
    GraspNet canonical basis; project runtime then applies ``R_ee_graspnet``,
    ``ee_axis`` and ``tcp_offset`` from the selected robot profile.
    """
    poses = np.asarray(grasps, dtype=np.float64)
    scores = np.asarray(confidences, dtype=np.float64).reshape(-1)
    tool_transform = np.asarray(tool_tcp_transform, dtype=np.float64)
    _validate_predictions(poses, scores)
    if tool_transform.shape != (4, 4):
        raise ValueError(
            f"tool_tcp_transform must be 4x4, got {tool_transform.shape}"
        )
    if not np.isfinite(tool_transform).all():
        raise ValueError("tool_tcp_transform contains non-finite values")
    if not np.allclose(tool_transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("tool_tcp_transform has an invalid homogeneous bottom row")
    tool_rotation = tool_transform[:3, :3]
    if not np.allclose(
        tool_rotation.T @ tool_rotation, np.eye(3), atol=1e-6
    ) or not np.isclose(np.linalg.det(tool_rotation), 1.0, atol=1e-6):
        raise ValueError("tool_tcp_transform does not contain a proper rotation")
    if height <= 0.0:
        raise ValueError("height must be positive")

    scores = np.clip(scores, 0.0, 1.0)
    legacy_scores = 0.1 + 0.9 * (1.0 - scores)
    order = np.argsort(legacy_scores, kind="stable")
    if count is not None:
        if count <= 0:
            raise ValueError("count must be positive")
        if len(order) < count:
            raise ValueError(
                f"GraspGenX returned {len(order)} candidates, fewer than requested {count}"
            )
        order = order[:count]

    selected = poses[order]
    rotations_graspgenx = selected[:, :3, :3]
    rotations_graspnet = rotations_graspgenx @ R_GRASPGENX_FROM_GRASPNET
    tcp_poses = selected @ tool_transform

    output = np.empty((len(selected), 17), dtype=np.float32)
    output[:, 0] = legacy_scores[order]
    output[:, 1] = profile.gripper_max_width
    output[:, 2] = float(height)
    output[:, 3] = float(insertion_depth)
    output[:, 4:13] = rotations_graspnet.reshape(-1, 9)
    output[:, 13:16] = tcp_poses[:, :3, 3]
    output[:, 16] = float(object_id)
    return output
