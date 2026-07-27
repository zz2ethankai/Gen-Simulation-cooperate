"""InterndataEngine sparse-grasp exporter and robot profile loader.

GraspGen predicts the pose of a model-specific gripper base. InterndataEngine
stores a robot-independent GraspNet-style TCP pose and lets the robot runtime
convert that pose to its own end-effector frame. This module bridges those two
contracts while validating the selected project robot against its CuRobo URDF.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import re
import xml.etree.ElementTree as ET

import numpy as np
import yaml


# Columns are the GraspNet basis vectors expressed in the GraspGen basis.
# GraspNet: x=approach, y=closing, z=height.
# GraspGen: x=closing, y=height, z=approach.
R_GRASPGEN_FROM_GRASPNET = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


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
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _resolve_urdf_path(curobo_config: Path, urdf_ref: str) -> Path:
    path = Path(urdf_ref)
    if path.is_absolute():
        return path

    content_dir = next(
        (parent for parent in curobo_config.parents if parent.name == "content"),
        None,
    )
    candidates = []
    if content_dir is not None:
        candidates.append(content_dir / "assets" / path)
    candidates.extend(
        [curobo_config.parent / path, curobo_config.parent.parent / path]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else path.resolve()


@dataclass(frozen=True)
class ArmUrdf:
    curobo_config: str
    urdf_path: str
    base_link: str
    ee_link: str
    validation: str


def _load_urdf_links(urdf_path: Path) -> tuple[set[str], str]:
    try:
        root = ET.parse(urdf_path).getroot()
        return (
            {node.attrib.get("name", "") for node in root.findall("link")},
            "xml",
        )
    except ET.ParseError:
        # Some project URDFs contain nested comments. CuRobo still consumes
        # their link definitions, so recover names without rewriting the asset.
        text = urdf_path.read_text(encoding="utf-8", errors="replace")
        links = set(
            re.findall(r"<link\s+name\s*=\s*['\"]([^'\"]+)['\"]", text)
        )
        if not links:
            raise
        return links, "recovered_link_names"


@dataclass(frozen=True)
class RobotProfile:
    """Robot parameters consumed by InterndataEngine's grasp pose conversion."""

    name: str
    robot_config: str
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
            raise ValueError(f"robot_file must contain at least one CuRobo config: {config_path}")

        arms = []
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
            name=str(config["target_class"]),
            robot_config=str(config_path),
            gripper_min_width=min_width,
            gripper_max_width=max_width,
            tcp_offset=float(config["tcp_offset"]),
            ee_axis=ee_axis,
            r_ee_graspnet=tuple(tuple(float(value) for value in row) for row in r_ee),
            arms=tuple(arms),
        )

    @property
    def recommended_source_grippers(self) -> tuple[str, ...]:
        """Order checkpoint embodiments by compatibility with this project robot."""
        name = self.name.lower()
        if "robotiq" in name:
            return ("robotiq_2f_140", "franka_panda")
        return ("franka_panda", "robotiq_2f_140")

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)


def resolve_model_config(
    models_dir: str | Path,
    profile: RobotProfile,
    *,
    source_gripper: str = "auto",
) -> Path:
    """Resolve a GraspGen checkpoint YAML without task-side gripper selection."""
    root = Path(models_dir).expanduser().resolve()
    names: Iterable[str]
    if source_gripper == "auto":
        names = profile.recommended_source_grippers
    else:
        names = (source_gripper,)

    checked = []
    for name in names:
        patterns = (
            f"graspgen_{name}.yml",
            f"graspgen_{name}.yaml",
            f"*{name}*.yml",
            f"*{name}*.yaml",
        )
        for base in (root / "checkpoints", root):
            for pattern in patterns:
                for candidate in sorted(base.glob(pattern)):
                    checked.append(str(candidate))
                    if candidate.is_file():
                        return candidate.resolve()
    raise FileNotFoundError(
        f"No GraspGen model config found in {root} for {tuple(names)}; checked {checked}"
    )


def load_source_gripper_geometry(
    gripper_name: str,
    *,
    graspgen_root: str | Path | None = None,
) -> dict[str, float]:
    if graspgen_root is None:
        root = Path(__file__).resolve().parents[2]
    else:
        root = Path(graspgen_root).expanduser().resolve()
    config_path = root / "config/grippers" / f"{gripper_name}.yaml"
    config = _load_yaml(config_path)
    if "depth" not in config:
        raise ValueError(f"Source gripper has no depth: {config_path}")
    return {
        "depth": float(config["depth"]),
        "width": float(config.get("width", 0.0)),
    }


def _validate_predictions(grasps: np.ndarray, confidences: np.ndarray) -> None:
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
        raise ValueError(f"Expected GraspGen poses [N,4,4], got {grasps.shape}")
    if confidences.shape != (len(grasps),):
        raise ValueError(
            f"Expected one confidence per pose, got {confidences.shape} for {len(grasps)} poses"
        )
    if not np.isfinite(grasps).all() or not np.isfinite(confidences).all():
        raise ValueError("GraspGen predictions contain non-finite values")
    rotations = grasps[:, :3, :3]
    should_be_identity = np.swapaxes(rotations, 1, 2) @ rotations
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-4):
        raise ValueError("GraspGen predictions contain non-orthonormal rotations")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-4):
        raise ValueError("GraspGen predictions contain improper rotations")


def export_interndata_grasps(
    grasps: np.ndarray,
    confidences: np.ndarray,
    profile: RobotProfile,
    *,
    source_gripper_depth: float,
    count: int | None = None,
    height: float = 0.02,
    insertion_depth: float = 0.0,
    object_id: int = -1,
) -> np.ndarray:
    """Convert GraspGen base-link poses to InterndataEngine ``N x 17`` rows.

    The stored center is the canonical tool contact point. The existing robot
    runtime applies ``R_ee_graspnet``, ``tcp_offset`` and any dynamic gripper
    correction when it turns these rows into embodiment-specific EE poses.
    """
    poses = np.asarray(grasps, dtype=np.float64)
    scores = np.asarray(confidences, dtype=np.float64).reshape(-1)
    _validate_predictions(poses, scores)
    if source_gripper_depth <= 0.0:
        raise ValueError("source_gripper_depth must be positive")
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
                f"GraspGen returned {len(order)} candidates, fewer than requested {count}"
            )
        order = order[:count]

    selected = poses[order]
    rotations_graspgen = selected[:, :3, :3]
    rotations_graspnet = rotations_graspgen @ R_GRASPGEN_FROM_GRASPNET
    tcp_centers = (
        selected[:, :3, 3]
        + rotations_graspgen[:, :, 2] * float(source_gripper_depth)
    )

    output = np.empty((len(selected), 17), dtype=np.float32)
    output[:, 0] = legacy_scores[order]
    output[:, 1] = profile.gripper_max_width
    output[:, 2] = float(height)
    output[:, 3] = float(insertion_depth)
    output[:, 4:13] = rotations_graspnet.reshape(-1, 9)
    output[:, 13:16] = tcp_centers
    output[:, 16] = float(object_id)
    return output
