"""Typed, deterministic loading for canonical SimBox robot profiles."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


class RobotProfileError(ValueError):
    """Raised when a canonical robot profile is missing or invalid."""


class PlacementFamily(str, Enum):
    FLOOR_STANDING = "floor_standing"
    SUPPORT_MOUNTED = "support_mounted"


@dataclass(frozen=True)
class RobotCollisionLayer:
    name: str
    center_xy_m: tuple[float, float]
    size_xy_m: tuple[float, float]
    min_z_m: float
    max_z_m: float


@dataclass(frozen=True)
class RobotPlacementProfile:
    family: PlacementFamily
    support_role: str
    footprint_m: tuple[float, float] | None
    collision_layers: tuple[RobotCollisionLayer, ...]
    base_contact_offset_m: float
    authored_forward_yaw_offset_deg: float


@dataclass(frozen=True)
class RobotAssetVariant:
    variant_id: str
    sha256: str


@dataclass(frozen=True)
class RobotBaseProfile:
    operation_mode: str
    locked_joint_names: tuple[str, ...]
    joint_groups: Mapping[str, Mapping[str, Any]]
    hold: Mapping[str, float]


@dataclass(frozen=True)
class RobotGripperProfile:
    joint_names: tuple[str, ...]
    joint_indices: tuple[int, ...]
    max_width: float
    min_width: float
    tcp_offset: float
    home: tuple[float, ...]
    keypoints: Mapping[str, tuple[float, ...]]
    action_adapter: str
    action_params: Mapping[str, Any]


@dataclass(frozen=True)
class RobotArmProfile:
    arm_id: str
    controller: str
    curobo_file: str
    command_joint_names: tuple[str, ...]
    trajectory_joint_names: tuple[str, ...]
    joint_indices: tuple[int, ...]
    ee_path: str
    base_path: str
    filter_paths: tuple[str, ...]
    forbid_collision_paths: tuple[str, ...]
    home: tuple[float, ...]
    home_std: tuple[float, ...]
    arm_base_xy_m: tuple[float, float] | None
    gripper: RobotGripperProfile


@dataclass(frozen=True)
class RobotCameraProfile:
    name_suffix: str
    save_name: str
    translation: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    camera_axes: str
    camera_file: str
    parent_suffix: str
    apply_randomization: bool
    max_translation_noise: float | None = None
    max_orientation_noise: float | None = None

    def to_task_camera(self, robot_instance: str) -> dict[str, Any]:
        camera: dict[str, Any] = {
            "name": f"{robot_instance}_{self.name_suffix}",
            "save_name": self.save_name,
            "translation": list(self.translation),
            "orientation": list(self.orientation),
            "camera_axes": self.camera_axes,
            "camera_file": self.camera_file,
            "parent": (
                f"{robot_instance}/{self.parent_suffix}"
                if self.parent_suffix
                else robot_instance
            ),
            "apply_randomization": self.apply_randomization,
            "record_to": robot_instance,
            "record_mode": "lmdb_and_video",
        }
        if self.max_translation_noise is not None:
            camera["max_translation_noise"] = self.max_translation_noise
        if self.max_orientation_noise is not None:
            camera["max_orientation_noise"] = self.max_orientation_noise
        return camera


@dataclass(frozen=True)
class RobotDataArmAdapter:
    action_name: str
    joint_position_key: str
    gripper_position_key: str
    gripper_pose_key: str | None


@dataclass(frozen=True)
class RobotDataAdapter:
    arms: Mapping[str, RobotDataArmAdapter]


@dataclass(frozen=True)
class RobotModelProfile:
    profile_id: str
    target_class: str
    path: str
    asset_variants: tuple[RobotAssetVariant, ...]
    placement: RobotPlacementProfile
    base: RobotBaseProfile
    arms: Mapping[str, RobotArmProfile]
    camera_rig: tuple[RobotCameraProfile, ...]
    data_adapter: RobotDataAdapter
    capabilities: frozenset[str]
    collision_world_modes: frozenset[str]
    physics: Mapping[str, Any]
    kinematics: Mapping[str, Any]
    profile_hash: str
    source_path: Path


_PROFILE_FIELDS = {
    "profile_id",
    "target_class",
    "path",
    "asset_variants",
    "placement",
    "base",
    "arms",
    "camera_rig",
    "data_adapter",
    "capabilities",
    "collision_world_modes",
    "physics",
    "kinematics",
}
_REQUIRED_PROFILE_FIELDS = _PROFILE_FIELDS - {"asset_variants"}
_INSTANCE_OVERRIDE_FIELDS = {
    "name",
    "euler",
    "robot_config_file",
    "left_joint_home",
    "right_joint_home",
    "left_joint_home_std",
    "right_joint_home_std",
    "left_gripper_home",
    "right_gripper_home",
    "ignore_substring",
    "use_batch",
    "collision_activation_distance",
    "constrain_grasp_approach",
    "tcp_offset",
}
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RobotProfileError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RobotProfileError(f"{field} must be a list")
    return value


def _text(value: Any, field: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise RobotProfileError(f"{field} must be non-empty")
    return text


def _texts(value: Any, field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    result = tuple(_text(item, field) for item in _sequence(value, field))
    if not allow_empty and not result:
        raise RobotProfileError(f"{field} must not be empty")
    if len(result) != len(set(result)):
        raise RobotProfileError(f"{field} values must be unique")
    return result


def _numbers(
    value: Any,
    field: str,
    *,
    length: int | None = None,
    positive: bool = False,
    allow_empty: bool = False,
) -> tuple[float, ...]:
    raw = _sequence(value, field)
    result: list[float] = []
    for item in raw:
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise RobotProfileError(f"{field} must contain numbers") from exc
        if not math.isfinite(number) or (positive and number <= 0.0):
            raise RobotProfileError(f"{field} contains an invalid number: {number}")
        result.append(number)
    if length is not None and len(result) != length:
        raise RobotProfileError(f"{field} must contain exactly {length} values")
    if not allow_empty and not result:
        raise RobotProfileError(f"{field} must not be empty")
    return tuple(result)


def _indices(value: Any, field: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    raw = _sequence(value, field)
    result: list[int] = []
    for item in raw:
        if isinstance(item, bool):
            raise RobotProfileError(f"{field} must contain integer indices")
        try:
            number = int(item)
        except (TypeError, ValueError) as exc:
            raise RobotProfileError(f"{field} must contain integer indices") from exc
        if number < 0 or number != float(item):
            raise RobotProfileError(f"{field} contains an invalid index: {item}")
        result.append(number)
    if not allow_empty and not result:
        raise RobotProfileError(f"{field} must not be empty")
    if len(result) != len(set(result)):
        raise RobotProfileError(f"{field} values must be unique")
    return tuple(result)


def _optional_xy(value: Any, field: str) -> tuple[float, float] | None:
    if value is None:
        return None
    values = _numbers(value, field, length=2)
    return values[0], values[1]


def _load_asset_variants(value: Any) -> tuple[RobotAssetVariant, ...]:
    variants: list[RobotAssetVariant] = []
    variant_ids: set[str] = set()
    hashes: set[str] = set()
    for index, item in enumerate(_sequence(value, "asset_variants")):
        raw = _mapping(item, f"asset_variants[{index}]")
        if set(raw) != {"variant_id", "sha256"}:
            raise RobotProfileError(
                f"asset_variants[{index}] must contain exactly variant_id and sha256"
            )
        variant_id = _text(raw["variant_id"], f"asset_variants[{index}].variant_id")
        sha256 = _text(raw["sha256"], f"asset_variants[{index}].sha256").lower()
        if len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256
        ):
            raise RobotProfileError(
                f"asset_variants[{index}].sha256 must be a 64-character hex digest"
            )
        if variant_id in variant_ids:
            raise RobotProfileError(f"asset variant_id must be unique: {variant_id!r}")
        if sha256 in hashes:
            raise RobotProfileError(f"asset variant sha256 must be unique: {sha256}")
        variant_ids.add(variant_id)
        hashes.add(sha256)
        variants.append(RobotAssetVariant(variant_id=variant_id, sha256=sha256))
    return tuple(variants)


def _load_placement(value: Any) -> RobotPlacementProfile:
    raw = _mapping(value, "placement")
    fields = {
        "family",
        "support_role",
        "footprint_m",
        "collision_layers",
        "base_contact_offset_m",
        "authored_forward_yaw_offset_deg",
    }
    if set(raw) != fields:
        raise RobotProfileError(
            f"placement fields must be exactly {sorted(fields)}, got {sorted(raw)}"
        )
    try:
        family = PlacementFamily(_text(raw.get("family"), "placement.family"))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in PlacementFamily)
        raise RobotProfileError(
            f"placement.family must be one of: {allowed}"
        ) from exc
    support_role = _text(raw.get("support_role"), "placement.support_role")
    if family is PlacementFamily.FLOOR_STANDING and support_role != "floor":
        raise RobotProfileError(
            "floor_standing profiles must use placement.support_role=floor"
        )
    footprint_value = raw.get("footprint_m")
    footprint = (
        None
        if footprint_value is None
        else tuple(_numbers(footprint_value, "placement.footprint_m", length=2, positive=True))
    )
    layers: list[RobotCollisionLayer] = []
    for index, item in enumerate(_sequence(raw.get("collision_layers", []), "placement.collision_layers")):
        layer = _mapping(item, f"placement.collision_layers[{index}]")
        center = _numbers(layer.get("center_xy_m"), f"collision_layers[{index}].center_xy_m", length=2)
        size = _numbers(
            layer.get("size_xy_m"),
            f"collision_layers[{index}].size_xy_m",
            length=2,
            positive=True,
        )
        min_z = float(layer.get("min_z_m"))
        max_z = float(layer.get("max_z_m"))
        if not all(math.isfinite(value) for value in (min_z, max_z)) or min_z >= max_z:
            raise RobotProfileError(
                f"placement.collision_layers[{index}] requires min_z_m < max_z_m"
            )
        layers.append(
            RobotCollisionLayer(
                name=_text(layer.get("name"), f"collision_layers[{index}].name"),
                center_xy_m=(center[0], center[1]),
                size_xy_m=(size[0], size[1]),
                min_z_m=min_z,
                max_z_m=max_z,
            )
        )
    base_contact_offset = float(raw.get("base_contact_offset_m", 0.0))
    authored_yaw_offset = float(raw.get("authored_forward_yaw_offset_deg", 0.0))
    if not math.isfinite(base_contact_offset) or not math.isfinite(authored_yaw_offset):
        raise RobotProfileError(
            "placement base_contact_offset_m and authored_forward_yaw_offset_deg must be finite"
        )
    return RobotPlacementProfile(
        family=family,
        support_role=support_role,
        footprint_m=(footprint[0], footprint[1]) if footprint is not None else None,
        collision_layers=tuple(layers),
        base_contact_offset_m=base_contact_offset,
        authored_forward_yaw_offset_deg=authored_yaw_offset,
    )


def _load_base(value: Any) -> RobotBaseProfile:
    raw = _mapping(value, "base")
    operation_mode = _text(raw.get("operation_mode"), "base.operation_mode")
    if operation_mode != "locked":
        raise RobotProfileError("base.operation_mode must be locked in v1")
    groups_raw = _mapping(raw.get("joint_groups", {}), "base.joint_groups")
    groups: dict[str, Mapping[str, Any]] = {}
    for name, item in groups_raw.items():
        group = dict(_mapping(item, f"base.joint_groups.{name}"))
        names = _texts(
            group.get("joint_names", []),
            f"base.joint_groups.{name}.joint_names",
            allow_empty=True,
        )
        indices = _indices(
            group.get("joint_indices", []),
            f"base.joint_groups.{name}.joint_indices",
            allow_empty=True,
        )
        home = _numbers(
            group.get("home", []),
            f"base.joint_groups.{name}.home",
            allow_empty=True,
        )
        if not names and not indices:
            raise RobotProfileError(
                f"base.joint_groups.{name} requires joint_names or joint_indices"
            )
        if home and len(home) not in {len(names), len(indices)}:
            raise RobotProfileError(
                f"base.joint_groups.{name}.home does not match its joints"
            )
        group["joint_names"] = list(names)
        group["joint_indices"] = list(indices)
        group["home"] = list(home)
        groups[str(name)] = group
    hold_raw = _mapping(raw.get("hold", {}), "base.hold")
    hold = {
        "stiffness": float(hold_raw.get("stiffness", 100000.0)),
        "damping": float(hold_raw.get("damping", 3000.0)),
        "max_effort": float(hold_raw.get("max_effort", 10000.0)),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in hold.values()):
        raise RobotProfileError("base.hold values must be finite and positive")
    return RobotBaseProfile(
        operation_mode=operation_mode,
        locked_joint_names=_texts(
            raw.get("locked_joint_names", []),
            "base.locked_joint_names",
            allow_empty=True,
        ),
        joint_groups=groups,
        hold=hold,
    )


def _load_gripper(value: Any, field: str) -> RobotGripperProfile:
    raw = _mapping(value, field)
    keypoints_raw = _mapping(raw.get("keypoints"), f"{field}.keypoints")
    keypoints = {
        str(name): _numbers(point, f"{field}.keypoints.{name}")
        for name, point in keypoints_raw.items()
    }
    action = _mapping(raw.get("action"), f"{field}.action")
    action_adapter = _text(action.get("adapter"), f"{field}.action.adapter")
    if action_adapter != "signed_clip":
        raise RobotProfileError(f"{field}.action.adapter must be signed_clip")
    action_params = dict(
        _mapping(action.get("params"), f"{field}.action.params")
    )
    if set(action_params) != {"command", "invert", "clip"}:
        raise RobotProfileError(
            f"{field}.action.params must contain command, invert, and clip"
        )
    max_width = float(raw.get("max_width"))
    min_width = float(raw.get("min_width"))
    tcp_offset = float(raw.get("tcp_offset"))
    if not all(math.isfinite(value) for value in (min_width, max_width, tcp_offset)):
        raise RobotProfileError(f"{field} width and tcp_offset must be finite")
    if min_width < 0.0 or max_width <= min_width or tcp_offset < 0.0:
        raise RobotProfileError(f"{field} has invalid width or tcp_offset")
    joint_indices = _indices(raw.get("joint_indices"), f"{field}.joint_indices")
    command = _numbers(
        action_params["command"],
        f"{field}.action.params.command",
        length=len(joint_indices),
    )
    clip = _numbers(
        action_params["clip"], f"{field}.action.params.clip", length=2
    )
    if clip[0] >= clip[1] or not isinstance(action_params["invert"], bool):
        raise RobotProfileError(f"{field}.action.params clip or invert is invalid")
    normalized_action_params = {
        "command": list(command),
        "invert": action_params["invert"],
        "clip": list(clip),
    }
    return RobotGripperProfile(
        joint_names=_texts(
            raw.get("joint_names", []), f"{field}.joint_names", allow_empty=True
        ),
        joint_indices=joint_indices,
        max_width=max_width,
        min_width=min_width,
        tcp_offset=tcp_offset,
        home=_numbers(raw.get("home"), f"{field}.home"),
        keypoints=keypoints,
        action_adapter=action_adapter,
        action_params=normalized_action_params,
    )


def _load_arms(value: Any) -> dict[str, RobotArmProfile]:
    raw = _mapping(value, "arms")
    if not raw:
        raise RobotProfileError("arms must not be empty")
    arms: dict[str, RobotArmProfile] = {}
    for arm_id, item in raw.items():
        arm_name = _text(arm_id, "arms key")
        arm = _mapping(item, f"arms.{arm_name}")
        command_names = _texts(
            arm.get("command_joint_names"), f"arms.{arm_name}.command_joint_names"
        )
        trajectory_names = _texts(
            arm.get("trajectory_joint_names"),
            f"arms.{arm_name}.trajectory_joint_names",
        )
        indices = _indices(arm.get("joint_indices"), f"arms.{arm_name}.joint_indices")
        home = _numbers(arm.get("home"), f"arms.{arm_name}.home")
        home_std = _numbers(arm.get("home_std"), f"arms.{arm_name}.home_std")
        lengths = {len(command_names), len(trajectory_names), len(indices), len(home), len(home_std)}
        if len(lengths) != 1:
            raise RobotProfileError(
                f"arms.{arm_name} joint names, indices, home and home_std must have equal lengths"
            )
        arms[arm_name] = RobotArmProfile(
            arm_id=arm_name,
            controller=_text(arm.get("controller"), f"arms.{arm_name}.controller"),
            curobo_file=_text(arm.get("curobo_file"), f"arms.{arm_name}.curobo_file"),
            command_joint_names=command_names,
            trajectory_joint_names=trajectory_names,
            joint_indices=indices,
            ee_path=_text(arm.get("ee_path"), f"arms.{arm_name}.ee_path"),
            base_path=_text(arm.get("base_path"), f"arms.{arm_name}.base_path"),
            filter_paths=_texts(
                arm.get("filter_paths", []),
                f"arms.{arm_name}.filter_paths",
                allow_empty=True,
            ),
            forbid_collision_paths=_texts(
                arm.get("forbid_collision_paths", []),
                f"arms.{arm_name}.forbid_collision_paths",
                allow_empty=True,
            ),
            home=home,
            home_std=home_std,
            arm_base_xy_m=_optional_xy(
                arm.get("arm_base_xy_m"), f"arms.{arm_name}.arm_base_xy_m"
            ),
            gripper=_load_gripper(arm.get("gripper"), f"arms.{arm_name}.gripper"),
        )
    return arms


def _load_camera_rig(value: Any) -> tuple[RobotCameraProfile, ...]:
    cameras: list[RobotCameraProfile] = []
    names: set[str] = set()
    save_names: set[str] = set()
    for index, item in enumerate(_sequence(value, "camera_rig")):
        raw = _mapping(item, f"camera_rig[{index}]")
        name = _text(raw.get("name_suffix"), f"camera_rig[{index}].name_suffix")
        save_name = _text(raw.get("save_name", name), f"camera_rig[{index}].save_name")
        if name in names or save_name in save_names:
            raise RobotProfileError("camera_rig names and save_names must be unique")
        names.add(name)
        save_names.add(save_name)
        translation = _numbers(
            raw.get("translation"), f"camera_rig[{index}].translation", length=3
        )
        orientation = _numbers(
            raw.get("orientation"), f"camera_rig[{index}].orientation", length=4
        )
        cameras.append(
            RobotCameraProfile(
                name_suffix=name,
                save_name=save_name,
                translation=(translation[0], translation[1], translation[2]),
                orientation=(orientation[0], orientation[1], orientation[2], orientation[3]),
                camera_axes=_text(raw.get("camera_axes", "usd"), f"camera_rig[{index}].camera_axes"),
                camera_file=_text(raw.get("camera_file"), f"camera_rig[{index}].camera_file"),
                parent_suffix=str(raw.get("parent_suffix", "")).strip(),
                apply_randomization=bool(raw.get("apply_randomization", False)),
                max_translation_noise=(
                    float(raw["max_translation_noise"])
                    if raw.get("max_translation_noise") is not None
                    else None
                ),
                max_orientation_noise=(
                    float(raw["max_orientation_noise"])
                    if raw.get("max_orientation_noise") is not None
                    else None
                ),
            )
        )
    return tuple(cameras)


def _load_data_adapter(value: Any, arms: Mapping[str, RobotArmProfile]) -> RobotDataAdapter:
    raw = _mapping(value, "data_adapter")
    arm_values = _mapping(raw.get("arms"), "data_adapter.arms")
    if set(arm_values) != set(arms):
        raise RobotProfileError("data_adapter.arms must exactly match arms")
    adapters: dict[str, RobotDataArmAdapter] = {}
    for arm_id, item in arm_values.items():
        adapter = _mapping(item, f"data_adapter.arms.{arm_id}")
        pose = adapter.get("gripper_pose_key")
        action_name = str(adapter.get("action_name", ""))
        if action_name not in {"", "left", "right"}:
            raise RobotProfileError(
                f"data_adapter.arms.{arm_id}.action_name must be left, right, or empty"
            )
        adapters[str(arm_id)] = RobotDataArmAdapter(
            action_name=action_name,
            joint_position_key=_text(
                adapter.get("joint_position_key"),
                f"data_adapter.arms.{arm_id}.joint_position_key",
            ),
            gripper_position_key=_text(
                adapter.get("gripper_position_key"),
                f"data_adapter.arms.{arm_id}.gripper_position_key",
            ),
            gripper_pose_key=(
                None
                if pose is None
                else _text(pose, f"data_adapter.arms.{arm_id}.gripper_pose_key")
            ),
        )
    return RobotDataAdapter(arms=adapters)


def _canonical_hash(raw: Mapping[str, Any]) -> str:
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_physics(value: Any) -> dict[str, Any]:
    raw = dict(_mapping(value, "physics"))
    required = {
        "solver_position_iteration_count",
        "solver_velocity_iteration_count",
        "stabilization_threshold",
    }
    if set(raw) != required:
        raise RobotProfileError(
            f"physics fields must be exactly {sorted(required)}, got {sorted(raw)}"
        )
    position_iterations = int(raw["solver_position_iteration_count"])
    velocity_iterations = int(raw["solver_velocity_iteration_count"])
    stabilization_threshold = float(raw["stabilization_threshold"])
    if (
        position_iterations <= 0
        or velocity_iterations <= 0
        or not math.isfinite(stabilization_threshold)
        or stabilization_threshold < 0.0
    ):
        raise RobotProfileError("physics solver settings are invalid")
    return {
        "solver_position_iteration_count": position_iterations,
        "solver_velocity_iteration_count": velocity_iterations,
        "stabilization_threshold": stabilization_threshold,
    }


def _load_kinematics(value: Any) -> dict[str, Any]:
    raw = dict(_mapping(value, "kinematics"))
    allowed = {"R_ee_graspnet", "ee_axis", "extra_depth_file"}
    if set(raw) - allowed or not {"R_ee_graspnet", "ee_axis"} <= set(raw):
        raise RobotProfileError(
            "kinematics requires R_ee_graspnet and ee_axis, with optional extra_depth_file"
        )
    rows = _sequence(raw["R_ee_graspnet"], "kinematics.R_ee_graspnet")
    if len(rows) != 3:
        raise RobotProfileError("kinematics.R_ee_graspnet must be a 3x3 matrix")
    matrix = [
        list(_numbers(row, f"kinematics.R_ee_graspnet[{index}]", length=3))
        for index, row in enumerate(rows)
    ]
    axis = _text(raw["ee_axis"], "kinematics.ee_axis")
    if axis not in {"x", "y", "z"}:
        raise RobotProfileError("kinematics.ee_axis must be x, y, or z")
    result: dict[str, Any] = {"R_ee_graspnet": matrix, "ee_axis": axis}
    if raw.get("extra_depth_file") is not None:
        result["extra_depth_file"] = _text(
            raw["extra_depth_file"], "kinematics.extra_depth_file"
        )
    return result


def load_robot_profile(path: str | Path) -> RobotModelProfile:
    source_path = Path(path).expanduser().resolve()
    try:
        raw_value = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RobotProfileError(f"robot profile does not exist: {source_path}") from exc
    raw = _mapping(raw_value, "robot profile")
    unknown = set(raw) - _PROFILE_FIELDS
    if unknown:
        raise RobotProfileError(
            f"robot profile contains non-canonical fields: {sorted(unknown)}"
        )
    missing = _REQUIRED_PROFILE_FIELDS - set(raw)
    if missing:
        raise RobotProfileError(f"robot profile is missing fields: {sorted(missing)}")
    arms = _load_arms(raw["arms"])
    capabilities = frozenset(_texts(raw["capabilities"], "capabilities"))
    collision_world_modes = frozenset(
        _texts(raw["collision_world_modes"], "collision_world_modes")
    )
    if not collision_world_modes <= {"physics_schema"}:
        raise RobotProfileError("collision_world_modes contains an unsupported mode")
    return RobotModelProfile(
        profile_id=_text(raw["profile_id"], "profile_id"),
        target_class=_text(raw["target_class"], "target_class"),
        path=_text(raw["path"], "path"),
        asset_variants=_load_asset_variants(raw.get("asset_variants", [])),
        placement=_load_placement(raw["placement"]),
        base=_load_base(raw["base"]),
        arms=arms,
        camera_rig=_load_camera_rig(raw["camera_rig"]),
        data_adapter=_load_data_adapter(raw["data_adapter"], arms),
        capabilities=capabilities,
        collision_world_modes=collision_world_modes,
        physics=_load_physics(raw["physics"]),
        kinematics=_load_kinematics(raw["kinematics"]),
        profile_hash=_canonical_hash(raw),
        source_path=source_path,
    )


def resolve_robot_profile_path(path_value: str | Path, task_path: Path) -> Path:
    path = Path(path_value).expanduser()
    candidates = (
        (path,)
        if path.is_absolute()
        else (Path.cwd() / path, task_path.parent / path, _PROJECT_ROOT / path)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RobotProfileError(
        f"robot_config_file does not exist: {path_value}; candidates={list(map(str, candidates))}"
    )


def load_robot_profile_for_task(
    robot: Mapping[str, Any], task_path: Path
) -> RobotModelProfile:
    config_file = _text(robot.get("robot_config_file"), "robots[].robot_config_file")
    profile = load_robot_profile(resolve_robot_profile_path(config_file, task_path))
    requested_profile = robot.get("profile_id")
    if requested_profile is not None and str(requested_profile) != profile.profile_id:
        raise RobotProfileError(
            f"robot requested profile_id {requested_profile!r}, but {config_file} defines {profile.profile_id!r}"
        )
    return profile


def resolve_robot_asset_path(profile: RobotModelProfile) -> Path:
    """Resolve the profile-owned USD independently of a task's asset root."""

    path = Path(profile.path).expanduser()
    candidates = (
        (path,)
        if path.is_absolute()
        else (
            _PROJECT_ROOT / path,
            _PROJECT_ROOT / "workflows" / "simbox" / "assets" / path,
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.absolute()
    raise RobotProfileError(
        f"robot profile asset does not exist: {profile.profile_id} "
        f"path={profile.path!r}; candidates={list(map(str, candidates))}"
    )


def _runtime_robot_asset_path(profile: RobotModelProfile) -> str:
    path = resolve_robot_asset_path(profile)
    try:
        return str(path.relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(path)


def project_runtime_config(
    profile: RobotModelProfile,
    overrides: Mapping[str, Any] | None = None,
    *,
    task_path: Path | None = None,
    asset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Project the canonical schema into the runtime's established flat attributes."""

    physics = profile.physics
    kinematics = profile.kinematics
    config: dict[str, Any] = {
        "profile_id": profile.profile_id,
        "profile_hash": profile.profile_hash,
        "target_class": profile.target_class,
        "path": _runtime_robot_asset_path(profile),
        "placement": {
            "family": profile.placement.family.value,
            "support_role": profile.placement.support_role,
            "footprint_m": (
                list(profile.placement.footprint_m)
                if profile.placement.footprint_m is not None
                else None
            ),
            "collision_layers": [
                {
                    "name": layer.name,
                    "center_xy_m": list(layer.center_xy_m),
                    "size_xy_m": list(layer.size_xy_m),
                    "min_z_m": layer.min_z_m,
                    "max_z_m": layer.max_z_m,
                }
                for layer in profile.placement.collision_layers
            ],
            "base_contact_offset_m": profile.placement.base_contact_offset_m,
            "authored_forward_yaw_offset_deg": (
                profile.placement.authored_forward_yaw_offset_deg
            ),
        },
        "base": {
            "operation_mode": profile.base.operation_mode,
            "locked_joint_names": list(profile.base.locked_joint_names),
        },
        "arms": {},
        "data_adapter": {
            "arms": {
                arm_id: {
                    "action_name": adapter.action_name,
                    "joint_position_key": adapter.joint_position_key,
                    "gripper_position_key": adapter.gripper_position_key,
                    "gripper_pose_key": adapter.gripper_pose_key,
                }
                for arm_id, adapter in profile.data_adapter.arms.items()
            }
        },
        "capabilities": sorted(profile.capabilities),
        "collision_world_modes": sorted(profile.collision_world_modes),
        "solver_position_iteration_count": int(physics["solver_position_iteration_count"]),
        "solver_velocity_iteration_count": int(physics["solver_velocity_iteration_count"]),
        "stabilization_threshold": float(physics["stabilization_threshold"]),
        "R_ee_graspnet": copy.deepcopy(kinematics["R_ee_graspnet"]),
        "ee_axis": str(kinematics["ee_axis"]),
    }
    if kinematics.get("extra_depth_file"):
        config["extra_depth_file"] = str(kinematics["extra_depth_file"])

    prefixes = {"left": ("left", "fl"), "right": ("right", "fr")}
    for arm_id, arm in profile.arms.items():
        config["arms"][arm_id] = {
            "arm_id": arm_id,
            "controller": arm.controller,
            "curobo_file": arm.curobo_file,
            "command_joint_names": list(arm.command_joint_names),
            "trajectory_joint_names": list(arm.trajectory_joint_names),
            "joint_indices": list(arm.joint_indices),
            "base_path": arm.base_path,
            "ee_path": arm.ee_path,
            "gripper": {
                "joint_names": list(arm.gripper.joint_names),
                "joint_indices": list(arm.gripper.joint_indices),
                "action": {
                    "adapter": arm.gripper.action_adapter,
                    **copy.deepcopy(dict(arm.gripper.action_params)),
                },
            },
        }
        if arm_id not in prefixes:
            continue
        long_prefix, short_prefix = prefixes[arm_id]
        config[f"{long_prefix}_joint_names"] = list(arm.command_joint_names)
        config[f"{long_prefix}_joint_indices"] = list(arm.joint_indices)
        if arm.gripper.joint_names:
            config[f"{long_prefix}_gripper_names"] = list(arm.gripper.joint_names)
        config[f"{long_prefix}_gripper_indices"] = list(arm.gripper.joint_indices)
        config[f"{long_prefix}_joint_home"] = list(arm.home)
        config[f"{long_prefix}_joint_home_std"] = list(arm.home_std)
        config[f"{long_prefix}_gripper_home"] = list(arm.gripper.home)
        config[f"{short_prefix}_ee_path"] = arm.ee_path
        config[f"{short_prefix}_base_path"] = arm.base_path
        config[f"{short_prefix}_gripper_keypoints"] = {
            name: list(point) for name, point in arm.gripper.keypoints.items()
        }
        config[f"{short_prefix}_filter_paths"] = list(arm.filter_paths)
        config[f"{short_prefix}_forbid_collision_paths"] = list(
            arm.forbid_collision_paths
        )

    grippers = [arm.gripper for arm in profile.arms.values()]
    gripper_contracts = {
        (gripper.max_width, gripper.min_width, gripper.tcp_offset)
        for gripper in grippers
    }
    if len(gripper_contracts) != 1:
        raise RobotProfileError(
            "runtime projection requires one shared gripper width/tcp contract"
        )
    max_width, min_width, tcp_offset = next(iter(gripper_contracts))
    config.update(
        {
            "gripper_max_width": max_width,
            "gripper_min_width": min_width,
            "tcp_offset": tcp_offset,
        }
    )

    for group_name, group in profile.base.joint_groups.items():
        config[f"{group_name}_names"] = list(group.get("joint_names", []))
        config[f"{group_name}_indices"] = list(group.get("joint_indices", []))
        config[f"{group_name}_home"] = list(group.get("home", []))
    if profile.base.locked_joint_names:
        config["manipulation_base_hold"] = {
            "enabled": True,
            "joint_names": list(profile.base.locked_joint_names),
            **dict(profile.base.hold),
        }
    if overrides:
        normalized_overrides = copy.deepcopy(dict(overrides))
        asserted_target_class = normalized_overrides.pop("target_class", None)
        if (
            asserted_target_class is not None
            and str(asserted_target_class) != profile.target_class
        ):
            raise RobotProfileError(
                "PROFILE_TARGET_CLASS_MISMATCH: robot instance declares "
                f"{asserted_target_class!r}, profile defines {profile.target_class!r}"
            )
        asserted_path = normalized_overrides.pop("path", None)
        if asserted_path is not None:
            config["path"] = _runtime_asset_path(
                _validate_instance_asset_path(
                    profile,
                    str(asserted_path),
                    task_path=task_path,
                    asset_root=asset_root,
                )
            )
        unknown_overrides = set(normalized_overrides) - _INSTANCE_OVERRIDE_FIELDS
        if unknown_overrides:
            raise RobotProfileError(
                "robot instance attempts to override canonical fields: "
                f"{sorted(unknown_overrides)}"
            )
        config.update(normalized_overrides)
    return config


def _asset_candidates(
    path_value: str,
    *,
    task_path: Path | None,
    asset_root: str | Path | None,
) -> tuple[Path, ...]:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return (path,)
    candidates: list[Path] = []
    if asset_root is not None:
        candidates.append(Path(asset_root).expanduser() / path)
    if task_path is not None:
        candidates.append(task_path.parent / path)
    candidates.extend(
        (
            Path.cwd() / path,
            _PROJECT_ROOT / path,
            _PROJECT_ROOT / "workflows" / "simbox" / "assets" / path,
        )
    )
    return tuple(candidates)


def _resolve_existing_asset(
    path_value: str,
    *,
    task_path: Path | None,
    asset_root: str | Path | None,
) -> Path | None:
    for candidate in _asset_candidates(
        path_value, task_path=task_path, asset_root=asset_root
    ):
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_asset_path(path: Path) -> str:
    return str(path.resolve())


def _validate_instance_asset_path(
    profile: RobotModelProfile,
    asserted_path: str,
    *,
    task_path: Path | None,
    asset_root: str | Path | None,
) -> Path:
    asserted_asset = _resolve_existing_asset(
        asserted_path, task_path=task_path, asset_root=asset_root
    )
    canonical_asset = resolve_robot_asset_path(profile).resolve()
    if asserted_asset is not None:
        if asserted_asset == canonical_asset:
            return asserted_asset
        asserted_hash = _file_hash(asserted_asset)
        accepted_hashes = {variant.sha256 for variant in profile.asset_variants}
        accepted_hashes.add(_file_hash(canonical_asset))
        if asserted_hash in accepted_hashes:
            return asserted_asset
    raise RobotProfileError(
        "PROFILE_ASSET_MISMATCH: robot instance path does not resolve to the "
        f"canonical asset or a registered variant; instance={asserted_path!r} "
        f"resolved={asserted_asset}, "
        f"profile={profile.path!r} resolved={canonical_asset}"
    )
