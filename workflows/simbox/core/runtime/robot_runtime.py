"""Robot and arm joint-order contracts independent of Isaac Sim.

Isaac articulation DOF order is an asset/runtime property and is not safe to
assume from YAML list positions.  ``RobotRuntime`` resolves named arm groups
against the observed runtime order once, then provides small reorder helpers
for native joint-state objects, mappings, and ordinary sequences.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


class RobotRuntimeError(ValueError):
    """Base class for invalid robot runtime contracts."""


class JointOrderError(RobotRuntimeError):
    """Raised when named joints cannot be resolved unambiguously."""


def _names(values: Sequence[Any] | None, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise JointOrderError(f"{field_name} must be a sequence of joint names")
    result = tuple(str(value) for value in values)
    if any(not value for value in result):
        raise JointOrderError(f"{field_name} contains an empty joint name")
    duplicate = sorted({value for value in result if result.count(value) > 1})
    if duplicate:
        raise JointOrderError(f"{field_name} contains duplicate names: {duplicate}")
    return result


@dataclass(frozen=True, init=False)
class ArmSpec:
    """Stable named contract for one manipulator arm."""

    name: str
    joint_names: tuple[str, ...]
    gripper_names: tuple[str, ...]
    tool_frame: str | None
    base_frame: str | None
    joint_indices: tuple[int, ...]
    gripper_indices: tuple[int, ...]
    metadata: Mapping[str, Any]

    def __init__(
        self,
        name: str = "arm",
        joint_names: Sequence[Any] | None = None,
        *,
        joints: Sequence[Any] | None = None,
        planner_joints: Sequence[Any] | Mapping[str, Sequence[Any]] | None = None,
        control_joints: Mapping[str, Sequence[Any]] | Sequence[Any] | None = None,
        gripper_names: Sequence[Any] | None = None,
        gripper_joints: Sequence[Any] | None = None,
        tool_frame: str | None = None,
        ee_link: str | None = None,
        base_frame: str | None = None,
        joint_indices: Sequence[int] | None = None,
        gripper_indices: Sequence[int] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if joint_names is None:
            joint_names = joints
        if joint_names is None:
            if isinstance(planner_joints, Mapping):
                if name in planner_joints:
                    joint_names = planner_joints[name]
                elif len(planner_joints) == 1:
                    joint_names = next(iter(planner_joints.values()))
            else:
                joint_names = planner_joints
        if gripper_names is None:
            gripper_names = gripper_joints
        resolved_joint_names = _names(joint_names, f"{name}.joint_names")
        if not resolved_joint_names:
            raise JointOrderError(f"{name}.joint_names must not be empty")
        resolved_gripper_names = _names(gripper_names, f"{name}.gripper_names")
        resolved_joint_indices = tuple(int(value) for value in (joint_indices or ()))
        resolved_gripper_indices = tuple(int(value) for value in (gripper_indices or ()))
        if len(resolved_joint_indices) not in {0, len(resolved_joint_names)}:
            raise JointOrderError(
                f"{name}.joint_indices length does not match joint_names"
            )
        if len(resolved_gripper_indices) not in {0, len(resolved_gripper_names)}:
            raise JointOrderError(
                f"{name}.gripper_indices length does not match gripper_names"
            )
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "joint_names", resolved_joint_names)
        object.__setattr__(self, "gripper_names", resolved_gripper_names)
        object.__setattr__(self, "tool_frame", tool_frame if tool_frame is not None else ee_link)
        object.__setattr__(self, "base_frame", base_frame)
        object.__setattr__(self, "joint_indices", resolved_joint_indices)
        object.__setattr__(self, "gripper_indices", resolved_gripper_indices)
        metadata_value = dict(metadata or {})
        if planner_joints is not None:
            metadata_value.setdefault("planner_joints", planner_joints)
        if control_joints is not None:
            metadata_value.setdefault("control_joints", control_joints)
        object.__setattr__(self, "metadata", MappingProxyType(metadata_value))

    @classmethod
    def from_mapping(
        cls,
        name: str,
        value: Mapping[str, Any],
        *,
        prefix: str | None = None,
    ) -> "ArmSpec":
        if isinstance(value, cls):
            return value
        raw = dict(value)
        prefix = prefix or str(name)
        # Both arm-local mappings and a complete robot config are accepted.
        planner_raw = raw.get("planner_joints")
        joint_names = raw.get("joint_names", raw.get("joints"))
        if joint_names is None and isinstance(planner_raw, Mapping):
            joint_names = planner_raw.get(name, planner_raw.get(prefix))
            if joint_names is None and len(planner_raw) == 1:
                joint_names = next(iter(planner_raw.values()))
        elif joint_names is None:
            joint_names = planner_raw
        if joint_names is None:
            joint_names = raw.get(f"{prefix}_joint_names")
        if joint_names is None and prefix in {"left", "right"}:
            joint_names = raw.get(f"{prefix}_joint_names")
        if joint_names is None:
            joint_indices = raw.get(f"{prefix}_joint_indices", raw.get("joint_indices"))
            # Numeric-only contracts cannot provide a stable named reorder;
            # make the omission explicit rather than inventing names.
            if joint_indices:
                raise JointOrderError(
                    f"{prefix} arm requires authoritative {prefix}_joint_names"
                )
        gripper_names = raw.get("gripper_names", raw.get("gripper_joints"))
        if gripper_names is None:
            gripper_names = raw.get(f"{prefix}_gripper_names")
        joint_indices = raw.get("joint_indices", raw.get(f"{prefix}_joint_indices"))
        gripper_indices = raw.get("gripper_indices", raw.get(f"{prefix}_gripper_indices"))
        metadata = raw.get("metadata", {})
        return cls(
            name,
            joint_names,
            planner_joints=raw.get("planner_joints"),
            control_joints=raw.get("control_joints"),
            gripper_names=gripper_names,
            tool_frame=raw.get("tool_frame", raw.get("ee_link", raw.get(f"{prefix}_ee_link"))),
            base_frame=raw.get("base_frame", raw.get(f"{prefix}_base_frame")),
            joint_indices=joint_indices,
            gripper_indices=gripper_indices,
            metadata=metadata,
        )

    @property
    def all_joint_names(self) -> tuple[str, ...]:
        return self.joint_names + self.gripper_names

    # Compatibility accessors mirror the controller ArmSpec vocabulary while
    # retaining a one-arm ``ArmSpec`` value object for RobotRuntime.
    @property
    def planner_joints(self) -> tuple[str, ...]:
        return self.joint_names

    def planner_joint_names(self, arm: str | None = None) -> tuple[str, ...]:
        if arm is None or str(arm) == self.name:
            return self.joint_names
        configured = self.metadata.get("planner_joints")
        if isinstance(configured, Mapping) and arm in configured:
            return _names(configured[arm], f"{arm}.planner_joints")
        raise JointOrderError(f"arm {arm!r} is not configured in planner_joints")

    def control_joint_names(self, arm: str | None = None) -> tuple[str, ...]:
        arm = self.name if arm is None else str(arm)
        configured = self.metadata.get("control_joints")
        if isinstance(configured, Mapping):
            if arm not in configured:
                raise JointOrderError(f"arm {arm!r} is not configured in control_joints")
            names = _names(configured[arm], f"{arm}.control_joints")
            if len(names) != len(self.joint_names):
                raise JointOrderError(
                    f"control joint count mismatch for {arm}: "
                    f"planner={len(self.joint_names)}, control={len(names)}"
                )
            return names
        if configured is not None:
            names = _names(configured, "control_joints")
            if len(names) != len(self.joint_names):
                raise JointOrderError("control joint count does not match planner joints")
            return names
        return self.joint_names


def _read_joint_state(state: Any) -> tuple[tuple[str, ...] | None, Any, str | None]:
    if isinstance(state, Mapping):
        names = state.get("joint_names", state.get("names"))
        values = state.get("position", state.get("positions", state.get("values")))
        key = "position" if "position" in state else "positions" if "positions" in state else "values"
        return (None if names is None else tuple(str(value) for value in names), values, key)
    names = getattr(state, "joint_names", getattr(state, "names", None))
    values = getattr(state, "position", getattr(state, "positions", getattr(state, "values", None)))
    key = "position" if hasattr(state, "position") else "positions" if hasattr(state, "positions") else "values"
    return (None if names is None else tuple(str(value) for value in names), values, key)


def _reorder_sequence(values: Any, indices: Sequence[int]) -> Any:
    if values is None:
        return None
    if hasattr(values, "__getitem__"):
        try:
            # Preserve numpy/torch-like vector types where practical.
            return values[list(indices)]
        except Exception:
            pass
    return type(values)(values[index] for index in indices) if not isinstance(values, tuple) else tuple(values[index] for index in indices)


class RobotRuntime:
    """Resolve and apply arm joint ordering for one observed robot."""

    def __init__(
        self,
        dof_names: Sequence[Any] | None = None,
        arms: Mapping[str, ArmSpec | Mapping[str, Any]] | None = None,
        *,
        runtime_dof_names: Sequence[Any] | None = None,
        config: Mapping[str, Any] | None = None,
        strict: bool = True,
    ) -> None:
        if dof_names is None:
            dof_names = runtime_dof_names
        self.dof_names = _names(dof_names, "runtime dof_names")
        if not self.dof_names:
            raise JointOrderError("runtime dof_names must not be empty")
        duplicate = sorted({value for value in self.dof_names if self.dof_names.count(value) > 1})
        if duplicate:
            raise JointOrderError(f"runtime DOF names are ambiguous: {duplicate}")
        self.strict = bool(strict)
        raw_config = dict(config or {})
        if arms is None:
            arms = self._arms_from_config(raw_config)
        self.arms: dict[str, ArmSpec] = {}
        for name, value in (arms or {}).items():
            spec = value if isinstance(value, ArmSpec) else ArmSpec.from_mapping(str(name), value)
            self.arms[str(name)] = spec
        self._indices: dict[str, tuple[int, ...]] = {}
        self._gripper_indices: dict[str, tuple[int, ...]] = {}
        for name, spec in self.arms.items():
            self._indices[name] = self._resolve(spec.joint_names, f"{name}.joint_names")
            self._gripper_indices[name] = self._resolve(spec.gripper_names, f"{name}.gripper_names")

    @staticmethod
    def _arms_from_config(config: Mapping[str, Any]) -> dict[str, ArmSpec]:
        nested = config.get("arms")
        if isinstance(nested, Mapping):
            return {
                str(name): value if isinstance(value, ArmSpec) else ArmSpec.from_mapping(str(name), value)
                for name, value in nested.items()
            }
        result: dict[str, ArmSpec] = {}
        for name in ("left", "right"):
            names = config.get(f"{name}_joint_names")
            if names:
                result[name] = ArmSpec.from_mapping(name, config)
        return result

    def _resolve(self, requested: Sequence[str], group: str) -> tuple[int, ...]:
        requested = _names(requested, group)
        missing = [name for name in requested if name not in self.dof_names]
        if missing:
            raise JointOrderError(
                f"runtime DOFs are missing {group} names {missing}; all DOFs={list(self.dof_names)}"
            )
        return tuple(self.dof_names.index(name) for name in requested)

    def arm(self, name: str) -> ArmSpec:
        try:
            return self.arms[str(name)]
        except KeyError as exc:
            raise RobotRuntimeError(f"unknown arm {name!r}; available={sorted(self.arms)}") from exc

    get_arm = arm

    def arm_indices(self, name: str) -> tuple[int, ...]:
        self.arm(name)
        return self._indices[str(name)]

    def gripper_indices(self, name: str) -> tuple[int, ...]:
        self.arm(name)
        return self._gripper_indices[str(name)]

    def indices(self, name: str, *, include_gripper: bool = False) -> tuple[int, ...]:
        result = self.arm_indices(name)
        if include_gripper:
            result += self.gripper_indices(name)
        return result

    def reorder(
        self,
        values: Any,
        source_names: Sequence[Any] | None = None,
        target_names: Sequence[Any] | None = None,
    ) -> Any:
        """Reorder a vector/state from ``source_names`` into ``target_names``.

        If a native state object exposes ``reorder(names)``, that method is
        preferred so derivatives and metadata stay intact.  Raw mappings and
        vectors are handled without importing numpy or torch.
        """

        target = _names(target_names, "target_names") if target_names is not None else self.dof_names
        if hasattr(values, "reorder") and callable(values.reorder):
            try:
                return values.reorder(list(target))
            except (TypeError, ValueError):
                return values.reorder(target)
        if isinstance(values, Mapping):
            names = source_names or values.get("joint_names", values.get("names"))
            if names is None:
                return {name: values[name] for name in target}
            names = _names(names, "source_names")
            missing = [name for name in target if name not in names]
            if missing:
                raise JointOrderError(f"source vector is missing target joints {missing}")
            vector = values.get("position", values.get("positions", values.get("values")))
            if vector is None:
                return {name: values[name] for name in target}
            index = [names.index(name) for name in target]
            return _reorder_sequence(vector, index)
        names, vector, _ = _read_joint_state(values)
        if names is None:
            if source_names is None:
                raise JointOrderError("source_names are required for an unnamed joint vector")
            names = _names(source_names, "source_names")
            vector = values
        else:
            names = _names(source_names or names, "source_names")
        missing = [name for name in target if name not in names]
        if missing:
            raise JointOrderError(f"source vector is missing target joints {missing}")
        indices = [names.index(name) for name in target]
        return _reorder_sequence(vector, indices)

    reorder_joints = reorder
    reorder_joint_state = reorder

    def arm_values(self, values: Any, arm: str, *, include_gripper: bool = False) -> Any:
        spec = self.arm(arm)
        names = spec.all_joint_names if include_gripper else spec.joint_names
        return self.reorder(values, target_names=names)

    extract_arm = arm_values
    arm_joint_state = arm_values

    def to_runtime_order(self, values: Any, source_names: Sequence[Any]) -> Any:
        return self.reorder(values, source_names=source_names, target_names=self.dof_names)

    def to_planner_order(self, values: Any, planner_names: Sequence[Any]) -> Any:
        return self.reorder(values, source_names=self.dof_names, target_names=planner_names)

    def describe(self) -> dict[str, Any]:
        return {
            "dof_names": list(self.dof_names),
            "arms": {
                name: {
                    "joint_names": list(spec.joint_names),
                    "gripper_names": list(spec.gripper_names),
                    "joint_indices": list(self._indices[name]),
                    "gripper_indices": list(self._gripper_indices[name]),
                    "tool_frame": spec.tool_frame,
                    "base_frame": spec.base_frame,
                }
                for name, spec in self.arms.items()
            },
        }

    @classmethod
    def from_config(
        cls,
        dof_names: Sequence[Any],
        config: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> "RobotRuntime":
        return cls(dof_names, config=config, strict=strict)


__all__ = ["ArmSpec", "JointOrderError", "RobotRuntime", "RobotRuntimeError"]
