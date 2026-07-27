"""Physics-schema collision discovery and manipulation-object state management."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

LOGGER = logging.getLogger("de_logger")
SUPPORTED_COLLIDER_TYPES = (
    UsdGeom.Mesh,
    UsdGeom.Cube,
    UsdGeom.Sphere,
    UsdGeom.Cylinder,
    UsdGeom.Capsule,
)


class CollisionSceneError(RuntimeError):
    """Raised when Physics and CuRobo cannot share an auditable world."""


class CollisionObjectState(str, Enum):
    WORLD_OBSTACLE = "world_obstacle"
    ACTIVE_TARGET_TRANSIT = "active_target_transit"
    ACTIVE_TARGET_APPROACH = "active_target_approach"
    ATTACHED = "attached"
    PLACEMENT_CONTACT = "placement_contact"
    PLACED_WORLD = "placed_world"
    DISABLED = "disabled"


@dataclass
class CollisionObjectRecord:
    entity_name: str
    root_prim_path: str
    collision_prim_paths: list[str]
    mobility: str
    tracking_prim_path: str | None = None
    state: CollisionObjectState = CollisionObjectState.WORLD_OBSTACLE
    owner_robot: str | None = None
    owner_arm: str | None = None
    pose_revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        return value


_ALLOWED_TRANSITIONS = {
    CollisionObjectState.WORLD_OBSTACLE: {
        CollisionObjectState.ACTIVE_TARGET_TRANSIT,
        CollisionObjectState.DISABLED,
    },
    CollisionObjectState.PLACED_WORLD: {
        CollisionObjectState.ACTIVE_TARGET_TRANSIT,
        CollisionObjectState.ACTIVE_TARGET_APPROACH,
        CollisionObjectState.WORLD_OBSTACLE,
        CollisionObjectState.DISABLED,
    },
    CollisionObjectState.ACTIVE_TARGET_TRANSIT: {
        CollisionObjectState.ACTIVE_TARGET_APPROACH,
        CollisionObjectState.WORLD_OBSTACLE,
    },
    CollisionObjectState.ACTIVE_TARGET_APPROACH: {
        CollisionObjectState.ACTIVE_TARGET_TRANSIT,
        CollisionObjectState.ATTACHED,
        CollisionObjectState.PLACED_WORLD,
        CollisionObjectState.WORLD_OBSTACLE,
    },
    CollisionObjectState.ATTACHED: {
        CollisionObjectState.PLACEMENT_CONTACT,
        CollisionObjectState.PLACED_WORLD,
    },
    CollisionObjectState.PLACEMENT_CONTACT: {
        CollisionObjectState.ATTACHED,
        CollisionObjectState.PLACED_WORLD,
    },
    CollisionObjectState.DISABLED: {CollisionObjectState.WORLD_OBSTACLE},
}


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


def validate_exact_exclusions(values: Iterable[Any] | None) -> dict[str, str]:
    """Validate exact, reasoned exclusions and return path -> reason."""

    result: dict[str, str] = {}
    for value in values or []:
        if not hasattr(value, "get"):
            raise ValueError("exact_exclusions entries must contain prim_path and reason")
        path = str(value.get("prim_path", "")).strip()
        reason = str(value.get("reason", "")).strip()
        sdf_path = Sdf.Path(path)
        if not path or not sdf_path.IsAbsolutePath() or not sdf_path.IsPrimPath():
            raise ValueError(f"exact exclusion must use a complete absolute Prim path: {path!r}")
        if not reason:
            raise ValueError(f"exact exclusion requires a non-empty reason: {path}")
        if path in result:
            raise ValueError(f"duplicate exact exclusion: {path}")
        result[path] = reason
    return result


class CollisionSceneManager:
    """Own the exact mapping between Stage Physics colliders and CuRobo worlds."""

    def __init__(
        self,
        stage: Usd.Stage,
        task: Any,
        config: Any | None = None,
        execution_safety_config: Any | None = None,
    ):
        self.stage = stage
        self.task = task
        self.config = config or {}
        self.strict = bool(_cfg_get(self.config, "strict", True))
        self.mode = str(_cfg_get(self.config, "mode", "physics_schema"))
        if self.mode != "physics_schema":
            raise ValueError(f"CollisionSceneManager only supports physics_schema, got {self.mode!r}")
        self.exclusions = validate_exact_exclusions(_cfg_get(self.config, "exact_exclusions", []))
        self.dynamic_translation_replan_m = float(
            _cfg_get(execution_safety_config, "dynamic_translation_replan_m", 0.01)
        )
        self.dynamic_rotation_replan_deg = float(
            _cfg_get(execution_safety_config, "dynamic_rotation_replan_deg", 3.0)
        )
        self.schema_exclusions: dict[str, str] = {}
        self.records: dict[str, CollisionObjectRecord] = {}
        self.attach_prim_paths: dict[str, list[str]] = {}
        self.path_to_entity: dict[str, str] = {}
        self.controllers: dict[tuple[str, str], Any] = {}
        self.controller_enabled: dict[tuple[str, str], dict[str, bool]] = {}
        self.controller_audits: dict[str, dict[str, list[str]]] = {}
        self._temporary_disabled: dict[tuple[str, str], set[str]] = {}
        self._pending_detach: set[str] = set()
        self._retreating_placed: set[str] = set()
        self._attached_relative_pose: dict[str, np.ndarray] = {}
        self.object_state_events: list[dict[str, Any]] = []
        self.world_revision = 0
        self._step_id = 0
        self._pose_matrices: dict[str, np.ndarray] = {}
        self._robot_environment_contact_views: dict[tuple[str, str], list[Any]] = {}
        self._finger_environment_contact_views: dict[tuple[str, str], list[Any]] = {}
        self._object_environment_contact_views: dict[str, list[Any]] = {}
        self._object_environment_filter_paths: dict[str, list[str]] = {}
        self._usd_helper = None
        self._discover()

    def _helper(self):
        if self._usd_helper is None:
            # Lazy import keeps Physics-schema discovery unit-testable outside
            # the Isaac/CuRobo Python environment.
            from curobo.util.usd_helper import UsdHelper

            self._usd_helper = UsdHelper()
            self._usd_helper.load_stage(self.stage)
        return self._usd_helper

    @staticmethod
    def _entity_root(entity: Any) -> str | None:
        for field in ("base_prim_path", "object_prim_path", "prim_path", "rigid_prim_path"):
            value = getattr(entity, field, None)
            if value:
                return str(value)
        return None

    @staticmethod
    def _is_supported(prim: Usd.Prim) -> bool:
        return any(prim.IsA(schema) for schema in SUPPORTED_COLLIDER_TYPES)

    @staticmethod
    def _explicitly_noncollidable(entity: Any) -> bool:
        """Return whether config explicitly declares an entity visual-only.

        Stage Physics schema remains the only source used to *register* an
        obstacle. Config is consulted only to distinguish an intentional
        visual-only entity from a broken asset that claims collision but has
        no enabled ``CollisionAPI`` at runtime.
        """

        cfg = getattr(entity, "cfg", {}) or {}
        physics = cfg.get("physics", {}) or {}
        source_physics = cfg.get("source_physics", {}) or {}
        explicit_flags = [
            physics.get("collision_enabled"),
            cfg.get("collision_enabled"),
            cfg.get("collision"),
            source_physics.get("collision_enabled"),
        ]
        if any(value is True for value in explicit_flags):
            return False
        return any(value is False for value in explicit_flags) or str(
            cfg.get("collider", "")
        ).lower() in {"none", "disabled"}

    @staticmethod
    def _mobility(root: Usd.Prim, collision_prims: list[Usd.Prim]) -> str:
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                return "articulated"
        root_path = root.GetPath()
        for collider in collision_prims:
            current = collider
            while current and current.IsValid() and current.GetPath().HasPrefix(root_path):
                if current.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid = UsdPhysics.RigidBodyAPI(current)
                    kinematic = rigid.GetKinematicEnabledAttr().Get()
                    return "kinematic" if kinematic is True else "dynamic"
                current = current.GetParent()
        return "static"

    def _iter_entities(self):
        seen: set[int] = set()
        for collection_name in ("fixtures", "objects", "distractors"):
            collection = getattr(self.task, collection_name, {}) or {}
            for name, entity in collection.items():
                if id(entity) in seen:
                    continue
                seen.add(id(entity))
                yield str(name), entity

    def _configured_manipulation_entities(self) -> set[str]:
        """Collect standard Pick/Place active objects without name heuristics."""

        result: set[str] = set()

        def visit(value):
            if isinstance(value, Mapping):
                skill_name = str(value.get("name", "")).lower()
                objects = value.get("objects", [])
                if skill_name in {"pick", "place"} and objects:
                    result.add(str(objects[0]))
                for child in value.values():
                    visit(child)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for child in value:
                    visit(child)

        visit(getattr(self.task, "cfg", {}).get("skills", []))
        return result

    def _rigid_body_paths(self, record: CollisionObjectRecord) -> list[str]:
        paths: set[str] = set()
        root_path = Sdf.Path(record.root_prim_path)
        for collision_path in record.collision_prim_paths:
            current = self.stage.GetPrimAtPath(collision_path)
            while current and current.IsValid() and current.GetPath().HasPrefix(root_path):
                if current.HasAPI(UsdPhysics.RigidBodyAPI):
                    paths.add(str(current.GetPath()))
                    break
                current = current.GetParent()
        return sorted(paths)

    def _rigid_body_ancestor_path(
        self, prim_path: str, root_prim_path: str
    ) -> str | None:
        """Resolve the Physics body that actually carries an exact collider."""

        root_path = Sdf.Path(root_prim_path)
        current = self.stage.GetPrimAtPath(prim_path)
        while current and current.IsValid() and current.GetPath().HasPrefix(root_path):
            if current.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(current.GetPath())
            current = current.GetParent()
        return None

    def _discover(self) -> None:
        discovered_paths: set[str] = set()
        matched_exclusions: set[str] = set()
        for entity_name, entity in self._iter_entities():
            root_path = self._entity_root(entity)
            if not root_path:
                if self.strict:
                    raise CollisionSceneError(f"entity {entity_name!r} has no resolvable root Prim path")
                continue
            root = self.stage.GetPrimAtPath(root_path)
            if not root or not root.IsValid():
                raise CollisionSceneError(f"entity root does not exist: {entity_name} -> {root_path}")

            enabled_collision_prims: list[Usd.Prim] = []
            for prim in Usd.PrimRange(root):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                path = str(prim.GetPath())
                enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                if enabled is False:
                    continue
                if path in self.exclusions:
                    matched_exclusions.add(path)
                    continue
                enabled_collision_prims.append(prim)

            collider_prims = [prim for prim in enabled_collision_prims if self._is_supported(prim)]
            supported_paths = [prim.GetPath() for prim in collider_prims]
            for prim in enabled_collision_prims:
                if self._is_supported(prim):
                    continue
                path = str(prim.GetPath())
                has_supported_descendant = any(
                    supported_path.HasPrefix(prim.GetPath()) and supported_path != prim.GetPath()
                    for supported_path in supported_paths
                )
                if has_supported_descendant:
                    self.schema_exclusions[path] = (
                        "non_geometry_collision_api_with_supported_descendant_colliders"
                    )
                    LOGGER.warning(
                        "[CollisionWorld] audit-only schema exclusion path=%s type=%s; "
                        "supported descendant colliders are authoritative",
                        path,
                        prim.GetTypeName(),
                    )
                    continue
                message = f"unsupported enabled CollisionAPI type {prim.GetTypeName()!r}: {path}"
                if self.strict:
                    raise CollisionSceneError(message)
                self.schema_exclusions[path] = "unsupported_collision_prim_non_strict"
                LOGGER.warning("[CollisionWorld] %s", message)

            for prim in collider_prims:
                path = str(prim.GetPath())
                if path in discovered_paths:
                    raise CollisionSceneError(f"collision Prim belongs to multiple entities: {path}")
                discovered_paths.add(path)

            if not collider_prims:
                if self._explicitly_noncollidable(entity):
                    self.schema_exclusions[root_path] = (
                        "config_declared_visual_only_and_stage_has_no_enabled_collider"
                    )
                    continue
                message = f"entity has no supported enabled collider: {entity_name} ({root_path})"
                if self.strict:
                    raise CollisionSceneError(message)
                LOGGER.warning("[CollisionWorld] %s", message)
                continue
            paths = [str(prim.GetPath()) for prim in collider_prims]
            configured_attach_paths = [
                str(path)
                for path in getattr(entity, "attach_collision_prim_paths", [])
            ]
            if configured_attach_paths:
                invalid_attach_paths = sorted(set(configured_attach_paths) - set(paths))
                if invalid_attach_paths:
                    raise CollisionSceneError(
                        f"attach collision Prim is not an enabled collider of {entity_name}: "
                        f"{invalid_attach_paths}"
                    )
                self.attach_prim_paths[entity_name] = configured_attach_paths
            tracking_source = (
                configured_attach_paths[0]
                if configured_attach_paths
                else paths[0]
            )
            tracking_prim_path = self._rigid_body_ancestor_path(
                tracking_source, root_path
            ) or root_path
            record = CollisionObjectRecord(
                entity_name=entity_name,
                root_prim_path=root_path,
                collision_prim_paths=paths,
                mobility=self._mobility(root, collider_prims),
                tracking_prim_path=tracking_prim_path,
            )
            self.records[entity_name] = record
            for path in paths:
                self.path_to_entity[path] = entity_name
                self._pose_matrices[path] = self._world_matrix(path)

        missing_exclusions = sorted(set(self.exclusions) - matched_exclusions)
        if missing_exclusions:
            raise CollisionSceneError(f"exact exclusions do not name enabled Stage colliders: {missing_exclusions}")
        if not self.records:
            raise CollisionSceneError("physics_schema discovered no collision entities")
        for entity_name in self._configured_manipulation_entities():
            if entity_name not in self.records:
                raise CollisionSceneError(
                    f"configured manipulation object has no collision record: {entity_name}"
                )
            if not self.attach_prim_paths.get(entity_name):
                raise CollisionSceneError(
                    f"configured manipulation object has no exact attach collision Prim: {entity_name}"
                )

    def _stage_collision_paths(self) -> set[str]:
        return {
            str(prim.GetPath())
            for prim in self.stage.Traverse()
            if prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
        }

    def _world_matrix(self, prim_path: str) -> np.ndarray:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        matrix = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(prim_path))
        return np.asarray(matrix, dtype=float).reshape(4, 4)

    @property
    def collision_prim_paths(self) -> list[str]:
        return [path for record in self.records.values() for path in record.collision_prim_paths]

    def build_world_config(self, reference_prim_path: str):
        return self._helper().get_obstacles_from_collision_prims(
            self.collision_prim_paths,
            reference_prim_path=reference_prim_path,
        ).get_collision_check_world()

    def bind_controller(self, controller: Any) -> None:
        key = (str(controller.name), str(controller.lr_name))
        if key in self.controllers:
            raise CollisionSceneError(f"controller already registered: {key}")
        self.controllers[key] = controller
        self.controller_enabled[key] = {path: True for path in self.collision_prim_paths}
        self._temporary_disabled[key] = set()
        capacity = int(
            controller.motion_gen.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(
                "attached_object"
            )
        )
        for entity_name in self._configured_manipulation_entities():
            record = self.records.get(entity_name)
            if record is None:
                continue
            attach_count = len(self.attach_prim_paths.get(entity_name, []))
            if attach_count > capacity:
                raise CollisionSceneError(
                    "attached sphere capacity is smaller than the target's exact collider count: "
                    f"controller={key} entity={entity_name} "
                    f"attach_prims={attach_count} capacity={capacity}"
                )
        self.audit_controller(controller)

    def initialize_contact_views(self) -> None:
        """Create PhysX views for non-finger robot links against all world colliders."""

        from omni.isaac.core.prims import RigidContactView

        self._robot_environment_contact_views.clear()
        self._finger_environment_contact_views.clear()
        self._object_environment_contact_views.clear()
        self._object_environment_filter_paths.clear()

        for key, controller in self.controllers.items():
            robot = controller.robot
            paths = (
                robot.fl_forbid_collision_paths
                if controller.lr_name == "left"
                else robot.fr_forbid_collision_paths
            )
            views = []
            for path in paths:
                view = RigidContactView(
                    prim_paths_expr=path,
                    filter_paths_expr=self.collision_prim_paths,
                )
                view.initialize()
                views.append(view)
            if not views and self.strict:
                raise CollisionSceneError(f"no forbidden-link contact views configured for {key}")
            self._robot_environment_contact_views[key] = views
            finger_paths = (
                robot.fl_filter_paths_expr
                if controller.lr_name == "left"
                else robot.fr_filter_paths_expr
            )
            finger_views = []
            for path in finger_paths:
                # Isaac Sim 4.1 requires one filter-pattern group per sensor
                # pattern. One view per finger also gives a stable matrix
                # layout for asymmetric grippers.
                view = RigidContactView(
                    prim_paths_expr=path,
                    filter_paths_expr=self.collision_prim_paths,
                )
                view.initialize()
                finger_views.append(view)
            if not finger_views and self.strict:
                raise CollisionSceneError(f"no finger contact views configured for {key}")
            self._finger_environment_contact_views[key] = finger_views

        manipulation_entities = self._configured_manipulation_entities()
        for entity_name in sorted(manipulation_entities):
            if entity_name not in self.records:
                raise CollisionSceneError(
                    f"configured manipulation object has no collision record: {entity_name}"
                )
            record = self.records[entity_name]
            sensor_paths = self._rigid_body_paths(record)
            if not sensor_paths:
                raise CollisionSceneError(
                    f"manipulation object has no RigidBodyAPI sensor root: {entity_name}"
                )
            filters = [
                path
                for other in self.records.values()
                if other.entity_name != entity_name
                for path in other.collision_prim_paths
            ]
            if not filters:
                continue
            views = []
            for sensor_path in sensor_paths:
                view = RigidContactView(
                    prim_paths_expr=sensor_path,
                    filter_paths_expr=filters,
                )
                view.initialize()
                views.append(view)
            self._object_environment_contact_views[entity_name] = views
            self._object_environment_filter_paths[entity_name] = filters

    @staticmethod
    def _filter_force_maxima(views: Iterable[Any], filter_count: int) -> np.ndarray:
        """Reduce one-sensor contact matrices to one maximum per filter."""

        maxima = np.zeros(int(filter_count), dtype=float)
        for view in views:
            values = np.asarray(view.get_contact_force_matrix(), dtype=float)
            if not values.size:
                continue
            magnitudes = np.linalg.norm(values, axis=-1).reshape(-1, int(filter_count))
            maxima = np.maximum(maxima, np.max(magnitudes, axis=0))
        return maxima

    def get_unexpected_robot_contact_force(self, robot: str, arm: str) -> float:
        maximum = 0.0
        for view in self._robot_environment_contact_views.get((str(robot), str(arm)), []):
            try:
                values = np.asarray(view.get_contact_force_matrix(), dtype=float)
                if values.size:
                    maximum = max(maximum, float(np.max(np.linalg.norm(values, axis=-1))))
            except Exception as exc:  # pragma: no cover - Isaac runtime failure path
                if self.strict:
                    raise CollisionSceneError(
                        f"failed to read robot/environment contact view for {robot}/{arm}: {exc}"
                    ) from exc
                LOGGER.exception("[CollisionWorld] contact view read failed for %s/%s", robot, arm)
        return maximum

    def get_object_environment_contact_forces(
        self, entity_name: str, support_entity: str | None = None
    ) -> tuple[float, float]:
        """Return (allowed support, all other) contact-force maxima."""

        views = self._object_environment_contact_views.get(entity_name)
        if not views:
            return 0.0, 0.0
        try:
            filters = self._object_environment_filter_paths[entity_name]
            maxima = self._filter_force_maxima(views, len(filters))
        except Exception as exc:  # pragma: no cover - Isaac runtime failure path
            if self.strict:
                raise CollisionSceneError(
                    f"failed to read object/environment contact view for {entity_name}: {exc}"
                ) from exc
            LOGGER.exception("[CollisionWorld] object contact view failed for %s", entity_name)
            return 0.0, 0.0
        support_paths = (
            set(self.records[support_entity].collision_prim_paths)
            if support_entity in self.records
            else set()
        )
        allowed_indices = [index for index, path in enumerate(filters) if path in support_paths]
        other_indices = [index for index, path in enumerate(filters) if path not in support_paths]
        allowed = (
            float(np.max(maxima[allowed_indices])) if allowed_indices else 0.0
        )
        unexpected = (
            float(np.max(maxima[other_indices])) if other_indices else 0.0
        )
        return allowed, unexpected

    def get_finger_environment_contact_forces(
        self,
        robot: str,
        arm: str,
        allowed_entity: str | None = None,
    ) -> tuple[float, float]:
        """Return (allowed target, all other) finger contact-force maxima."""

        key = self._controller_key(robot, arm)
        views = self._finger_environment_contact_views.get(key)
        if not views:
            return 0.0, 0.0
        maxima = self._filter_force_maxima(views, len(self.collision_prim_paths))
        allowed_paths = (
            set(self.records[allowed_entity].collision_prim_paths)
            if allowed_entity in self.records
            else set()
        )
        allowed_indices = [
            index for index, path in enumerate(self.collision_prim_paths) if path in allowed_paths
        ]
        other_indices = [
            index for index, path in enumerate(self.collision_prim_paths) if path not in allowed_paths
        ]
        allowed = float(np.max(maxima[allowed_indices])) if allowed_indices else 0.0
        unexpected = float(np.max(maxima[other_indices])) if other_indices else 0.0
        return allowed, unexpected

    def get_attached_object_slip(self, entity_name: str) -> tuple[float, float]:
        """Return object-to-EE translation and rotation drift since attach."""

        record = self.records[entity_name]
        if record.state not in {
            CollisionObjectState.ATTACHED,
            CollisionObjectState.PLACEMENT_CONTACT,
        }:
            return 0.0, 0.0
        initial = self._attached_relative_pose.get(entity_name)
        if initial is None:
            raise CollisionSceneError(f"attached pose baseline missing: {entity_name}")
        owner = self._controller_key(record.owner_robot, record.owner_arm)
        ee_world = self._world_matrix(self.controllers[owner].robot_ee_path)
        object_world = self._world_matrix(
            record.tracking_prim_path or record.root_prim_path
        )
        current = object_world @ np.linalg.inv(ee_world)
        translation = float(np.linalg.norm(current[3, :3] - initial[3, :3]))
        relative_rotation = current[:3, :3] @ initial[:3, :3].T
        cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
        return translation, float(np.degrees(np.arccos(cosine)))

    def _controller_key(self, robot: str, arm: str) -> tuple[str, str]:
        key = (str(robot), str(arm))
        if key not in self.controllers:
            raise CollisionSceneError(f"unknown collision-world controller: {key}")
        return key

    def _set_enabled(self, key: tuple[str, str], paths: Iterable[str], enabled: bool) -> None:
        controller = self.controllers[key]
        for path in paths:
            if controller.motion_gen.world_model.get_obstacle(path) is None:
                raise CollisionSceneError(f"CuRobo obstacle missing before enable change: {key} {path}")
            controller.motion_gen.world_collision.enable_obstacle(path, bool(enabled))
            self.controller_enabled[key][path] = bool(enabled)

    def _transition(
        self,
        entity_name: str,
        state: CollisionObjectState,
        robot: str | None = None,
        arm: str | None = None,
        reason: str = "",
    ) -> CollisionObjectRecord:
        record = self.records[entity_name]
        if state != record.state and state not in _ALLOWED_TRANSITIONS[record.state]:
            raise CollisionSceneError(
                f"illegal collision state transition for {entity_name}: {record.state.value} -> {state.value}"
            )
        if state in {
            CollisionObjectState.ACTIVE_TARGET_TRANSIT,
            CollisionObjectState.ACTIVE_TARGET_APPROACH,
            CollisionObjectState.ATTACHED,
            CollisionObjectState.PLACEMENT_CONTACT,
        }:
            if not robot or not arm:
                raise CollisionSceneError(f"state {state.value} requires an owner")
            active_other = [
                other.entity_name
                for other in self.records.values()
                if other.entity_name != entity_name
                and other.owner_robot is not None
                and (other.owner_robot, other.owner_arm) != (robot, arm)
                and other.state
                in {
                    CollisionObjectState.ACTIVE_TARGET_TRANSIT,
                    CollisionObjectState.ACTIVE_TARGET_APPROACH,
                    CollisionObjectState.ATTACHED,
                    CollisionObjectState.PLACEMENT_CONTACT,
                }
            ]
            if active_other:
                raise CollisionSceneError(
                    "UNSUPPORTED_CONCURRENT_MANIPULATION: " + ", ".join(active_other)
                )
            record.owner_robot, record.owner_arm = str(robot), str(arm)
        elif state in {
            CollisionObjectState.WORLD_OBSTACLE,
            CollisionObjectState.PLACED_WORLD,
            CollisionObjectState.DISABLED,
        }:
            record.owner_robot = None
            record.owner_arm = None
        old = record.state
        record.state = state
        self.world_revision += 1
        self.object_state_events.append(
            {
                "step_id": self._step_id,
                "entity": entity_name,
                "from": old.value,
                "to": state.value,
                "owner_robot": record.owner_robot,
                "owner_arm": record.owner_arm,
                "reason": reason,
                "world_revision": self.world_revision,
            }
        )
        return record

    def begin_target_transit(self, entity_name: str, robot: str, arm: str) -> None:
        record = self._transition(
            entity_name, CollisionObjectState.ACTIVE_TARGET_TRANSIT, robot, arm, "pick_transit"
        )
        for key in self.controllers:
            self._set_enabled(key, record.collision_prim_paths, True)

    def begin_target_approach(self, entity_name: str, robot: str, arm: str) -> None:
        record = self._transition(
            entity_name, CollisionObjectState.ACTIVE_TARGET_APPROACH, robot, arm, "terminal_grasp"
        )
        owner = self._controller_key(robot, arm)
        for key in self.controllers:
            self._set_enabled(key, record.collision_prim_paths, key != owner)
        self.assert_invariants()

    def attach_target(self, entity_name: str, robot: str, arm: str) -> None:
        record = self.records[entity_name]
        attach_paths = self.attach_prim_paths[entity_name]
        owner = self._controller_key(robot, arm)
        # Restore first; CuRobo's attach call then atomically disables these
        # world obstacles while adding attached collision spheres.
        self._set_enabled(owner, record.collision_prim_paths, True)
        try:
            attached = self.controllers[owner].attach_objects(attach_paths)
        except Exception:
            # Preserve the ACTIVE_TARGET_APPROACH invariant on any CuRobo
            # attach failure; the execution supervisor can then hold/abort
            # without leaving the target accidentally enabled for its owner.
            for key in self.controllers:
                self._set_enabled(key, record.collision_prim_paths, key != owner)
            raise
        if attached is False:
            for key in self.controllers:
                self._set_enabled(key, record.collision_prim_paths, key != owner)
            raise CollisionSceneError(f"CuRobo attach failed: {entity_name}")
        # attach_objects_to_robot disables the selected consolidated attach
        # proxy. Explicitly disable every other exact world collider of the
        # same entity as part of the identity switch.
        self._set_enabled(owner, record.collision_prim_paths, False)
        ee_world = self._world_matrix(self.controllers[owner].robot_ee_path)
        object_world = self._world_matrix(
            record.tracking_prim_path or record.root_prim_path
        )
        self._attached_relative_pose[entity_name] = object_world @ np.linalg.inv(ee_world)
        self._transition(entity_name, CollisionObjectState.ATTACHED, robot, arm, "attach")
        self.assert_invariants()

    def begin_placement_contact(self, entity_name: str, robot: str, arm: str) -> None:
        self._transition(
            entity_name, CollisionObjectState.PLACEMENT_CONTACT, robot, arm, "terminal_place"
        )

    def begin_placement_descent(
        self, entity_name: str, support_entity: str, robot: str, arm: str
    ) -> None:
        """Allow only the terminal owner to enter the support collision volume.

        PhysX collision remains enabled.  This only changes the owner's CuRobo
        world, while the execution monitor is responsible for rejecting robot-
        support contact and accepting object-support contact.
        """

        self.begin_placement_contact(entity_name, robot, arm)
        support = self.records[support_entity]
        owner = self._controller_key(robot, arm)
        self._set_enabled(owner, support.collision_prim_paths, False)
        self._temporary_disabled[owner].update(support.collision_prim_paths)

    def begin_terminal_retreat(self, entity_name: str, robot: str, arm: str) -> None:
        record = self.records[entity_name]
        owner = self._controller_key(robot, arm)
        if record.state != CollisionObjectState.PLACED_WORLD:
            raise CollisionSceneError(
                f"terminal retreat requires PLACED_WORLD, got {record.state.value}: {entity_name}"
            )
        self._retreating_placed.add(entity_name)
        self._transition(
            entity_name,
            CollisionObjectState.ACTIVE_TARGET_APPROACH,
            robot,
            arm,
            "terminal_retreat",
        )
        self._set_enabled(owner, record.collision_prim_paths, False)
        self._temporary_disabled[owner].update(record.collision_prim_paths)

    def _restore_temporary(self, key: tuple[str, str]) -> None:
        paths = list(self._temporary_disabled.get(key, set()))
        if paths:
            self._set_enabled(key, paths, True)
            self._temporary_disabled[key].clear()

    def detach_target(self, entity_name: str, robot: str, arm: str) -> None:
        record = self.records[entity_name]
        owner = self._controller_key(robot, arm)
        self.controllers[owner].detach_obj()
        self._pending_detach.add(entity_name)
        self._attached_relative_pose.pop(entity_name, None)
        self._restore_temporary(owner)
        # The object is deliberately absent from the owner's planning world
        # during the configured physics settle window.  No robot motion is
        # allowed in that bookkeeping phase.
        self._set_enabled(owner, record.collision_prim_paths, False)

    def finalize_detach_target(self, entity_name: str, robot: str, arm: str) -> None:
        """Read the settled Stage pose, then restore the object to every world."""

        record = self.records[entity_name]
        if entity_name not in self._pending_detach:
            raise CollisionSceneError(f"detach settle was not started: {entity_name}")
        self._sync_record_poses(record, force=True)
        for key in self.controllers:
            self._set_enabled(key, record.collision_prim_paths, True)
        self._transition(entity_name, CollisionObjectState.PLACED_WORLD, reason="detach")
        self._pending_detach.remove(entity_name)
        self.assert_invariants()

    def restore_world(self, entity_name: str) -> None:
        record = self.records[entity_name]
        self._sync_record_poses(record, force=True)
        for key in self.controllers:
            self._restore_temporary(key)
            self._set_enabled(key, record.collision_prim_paths, True)
        was_placed = (
            record.state == CollisionObjectState.PLACED_WORLD
            or entity_name in self._retreating_placed
        )
        target = (
            CollisionObjectState.PLACED_WORLD
            if was_placed
            else CollisionObjectState.WORLD_OBSTACLE
        )
        self._transition(entity_name, target, reason="restore_world")
        self._retreating_placed.discard(entity_name)

    def _sync_record_poses(self, record: CollisionObjectRecord, force: bool = False) -> bool:
        changed = False
        significant = False
        for path in record.collision_prim_paths:
            matrix = self._world_matrix(path)
            previous = self._pose_matrices.get(path)
            if not force and previous is not None and np.allclose(matrix, previous, atol=1e-6, rtol=0.0):
                continue
            if previous is not None:
                translation_delta = float(np.linalg.norm(matrix[3, :3] - previous[3, :3]))
                relative_rotation = matrix[:3, :3] @ previous[:3, :3].T
                cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
                rotation_delta_deg = float(np.degrees(np.arccos(cosine)))
                significant = significant or (
                    translation_delta > self.dynamic_translation_replan_m
                    or rotation_delta_deg > self.dynamic_rotation_replan_deg
                )
            self._pose_matrices[path] = matrix
            changed = True
            for controller in self.controllers.values():
                from curobo.types.math import Pose

                obstacle_pose = self._helper().get_collision_prim_pose(
                    path, reference_prim_path=controller.reference_prim_path
                )
                controller.motion_gen.world_collision.update_obstacle_pose(
                    path, Pose.from_list(obstacle_pose, controller.tensor_args)
                )
        if changed:
            record.pose_revision += 1
            self.world_revision += 1
        return significant if not force else changed

    def sync_dynamic_poses(self, step_id: int, interval_steps: int = 5, force: bool = False) -> list[str]:
        self._step_id = int(step_id)
        if not force and interval_steps > 0 and step_id % interval_steps != 0:
            return []
        changed = []
        for record in self.records.values():
            if record.mobility == "static" or record.state == CollisionObjectState.ATTACHED:
                continue
            if self._sync_record_poses(record, force=force):
                changed.append(record.entity_name)
        return changed

    def audit_controller(self, controller: Any) -> None:
        expected = set(self.collision_prim_paths)
        actual = {str(obstacle.name) for obstacle in controller.motion_gen.world_model.objects}
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        key = f"{controller.name}/{controller.lr_name}"
        self.controller_audits[key] = {
            "missing_in_curobo": missing,
            "unexpected_in_curobo": unexpected,
        }
        if missing or unexpected:
            raise CollisionSceneError(
                f"Physics/CuRobo collider mismatch for {key}: "
                f"missing={missing}, unexpected={unexpected}"
            )

    def assert_invariants(self) -> None:
        for record in self.records.values():
            if record.state in {
                CollisionObjectState.ATTACHED,
                CollisionObjectState.PLACEMENT_CONTACT,
            }:
                owner = self._controller_key(record.owner_robot, record.owner_arm)
                if any(self.controller_enabled[owner].get(path, True) for path in record.collision_prim_paths):
                    raise CollisionSceneError(f"attached object remains world-enabled: {record.entity_name}")
                if (
                    record.entity_name not in self._pending_detach
                    and not self.controllers[owner].has_attached_collision_spheres()
                ):
                    raise CollisionSceneError(
                        f"attached object has no active collision spheres: {record.entity_name}"
                    )
            elif record.state not in {
                CollisionObjectState.ACTIVE_TARGET_APPROACH,
                CollisionObjectState.DISABLED,
            }:
                for key in self.controllers:
                    disabled_paths = {
                        path
                        for path in record.collision_prim_paths
                        if not self.controller_enabled[key].get(path, False)
                    }
                    unexpected_disabled = disabled_paths - self._temporary_disabled.get(
                        key, set()
                    )
                    if unexpected_disabled:
                        raise CollisionSceneError(
                            f"world object is disabled for {key}: {record.entity_name} "
                            f"({record.state.value}) paths={sorted(unexpected_disabled)}"
                        )

    def reset_episode(self) -> None:
        """Restore every entity to a world obstacle after simulator reset."""

        for record in self.records.values():
            if record.state in {
                CollisionObjectState.ATTACHED,
                CollisionObjectState.PLACEMENT_CONTACT,
            } and record.owner_robot:
                key = self._controller_key(record.owner_robot, record.owner_arm)
                self.controllers[key].detach_obj()
            record.state = CollisionObjectState.WORLD_OBSTACLE
            record.owner_robot = None
            record.owner_arm = None
            record.pose_revision = 0
        for key in self.controllers:
            self._restore_temporary(key)
            self._set_enabled(key, self.collision_prim_paths, True)
        self.object_state_events.clear()
        self._pending_detach.clear()
        self._retreating_placed.clear()
        self._attached_relative_pose.clear()
        self.world_revision += 1
        self.sync_dynamic_poses(0, interval_steps=1, force=True)
        self.assert_invariants()

    def export(self, episode_dir: str | Path) -> None:
        directory = Path(episode_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for controller in self.controllers.values():
            self.audit_controller(controller)
        audit = {
            "mode": self.mode,
            "strict": self.strict,
            "world_revision": self.world_revision,
            "collision_prim_count": len(self.collision_prim_paths),
            "records": [record.to_dict() for record in self.records.values()],
            "attach_prim_paths": dict(self.attach_prim_paths),
            "exact_exclusions": [
                {"prim_path": path, "reason": reason} for path, reason in self.exclusions.items()
            ],
            "schema_exclusions": [
                {"prim_path": path, "reason": reason}
                for path, reason in self.schema_exclusions.items()
            ],
            "physics_curobo_difference": self.controller_audits,
        }
        (directory / "collision_world_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (directory / "object_state_events.jsonl").open("w", encoding="utf-8") as stream:
            for event in self.object_state_events:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
