"""Physics-schema collision discovery and manipulation-object state management."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

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


@dataclass(frozen=True)
class PlannerScenePort:
    """Explicit scene-facing controller dependencies.

    The manager stores this port, never a controller façade.  Runtime and
    operation callbacks are supplied by composition at bind time.
    """

    name: str
    lr_name: str
    reference_prim_path: str
    robot_ee_path: str
    tensor_args: Any
    robot: Any
    runtime: Any
    check_current_start_state: Callable[[], tuple[bool, Any]] | None = None
    attach_collision_object: Callable[[Sequence[str]], Any] | None = None
    detach_attachment: Callable[[], Any] | None = None
    has_attached_collision_spheres: Callable[[], bool] | None = None
    collision_world_mode: str = "physics_schema"


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
        self.geometry_mode = str(
            _cfg_get(self.config, "geometry_mode", "bbox")
        ).strip().lower()
        if self.geometry_mode not in {"bbox", "native"}:
            raise ValueError(
                "collision_world.geometry_mode must be 'bbox' or 'native', "
                f"got {self.geometry_mode!r}"
            )
        planning_exclusions = _cfg_get(self.config, "planning_exclusions", [])
        if not isinstance(planning_exclusions, list):
            raise ValueError("planning_exclusions must be a YAML list of exact entity names")
        names: list[str] = []
        seen_names: set[str] = set()
        for value in planning_exclusions:
            if not isinstance(value, str):
                raise ValueError("planning_exclusions entries must be exact entity-name strings")
            name = value.strip()
            if (
                value != name
                or not name
                or name in {".", ".."}
                or name.startswith("/")
                or "/" in name
                or "\\" in name
                or any(char.isspace() for char in name)
                or any(char in name for char in "*?[]{}()")
            ):
                raise ValueError(
                    "planning exclusion must be one exact task entity name, not a "
                    f"path, substring, or glob: {value!r}"
                )
            if name in seen_names:
                raise ValueError(f"duplicate planning exclusion: {name}")
            seen_names.add(name)
            names.append(name)
        self._planning_exclusion_names = tuple(names)
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
        self.scene_ports: dict[tuple[str, str], PlannerScenePort] = {}
        self.controller_enabled: dict[tuple[str, str], dict[str, bool]] = {}
        self._controller_reference_matrices: dict[tuple[str, str], np.ndarray] = {}
        self.controller_audits: dict[str, dict[str, list[str]]] = {}
        self._temporary_disabled: dict[tuple[str, str], set[str]] = {}
        self._diagnostic_forced_disabled: dict[tuple[str, str], set[str]] = {}
        self._diagnostic_physics_disabled_paths: set[str] = set()
        self._pending_detach: set[str] = set()
        self._retreating_placed: set[str] = set()
        self._attached_relative_pose: dict[str, np.ndarray] = {}
        self._slip_ignore_axis: dict[str, np.ndarray | None] = {}
        self.object_state_events: list[dict[str, Any]] = []
        self.world_revision = 0
        self._step_id = 0
        self._pose_matrices: dict[str, np.ndarray] = {}
        self._tracking_pose_matrices: dict[str, np.ndarray] = {}
        # Last native-frame pose sent for each exact collider/controller.  A
        # planner materialized after dynamic synchronization replays this
        # cache before its first query instead of relying on the original
        # world snapshot alone.
        self._native_pose_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._robot_environment_contact_views: dict[tuple[str, str], list[Any]] = {}
        self._robot_contact_debug_reported: set[tuple[str, str]] = set()
        self._finger_environment_contact_views: dict[tuple[str, str], list[Any]] = {}
        self._object_environment_contact_views: dict[str, list[Any]] = {}
        self._object_environment_filter_paths: dict[str, list[str]] = {}
        self._object_contact_debug_reported: set[tuple[str, str | None]] = set()
        self._usd_parser = None
        self._discover()

    def _helper(self):
        if self._usd_parser is None:
            # Lazy import keeps Physics-schema discovery unit-testable outside
            # the Isaac/CuRobo Python environment while using the native v2
            # USD parser directly.
            from core.planning.native_bridge import UsdSceneParser

            self._usd_parser = UsdSceneParser()
            self._usd_parser.load_stage(self.stage)
        return self._usd_parser

    @staticmethod
    def _entity_root(entity: Any) -> str | None:
        for field in ("base_prim_path", "object_prim_path", "prim_path", "rigid_prim_path"):
            value = getattr(entity, field, None)
            if value:
                return str(value)
        return None

    @staticmethod
    def _parse_slip_ignore_axis(value: Any) -> np.ndarray | None:
        """Resolve a rotational-symmetry axis declared by an object profile."""

        if value is None:
            return None
        if isinstance(value, str):
            axis = {
                "x": [1.0, 0.0, 0.0],
                "y": [0.0, 1.0, 0.0],
                "z": [0.0, 0.0, 1.0],
            }.get(value.strip().lower())
            if axis is None:
                raise CollisionSceneError(
                    f"unknown attach_slip_ignore_axis: {value!r}"
                )
            return np.asarray(axis, dtype=float)
        axis = np.asarray(value, dtype=float).reshape(-1)
        if axis.shape != (3,):
            raise CollisionSceneError(
                "attach_slip_ignore_axis must be a local axis name or 3-vector: "
                f"{value!r}"
            )
        norm = float(np.linalg.norm(axis))
        if norm <= 0.0:
            raise CollisionSceneError(
                f"attach_slip_ignore_axis must be non-zero: {value!r}"
            )
        return axis / norm

    @staticmethod
    def _is_supported(prim: Usd.Prim) -> bool:
        return any(prim.IsA(schema) for schema in SUPPORTED_COLLIDER_TYPES)

    @staticmethod
    def _has_nonempty_bound(prim: Usd.Prim) -> bool:
        bound = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
                UsdGeom.Tokens.guide,
            ],
            useExtentsHint=False,
        ).ComputeLocalBound(prim).ComputeAlignedBox()
        if bound.IsEmpty():
            return False
        size = bound.GetSize()
        return all(float(size[index]) > 0.0 for index in range(3))

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

    def _configured_entity(self, entity_name: str) -> Any | None:
        for name, entity in self._iter_entities():
            if name == str(entity_name):
                return entity
        return None

    def _configured_entity_cfg(self, entity_name: str) -> Any:
        """Look up raw task metadata when an Isaac wrapper drops it."""

        task_cfg = getattr(self.task, "cfg", {}) or {}
        candidates = list(task_cfg.get("objects", []) or [])
        arena_cfg = task_cfg.get("arena", {}) or {}
        candidates.extend(list(arena_cfg.get("fixtures", []) or []))
        for candidate in candidates:
            if str(_cfg_get(candidate, "name", "")) == str(entity_name):
                return candidate
        entity = self._configured_entity(entity_name)
        return getattr(entity, "cfg", {}) or {}

    def get_source_support_entity(self, entity_name: str) -> str | None:
        """Return the configured source fixture for a movable object.

        Pick's post-grasp lift starts while many tabletop assets are still in
        contact with their source support.  The support is configuration
        metadata, not a manipulation target, so expose it explicitly to the
        phase-level safety monitor instead of treating every environment
        contact as unexpected.
        """

        cfg = self._configured_entity_cfg(entity_name)
        parent_fixture = str(_cfg_get(cfg, "parent_fixture", "") or "").strip()
        if not parent_fixture:
            task_cfg = getattr(self.task, "cfg", {}) or {}
            for region in task_cfg.get("regions", []) or []:
                if str(_cfg_get(region, "object", "")) != str(entity_name):
                    continue
                parent_fixture = str(
                    _cfg_get(
                        region,
                        "parent_fixture",
                        _cfg_get(
                            region,
                            "support_target_fixture",
                            _cfg_get(region, "target", ""),
                        ),
                    )
                    or ""
                ).strip()
                break
        if parent_fixture and parent_fixture in self.records:
            return parent_fixture
        return None

    def support_collision_paths(self, support_entity: str | None) -> set[str]:
        """Resolve a fixture and its explicit support-plane descendants."""

        if not support_entity:
            return set()
        support_names = {str(support_entity)}
        for name, entity in self._iter_entities():
            cfg = self._configured_entity_cfg(name)
            parent_fixture = str(_cfg_get(cfg, "parent_fixture", "") or "").strip()
            role = str(_cfg_get(cfg, "role", "") or "").lower()
            if parent_fixture == str(support_entity) and (
                role == "support_collision_plane" or "support_plane" in name
            ):
                support_names.add(name)
        paths: set[str] = set()
        for name in support_names:
            record = self.records.get(name)
            if record is not None:
                paths.update(record.collision_prim_paths)
        return paths

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
        # A number of imported object assets place the collision mesh under a
        # sibling branch of the rigid body (rather than below it).  In that
        # layout the collider-to-ancestor walk above is intentionally empty,
        # but the object still has a valid rigid body sensor root.  Resolve
        # those roots from the object subtree before declaring the asset
        # incompatible with contact auditing.
        if not paths:
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath(root_path)):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    paths.add(str(prim.GetPath()))
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
        discovered_path_owners: dict[str, tuple[str, str]] = {}
        discovered_entity_names: set[str] = set()
        for entity_name, entity in self._iter_entities():
            if entity_name in discovered_entity_names:
                raise CollisionSceneError(
                    f"task entity name resolves to multiple records: {entity_name}"
                )
            discovered_entity_names.add(entity_name)
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
                enabled_collision_prims.append(prim)

            collider_prims = []
            for prim in enabled_collision_prims:
                if not self._is_supported(prim):
                    continue
                if not self._has_nonempty_bound(prim):
                    path = str(prim.GetPath())
                    self.schema_exclusions[path] = "empty_enabled_geometry_collider"
                    LOGGER.warning(
                        "[CollisionWorld] audit-only empty collider exclusion path=%s",
                        path,
                    )
                    continue
                collider_prims.append(prim)
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
                    previous_entity, previous_root = discovered_path_owners[path]
                    LOGGER.error(
                        "[CollisionWorldDuplicate] path=%s previous_entity=%s "
                        "previous_root=%s current_entity=%s current_root=%s",
                        path,
                        previous_entity,
                        previous_root,
                        entity_name,
                        root_path,
                    )
                    raise CollisionSceneError(f"collision Prim belongs to multiple entities: {path}")
                discovered_paths.add(path)
                discovered_path_owners[path] = (entity_name, root_path)

            if not collider_prims:
                if self._explicitly_noncollidable(entity):
                    self.schema_exclusions[root_path] = (
                        "config_declared_visual_only_and_stage_has_no_enabled_collider"
                    )
                    continue
                # Some legacy arenas register visual-only fixture roots (for
                # example ``background0``) in the task entity collection even
                # though the referenced USD subtree contains no CollisionAPI
                # at all.  There is no physics contract to audit in that
                # case, so keep it out of the physics-schema world.  A
                # configured/claimed collider still follows the strict path
                # below and cannot be silently ignored.
                has_collision_api = any(
                    prim.HasAPI(UsdPhysics.CollisionAPI)
                    for prim in Usd.PrimRange(root)
                )
                cfg = getattr(entity, "cfg", {}) or {}
                physics_cfg = cfg.get("physics", {}) or {}
                claims_collision = any(
                    value is True
                    for value in (
                        physics_cfg.get("collision_enabled"),
                        cfg.get("collision_enabled"),
                        cfg.get("collision"),
                        (cfg.get("source_physics", {}) or {}).get("collision_enabled"),
                    )
                )
                if not enabled_collision_prims and not has_collision_api and not claims_collision:
                    self.schema_exclusions[root_path] = "visual_only_no_collision_api"
                    LOGGER.info(
                        "[CollisionWorld] skipping visual-only entity without CollisionAPI: %s (%s)",
                        entity_name,
                        root_path,
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
            self._slip_ignore_axis[entity_name] = self._parse_slip_ignore_axis(
                getattr(entity, "attach_slip_ignore_axis", None)
            )
            self._tracking_pose_matrices[entity_name] = self._world_matrix(
                tracking_prim_path
            )
            for path in paths:
                self.path_to_entity[path] = entity_name
                self._pose_matrices[path] = self._world_matrix(path)

        if not self.records:
            raise CollisionSceneError("physics_schema discovered no collision entities")
        for entity_name in self._planning_exclusion_names:
            matches = [
                record
                for record in self.records.values()
                if record.entity_name == entity_name
            ]
            if not matches:
                raise CollisionSceneError(
                    "planning_exclusions entry does not name a collision entity record: "
                    f"{entity_name}"
                )
            if len(matches) != 1:
                raise CollisionSceneError(
                    "planning_exclusions entry resolves to multiple collision records: "
                    f"{entity_name}"
                )
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

    @staticmethod
    def _rotation_from_affine(matrix: np.ndarray) -> np.ndarray:
        """Extract the closest proper rotation from an affine transform.

        USD collision assets can carry non-unit or non-uniform scale in the
        same 3x3 block as their orientation.  Computing an angle directly
        from that affine block turns ordinary scale into large false rotation
        drift.  The polar factor is the nearest orthonormal rotation and is
        stable for the rigid transforms used by the scene manager.
        """

        linear = np.asarray(matrix, dtype=float).reshape(4, 4)[:3, :3]
        if not np.all(np.isfinite(linear)):
            raise CollisionSceneError("non-finite affine transform rotation block")
        left, _, right_transpose = np.linalg.svd(linear)
        rotation = left @ right_transpose
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right_transpose
        return rotation

    @property
    def collision_prim_paths(self) -> list[str]:
        return [path for record in self.records.values() for path in record.collision_prim_paths]

    def build_world_config(self, reference_prim_path: str):
        helper = self._helper()
        exact_loader = getattr(helper, "get_obstacles_from_collision_prims", None)
        # The pre-native-v2 implementation deliberately used one oriented
        # cuboid per exact CollisionAPI path.  Feeding a high-poly USD Mesh
        # (for example, the ball asset has roughly 44k triangles) into every
        # rollout is much slower.  Keep that fast conservative representation
        # as the default and require an explicit ``geometry_mode: native`` for
        # full mesh collision fidelity.
        if self.geometry_mode == "native" and callable(exact_loader):
            world = exact_loader(
                self.collision_prim_paths,
                reference_prim_path=reference_prim_path,
            )
        else:
            stage_loader = getattr(helper, "get_obstacles_from_stage", None)
            if self.geometry_mode == "native" and callable(stage_loader):
                world = stage_loader(
                    only_paths=self.collision_prim_paths,
                    reference_prim_path=reference_prim_path,
                )
                allowed = set(self.collision_prim_paths)
                for field in ("cuboid", "sphere", "mesh", "cylinder", "capsule"):
                    obstacles = getattr(world, field, None)
                    if obstacles is not None:
                        setattr(
                            world,
                            field,
                            [obstacle for obstacle in obstacles if obstacle.name in allowed],
                        )
                parsed_names = {
                    obstacle.name
                    for obstacle in getattr(world, "objects", [])
                    if obstacle.name in allowed
                }
                missing = sorted(allowed - parsed_names)
                if missing:
                    raise CollisionSceneError(
                        "native v2 USD parser did not produce exact collision geometry: "
                        f"{missing}"
                    )
                world.objects = [
                    obstacle
                    for field in ("sphere", "cuboid", "capsule", "mesh", "cylinder", "voxel")
                    for obstacle in (getattr(world, field, None) or [])
                ]
                LOGGER.info(
                    "[CollisionWorld] native USD parser loaded %d exact-path geometries",
                    len(parsed_names),
                )
            else:
                # Fast/default mode, and unit-test doubles or non-Isaac
                # environments without a native loader, all use the same
                # exact-path proxy representation.
                from core.planning.native_bridge import SceneCfg

                proxies = [
                    self._bbox_collision_proxy(path, reference_prim_path)
                    for path in self.collision_prim_paths
                ]
                world = SceneCfg(cuboid=proxies)
                LOGGER.info(
                    "[CollisionWorld] using %d exact-path oriented bbox proxies "
                    "(geometry_mode=%s)",
                    len(proxies),
                    self.geometry_mode,
                )
        return world.get_collision_check_world()

    def _bbox_collision_proxy(self, prim_path: str, reference_prim_path: str):
        """Create a conservative exact-name proxy for a native USD scene."""

        from core.planning.native_bridge import Cuboid

        prim = self.stage.GetPrimAtPath(prim_path)
        reference = self.stage.GetPrimAtPath(reference_prim_path)
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        local_range = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
                UsdGeom.Tokens.guide,
            ],
            useExtentsHint=False,
        ).ComputeUntransformedBound(prim).ComputeAlignedBox()
        if local_range.IsEmpty():
            raise CollisionSceneError(
                f"cannot build collision proxy for empty collider: {prim_path}"
            )
        minimum = local_range.GetMin()
        maximum = local_range.GetMax()
        center = (minimum + maximum) * 0.5
        local_dims = maximum - minimum
        prim_world = cache.GetLocalToWorldTransform(prim)
        reference_world = cache.GetLocalToWorldTransform(reference)
        relative = prim_world * reference_world.GetInverse()
        # ``Gf.Transform.GetScale`` clamps very small scales to 1e-10.  The
        # legacy assets used here legitimately produce relative scales below
        # that threshold after USD unit conversion, so using it collapses a
        # valid collider into a near-zero CuRobo cuboid.  Measure each local
        # axis directly from the affine matrix and normalize a copy only for
        # extracting the orientation.
        axis_rows = [relative.GetRow3(index) for index in range(3)]
        scale = [float(row.GetLength()) for row in axis_rows]
        if not all(np.isfinite(value) and value > 0.0 for value in scale):
            raise CollisionSceneError(
                f"cannot build collision proxy with invalid scale: {prim_path} {scale}"
            )
        rotation_matrix = Gf.Matrix4d(relative)
        for index, row in enumerate(axis_rows):
            rotation_matrix.SetRow3(index, row / scale[index])
        transform = Gf.Transform(rotation_matrix.GetOrthonormalized())
        dims = [abs(float(local_dims[index]) * scale[index]) for index in range(3)]
        if min(dims) <= 0.0:
            raise CollisionSceneError(
                f"cannot build collision proxy with non-positive dimensions: {prim_path} {dims}"
            )
        center_reference = relative.Transform(center)
        quaternion = transform.GetRotation().GetQuat()
        imaginary = quaternion.GetImaginary()
        if (
            os.environ.get("SIMBOX_DEBUG_COLLISION_GEOMETRY") == "1"
            or os.environ.get("CUROBO_DEBUG_WORLD_COLLISION") == "1"
        ):
            LOGGER.warning(
                "[CollisionGeometryDebug] path=%s reference=%s prim_type=%s "
                "local_min=%s local_max=%s relative_scale=%s dims=%s "
                "center_reference=%s",
                prim_path,
                reference_prim_path,
                prim.GetTypeName(),
                [float(value) for value in minimum],
                [float(value) for value in maximum],
                scale,
                dims,
                [float(value) for value in center_reference],
            )
        return Cuboid(
            name=prim_path,
            pose=[
                float(center_reference[0]),
                float(center_reference[1]),
                float(center_reference[2]),
                float(quaternion.GetReal()),
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
            ],
            dims=dims,
        )

    def _port_obstacle_pose(self, port: PlannerScenePort, prim_path: str):
        """Return a native CuRobo Pose in the planner reference frame."""

        from curobo.types import Pose

        helper = self._helper()
        reference_from_world = np.asarray(
            helper.get_pose(port.reference_prim_path, inverse=True), dtype=float
        )
        world_from_prim = np.asarray(helper.get_pose(prim_path), dtype=float)
        reference_from_prim = reference_from_world @ world_from_prim
        return Pose.from_matrix(
            port.tensor_args.to_device(reference_from_prim)
        )

    def bind_scene_port(self, port: PlannerScenePort) -> None:
        if not isinstance(port, PlannerScenePort):
            raise TypeError("CollisionSceneManager requires a PlannerScenePort")
        if port.collision_world_mode == "physics_schema":
            missing_callbacks = [
                name
                for name, callback in (
                    ("attach_collision_object", port.attach_collision_object),
                    ("detach_attachment", port.detach_attachment),
                    (
                        "has_attached_collision_spheres",
                        port.has_attached_collision_spheres,
                    ),
                )
                if callback is None
            ]
            if missing_callbacks:
                raise CollisionSceneError(
                    "physics_schema PlannerScenePort is missing required callbacks: "
                    f"{missing_callbacks}"
                )
        key = (str(port.name), str(port.lr_name))
        if key in self.scene_ports:
            raise CollisionSceneError(f"controller already registered: {key}")
        self.scene_ports[key] = port
        try:
            self.world_revision = max(self.world_revision, int(port.runtime.scene_revision))
            self.controller_enabled[key] = {path: True for path in self.collision_prim_paths}
            self._native_pose_cache[key] = {}
            self._controller_reference_matrices[key] = self._world_matrix(
                port.reference_prim_path
            )
            self._temporary_disabled[key] = set()
            self._diagnostic_forced_disabled[key] = set()
            planner = port.runtime.native_planner
            if planner is None:
                raise CollisionSceneError(f"controller has no native CuRobo planner: {key}")
            capacity = int(
                planner.kinematics.config.kinematics_config.get_number_of_spheres(
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
            self.audit_controller(port)
        except Exception:
            self.scene_ports.pop(key, None)
            self.controller_enabled.pop(key, None)
            self._native_pose_cache.pop(key, None)
            self._controller_reference_matrices.pop(key, None)
            self._temporary_disabled.pop(key, None)
            self._diagnostic_forced_disabled.pop(key, None)
            raise

    def _physics_controller_keys(self) -> list[tuple[str, str]]:
        return [
            key
            for key, controller in self.scene_ports.items()
            if controller.collision_world_mode == "physics_schema"
        ]

    def get_attached_entity(self, robot: str, arm: str) -> str | None:
        """Return the object currently owned by one controller, if any."""

        owner = (str(robot), str(arm))
        attached = sorted(
            record.entity_name
            for record in self.records.values()
            if record.state == CollisionObjectState.ATTACHED
            and (record.owner_robot, record.owner_arm) == owner
        )
        if len(attached) > 1:
            raise CollisionSceneError(
                f"controller owns multiple attached objects: owner={owner} objects={attached}"
            )
        return attached[0] if attached else None

    def assert_attached_owner(
        self, entity_name: str, robot: str, arm: str
    ) -> CollisionObjectRecord:
        """Validate that a carry phase preserves the existing attachment."""

        record = self.records.get(str(entity_name))
        owner = (str(robot), str(arm))
        if record is None:
            raise CollisionSceneError(f"unknown attached object: {entity_name}")
        if record.state != CollisionObjectState.ATTACHED:
            raise CollisionSceneError(
                f"carry phase requires ATTACHED object, got {record.state.value}: {entity_name}"
            )
        if (record.owner_robot, record.owner_arm) != owner:
            raise CollisionSceneError(
                "carry phase attachment owner mismatch: "
                f"object={entity_name} expected={owner} "
                f"actual={(record.owner_robot, record.owner_arm)}"
            )
        self.assert_invariants()
        return record

    def refresh_controller_reference_world(self, port: PlannerScenePort, force: bool = False) -> bool:
        """Refresh every obstacle pose after the mobile planning reference moves."""

        key = self._controller_key(port.name, port.lr_name)
        current = self._world_matrix(port.reference_prim_path)
        previous = self._controller_reference_matrices.get(key)
        if not force and previous is not None and np.allclose(
            current, previous, atol=1e-6, rtol=0.0
        ):
            return False
        poses = {}
        for path in self.collision_prim_paths:
            obstacle_pose = self._port_obstacle_pose(port, path)
            poses[path] = obstacle_pose
            self._native_pose_cache.setdefault(key, {})[path] = obstacle_pose
        port.runtime.update_obstacle_poses(poses, revision=self.world_revision + 1)
        self._controller_reference_matrices[key] = current
        self._adopt_world_revision()
        LOGGER.info(
            "[CollisionWorld] refreshed moving reference controller=%s/%s obstacles=%d world_revision=%d",
            port.name,
            port.lr_name,
            len(self.collision_prim_paths),
            self.world_revision,
        )
        return True

    def diagnose_controller_world_collision(self, port: PlannerScenePort) -> dict[str, Any]:
        """Identify which entity groups invalidate the controller's live start state."""

        key = self._controller_key(port.name, port.lr_name)
        check_start = port.check_current_start_state
        if check_start is None:
            raise CollisionSceneError(
                "PlannerScenePort is missing check_current_start_state callback: "
                f"{key}"
            )

        original = dict(self.controller_enabled[key])
        enabled_paths = [
            path for path in self.collision_prim_paths if original.get(path, False)
        ]
        grouped_paths: dict[str, list[str]] = {}
        for path in enabled_paths:
            grouped_paths.setdefault(self.path_to_entity.get(path, path), []).append(path)

        colliding_entities = []
        try:
            self._set_enabled(key, enabled_paths, False)
            baseline_valid, baseline_status = check_start()
            for entity_name, paths in grouped_paths.items():
                self._set_enabled(key, paths, True)
                valid, status = check_start()
                if not valid:
                    colliding_entities.append(
                        {
                            "entity": entity_name,
                            "paths": list(paths),
                            "status": status,
                        }
                    )
                self._set_enabled(key, paths, False)
        finally:
            for path, enabled in original.items():
                self._set_enabled(key, [path], enabled)

        return {
            "available": True,
            "baseline_without_world_valid": baseline_valid,
            "baseline_without_world_status": baseline_status,
            "tested_entity_count": len(grouped_paths),
            "colliding_entities": colliding_entities,
        }

    def initialize_contact_views(self, physics_sim_view=None) -> None:
        """Create PhysX views for non-finger robot links against all world colliders."""

        from isaacsim.core.api.sensors import RigidContactView
        from isaacsim.core.api.simulation_context import SimulationContext

        # Isaac Sim 6 does not reliably attach a newly-created tensor view to
        # the current USD stage.  All contact sensors must share the view
        # finalized by World.reset()/Scene._finalize().
        if physics_sim_view is None:
            simulation_context = SimulationContext.instance()
            if simulation_context is not None:
                # This is the deprecated tensor view consumed by
                # isaacsim.core.api.sensors.RigidContactView.  Do not use
                # SimulationManager.get_physics_simulation_view(), which is
                # Isaac Sim 6's separate Warp view.
                physics_sim_view = simulation_context.physics_sim_view
        if physics_sim_view is None:
            raise CollisionSceneError("PhysX simulation view is not initialized")

        self._robot_environment_contact_views.clear()
        self._finger_environment_contact_views.clear()
        self._object_environment_contact_views.clear()
        self._object_environment_filter_paths.clear()

        for key, controller in self.scene_ports.items():
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
                view.initialize(physics_sim_view=physics_sim_view)
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
                view.initialize(physics_sim_view=physics_sim_view)
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
                view.initialize(physics_sim_view=physics_sim_view)
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

    def get_unexpected_robot_contact_force(
        self,
        robot: str,
        arm: str,
        allowed_entity: str | None = None,
    ) -> float:
        """Return robot/environment contact excluding one expected entity."""

        key = (str(robot), str(arm))
        filters = self.collision_prim_paths
        allowed_paths = (
            set(self.records[allowed_entity].collision_prim_paths)
            if allowed_entity in self.records
            else set()
        )
        allowed_indices = {
            index for index, path in enumerate(filters) if path in allowed_paths
        }
        other_indices = [
            index for index in range(len(filters)) if index not in allowed_indices
        ]
        maximum = 0.0
        top_contacts: list[tuple[float, str]] = []
        for view in self._robot_environment_contact_views.get(key, []):
            try:
                values = np.asarray(view.get_contact_force_matrix(), dtype=float)
                if values.size:
                    magnitudes = np.linalg.norm(values, axis=-1).reshape(-1, len(filters))
                    all_maxima = np.max(magnitudes, axis=0)
                    maxima = all_maxima[other_indices] if other_indices else np.zeros(0)
                    maximum = max(maximum, float(np.max(maxima)) if maxima.size else 0.0)
                    top_contacts.extend(
                        (float(force), str(path))
                        for index, (force, path) in enumerate(zip(all_maxima, filters))
                        if index in other_indices
                        if float(force) > 0.0
                    )
            except Exception as exc:  # pragma: no cover - Isaac runtime failure path
                if self.strict:
                    raise CollisionSceneError(
                        f"failed to read robot/environment contact view for {robot}/{arm}: {exc}"
                    ) from exc
                LOGGER.exception("[CollisionWorld] contact view read failed for %s/%s", robot, arm)
        if maximum > 0.0 and key not in self._robot_contact_debug_reported:
            self._robot_contact_debug_reported.add(key)
            LOGGER.warning(
                "[CollisionWorld] robot contact detail robot=%s arm=%s allowed=%s "
                "maximum=%s top=%s",
                robot,
                arm,
                allowed_entity,
                maximum,
                sorted(top_contacts, reverse=True)[:8],
            )
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
        support_paths = self.support_collision_paths(support_entity)
        allowed_indices = [index for index, path in enumerate(filters) if path in support_paths]
        other_indices = [index for index, path in enumerate(filters) if path not in support_paths]
        allowed = (
            float(np.max(maxima[allowed_indices])) if allowed_indices else 0.0
        )
        unexpected = (
            float(np.max(maxima[other_indices])) if other_indices else 0.0
        )
        debug_key = (str(entity_name), str(support_entity) if support_entity else None)
        if unexpected > 0.0 and debug_key not in self._object_contact_debug_reported:
            ranked = sorted(
                (
                    (float(force), str(path))
                    for index, (force, path) in enumerate(zip(maxima, filters))
                    if index in other_indices and float(force) > 0.0
                ),
                reverse=True,
            )[:8]
            self._object_contact_debug_reported.add(debug_key)
            LOGGER.warning(
                "[CollisionWorld] object contact detail entity=%s support=%s "
                "allowed=%s unexpected=%s top=%s",
                entity_name,
                support_entity,
                allowed,
                unexpected,
                ranked,
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
        ee_world = self._world_matrix(self.scene_ports[owner].robot_ee_path)
        object_world = self._world_matrix(
            record.tracking_prim_path or record.root_prim_path
        )
        current = object_world @ np.linalg.inv(ee_world)
        translation = float(np.linalg.norm(current[3, :3] - initial[3, :3]))
        current_rotation = self._rotation_from_affine(current)
        initial_rotation = self._rotation_from_affine(initial)
        ignore_axis = self._slip_ignore_axis.get(entity_name)
        if ignore_axis is not None:
            baseline_axis = initial_rotation @ ignore_axis
            current_axis = current_rotation @ ignore_axis
            cosine = float(
                np.clip(
                    float(baseline_axis @ current_axis)
                    / max(
                        float(np.linalg.norm(baseline_axis))
                        * float(np.linalg.norm(current_axis)),
                        1e-12,
                    ),
                    -1.0,
                    1.0,
                )
            )
        else:
            relative_rotation = current_rotation @ initial_rotation.T
            cosine = float(
                np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
            )
        return translation, float(np.degrees(np.arccos(cosine)))

    def _controller_key(self, robot: str, arm: str) -> tuple[str, str]:
        key = (str(robot), str(arm))
        if key not in self.scene_ports:
            raise CollisionSceneError(f"unknown collision-world controller: {key}")
        return key

    def _adopt_world_revision(self) -> int:
        """Publish the manager revision through every bound scene port."""

        self.world_revision = max(
            [self.world_revision + 1]
            + [int(port.runtime.scene_revision) for port in self.scene_ports.values()]
        )
        for port in self.scene_ports.values():
            port.runtime.adopt_scene_revision(self.world_revision)
        return self.world_revision

    def unbind_scene_port(self, port_or_key: PlannerScenePort | tuple[str, str]) -> None:
        """Remove a controller scene port."""

        key = (
            (str(port_or_key.name), str(port_or_key.lr_name))
            if isinstance(port_or_key, PlannerScenePort)
            else (str(port_or_key[0]), str(port_or_key[1]))
        )
        port = self.scene_ports.get(key)
        self.scene_ports.pop(key, None)
        self.controller_enabled.pop(key, None)
        self._native_pose_cache.pop(key, None)
        self._controller_reference_matrices.pop(key, None)
        self._temporary_disabled.pop(key, None)
        self._diagnostic_forced_disabled.pop(key, None)

    unregister_scene_port = unbind_scene_port

    def has_native_obstacle(self, port: PlannerScenePort, path: str) -> bool:
        """Check one exact collider through the unique typed scene owner."""

        return port.runtime.has_obstacle(str(path))

    def _set_enabled(self, key: tuple[str, str], paths: Iterable[str], enabled: bool) -> None:
        port = self.scene_ports[key]
        requested_paths = tuple(str(path) for path in paths)
        missing = tuple(path for path in requested_paths if not port.runtime.has_obstacle(path))
        if missing:
            raise CollisionSceneError(
                "CuRobo obstacle missing before enable change: "
                f"{key} missing={list(missing)}"
            )
        changed = False
        for path in requested_paths:
            effective = bool(enabled) and path not in self._diagnostic_forced_disabled.get(key, set())
            port.runtime.set_obstacle_enabled(path, effective)
            changed = changed or self.controller_enabled[key].get(path, True) != effective
            self.controller_enabled[key][path] = effective
        if changed:
            self._adopt_world_revision()

    def _validate_diagnostic_paths(
        self, port: PlannerScenePort, prim_paths: Iterable[str]
    ) -> tuple[tuple[str, str], list[str]]:
        if isinstance(prim_paths, (str, bytes)):
            raise CollisionSceneError(
                "diagnostic obstacle paths must be a collection of exact paths"
            )
        paths = [str(value).strip() for value in prim_paths]
        if any(not path for path in paths) or len(paths) != len(set(paths)):
            raise CollisionSceneError(
                "diagnostic obstacle paths must be non-empty and unique"
            )
        for path in paths:
            sdf_path = Sdf.Path(path)
            if not sdf_path.IsAbsolutePath() or not sdf_path.IsPrimPath():
                raise CollisionSceneError(
                    f"diagnostic obstacle path must be an absolute Prim path: {path!r}"
                )
        key = self._controller_key(port.name, port.lr_name)
        if self.scene_ports.get(key) is not port:
            raise CollisionSceneError(
                f"diagnostic obstacle change requires the bound scene port: {key}"
            )
        missing = sorted(set(paths) - set(self.controller_enabled.get(key, {})))
        if missing:
            raise CollisionSceneError(
                f"paths are not exact planner-world obstacles: {missing}"
            )
        return key, paths

    def resolve_diagnostic_collision_entities(
        self, entity_names: Iterable[str]
    ) -> dict[str, list[str]]:
        if isinstance(entity_names, (str, bytes)):
            raise CollisionSceneError(
                "diagnostic collision entities must be a collection of names"
            )
        names = [str(value).strip() for value in entity_names]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise CollisionSceneError(
                "diagnostic collision entity names must be non-empty and unique"
            )
        missing = sorted(set(names) - set(self.records))
        if missing:
            raise CollisionSceneError(
                "diagnostic collision entities are not registered world entities: "
                f"{missing}"
            )
        return {
            name: list(self.records[name].collision_prim_paths) for name in names
        }

    @contextmanager
    def diagnostic_curobo_obstacles_disabled(
        self, port: PlannerScenePort, prim_paths: Iterable[str]
    ):
        """Disable exact planner obstacles for one probe and always roll back."""

        key, paths = self._validate_diagnostic_paths(port, prim_paths)
        previous = {
            path: bool(self.controller_enabled[key].get(path, True)) for path in paths
        }
        self._diagnostic_forced_disabled[key].update(paths)
        try:
            self._set_enabled(key, paths, False)
            yield tuple(paths)
        finally:
            self._diagnostic_forced_disabled[key].difference_update(paths)
            for path in reversed(paths):
                self._set_enabled(key, [path], previous[path])

    @contextmanager
    def diagnostic_physics_and_curobo_obstacles_disabled(
        self, port: PlannerScenePort, prim_paths: Iterable[str]
    ):
        """Disable exact PhysX and planner obstacles with authored-state rollback."""

        key, paths = self._validate_diagnostic_paths(port, prim_paths)
        overlapping = sorted(
            set(paths).intersection(self._diagnostic_physics_disabled_paths)
        )
        if overlapping:
            raise CollisionSceneError(
                "diagnostic Physics collision overlay is already active: "
                f"{overlapping}"
            )
        self.sync_dynamic_poses(self._step_id, interval_steps=1, force=True)
        physics_attrs: dict[str, tuple[Any, Any, bool]] = {}
        for path in paths:
            prim = self.stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise CollisionSceneError(
                    f"diagnostic path is not a live Physics collider: {path}"
                )
            attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
            physics_attrs[path] = (attr, attr.Get(), bool(attr.HasAuthoredValueOpinion()))
        previous = {
            path: bool(self.controller_enabled[key].get(path, True)) for path in paths
        }
        self._diagnostic_physics_disabled_paths.update(paths)
        self._diagnostic_forced_disabled[key].update(paths)
        try:
            for path in paths:
                physics_attrs[path][0].Set(False)
            self._set_enabled(key, paths, False)
            yield tuple(paths)
        finally:
            restore_error: BaseException | None = None
            self._diagnostic_forced_disabled[key].difference_update(paths)
            for path in reversed(paths):
                try:
                    self._set_enabled(key, [path], previous[path])
                except BaseException as exc:
                    restore_error = restore_error or exc
            for path in reversed(paths):
                attr, value, authored = physics_attrs[path]
                try:
                    attr.Set(value) if authored else attr.Clear()
                except BaseException as exc:
                    restore_error = restore_error or exc
            self._diagnostic_physics_disabled_paths.difference_update(paths)
            if restore_error is not None:
                raise restore_error

    def apply_controller_planning_exclusions(self, port: PlannerScenePort) -> None:
        """Disable every native obstacle belonging to each exact task entity."""

        if not self._planning_exclusion_names:
            return
        key = self._controller_key(port.name, port.lr_name)
        ignored: list[str] = []
        for entity_name in self._planning_exclusion_names:
            record = self.records.get(entity_name)
            if record is None:
                raise CollisionSceneError(
                    "planning_exclusions entry does not name a collision entity record: "
                    f"{entity_name}"
                )
            paths = list(record.collision_prim_paths)
            self._set_enabled(key, paths, False)
            self._temporary_disabled[key].update(paths)
            ignored.extend(paths)
        if ignored:
            LOGGER.info(
                "[CollisionWorld] applied native-v2 planning exclusions "
                "controller=%s/%s names=%s paths=%s",
                port.name,
                port.lr_name,
                list(self._planning_exclusion_names),
                ignored,
            )

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
        self._adopt_world_revision()
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
        for key in self._physics_controller_keys():
            self._set_enabled(key, record.collision_prim_paths, True)

    def begin_target_approach(self, entity_name: str, robot: str, arm: str) -> None:
        record = self._transition(
            entity_name, CollisionObjectState.ACTIVE_TARGET_APPROACH, robot, arm, "terminal_grasp"
        )
        owner = self._controller_key(robot, arm)
        for key in self._physics_controller_keys():
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
            attached = self.scene_ports[owner].attach_collision_object(attach_paths)
        except Exception:
            # Preserve the ACTIVE_TARGET_APPROACH invariant on any CuRobo
            # attach failure; the execution supervisor can then hold/abort
            # without leaving the target accidentally enabled for its owner.
            for key in self._physics_controller_keys():
                self._set_enabled(key, record.collision_prim_paths, key != owner)
            raise
        if attached is False:
            for key in self._physics_controller_keys():
                self._set_enabled(key, record.collision_prim_paths, key != owner)
            raise CollisionSceneError(f"CuRobo attach failed: {entity_name}")
        # The native attachment manager disables the selected consolidated
        # proxy. Explicitly disable every other exact world collider of the
        # same entity as part of the identity switch.
        self._set_enabled(owner, record.collision_prim_paths, False)
        ee_world = self._world_matrix(self.scene_ports[owner].robot_ee_path)
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
        # Only remember paths changed by this placement transaction.  A
        # support can also be disabled by a persistent planning exclusion;
        # that state must not be re-enabled by candidate cleanup.
        paths = tuple(
            path
            for path in self.support_collision_paths(support_entity)
            if self.controller_enabled[owner].get(path, True)
        )
        if paths:
            self._set_enabled(owner, paths, False)
            self._temporary_disabled[owner].update(paths)

    def restore_placement_support(
        self,
        entity_name: str,
        support_entity: str,
        robot: str,
        arm: str,
    ) -> None:
        """Restore support colliders after a candidate place query.

        Candidate validation may temporarily disable the support in the
        controller's planning world while the carried object enters its
        placement volume.  This cleanup restores only paths changed by this
        transaction and leaves the carried object in ``ATTACHED`` state;
        execution will enter ``PLACEMENT_CONTACT`` again when the terminal
        place phase actually begins.
        """

        entity = self.records.get(str(entity_name))
        support = self.records.get(str(support_entity))
        if entity is None:
            raise CollisionSceneError(
                f"unknown carried object during placement cleanup: {entity_name}"
            )
        if support is None:
            raise CollisionSceneError(
                f"unknown placement support during planning cleanup: {support_entity}"
            )
        owner = self._controller_key(robot, arm)
        paths = tuple(
            path
            for path in self.support_collision_paths(support_entity)
            if path in self._temporary_disabled.get(owner, set())
            and not self.controller_enabled[owner].get(path, True)
        )
        if paths:
            self._set_enabled(owner, paths, True)
            self._temporary_disabled[owner].difference_update(paths)
        if entity.state == CollisionObjectState.PLACEMENT_CONTACT:
            self._transition(
                entity_name,
                CollisionObjectState.ATTACHED,
                robot,
                arm,
                "placement_query_cleanup",
            )

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
        self.scene_ports[owner].detach_attachment()
        self._pending_detach.add(entity_name)
        self._attached_relative_pose.pop(entity_name, None)
        self._restore_temporary(owner)
        # The object is deliberately absent from the owner's planning world
        # during the configured physics settle window.  No robot motion is
        # allowed in that bookkeeping phase.
        self._set_enabled(owner, record.collision_prim_paths, False)

    def is_pending_detach(self, entity_name: str | None) -> bool:
        """Return whether an object is inside its post-detach settle window."""

        return entity_name is not None and str(entity_name) in self._pending_detach

    def finalize_detach_target(self, entity_name: str, robot: str, arm: str) -> None:
        """Read the settled Stage pose, then restore the object to every world."""

        record = self.records[entity_name]
        if entity_name not in self._pending_detach:
            raise CollisionSceneError(f"detach settle was not started: {entity_name}")
        self._sync_record_poses(record, force=True)
        for key in self._physics_controller_keys():
            self._set_enabled(key, record.collision_prim_paths, True)
        self._transition(entity_name, CollisionObjectState.PLACED_WORLD, reason="detach")
        self._pending_detach.remove(entity_name)
        self.assert_invariants()

    def restore_world(self, entity_name: str) -> None:
        record = self.records[entity_name]
        self._sync_record_poses(record, force=True)
        for key in self._physics_controller_keys():
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
        tracking_path = record.tracking_prim_path or record.root_prim_path
        tracking_matrix = self._world_matrix(tracking_path)
        previous_tracking = self._tracking_pose_matrices.get(record.entity_name)
        tracking_translation_delta = 0.0
        tracking_rotation_delta_deg = 0.0
        if previous_tracking is not None:
            tracking_translation_delta = float(
                np.linalg.norm(tracking_matrix[3, :3] - previous_tracking[3, :3])
            )
            relative_rotation = self._rotation_from_affine(tracking_matrix) @ self._rotation_from_affine(previous_tracking).T
            cosine = float(
                np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
            )
            tracking_rotation_delta_deg = float(np.degrees(np.arccos(cosine)))
        self._tracking_pose_matrices[record.entity_name] = tracking_matrix

        for path in record.collision_prim_paths:
            if path in self._diagnostic_physics_disabled_paths:
                continue
            matrix = self._world_matrix(path)
            previous = self._pose_matrices.get(path)
            if not force and previous is not None and np.allclose(matrix, previous, atol=1e-6, rtol=0.0):
                continue
            self._pose_matrices[path] = matrix
            changed = True
            for key in self._physics_controller_keys():
                controller = self.scene_ports[key]
                obstacle_pose = self._port_obstacle_pose(controller, path)
                controller.runtime.update_obstacle_pose(
                    path, obstacle_pose, revision=self.world_revision + 1
                )
                self._native_pose_cache.setdefault(key, {})[path] = obstacle_pose
        if changed:
            record.pose_revision += 1
            self._adopt_world_revision()
        significant = (
            previous_tracking is not None
            and not force
            and (
                tracking_translation_delta > self.dynamic_translation_replan_m
                or tracking_rotation_delta_deg > self.dynamic_rotation_replan_deg
            )
        )
        if significant:
            velocity = None
            entity = self._configured_entity(record.entity_name)
            get_linear_velocity = getattr(entity, "get_linear_velocity", None)
            if callable(get_linear_velocity):
                try:
                    velocity = np.asarray(get_linear_velocity(), dtype=float).reshape(-1).tolist()
                except Exception:  # pragma: no cover - simulator-only diagnostic
                    velocity = None
            LOGGER.warning(
                "[CollisionWorld] significant dynamic pose entity=%s tracking_path=%s "
                "translation_delta_m=%.6f rotation_delta_deg=%.3f "
                "world_position=%s linear_velocity=%s updated_colliders=%d",
                record.entity_name,
                tracking_path,
                tracking_translation_delta,
                tracking_rotation_delta_deg,
                np.asarray(tracking_matrix[3, :3], dtype=float).round(6).tolist(),
                velocity,
                int(changed),
            )
        return significant if not force else changed

    def sync_dynamic_poses(self, step_id: int, interval_steps: int = 5, force: bool = False) -> list[str]:
        self._step_id = int(step_id)
        if not force and interval_steps > 0 and step_id % interval_steps != 0:
            return []
        changed = []
        for record in self.records.values():
            # ATTACHED and PLACEMENT_CONTACT objects are represented by the
            # controller's attached collision spheres, not as world obstacles.
            # In particular, the carried object naturally moves during the
            # terminal placement descent; syncing its disabled world collider
            # would falsely request a dynamic-obstacle replan every few steps.
            if record.mobility == "static" or record.state in {
                CollisionObjectState.ATTACHED,
                CollisionObjectState.PLACEMENT_CONTACT,
            }:
                continue
            if self._sync_record_poses(record, force=force):
                changed.append(record.entity_name)
        return changed

    def audit_controller(self, port: PlannerScenePort) -> None:
        expected = set(self.collision_prim_paths)
        try:
            actual = set(port.runtime.obstacle_names())
        except Exception as exc:
            raise CollisionSceneError(
                f"Physics/CuRobo collider audit cannot enumerate native world for "
                f"{port.name}/{port.lr_name}: {exc}"
            ) from exc
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        key = f"{port.name}/{port.lr_name}"
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
                    and not self.scene_ports[owner].has_attached_collision_spheres()
                ):
                    raise CollisionSceneError(
                        f"attached object has no active collision spheres: {record.entity_name}"
                    )
            elif record.state not in {
                CollisionObjectState.ACTIVE_TARGET_APPROACH,
                CollisionObjectState.DISABLED,
            }:
                for key in self._physics_controller_keys():
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

        self._diagnostic_physics_disabled_paths.clear()
        for paths in self._diagnostic_forced_disabled.values():
            paths.clear()
        for record in self.records.values():
            if record.state in {
                CollisionObjectState.ATTACHED,
                CollisionObjectState.PLACEMENT_CONTACT,
            } and record.owner_robot:
                key = self._controller_key(record.owner_robot, record.owner_arm)
                self.scene_ports[key].detach_attachment()
            record.state = CollisionObjectState.WORLD_OBSTACLE
            record.owner_robot = None
            record.owner_arm = None
            record.pose_revision = 0
        for key in self._physics_controller_keys():
            self._restore_temporary(key)
            self._set_enabled(key, self.collision_prim_paths, True)
        self.object_state_events.clear()
        self._pending_detach.clear()
        self._retreating_placed.clear()
        self._attached_relative_pose.clear()
        self._adopt_world_revision()
        self.sync_dynamic_poses(0, interval_steps=1, force=True)
        self.assert_invariants()

    def sync_after_task_state_restore(self, *, label: str = "task_state_restore") -> list[str]:
        """Publish exact post-reset USD poses to every bound planning scene.

        Fixed rigid bodies are restored after the normal physics warmup.  That
        restore happens after ``reset_episode`` has already synchronized the
        initial world, so force one final collider-pose pass here.  Include
        static records as well as dynamic records: the task restore contract
        is world-pose based and must not leave a stale static collider in the
        native scene.  The existing attachment/world-state transitions remain
        untouched.
        """

        self._adopt_world_revision()
        changed: list[str] = []
        for record in self.records.values():
            if record.state in {
                CollisionObjectState.ATTACHED,
                CollisionObjectState.PLACEMENT_CONTACT,
            }:
                continue
            if self._sync_record_poses(record, force=True):
                changed.append(record.entity_name)
        self.assert_invariants()
        LOGGER.info(
            "[CollisionWorld] post-reset task state synchronized label=%s "
            "changed_entities=%s world_revision=%d",
            label,
            changed,
            self.world_revision,
        )
        return changed

    def refresh_after_task_reset(self) -> None:
        """Re-discover colliders after a task reload and rebuild bound worlds.

        A randomized retry can delete and recreate a rigid-object USD.  The
        replacement may expose a different exact collider path, while the
        manager records and CuRobo world still contain the previous path.  A
        normal ``reset_episode`` only resets state on those old records, so
        this refresh must run before controller ``reset()`` audits the world.
        """

        # Clear any attachment left by the failed episode before replacing the
        # world.  This is intentionally done against the currently bound
        # controllers, before their old records are discarded below.
        for key, controller in self.scene_ports.items():
            if controller.collision_world_mode != "physics_schema":
                continue
            has_attached = controller.has_attached_collision_spheres
            detach = controller.detach_attachment
            if has_attached is None or detach is None:
                raise CollisionSceneError(
                    "physics_schema PlannerScenePort is missing attachment reset callbacks: "
                    f"{key}"
                )
            try:
                if has_attached():
                    detach()
            except Exception as exc:  # pragma: no cover - Isaac/CuRobo reset path
                raise CollisionSceneError(
                    f"failed to clear attached CuRobo state before task reset: {key}: {exc}"
                ) from exc

        # Every structure below is keyed by exact Stage prim paths.  Do not
        # carry any of it over when the task has replaced a USD subtree.
        self.schema_exclusions.clear()
        self.records.clear()
        self.attach_prim_paths.clear()
        self.path_to_entity.clear()
        self._pose_matrices.clear()
        self._tracking_pose_matrices.clear()
        self.controller_enabled.clear()
        self._controller_reference_matrices.clear()
        self.controller_audits.clear()
        self._temporary_disabled.clear()
        self._diagnostic_forced_disabled.clear()
        self._diagnostic_physics_disabled_paths.clear()
        self._native_pose_cache.clear()
        self._pending_detach.clear()
        self._retreating_placed.clear()
        self._attached_relative_pose.clear()
        self._slip_ignore_axis.clear()
        self.object_state_events.clear()
        self._robot_environment_contact_views.clear()
        self._finger_environment_contact_views.clear()
        self._object_environment_contact_views.clear()
        self._object_environment_filter_paths.clear()

        self._discover()

        # Rebuild each bound Physics-schema CuRobo world before the workflow
        # calls TemplateController.reset().  That reset calls update() and
        # audit_controller(), so doing this afterwards would audit a stale
        # world and reproduce ATTACH_COLLISION_PRIM_NOT_IN_CUROBO_WORLD.
        for key, controller in self.scene_ports.items():
            if controller.collision_world_mode != "physics_schema":
                continue
            self.controller_enabled[key] = {
                path: True for path in self.collision_prim_paths
            }
            self._temporary_disabled[key] = set()
            self._diagnostic_forced_disabled[key] = set()
            self._controller_reference_matrices[key] = self._world_matrix(
                controller.reference_prim_path
            )

            world = self.build_world_config(controller.reference_prim_path)
            planner = controller.runtime.native_planner
            if planner is None:
                raise CollisionSceneError(
                    f"controller has no native CuRobo planner during task reset: {key}"
                )
            # The scene port owns the revision-bearing runtime update.
            controller.runtime.update_world(world)
            self.audit_controller(controller)

        self._adopt_world_revision()

    def export(self, episode_dir: str | Path) -> None:
        directory = Path(episode_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for key in self._physics_controller_keys():
            controller = self.scene_ports[key]
            self.audit_controller(controller)
        audit = {
            "mode": self.mode,
            "strict": self.strict,
            "world_revision": self.world_revision,
            "collision_prim_count": len(self.collision_prim_paths),
            "records": [record.to_dict() for record in self.records.values()],
            "attach_prim_paths": dict(self.attach_prim_paths),
            "planning_exclusions": list(self._planning_exclusion_names),
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
