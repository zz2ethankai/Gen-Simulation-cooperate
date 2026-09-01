import glob
import inspect
import logging
import os
import random
from copy import deepcopy
from typing import Dict

import numpy as np
import yaml
from core.cameras import CustomCamera
from core.objects import get_object_cls
from core.robots import get_robot_cls
from core.robots.profile import resolve_fixed_robot_start_pose
from core.tasks.base_task import register_task
from core.utils.dr import update_articulated_objs, update_rigid_objs, update_scenes
from core.utils.asset_path_utils import resolve_asset_root
from core.utils.language import update_language
from core.utils.layout import optimize_2d_manip_layout
from core.utils.region_metadata import (
    merge_source_region_sampling_metadata,
    normalize_legacy_runtime_region_offsets,
    resolve_region_target_name,
)
from core.utils.region_sampler import RandomRegionSampler
from core.utils.rigid_pose import upright_world_orientation
from core.utils.scene_utils import deactivate_selected_prims
from core.utils.transformation_utils import get_orientation
from core.utils.usd_geom_utils import compute_bbox
from core.utils.visual_distractor import set_distractors
from omegaconf import DictConfig
from isaacsim.core.api.materials import PreviewSurface
from isaacsim.core.api.sensors import RigidContactView
from isaacsim.core.api.scenes import Scene
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.prims import (
    delete_prim,
    get_prim_at_path,
    is_prim_path_valid,
)
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.experimental.objects import DomeLight
from omni.physx.scripts import particleUtils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdShade, Vt
from scipy.spatial.transform import Rotation as R


LOGGER = logging.getLogger("de_logger")


@register_task
class BananaBaseTask(BaseTask):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__(name=cfg["name"], offset=cfg["offset"])
        self.cfg = cfg
        self._merge_source_region_sampling_metadata(self.cfg)
        normalize_legacy_runtime_region_offsets(self.cfg)
        self._render = cfg.get("render", True)
        self.asset_root = os.path.abspath(self.cfg["asset_root"])
        self.root_prim_path = os.path.join("/World", f"task_{cfg['task_id']}")
        self.robots = {}
        self.cameras = {}
        self.cameras_info = {}
        self.objects = {}
        self.distractors = {}
        self.fixtures = {}
        self.visuals = {}
        self.pickcontact_views = {}
        self.artcontact_views = {}
        self._contact_views_physics_sim_view = None
        # Isaac Sim 6 invalidates the shared physics tensor view when a prim
        # that is already represented by a SingleRigidPrim is deleted.  Keep
        # the loaded USD subtrees stable across episode resets and restore
        # their state after the current physics view has been initialized.
        self._rigid_object_reset_states = {}
        self._reset_reuse_warning_emitted = False
        self._debug_reset_lifecycle = os.environ.get("INTERNDATA_DEBUG_RESET_LIFECYCLE") == "1"
        self.stage = get_current_stage()
        self.random_region_list = self.cfg.get("random_region_list", [])
        self.current_id = 0

        self.first_set_fluid = True
        self.particleSystemPath = None
        self.particlesPath = None
        self.particlesPbdMaterialPath = None
        self.particlesVisualMaterialPath = None
        self._defaultFluidPath = Sdf.Path("/World/task_0/fluid")

    @staticmethod
    def _merge_source_region_sampling_metadata(cfg):
        """Preserve source-region sampling contracts in runtime regions."""

        merge_source_region_sampling_metadata(cfg)

    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)
        self._set_envmap()
        self.cfg = update_scenes(self.cfg)
        for cfg in self.cfg["arena"]["fixtures"]:
            self.fixtures[cfg["name"]] = self._load_obj(cfg)
            if cfg["target_class"] == "ConveyorObject":
                self.conveyor_velocity = cfg["linear_velocity"][0]

        self.cfg = update_rigid_objs(self.cfg)
        self.cfg = update_articulated_objs(self.cfg)

        for cfg in self.cfg["objects"]:
            self.objects[cfg["name"]] = self._load_obj(cfg)

        for cfg in self.cfg["robots"]:
            self._load_robot(cfg)
        for cfg in self.cfg["cameras"]:
            self._load_camera(cfg)
            if cfg.get("apply_randomization", False):
                self._perturb_camera(
                    self.cameras[cfg["name"]],
                    cfg,
                    max_translation_noise=cfg.get("max_translation_noise", 0.05),
                    max_orientation_noise=cfg.get("max_orientation_noise", 10.0),
                )

        self._task_objects |= self.fixtures | self.objects | self.robots | self.cameras

        # Initialize object regions according to region configs
        self._set_regions()

        optimize_2d_manip_layout(self.cfg["objects"], self.cfg["regions"], self.objects)

        # Object collision filtering (mainly for dynamicpick)
        self.ignore_objects = [obj["name"] for obj in self.cfg["objects"]]
        self.pickcontact_views = self._set_pickcontact_view(self.cfg)
        self.artcontact_views = self._set_artcontact_view(self.cfg)

        # Set up visual distrator (if exists)
        if self.cfg.get("distractors", None):
            cfgs = self._create_distractor_cfg()
            for cfg in cfgs:
                self.distractors[cfg["name"]] = self._load_obj(cfg)

            self.cfg["mem_distractors"] = cfgs
            set_distractors(
                self.objects,
                self.distractors,
                self._task_objects[self.cfg["distractors"]["target"]],
                self.cfg["distractors"],
                cfgs,
            )

        self.capture_rigid_object_states()

        # Isaac Sim 6 finalizes the physics scene immediately after this
        # callback returns from ``set_up_scene``.  Let the workflow author
        # collision collections and any native support proxies at this exact
        # lifecycle boundary, while all task objects are present in USD but
        # before PhysX has built its shape/view representation.  Authoring
        # these schemas after the first ``World.reset`` leaves the USD prims
        # visible while their colliders are absent from the active physics
        # scene.
        before_physics_scene_finalize = getattr(
            self, "_before_physics_scene_finalize", None
        )
        if callable(before_physics_scene_finalize):
            before_physics_scene_finalize()

        # Update language
        self.language_instruction, self.detailed_language_instruction = update_language(self.cfg)

    def _iter_rigid_objects(self):
        """Yield each loaded rigid object exactly once.

        Fixtures, task objects, and distractors can share the same object
        instance in generated scenes.  De-duplicating by identity keeps the
        reset snapshot stable and avoids applying a reset twice.
        """

        seen = set()
        for collection in (self.fixtures, self.objects, self.distractors):
            for obj in collection.values():
                if not isinstance(obj, SingleRigidPrim) or id(obj) in seen:
                    continue
                seen.add(id(obj))
                yield obj

    @staticmethod
    def _rigid_object_state_key(obj):
        return str(getattr(obj, "prim_path", getattr(obj, "base_prim_path", id(obj))))

    @staticmethod
    def _state_value_to_numpy(value):
        """Convert Isaac v2 backend values into CPU NumPy state arrays.

        Isaac Sim 6's native v2 wrappers return CUDA Torch tensors when the
        active backend is Torch.  Calling ``np.asarray`` on such a tensor
        raises before a reset snapshot can be recorded.  The same snapshot
        path is also used with Warp and USD/Gf values, so normalize those
        representations at this boundary rather than making every reset
        caller backend-aware.
        """

        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=float).copy()

    @staticmethod
    def _set_sampled_world_pose(obj, pose):
        """Apply a region-sampler pose in the world frame.

        ``RandomRegionSampler`` computes its translation from world-space USD
        bounds.  Feeding that result to ``set_local_pose`` only happened to
        work for assets whose rigid-body parent had an identity transform.  A
        scaled referenced asset (for example google_scan-handbag) then gets
        scaled toward the origin once more.  Isaac v2 exposes the intended
        world-frame API directly, so keep sampling and application in the
        same frame for every rigid/object implementation.
        """

        translation, orientation = pose
        scale = None
        if hasattr(obj, "get_local_scale"):
            try:
                scale = np.asarray(obj.get_local_scale(), dtype=float).copy()
            except Exception:
                scale = None
        if hasattr(obj, "set_world_pose"):
            obj.set_world_pose(position=translation, orientation=orientation)
        else:
            # Keep the fallback for non-prim task objects that implement only
            # the legacy local-pose interface; all Isaac rigid prims take the
            # native v2 branch above.
            obj.set_local_pose(translation=translation, orientation=orientation)
        # Isaac Sim 6 may rebuild the referenced rigid root while applying a
        # world pose and restore its authored scale.  Preserve the effective
        # task scale across that pose write; otherwise legacy assets whose
        # source stage uses centimeters become tiny again before physics cook.
        if scale is not None and hasattr(obj, "set_local_scale"):
            obj.set_local_scale(scale)

    def _remember_rigid_object_state(self, obj, translation=None, orientation=None):
        """Remember a desired pose without touching the physics view."""

        if not isinstance(obj, SingleRigidPrim):
            return
        try:
            if translation is None or orientation is None:
                translation, orientation = obj.get_local_pose()
            world_translation, world_orientation = obj.get_world_pose()
            state = {
                "translation": self._state_value_to_numpy(translation),
                "orientation": self._state_value_to_numpy(orientation),
                # Isaac v2 rigid prim defaults and set_world_pose both use
                # world coordinates.  Keep the local values only for
                # diagnostics/backward compatibility with old snapshots.
                "world_translation": self._state_value_to_numpy(world_translation),
                "world_orientation": self._state_value_to_numpy(world_orientation),
            }
            if hasattr(obj, "get_local_scale"):
                state["scale"] = self._state_value_to_numpy(obj.get_local_scale())
            if hasattr(obj, "get_visibility"):
                try:
                    state["visibility"] = bool(obj.get_visibility())
                except Exception:
                    LOGGER.debug("Could not snapshot visibility for %s", getattr(obj, "name", obj), exc_info=True)
            for method_name, state_name in (
                ("get_linear_velocity", "linear_velocity"),
                ("get_angular_velocity", "angular_velocity"),
            ):
                getter = getattr(obj, method_name, None)
                if callable(getter):
                    try:
                        state[state_name] = self._state_value_to_numpy(getter())
                    except Exception:
                        LOGGER.debug(
                            "Could not snapshot %s for %s",
                            state_name,
                            getattr(obj, "name", obj),
                            exc_info=True,
                        )
            self._rigid_object_reset_states[self._rigid_object_state_key(obj)] = state

            # Native v2 default state is reapplied by post_reset/world.reset.
            # Recording it here makes every reset path start from the same
            # world-frame pose, including before the workflow's explicit
            # post-step restore runs.
            if hasattr(obj, "set_default_state"):
                obj.set_default_state(
                    position=state["world_translation"],
                    orientation=state["world_orientation"],
                    linear_velocity=np.zeros(3),
                    angular_velocity=np.zeros(3),
                )
        except Exception:
            # A state snapshot is diagnostic/recovery metadata.  It must not
            # turn a scene load into a failure when an asset is not yet bound
            # to the physics backend.
            LOGGER.debug("Could not snapshot rigid object %s", getattr(obj, "name", obj), exc_info=True)

    def capture_rigid_object_states(self):
        """Capture the poses that should survive the next world reset."""

        states = {}
        for obj in self._iter_rigid_objects():
            self._remember_rigid_object_state(obj)
            key = self._rigid_object_state_key(obj)
            if key in self._rigid_object_reset_states:
                states[key] = {
                    name: value.copy() if isinstance(value, np.ndarray) else value
                    for name, value in self._rigid_object_reset_states[key].items()
                }
        if self._debug_reset_lifecycle:
            debug_entries = []
            for key, state in states.items():
                debug_entries.append(
                    {
                        "key": key,
                        "translation": state["translation"].tolist(),
                        "orientation": state["orientation"].tolist(),
                        "world_translation": state.get("world_translation", state["translation"]).tolist(),
                        "world_orientation": state.get("world_orientation", state["orientation"]).tolist(),
                    }
                )
            LOGGER.warning(
                "[ResetLifecycle] captured %d rigid states: %s",
                len(states),
                debug_entries,
            )
        return states

    def restore_rigid_object_states(self, states=None, object_keys=None, world_orientation_overrides=None):
        """Restore rigid object state after the active physics view is ready."""

        states = states or self._rigid_object_reset_states
        if not states:
            return

        objects_by_path = {
            self._rigid_object_state_key(obj): obj for obj in self._iter_rigid_objects()
        }
        for key, state in states.items():
            if object_keys is not None and key not in object_keys:
                continue
            obj = objects_by_path.get(key)
            if obj is None:
                continue
            try:
                if "visibility" in state and hasattr(obj, "set_visibility"):
                    obj.set_visibility(bool(state["visibility"]))
            except Exception:
                LOGGER.debug("Could not restore visibility for %s", key, exc_info=True)

            try:
                # A reset can change the effective parent transform of a
                # referenced asset and it always replaces the PhysX tensor
                # view.  Local pose values are therefore not stable across a
                # reset: the same local numbers can map to a different world
                # pose (the handbag reproduced this as world z ~= -0.0004).
                # Isaac v2 defines SingleRigidPrim's default/set-world pose in
                # world coordinates, so use the captured world pose as the
                # sole restore authority.  A large resulting local value is
                # expected for a child below a 1e-3-scaled asset.
                if "world_translation" in state and "world_orientation" in state:
                    world_orientation = state["world_orientation"]
                    if world_orientation_overrides and key in world_orientation_overrides:
                        world_orientation = world_orientation_overrides[key]
                    obj.set_world_pose(
                        position=state["world_translation"],
                        orientation=world_orientation,
                    )
                elif "translation" in state and "orientation" in state:
                    # Backward-compatible fallback for snapshots created
                    # before world coordinates were recorded.
                    obj.set_local_pose(
                        translation=state["translation"],
                        orientation=state["orientation"],
                    )
                if "scale" in state and hasattr(obj, "set_local_scale"):
                    # set_world_pose/set_local_pose can restore the authored
                    # reference scale.  Apply the captured effective scale
                    # after the pose so reset recovery does not shrink the
                    # rigid asset back to source units.
                    obj.set_local_scale(state["scale"])
                # Placement/reset snapshots are captured at rest.  Keep the
                # explicit zero fallback for old snapshots that predate
                # velocity fields, while preserving a captured state when a
                # caller intentionally records one.
                linear_velocity = state.get("linear_velocity")
                angular_velocity = state.get("angular_velocity")
                if linear_velocity is not None and hasattr(obj, "set_linear_velocity"):
                    obj.set_linear_velocity(linear_velocity)
                if angular_velocity is not None and hasattr(obj, "set_angular_velocity"):
                    obj.set_angular_velocity(angular_velocity)
                if linear_velocity is None or angular_velocity is None:
                    self._zero_object_velocity(obj)
                if self._debug_reset_lifecycle:
                    local_translation, _ = obj.get_local_pose()
                    world_translation, _ = obj.get_world_pose()
                    LOGGER.warning(
                        "[ResetLifecycle] restored key=%s local=%s world=%s",
                        key,
                        self._state_value_to_numpy(local_translation).tolist(),
                        self._state_value_to_numpy(world_translation).tolist(),
                    )
            except Exception:
                LOGGER.warning("Could not restore rigid object state for %s", key, exc_info=True)

    def _fixed_rigid_object_state_keys(self):
        """Return loaded rigid prim keys whose asset selection is fixed.

        ``apply_randomization: false`` means the loaded USD and authored
        placement are fixed; it does not make the body kinematic.  Restricting
        this list to task objects and fixtures avoids resetting a distractor
        pool that is deliberately re-sampled on each episode.
        """

        keys = set()
        configured = (
            (self.cfg.get("objects", []) or [], self.objects),
            ((self.cfg.get("arena", {}) or {}).get("fixtures", []) or [], self.fixtures),
        )
        for cfgs, collection in configured:
            for cfg in cfgs:
                if cfg.get("target_class") != "RigidObject":
                    continue
                if bool(cfg.get("apply_randomization", False)):
                    continue
                obj = collection.get(cfg.get("name"))
                if isinstance(obj, SingleRigidPrim):
                    keys.add(self._rigid_object_state_key(obj))
        return keys

    def audit_fixed_rigid_object_reset(self, label="audit"):
        """Emit a reset audit that is visible on the workflow logger.

        Keep this diagnostic independent of the restore result: a missing
        fixed-object line must distinguish an empty effective selection from
        a task type/wrapper mismatch or a missing captured state.  The audit
        records the effective config flag, wrapper type, state key, configured
        keep-upright flag, and both current/captured world poses.
        """

        rows = []
        configured = (
            ("object", self.cfg.get("objects", []) or [], self.objects),
            (
                "fixture",
                ((self.cfg.get("arena", {}) or {}).get("fixtures", []) or []),
                self.fixtures,
            ),
        )

        def safe_pose(obj):
            try:
                position, orientation = obj.get_world_pose()
                return {
                    "world_translation": self._state_value_to_numpy(position).tolist(),
                    "world_orientation": self._state_value_to_numpy(orientation).tolist(),
                }
            except Exception as exc:
                return {"pose_error": repr(exc)}

        for collection_name, cfgs, collection in configured:
            for cfg in cfgs:
                if cfg.get("target_class") != "RigidObject":
                    continue
                object_name = str(cfg.get("name"))
                obj = collection.get(cfg.get("name"))
                row = {
                    "collection": collection_name,
                    "name": object_name,
                    "apply_randomization": bool(cfg.get("apply_randomization", False)),
                    "object_type": None if obj is None else type(obj).__name__,
                    "is_single_rigid_prim": isinstance(obj, SingleRigidPrim),
                }
                region_cfg = self._get_region_cfg_for_object(object_name)
                sampling_cfg = (region_cfg or {}).get("sampling", {}) or {}
                row["keep_upright"] = (
                    None
                    if sampling_cfg.get("keep_upright") is None
                    else bool(sampling_cfg.get("keep_upright"))
                )
                if isinstance(obj, SingleRigidPrim):
                    key = self._rigid_object_state_key(obj)
                    row["state_key"] = key
                    row["state_present"] = key in self._rigid_object_reset_states
                    row.update(safe_pose(obj))
                    state = self._rigid_object_reset_states.get(key)
                    if state is not None:
                        if "world_translation" in state:
                            row["captured_world_translation"] = self._state_value_to_numpy(
                                state["world_translation"]
                            ).tolist()
                        if "world_orientation" in state:
                            row["captured_world_orientation"] = self._state_value_to_numpy(
                                state["world_orientation"]
                            ).tolist()
                rows.append(row)

        LOGGER.warning(
            "[ResetLifecycle] fixed rigid audit label=%s task_type=%s rows=%s",
            label,
            f"{type(self).__module__}.{type(self).__qualname__}",
            rows,
        )
        return rows

    def restore_fixed_rigid_object_states(self, label="warmup"):
        """Restore fixed rigid prims after reset warmup and report pose deltas.

        Region sampling and the first physics ticks happen before the normal
        planning world is rebuilt.  A dynamic fixed asset can settle or roll
        during that interval; restore the captured world state at this final
        reset boundary so planning starts from the same configured pose.
        Existing prims and tensor views are reused.
        """

        audit_rows = self.audit_fixed_rigid_object_reset(label=f"{label}:before")
        object_keys = self._fixed_rigid_object_state_keys()
        if not object_keys:
            LOGGER.warning(
                "[ResetLifecycle] fixed rigid restore label=%s selected=0 "
                "audit_rows=%d captured_state_keys=%d",
                label,
                len(audit_rows),
                len(self._rigid_object_reset_states),
            )
            return []
        objects_by_path = {
            self._rigid_object_state_key(obj): obj for obj in self._iter_rigid_objects()
        }
        states = self._rigid_object_reset_states
        restored = []
        for key in sorted(object_keys):
            obj = objects_by_path.get(key)
            state = states.get(key)
            if obj is None or state is None:
                continue
            try:
                before_position, before_orientation = obj.get_world_pose()
                before_position = self._state_value_to_numpy(before_position)
                before_orientation = self._state_value_to_numpy(before_orientation)
            except Exception:
                before_position = None
                before_orientation = None

            object_name = getattr(obj, "name", key)
            region_cfg = self._get_region_cfg_for_object(object_name)
            sampling_cfg = (region_cfg or {}).get("sampling", {}) or {}
            keep_upright = bool(sampling_cfg.get("keep_upright", False))
            desired_orientation = self._state_value_to_numpy(state["world_orientation"])
            if keep_upright:
                desired_orientation = upright_world_orientation(desired_orientation)

            # Reuse the common state restore path, including scale,
            # visibility and velocity reset.  Pass one key at a time so the
            # diagnostic below identifies the exact prim.
            self.restore_rigid_object_states(
                states=states,
                object_keys={key},
                world_orientation_overrides={key: desired_orientation},
            )
            self._zero_object_velocity(obj)

            try:
                after_position, after_orientation = obj.get_world_pose()
                after_position = self._state_value_to_numpy(after_position)
                after_orientation = self._state_value_to_numpy(after_orientation)
                desired_position = self._state_value_to_numpy(state["world_translation"])
                translation_delta = float(np.linalg.norm(before_position - desired_position)) if before_position is not None else None
                relative = R.from_quat(before_orientation, scalar_first=True).inv() * R.from_quat(
                    desired_orientation, scalar_first=True
                )
                rotation_delta = float(np.degrees(np.linalg.norm(relative.as_rotvec())))
                restore_relative = R.from_quat(after_orientation, scalar_first=True).inv() * R.from_quat(
                    desired_orientation, scalar_first=True
                )
                restore_error_deg = float(np.degrees(np.linalg.norm(restore_relative.as_rotvec())))
            except Exception:
                after_position = None
                translation_delta = None
                rotation_delta = None
                restore_error_deg = None

            entry = {
                "key": key,
                "label": label,
                "keep_upright": keep_upright,
                "translation_delta_m": translation_delta,
                "rotation_delta_deg": rotation_delta,
                "restore_error_deg": restore_error_deg,
                "normalized_orientation": desired_orientation.tolist(),
            }
            restored.append(entry)
            LOGGER.info("[ResetLifecycle] fixed rigid restore %s", entry)
        LOGGER.warning(
            "[ResetLifecycle] fixed rigid restore label=%s selected=%d restored=%d "
            "entries=%s",
            label,
            len(object_keys),
            len(restored),
            restored,
        )
        return restored

    def debug_reset_dynamics(self, label):
        """Log rigid poses/velocities and fixture collision USD state.

        Isaac Sim 6 can leave a valid world pose in the tensor view while the
        corresponding collision shape is missing or filtered. The diagnostic
        is intentionally opt-in and read-only so normal episodes do not pay
        for extra USD traversal or log volume.
        """

        if not self._debug_reset_lifecycle:
            return

        entries = []
        for obj in self._iter_rigid_objects():
            entry = {"key": self._rigid_object_state_key(obj)}
            prim_view = getattr(obj, "_prim_view", None)
            if prim_view is not None:
                physics_view = getattr(prim_view, "_physics_view", None)
                entry["physics_view_bound"] = physics_view is not None
                entry["physics_view_type"] = (
                    None if physics_view is None else type(physics_view).__name__
                )
                entry["physics_num_shapes"] = getattr(prim_view, "_num_shapes", None)
            try:
                entry["local_scale"] = self._state_value_to_numpy(obj.get_local_scale()).tolist()
            except Exception as exc:
                entry["local_scale_error"] = repr(exc)
            proxy_path = getattr(obj, "_physics_collision_proxy_path", None)
            if proxy_path:
                proxy_prim = get_prim_at_path(proxy_path)
                proxy_entry = {
                    "path": proxy_path,
                    "valid": bool(proxy_prim.IsValid()),
                }
                if proxy_prim.IsValid():
                    collision_enabled = proxy_prim.GetAttribute("physics:collisionEnabled")
                    proxy_entry["collision_enabled"] = (
                        None if not collision_enabled.IsValid() else collision_enabled.Get()
                    )
                    try:
                        proxy_bbox = UsdGeom.BBoxCache(
                            Usd.TimeCode.Default(),
                            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
                            useExtentsHint=False,
                        ).ComputeWorldBound(proxy_prim).ComputeAlignedBox()
                        proxy_entry["world_bbox_min"] = [float(value) for value in proxy_bbox.GetMin()]
                        proxy_entry["world_bbox_max"] = [float(value) for value in proxy_bbox.GetMax()]
                    except Exception as exc:
                        proxy_entry["world_bbox_error"] = repr(exc)
                entry["collision_proxy"] = proxy_entry
            try:
                world_translation, world_orientation = obj.get_world_pose()
                entry["world"] = self._state_value_to_numpy(world_translation).tolist()
                entry["orientation"] = self._state_value_to_numpy(world_orientation).tolist()
            except Exception as exc:
                entry["pose_error"] = repr(exc)
            for method_name, key_name in (
                ("get_linear_velocity", "linear_velocity"),
                ("get_angular_velocity", "angular_velocity"),
            ):
                try:
                    entry[key_name] = self._state_value_to_numpy(getattr(obj, method_name)()).tolist()
                except Exception as exc:
                    entry[f"{key_name}_error"] = repr(exc)
            entries.append(entry)

        fixture_collisions = {}
        for name, obj in self.fixtures.items():
            get_debug_info = getattr(obj, "get_collision_debug_info", None)
            if callable(get_debug_info):
                try:
                    fixture_collisions[name] = get_debug_info()
                except Exception as exc:
                    fixture_collisions[name] = {"error": repr(exc)}

        LOGGER.warning(
            "[ResetLifecycle] dynamics label=%s rigid_states=%s fixture_collisions=%s",
            label,
            entries,
            fixture_collisions,
        )

    def individual_reset(self):
        """Reset episode bookkeeping without deleting live USD rigid bodies.

        The old implementation changed randomized asset paths and deleted
        their prims here.  In Isaac Sim 6 those prims are already represented
        by tensor views, so deletion invalidates the shared simulation view
        before the replacement object can initialize.  Asset selection is
        intentionally performed once during set_up_scene; episode resets
        randomize/reapply poses in-place instead.
        """

        self.current_id += 1
        if not self._reset_reuse_warning_emitted:
            randomized = [
                cfg["name"]
                for cfg in list(self.cfg.get("objects", []))
                + list(self.cfg.get("arena", {}).get("fixtures", []))
                if cfg.get("apply_randomization", False)
            ]
            if randomized or self.cfg.get("distractors"):
                LOGGER.info(
                    "[ResetLifecycle] reusing loaded USD prims for episode reset; "
                    "asset reload disabled for %s",
                    randomized or ["distractors"],
                )
            self._reset_reuse_warning_emitted = True

        optimize_2d_manip_layout(self.cfg["objects"], self.cfg["regions"], self.objects)
        self.pickcontact_views = self._set_pickcontact_view(self.cfg)
        self.artcontact_views = self._set_artcontact_view(self.cfg)

    def reset_fixed_rigid_objects(self):
        """Restore non-randomized rigid objects after a failed generation retry."""
        for cfg in self.cfg["objects"]:
            if cfg.get("apply_randomization", False):
                continue
            if cfg.get("target_class") != "RigidObject":
                continue
            obj = self.objects.get(cfg["name"])
            if obj is None:
                continue

            # ``individual_reset`` intentionally does not resample regions.
            # Restore the pose captured after the current layout was
            # randomized.  Re-running A_on_B_region_sampler here derives Z
            # from the post-failure pose (for example, an apple held over the
            # tray), which changes the canonical reset state and contaminates
            # the next plan_with_render episode.
            key = self._rigid_object_state_key(obj)
            state = self._rigid_object_reset_states.get(key)
            if (
                state is not None
                and "world_translation" in state
                and "world_orientation" in state
            ):
                self.restore_rigid_object_states(states={key: state}, object_keys={key})
                self._zero_object_velocity(obj)
                continue

            orientation = get_orientation(cfg.get("euler"), cfg.get("quaternion"))
            translation = cfg.get("translation")
            if translation is None:
                translation = deepcopy(obj.get_local_pose()[0])
            obj.set_local_pose(translation=translation, orientation=orientation)

            region_cfg = self._get_region_cfg_for_object(cfg["name"])
            if region_cfg is not None:
                pose = self._get_deterministic_region_pose(obj, region_cfg)
                if pose is not None:
                    self._set_sampled_world_pose(obj, pose)

            self._zero_object_velocity(obj)
            self._remember_rigid_object_state(obj)

    def _get_region_cfg_for_object(self, object_name):
        for region_cfg in self.cfg.get("regions", []):
            if region_cfg.get("object") == object_name:
                return region_cfg
        return None

    def _get_deterministic_region_pose(self, obj, region_cfg):
        if region_cfg.get("random_type") != "A_on_B_region_sampler":
            return None
        random_config = deepcopy(region_cfg.get("random_config", {}))
        sampling_cfg = region_cfg.get("sampling", {}) or {}
        if "keep_upright" not in random_config and "keep_upright" in sampling_cfg:
            random_config["keep_upright"] = bool(sampling_cfg["keep_upright"])
        pos_range = random_config.get("pos_range")
        yaw_rotation = random_config.get("yaw_rotation")
        if pos_range is None or yaw_rotation is None:
            return None

        pos_mid = ((np.asarray(pos_range[0], dtype=float) + np.asarray(pos_range[1], dtype=float)) / 2.0).tolist()
        yaw_mid = float((float(yaw_rotation[0]) + float(yaw_rotation[1])) / 2.0)
        random_config["pos_range"] = [pos_mid, pos_mid]
        random_config["yaw_rotation"] = [yaw_mid, yaw_mid]

        target_name = region_cfg.get("target")
        if target_name not in self._task_objects:
            return None
        target = self._task_objects[target_name]
        if "sub_tgt_prim" in region_cfg:
            target = SingleXFormPrim(prim_path=target.prim_path + region_cfg["sub_tgt_prim"])

        sampler_fn = RandomRegionSampler.A_on_B_region_sampler
        return sampler_fn(obj, target, **self._filter_sampler_random_config(sampler_fn, random_config))

    @staticmethod
    def _zero_object_velocity(obj):
        if hasattr(obj, "set_linear_velocity"):
            obj.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        if hasattr(obj, "set_angular_velocity"):
            obj.set_angular_velocity(np.array([0.0, 0.0, 0.0]))

    @staticmethod
    def _object_reload_root_path(obj, cfg):
        target_class = cfg.get("target_class")
        if target_class == "RigidObject":
            return getattr(obj, "base_prim_path", os.path.dirname(obj.prim_path))
        if target_class == "ArticulatedObject":
            return getattr(obj, "object_prim_path", obj.prim_path)
        return obj.prim_path

    def individual_reset_from_mem(self):
        optimize_2d_manip_layout(self.cfg["objects"], self.cfg["regions"], self.objects)
        self.pickcontact_views = self._set_pickcontact_view(self.cfg)
        self.artcontact_views = self._set_artcontact_view(self.cfg)

    def individual_randomize(self):
        # Randomize objects in regions
        self._set_regions()

        # Update envmap, fixture textures, and camera poses
        self._set_envmap()
        self._set_fixture_textures()
        self._set_camera_poses()

        # Set up visual distractor (if exists, resample and update mem)
        if self.cfg.get("distractors", None):
            cfgs = self.cfg.get("mem_distractors") or []
            if not self.distractors and cfgs:
                self._rebuild_distractors(cfgs, use_mem=False)
            if self.distractors and cfgs:
                # Reposition the existing distractor pool; never delete a
                # rigid USD subtree after its tensor view has been created.
                active_cfgs = [cfg for cfg in cfgs if cfg["name"] in self.distractors]
                if len(active_cfgs) == len(self.distractors):
                    set_distractors(
                        self.objects,
                        self.distractors,
                        self._task_objects[self.cfg["distractors"]["target"]],
                        self.cfg["distractors"],
                        active_cfgs,
                    )

        self.capture_rigid_object_states()

        # Update language
        self.language_instruction, self.detailed_language_instruction = update_language(self.cfg)

    def individual_randomize_from_mem(self):
        # Randomize objects in regions (re-sample placements)
        self._set_regions()

        # Update envmap, fixture textures, and camera poses
        self._set_envmap()
        self._set_fixture_textures()
        self._set_camera_poses()

        # Rebuild visual distractors from mem (no new random placement)
        if self.cfg.get("mem_distractors", None) and self.cfg.get("distractors", None):
            cfgs = self.cfg["mem_distractors"]
            if not self.distractors:
                self._rebuild_distractors(cfgs, use_mem=True)
            if self.distractors:
                active_cfgs = [cfg for cfg in cfgs if cfg["name"] in self.distractors]
                if len(active_cfgs) == len(self.distractors):
                    set_distractors(
                        self.objects,
                        self.distractors,
                        self._task_objects[self.cfg["distractors"]["target"]],
                        self.cfg["distractors"],
                        active_cfgs,
                    )

        self.capture_rigid_object_states()

        # Update language
        self.language_instruction, self.detailed_language_instruction = update_language(self.cfg)

    def post_reset(self):
        for _, robot in self.robots.items():
            robot.initialize()
        self.set_fixed_robot_start_poses()
        for cfg in self.cfg["objects"]:
            if cfg["target_class"] == "ArticulatedObject":
                self.objects[cfg["name"]].initialize()
        # Cameras are task-owned sensors rather than World scene objects, so
        # World.reset() does not invoke their acquisition-timing reset.  Keep
        # their configured/mounted pose intact while clearing the previous
        # episode's render timestamp and frame cache.
        for camera in self.cameras.values():
            post_reset = getattr(camera, "post_reset", None)
            if callable(post_reset):
                post_reset()

    def initialize_rigid_objects(self, physics_sim_view=None):
        """Attach newly referenced rigid USDs to the Isaac Sim 6 tensor view.

        The wrapper is constructed before reset, while the workflow attaches
        its physics handle after one world step, when the referenced USD body
        has been consumed by the PhysX tensor backend.
        """
        candidates = list(self.fixtures.values()) + list(self.objects.values()) + list(self.distractors.values())
        for obj in candidates:
            if not isinstance(obj, SingleRigidPrim):
                continue
            # ``SingleRigidPrim.initialize`` delegates to ``RigidPrim``.  In
            # Isaac Sim 6 that method is a no-op whenever its private
            # ``_physics_view`` is non-null, even if World.reset() has already
            # replaced the underlying simulation view.  A reset can therefore
            # leave the wrapper attached to an invalid backend without any
            # public API reporting it.  Clear only the Python-side handle and
            # ask the native v2 wrapper to bind a fresh view; the USD prim is
            # never deleted or recreated.
            prim_view = getattr(obj, "_prim_view", None)
            if prim_view is not None and hasattr(prim_view, "_physics_view"):
                prim_view._physics_view = None
                prim_view._num_shapes = None
            # Always pass the active World view explicitly.  Calling the
            # private callback with ``None`` can create a wrapper whose USD
            # local pose changes while its PhysX world pose remains attached
            # to the default/origin view.
            obj.initialize(physics_sim_view=physics_sim_view)

    def initialize_contact_views(self, physics_sim_view=None):
        """Initialize contact sensors after the first post-reset physics tick.

        Isaac Sim 6 can invoke ``post_reset`` before PhysX has attached the
        USD stage to the tensor backend.  Initializing ``RigidContactView``
        from that callback invalidates the simulation view for Panda-Omron
        Scene-8 tasks.  The workflow calls this method after one world step,
        when the PhysX stage is live.  A view from the active World is always
        passed through; creating an implicit tensor view here is forbidden
        because it can invalidate the view finalized by ``World.reset``.
        """
        if physics_sim_view is None:
            from isaacsim.core.api.simulation_context import SimulationContext

            simulation_context = SimulationContext.instance()
            if simulation_context is not None:
                physics_sim_view = simulation_context.physics_sim_view
        if physics_sim_view is None:
            raise RuntimeError(
                "Cannot initialize task contact views before Isaac Sim physics is ready"
            )

        all_views = []
        for views_by_robot in (self.pickcontact_views, self.artcontact_views):
            for views_by_arm in views_by_robot.values():
                for views_by_object in views_by_arm.values():
                    all_views.extend(views_by_object.values())

        for view in all_views:
            # RigidContactView does not validate that its cached handle
            # belongs to the current World after a hard reset.  Compare the
            # actual parent view so retries never retain a handle from the
            # invalidated pre-reset simulation.
            if (
                getattr(view, "_physics_sim_view", None) is physics_sim_view
                and view.is_physics_handle_valid()
            ):
                continue
            view.initialize(physics_sim_view=physics_sim_view)

        self._contact_views_physics_sim_view = physics_sim_view

    def apply_action(self, action: Dict[str, Dict[str, np.ndarray]]):
        for name in action.keys():
            self.robots[name].apply_action(**action[name])

    def get_observations(self):
        obs = {
            "robots": {},
            "objects": {},
            "cameras": {},
        }
        for name, robot in self.robots.items():
            obs["robots"][name] = robot.get_observations()
        for name, obj in self.objects.items():
            obs["objects"][name] = obj.get_observations()
        if self._render:
            for name, camera in self.cameras.items():
                obs["cameras"][name] = camera.get_observations()
        return obs

    # Load robot, camera and objects
    def _load_robot(self, cfg):
        robot = get_robot_cls(cfg["target_class"])(
            self.asset_root,
            self.root_prim_path,
            cfg,
        )
        orientation = get_orientation(cfg.get("euler"), cfg.get("quaternion"))
        robot.set_local_pose(
            translation=cfg.get("translation", [0.0, 0.0, 0.0]),
            orientation=orientation,
        )
        robot.set_local_scale(cfg.get("scale", [1.0, 1.0, 1.0]))
        self.robots[cfg["name"]] = robot

    def _load_camera(self, cfg):
        cameras_root = os.path.join(self.root_prim_path, "cameras")
        parent_cfg = str(cfg.get("parent", "") or "").strip()
        if parent_cfg:
            camera_parent_path = parent_cfg if parent_cfg.startswith("/") else os.path.join(self.root_prim_path, parent_cfg)
            if not is_prim_path_valid(camera_parent_path):
                raise ValueError(f"Camera parent prim does not exist: {camera_parent_path}")
            camera_mount_path = os.path.join(camera_parent_path, f"{cfg['name']}_mount")
            mount_paths = [camera_mount_path]
        else:
            camera_mount_path = os.path.join(cameras_root, cfg["name"])
            mount_paths = [cameras_root, camera_mount_path]

        camera_prim_path = os.path.join(camera_mount_path, "camera")
        if is_prim_path_valid(camera_mount_path):
            delete_prim(camera_mount_path)
            self.cameras.pop(cfg["name"], None)
            self.cameras_info.pop(cfg["name"], None)

        for path in mount_paths:
            if is_prim_path_valid(path):
                continue
            xform = SingleXFormPrim(prim_path=path)
            xform.set_local_pose(translation=[0.0, 0.0, 0.0], orientation=[1.0, 0.0, 0.0, 0.0])

        # Load camera params from external file if camera_file is specified
        camera_file_path = cfg["camera_file"]
        with open(camera_file_path, "r", encoding="utf-8") as f:
            camera_params = yaml.safe_load(f)
        cfg = dict(cfg)
        cfg["params"] = camera_params

        # Use a single generic camera implementation.
        camera = CustomCamera(
            cfg=cfg,
            prim_path=camera_prim_path,
            root_prim_path=self.root_prim_path,
            name=cfg["name"],
        )

        camera.set_local_pose(
            translation=cfg["translation"],
            orientation=cfg["orientation"],
            camera_axes=cfg["camera_axes"],
        )

        self.cameras[cfg["name"]] = camera
        self.cameras_info[cfg["name"]] = {
            "translation": deepcopy(camera.get_local_pose()[0]),
            "orientation": deepcopy(camera.get_local_pose()[1]),
        }

    def _load_obj(self, cfg: DictConfig):
        """Create and initialize any object based on cfg['target_class']."""
        target_class = cfg.get("target_class") or cfg["target_class"]
        obj_cls = get_object_cls(target_class)

        # Decide root prim and constructor args
        root_prim_path = self.root_prim_path
        object_asset_root = resolve_asset_root(self.asset_root, cfg)
        ctor_args = [object_asset_root]

        if target_class == "XFormObject" and cfg.get("parent_obj", None):
            root_prim_path = self.objects[cfg["parent_obj"]].prim_path

        ctor_args.append(root_prim_path)

        if target_class == "ConveyorObject":
            ctor_args.append(self.stage)

        ctor_args.append(cfg)
        obj = obj_cls(*ctor_args)

        # Optional texture (for non-shape objects)
        if cfg.get("texture") and target_class not in ("ShapeObject",):
            obj.apply_texture(object_asset_root, cfg.get("texture"))

        orientation = get_orientation(cfg.get("euler"), cfg.get("quaternion"))
        obj.set_local_pose(translation=cfg.get("translation"), orientation=orientation)
        configured_scale = np.asarray(cfg.get("scale", [1.0, 1.0, 1.0]), dtype=float)
        configured_scale *= float(getattr(obj, "asset_scale_correction", 1.0))
        obj.set_local_scale(configured_scale)
        obj.set_visibility(cfg.get("visible", True))

        # Extra behavior per type
        if target_class == "ArticulatedObject":
            obj.get_joint_position(self.stage)
        elif target_class == "ShapeObject":
            material = PreviewSurface(
                prim_path="/World/Materials/Red",
                color=np.array(cfg.get("color", np.array([1, 0, 0]))),
            )
            obj.apply_visual_material(material)

        # Special handling for scene object (only for general rigid/geometry)
        if target_class in ("RigidObject", "GeometryObject") and obj.name == "scene":
            deactivate_selected_prims(
                obj.prim, ["pan", "hearth", "ceiling", "__default_setting", "other", "microwave"], ["light"]
            )

        return obj

    # Set
    def _set_artcontact_view(self, cfg):
        artcontact_views = {}
        for cfg_skill_dict in cfg["skills"]:
            for robot_name, robot_skill_list in cfg_skill_dict.items():
                for lr_skill_dict in robot_skill_list:
                    for lr_name, lr_skill_list in lr_skill_dict.items():
                        for lr_skill in lr_skill_list:
                            if lr_skill.get("name") == "open" or lr_skill.get("name") == "close":
                                if robot_name not in artcontact_views:
                                    artcontact_views[robot_name] = {}
                                if lr_name not in artcontact_views[robot_name]:
                                    artcontact_views[robot_name][lr_name] = {}

                                object_name = lr_skill["objects"][0]
                                robot = self.robots[robot_name]
                                filter_paths_expr = (
                                    robot.fl_filter_paths_expr if lr_name == "left" else robot.fr_filter_paths_expr
                                )
                                forbid_collision_paths = (
                                    robot.fl_forbid_collision_paths
                                    if lr_name == "left"
                                    else robot.fr_forbid_collision_paths
                                )
                                if (object_name + "_fingers_link") not in artcontact_views[robot_name][lr_name]:
                                    artcontact_views[robot_name][lr_name][
                                        object_name + "_fingers_link"
                                    ] = RigidContactView(
                                        prim_paths_expr=self._task_objects[object_name].object_link_path,
                                        filter_paths_expr=filter_paths_expr,
                                    )
                                if (object_name + "_fingers_base") not in artcontact_views[robot_name][lr_name]:
                                    artcontact_views[robot_name][lr_name][
                                        object_name + "_fingers_base"
                                    ] = RigidContactView(
                                        prim_paths_expr=self._task_objects[object_name].object_base_path,
                                        filter_paths_expr=filter_paths_expr,
                                    )

                                if (object_name + "_forbid_collision") not in artcontact_views[robot_name][lr_name]:
                                    art_obj = self._task_objects[object_name]
                                    art_obj_cfg = getattr(art_obj, "cfg", {})
                                    art_forbid_paths = art_obj_cfg.get(
                                        "forbid_collision_paths", getattr(art_obj, "forbid_collision_paths", None)
                                    )
                                    if art_forbid_paths:
                                        art_forbid_paths = [
                                            path
                                            if path.startswith("/")
                                            else f"{art_obj.object_prim_path}/{path}"
                                            for path in art_forbid_paths
                                        ]
                                        art_forbid_paths = self._collapse_contact_sensor_paths(art_forbid_paths)
                                    else:
                                        art_forbid_paths = art_obj.object_prim_path + "/instance/*"
                                    artcontact_views[robot_name][lr_name][
                                        object_name + "_forbid_collision"
                                    ] = RigidContactView(
                                        prim_paths_expr=art_forbid_paths,
                                        filter_paths_expr=forbid_collision_paths,
                                    )

        return artcontact_views

    @staticmethod
    def _collapse_contact_sensor_paths(paths):
        if isinstance(paths, str):
            return paths
        if not paths:
            return paths
        parents = {path.rsplit("/", 1)[0] for path in paths}
        if len(parents) == 1:
            return f"{parents.pop()}/*"
        common_prefix = os.path.commonpath(paths)
        if common_prefix in paths:
            common_prefix = common_prefix.rsplit("/", 1)[0]
        return f"{common_prefix}/*"

    def _set_pickcontact_view(self, cfg):
        pickcontact_views = {}
        for cfg_skill_dict in cfg["skills"]:
            for robot_name, robot_skill_list in cfg_skill_dict.items():
                for lr_skill_dict in robot_skill_list:
                    for lr_name, lr_skill_list in lr_skill_dict.items():
                        for lr_skill in lr_skill_list:
                            if "pick" in lr_skill.get("name"):
                                if robot_name not in pickcontact_views:
                                    pickcontact_views[robot_name] = {}
                                if lr_name not in pickcontact_views[robot_name]:
                                    pickcontact_views[robot_name][lr_name] = {}
                                object_name = lr_skill["objects"][0]
                                obj = self.objects[object_name]
                                # RigidContactView must match the RigidBodyAPI prim, not the mesh prim.
                                prim_paths_expr = obj.prim_path
                                robot = self.robots[robot_name]
                                filter_paths_expr = (
                                    robot.fl_filter_paths_expr if lr_name == "left" else robot.fr_filter_paths_expr
                                )
                                if object_name not in pickcontact_views[robot_name][lr_name]:
                                    pickcontact_views[robot_name][lr_name][object_name] = RigidContactView(
                                        prim_paths_expr=prim_paths_expr, filter_paths_expr=filter_paths_expr
                                    )
        return pickcontact_views

    def _set_regions(self):
        """Randomize object poses according to region configs."""
        random_region_list = deepcopy(self.random_region_list)
        for cfg in self.cfg["regions"]:
            obj = self._task_objects[cfg["object"]]
            if cfg["object"] in self.robots:
                robot_cfg = self._robot_cfg_by_name(cfg["object"])
                fixed_pose = resolve_fixed_robot_start_pose(cfg, robot_cfg)
                if fixed_pose is not None:
                    translation, euler, quaternion = fixed_pose
                    obj.set_mobile_base_world_pose(
                        translation,
                        get_orientation(euler, quaternion),
                    )
                    continue
            target_name = resolve_region_target_name(cfg)
            tgt = self._task_objects[target_name]
            if "sub_tgt_prim" in cfg:
                tgt = SingleXFormPrim(prim_path=tgt.prim_path + cfg["sub_tgt_prim"])
            if self._debug_reset_lifecycle:
                try:
                    obj_bbox = compute_bbox(obj.prim)
                    tgt_bbox = compute_bbox(tgt.prim)
                    obj_world = self._state_value_to_numpy(obj.get_world_pose()[0]).tolist()
                    obj_scale = self._state_value_to_numpy(obj.get_local_scale()).tolist()
                    xform_rows = []
                    for prim in Usd.PrimRange(obj.prim):
                        if not prim.IsA(UsdGeom.Xformable):
                            continue
                        xform_rows.append(
                            {
                                "path": str(prim.GetPath()),
                                "translate": prim.GetAttribute("xformOp:translate").Get(),
                                "scale": prim.GetAttribute("xformOp:scale").Get(),
                            }
                        )
                        if len(xform_rows) >= 8:
                            break
                    LOGGER.warning(
                        "[RegionDebug] object=%s target=%s obj_world=%s "
                        "scale=%s usd=%s obj_bbox_z=(%.6f,%.6f) "
                        "target_bbox_z=(%.6f,%.6f) stage_mpu=%s "
                        "prim=%s xforms=%s",
                        cfg["object"],
                        target_name,
                        obj_world,
                        obj_scale,
                        getattr(obj, "usd_path", "<unknown>"),
                        float(obj_bbox.min[2]),
                        float(obj_bbox.max[2]),
                        float(tgt_bbox.min[2]),
                        float(tgt_bbox.max[2]),
                        UsdGeom.GetStageMetersPerUnit(self.stage),
                        str(obj.prim.GetPath()),
                        xform_rows,
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "[RegionDebug] object=%s target=%s pre_sample_error=%s",
                        cfg["object"],
                        target_name,
                        exc,
                    )
            random_config = deepcopy(cfg.get("random_config", {}))
            sampling_cfg = cfg.get("sampling", {}) or {}
            if "keep_upright" not in random_config and "keep_upright" in sampling_cfg:
                random_config["keep_upright"] = bool(sampling_cfg["keep_upright"])
            if "priority" in cfg:
                if cfg["priority"]:
                    idx = random.choice(cfg["priority"])
                else:
                    idx = random.randint(0, len(random_region_list) - 1)
                random_config = (random_region_list.pop(idx))["random_config"]
                if "keep_upright" not in random_config and "keep_upright" in sampling_cfg:
                    random_config["keep_upright"] = bool(sampling_cfg["keep_upright"])
                sampler_fn = getattr(RandomRegionSampler, cfg["random_type"])
                pose = sampler_fn(obj, tgt, **self._filter_sampler_random_config(sampler_fn, random_config))
                self._set_sampled_world_pose(obj, pose)
            elif "container" in cfg:
                container = self._task_objects[cfg["container"]]
                obj_trans = container.get_local_pose()[0]
                x_bias = random.choice(container.gap) if container.gap else 0
                obj_trans[0] += x_bias
                obj_trans[2] += cfg["z_init"]
                obj_ori = obj.get_local_pose()[1]
                obj.set_local_pose(obj_trans, obj_ori)
            elif "target2" in cfg:
                tgt2 = self._task_objects[cfg["target2"]]
                sampler_fn = getattr(RandomRegionSampler, cfg["random_type"])
                pose = sampler_fn(obj, tgt, tgt2, **self._filter_sampler_random_config(sampler_fn, random_config))
                self._set_sampled_world_pose(obj, pose)
            else:
                sampler_fn = getattr(RandomRegionSampler, cfg["random_type"])
                pose = sampler_fn(obj, tgt, **self._filter_sampler_random_config(sampler_fn, random_config))
                self._set_sampled_world_pose(obj, pose)

            if self._debug_reset_lifecycle:
                try:
                    sampled_translation = self._state_value_to_numpy(pose[0]).tolist()
                    sampled_world = self._state_value_to_numpy(obj.get_world_pose()[0]).tolist()
                    LOGGER.warning(
                        "[RegionDebug] object=%s target=%s sampled=%s applied_world=%s",
                        cfg["object"],
                        target_name,
                        sampled_translation,
                        sampled_world,
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "[RegionDebug] object=%s target=%s post_sample_error=%s",
                        cfg["object"],
                        target_name,
                        exc,
                    )

            # A placed RigidObject starts from rest.  Without this reset, a
            # previous failed episode can leave residual velocity on a fixed
            # tabletop target, making the first Pick transit look like a
            # moving obstacle and consuming the dynamic replan budget.
            if isinstance(obj, SingleRigidPrim):
                self._zero_object_velocity(obj)

    def set_fixed_robot_start_poses(self):
        for region_cfg in self.cfg.get("regions", []):
            robot_name = region_cfg.get("object")
            if robot_name not in self.robots:
                continue
            robot_cfg = self._robot_cfg_by_name(robot_name)
            fixed_pose = resolve_fixed_robot_start_pose(region_cfg, robot_cfg)
            if fixed_pose is None:
                continue
            translation, euler, quaternion = fixed_pose
            self.robots[robot_name].reset_mobile_base_world_state(
                translation,
                get_orientation(euler, quaternion),
            )

    @staticmethod
    def _filter_sampler_random_config(sampler_fn, random_config):
        """Pass only parameters declared by the region sampler."""
        signature = inspect.signature(sampler_fn)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return random_config
        accepted = set(signature.parameters)
        return {key: value for key, value in random_config.items() if key in accepted}

    def _robot_cfg_by_name(self, robot_name):
        for robot_cfg in self.cfg["robots"]:
            if robot_cfg["name"] == robot_name:
                return robot_cfg
        raise KeyError(f"Robot region references unknown robot config: {robot_name}")

    def _set_fixture_textures(self):
        """Apply or randomize textures for arena fixtures (table, floor, background)."""
        for cfg in self.cfg["arena"]["fixtures"]:
            if cfg.get("texture"):
                self.fixtures[cfg["name"]].apply_texture(self.asset_root, cfg.get("texture"))

    def _set_camera_poses(self):
        """Randomize camera poses according to camera configs."""
        for cfg in self.cfg["cameras"]:
            if cfg.get("apply_randomization", False):
                self._perturb_camera(
                    self.cameras[cfg["name"]],
                    cfg,
                    max_translation_noise=cfg.get("max_translation_noise", 0.05),
                    max_orientation_noise=cfg.get("max_orientation_noise", 10.0),
                )

    def _set_envmap(self):
        """Randomize or reset the environment map (HDR dome light)."""
        cfg = self.cfg["env_map"]
        if cfg.get("light_type", "DomeLight") == "DomeLight":
            direct_path = cfg.get("path")
            if direct_path:
                direct_path = os.path.expanduser(str(direct_path))
                if not os.path.isabs(direct_path):
                    direct_path = os.path.join(self.asset_root, direct_path)
                envmap_hdr_path_list = [os.path.abspath(direct_path)]
                searched = envmap_hdr_path_list[0]
            else:
                envmap_lib = cfg.get("envmap_lib")
                if not envmap_lib:
                    raise ValueError(
                        "env_map must declare either a direct HDR path in 'path' "
                        "or an HDR directory in 'envmap_lib'"
                    )
                envmap_dir = os.path.join(self.asset_root, str(envmap_lib))
                envmap_hdr_path_list = glob.glob(os.path.join(envmap_dir, "*.hdr"))
                searched = os.path.abspath(envmap_dir)
            envmap_hdr_path_list.sort()
            if not envmap_hdr_path_list or not all(
                os.path.isfile(path) for path in envmap_hdr_path_list
            ):
                raise FileNotFoundError(
                    "No HDR envmap files found for task "
                    f"{self.cfg.get('name')!r}: asset_root={self.asset_root!r}, "
                    f"path={cfg.get('path')!r}, envmap_lib={cfg.get('envmap_lib')!r}, "
                    f"searched={searched!r}"
                )
            if cfg.get("apply_randomization", False):
                envmap_id = int(np.random.randint(0, len(envmap_hdr_path_list)))
                intensity = float(np.random.uniform(cfg["intensity_range"][0], cfg["intensity_range"][1]))
                rotation = [
                    float(np.random.uniform(cfg["rotation_range"][0], cfg["rotation_range"][1]))
                    for _ in range(3)
                ]
            else:
                envmap_id = 0
                # Direct-path conversion output uses a normalized multiplier
                # (1.0 = the legacy fixed DomeLight intensity of 1000).
                intensity = 1000.0 * float(cfg.get("intensity", 1.0))
                rotation_deg = cfg.get("rotation_deg", 0.0)
                if isinstance(rotation_deg, (list, tuple)):
                    rotation = [float(value) for value in rotation_deg]
                    if len(rotation) != 3:
                        raise ValueError("env_map.rotation_deg must be a scalar or xyz triple")
                else:
                    rotation = [0.0, 0.0, float(rotation_deg)]
            dome_prim_path = f"{self.root_prim_path}/DomeLight"
            envmap_hdr_path = envmap_hdr_path_list[envmap_id]

            # Isaac Sim 6 lighting API: this wrapper creates a missing DomeLight
            # or wraps the existing scene light at the same USD path. The
            # wrapper's default transform setup creates the xformOp:orient
            # property required by set_local_poses().
            self.dome_light_prim = DomeLight(dome_prim_path)
            rotation_quaternion = R.from_euler(
                "xyz", rotation, degrees=True
            ).as_quat(scalar_first=True)
            self.dome_light_prim.set_local_poses(orientations=[rotation_quaternion])
            self.dome_light_prim.set_intensities(float(intensity))
            self.dome_light_prim.set_texture_files(envmap_hdr_path)

    def _set_fluid(self):
        # Particle params
        self.particleContactOffset = self.cfg["fluid"].get("particleContactOffset", 0.005)
        self.particleSpacing = self.particleContactOffset * self.cfg["fluid"].get("spacing_scale", 1.2)

        offset = self._get_container_center()
        numParticlesX = self.cfg["fluid"].get("numParticlesX", 7)
        numParticlesY = self.cfg["fluid"].get("numParticlesY", 7)
        numParticlesZ = self.cfg["fluid"].get("numParticlesZ", 450)
        lower_x = (numParticlesX - 1) * self.particleSpacing * -0.5 + offset[0].item()
        lower_y = (numParticlesY - 1) * self.particleSpacing * -0.5 + offset[1].item()
        lower_z = (
            (numParticlesZ - 1) * self.particleSpacing * -0.5 + offset[2].item()
            if self.cfg["fluid"].get("center_z", False)
            else offset[2].item()
        )
        z_offset = self.cfg["fluid"].get("z_offset", 0.0)
        lower_z += z_offset
        lower = Gf.Vec3f(lower_x, lower_y, lower_z)

        positions, velocities = particleUtils.create_particles_grid(
            lower, self.particleSpacing, numParticlesX, numParticlesY, numParticlesZ
        )
        widths = [self.particleSpacing] * len(positions)

        positions = Vt.Vec3fArray(positions)
        velocities = Vt.Vec3fArray(velocities)
        widths = Vt.FloatArray(widths)

        if self.first_set_fluid:
            # Particle system
            self.particleSystemPath = self._defaultFluidPath.AppendChild("particleSystem0")

            self.particle_system = particleUtils.add_physx_particle_system(
                stage=self.stage,
                particle_system_path=self.particleSystemPath,
                particle_system_enabled=True,
                simulation_owner=None,
                # contact_offset=self.particleContactOffset,
                # rest_offset=self.particleContactOffset * 0.99,
                particle_contact_offset=self.particleContactOffset,
                # solid_rest_offset=self.particleContactOffset * 0.99,
                # fluid_rest_offset=self.particleContactOffset * 0.99 * 0.6,
                enable_ccd=True,
                solver_position_iterations=16,
                max_depenetration_velocity=None,
                wind=None,
                max_neighborhood=96,
                neighborhood_scale=1.01,
                max_velocity=self.cfg["fluid"].get("max_velocity", 0.8),
                global_self_collision_enabled=True,
                non_particle_collision_enabled=None,
            )
            particleUtils.add_physx_particle_isosurface(self.stage, self.particleSystemPath)

            smoothingAPI = PhysxSchema.PhysxParticleSmoothingAPI.Apply(self.particle_system.GetPrim())
            smoothingAPI.CreateParticleSmoothingEnabledAttr().Set(True)
            smoothingAPI.CreateStrengthAttr().Set(50.0)

            self.particlesPath = self._defaultFluidPath.AppendChild("particles")

            self.stage.SetInterpolationType(Usd.InterpolationTypeLinear)

            self.particles = particleUtils.add_physx_particleset_points(
                stage=self.stage,
                path=self.particlesPath,
                positions_list=positions,
                velocities_list=velocities,
                widths_list=widths,
                particle_system_path=self.particleSystemPath,
                self_collision=True,
                fluid=True,
                particle_group=0,
                particle_mass=self.cfg["fluid"].get("mass", 0.000000),
                density=self.cfg["fluid"].get("density", 0.000000),
            )

            self.particlesPbdMaterialPath = self._defaultFluidPath.AppendChild("pdbMaterial")

            self.particlesVisualMaterialPath = self._defaultFluidPath.AppendChild("visualMaterial")

            particleUtils.add_pbd_particle_material(
                stage=self.stage,
                path=self.particlesPbdMaterialPath,
                cohesion=0.01,
                drag=0,
                lift=0,
                damping=0,
                friction=0.1,
                surface_tension=0.0074,
                viscosity=0.0000017,
                vorticity_confinement=0,
            )
            particlesPbdMaterial_prim = get_prim_at_path(self.particlesPbdMaterialPath)
            material = UsdShade.Material(particlesPbdMaterial_prim)

            particleSystem_prim = get_prim_at_path(self.particleSystemPath)
            binding_api = UsdShade.MaterialBindingAPI.Apply(particleSystem_prim)
            binding_api.Bind(material)

            material = self._create_colored_material(
                self.stage,
                self.particlesVisualMaterialPath,
                color=self.cfg["fluid"].get("color", [1.0, 1.0, 1.0]),
                emissiveColor=self.cfg["fluid"].get("emissiveColor", [0.0, 0.0, 0.0]),
                opacity=self.cfg["fluid"].get("opacity", 1),
            )
            binding_api.Bind(material)

            self.first_set_fluid = False

        else:
            self.particles.GetPointsAttr().Set(positions)
            self.particles.GetVelocitiesAttr().Set(velocities)
            self.particles.GetWidthsAttr().Set(widths)

        particles_prim = self.stage.GetPrimAtPath(self.particlesPath)
        if particles_prim:
            purpose_attr = particles_prim.CreateAttribute("purpose", Sdf.ValueTypeNames.Token)
            purpose_attr.Set("proxy")

        return self.particles

    # Utilities
    def _perturb_camera(self, camera, cfg, max_translation_noise=0.05, max_orientation_noise=10.0):
        translation = np.array(cfg["translation"])
        orientation = np.array(cfg["orientation"])

        random_direction = np.random.randn(3)
        random_direction /= np.linalg.norm(random_direction)
        random_distance = np.random.uniform(0, max_translation_noise)
        perturbed_translation = translation + random_direction * random_distance

        original_rot = R.from_quat(orientation, scalar_first=True)
        random_axis = np.random.randn(3)
        random_axis /= np.linalg.norm(random_axis)
        random_angle_deg = np.random.uniform(-max_orientation_noise, max_orientation_noise)
        random_angle_rad = np.radians(random_angle_deg)
        perturbation_rot = R.from_rotvec(random_axis * random_angle_rad)
        perturbed_rot = perturbation_rot * original_rot
        perturbed_orientation = perturbed_rot.as_quat(scalar_first=True)

        camera.set_local_pose(
            translation=perturbed_translation,
            orientation=perturbed_orientation,
            camera_axes=cfg["camera_axes"],
        )

    def _create_distractor_cfg(self):
        distractors_cfg = self.cfg["distractors"]

        # Collect all available distractor asset paths
        distractor_paths = glob.glob(
            os.path.join(self.asset_root, distractors_cfg["path"], "*", "*", "*.usd")  # category  # subcategory
        )
        distractor_paths.sort()

        # Categories already used by main objects in the scene
        current_categories = {obj_cfg["path"].split("/")[-3] for obj_cfg in self.cfg["objects"]}

        # Optional: categories to be excluded from distractors via config
        # Example in config:
        # distractors:
        #   exclude_categories: ["omniobject3d-shoe", "omniobject3d-book"]
        #   exclude_keywords: ["shoe", "book"]
        excluded_categories = set(distractors_cfg.get("exclude_categories", []))
        exclude_keywords = [k.lower() for k in distractors_cfg.get("exclude_keywords", [])]

        filtered_distractors = []
        for path in distractor_paths:
            category = path.split("/")[-3]
            category_lower = category.lower()

            # Skip if category is already used by main objects
            if category in current_categories:
                continue

            # Skip if category is explicitly excluded
            if category in excluded_categories:
                continue

            # Skip if any keyword appears in the category name (case-insensitive)
            if any(kw in category_lower for kw in exclude_keywords):
                continue

            filtered_distractors.append(path)

        num_samples = random.randint(
            distractors_cfg["min_num"], min(distractors_cfg["max_num"], len(filtered_distractors))
        )
        filtered_distractors = random.sample(filtered_distractors, num_samples)

        cfgs = []
        for path in filtered_distractors:
            tmp_cfg = {}
            tmp_cfg["name"] = "distractors" + "/" + path.split("/")[-2]
            tmp_cfg["name"] = (tmp_cfg["name"]).replace("-", "_")
            tmp_cfg["path"] = path.replace(self.asset_root, "")
            tmp_cfg["target_class"] = distractors_cfg.get("target_class", "RigidObject")  # "RigidObject"
            tmp_cfg["prim_path_child"] = distractors_cfg.get("prim_path_child", "Aligned")  # "Aligned"
            tmp_cfg["translation"] = distractors_cfg.get("translation", [0.0, 0.0, 0.0])
            tmp_cfg["scale"] = distractors_cfg.get("scale", [1.0, 1.0, 1.0])
            tmp_category = path.split("/")[-3]
            tmp_cfg["category"] = tmp_category
            tmp_cfg = DictConfig(tmp_cfg)
            cfgs.append(tmp_cfg)

        return cfgs

    def _rebuild_distractors(self, cfgs, use_mem: bool):
        """Rebuild distractor objects from a list of configs.

        - If use_mem is True, only rebuild objects (no new random placement via set_distractors).
        - If use_mem is False, also call set_distractors to (re)sample placements.
        """
        for cfg in cfgs:
            if cfg["target_class"] == "RigidObject":
                self.distractors[cfg["name"]] = self._load_obj(cfg)
            else:
                raise NotImplementedError

        if (not use_mem) and self.cfg.get("distractors", None):
            set_distractors(
                self.objects,
                self.distractors,
                self._task_objects[self.cfg["distractors"]["target"]],
                self.cfg["distractors"],
                cfgs,
            )

    def _get_container_center(self):
        container_name = self.cfg["fluid"]["container_name"]
        container = self.objects[container_name]
        container_trans, _ = container.get_world_pose()

        return container_trans

    def _create_colored_material(
        self, stage, material_path, color=(1.0, 0.0, 0.0), emissiveColor=(0.0, 0.0, 0.0), opacity=1.0
    ):
        material_prim = stage.DefinePrim(material_path, "Material")
        material = UsdShade.Material(material_prim)

        shader_path = f"{material_path}/PreviewSurface"
        shader_prim = stage.DefinePrim(shader_path, "Shader")
        shader = UsdShade.Shader(shader_prim)

        shader.CreateIdAttr("UsdPreviewSurface")

        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissiveColor))

        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)

        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        return material
