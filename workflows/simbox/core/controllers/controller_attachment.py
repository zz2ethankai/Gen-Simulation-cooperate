"""Attachment lifecycle for the controller-owned planner runtime."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
from core.planning.domain_types import AttachmentSpec
from core.controllers.controller_component import ComponentState

LOGGER = logging.getLogger("de_logger")


def _record_attachment_rollback_failure(primary_error: Exception, operation: str, rollback_error: Exception) -> None:
    failures = list(getattr(primary_error, "_attachment_rollback_failures", ()))
    failures.append((operation, rollback_error))
    try:
        primary_error._attachment_rollback_failures = tuple(failures)
    except Exception:
        pass
    add_note = getattr(primary_error, "add_note", None)
    if callable(add_note):
        try:
            add_note(
                "Attachment rollback failed during "
                f"{operation}: {type(rollback_error).__name__}: {rollback_error}"
            )
        except Exception:
            pass


class ControllerAttachment(ComponentState):
    def _attach_native_planner(
        self,
        planner,
        object_names: List[str],
        *,
        link_name="attached_object",
        joint_state=None,
        world_objects_pose_offset=None,
        scene_adapter=None,
    ):
        """Attach one resolved object set to a native planner.

        Pick execution, Place candidate evaluation, and transient post-grasp
        validation all use the same CuRobo attachment contract.  Keeping the
        validation, mesh fitting, sphere count, and named-joint conversion in
        one helper prevents those paths from drifting apart.
        """

        if not object_names or any(
            not isinstance(path, str) or not path.strip() for path in object_names
        ):
            raise ValueError("native attachment requires non-empty obstacle path strings")
        paths = [path.strip() for path in object_names]
        if planner is None:
            raise RuntimeError("native attachment requires an initialized planner")
        # Presence is checked against the exact target planner adapter.  The
        # logical SceneCfg remains useful to setup code, but attachment must
        # never infer native addressability from that snapshot or synthesize
        # an unregistered adapter here.
        if scene_adapter is None or not callable(
            getattr(scene_adapter, "require_obstacles", None)
        ):
            raise RuntimeError(
                "native attachment requires a registered target scene adapter"
            )
        scene_adapter.require_obstacles(paths)
        if joint_state is None:
            joint_state = self._arm_joint_state(self.robot.get_joints_state())
        attachment_meshes, attachment_offset = self.runtime._attachment_geometry(paths)
        if world_objects_pose_offset is not None:
            attachment_offset = world_objects_pose_offset
        from curobo.sphere_fit import SphereFitType

        planner.attachment_manager.attach(
            joint_state,
            attachment_meshes,
            link_name=link_name,
            num_spheres=max(
                1,
                self.runtime._attached_sphere_count(link_name, 1, planner=planner),
            ),
            surface_radius=0.001,
            sphere_fit_type=SphereFitType.VOXEL,
            world_objects_pose_offset=attachment_offset,
            disable_obstacle_names=paths,
        )
        return paths

    def attach_collision_object(
        self,
        obj_prim_paths: List[str],
        link_name="attached_object",
        world_objects_pose_offset=None,
    ):
        paths = [str(path).strip() for path in obj_prim_paths]
        if not paths or any(not path for path in paths):
            raise ValueError("attachment requires non-empty exact collider paths")
        meshes, attachment_offset = self.runtime._attachment_geometry(paths)
        if world_objects_pose_offset is not None:
            attachment_offset = world_objects_pose_offset
        joint_state = self._arm_joint_state(self.robot.get_joints_state())
        spec = AttachmentSpec(
            name="|".join(paths),
            state=joint_state,
            meshes=meshes,
            link_name=link_name,
            pose_offset=attachment_offset,
            disable_obstacle_names=tuple(paths),
        )
        self.runtime.attach_object(spec)
        if self.runtime.batch_attachment_runtime is not None:
            self.runtime.batch_attachment_runtime.detach()
        return True

    def sync_native_batch_attachment(
        self,
        link_name="attached_object",
        world_objects_pose_offset=None,
    ) -> bool:
        """Attach the currently held object to the native batch planner.

        The batch planner is used only to rank Place candidates.  It must see
        the same attached geometry as the execution planner, but it must not
        be attached during Pick: two CUDA-backed AttachmentManagers share no
        solver state contract and the second attach can invalidate the live
        single-planner start-state query.  This method is therefore an
        explicit Place-side synchronization point.
        """

        if not self.batch_capability:
            return False
        batch_planner = self._ensure_batch_planner()
        require_adapter = getattr(self, "_require_batch_scene_adapter", None)
        if not callable(require_adapter):
            raise RuntimeError(
                "native batch attachment requires the collision-scene "
                "manager's strict target-batch adapter"
            )
        batch_adapter = require_adapter()
        if batch_adapter is None or not callable(
            getattr(batch_adapter, "require_obstacles", None)
        ):
            raise RuntimeError(
                "collision-scene manager did not provide a strict target-batch "
                "scene adapter"
            )
        paths = list(self.runtime.attachment_runtime.attached_obstacle_names)
        if not paths:
            raise RuntimeError(
                "cannot synchronize native batch attachment without an attached object"
            )
        batch_adapter.require_obstacles(paths)
        batch_attachment_runtime = self.runtime.batch_attachment_runtime
        if batch_attachment_runtime is None:
            raise RuntimeError("batch attachment runtime was not initialized")
        if list(batch_attachment_runtime.attached_obstacle_names) == paths:
            return True

        # AttachmentRuntime owns the candidate planner's logical state and
        # performs detach/reattach rollback around the native operation.
        self._attach_batch_runtime_spec(
            paths,
            link_name=link_name,
            world_objects_pose_offset=world_objects_pose_offset,
        )
        return True

    def _attach_batch_runtime_spec(
        self, paths, *, link_name="attached_object", world_objects_pose_offset=None
    ):
        """Record batch attachment ownership in the shared AttachmentRuntime."""

        if not paths:
            return
        sim_js = self.robot.get_joints_state()
        cu_js = self._arm_joint_state(sim_js)
        meshes, attachment_offset = self.runtime._attachment_geometry(paths)
        if world_objects_pose_offset is not None:
            attachment_offset = world_objects_pose_offset
        self.runtime.batch_attachment_runtime.attach(
            AttachmentSpec(
                name="|".join(paths),
                state=cu_js,
                meshes=meshes,
                link_name=link_name,
                pose_offset=attachment_offset,
                disable_obstacle_names=tuple(paths),
            )
        )

    def _ensure_batch_planner(self):
        ensure = getattr(self.runtime, "ensure_batch_planner", None)
        if not callable(ensure):
            raise RuntimeError("batch planning was not enabled for this controller")
        return ensure()

    def plan_attached_pose_from_joint_positions(
        self,
        ee_trans: np.ndarray,
        ee_ori: np.ndarray,
        start_arm_positions: np.ndarray,
        obj_prim_paths: List[str],
    ):
        """Plan a post-grasp target with the object attached at the grasp endpoint.

        Candidate validation must use the same attached collision geometry as
        execution.  The attachment is deliberately transient and is always
        removed before returning to the caller.
        """
        if not obj_prim_paths:
            raise ValueError("post-grasp validation requires attach collision prim paths")
        start_arm_positions = np.asarray(start_arm_positions, dtype=float).reshape(-1)
        if len(start_arm_positions) != len(self.arm_indices):
            raise ValueError(
                "start_arm_positions must match the controller arm joint count: "
                f"got {len(start_arm_positions)}, expected {len(self.arm_indices)}"
            )

        sim_js = self.robot.get_joints_state()
        sim_js.positions = np.asarray(sim_js.positions, dtype=float).copy()
        sim_js.positions[self.arm_indices] = start_arm_positions
        cu_js = self._arm_joint_state(sim_js)

        paths = [str(path).strip() for path in obj_prim_paths]
        meshes, attachment_offset = self.runtime._attachment_geometry(paths)
        spec = AttachmentSpec(
            name="|".join(paths),
            state=cu_js,
            meshes=meshes,
            link_name="attached_object",
            pose_offset=attachment_offset,
            disable_obstacle_names=tuple(paths),
        )
        self.runtime.attachment_runtime.attach(spec)
        try:
            success, end_positions, result = self._plan_pose_from_joint_positions(
                ee_trans,
                ee_ori,
                start_arm_positions=start_arm_positions,
            )
            self._log_plan_result("attached_forward", result, target=ee_trans)
            return success, end_positions, result
        finally:
            self.runtime.detach_object()

    def detach_attachment(self):
        """Detach the active object through the typed attachment runtime."""

        self.runtime.detach_object()
        if self.runtime.batch_attachment_runtime is not None:
            self.runtime.batch_attachment_runtime.detach()

    def has_attached_collision_spheres(self, link_name="attached_object") -> bool:
        spheres = (
            self.runtime.native_planner.kinematics.config.kinematics_config.get_link_spheres(link_name)
        )
        return bool(torch.any(spheres[:, 3] > 0.0).item())

    def _attached_sphere_count(
        self, link_name: str, object_count: int, *, planner=None
    ) -> int:
        planner = planner or self.runtime.native_planner
        total = planner.kinematics.config.kinematics_config.get_number_of_spheres(
            link_name
        )
        return max(1, int(total) // max(1, int(object_count)))
