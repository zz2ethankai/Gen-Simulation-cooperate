"""Pick phase construction and mobile-reference coordination."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
from isaacsim.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.xforms import get_world_pose
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.controllers.curobo.components import ComponentState

LOGGER = logging.getLogger("de_logger")


class ControllerPhases(ComponentState):
    def forward(self, command, eps=5e-3):
        """Execute one typed motion command.

        ``forward`` remains the simulator-facing spelling for structured
        commands.  Skills that own direct joint interpolation use the
        controller's explicit ``dummy_forward`` interface instead of this
        phase builder.  ``eps`` is retained as a harmless call-site
        compatibility parameter for workflow integrations that pass it
        positionally.
        """

        del eps
        return self.execute(command)

    def _begin_phase_command(self, command: MotionPhaseCommand) -> bool:
        if command is self._active_phase_command:
            return False
        self._active_phase_command = command
        self._phase_plan_started = False
        self._phase_plan_finished = False
        self._phase_bookkeeping_done = False
        self._phase_dwell_count = 0
        self._phase_tracking_failed = False
        self._phase_plan_failed = False
        self._phase_completion_logged = False
        self._last_command_name = command.phase.value
        self._phase_base_position, self._phase_base_orientation = self.get_armbase_pose()
        LOGGER.info(
            "[PhaseDebug] start robot=%s arm=%s phase=%s object=%s support=%s target=%s",
            self.name,
            self.lr_name,
            command.phase.value,
            command.active_object,
            command.support_object,
            None
            if command.target_position is None
            else np.asarray(command.target_position, dtype=float).round(5).tolist(),
        )
        return True

    def _install_preplanned_phase_path(self, command: MotionPhaseCommand):
        """Install a named native-v2 path for a phase without replanning it.

        Pick and Place candidate validation may run against a planner that has
        already solved the exact attached-object world.  Re-running the
        single planner at execution time can therefore reject a path that was
        just validated (and can also choose a different collision branch).
        Keep the same explicit named-joint normalization as the ordinary
        planning path, then let ``ee_forward`` consume the installed path by
        setting its phase target to the command target.
        """

        preplanned_path = command.params.get("preplanned_joint_path")
        if preplanned_path is None:
            return False
        trajectory = self._command_path(preplanned_path)
        self._install_command_plan(
            trajectory,
            target_position=command.target_position,
            target_orientation=command.target_orientation,
            phase_name=command.phase.value,
            cached=True,
        )
        return True

    @staticmethod
    def _pick_terminal_samples(start, goal, step_m: float) -> list[np.ndarray]:
        """Discretize a terminal grasp approach for controllers without a cached path."""

        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        distance = float(np.linalg.norm(goal - start))
        count = max(1, int(np.ceil(distance / float(step_m))))
        return [start + (goal - start) * (index / count) for index in range(1, count + 1)]

    def build_pick_phase_commands(
        self,
        *,
        object_name: str,
        pregrasp_position,
        pregrasp_orientation,
        grasp_position,
        grasp_orientation,
        gripper_action: str,
        post_grasp_offset: float = 0.0,
        source_support=None,
        pregrasp_path=None,
        terminal_path=None,
        terminal_path_length_ratio=None,
        terminal_path_max_deviation_m=None,
        return_to_pregrasp: bool = False,
        completion_tolerance: Optional[dict] = None,
        terminal_step_m: Optional[float] = None,
        gripper_change_steps: int = 40,
        contact_threshold_n: float = 0.0,
        verify_grasp_contact=None,
    ) -> list[MotionPhaseCommand]:
        """Build the executable Physics-schema Pick sequence.

        Pick owns grasp annotation sampling and candidate selection.  Once a
        candidate is selected, the controller owns the execution protocol:
        world synchronization, pre-grasp motion, terminal approach, gripper
        close, attachment, and post-grasp lift.  Keeping this policy here
        makes the skill independent of the low-level Physics phase details
        while retaining the structured command interface used by the
        workflow.
        """

        if verify_grasp_contact is None or not callable(verify_grasp_contact):
            raise ValueError("Physics Pick execution requires verify_grasp_contact callback")
        if terminal_step_m is None:
            terminal_step_m = float(
                self.task.cfg.get("planning", {})
                .get("pick_place", {})
                .get("terminal_step_m", 0.005)
            )
        # ``terminal_step_m`` controls terminal-path waypoint sampling only.
        # The phase completion threshold comes from the skill's
        # ``completion_tolerance`` (for Pick, this is ``t_eps``).
        terminal_step_m = max(float(terminal_step_m), 1e-6)
        tolerance = dict(
            completion_tolerance
            or {
                "position_m": 0.005,
                "orientation_rad": 0.05,
            }
        )
        if source_support is None and self.collision_scene_manager is not None:
            source_support = self.collision_scene_manager.get_source_support_entity(
                object_name
            )

        pregrasp_position = np.asarray(pregrasp_position, dtype=float)
        pregrasp_orientation = np.asarray(pregrasp_orientation, dtype=float)
        grasp_position = np.asarray(grasp_position, dtype=float)
        grasp_orientation = np.asarray(grasp_orientation, dtype=float)
        commands = [
            MotionPhaseCommand(
                MotionPhase.SYNC_WORLD,
                active_object=object_name,
                replan_allowed=False,
            ),
            MotionPhaseCommand(
                MotionPhase.TRANSIT_PREGRASP,
                pregrasp_position,
                pregrasp_orientation,
                gripper_action="open_gripper",
                active_object=object_name,
                completion_tolerance=tolerance,
                params=(
                    {"preplanned_joint_path": pregrasp_path}
                    if pregrasp_path is not None
                    else {}
                ),
            ),
        ]

        if terminal_path is not None:
            commands.append(
                MotionPhaseCommand(
                    MotionPhase.TERMINAL_GRASP_APPROACH,
                    grasp_position,
                    grasp_orientation,
                    gripper_action="open_gripper",
                    active_object=object_name,
                    allow_target_finger_contact=True,
                    completion_tolerance={
                        "position_m": tolerance["position_m"],
                        "orientation_rad": tolerance["orientation_rad"],
                    },
                    params={
                        "preplanned_joint_path": terminal_path,
                        "cartesian_step_m": terminal_step_m,
                        "path_length_ratio": terminal_path_length_ratio,
                        "path_max_deviation_m": terminal_path_max_deviation_m,
                    },
                )
            )
        else:
            # Lightweight controllers may not return a chained pre-grasp ->
            # grasp path.  Keep their fallback behavior in the controller so
            # the skill does not need to know how terminal motion is sampled.
            terminal_points = self._pick_terminal_samples(
                pregrasp_position, grasp_position, terminal_step_m
            )
            for point_index, point in enumerate(terminal_points):
                ratio = (point_index + 1) / len(terminal_points)
                quat = (1.0 - ratio) * pregrasp_orientation + ratio * grasp_orientation
                quat = quat / np.linalg.norm(quat)
                commands.append(
                    MotionPhaseCommand(
                        MotionPhase.TERMINAL_GRASP_APPROACH,
                        point,
                        quat,
                        gripper_action="open_gripper",
                        active_object=object_name,
                        allow_target_finger_contact=True,
                        completion_tolerance={
                            "position_m": tolerance["position_m"],
                            "orientation_rad": tolerance["orientation_rad"],
                        },
                    )
                )

        commands.extend(
            [
                MotionPhaseCommand(
                    MotionPhase.GRIPPER_CLOSE,
                    grasp_position,
                    grasp_orientation,
                    gripper_action=gripper_action,
                    active_object=object_name,
                    allow_target_finger_contact=True,
                    replan_allowed=False,
                    dwell_steps=int(gripper_change_steps),
                    params={"contact_threshold_n": float(contact_threshold_n)},
                ),
                MotionPhaseCommand(
                    MotionPhase.ATTACH,
                    active_object=object_name,
                    allow_target_finger_contact=True,
                    replan_allowed=False,
                    params={"verify_grasp_contact": verify_grasp_contact},
                ),
            ]
        )

        def append_lift(target_position, target_orientation):
            commands.append(
                MotionPhaseCommand(
                    MotionPhase.POST_GRASP_LIFT,
                    target_position,
                    target_orientation,
                    gripper_action=gripper_action,
                    active_object=object_name,
                    support_object=source_support,
                    allow_target_finger_contact=True,
                    # Native attachment leaves the physical target in the
                    # gripper; permit the attached target to touch robot links
                    # while the explicit target collision gate remains active.
                    allow_target_robot_contact=True,
                    allow_object_support_contact=source_support is not None,
                    completion_tolerance=tolerance,
                )
            )

        if post_grasp_offset:
            post_position = grasp_position.copy()
            post_position[2] += float(post_grasp_offset)
            append_lift(post_position, grasp_orientation)
        if return_to_pregrasp:
            append_lift(pregrasp_position, pregrasp_orientation)
        return commands

    def get_pick_armbase_transform(self):
        """Return the current world transform of this controller's arm base."""

        armbase_tf_getter = getattr(self.robot, "get_armbase_world_transform", None)
        if callable(armbase_tf_getter):
            return armbase_tf_getter()

        reference_prim_path = str(getattr(self, "reference_prim_path", "")).strip()
        if reference_prim_path:
            reference_prim = get_prim_at_path(reference_prim_path)
            if reference_prim.IsValid():
                try:
                    reference_t, reference_q = get_world_pose(reference_prim_path)
                    world_armbase = tf_matrix_from_pose(reference_t, reference_q)
                    if hasattr(self.robot, "get_mobile_base_pose"):
                        try:
                            mobile_base_t, mobile_base_q = self.robot.get_mobile_base_pose()
                            world_mobile = tf_matrix_from_pose(mobile_base_t, mobile_base_q)
                            self._pick_cached_mobile_to_armbase_tf = (
                                np.linalg.inv(world_mobile) @ world_armbase
                            )
                        except Exception:
                            pass
                    return world_armbase
                except Exception:
                    pass

        if self._pick_configured_mobile_to_armbase_translation.shape == (3,):
            if hasattr(self.robot, "get_mobile_base_pose"):
                mobile_base_t, mobile_base_q = self.robot.get_mobile_base_pose()
            else:
                mobile_base_t, mobile_base_q = self.robot.get_world_pose()
            world_mobile = tf_matrix_from_pose(mobile_base_t, mobile_base_q)
            mobile_to_armbase = tf_matrix_from_pose(
                self._pick_configured_mobile_to_armbase_translation,
                self._pick_configured_mobile_to_armbase_orientation,
            )
            self._pick_cached_mobile_to_armbase_tf = mobile_to_armbase
            return world_mobile @ mobile_to_armbase

        reference_prim = get_prim_at_path(self.reference_prim_path)
        task_prim = get_prim_at_path(self.task.root_prim_path)
        raw_task_armbase = get_relative_transform(reference_prim, task_prim)
        mobile_base_prim_path = str(self._pick_mobile_base_prim_path or "").strip()
        if not mobile_base_prim_path:
            return raw_task_armbase

        mobile_base_prim = get_prim_at_path(mobile_base_prim_path)
        if not mobile_base_prim.IsValid():
            return raw_task_armbase
        task_mobile = get_relative_transform(mobile_base_prim, task_prim)
        if self._pick_cached_mobile_to_armbase_tf is None:
            self._pick_cached_mobile_to_armbase_tf = (
                np.linalg.inv(task_mobile) @ raw_task_armbase
            )
        return task_mobile @ self._pick_cached_mobile_to_armbase_tf

    def get_pick_frame_debug(self):
        """Expose frame-resolution details for Pick's non-critical diagnostics."""

        return {
            "mobile_base_prim_path": self._pick_mobile_base_prim_path,
            "cached_mobile_to_armbase_tf": self._pick_cached_mobile_to_armbase_tf,
            "configured_mobile_to_armbase_translation": self._pick_configured_mobile_to_armbase_translation,
            "configured_mobile_to_armbase_orientation": self._pick_configured_mobile_to_armbase_orientation,
        }

    def _get_pick_object_world_pose(self, object_name: str):
        pick_object = self.task.objects[object_name]
        get_world_pose_fn = getattr(pick_object, "get_world_pose", None)
        if callable(get_world_pose_fn):
            return get_world_pose_fn()
        return pick_object.get_local_pose()

    def capture_pick_plan_reference(self, object_name: str):
        """Capture the object and arm-base frames used by a Physics Pick plan."""

        object_translation, object_orientation = self._get_pick_object_world_pose(object_name)
        self._pick_plan_references[object_name] = {
            "object_pose": (
                np.asarray(object_translation, dtype=float).reshape(3).copy(),
                np.asarray(object_orientation, dtype=float).reshape(4).copy(),
            ),
            "world_armbase_tf": np.asarray(self.get_pick_armbase_transform(), dtype=float).copy(),
        }

    def retarget_pick_phase_commands(self, object_name: str, commands):
        """Retarget pending Physics Pick phases after rigid target motion.

        A command sequence is also retargeted once immediately after candidate
        evaluation.  Preserve a validated terminal path for that no-motion
        case; invalidate it only when the object or the arm base actually
        changed pose.
        """

        reference = self._pick_plan_references.get(object_name)
        if reference is None:
            self.capture_pick_plan_reference(object_name)
            return 0.0, 0.0

        current_translation, current_orientation = self._get_pick_object_world_pose(object_name)
        current_object_pose = (
            np.asarray(current_translation, dtype=float).reshape(3).copy(),
            np.asarray(current_orientation, dtype=float).reshape(4).copy(),
        )
        current_world_armbase_tf = np.asarray(self.get_pick_armbase_transform(), dtype=float)
        old_object_tf = tf_matrix_from_pose(*reference["object_pose"])
        current_object_tf = tf_matrix_from_pose(*current_object_pose)
        object_delta = current_object_tf @ np.linalg.inv(old_object_tf)
        relative_rotation = object_delta[:3, :3]
        cosine = float(
            np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
        )
        rotation_delta_deg = float(np.degrees(np.arccos(cosine)))
        # ``object_delta`` is a rigid transform about the world origin.  Its
        # translation component includes the lever-arm term introduced by a
        # target rotation (for an object 0.2 m from the origin, a 9 degree
        # rotation alone looks like roughly 3 cm of translation).  Safety
        # retargeting must compare the two object-frame origins directly; the
        # rigid transform above is still the correct transform for retargeting
        # the object-relative EE targets.
        translation_delta = float(
            np.linalg.norm(current_object_pose[0] - reference["object_pose"][0])
        )
        reference_world_armbase_tf = np.asarray(
            reference["world_armbase_tf"], dtype=float
        )
        base_pose_changed = not np.allclose(
            current_world_armbase_tf,
            reference_world_armbase_tf,
            atol=1e-6,
            rtol=0.0,
        )
        target_pose_changed = (
            translation_delta > 1e-6 or rotation_delta_deg > 1e-4 or base_pose_changed
        )

        current_base_inverse = np.linalg.inv(current_world_armbase_tf)
        for pending in commands:
            if not isinstance(pending, MotionPhaseCommand) or pending.target_position is None:
                continue
            old_base_ee_tf = tf_matrix_from_pose(
                pending.target_position, pending.target_orientation
            )
            old_world_ee_tf = reference["world_armbase_tf"] @ old_base_ee_tf
            current_world_ee_tf = object_delta @ old_world_ee_tf
            current_base_ee_tf = current_base_inverse @ current_world_ee_tf
            target_position, target_orientation = pose_from_tf_matrix(current_base_ee_tf)
            pending.target_position = np.asarray(target_position, dtype=float).reshape(3)
            pending.target_orientation = np.asarray(target_orientation, dtype=float).reshape(4)
            if target_pose_changed and pending.phase in {
                MotionPhase.TRANSIT_PREGRASP,
                MotionPhase.TERMINAL_GRASP_APPROACH,
            }:
                pending.params.pop("preplanned_joint_path", None)
                pending.params.pop("path_length_ratio", None)
                pending.params.pop("path_max_deviation_m", None)

        self._pick_plan_references[object_name] = {
            "object_pose": current_object_pose,
            "world_armbase_tf": current_world_armbase_tf.copy(),
        }
        return translation_delta, rotation_delta_deg

    def replan_pick_after_safety(self, object_name: str, command, commands):
        """Retarget remaining Pick phases when the active object moved."""

        if not isinstance(command, MotionPhaseCommand):
            return True
        if command.active_object != object_name:
            return True
        if command.phase not in {
            MotionPhase.TRANSIT_PREGRASP,
            MotionPhase.TERMINAL_GRASP_APPROACH,
        }:
            return True
        try:
            translation_delta, rotation_delta_deg = self.retarget_pick_phase_commands(
                object_name, commands
            )
        except Exception:
            LOGGER.exception(
                "[PickSafety] failed to retarget moving object=%s phase=%s",
                object_name,
                command.phase.value,
            )
            return False
        if translation_delta > 1e-6 or rotation_delta_deg > 1e-4:
            LOGGER.warning(
                "[PickSafety] retargeted active object=%s phase=%s "
                "translation_delta_m=%.6f rotation_delta_deg=%.3f "
                "cached_pick_paths_invalidated=true",
                object_name,
                command.phase.value,
                translation_delta,
                rotation_delta_deg,
            )
        return True
