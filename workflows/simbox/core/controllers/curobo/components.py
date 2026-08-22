"""Explicit state injection helpers for controller operation components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class MutableExecutionState:
    """Single mutable owner for phase lifecycle and command bookkeeping."""

    active_phase_command: Any = None
    last_command_name: str = "unknown"
    phase_base_position: Any = None
    phase_base_orientation: Any = None
    phase_bookkeeping_done: bool = False
    phase_dwell_count: int = 0
    phase_plan_started: bool = False
    phase_plan_finished: bool = False
    phase_tracking_failed: bool = False
    phase_plan_failed: bool = False
    phase_completion_logged: bool = False
    step_idx: int = 0
    num_last_cmd: int = 0
    num_plan_failed: int = 0
    last_arm_action: Any = None
    last_commanded_arm_position: Any = None
    idx_list: Any = None
    ee_trans: Any = 0.0
    ee_ori: Any = 0.0
    gripper_state: float = 1.0
    gripper_joint_position: Any = None

    def reset(self) -> None:
        """Reset phase/trajectory state while preserving robot configuration."""

        self.active_phase_command = None
        self.last_command_name = "unknown"
        self.phase_base_position = None
        self.phase_base_orientation = None
        self.phase_bookkeeping_done = False
        self.phase_dwell_count = 0
        self.phase_plan_started = False
        self.phase_plan_finished = False
        self.phase_tracking_failed = False
        self.phase_plan_failed = False
        self.phase_completion_logged = False
        self.step_idx = 0
        self.num_last_cmd = 0
        self.num_plan_failed = 0
        self.last_arm_action = None
        self.last_commanded_arm_position = None
        self.idx_list = None


@dataclass
class PlanningConfig:
    """Setup-time planning configuration shared by planning components."""

    ds_ratio: int = 1


def _state_property(field: str):
    """Build an explicit component property backed by shared state."""

    def getter(component):
        return getattr(component.execution_state, field)

    def setter(component, value):
        setattr(component.execution_state, field, value)

    return property(getter, setter)


class ComponentPort:
    """Immutable-by-convention collection of dependencies for one component.

    The port is deliberately a plain value container.  It never keeps the
    façade that created it and has no attribute fallback or write-through
    behavior.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = dict(values)

    def items(self):
        return self._values.items()


class ComponentState:
    """Base for operation components with explicit, shared dependencies."""

    def __init__(self, port: ComponentPort) -> None:
        if not isinstance(port, ComponentPort):
            raise TypeError("components require an explicit ComponentPort")
        values = dict(port.items())
        execution_state = values.pop("execution_state", None)
        if execution_state is None:
            execution_state = MutableExecutionState()
        if not isinstance(execution_state, MutableExecutionState):
            raise TypeError("execution_state must be a MutableExecutionState")
        planning_config = values.pop("planning_config", None)
        if planning_config is None:
            planning_config = PlanningConfig()
        if not isinstance(planning_config, PlanningConfig):
            raise TypeError("planning_config must be a PlanningConfig")
        object.__setattr__(self, "execution_state", execution_state)
        object.__setattr__(self, "planning_config", planning_config)
        for name, value in values.items():
            setattr(self, name, value)

    _active_phase_command = _state_property("active_phase_command")
    _last_command_name = _state_property("last_command_name")
    _phase_base_position = _state_property("phase_base_position")
    _phase_base_orientation = _state_property("phase_base_orientation")
    _phase_bookkeeping_done = _state_property("phase_bookkeeping_done")
    _phase_dwell_count = _state_property("phase_dwell_count")
    _phase_plan_started = _state_property("phase_plan_started")
    _phase_plan_finished = _state_property("phase_plan_finished")
    _phase_tracking_failed = _state_property("phase_tracking_failed")
    _phase_plan_failed = _state_property("phase_plan_failed")
    _phase_completion_logged = _state_property("phase_completion_logged")
    _step_idx = _state_property("step_idx")
    num_last_cmd = _state_property("num_last_cmd")
    num_plan_failed = _state_property("num_plan_failed")
    _last_arm_action = _state_property("last_arm_action")
    _last_commanded_arm_position = _state_property("last_commanded_arm_position")
    idx_list = _state_property("idx_list")
    _ee_trans = _state_property("ee_trans")
    _ee_ori = _state_property("ee_ori")
    _gripper_state = _state_property("gripper_state")
    _gripper_joint_position = _state_property("gripper_joint_position")

    @property
    def ds_ratio(self) -> int:
        return int(self.planning_config.ds_ratio)

    @ds_ratio.setter
    def ds_ratio(self, value: int) -> None:
        self.planning_config.ds_ratio = max(1, int(value))


class SetupPort(ComponentPort):
    """Dependencies for scene/setup operations."""


class StatePlanningPort(ComponentPort):
    """Dependencies for typed state conversion and planning."""


class PlanningQueriesPort(ComponentPort):
    """Dependencies for candidate/query operations."""


class PhasesPort(ComponentPort):
    """Dependencies for phase-command construction."""


class AttachmentPort(ComponentPort):
    """Dependencies for attachment lifecycle operations."""


@dataclass(frozen=True)
class ControllerComponents:
    """The operation components assembled for one controller instance.

    This is an assembly result, not a controller context.  Components receive
    only the explicit ports produced by ``port_factory`` and the returned
    bundle contains no reference to the façade that created them.
    """

    setup: Any
    state_planning: Any
    planning_queries: Any
    phases: Any
    execution: Any
    attachment: Any


def wire_controller_components(port_factory, *, prepare_setup=None) -> ControllerComponents:
    """Construct and cross-wire operation components from explicit ports.

    ``port_factory`` is used only during construction to materialize the
    named dependencies for each component.  Keeping this repetitive wiring
    here leaves ``TemplateController`` focused on lifecycle orchestration.
    """

    from core.controllers.curobo.attachments import ControllerAttachment
    from core.controllers.curobo.execution import ControllerExecution
    from core.controllers.curobo.motion_phases import ControllerPhases
    from core.controllers.curobo.planning_queries import ControllerPlanningQueries
    from core.controllers.curobo.scene_setup import ControllerSetup
    from core.controllers.curobo.state_planning import ControllerStatePlanning

    setup = ControllerSetup(
        port_factory(
            (
                "name", "world", "task", "robot", "arm_spec", "robot_file",
                "trajectory_visualizer", "timing_recorder", "collision_scene_manager",
                "tensor_args", "batch_capability", "batch_enabled",
                "constrain_grasp_approach", "collision_activation_distance",
                "raw_js_names", "cmd_js_names", "arm_indices", "gripper_indices",
                "reference_prim_path", "lr_name", "planning_config", "execution_state",
                "_world_cache_invalidated", "_world_cleanup_failed", "_world_cleanup_error",
                "_world_update_signature", "_pending_pose_criteria", "phase_executor",
                "_curobo_plan_debug_counter", "_curobo_plan_debug_dir", "runtime",
            ),
            _joint_state_derivatives=ControllerStatePlanning._joint_state_derivatives,
            port_type=SetupPort,
        )
    )

    # Setup owns arm selection, runtime joint resolution, and robot frame
    # paths.  Resolve those values before any sibling component receives its
    # narrow port; otherwise copied ``lr_name``/indices/path fields remain
    # the controller's construction-time placeholders.
    if prepare_setup is not None:
        prepare_setup(setup)

    def component_port(names, *, port_type, **overrides):
        setup_values = {
            name: getattr(setup, name)
            for name in names
            if hasattr(setup, name)
        }
        setup_values.update(overrides)
        return port_factory(names, port_type=port_type, **setup_values)

    state_planning = ControllerStatePlanning(
        component_port(
            (
                "name", "robot", "arm_spec", "tensor_args", "raw_js_names", "cmd_js_names",
                "arm_indices", "gripper_indices", "lr_name", "phase_executor", "runtime",
                "planning_config", "execution_state",
            ),
            _log_plan_result=setup._log_plan_result,
            _visualize_selected_plan=setup._visualize_selected_plan,
            _refresh_reference_world_for_planning=setup._refresh_reference_world_for_planning,
            port_type=StatePlanningPort,
        )
    )
    planning_queries = ControllerPlanningQueries(
        component_port(
            (
                "robot", "tensor_args", "raw_js_names", "arm_indices", "runtime",
                "phase_executor", "lr_name", "planning_config", "execution_state",
            ),
            _arm_joint_state=state_planning._arm_joint_state,
            _planner_joint_names=state_planning._planner_joint_names,
            _planner_state=state_planning._planner_state,
            _result_success=state_planning._result_success,
            _result_path=state_planning._result_path,
            _command_path=state_planning._command_path,
            _plan_pose_from_state=state_planning._plan_pose_from_state,
            _plan_batch_from_state=state_planning._plan_batch_from_state,
            _native_plan_pose=state_planning._native_plan_pose,
            _native_plan_pose_batch=state_planning._native_plan_pose_batch,
            _log_plan_result=setup._log_plan_result,
            _refresh_reference_world_for_planning=setup._refresh_reference_world_for_planning,
            port_type=PlanningQueriesPort,
        )
    )
    phases = ControllerPhases(
        component_port(
            (
                "name", "lr_name", "robot", "task", "reference_prim_path",
                "collision_scene_manager", "execution_state", "planning_config",
                "_pick_mobile_base_prim_path", "_pick_cached_mobile_to_armbase_tf",
                "_pick_configured_mobile_to_armbase_translation",
                "_pick_configured_mobile_to_armbase_orientation", "_pick_plan_references",
                "phase_executor", "runtime",
            ),
            _command_path=state_planning._command_path,
            _install_command_plan=state_planning._install_command_plan,
            port_type=PhasesPort,
        )
    )
    execution = ControllerExecution(
        component_port(
            (
                "name", "lr_name", "robot", "arm_spec", "tensor_args",
                "arm_indices", "gripper_indices", "phase_executor", "runtime",
                "collision_scene_manager",
                "robot_base_path", "robot_ee_path", "task_root_prim_path",
                "planning_config", "execution_state", "batch_capability", "batch_enabled",
            ),
            _begin_phase_command=phases._begin_phase_command,
            _install_preplanned_phase_path=phases._install_preplanned_phase_path,
            _install_command_plan=state_planning._install_command_plan,
            _arm_joint_state=state_planning._arm_joint_state,
            _planner_state=state_planning._planner_state,
            _planner_joint_names=state_planning._planner_joint_names,
            _command_path=state_planning._command_path,
            _result_path=state_planning._result_path,
            _result_success=state_planning._result_success,
            _native_plan_pose=state_planning._native_plan_pose,
            _log_plan_result=setup._log_plan_result,
            _write_curobo_plan_debug=setup._write_curobo_plan_debug,
            port_type=ComponentPort,
        )
    )
    attachment = ControllerAttachment(
        component_port(
            (
                "robot", "runtime", "arm_indices", "batch_capability", "phase_executor",
                "planning_config", "execution_state", "_require_batch_scene_adapter",
            ),
            _arm_joint_state=state_planning._arm_joint_state,
            _plan_pose_from_joint_positions=planning_queries.plan_pose_from_joint_positions,
            _log_plan_result=setup._log_plan_result,
            port_type=AttachmentPort,
        )
    )
    phases.execute = execution.forward_phase_command
    phases.get_armbase_pose = execution.get_armbase_pose
    phases.update_pose_cost_metric = setup.update_pose_cost_metric
    setup._get_ee_pose = execution.get_ee_pose
    planning_queries.forward_kinematic = execution.forward_kinematic
    planning_queries._forward_kinematic_batch = execution._forward_kinematic_batch
    return ControllerComponents(setup, state_planning, planning_queries, phases, execution, attachment)


# The execution component is being migrated independently.  This name is a
# temporary import alias only; it has no façade binding or magic dispatch.
ControllerComponent = ComponentState


__all__ = [
    "AttachmentPort",
    "ComponentPort",
    "ComponentState",
    "ControllerComponent",
    "ControllerComponents",
    "MutableExecutionState",
    "PlanningConfig",
    "PhasesPort",
    "PlanningQueriesPort",
    "SetupPort",
    "StatePlanningPort",
    "wire_controller_components",
]
