"""Construction-order contract for controller-owned robot frame paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def simulation_app():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    yield app
    app.close()


def test_robot_ee_path_is_resolved_before_runtime_and_scene_ports(
    simulation_app, monkeypatch
):
    from core.controllers.controller_registry import ArmSpec
    from core.controllers.curobo.controller import TemplateController
    import core.controllers.curobo.controller as template_module

    expected_base = "/World/robot/fl_base"
    expected_ee = "/World/robot/fl_ee"
    observed = {}

    class FakeRobot:
        cfg = {}
        dof_names = ("joint1", "gripper")
        left_joint_indices = [0]
        left_gripper_indices = [1]
        left_gripper_state = 1.0
        fl_base_path = expected_base
        fl_ee_path = expected_ee

    class FakeTask:
        robots = {"robot": FakeRobot()}
        cfg = {"planning": {"execution_safety": {"max_waypoint_stride": 2}}}
        root_prim_path = "/World"

    class FakeWorld:
        def get_physics_dt(self):
            return 0.01

    class FakeSceneManager:
        def build_world_config(self, reference_prim_path):
            assert reference_prim_path == expected_base
            return object()

        def bind_scene_port(self, port):
            observed["bound_scene_port"] = port

        def _port_obstacle_pose(self, _port, _path):
            return None

    class FakeRuntime:
        def __init__(self):
            self.robot_port = SimpleNamespace(
                kin_model=object(), interpolation_dt=0.01, obstacle_pose=None
            )
            self.check_current_start_state = lambda: (True, "valid")
            self.attach_collision_object = lambda *args, **kwargs: True
            self.detach_attachment = lambda: None
            self.has_attached_collision_spheres = lambda: False

    def fake_build_runtime(self, planning_world):
        assert planning_world is not None
        observed["runtime_ee_path"] = self._setup.robot_ee_path
        observed["runtime_reference_path"] = self._setup.reference_prim_path
        observed["execution_base_path"] = self._execution.robot_base_path
        observed["execution_ee_path"] = self._execution.robot_ee_path
        observed["execution_task_root"] = self._execution.task_root_prim_path
        assert not hasattr(self._execution, "task")
        return FakeRuntime()

    def fake_scene_port(**kwargs):
        observed["scene_port_kwargs"] = kwargs
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(TemplateController, "_build_runtime", fake_build_runtime)
    monkeypatch.setattr(template_module, "PlannerScenePort", fake_scene_port)

    class FakeController(TemplateController):
        arm_spec = ArmSpec(
            planner_joints=("joint1",),
            control_joints={"left": ("joint1",)},
            supported_arms=("left",),
        )

    controller = FakeController(
        name="robot",
        robot_file="left_robot.yaml",
        arm_name="left",
        task=FakeTask(),
        world=FakeWorld(),
        collision_scene_manager=FakeSceneManager(),
    )

    assert controller._setup.robot_ee_path == expected_ee
    assert observed["runtime_ee_path"] == expected_ee
    assert observed["runtime_reference_path"] == expected_base
    assert observed["execution_base_path"] == expected_base
    assert observed["execution_ee_path"] == expected_ee
    assert observed["execution_task_root"] == "/World"
    assert observed["scene_port_kwargs"]["robot_ee_path"] == expected_ee
    assert observed["bound_scene_port"].robot_ee_path == expected_ee

    # The task root belongs to ControllerSetup and is injected into execution;
    # Skills receive the one concrete typed runtime owned by the controller.
    assert not hasattr(controller, "task_root_prim_path")
    assert controller._setup.task_root_prim_path == "/World"
    assert controller.skill_runtime is controller.runtime
