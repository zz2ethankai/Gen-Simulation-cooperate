from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = REPO_ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.mobile.g1_decoupled_wbc import MotorCommand, RobotState  # noqa: E402
from core.mobile.g1_locomotion_driver import G1LocomotionDriver  # noqa: E402


_WORKFLOW_PATH = REPO_ROOT / "workflows" / "simbox_dual_workflow.py"


def _load_workflow_warmup_methods():
    tree = ast.parse(_WORKFLOW_PATH.read_text(encoding="utf-8"))
    workflow_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SimBoxDualWorkFlow"
    )
    method_names = {
        "_finish_reset_warmup",
        "_resolved_navigation_warmup_steps",
    }
    method_nodes = [
        node
        for node in workflow_node.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    if {node.name for node in method_nodes} != method_names:
        raise AssertionError("workflow is missing the G1 warmup handoff methods")
    namespace = {}
    module = ast.fix_missing_locations(ast.Module(body=method_nodes, type_ignores=[]))
    exec(compile(module, _WORKFLOW_PATH, "exec"), namespace)
    return namespace


class _FakeWorld:
    def __init__(self):
        self.current_time = 0.0
        self.current_time_step_index = 0

    def get_physics_dt(self):
        return 0.005


class _FakeRobot:
    def __init__(self):
        self.name = "unitree_g1"
        self.pose = (
            np.array([0.0, 0.0, 0.8], dtype=np.float64),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        self.commands = []
        self.state = RobotState(
            body_q=np.zeros(29),
            body_dq=np.zeros(29),
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            base_ang_vel=np.zeros(3),
            pelvis_z=0.8,
        )

    def get_base_interface(self):
        return {
            "steering_joint_names": [],
            "wheel_joint_names": [],
            "steering_joint_indices": [],
            "wheel_joint_indices": [],
            "base_cfg": {
                "platform": {
                    "profile": "unitree_g1_decoupled_wbc",
                    "local_navigation": {
                        "controller_hard_limits": {
                            "max_velocity": [0.35, 0.0, 0.5],
                            "min_velocity": [0.0, 0.0, -0.5],
                        }
                    },
                },
                "command_timeout": 0.25,
                "debug_history_size": 8,
            },
        }

    def get_nav_base_pose(self):
        return self.pose

    def get_locomotion_state(self):
        return self.state

    def apply_locomotion_command(self, command):
        self.commands.append(command)


class _FakePolicy:
    def __init__(self):
        self.calls = []
        self.reset_count = 0
        self.last_mode = "balance"
        self.inference_count = 0

    def reset(self):
        self.reset_count += 1

    def step(self, robot_state, navigate_cmd, *, env_step_dt):
        self.calls.append((robot_state, navigate_cmd.copy(), env_step_dt))
        self.inference_count += 1
        self.last_mode = "walk" if navigate_cmd[0] > 0.0 else "balance"
        return MotorCommand(q_target=np.full(29, navigate_cmd[0]))


class _FakeCommandAdapter:
    def translate(self, *, vx_body, vy_body, wz_body, current_yaw):
        return SimpleNamespace(
            navigate_cmd=np.asarray([vx_body, vy_body, float(abs(wz_body) > 0.01), current_yaw])
        )


class G1NavigationConfigTest(unittest.TestCase):
    def test_g1_disables_final_yaw_rotate_first_gate(self):
        config_path = (
            REPO_ROOT / "workflows/simbox/core/configs/robots/unitree_g1.yaml"
        )
        robot_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        rotate_first_error = robot_cfg["base"]["local_navigation"]["controller"][
            "rotate_first_error_rad"
        ]
        self.assertAlmostEqual(rotate_first_error, np.pi)

    def test_navigation_task_uses_dedicated_untextured_arena(self):
        task_path = (
            REPO_ROOT
            / "workflows/simbox/core/configs/tasks/navigation/unitree_g1/navigate_empty_validation.yaml"
        )
        task_cfg = yaml.safe_load(task_path.read_text(encoding="utf-8"))["tasks"][0]

        arena_relpath = (
            "workflows/simbox/core/configs/arenas/"
            "unitree_g1_navigate_floor_untextured_arena.yaml"
        )
        self.assertEqual(task_cfg["arena_file"], arena_relpath)

        arena_path = REPO_ROOT / arena_relpath
        arena_cfg = yaml.safe_load(arena_path.read_text(encoding="utf-8"))
        fixtures = arena_cfg["fixtures"]
        self.assertTrue(fixtures)
        self.assertTrue(all("texture" not in fixture for fixture in fixtures))
        floor = next(fixture for fixture in fixtures if fixture["name"] == "floor")
        self.assertTrue(floor["collision_enabled"])

    def test_navigation_handoff_uses_50_render_steps(self):
        config_path = (
            REPO_ROOT
            / "workflows/simbox/core/configs/bases/unitree_g1_sonic.yaml"
        )
        base_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            base_cfg["platform"]["decoupled_wbc"]["navigation_warmup_steps"],
            50,
        )

    def test_g1_settle_tolerances_cover_balance_policy_jitter(self):
        config_path = (
            REPO_ROOT
            / "workflows/simbox/core/configs/bases/unitree_g1_sonic.yaml"
        )
        base_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        settle_cfg = base_cfg["platform"]["local_navigation"]["settle"]
        self.assertEqual(settle_cfg["linear_speed_tolerance"], 0.01)
        self.assertEqual(settle_cfg["angular_speed_tolerance"], 0.01)
        self.assertEqual(settle_cfg["consecutive_steps"], 8)

    def test_g1_navigation_adapter_thresholds_are_explicit(self):
        config_path = (
            REPO_ROOT
            / "workflows/simbox/core/configs/bases/unitree_g1_sonic.yaml"
        )
        base_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        wbc_cfg = base_cfg["platform"]["decoupled_wbc"]
        self.assertEqual(wbc_cfg["walk_heading_tolerance_rad"], 0.2)
        self.assertEqual(
            wbc_cfg["final_heading_linear_speed_threshold"],
            0.24,
        )

    def test_g1_enables_terminal_approach_planning(self):
        config_path = (
            REPO_ROOT / "workflows/simbox/core/configs/robots/unitree_g1.yaml"
        )
        robot_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        planner_cfg = robot_cfg["base"]["local_navigation"]["planner"]
        self.assertEqual(planner_cfg["terminal_approach_distance_m"], 0.6)
        self.assertEqual(planner_cfg["terminal_approach_step_m"], 0.1)

    def test_g1_selects_opt_in_phased_waypoint_controller(self):
        config_path = (
            REPO_ROOT / "workflows/simbox/core/configs/robots/unitree_g1.yaml"
        )
        robot_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        controller_cfg = robot_cfg["base"]["local_navigation"]["controller"]
        self.assertEqual(controller_cfg["controller_type"], "phased_waypoint")
        self.assertEqual(controller_cfg["terminal_max_linear_velocity"], 0.15)
        self.assertEqual(controller_cfg["max_angular_velocity"], 0.5)


class G1LocomotionDriverTest(unittest.TestCase):
    def test_step_runs_decoupled_wbc_and_applies_joint_command(self):
        world = _FakeWorld()
        robot = _FakeRobot()
        policy = _FakePolicy()
        driver = G1LocomotionDriver(
            robot,
            world=world,
            policy=policy,
            command_adapter=_FakeCommandAdapter(),
        )

        driver.prepare_for_navigation()
        driver.set_command(0.3, 0.0, 0.0)
        world.current_time = 0.02
        driver.step(step_dt=0.02)

        self.assertEqual(policy.reset_count, 0)
        self.assertEqual(len(policy.calls), 1)
        self.assertEqual(len(robot.commands), 1)
        np.testing.assert_allclose(robot.commands[0].q_target, np.full(29, 0.3))
        snapshot = driver.get_logging_action_snapshot()
        self.assertEqual(snapshot["locomotion_mode"], 1)
        self.assertEqual(len(snapshot["joint_position_targets"]), 29)

    def test_driver_passes_each_physics_tick_to_wbc_rate_gate(self):
        world = _FakeWorld()
        robot = _FakeRobot()
        policy = _FakePolicy()
        driver = G1LocomotionDriver(
            robot,
            world=world,
            policy=policy,
            command_adapter=_FakeCommandAdapter(),
        )
        driver.prepare_for_navigation()
        driver.set_command(0.3, 0.0, 0.0)

        world.current_time = 0.005
        world.current_time_step_index = 1
        driver.step(step_dt=0.005)
        world.current_time = 0.010
        world.current_time_step_index = 2
        driver.step(step_dt=0.005)

        self.assertEqual(len(policy.calls), 2)
        self.assertEqual(policy.calls[0][2], 0.005)
        self.assertEqual(policy.calls[1][2], 0.005)

    def test_driver_uses_physics_step_delta_when_render_batches_physics(self):
        world = _FakeWorld()
        policy = _FakePolicy()
        driver = G1LocomotionDriver(
            _FakeRobot(),
            world=world,
            policy=policy,
            command_adapter=_FakeCommandAdapter(),
        )
        driver.prepare_for_navigation()
        driver.set_command(0.3, 0.0, 0.0)

        world.current_time = 0.02
        world.current_time_step_index = 4
        driver.step(step_dt=0.005)

        self.assertEqual(policy.calls[-1][2], 0.02)

    def test_driver_restarts_physics_step_tracking_after_reset(self):
        world = _FakeWorld()
        policy = _FakePolicy()
        driver = G1LocomotionDriver(
            _FakeRobot(),
            world=world,
            policy=policy,
            command_adapter=_FakeCommandAdapter(),
        )

        world.current_time = 0.02
        world.current_time_step_index = 4
        driver.step(step_dt=0.005)
        world.current_time = 0.0
        world.current_time_step_index = 0
        driver.reset()
        driver.step(step_dt=0.005)

        self.assertEqual(policy.calls[-1][2], 0.005)

    def test_driver_without_world_uses_explicit_step_dt(self):
        policy = _FakePolicy()
        driver = G1LocomotionDriver(
            _FakeRobot(),
            world=None,
            policy=policy,
            command_adapter=_FakeCommandAdapter(),
        )

        driver.step(step_dt=0.005)

        self.assertEqual(policy.calls[-1][2], 0.005)

    def test_stale_command_becomes_idle(self):
        world = _FakeWorld()
        robot = _FakeRobot()
        policy = _FakePolicy()
        driver = G1LocomotionDriver(
            robot,
            world=world,
            policy=policy,
            command_adapter=_FakeCommandAdapter(),
        )
        driver.prepare_for_navigation()
        driver.set_command(0.4, 0.0, 0.0)

        world.current_time = 1.0
        driver.step(step_dt=0.02)

        self.assertEqual(policy.calls[-1][1][0], 0.0)

    def test_driver_reset_still_clears_wbc_state(self):
        policy = _FakePolicy()
        driver = G1LocomotionDriver(
            _FakeRobot(),
            world=_FakeWorld(),
            policy=policy,
            command_adapter=_FakeCommandAdapter(),
        )

        driver.reset(clear_debug_history=True)

        self.assertEqual(policy.reset_count, 1)

    def test_finish_reset_warmup_preserves_policy_and_clears_debug_counters(self):
        world = _FakeWorld()
        policy = _FakePolicy()
        driver = G1LocomotionDriver(
            _FakeRobot(),
            world=world,
            policy=policy,
            command_adapter=_FakeCommandAdapter(),
        )
        driver.prepare_for_navigation()
        driver.set_command(0.3, 0.0, 0.0)
        world.current_time = 0.02
        driver.step(step_dt=0.02)

        driver.finish_reset_warmup(clear_debug_history=True)

        self.assertEqual(policy.reset_count, 0)
        self.assertEqual(driver.get_logging_action_snapshot()["driver_command_message_count"], 0)
        self.assertEqual(driver.required_navigation_warmup_steps, 50)


class WorkflowWarmupHandoffTest(unittest.TestCase):
    @staticmethod
    def _workflow(drivers):
        methods = _load_workflow_warmup_methods()
        workflow = SimpleNamespace(
            _local_base_drivers=drivers,
            reset_calls=[],
        )
        workflow._reset_fixed_robot_start_states_after_physics = (
            lambda *, clear_debug_history: workflow.reset_calls.append(clear_debug_history)
        )
        workflow._finish_reset_warmup = methods["_finish_reset_warmup"].__get__(workflow)
        workflow._resolved_navigation_warmup_steps = methods[
            "_resolved_navigation_warmup_steps"
        ].__get__(workflow)
        return workflow

    def test_existing_driver_keeps_original_post_warmup_reset(self):
        workflow = self._workflow({"legacy": SimpleNamespace()})

        workflow._finish_reset_warmup(clear_debug_history=True)

        self.assertEqual(workflow.reset_calls, [True])
        self.assertEqual(workflow._resolved_navigation_warmup_steps(10), 10)

    def test_g1_driver_preserves_stabilized_state_and_requests_50_steps(self):
        finish_calls = []
        driver = SimpleNamespace(
            preserve_reset_warmup_state=True,
            required_navigation_warmup_steps=50,
            finish_reset_warmup=(
                lambda *, clear_debug_history: finish_calls.append(clear_debug_history)
            ),
        )
        workflow = self._workflow({"unitree_g1": driver})

        workflow._finish_reset_warmup(clear_debug_history=True)

        self.assertEqual(workflow.reset_calls, [])
        self.assertEqual(finish_calls, [True])
        self.assertEqual(workflow._resolved_navigation_warmup_steps(10), 50)


if __name__ == "__main__":
    unittest.main()
