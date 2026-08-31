from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_admitted_runtime_does_not_infer_arm_from_curobo_filename():
    paths = [
        "workflows/simbox_dual_workflow.py",
        "workflows/simbox/core/controllers/curobo/controller.py",
        "workflows/simbox/core/controllers/curobo/scene_setup.py",
        "workflows/simbox/core/controllers/splitaloha_controller.py",
        "workflows/simbox/core/controllers/lift2_controller.py",
        "workflows/simbox/core/controllers/fr3_controller.py",
        "workflows/simbox/core/controllers/frankarobotiq85_controller.py",
        "workflows/simbox/core/skills/pick.py",
        "workflows/simbox/core/skills/place.py",
        "workflows/simbox/core/robots/profile.py",
    ]
    joined = "\n".join(_source(path) for path in paths)
    assert '"left" in robot_file' not in joined
    assert '"right" in robot_file' not in joined
    assert '"r5a" in' not in joined
    assert "arm_name=controller_name" in joined
    assert "arm_name=arm_name" in joined


def test_runtime_logging_does_not_branch_on_robot_instance_name():
    joined = _source("workflows/simbox/core/loggers/utils.py") + _source(
        "workflows/simbox/core/loggers/lmdb_logger.py"
    )
    for profile_name in (
        "split_aloha",
        "lift2",
        "franka",
        "robotiq",
        "genie",
    ):
        assert profile_name not in joined.lower()


def test_typed_trajectory_visualization_is_bound_to_runtime_and_skills():
    controller = _source(
        "workflows/simbox/core/controllers/curobo/controller.py"
    )
    base_skill = _source("workflows/simbox/core/skills/base_skill.py")
    pick = _source("workflows/simbox/core/skills/pick.py")
    place = _source("workflows/simbox/core/skills/place.py")

    assert "TrajectoryVisualizationFrame(" in controller
    assert "self.runtime.trajectory_visualization_frame" in controller
    assert "def record_selected_trajectory(" in base_skill
    assert "record_selected_trajectory(" in pick
    assert "record_selected_trajectory(" in place


def test_post_place_return_to_initial_is_a_typed_dag_node():
    workflow = _source("workflows/simbox_dual_workflow.py")
    skill = _source(
        "workflows/simbox/core/skills/return_to_episode_initial.py"
    )

    assert "def _append_return_to_initial_nodes(" in workflow
    assert 'reset_id = f"{place_node[\'id\']}:return_to_episode_initial"' in workflow
    assert '"name": "return_to_episode_initial"' in workflow
    assert '"depends_on": [place_node["id"]]' in workflow
    assert "self.joint_command(" in skill
    assert "MotionPhase.CARRY_HOME" in skill
    assert ".controller" not in skill
