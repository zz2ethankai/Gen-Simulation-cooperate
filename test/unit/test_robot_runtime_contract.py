from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_admitted_runtime_does_not_infer_arm_from_curobo_filename():
    paths = [
        "workflows/simbox_dual_workflow.py",
        "workflows/simbox/core/controllers/template_controller.py",
        "workflows/simbox/core/controllers/splitaloha_controller.py",
        "workflows/simbox/core/controllers/lift2_controller.py",
        "workflows/simbox/core/controllers/fr3_controller.py",
        "workflows/simbox/core/controllers/frankarobotiq85_controller.py",
        "workflows/simbox/core/skills/pick.py",
        "workflows/simbox/core/skills/place.py",
        "workflows/simbox/core/planning/grasp_plan_evaluator.py",
    ]
    joined = "\n".join(_source(path) for path in paths)
    assert '"left" in robot_file' not in joined
    assert '"right" in robot_file' not in joined
    assert '"r5a" in' not in joined


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
