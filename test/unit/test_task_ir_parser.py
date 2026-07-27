"""Unit tests for TaskIR parser."""

from pathlib import Path

from agent.task_ir import parse_task_yaml_to_ir, parse_tasks_yaml_to_ir, validate_task_ir


ROOT = Path(__file__).resolve().parents[2]


def test_parse_basic_task_yaml_to_ir():
    yaml_path = (
        ROOT
        / "workflows/simbox/core/configs/tasks/basic/lift2/arrange_the_tableware/arrange_the_tableware_part0.yaml"
    )
    task_ir = parse_task_yaml_to_ir(yaml_path)

    assert task_ir["task_name"] == "banana_base_task"
    assert task_ir["task_class"] == "BananaBaseTask"
    assert task_ir["task_family"] == "basic"
    assert isinstance(task_ir["robots"], list) and task_ir["robots"]
    assert isinstance(task_ir["objects"], list) and task_ir["objects"]
    assert isinstance(task_ir["skill_steps"], list) and task_ir["skill_steps"]

    first_step = task_ir["skill_steps"][0]
    assert first_step["step_id"] == "step_001"
    assert first_step["robot_name"] == "lift2"
    assert first_step["arm"] in {"left", "right"}
    assert first_step["skill_name"] in {"pick", "place", "heuristic__skill"}
    assert isinstance(first_step["object_refs"], list)


def test_parse_navigation_task_yaml_to_ir_and_validate_without_assets():
    yaml_path = ROOT / "workflows/simbox/core/configs/tasks/navigation/split_aloha/navigate_asset_obstacles.yaml"
    task_ir = parse_task_yaml_to_ir(yaml_path)

    assert task_ir["task_family"] == "navigation"
    assert any(step["skill_name"] == "navigate" for step in task_ir["skill_steps"])

    validation = validate_task_ir(task_ir, repo_root=ROOT, check_assets=False)
    assert validation["schema_ok"] is True
    assert validation["references_ok"] is True
    assert validation["compatibility_ok"] is True
    assert validation["assets_ok"] is None


def test_parse_all_tasks_returns_list():
    yaml_path = ROOT / "workflows/simbox/core/configs/tasks/art/franka/open_the_pot/open_the_pot.yaml"
    all_task_irs = parse_tasks_yaml_to_ir(yaml_path)
    assert isinstance(all_task_irs, list)
    assert len(all_task_irs) == 1
    assert all_task_irs[0]["task_family"] == "art"
