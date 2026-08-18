"""Round-trip tests for TaskIR -> native YAML assembly."""

from pathlib import Path

import yaml
from omegaconf import OmegaConf

from agent.task_ir import assemble_task_ir_to_document, parse_task_yaml_to_ir, validate_task_ir
from workflows.simbox.utils.task_config_parser import TaskConfigParser


ROOT = Path(__file__).resolve().parents[2]

ROUNDTRIP_CASES = [
    "workflows/simbox/core/configs/tasks/basic/lift2/arrange_the_tableware/arrange_the_tableware_part0.yaml",
    "workflows/simbox/core/configs/tasks/art/franka/open_the_pot/open_the_pot.yaml",
    "workflows/simbox/core/configs/tasks/basic/split_aloha/insert_the_markpen_in_penholder/insert_the_markpen_in_penholder_part0.yaml",
]


def _load_resolved_task(task_yaml_path: Path, task_index: int = 0) -> dict:
    yaml_conf = OmegaConf.load(str(task_yaml_path))
    tasks = OmegaConf.to_container(yaml_conf["tasks"], resolve=True)
    return tasks[task_index]


def test_roundtrip_yaml_task_semantics(tmp_path):
    for rel_path in ROUNDTRIP_CASES:
        source_path = ROOT / rel_path
        original_task = _load_resolved_task(source_path)

        task_ir = parse_task_yaml_to_ir(source_path)
        roundtrip_doc = assemble_task_ir_to_document(task_ir)

        output_path = tmp_path / f"{source_path.stem}.roundtrip.yaml"
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(roundtrip_doc, f, sort_keys=False, allow_unicode=True)

        roundtrip_task = _load_resolved_task(output_path)
        assert roundtrip_task == original_task, f"resolved task dict changed after roundtrip for {rel_path}"


def test_roundtrip_yaml_can_be_loaded_by_task_config_parser(tmp_path):
    source_path = ROOT / ROUNDTRIP_CASES[0]
    task_ir = parse_task_yaml_to_ir(source_path)
    roundtrip_doc = assemble_task_ir_to_document(task_ir)

    output_path = tmp_path / "basic_roundtrip.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(roundtrip_doc, f, sort_keys=False, allow_unicode=True)

    parsed_tasks = TaskConfigParser(str(output_path)).parse_tasks()
    assert len(parsed_tasks) == 1
    assert parsed_tasks[0]["name"] == "banana_base_task"
    assert parsed_tasks[0]["task"] == "BananaBaseTask"
    assert "skills" in parsed_tasks[0]


def test_roundtrip_yaml_reparsed_task_ir_still_valid(tmp_path):
    source_path = ROOT / ROUNDTRIP_CASES[2]
    task_ir = parse_task_yaml_to_ir(source_path)
    roundtrip_doc = assemble_task_ir_to_document(task_ir)

    output_path = tmp_path / "split_aloha_roundtrip.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(roundtrip_doc, f, sort_keys=False, allow_unicode=True)

    reparsed_ir = parse_task_yaml_to_ir(output_path)
    validation = validate_task_ir(reparsed_ir, repo_root=ROOT, check_assets=False)

    assert validation["schema_ok"] is True
    assert validation["references_ok"] is True
    assert validation["compatibility_ok"] is True
