from workflows.simbox.core.utils.task_data import normalize_runtime_data_config


def test_empty_converted_data_uses_task_name_defaults():
    task = {"name": "s04_map04", "task": "BananaBaseTask", "data": {}}

    data = normalize_runtime_data_config(task, "/tmp/simbox_task.yaml")

    assert data == {
        "task_dir": "s04_map04",
        "collect_info": "s04_map04",
        "version": "v1.0",
        "update": True,
    }
    assert task["data"] is data


def test_explicit_runtime_data_is_preserved():
    task = {
        "name": "scene",
        "data": {
            "task_dir": "custom/path",
            "collect_info": "collector",
            "version": "v2.0",
            "update": False,
        },
    }

    data = normalize_runtime_data_config(task, "/tmp/task.yaml")

    assert data == task["data"]
    assert data["task_dir"] == "custom/path"
    assert data["collect_info"] == "collector"
    assert data["version"] == "v2.0"
    assert data["update"] is False


def test_missing_data_falls_back_to_config_filename():
    task = {}

    data = normalize_runtime_data_config(task, "/tmp/generated_scene.yaml")

    assert data["task_dir"] == "generated_scene"
    assert data["collect_info"] == "generated_scene"
