from workflows.simbox.core.utils.task_data import normalize_runtime_data_config


def test_empty_converted_data_uses_task_name_defaults():
    task = {
        "name": "s04_map04",
        "task": "BananaBaseTask",
        "max_episode_length": 4321,
        "data": {},
    }

    data = normalize_runtime_data_config(task, "/tmp/simbox_task.yaml")

    assert data == {
        "task_dir": "s04_map04",
        "collect_info": "s04_map04",
        "version": "v1.0",
        "update": True,
        "max_episode_length": 4321,
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
            "max_episode_length": 2500,
        },
    }

    data = normalize_runtime_data_config(task, "/tmp/task.yaml")

    assert data == task["data"]
    assert data["task_dir"] == "custom/path"
    assert data["collect_info"] == "collector"
    assert data["version"] == "v2.0"
    assert data["update"] is False
    assert data["max_episode_length"] == 2500


def test_missing_data_falls_back_to_config_filename():
    task = {}

    data = normalize_runtime_data_config(task, "/tmp/generated_scene.yaml")

    assert data["task_dir"] == "generated_scene"
    assert data["collect_info"] == "generated_scene"
    assert data["max_episode_length"] == 10_000
    assert task["max_episode_length"] == 10_000


def test_nested_episode_length_takes_precedence_over_top_level_value():
    task = {
        "max_episode_length": 1000,
        "data": {"max_episode_length": "2000"},
    }

    data = normalize_runtime_data_config(task, "/tmp/task.yaml")

    assert data["max_episode_length"] == 2000
    assert task["max_episode_length"] == 1000


def test_invalid_episode_length_is_rejected():
    task = {"max_episode_length": 0, "data": {}}

    try:
        normalize_runtime_data_config(task, "/tmp/task.yaml")
    except ValueError as exc:
        assert str(exc) == "max_episode_length must be a positive integer"
    else:
        raise AssertionError("expected invalid max_episode_length to fail")
