import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import time

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / "docker" / "docker-compose.yml"
SCRIPT_PATH = REPO_ROOT / "scripts" / "docker" / "prepare_simbox_run.py"
SPEC = importlib.util.spec_from_file_location("prepare_simbox_run", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_simbox_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_simbox_run)


def _args(task_config: str, **overrides):
    values = {
        "task_config": task_config,
        "launcher_config": "configs/de_plan_with_render_template.yaml",
        "run_name": "agent/test/attempt_00",
        "random_num": 1,
        "random_seed": 0,
        "output_dir": "",
        "seq_output_dir": "",
        "episode_event_path": "",
        "debug_output_dir": "",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_compose_defines_only_isaac_without_ros_nav2_runtime_fields():
    compose = COMPOSE_PATH.read_text(encoding="utf-8")

    assert "  isaac:" in compose
    assert "  nav2:" not in compose
    assert "ROS_DOMAIN_ID" not in compose
    assert "INTERNDATA_NAV2_" not in compose


def test_contract_is_isaac_only_and_uses_container_local_gpu_indices():
    task = "workflows/simbox/core/configs/tasks/example/sort_the_rubbish.yaml"

    contract = prepare_simbox_run.build_contract(_args(task))

    assert contract["task_container"] == f"/workspace/{task}"
    assert contract["launcher_container"] == "/workspace/configs/de_plan_with_render_template.yaml"
    assert "needs_nav2" not in contract
    assert not any(key.startswith("robot_") or key.startswith("base_") for key in contract)
    assert "--load_stage.scene_loader.args.simulator.active_gpu=0" in contract["launcher_args"]
    assert "--load_stage.scene_loader.args.simulator.physics_gpu=0" in contract["launcher_args"]


def test_contract_rejects_paths_outside_repository(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("tasks: []\n", encoding="utf-8")
    monkeypatch.setattr(prepare_simbox_run, "REPO_ROOT", repo_root)

    with pytest.raises(ValueError, match="inside the repository"):
        prepare_simbox_run.build_contract(_args(str(outside)))


def test_docker_runner_dry_run_never_requires_local_isaac_or_conda():
    output_root = REPO_ROOT / "output"
    output_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="docker_runner_test_", dir=output_root) as raw_dir:
        run_dir = Path(raw_dir)
        metadata_path = run_dir / "docker_runtime.json"
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "ROS_DOMAIN_ID": "233",
                "INTERNDATA_NAV2_ROBOT_NAME": "must-not-be-exported",
                "TASK_CONFIG": "workflows/simbox/core/configs/tasks/example/sort_the_rubbish.yaml",
                "RUN_NAME": "agent/test/attempt_00",
                "OUTPUT_DIR": str(run_dir / "data"),
                "INTERNDATA_EPISODE_EVENT_PATH": str(run_dir / "episode_events.jsonl"),
                "INTERNDATA_DOCKER_METADATA_PATH": str(metadata_path),
                "SIMBOX_DEBUG_OUTPUT_DIR": str(run_dir / "simbox_debug"),
            }
        )

        completed = subprocess.run(
            ["bash", "scripts/docker/run_simbox_task.sh"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert "up_simbox_isaac.sh" in completed.stdout
        assert "isaac nav2" not in completed.stdout
        assert "ROS_DOMAIN_ID" not in completed.stdout
        assert "INTERNDATA_NAV2_" not in completed.stdout
        assert "conda" not in completed.stdout.lower()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["status"] == "dry_run"
        assert metadata["host_gpu_id"] == 0
        assert "nav2_container" not in metadata
        assert "ros_domain_id" not in metadata


def test_docker_runner_ignores_ros_domain_environment():
    output_root = REPO_ROOT / "output"
    output_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="docker_invalid_domain_test_", dir=output_root) as raw_dir:
        run_dir = Path(raw_dir)
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "ROS_DOMAIN_ID": "not-used",
                "TASK_CONFIG": "workflows/simbox/core/configs/tasks/example/sort_the_rubbish.yaml",
                "INTERNDATA_DOCKER_METADATA_PATH": str(run_dir / "docker_runtime.json"),
                "SIMBOX_DEBUG_OUTPUT_DIR": str(run_dir / "simbox_debug"),
            }
        )

        completed = subprocess.run(
            ["bash", "scripts/docker/run_simbox_task.sh"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert "ROS_DOMAIN_ID" not in completed.stdout


def _write_fake_docker(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o755)


def _runner_env(run_dir: Path, fake_bin: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "TASK_CONFIG": "workflows/simbox/core/configs/tasks/example/sort_the_rubbish.yaml",
            "RUN_NAME": "agent/test/attempt_00",
            "INTERNDATA_DOCKER_METADATA_PATH": str(run_dir / "docker_runtime.json"),
            "SIMBOX_DEBUG_OUTPUT_DIR": str(run_dir / "simbox_debug"),
        }
    )
    return env


def test_docker_runner_reports_missing_required_image(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_docker(
        fake_bin / "docker",
        """
        if [[ "$1" == "compose" && "$2" == "version" ]]; then exit 0; fi
        if [[ "$1" == "info" ]]; then exit 0; fi
        if [[ "$1" == "compose" && "$*" == *"config --images"* ]]; then
            printf '%s\\n' local/isaac-sim-test:latest
            exit 0
        fi
        if [[ "$1" == "image" && "$2" == "inspect" ]]; then exit 1; fi
        exit 99
        """,
    )

    output_root = REPO_ROOT / "output"
    output_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="docker_missing_image_test_", dir=output_root) as raw_dir:
        run_dir = Path(raw_dir)
        completed = subprocess.run(
            ["bash", "scripts/docker/run_simbox_task.sh"],
            cwd=REPO_ROOT,
            env=_runner_env(run_dir, fake_bin),
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 5
        assert "DOCKER_IMAGE_MISSING" in completed.stderr
        metadata = json.loads((run_dir / "docker_runtime.json").read_text(encoding="utf-8"))
        assert metadata["status"] == "image_missing"
        assert metadata["exit_code"] == 5


def test_docker_runner_sigterm_records_interrupt_and_cleans_stack(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls_path = tmp_path / "docker_calls.log"
    _write_fake_docker(
        fake_bin / "docker",
        """
        printf '%s\\n' "$*" >> "${FAKE_DOCKER_CALLS}"
        if [[ "$1" == "compose" && "$2" == "version" ]]; then exit 0; fi
        if [[ "$1" == "info" ]]; then exit 0; fi
        if [[ "$1" == "compose" && "$*" == *"config --images"* ]]; then
            printf '%s\\n' local/isaac-sim-test:latest
            exit 0
        fi
        if [[ "$1" == "image" && "$2" == "inspect" ]]; then exit 0; fi
        if [[ "$1" == "compose" && "$*" == *" up -d "* ]]; then exit 0; fi
        if [[ "$1" == "logs" ]]; then sleep 60; exit 0; fi
        if [[ "$1" == "wait" ]]; then sleep 60; exit 0; fi
        if [[ "$1" == "stop" ]]; then exit 0; fi
        if [[ "$1" == "compose" && "$*" == *" down --remove-orphans"* ]]; then exit 0; fi
        exit 99
        """,
    )
    output_root = REPO_ROOT / "output"
    output_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="docker_signal_test_", dir=output_root) as raw_dir:
        run_dir = Path(raw_dir)
        env = _runner_env(run_dir, fake_bin)
        env["FAKE_DOCKER_CALLS"] = str(calls_path)
        process = subprocess.Popen(
            ["bash", "scripts/docker/run_simbox_task.sh"],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        metadata_path = run_dir / "docker_runtime.json"
        for _ in range(100):
            if metadata_path.is_file() and json.loads(metadata_path.read_text())["status"] == "running":
                break
            time.sleep(0.05)
        else:
            process.kill()
            pytest.fail("runner did not reach running state")

        os.killpg(process.pid, 15)
        assert process.wait(timeout=10) == 143

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["status"] == "interrupted"
        assert metadata["exit_code"] == 143
        calls = calls_path.read_text(encoding="utf-8")
        assert "stop isaac-" in calls
        assert "down --remove-orphans" in calls
