"""Contracts for the native Isaac Sim 6 / CuRobo v2 Conda runtime."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "simbox" / "run_simbox_task.sh"
SETUP = ROOT / "scripts" / "conda" / "setup_isaac6_env.sh"
ACTIVATE = ROOT / "scripts" / "conda" / "activate_isaac6_env.sh"
RESOLVER_PATH = ROOT / "scripts" / "conda" / "resolve_curobo_requirements.py"
CUDA_PREPARE_PATH = ROOT / "scripts" / "conda" / "prepare_cuda_home.py"
WORKSPACE_VALIDATOR = ROOT / "scripts" / "simbox" / "validate_workspace_candidates.py"
WORKSPACE_TEMPLATE = ROOT / "configs" / "simbox" / "de_workspace_probe_template.yaml"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_resolver():
    return _load_module("resolve_curobo_requirements", RESOLVER_PATH)


def test_conda_setup_dry_run_pins_isaac6_torch_and_python():
    completed = subprocess.run(
        ["bash", str(SETUP), "--env", "test-isaac6", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "python=3.12" in completed.stdout
    assert "torch==2.11.0" in completed.stdout
    assert "whl/cu128" in completed.stdout
    normalized = completed.stdout.replace("\\", "")
    assert "isaacsim[all,extscache]==6.0.1.0" in normalized
    assert "InternDataAssets/curobov2/pyproject.toml" in completed.stdout


def test_conda_runner_dry_run_uses_absolute_physical_gpu_index():
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "CONDA_ENV": "test-isaac6",
            "GPU_ID": "3",
            "TASK_CONFIG": (
                "workflows/simbox/core/configs/tasks/example/"
                "sort_the_rubbish.yaml"
            ),
        }
    )
    completed = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "-u CUDA_VISIBLE_DEVICES" in completed.stdout
    assert "test-isaac6" in completed.stdout
    assert "simulator.active_gpu=3" in completed.stdout
    assert "simulator.physics_gpu=3" in completed.stdout
    assert "simulator.cuda_device=3" in completed.stdout
    assert "CUDA_VISIBLE_DEVICES=3" not in completed.stdout
    assert "launcher.py" in completed.stdout


def test_conda_activation_and_runner_require_canonical_curobo_v2():
    activation = ACTIVATE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert "InternDataAssets/curobov2" in activation
    assert "InternDataAssets/curobo\n" not in activation
    assert "verify_isaac6_env.py" in runner
    assert "CONDA_PREFLIGHT_ONLY" in runner


def test_workspace_validation_uses_one_backend_and_existing_probe_template():
    validator = _load_module("workspace_backend_validator", WORKSPACE_VALIDATOR)

    assert validator._runner_command("docker") == [
        "bash",
        "scripts/docker/up_simbox_isaac.sh",
    ]
    assert validator._runner_command("conda") == [
        "bash",
        "scripts/simbox/run_simbox_task.sh",
    ]
    assert WORKSPACE_TEMPLATE.is_file()
    source = WORKSPACE_VALIDATOR.read_text(encoding="utf-8")
    assert "configs/de_workspace_probe_template.yaml" not in source
    assert source.count('"LAUNCH_TEMPLATE": "configs/simbox/de_workspace_probe_template.yaml"') == 2
    assert source.count("_runner_command(simulator_backend),") == 3


def test_curobo_dependency_filter_preserves_isaac_owned_packages(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
dependencies = ["numpy>=1.26", "torch>=2.0", "warp-lang>=1.0"]

[project.optional-dependencies]
cu12 = [
  "nvidia-curobo==0.8.0",
  "cuda-toolkit>=12.0",
  "nvidia-cuda-nvcc-cu12>=12.0",
]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    requirements = _load_resolver().resolve_requirements(pyproject)

    assert "numpy>=1.26" in requirements
    assert "cuda-toolkit==12.8.1" in requirements
    assert "nvidia-cuda-nvcc-cu12==12.8.93" in requirements
    assert "setuptools<82" in requirements
    assert not any(value.startswith("torch") for value in requirements)
    assert not any("warp-lang" in value for value in requirements)
    assert not any("nvidia-curobo" in value for value in requirements)


def test_cuda_home_assembly_matches_runtime_package_layout(tmp_path, monkeypatch):
    roots = {}
    for distribution, relative in (
        ("nvidia-cuda-runtime-cu12", "nvidia/cuda_runtime"),
        ("nvidia-cuda-nvrtc-cu12", "nvidia/cuda_nvrtc"),
        ("nvidia-cuda-nvcc-cu12", "nvidia/cuda_nvcc"),
    ):
        root = tmp_path / distribution
        roots[(distribution, relative)] = root
        root.mkdir(parents=True)

    runtime = roots[("nvidia-cuda-runtime-cu12", "nvidia/cuda_runtime")]
    nvrtc = roots[("nvidia-cuda-nvrtc-cu12", "nvidia/cuda_nvrtc")]
    nvcc = roots[("nvidia-cuda-nvcc-cu12", "nvidia/cuda_nvcc")]
    for path in (
        runtime / "include" / "cuda.h",
        runtime / "lib" / "libcudart.so",
        nvrtc / "include" / "nvrtc.h",
        nvrtc / "lib" / "libnvrtc.so",
        nvcc / "include" / "crt" / "host_config.h",
        nvcc / "bin" / "ptxas",
        nvcc / "nvvm" / "libdevice" / "libdevice.10.bc",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")

    module = _load_module("prepare_cuda_home", CUDA_PREPARE_PATH)
    monkeypatch.setattr(
        module,
        "_distribution_path",
        lambda distribution, relative: roots[(distribution, relative)],
    )

    cuda_home = module.prepare(tmp_path / "env")
    assert (cuda_home / "include" / "cuda.h").is_file()
    assert (cuda_home / "include" / "nvrtc.h").is_file()
    assert (cuda_home / "include" / "crt" / "host_config.h").is_file()
    assert (cuda_home / "bin" / "ptxas").resolve() == (
        nvcc / "bin" / "ptxas"
    ).resolve()
    assert (cuda_home / "lib" / "libcudart.so").resolve() == (
        runtime / "lib" / "libcudart.so"
    ).resolve()
    assert module.prepare(tmp_path / "env") == cuda_home
