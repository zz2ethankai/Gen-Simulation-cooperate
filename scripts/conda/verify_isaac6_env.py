#!/usr/bin/env python3
"""Verify the native Isaac Sim 6.0.1 and CuRobo v2 runtime contract."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from packaging.utils import canonicalize_name


EXPECTED_ISAAC_VERSION = "6.0.1.0"
EXPECTED_TORCH_VERSION = "2.11.0"
EXPECTED_CUDA_VERSION = "12.8"
EXPECTED_CUROBO_VERSION = "0.8.0"
EXPECTED_CUROBO_COMMIT = "4ea77366ca48ee453e7df139e39fa6532af49f3b"


def _normalized_version(value: str) -> str:
    return value.removeprefix("v").split("+", 1)[0].removesuffix("-no-tag")


def _distribution_locations(name: str) -> list[str]:
    normalized = canonicalize_name(name)
    return [
        str(distribution.locate_file(""))
        for distribution in metadata.distributions()
        if canonicalize_name(distribution.metadata.get("Name") or "") == normalized
    ]


def _source_commit(source_root: Path) -> str:
    marker = source_root / ".curobo_commit"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"CuRobo source has neither .curobo_commit nor a readable Git HEAD: "
            f"{source_root}"
        )
    return completed.stdout.strip()


def verify(repo_root: Path) -> dict[str, str]:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Isaac Sim 6.0.1 Conda runtime requires Python 3.12, got "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )

    source_root = (repo_root / "InternDataAssets" / "curobov2").resolve()
    configured_root = Path(os.environ.get("CUROBO_PATH", source_root)).resolve()
    python_root = Path(os.environ.get("CUROBO_PYTHON_PATH", source_root)).resolve()
    if configured_root != source_root or python_root != source_root:
        raise RuntimeError(
            "CUROBO_PATH and CUROBO_PYTHON_PATH must resolve to "
            f"{source_root}"
        )
    if not (source_root / "curobo" / "__init__.py").is_file():
        raise RuntimeError(f"CuRobo v2 checkout is missing: {source_root}")

    isaac_version = metadata.version("isaacsim")
    if isaac_version != EXPECTED_ISAAC_VERSION:
        raise RuntimeError(
            f"Isaac Sim version mismatch: expected {EXPECTED_ISAAC_VERSION}, "
            f"got {isaac_version}"
        )

    torch = importlib.import_module("torch")
    torch_version = _normalized_version(str(torch.__version__))
    cuda_version = str(torch.version.cuda)
    if torch_version != EXPECTED_TORCH_VERSION or cuda_version != EXPECTED_CUDA_VERSION:
        raise RuntimeError(
            "Torch/CUDA mismatch: expected "
            f"{EXPECTED_TORCH_VERSION}/cu{EXPECTED_CUDA_VERSION.replace('.', '')}, "
            f"got {torch.__version__}/cuda-{cuda_version}"
        )

    if _distribution_locations("nvidia-curobo"):
        raise RuntimeError(
            "nvidia-curobo must not be installed; runtime imports must come from "
            f"{source_root}"
        )
    curobo = importlib.import_module("curobo")
    module_path = Path(curobo.__file__).resolve()
    package_root = (source_root / "curobo").resolve()
    if not module_path.is_relative_to(package_root):
        raise RuntimeError(
            f"CuRobo import escaped the pinned checkout: {module_path}"
        )
    duplicate_paths = []
    for path_entry in sys.path:
        if not path_entry:
            continue
        candidate = Path(path_entry) / "curobo" / "__init__.py"
        if candidate.is_file() and candidate.resolve() != module_path:
            duplicate_paths.append(str(candidate.resolve()))
    if duplicate_paths:
        raise RuntimeError(
            f"multiple CuRobo source paths are importable: {duplicate_paths}"
        )
    curobo_version = str(curobo.__version__)
    if _normalized_version(curobo_version) != EXPECTED_CUROBO_VERSION:
        raise RuntimeError(
            f"CuRobo version mismatch: expected {EXPECTED_CUROBO_VERSION}, "
            f"got {curobo_version}"
        )
    commit = _source_commit(source_root)
    if commit != EXPECTED_CUROBO_COMMIT:
        raise RuntimeError(
            f"CuRobo commit mismatch: expected {EXPECTED_CUROBO_COMMIT}, got {commit}"
        )

    cuda_home = Path(os.environ.get("CUDA_HOME", "")).resolve()
    required_cuda_entries = (
        cuda_home / "include" / "cuda.h",
        cuda_home / "include" / "nvrtc.h",
        cuda_home / "bin" / "ptxas",
    )
    if not os.environ.get("CUDA_HOME") or not all(
        path.exists() for path in required_cuda_entries
    ):
        raise RuntimeError(
            "CUDA_HOME is not the prepared InterData CUDA toolkit view; run "
            "scripts/conda/setup_isaac6_env.sh"
        )

    importlib.import_module("isaacsim")
    if importlib.util.find_spec("isaacsim.core.api") is None:
        raise RuntimeError("Isaac Sim core API package is unavailable")
    if importlib.util.find_spec("curobo.motion_planner") is None:
        raise RuntimeError("CuRobo motion_planner package is unavailable")

    return {
        "python": sys.version.split()[0],
        "isaacsim": isaac_version,
        "torch": str(torch.__version__),
        "cuda": cuda_version,
        "curobo": curobo_version,
        "curobo_commit": commit,
        "curobo_path": str(module_path),
        "cuda_home": str(cuda_home),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.repo_root.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
