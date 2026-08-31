#!/usr/bin/env python3
"""Assemble an environment-local CUDA_HOME from NVIDIA Python packages."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil


def _distribution_path(distribution: str, relative: str) -> Path:
    path = Path(metadata.distribution(distribution).locate_file(relative)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{distribution} does not provide {relative}: {path}")
    return path


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(
            f"refusing to replace existing CUDA_HOME entry: {destination}"
        )
    destination.symlink_to(source)


def prepare(prefix: Path) -> Path:
    root = prefix.resolve() / ".interndata" / "isaac-cuda"
    include = root / "include"
    library = root / "lib"
    binary = root / "bin"
    for directory in (include, library, binary):
        directory.mkdir(parents=True, exist_ok=True)

    runtime_root = _distribution_path(
        "nvidia-cuda-runtime-cu12", "nvidia/cuda_runtime"
    )
    nvrtc_root = _distribution_path("nvidia-cuda-nvrtc-cu12", "nvidia/cuda_nvrtc")
    nvcc_root = _distribution_path("nvidia-cuda-nvcc-cu12", "nvidia/cuda_nvcc")

    shutil.copytree(runtime_root / "include", include, dirs_exist_ok=True)
    shutil.copytree(nvrtc_root / "include", include, dirs_exist_ok=True)
    shutil.copytree(
        nvcc_root / "include" / "crt",
        include / "crt",
        dirs_exist_ok=True,
    )
    _link(nvcc_root / "bin" / "ptxas", binary / "ptxas")
    _link(nvcc_root / "nvvm", root / "nvvm")

    for source_root in (runtime_root / "lib", nvrtc_root / "lib"):
        for library_path in sorted(source_root.glob("*.so*")):
            _link(library_path, library / library_path.name)

    marker = {
        "cuda_home": str(root),
        "runtime_root": str(runtime_root),
        "nvrtc_root": str(nvrtc_root),
        "nvcc_root": str(nvcc_root),
    }
    (root / "interndata_cuda_home.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    args = parser.parse_args()
    print(prepare(args.prefix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
