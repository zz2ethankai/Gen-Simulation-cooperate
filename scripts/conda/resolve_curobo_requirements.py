#!/usr/bin/env python3
"""Print the CuRobo v2 dependencies allowed in the Isaac Sim runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10/3.9 developer test environments
    import tomli as tomllib


FORBIDDEN = {"nvidia-curobo", "torch", "warp-lang"}
PINNED = {
    "cuda-toolkit": "cuda-toolkit==12.8.1",
    "nvidia-cuda-nvcc-cu12": "nvidia-cuda-nvcc-cu12==12.8.93",
    "setuptools": "setuptools<82",
}


def resolve_requirements(pyproject_path: Path) -> list[str]:
    metadata = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = metadata.get("project", {})
    optional = project.get("optional-dependencies", {})
    raw = [
        *project.get("dependencies", []),
        *optional.get("cu12", []),
        "importlib-resources>=6.0",
        *PINNED.values(),
    ]
    resolved: dict[str, str] = {}
    for value in raw:
        requirement = Requirement(str(value))
        name = canonicalize_name(requirement.name)
        if name in FORBIDDEN:
            continue
        resolved[name] = PINNED.get(name, str(requirement))
    return [resolved[name] for name in sorted(resolved)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pyproject", type=Path)
    args = parser.parse_args()
    for requirement in resolve_requirements(args.pyproject.resolve()):
        print(requirement)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
