#!/usr/bin/env bash
set -euo pipefail

export ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/isaac-sim}"
expected_curobo_path="/workspace/InternDataAssets/curobov2"
expected_curobo_python_path="${expected_curobo_path}"
if [ "${CUROBO_PATH:-${expected_curobo_path}}" != "${expected_curobo_path}" ]; then
  echo "[isaac] Error: CUROBO_PATH must be ${expected_curobo_path}" >&2
  exit 1
fi
if [ "${CUROBO_PYTHON_PATH:-${expected_curobo_python_path}}" != "${expected_curobo_python_path}" ]; then
  echo "[isaac] Error: CUROBO_PYTHON_PATH must be ${expected_curobo_python_path}" >&2
  exit 1
fi
export CUROBO_PATH="${expected_curobo_path}"
export CUROBO_PYTHON_PATH="${expected_curobo_python_path}"
export CUROBO_EXPECTED_VERSION="${CUROBO_EXPECTED_VERSION:-0.8.0}"
export CUROBO_EXPECTED_COMMIT="${CUROBO_EXPECTED_COMMIT:-4ea77366ca48ee453e7df139e39fa6532af49f3b}"
export ISAAC_SIM_TORCH_PATH="${ISAAC_SIM_TORCH_PATH:-${ISAAC_SIM_PATH}/extsDeprecated/omni.isaac.ml_archive/pip_prebundle}"
export HOME="${HOME:-/workspace}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${HOME}/.config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
export INTERNDATA_RUN_IMPORT_CHECKS="${INTERNDATA_RUN_IMPORT_CHECKS:-0}"
export INTERNDATA_AUTOSTART_LAUNCHER="${INTERNDATA_AUTOSTART_LAUNCHER:-0}"
export INTERNDATA_LAUNCHER_CONFIG="${INTERNDATA_LAUNCHER_CONFIG:-configs/de_plan_with_render_template.yaml}"
export INTERNDATA_LAUNCHER_EXTRA_ARGS="${INTERNDATA_LAUNCHER_EXTRA_ARGS:-}"
export INTERNDATA_LAUNCHER_ARGS_JSON="${INTERNDATA_LAUNCHER_ARGS_JSON:-[]}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.8.0}"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUROBO="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUROBO:-${SETUPTOOLS_SCM_PRETEND_VERSION}}"

if [ ! -f "${CUROBO_PYTHON_PATH}/curobo/__init__.py" ]; then
  echo "[isaac] Error: CuRobo v2 source is not mounted at ${CUROBO_PYTHON_PATH}" >&2
  exit 1
fi

# Keep the runtime import path deterministic. CuRobo has exactly one source
# entry, and Isaac's ML prebundle remains the only Torch provider.
export PYTHONPATH="${CUROBO_PYTHON_PATH}:${ISAAC_SIM_TORCH_PATH}:${ISAAC_SIM_PATH}/extsDeprecated/isaacsim.core.api:${ISAAC_SIM_PATH}/extsDeprecated/isaacsim.core.prims:${ISAAC_SIM_PATH}/extsDeprecated/isaacsim.core.utils:${ISAAC_SIM_PATH}/extsDeprecated/isaacsim.sensors.camera"

if [ -d "${CUDA_HOME:-}" ]; then
  export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
fi

for lib_dir in \
  "${CUDA_HOME:-}/lib" \
  "${ISAAC_SIM_TORCH_PATH}/nvidia/cuda_nvrtc/lib" \
  "${ISAAC_SIM_TORCH_PATH}/nvidia/cuda_runtime/lib" \
  "${ISAAC_SIM_TORCH_PATH}/torch/lib" \
  "${ISAAC_SIM_PATH}/kit/exts/omni.cuda.libs/bin"
do
  if [ -d "${lib_dir}" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${lib_dir}"
  fi
done

"${ISAAC_SIM_PATH}/python.sh" - <<'PY'
import importlib.metadata as metadata
import os
from pathlib import Path
import sys

from packaging.utils import canonicalize_name

import curobo
import torch

source_root = Path(os.environ["CUROBO_PATH"]).resolve(strict=True)
python_root = Path(os.environ["CUROBO_PYTHON_PATH"]).absolute()
module_lexical = Path(curobo.__file__).absolute()
module_real = module_lexical.resolve(strict=True)
package_real = (source_root / "curobo").resolve(strict=True)

if not module_lexical.is_relative_to(python_root / "curobo"):
    raise RuntimeError(f"CuRobo was not imported through {python_root}: {module_lexical}")
if not module_real.is_relative_to(package_real):
    raise RuntimeError(f"CuRobo realpath escaped the pinned checkout: {module_real}")

# The vendored source carries a commit marker so a clean outer-repository
# checkout does not need to preserve an inner Git directory.  Accept the
# existing immutable HEAD marker as a backward-compatible local-development
# fallback only when the vendored marker is absent.
commit_marker = source_root / ".curobo_commit"
git_head = source_root / ".git" / "HEAD"
if commit_marker.is_file():
    actual_commit = commit_marker.read_text().strip()
elif git_head.is_file():
    actual_commit = git_head.read_text().strip()
else:
    raise RuntimeError(f"CuRobo source has no pinned commit marker: {source_root}")
expected_commit = os.environ["CUROBO_EXPECTED_COMMIT"]
if actual_commit != expected_commit:
    raise RuntimeError(f"CuRobo commit mismatch: expected {expected_commit}, got {actual_commit}")

expected_version = os.environ["CUROBO_EXPECTED_VERSION"]
actual_version = str(curobo.__version__)
normalized_version = actual_version.removeprefix("v").split("+", 1)[0].removesuffix("-no-tag")
if normalized_version != expected_version:
    raise RuntimeError(
        f"CuRobo version mismatch: expected {expected_version}, got {actual_version}"
    )

def distributions(name):
    normalized = canonicalize_name(name)
    return [
        dist
        for dist in metadata.distributions()
        if canonicalize_name(dist.metadata.get("Name") or "") == normalized
    ]

curobo_distributions = distributions("nvidia-curobo")
if curobo_distributions:
    locations = [str(dist.locate_file("")) for dist in curobo_distributions]
    raise RuntimeError(f"site-packages contains a CuRobo distribution: {locations}")

other_curobo_paths = []
for path_entry in sys.path:
    if not path_entry:
        continue
    candidate = Path(path_entry) / "curobo" / "__init__.py"
    if candidate.is_file() and candidate.resolve() != module_real:
        other_curobo_paths.append(str(candidate.resolve()))
if other_curobo_paths:
    raise RuntimeError(f"multiple CuRobo package paths found: {other_curobo_paths}")

torch_root = Path(os.environ["ISAAC_SIM_TORCH_PATH"]).resolve(strict=True)
torch_file = Path(torch.__file__).resolve(strict=True)
torch_distributions = distributions("torch")
if (
    len(torch_distributions) != 1
    or not torch_file.is_relative_to(torch_root)
    or torch.__version__ != "2.11.0+cu128"
    or torch.version.cuda != "12.8"
):
    raise RuntimeError(
        f"expected one Isaac-bundled Torch, found {len(torch_distributions)} at {torch_file}"
    )

print(
    "[isaac] CuRobo source verified:",
    f"version={actual_version}",
    f"commit={actual_commit}",
    f"file={module_lexical}",
    f"realpath={module_real}",
)
print(
    "[isaac] Isaac Torch verified:",
    f"version={torch.__version__}",
    f"file={torch_file}",
    "distributions=1",
)
print("[isaac] CuRobo package import verified:", curobo.__version__)
PY

run_default_entry() {
  if [ "${INTERNDATA_AUTOSTART_LAUNCHER}" = "1" ]; then
    cd /workspace
    launcher_extra_args=()
    if [ "${INTERNDATA_LAUNCHER_ARGS_JSON}" != "[]" ]; then
      launcher_args_file="$(mktemp)"
      "${ISAAC_SIM_PATH}/python.sh" -c \
        'import json, sys; [print(value) for value in json.loads(sys.argv[1])]' \
        "${INTERNDATA_LAUNCHER_ARGS_JSON}" > "${launcher_args_file}"
      mapfile -t launcher_extra_args < "${launcher_args_file}"
      rm -f "${launcher_args_file}"
    elif [ -n "${INTERNDATA_LAUNCHER_EXTRA_ARGS}" ]; then
      read -r -a launcher_extra_args <<< "${INTERNDATA_LAUNCHER_EXTRA_ARGS}"
    fi
    exec "${ISAAC_SIM_PATH}/python.sh" launcher.py --config "${INTERNDATA_LAUNCHER_CONFIG}" "${launcher_extra_args[@]}"
  fi

  if [ "${LIVESTREAM:-0}" = "1" ]; then
    exec "${ISAAC_SIM_PATH}/isaac-sim.sh" \
      --no-window
  fi

  exec bash
}

if [ "${INTERNDATA_RUN_IMPORT_CHECKS}" = "1" ]; then
  echo "[isaac] Python version:"
  "${ISAAC_SIM_PATH}/python.sh" -c "import sys; print(sys.version)"

  echo "[isaac] Test imports..."
  "${ISAAC_SIM_PATH}/python.sh" - <<'PY'
mods = [
    "trimesh",
    "open3d",
    "cv2",
    "imageio",
    "plyfile",
    "omegaconf",
    "pydantic",
    "toml",
    "shapely",
    "ray",
    "pympler",
    "skimage",
    "lmdb",
    "numpy",
    "scipy",
    "torch",
    "yaml",
    "sklearn",
    "transforms3d",
    "curobo",
    "curobo.motion_planner",
    "curobo.types",
]
for module in mods:
    try:
        __import__(module)
        print("[ok]", module)
    except Exception as exc:
        print("[fail]", module, exc)

PY
fi
if [ "$#" -eq 0 ]; then
  run_default_entry
fi

exec "$@"
