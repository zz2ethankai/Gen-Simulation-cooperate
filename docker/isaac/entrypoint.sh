#!/usr/bin/env bash
set -e

export ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/isaac-sim}"
export CUROBO_PATH="${CUROBO_PATH:-/opt/curobo}"
export INTERNDATA_RUN_IMPORT_CHECKS="${INTERNDATA_RUN_IMPORT_CHECKS:-0}"
export INTERNDATA_AUTOSTART_LAUNCHER="${INTERNDATA_AUTOSTART_LAUNCHER:-0}"
export INTERNDATA_LAUNCHER_CONFIG="${INTERNDATA_LAUNCHER_CONFIG:-configs/de_plan_with_render_template.yaml}"
export INTERNDATA_LAUNCHER_EXTRA_ARGS="${INTERNDATA_LAUNCHER_EXTRA_ARGS:-}"
export INTERNDATA_LAUNCHER_ARGS_JSON="${INTERNDATA_LAUNCHER_ARGS_JSON:-[]}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.7.0}"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUROBO="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUROBO:-${SETUPTOOLS_SCM_PRETEND_VERSION}}"

for lib_dir in \
  "${ISAAC_SIM_PATH}/exts/omni.isaac.ml_archive/pip_prebundle/torch/lib" \
  "${ISAAC_SIM_PATH}/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cuda_runtime/lib" \
  "${ISAAC_SIM_PATH}/kit/exts/omni.cuda.libs/bin"
do
  if [ -d "${lib_dir}" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${lib_dir}"
  fi
done

g1_wbc_env_file="/workspace/docker/isaac/env.d/g1_wbc_onnxruntime.sh"
if [ -f "${g1_wbc_env_file}" ]; then
  source "${g1_wbc_env_file}"
fi
unset g1_wbc_env_file

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

if [ -d /workspace ]; then
  if [ -d "${CUROBO_PATH}" ] && [ ! -e /workspace/InternDataAssets/curobo ]; then
    mkdir -p /workspace/InternDataAssets
    ln -s "${CUROBO_PATH}" /workspace/InternDataAssets/curobo
  fi

  if [ -d /workspace/workflows/simbox ] && [ ! -e /workspace/workflows/simbox/curobo ]; then
    ln -s ../../InternDataAssets/curobo /workspace/workflows/simbox/curobo
  fi
fi

if [ -d "${CUROBO_PATH}/src" ]; then
  export PYTHONPATH="${CUROBO_PATH}/src:${PYTHONPATH}"
fi

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
    "yaml",
    "sklearn",
    "transforms3d",
    "onnxruntime",
    "curobo",
    "curobo.curobolib.lbfgs_step_cu",
    "curobo.curobolib.kinematics_fused_cu",
    "curobo.curobolib.line_search_cu",
    "curobo.curobolib.tensor_step_cu",
    "curobo.curobolib.geom_cu",
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
