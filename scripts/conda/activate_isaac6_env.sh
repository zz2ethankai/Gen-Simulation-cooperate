#!/usr/bin/env bash
# Source this file to activate the maintained Isaac Sim 6 Conda runtime.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Source this script instead of executing it: source %s\n' "$0" >&2
  exit 2
fi

_interndata_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_interndata_repo_root="$(cd "${_interndata_script_dir}/../.." && pwd)"
export CONDA_ENV="${CONDA_ENV:-interndata-isaac6}"

if ! command -v conda >/dev/null 2>&1; then
  if [[ -x /opt/anaconda3/bin/conda ]]; then
    _interndata_conda_bin=/opt/anaconda3/bin/conda
  else
    printf 'conda was not found.\n' >&2
    return 2
  fi
else
  _interndata_conda_bin="$(command -v conda)"
fi

_interndata_conda_base="$("${_interndata_conda_bin}" info --base)"
set +u
source "${_interndata_conda_base}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

export CUROBO_PATH="${_interndata_repo_root}/InternDataAssets/curobov2"
export CUROBO_PYTHON_PATH="${CUROBO_PATH}"
export CUROBO_EXPECTED_VERSION="${CUROBO_EXPECTED_VERSION:-0.8.0}"
export CUROBO_EXPECTED_COMMIT="${CUROBO_EXPECTED_COMMIT:-4ea77366ca48ee453e7df139e39fa6532af49f3b}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.8.0}"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUROBO="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUROBO:-${SETUPTOOLS_SCM_PRETEND_VERSION}}"
export CUDA_HOME="${CONDA_PREFIX}/.interndata/isaac-cuda"
export CUDA_PATH="${CUDA_HOME}"
export PYTHONPATH="${CUROBO_PYTHON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${CUDA_HOME}/lib"

unset _interndata_script_dir _interndata_repo_root _interndata_conda_bin _interndata_conda_base
