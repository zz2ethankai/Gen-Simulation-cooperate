#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-interndata-isaac6}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/conda/setup_isaac6_env.sh [--env NAME] [--dry-run]

Create or update the maintained Isaac Sim 6.0.1 / CuRobo v2 Conda environment.
The environment is separate from the lightweight host Agent environment.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --env)
      CONDA_ENV="${2:?--env requires a value}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'conda create -n %q python=3.12 -y  # skipped when the environment exists\n' "${CONDA_ENV}"
  printf 'conda run -n %q python -m pip install --upgrade pip packaging\n' "${CONDA_ENV}"
  printf 'conda run -n %q python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128\n' "${CONDA_ENV}"
  printf 'conda run -n %q python -m pip install %q --extra-index-url https://pypi.nvidia.com\n' "${CONDA_ENV}" 'isaacsim[all,extscache]==6.0.1.0'
  printf 'conda run -n %q python -m pip install -r %q\n' "${CONDA_ENV}" "${REPO_ROOT}/docker/isaac/requirements.isaac.txt"
  printf '# Install filtered dependencies from InternDataAssets/curobov2/pyproject.toml\n'
  printf '# Prepare CUDA_HOME and run scripts/conda/verify_isaac6_env.py\n'
  exit 0
fi

command -v conda >/dev/null 2>&1 || {
  printf 'conda was not found.\n' >&2
  exit 2
}
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || {
  printf 'Isaac Sim 6.0.1 pip runtime requires Linux x86_64.\n' >&2
  exit 2
}
command -v ldd >/dev/null 2>&1 || {
  printf 'ldd was not found; Isaac Sim pip installation requires Linux glibc.\n' >&2
  exit 2
}
GLIBC_VERSION="$(ldd --version 2>&1 | head -n 1)"
GLIBC_VERSION="${GLIBC_VERSION##* }"
if [[ ! "${GLIBC_VERSION}" =~ ^([0-9]+)\.([0-9]+) ]] || \
   (( BASH_REMATCH[1] < 2 || (BASH_REMATCH[1] == 2 && BASH_REMATCH[2] < 35) )); then
  printf 'Isaac Sim 6.0.1 pip runtime requires glibc >= 2.35; detected %s.\n' "${GLIBC_VERSION}" >&2
  exit 2
fi
CUROBO_PYPROJECT="${REPO_ROOT}/InternDataAssets/curobov2/pyproject.toml"
[[ -f "${CUROBO_PYPROJECT}" ]] || {
  printf 'CuRobo v2 checkout is missing: %s\n' "${CUROBO_PYPROJECT}" >&2
  exit 2
}
CUROBO_ROOT="${REPO_ROOT}/InternDataAssets/curobov2"
EXPECTED_CUROBO_COMMIT=4ea77366ca48ee453e7df139e39fa6532af49f3b
if [[ -f "${CUROBO_ROOT}/.curobo_commit" ]]; then
  ACTUAL_CUROBO_COMMIT="$(tr -d '[:space:]' < "${CUROBO_ROOT}/.curobo_commit")"
else
  ACTUAL_CUROBO_COMMIT="$(git -C "${CUROBO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
fi
[[ "${ACTUAL_CUROBO_COMMIT}" == "${EXPECTED_CUROBO_COMMIT}" ]] || {
  printf 'CuRobo commit mismatch: expected %s, got %s\n' "${EXPECTED_CUROBO_COMMIT}" "${ACTUAL_CUROBO_COMMIT:-missing}" >&2
  exit 2
}

if ! conda run -n "${CONDA_ENV}" python -c 'import sys' >/dev/null 2>&1; then
  conda create -n "${CONDA_ENV}" python=3.12 -y
fi
CONDA_RUN=(conda run --no-capture-output -n "${CONDA_ENV}")
"${CONDA_RUN[@]}" python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
"${CONDA_RUN[@]}" python -m pip install --upgrade pip packaging
"${CONDA_RUN[@]}" python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
"${CONDA_RUN[@]}" python -m pip install 'isaacsim[all,extscache]==6.0.1.0' --extra-index-url https://pypi.nvidia.com
"${CONDA_RUN[@]}" python -m pip install -r "${REPO_ROOT}/docker/isaac/requirements.isaac.txt"

mapfile -t CUROBO_REQUIREMENTS < <(
  "${CONDA_RUN[@]}" python "${SCRIPT_DIR}/resolve_curobo_requirements.py" "${CUROBO_PYPROJECT}"
)
[[ "${#CUROBO_REQUIREMENTS[@]}" -gt 0 ]] || {
  printf 'No CuRobo dependencies were resolved from %s\n' "${CUROBO_PYPROJECT}" >&2
  exit 2
}
"${CONDA_RUN[@]}" python -m pip install "${CUROBO_REQUIREMENTS[@]}"

CONDA_PREFIX_PATH="$("${CONDA_RUN[@]}" python -c 'import sys; print(sys.prefix)')"
"${CONDA_RUN[@]}" python "${SCRIPT_DIR}/prepare_cuda_home.py" --prefix "${CONDA_PREFIX_PATH}"

export CONDA_ENV
set +u
source "${SCRIPT_DIR}/activate_isaac6_env.sh"
set -u
python "${SCRIPT_DIR}/verify_isaac6_env.py" --repo-root "${REPO_ROOT}"

printf 'Conda runtime is ready. Activate it with:\n'
printf '  CONDA_ENV=%q source scripts/conda/activate_isaac6_env.sh\n' "${CONDA_ENV}"
