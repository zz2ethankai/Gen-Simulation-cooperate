#!/usr/bin/env bash
# Download the public assets required by the Unitree G1 navigation example.

set -euo pipefail

readonly UNITREE_REV="394cf2448f8a9ed815c77c701a761f3d1ff1c8fb"
readonly WBC_REV="a0732b642c0333077e127a2f56ab0014c196bca4"
readonly WBC_REPO="https://github.com/NVlabs/GR00T-WholeBodyControl.git"
readonly WBC_MODEL_DIR="decoupled_wbc/sim2mujoco/resources/robots/g1/policy"
readonly SCRATCH_ROOT="tmp/g1_install"
readonly UNITREE_ARCHIVE="${SCRATCH_ROOT}/downloads/unitree_sim_isaaclab_assets.zip"
readonly UNITREE_EXTRACT="${SCRATCH_ROOT}/unitree_extract"
readonly WBC_SOURCE="${SCRATCH_ROOT}/sources/GR00T-WholeBodyControl"
readonly ASSET_ROOT="InternDataAssets/assets/unitree_g1_sonic"

for command_name in curl 7z rsync git; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing required command: ${command_name}" >&2
        exit 1
    fi
done
if ! git lfs version >/dev/null 2>&1; then
    echo "Git LFS is required. Install it and run: git lfs install" >&2
    exit 1
fi

mkdir -p "${SCRATCH_ROOT}/downloads" "${SCRATCH_ROOT}/sources"
mkdir -p "${UNITREE_EXTRACT}" "${ASSET_ROOT}/g1" "${ASSET_ROOT}/wbc"

curl -fL --retry 20 -C - \
    -o "${UNITREE_ARCHIVE}" \
    "https://hf-mirror.com/datasets/unitreerobotics/unitree_sim_isaaclab_usds/resolve/${UNITREE_REV}/assets.zip"

7z x -y "-o${UNITREE_EXTRACT}" "${UNITREE_ARCHIVE}" \
    "assets/robots/g1-29dof_wholebody_dex3/*"
rsync -a --exclude=.thumbs --exclude=.asset_hash \
    "${UNITREE_EXTRACT}/assets/robots/g1-29dof_wholebody_dex3/" \
    "${ASSET_ROOT}/g1/"

if [[ ! -d "${WBC_SOURCE}/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none "${WBC_REPO}" "${WBC_SOURCE}"
fi
git -C "${WBC_SOURCE}" fetch origin "${WBC_REV}"
GIT_LFS_SKIP_SMUDGE=1 git -C "${WBC_SOURCE}" checkout --detach "${WBC_REV}"
git -C "${WBC_SOURCE}" lfs pull \
    --include="${WBC_MODEL_DIR}/GR00T-WholeBodyControl-Balance.onnx,${WBC_MODEL_DIR}/GR00T-WholeBodyControl-Walk.onnx" \
    --exclude=""

for model_name in GR00T-WholeBodyControl-Balance.onnx GR00T-WholeBodyControl-Walk.onnx; do
    source_path="${WBC_SOURCE}/${WBC_MODEL_DIR}/${model_name}"
    if [[ ! -s "${source_path}" ]]; then
        echo "Downloaded WBC model is missing or empty: ${source_path}" >&2
        exit 1
    fi
    cp "${source_path}" "${ASSET_ROOT}/wbc/${model_name}"
done

echo "Unitree G1 assets installed under ${ASSET_ROOT}/g1/"
echo "NVIDIA decoupled-WBC models installed under ${ASSET_ROOT}/wbc/"
