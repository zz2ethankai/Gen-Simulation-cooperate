#!/usr/bin/env bash

# CUDA libraries used by the G1 decoupled-WBC ONNX Runtime sessions.
for g1_wbc_cuda_lib_dir in \
  "${ISAAC_SIM_PATH}/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cublas/lib" \
  "${ISAAC_SIM_PATH}/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cudnn/lib" \
  "${ISAAC_SIM_PATH}/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/curand/lib" \
  "${ISAAC_SIM_PATH}/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cufft/lib" \
  "${ISAAC_SIM_PATH}/kit/testlibs"
do
  if [ -d "${g1_wbc_cuda_lib_dir}" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${g1_wbc_cuda_lib_dir}"
  fi
done

unset g1_wbc_cuda_lib_dir
