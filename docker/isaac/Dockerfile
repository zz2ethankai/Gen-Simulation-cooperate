ARG ISAAC_SIM_VERSION=6.0.1
ARG CUROBO_VERSION=0.8.0
ARG CUROBO_COMMIT=4ea77366ca48ee453e7df139e39fa6532af49f3b

FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}

ARG CUROBO_VERSION
ARG CUROBO_COMMIT

ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ISAAC_SIM_PATH=/isaac-sim
ENV CUROBO_PATH=/workspace/InternDataAssets/curobov2
ENV CUROBO_PYTHON_PATH=/workspace/InternDataAssets/curobov2
ENV CUROBO_EXPECTED_VERSION=${CUROBO_VERSION}
ENV CUROBO_EXPECTED_COMMIT=${CUROBO_COMMIT}
ENV ISAAC_SIM_TORCH_PATH=/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle
ENV PYTHONPATH=/workspace/InternDataAssets/curobov2:/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:/isaac-sim/extsDeprecated/isaacsim.core.api:/isaac-sim/extsDeprecated/isaacsim.core.prims:/isaac-sim/extsDeprecated/isaacsim.core.utils:/isaac-sim/extsDeprecated/isaacsim.sensors.camera
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV PIP_DEFAULT_TIMEOUT=60

SHELL ["/bin/bash", "-lc"]

# Isaac Sim 6 containers are intended to run as a non-root user. Use root only
# while assembling the image, then drop back to the documented numeric uid.
USER root

RUN if [ -f /etc/apt/sources.list ]; then \
      sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g; s|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g; s|https://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g; s|https://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list; \
    fi && \
    find /etc/apt/sources.list.d -type f \( -name '*.list' -o -name '*.sources' \) -print0 | xargs -0 -r sed -i \
      -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
      -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
      -e 's|https://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
      -e 's|https://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' && \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git git-lfs \
      build-essential cmake pkg-config \
      libatomic1 libegl1 libglu1-mesa libgomp1 libxi6 libxrandr2 libxt6 \
      libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
      libfreetype-dev libfontconfig1 \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      ffmpeg \
      && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.pip && \
    cat > /etc/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

COPY docker/isaac/requirements.isaac.txt /tmp/requirements.isaac.txt
COPY InternDataAssets/curobov2/pyproject.toml /tmp/curobov2.pyproject.toml

RUN /isaac-sim/python.sh -m pip install --no-cache-dir -r /tmp/requirements.isaac.txt

# Install the dependencies declared by the pinned CuRobo v0.8 source tree, but
# deliberately do not install nvidia-curobo itself.  Runtime imports come from
# the single read-only checkout mounted at ${CUROBO_PYTHON_PATH}.  Selecting
# the cu12 extra (not cu12-torch) preserves Isaac Sim's bundled Torch as the
# only Torch distribution in the image.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
      --index-url https://pypi.org/simple \
      "importlib-resources>=6.0"
RUN /isaac-sim/python.sh - <<'PY'
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

metadata = tomllib.loads(Path("/tmp/curobov2.pyproject.toml").read_text())
requirements = [
    *metadata["project"]["dependencies"],
    *metadata["project"]["optional-dependencies"]["cu12"],
    # Match Isaac's Torch 2.11 + cu128 runtime and NVRTC headers.  Without
    # these constraints cuda-core's broad cu12 extra selects CUDA 12.9.
    "cuda-toolkit==12.8.1",
    "nvidia-cuda-nvcc-cu12==12.8.93",
    "setuptools<82",
]
install_requirements = []
for requirement in requirements:
    name = canonicalize_name(Requirement(requirement).name)
    if name == "warp-lang":
        # Isaac Sim loads omni.warp.core during Kit startup.  Do not install a
        # second PyPI Warp wheel that could shadow the Isaac-provided Warp
        # runtime; the maintained CuRobo fork uses its public top-level API.
        continue
    if name in {"torch", "nvidia-curobo"}:
        raise RuntimeError(f"forbidden CuRobo image dependency: {requirement}")
    install_requirements.append(requirement)

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        *install_requirements,
    ]
)
PY

# Fail the build if any CuRobo package copy or pip-provided Torch appeared.
RUN /isaac-sim/python.sh - <<'PY'
import importlib.metadata as metadata
import importlib.util
from pathlib import Path

import torch
from packaging.version import Version

def distributions(name):
    normalized = name.lower().replace("_", "-")
    return [
        dist
        for dist in metadata.distributions()
        if (dist.metadata.get("Name") or "").lower().replace("_", "-") == normalized
    ]

if (
    Path("/opt/curobo").exists()
    or importlib.util.find_spec("curobo") is not None
    or distributions("nvidia-curobo")
):
    raise RuntimeError("CuRobo must not be installed in the image or site-packages")

torch_distributions = distributions("torch")
torch_root = Path("/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle").resolve()
torch_file = Path(torch.__file__).resolve()
if (
    len(torch_distributions) != 1
    or not torch_file.is_relative_to(torch_root)
    or torch.__version__ != "2.11.0+cu128"
    or torch.version.cuda != "12.8"
):
    raise RuntimeError(
        f"expected one Isaac-bundled Torch, found {len(torch_distributions)} at {torch_file}"
    )
if metadata.version("cuda-toolkit") != "12.8.1":
    raise RuntimeError(f"unexpected CUDA toolkit: {metadata.version('cuda-toolkit')}")
if metadata.version("nvidia-cuda-nvcc-cu12") != "12.8.93":
    raise RuntimeError(f"unexpected CUDA NVCC package: {metadata.version('nvidia-cuda-nvcc-cu12')}")
if Version(metadata.version("setuptools")) >= Version("82"):
    raise RuntimeError(f"setuptools is incompatible with Isaac Torch: {metadata.version('setuptools')}")
print(
    "verified image packages: no CuRobo distribution; Isaac Torch",
    torch.__version__,
    torch_file,
    "CUDA toolkit",
    metadata.version("cuda-toolkit"),
    "NVCC",
    metadata.version("nvidia-cuda-nvcc-cu12"),
)
PY

# Isaac Sim keeps the CUDA headers and runtime libraries in its ML prebundle
# instead of /usr/local/cuda.  CuRobo v0.8's cuda.core backend discovers the
# NVRTC include directory through CUDA_HOME, so expose a small merged toolkit
# view made exclusively from those Isaac-supplied files.  No compiler or Torch
# wheel is added here; the nvcc package only contributes the CRT headers and
# ptxas that cuda.core may use for runtime compilation.
ENV CUDA_HOME=/opt/isaac-cuda
ENV CUDA_PATH=/opt/isaac-cuda
RUN mkdir -p "${CUDA_HOME}/include" "${CUDA_HOME}/lib" "${CUDA_HOME}/bin" && \
    cp -a "${ISAAC_SIM_TORCH_PATH}/nvidia/cuda_runtime/include/." "${CUDA_HOME}/include/" && \
    cp -a "${ISAAC_SIM_TORCH_PATH}/nvidia/cuda_nvrtc/include/." "${CUDA_HOME}/include/" && \
    cp -a \
      /isaac-sim/kit/python/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt \
      "${CUDA_HOME}/include/" && \
    ln -s \
      /isaac-sim/kit/python/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas \
      "${CUDA_HOME}/bin/ptxas" && \
    ln -s \
      /isaac-sim/kit/python/lib/python3.12/site-packages/nvidia/cuda_nvcc/nvvm \
      "${CUDA_HOME}/nvvm" && \
    for lib_dir in \
      "${ISAAC_SIM_TORCH_PATH}/nvidia/cuda_nvrtc/lib" \
      "${ISAAC_SIM_TORCH_PATH}/nvidia/cuda_runtime/lib"; do \
      for lib in "${lib_dir}"/*.so*; do \
        [ -e "${lib}" ] || continue; \
        ln -s "${lib}" "${CUDA_HOME}/lib/$(basename "${lib}")"; \
      done; \
    done

COPY docker/isaac/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    # The host UID is injected by docker-compose so that mounted workspace
    # outputs remain writable.  Isaac's base image leaves /isaac-sim at 750
    # for its baked-in uid 1234, which prevents another valid host uid from
    # even reaching python.sh.
    chmod 755 /isaac-sim /isaac-sim/python.sh /isaac-sim/runheadless.sh /isaac-sim/isaac-sim.sh && \
    mkdir -p /isaac-sim/.cache /isaac-sim/.nv/ComputeCache \
      /isaac-sim/.nvidia-omniverse/logs /isaac-sim/.nvidia-omniverse/config \
      /isaac-sim/.local/share/ov/data /isaac-sim/.local/share/ov/pkg && \
    chown -R 1234:1234 /entrypoint.sh /isaac-sim/.cache \
      /isaac-sim/.nv /isaac-sim/.nvidia-omniverse /isaac-sim/.local

WORKDIR /workspace
USER 1234:1234
