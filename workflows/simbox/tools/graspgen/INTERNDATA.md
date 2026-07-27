# InterndataEngine integration

This vendored GraspGen checkout lives independently from the legacy
`workflows/simbox/tools/grasp` generator.

## Environment

Follow the upstream README in this directory. The tested Conda environment name
is `graspgen-interndata`:

```bash
conda create -n graspgen-interndata python=3.10 -y
conda run -n graspgen-interndata python -m pip install \
  torch==2.1.0 torchvision==0.16.0 torch-cluster torch-scatter \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
conda run -n graspgen-interndata python -m pip install -e \
  workflows/simbox/tools/graspgen
cd workflows/simbox/tools/graspgen
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9 ./install_pointnet.sh

# Required by the existing scripts/visualize_grasp_html.py viewer.
conda run -n graspgen-interndata python -m pip install open3d plotly
```

Download the upstream model repository separately:

```bash
git clone https://huggingface.co/adithyamurali/GraspGenModels
```

## Generate an annotation

Only the project robot configuration is required. The wrapper validates every
arm's CuRobo `base_link` and `ee_link` against the URDF referenced by that
configuration. It then automatically selects the closest available supported
checkpoint embodiment from the model directory.

```bash
conda run -n graspgen-interndata python \
  workflows/simbox/tools/graspgen/scripts/generate_interndata_grasps.py \
  --mesh path/to/Aligned_obj.obj \
  --robot-config workflows/simbox/core/configs/robots/split_aloha.yaml \
  --models-dir /path/to/GraspGenModels \
  --output path/to/Aligned_grasp_sparse_graspgen.npy \
  --unit m --count 256
```

The output is a `float32` `N x 17` array accepted by the existing pick skills.
To use a non-default filename, set the pick skill's existing `npy_name` option.
The adjacent `.npy.meta.json` records the source model, project robot config,
URDF paths, EE links, and conversion parameters.

The neural checkpoint remains embodiment-specific because that is part of the
trained model. The exported annotation is converted to the project's generic
GraspNet TCP convention; the existing robot runtime applies each robot's
`R_ee_graspnet`, `ee_axis`, `tcp_offset`, and gripper width limits.

Panda-family, Piper, R5A, and Genie parallel-jaw profiles use the official
Franka checkpoint by default. Robotiq profiles prefer the official Robotiq
2F-140 checkpoint. This makes the generated file executable by each project's
runtime profile, but it does not turn the official checkpoint into a model
trained specifically for Piper, R5A, or Genie.
