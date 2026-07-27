# InterndataEngine integration

This vendored GraspGenX checkout is independent of both the legacy
`workflows/simbox/tools/grasp/` generator and the earlier
`workflows/simbox/tools/graspgen/` adapter.  It is an offline mesh-to-annotation
tool; the runtime Pick path still loads an `N x 17` NumPy file through the
existing `npy_name` option.

## Environment

The tested environment is `graspgenx-interndata`:

```bash
conda create -n graspgenx-interndata python=3.10 -y
conda run -n graspgenx-interndata python -m pip install \
  torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
conda run -n graspgenx-interndata python -m pip install -e \
  workflows/simbox/tools/graspgenx

# Existing project HTML grasp viewer.  The chardet pin matches requests 2.28.
conda run -n graspgenx-interndata python -m pip install \
  open3d plotly chardet==5.2.0
```

The validated versions are Python 3.10, PyTorch 2.1.0+cu121, torchvision
0.16.0+cu121, and NumPy 1.26.4.  The base adapter uses GraspGenX's pure-PyTorch
`ptv3vanilla` backbone and does not require the optional end-to-end
cuRobo/Newton/MuJoCo stack.

## External models and gripper descriptions

Keep the large repositories outside this vendored source directory:

```bash
git clone --depth 1 \
  https://huggingface.co/adithyamurali/GraspGenXModel \
  /path/to/GraspGenXModels

git clone --depth 1 \
  https://huggingface.co/datasets/adithyamurali/gripper_descriptions \
  /path/to/GraspGenXGripperDescriptions
```

The project CLI disables GraspGenX's import-time automatic download and
requires both paths explicitly.  This prevents an import or unit test from
silently placing multi-gigabyte data under the source tree.

## Generate an annotation

Only the project robot configuration selects the normal gripper descriptor;
the pick task does not specify a gripper:

```bash
conda run -n graspgenx-interndata python \
  workflows/simbox/tools/graspgenx/scripts/generate_interndata_grasps.py \
  --mesh path/to/Aligned_obj.obj \
  --robot-config workflows/simbox/core/configs/robots/panda_omron.yaml \
  --models-dir /path/to/GraspGenXModels \
  --gripper-descriptions-dir /path/to/GraspGenXGripperDescriptions \
  --output path/to/Aligned_grasp_sparse_graspgenx.npy \
  --unit m --count 256
```

Default mappings are:

| Project robot | GraspGenX descriptor |
|---|---|
| PandaOmron, PandaOmronVirtual, FR3, Tracer2Franka | `franka_panda` |
| SplitAloha, SplitAlohaActual | `piper_hand` |
| Lift2 | `arx_x5` |
| Genie1 | `galaxea_g1` |
| FrankaRobotiq85 | `robotiq_2f_85` |

Future robots can be added centrally in the adapter.  A robot YAML may also
opt into an explicit override with `graspgenx.gripper`, but ordinary task YAML
does not need or accept a gripper selector.

The wrapper validates each configured CuRobo `base_link` and `ee_link` against
the URDF referenced by the current project robot config.  GraspGenX's canonical
pose (`+X` closing, `+Z` approach) is converted to the project's generic
GraspNet TCP convention, and the descriptor's full `tool_tcp_transform` is
used for the stored TCP center.  Runtime then applies the selected robot's
existing `R_ee_graspnet`, `ee_axis`, `tcp_offset`, and width limits.

The output and adjacent metadata file are:

```text
Aligned_grasp_sparse_graspgenx.npy
Aligned_grasp_sparse_graspgenx.npy.meta.json
```

The array is strictly `(256, 17) float32` by default.  To use it without
overwriting a legacy annotation, set the pick skill's existing `npy_name` to
the new filename.

## Visualize

```bash
conda run -n graspgenx-interndata python scripts/visualize_grasp_html.py \
  --obj-path path/to/Aligned_obj.obj \
  --grasp-path path/to/Aligned_grasp_sparse_graspgenx.npy \
  --output output/grasp_visualizations/graspgenx/example.html \
  --unit m --count 256
```

Green is a better (lower) project score and red is a worse score within the
displayed set.  These checks validate the file contract, coordinate conversion,
and model inference; they are not evidence of physical grasp success.
