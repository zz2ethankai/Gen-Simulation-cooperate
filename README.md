<div align="center">

# InternDataEngine: Pioneering High-Fidelity Synthetic Data Generator for Robotic Manipulation

</div>

<div align="center">

[![Paper InternData-A1](https://img.shields.io/badge/Paper-InternData--A1-red.svg)](https://arxiv.org/abs/2511.16651)
[![Paper Nimbus](https://img.shields.io/badge/Paper-Nimbus-red.svg)](https://arxiv.org/abs/2601.21449)
[![Paper InternVLA-M1](https://img.shields.io/badge/Paper-InternVLA--M1-red.svg)](https://arxiv.org/abs/2510.13778)
[![Data InternData-A1](https://img.shields.io/badge/Data-InternData--A1-blue?logo=huggingface)](https://huggingface.co/datasets/InternRobotics/InternData-A1)
[![Data InternData-M1](https://img.shields.io/badge/Data-InternData--M1-blue?logo=huggingface)](https://huggingface.co/datasets/InternRobotics/InternData-M1)
[![Docs](https://img.shields.io/badge/Docs-Online-green.svg)](https://internrobotics.github.io/InternDataEngine-Docs/)

</div>

## 💻 About

<div align="center">
  <img src="./docs/images/intern_data_engine.jpeg" alt="InternDataEngine Overview" width="80%">
</div>

InternDataEngine is a synthetic data generation engine for embodied AI that powers large-scale model training and iteration. Built on NVIDIA Isaac Sim, it unifies high-fidelity physical interaction from InternData-A1, semantic task and scene generation from InternData-M1, and high-throughput scheduling from the Nimbus framework to deliver realistic, task-aligned, and massively scalable robotic manipulation data.

- **More realistic physical interaction**: Unified simulation of rigid, articulated, deformable, and fluid objects across single-arm, dual-arm, and humanoid robots, enabling long-horizon, skill-composed manipulation that better supports sim-to-real transfer.
- **More diverse data generation**: By leveraging the internal state of the simulation engine to extract high-quality ground truth, coupled with multi-dimensional domain randomization (e.g., layout, texture, structure, and lighting), the data distribution is significantly expanded. This approach produces precise and diverse operational data, while simultaneously exporting rich multimodal annotations such as bounding boxes, segmentation masks, and keypoints.
- **More efficient large-scale production**: Nimbus-powered asynchronous pipelines that decouple planning, rendering, and storage, achieving 2–3× end-to-end throughput, cluster-level load balancing and fault tolerance for billion-scale data generation.

## 🔥 Latest News

- **[2026/03]** We release the InternDataEngine codebase v1.0, which includes the core modules: InternData-A1 and Nimbus.

## 🚀 Quickstart

The local Docker workflow depends on the full `InternDataAssets/` directory.
Install the repository in this order so the Docker build can find
`InternDataAssets/curobo` and SimBox can resolve its asset links.

### 1. System prerequisites

- Linux host with an NVIDIA GPU and a working NVIDIA driver
- Docker Engine with Compose v2
- NVIDIA Container Toolkit
- Python 3.10+ on the host for the asset download helper
- `7z` command line tool
- Enough disk space for the asset archive, extracted assets, Docker images, and
  Isaac Sim caches. The current extracted `InternDataAssets/` tree is about
  200 GB.

Quick checks:

```bash
nvidia-smi
docker compose version
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
7z
```

On Ubuntu, install missing host tools with:

```bash
sudo apt-get update
sudo apt-get install -y p7zip-full python3-pip
python3 -m pip install -U modelscope
```

### 2. Download assets

Download and extract the ModelScope split archive from the repository root:

```bash
python3 scripts/download_modelscope.py --token <MODEL_SCOPE_TOKEN>
```

The script downloads `MinMaxMex/InterndataAssets/InternDataAssets_7z`, extracts
`InternDataAssets/`, and creates the required SimBox relative symlinks:

```text
workflows/simbox/assets -> ../../InternDataAssets/assets
workflows/simbox/curobo -> ../../InternDataAssets/curobo
workflows/simbox/panda_drake -> ../../InternDataAssets/panda_drake
```

It refuses to overwrite an existing `InternDataAssets/` directory. If you need
to reinstall assets, move or remove the old directory first.

### 3. Verify the checkout

Run these checks before building Docker images:

```bash
test -d InternDataAssets/assets
test -d InternDataAssets/curobo
test -d InternDataAssets/panda_drake
test -L workflows/simbox/assets
test -L workflows/simbox/curobo
test -L workflows/simbox/panda_drake
```

If `docker/isaac/entrypoint.sh` is not executable on your checkout, fix it
before building:

```bash
chmod +x docker/isaac/entrypoint.sh docker/nav2/entrypoint.sh
```

### 4. Build and start

Build the split Isaac/Nav2 stack:

```bash
docker compose -f docker/docker-compose.yml build
```

Start the default single-GPU stack:

```bash
scripts/docker/up_nav2_stack_single_gpu.sh
```

Stop the stack:

```bash
scripts/docker/stop_all_docker.sh
```

中文项目内文档：[数据生成 README / Quick Start](./docs/data_generation/README.md)，包含单任务启动、配置分类、Docker 并行生成、资产替换和 SimBox skill 替换。

For more details, please check [Documentation](https://internrobotics.github.io/InternDataEngine-Docs/).

## Split ROS / Isaac Sim Deployment

The repository now includes an in-repo split deployment layout for running Isaac Sim and ROS/Nav2 as separate services:

- `docker/isaac/`: Isaac Sim image and runtime entrypoint
- `nav2/`: standalone Nav2 package containing container assets, ROS-side helpers, and shared client/runtime code

Prerequisites:

- Docker Engine with Compose v2
- NVIDIA Container Toolkit and a visible GPU on the host
- Enough local disk space for Isaac caches under `.docker/isaac-sim/`

Build the split stack from the repository root with:

```bash
docker compose -f docker/docker-compose.yml build
```

Start the default single-GPU stack with:

```bash
scripts/docker/up_nav2_stack_single_gpu.sh
```

The helper script can be run directly. Its default settings live at the top of
`scripts/docker/up_nav2_stack_single_gpu.sh`:

```bash
DEFAULT_LAUNCHER_CONFIG="configs/de_plan_with_render_template.yaml"
DEFAULT_SINGLE_GPU_DEVICE_IDS="0"
DEFAULT_ROS_DOMAIN_ID="0"
DEFAULT_SERVICES=(isaac nav2)
```

To limit CPU scheduling quota per container, use the wrapper options (or set
the equivalent `INTERNDATA_ISAAC_CPUS` and `INTERNDATA_NAV2_CPUS` environment
variables):

```bash
scripts/docker/up_nav2_stack.sh --isaac-cpus 16 --nav2-cpus 2 isaac nav2
```

`cpus` is a Docker CPU quota, not a physical-core pinning policy.

Use `configs/de_plan_with_render_template.yaml` for plan-with-render runs, or
change `DEFAULT_LAUNCHER_CONFIG` to `configs/de_pipe_template.yaml` for the
pipeline template. The script exports the selected config into Compose; the
choice is intentionally not hardcoded in `docker/docker-compose.yml`.

Start multiple isolated GPU stacks with:

```bash
scripts/docker/up_nav2_stack_multi_gpu.sh
```

Give every parallel stack the same Docker CPU quota with:

```bash
INTERNDATA_PARALLEL_ISAAC_CPUS=16 INTERNDATA_PARALLEL_NAV2_CPUS=2 \
  scripts/docker/up_nav2_stack_multi_gpu.sh
```

The multi-GPU defaults are `12` CPUs for Isaac and `6` for Nav2 in each stack.

The multi-GPU defaults live at the top of
`scripts/docker/up_nav2_stack_multi_gpu.sh`:

```bash
DEFAULT_LAUNCHER_CONFIG="configs/de_plan_with_render_template.yaml"
DEFAULT_PARALLEL_GPU_COUNT="4"
DEFAULT_PARALLEL_GPUS=""
DEFAULT_STACKS_PER_GPU="2"
DEFAULT_ROS_DOMAIN_BASE="10"
DEFAULT_STOP_NAV2_WHEN_ISAAC_EXITS="1"
```

When `DEFAULT_PARALLEL_GPUS` is empty, the script starts GPUs
`0..DEFAULT_PARALLEL_GPU_COUNT-1`. To use a non-contiguous set, set it to a
comma-separated list such as `0,2,3`. Set `DEFAULT_STACKS_PER_GPU` to run more
than one stack on each selected GPU. For example, `DEFAULT_PARALLEL_GPUS="0,1"`
and `DEFAULT_STACKS_PER_GPU="2"` starts four stacks: two on GPU 0 and two on
GPU 1.

Each stack gets separate container names, Nav2 session IDs, ROS domain IDs, and
Isaac cache/log directories. The launcher output name is not suffixed per GPU,
so generated data continues to use the output directory selected by the data
engine config, such as `output/simbox_plan_with_render/`.

By default, the scripts watch each Isaac container and stop the matching Nav2
container after Isaac exits. Set `DEFAULT_STOP_NAV2_WHEN_ISAAC_EXITS="0"` in the
script if you want Nav2 to keep running after Isaac finishes.

Watch logs:

```bash
docker compose -f docker/docker-compose.yml logs -f isaac
docker compose -f docker/docker-compose.yml logs -f nav2
```

Stop the stack:

```bash
scripts/docker/stop_all_docker.sh
```

By default, this stop script only stops containers named `isaac`, `nav2`,
`isaac-*`, and `nav2-*`. To stop every running Docker container on the host,
edit `DEFAULT_STOP_EVERY_RUNNING_CONTAINER="1"` at the top of the script.

The default single-stack startup behavior is:

- `isaac` autostarts `launcher.py` with the config selected in `scripts/docker/up_nav2_stack_single_gpu.sh`
- `nav2` autostarts the in-repo Nav2 bridge and bringup stack

Generated data and logs are written to:

- run logs: `output/simbox_plan_with_render/de_time_profile_*.log`
- rendered episodes and LMDB exports: `output/simbox_plan_with_render/...`
- Isaac container logs/cache mounts: `.docker/isaac-sim/`

If you prefer to run from the `docker/` directory directly, the equivalent commands are:

```bash
cd docker
docker compose build
../scripts/docker/up_nav2_stack_single_gpu.sh
```

By default, the Isaac workflow writes external Nav2 runtime requests to `output/ros_bridge/runtime_requests`, and the ROS service watches that directory to launch or refresh the corresponding Nav2 stack.

## License and Citation
All the code within this repo are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please consider citing our papers if it helps your research.

```BibTeX
@article{tian2025interndata,
  title={Interndata-a1: Pioneering high-fidelity synthetic data for pre-training generalist policy},
  author={Tian, Yang and Yang, Yuyin and Xie, Yiman and Cai, Zetao and Shi, Xu and Gao, Ning and Liu, Hangxu and Jiang, Xuekun and Qiu, Zherui and Yuan, Feng and others},
  journal={arXiv preprint arXiv:2511.16651},
  year={2025}
}

@article{he2026nimbus,
  title={Nimbus: A Unified Embodied Synthetic Data Generation Framework},
  author={He, Zeyu and Zhang, Yuchang and Zhou, Yuanzhen and Tao, Miao and Li, Hengjie and Tian, Yang and Zeng, Jia and Wang, Tai and Cai, Wenzhe and Chen, Yilun and others},
  journal={arXiv preprint arXiv:2601.21449},
  year={2026}
}

@article{chen2025internvla,
  title={Internvla-m1: A spatially guided vision-language-action framework for generalist robot policy},
  author={Chen, Xinyi and Chen, Yilun and Fu, Yanwei and Gao, Ning and Jia, Jiaya and Jin, Weiyang and Li, Hao and Mu, Yao and Pang, Jiangmiao and Qiao, Yu and others},
  journal={arXiv preprint arXiv:2510.13778},
  year={2025}
}
```

<!--
```BibTeX
@misc{interndataengine2026,
  title={InternDataEngine: A Synthetic Data Generation Engine for Robotic Learning},
  author={InternDataEngine Contributors},
  year={2026},
  }
}
``` -->
