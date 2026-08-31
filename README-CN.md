# InternDataEngine

## 📚 目录

- [🚀 快速开始](#quickstart)
  - [🧰 公共资产准备](#common-asset-setup)
  - [Docker 第一章：Docker Only](#chapter-1-docker-only)
  - [Conda 第二章：Conda](#chapter-2-conda)

InternDataEngine 是面向具身智能的合成数据生成引擎，基于 NVIDIA Isaac Sim 和 Nimbus 运行。SimBox 移动操作中的导航由 Isaac 内部的本地 A* 规划器和移动底盘驱动执行，无需 ROS 或 Nav2。



## 🚀 快速开始

仓库提供两个相互独立的仿真章节。公共资产只需准备一次，然后在每台机器上选择
一个运行章节。



### 公共资产准备

公共依赖为 Linux 主机、NVIDIA GPU 和可用驱动、Python 3.10+、Git、`7z`，
以及足够的资产和 CuRobo 磁盘空间：

```bash
nvidia-smi
python3 --version
git --version
7z
```

Ubuntu 缺少资产下载工具时：

```bash
sudo apt-get update
sudo apt-get install -y git p7zip-full python3-pip
python3 -m pip install -U modelscope
```

在仓库根目录下载并解压 ModelScope 资产：

```bash
python3 scripts/download_modelscope.py --token <MODEL_SCOPE_TOKEN>
```

随后拉取固定版本的 CuRobo v2：

```bash
git clone https://github.com/MaxDYF/curobo.git InternDataAssets/curobov2
git -C InternDataAssets/curobov2 checkout 4ea77366ca48ee453e7df139e39fa6532af49f3b
```

运行时只使用 `InternDataAssets/curobov2`，不依赖旧的
`InternDataAssets/curobo`。检查资产和软链接：

```bash
test -d InternDataAssets/assets
test -d InternDataAssets/assets/custom
test -d InternDataAssets/robots
test -f InternDataAssets/curobov2/curobo/__init__.py
test -d InternDataAssets/panda_drake
test -L workflows/simbox/assets
test -L workflows/simbox/panda_drake
```



## 🐳 第一章：Docker Only

本章是生产默认路径，只通过
`scripts/docker/up_simbox_isaac.sh` 运行，不需要宿主机 Isaac Sim 或 Conda
仿真环境。

### 前置条件

- Docker Engine with Compose v2
- NVIDIA Container Toolkit
- 足够的 Docker 镜像和 Isaac Sim cache 磁盘空间

检查 Docker 和 GPU：

```bash
docker compose version
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

如果入口脚本没有执行权限：

```bash
chmod +x docker/isaac/entrypoint.sh scripts/docker/up_simbox_isaac.sh
```

### 构建与运行

```bash
docker compose -f docker/docker-compose.yml build
scripts/docker/up_simbox_isaac.sh
```

指定 GPU、CPU 配额或 pipeline 配置：

```bash
scripts/docker/up_simbox_isaac.sh --gpu 0 --isaac-cpus 16
scripts/docker/up_simbox_isaac.sh --launcher-config configs/de_pipe_template.yaml
```

`cpus` 是 Docker CPU 配额，不是物理 CPU 核绑定。并行任务请使用不同的
`--stack-id` 和 `--gpu`：

```bash
scripts/docker/up_simbox_isaac.sh --stack-id worker0 --gpu 0
scripts/docker/up_simbox_isaac.sh --stack-id worker1 --gpu 1
```

### 开发、日志与停止

进入独立的 Isaac Bash 开发容器：

```bash
scripts/docker/isaac_dev.sh shell --gpu 0 --build
scripts/docker/isaac_dev.sh start --gpu 0
scripts/docker/isaac_dev.sh exec -- python -c 'import torch; print(torch.__version__)'
scripts/docker/isaac_dev.sh stop
```

查看日志并停止任务：

```bash
docker logs -f isaac
scripts/docker/stop_all_docker.sh
```

默认只停止 `isaac` 和 `isaac-*` 容器。如确需停止宿主机上的所有 Docker
容器，将停止脚本顶部的 `DEFAULT_STOP_EVERY_RUNNING_CONTAINER="1"` 打开后再执行。

常见输出位置：

- plan-with-render 日志：`output/simbox_plan_with_render*/de_time_profile_*.log`
- 渲染数据和 LMDB：`output/simbox_plan_with_render*/`
- Isaac cache/log：`.docker/isaac-sim/`



## 🟢 第二章：Conda

本章在宿主机 Conda 环境中原生运行 Isaac Sim。它不需要 Docker 或 NVIDIA
Container Toolkit，但要求 Linux x86_64、glibc 2.35+、NVIDIA GPU 和 Conda。

### 安装环境

```bash
scripts/conda/setup_isaac6_env.sh --env interndata-isaac6
```

脚本安装 Python 3.12、Torch 2.11/cu128、Isaac Sim 6.0.1.0，以及项目和
CuRobo v2 依赖。CuRobo 始终从 `InternDataAssets/curobov2` 导入，并拒绝
`nvidia-curobo` wheel。

### 激活、校验与预检

```bash
CONDA_ENV=interndata-isaac6 source scripts/conda/activate_isaac6_env.sh
python scripts/conda/verify_isaac6_env.py
```

不启动 Isaac，先检查版本、源码身份、`CUDA_HOME` 和 GPU：

```bash
TASK_CONFIG=workflows/simbox/core/configs/tasks/example/sort_the_rubbish.yaml \
GPU_ID=0 \
CONDA_ENV=interndata-isaac6 \
CONDA_PREFLIGHT_ONLY=1 \
scripts/simbox/run_simbox_task.sh
```

### 运行任务与 Agent

预检通过后运行单个任务：

```bash
TASK_CONFIG=workflows/simbox/core/configs/tasks/example/sort_the_rubbish.yaml \
GPU_ID=0 \
CONDA_ENV=interndata-isaac6 \
scripts/simbox/run_simbox_task.sh
```

Agent 控制进程可以继续使用轻量 `interndata` 环境；显式选择 Conda 仿真：

```bash
conda run -n interndata python -m agent run \
  --prompt "把杯子放到托盘里" \
  --gpu 0 \
  --simulator-backend conda \
  --conda-env interndata-isaac6
```

`setup_isaac6_env.sh --dry-run` 可查看安装步骤，任务入口支持
`DRY_RUN=1` 和 `CONDA_PREFLIGHT_ONLY=1`。Agent 默认仍为 Docker，除非指定
`--simulator-backend conda` 或设置 `execution.simulator_backend: conda`。

更多 API、开发和历史文档见 [文档索引](./docs/README.md)。