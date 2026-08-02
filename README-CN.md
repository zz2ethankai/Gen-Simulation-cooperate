<div align="center">

# InternDataEngine

</div>

InternDataEngine 是面向具身智能的合成数据生成引擎，基于 NVIDIA Isaac Sim、Nimbus 和仓库内置的 ROS/Nav2 拆分部署组件运行。

## 快速开始

当前 Docker 工作流依赖完整的 `InternDataAssets/` 目录。请按下面顺序安装，
否则 Docker 构建会因为找不到 `InternDataAssets/curobo` 失败，SimBox 运行时也
无法解析资产路径。

### 1. 系统依赖

- Linux 主机，NVIDIA GPU 和可用驱动
- Docker Engine with Compose v2
- NVIDIA Container Toolkit
- 宿主机 Python 3.10+
- `7z` 命令行工具
- 足够磁盘空间存放资产压缩包、解压后的资产、Docker 镜像和 Isaac Sim cache。
  当前 `InternDataAssets/` 解压后约 200 GB。

快速检查：

```bash
nvidia-smi
docker compose version
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
7z
```

Ubuntu 上缺少工具时可安装：

```bash
sudo apt-get update
sudo apt-get install -y p7zip-full python3-pip
python3 -m pip install -U modelscope
```

### 2. 下载资产

在仓库根目录运行：

```bash
python3 scripts/download_modelscope.py --token <MODEL_SCOPE_TOKEN>
```

脚本会下载 `MinMaxMex/InterndataAssets/InternDataAssets_7z` 分卷压缩包，
解压出 `InternDataAssets/`，并在 SimBox 下创建所需的相对软链接：

```text
workflows/simbox/assets -> ../../InternDataAssets/assets
workflows/simbox/curobo -> ../../InternDataAssets/curobo
workflows/simbox/panda_drake -> ../../InternDataAssets/panda_drake
```

如果本地已经存在 `InternDataAssets/`，脚本会拒绝覆盖。需要重新安装资产时，
请先移动或删除旧目录。

### 3. 安装校验

构建镜像前先确认：

```bash
test -d InternDataAssets/assets
test -d InternDataAssets/curobo
test -d InternDataAssets/panda_drake
test -L workflows/simbox/assets
test -L workflows/simbox/curobo
test -L workflows/simbox/panda_drake
```

如果入口脚本没有执行权限，先修复：

```bash
chmod +x docker/isaac/entrypoint.sh docker/nav2/entrypoint.sh
```

## Split ROS / Isaac Sim Docker 部署

仓库提供了 Isaac Sim 与 ROS/Nav2 分离运行的 Docker 结构：

- `docker/isaac/`：Isaac Sim 镜像和入口脚本
- `nav2/`：仓库内置 Nav2 包、ROS 侧 bridge 和运行时代码
- `scripts/docker/`：启动和停止脚本

依赖：

- Docker Engine with Compose v2
- NVIDIA Container Toolkit
- 宿主机可见的 NVIDIA GPU
- `.docker/isaac-sim/` 下有足够空间存放 Isaac cache/log/config/data

先构建镜像：

```bash
docker compose -f docker/docker-compose.yml build
```

## 单 GPU 启动

直接运行：

```bash
scripts/docker/up_nav2_stack_single_gpu.sh
```

脚本顶部包含默认配置：

```bash
DEFAULT_LAUNCHER_CONFIG="configs/de_plan_with_render_template.yaml"
DEFAULT_SINGLE_GPU_DEVICE_IDS="0"
DEFAULT_ROS_DOMAIN_ID="0"
DEFAULT_SERVICES=(isaac nav2)
```

可通过 wrapper 为两个容器分别限制 Docker CPU 配额（也可设置同名环境变量
`INTERNDATA_ISAAC_CPUS` 和 `INTERNDATA_NAV2_CPUS`）：

```bash
scripts/docker/up_nav2_stack.sh --isaac-cpus 16 --nav2-cpus 2 isaac nav2
```

`cpus` 是 Docker CPU 配额，不是物理 CPU 核绑定。

默认行为：

- 使用宿主机 GPU `0`
- 启动 `isaac` 和 `nav2`
- Isaac 自动运行 `launcher.py`
- Nav2 自动运行仓库内置 bridge 和 bringup stack
- 默认运行 `configs/de_plan_with_render_template.yaml`

如果要切换到 pipeline 模板，直接把脚本顶部改成：

```bash
DEFAULT_LAUNCHER_CONFIG="configs/de_pipe_template.yaml"
```

运行模式选择放在脚本里，不放在 `docker/docker-compose.yml` 里。

## 多 GPU 并行启动

直接运行：

```bash
scripts/docker/up_nav2_stack_multi_gpu.sh
```

为每个并行栈设置相同的 Docker CPU 配额：

```bash
INTERNDATA_PARALLEL_ISAAC_CPUS=16 INTERNDATA_PARALLEL_NAV2_CPUS=2 \
  scripts/docker/up_nav2_stack_multi_gpu.sh
```

多 GPU 脚本默认每个栈为 Isaac 分配 `12` CPUs、为 Nav2 分配 `6` CPUs。

脚本顶部包含默认配置：

```bash
DEFAULT_LAUNCHER_CONFIG="configs/de_plan_with_render_template.yaml"
DEFAULT_PARALLEL_GPU_COUNT="4"
DEFAULT_PARALLEL_GPUS=""
DEFAULT_STACKS_PER_GPU="2"
DEFAULT_ROS_DOMAIN_BASE="10"
DEFAULT_STOP_NAV2_WHEN_ISAAC_EXITS="1"
```

当 `DEFAULT_PARALLEL_GPUS` 为空时，脚本会按 `DEFAULT_PARALLEL_GPU_COUNT` 生成连续 GPU 列表。例如默认值 `4` 会启动 GPU `0`、`1`、`2`、`3` 四组容器。

如果要启动 4 组，修改：

```bash
DEFAULT_PARALLEL_GPU_COUNT="4"
```

如果要指定非连续 GPU，例如 `0,2,3`，修改：

```bash
DEFAULT_PARALLEL_GPUS="0,2,3"
```

如果要在每张 GPU 上启动多组容器，修改：

```bash
DEFAULT_STACKS_PER_GPU="2"
```

例如：

```bash
DEFAULT_PARALLEL_GPUS="0,1"
DEFAULT_STACKS_PER_GPU="2"
```

会启动四组容器，其中两组使用 GPU `0`，两组使用 GPU `1`。

每组都会自动隔离：

- Compose project
- 容器名，例如 `isaac-gpu0`、`nav2-gpu0`
- Nav2 session UUID
- `ROS_DOMAIN_ID`
- Isaac cache/log/config/data 目录

输出目录不会自动加 `gpu0`、`gpu1` 之类的后缀。plan-with-render 会继续使用数据引擎配置里的 `name`，例如写到 `output/simbox_plan_with_render/`。

默认情况下，脚本会在 Isaac 结束后自动停止对应的 Nav2 容器。若要让 Nav2 在 Isaac 结束后继续运行，把 `DEFAULT_STOP_NAV2_WHEN_ISAAC_EXITS="0"`。

注意：多 GPU 脚本可以在每张宿主机 GPU 上启动多组 Docker stack。每个 stack 的容器内只暴露一张 GPU。

## 查看日志

单组默认容器：

```bash
docker logs -f isaac
docker logs -f nav2
```

多组容器示例：

```bash
docker logs -f isaac-gpu0
docker logs -f nav2-gpu0
docker logs -f isaac-gpu1
docker logs -f nav2-gpu1
```

常见输出位置：

- plan-with-render 日志：`output/simbox_plan_with_render*/de_time_profile_*.log`
- 渲染数据和 LMDB：`output/simbox_plan_with_render*/`
- Nav2 runtime：`output/nav2_runtime/`
- Isaac cache/log 挂载：`.docker/isaac-sim/`

## 停止容器

停止本项目启动的 Isaac/Nav2 容器：

```bash
scripts/docker/stop_all_docker.sh
```

默认只停止：

- `isaac`
- `nav2`
- `isaac-*`
- `nav2-*`

如果确实要停止宿主机上所有正在运行的 Docker 容器，把脚本顶部改成：

```bash
DEFAULT_STOP_EVERY_RUNNING_CONTAINER="1"
```

然后再运行：

```bash
scripts/docker/stop_all_docker.sh
```
