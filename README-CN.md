<div align="center">

# InternDataEngine

</div>

InternDataEngine 是面向具身智能的合成数据生成引擎，基于 NVIDIA Isaac Sim 和 Nimbus 运行。SimBox 移动操作中的导航由 Isaac 内部的本地 A* 规划器和移动底盘驱动执行，无需 ROS 或 Nav2。

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
chmod +x docker/isaac/entrypoint.sh scripts/docker/up_simbox_isaac.sh
```

## Isaac Sim Docker 部署

SimBox 导航在 Isaac 内部通过本地 A* 规划器和移动底盘驱动运行。Docker 部署只有一个 `isaac` 服务，不启动 ROS、Nav2 或额外 bridge 容器。

构建镜像：

```bash
docker compose -f docker/docker-compose.yml build
```

启动默认单 GPU 容器（GPU `0`，默认配置为 `configs/de_plan_with_render_template.yaml`）：

```bash
scripts/docker/up_simbox_isaac.sh
```

指定 GPU、CPU 配额或 launcher 配置：

```bash
scripts/docker/up_simbox_isaac.sh --gpu 0 --isaac-cpus 16
scripts/docker/up_simbox_isaac.sh --launcher-config configs/de_pipe_template.yaml
```

`cpus` 是 Docker CPU 配额，不是物理 CPU 核绑定。并行任务请为每个实例指定不同的 `--stack-id` 和 `--gpu`，以隔离容器名和 Isaac cache 目录：

```bash
scripts/docker/up_simbox_isaac.sh --stack-id worker0 --gpu 0
scripts/docker/up_simbox_isaac.sh --stack-id worker1 --gpu 1
```

## 查看日志

默认容器：

```bash
docker logs -f isaac
```

指定 `--stack-id worker0` 的容器示例：

```bash
docker logs -f isaac-worker0
```

常见输出位置：

- plan-with-render 日志：`output/simbox_plan_with_render*/de_time_profile_*.log`
- 渲染数据和 LMDB：`output/simbox_plan_with_render*/`
- Isaac cache/log 挂载：`.docker/isaac-sim/`

## 停止容器

停止本项目启动的 Isaac 容器：

```bash
scripts/docker/stop_all_docker.sh
```

默认只停止：

- `isaac`
- `isaac-*`

如果确实要停止宿主机上所有正在运行的 Docker 容器，把脚本顶部改成：

```bash
DEFAULT_STOP_EVERY_RUNNING_CONTAINER="1"
```

然后再运行：

```bash
scripts/docker/stop_all_docker.sh
```
