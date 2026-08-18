# InternDataEngine

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

### 执行配置文件功能解析

#### 模板家族（同 `configs/` 目录）

| 模板                                      | 顶层 stage                                                              | 定位                                                  |
| ----------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------- |
| `de_plan_with_render_template.yaml`     | load →**plan_with_render** → store                              | 规划+渲染合一，默认 launcher 配置                     |
| `de_plan_template.yaml`                 | load → plan → store                                                   | 纯规划，不渲染                                        |
| `de_render_template.yaml`               | load → plan → render → store                                         | 规划/渲染拆成两个 stage                               |
| `de_plan_and_render_template.yaml`      | load → plan → render → store                                         | 同上（较旧）                                          |
| `de_pipe_template.yaml`                 | load → plan → dump → dedump → render → store +**stage_pipe** | 并行/分布式（Ray）管线                                |
| `de_workspace_probe_template.yaml`      | load → plan_with_render → store                                       | workspace 探测                                        |
| `de_plan_with_render_*_validation.yaml` | 同主模板                                                                | 具体场景的验证配置（scene8 / split_aloha_navigation） |

目前默认是用`de_plan_with_render_template.yaml 换task直接替换 yaml文件中的`

```yaml
      cfg_path: InternDataAssets/assets/custom/scene_8/01_kitchen/assets/basic/kitchen_apple_orange_to_tray/simbox_task.yaml  # Task config path 为具体的 simbox_task.yaml即可
```

## 架构总览

InternDataEngine 分两层：**Nimbus 流水线层**（`configs/*.yaml` launcher 配置驱动，负责加载场景、随机化、规划渲染、落盘）和 **SimBox 工作流层**（任务 YAML 驱动，`SimBoxDualWorkFlow` 负责场景组装、技能编排与执行）。

```text
launcher 配置 ──ConfigProcessor──▶ run_data_engine
   └─▶ 流水线 stages: scene_loader(EnvLoader) → layout_random_generator → plan_with_render → writer
         └─▶ EnvLoader 创建 SimBoxDualWorkFlow（任务 YAML）
               └─▶ robots / controllers / skills / objects 装配 → 执行 skill 图 → 产出 obs / seq
```

核心目录（`workflows/simbox/core/`）：

| 模块                                 | 职责                                                                                                                     |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `robots/`                          | 机器人实现（`@register_robot`）：解析 robot yaml，拼 joints / ee / mobile_base 路径                                    |
| `controllers/`                     | 臂控制器（`@register_controller`，继承 `TemplateController`，CuRobo 运动规划）；加新臂先看 `controllers/README.md` |
| `skills/`                          | 技能实现（`@register_skill`，继承 `BaseSkill`）：`pick` / `place` / `navigate` / `dexpick` / `dexplace`…  |
| `tasks/`                           | 任务基类与工作流入口                                                                                                     |
| `objects/`                         | 物体类：`RigidObject` / `GeometryObject` / `ArticulatedObject`…                                                   |
| `planning/`                        | 抓取 / 放置 / 碰撞规划与合同校验                                                                                         |
| `execution/`                       | 执行监督与安全监控（滑动检测等）                                                                                         |
| `mobile/`                          | 移动底盘驱动（差速 / 虚拟 base）                                                                                         |
| `cameras/` + `utils/camera_*.py` | 相机与模板（realsense / astra 配置）                                                                                     |
| `loggers/`                         | LMDB / 事件日志                                                                                                          |
| `workspace/`                       | 工作区与可达性规划                                                                                                       |
| `configs/`                         | 任务 / 机器人 / 相机 / 导航 / 底盘 YAML                                                                                  |

核心概念：

| 概念                              | 说明                                                                                     |
| --------------------------------- | ---------------------------------------------------------------------------------------- |
| 任务 YAML（`simbox_task.yaml`） | 定义场景、机器人、物体、region 采样器、相机、skill 图、数据输出                          |
| `target_class`                  | robot yaml 里的键，映射到`robots/*.py` 中 `@register_robot` 的类                     |
| `robot_file`                    | curobo 配置（`robot_cfg.kinematics`：urdf_path / base_link / ee_link），运动规划依赖它 |
| 技能（skill）                     | 任务 YAML`skills:` 段按 `name` 映射到 `SKILL_DICT`，按 `depends_on` 串成 DAG     |
| region 采样器                     | 物体 / 机器人初始位姿的随机采样（`A_on_B_region_sampler` 等）                          |

## 文档索引

| 文档                                            | 内容                                                                  |
| ----------------------------------------------- | --------------------------------------------------------------------- |
| `docs/PROJECT_DOCUMENTATION.md`               | 生产级项目总览：定位 / 技术栈 / 数据流                                |
| `docs/SIMBOX_ARENA_TASK_YAML_API.md`          | 任务 YAML 完整 schema 与加载流程                                      |
| `docs/SIMBOX_CONFIG_REFERENCE.md`             | 任务 YAML 各段参考（robots / objects / cameras / regions / skills…） |
| `docs/SIMBOX_SKILLS_API.md`                   | Skill API：编排模式、方向过滤、注册技能清单                           |
| `docs/simbox_skill_reference.md`              | Skill 参数速查（几何 / 方向滤波 / bounding box）                      |
| `docs/docker并行生成使用说明.md`              | 多 GPU / 多容器并行生成                                               |
| `workflows/simbox/core/controllers/README.md` | **加新臂控制器**的完整教程                                      |
|                                                 |                                                                       |
| `AGENTS.md`                                   | 工程工作笔记：验证标准、重置 / 随机化、导航调试                       |
| `MERGE_TODO.md`                               | 当前开发分支未完成的合并项                                            |

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
