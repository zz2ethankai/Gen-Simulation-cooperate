# Unitree G1 navigation 数据生成指南

**Last Updated:** 2026-09-01
**Entry Points:** `scripts/docker/run_simbox_task.sh`, `configs/de_plan_with_render_unitree_g1_navigation_validation.yaml`

本文说明如何在 InternDataEngine 中使用 Unitree G1 执行简单 navigation 任务、生成完整
LMDB episode，并将数据转换为 LeRobot v2.1。所有命令均从仓库根目录执行。

## 1. 功能范围

当前 G1 数据生成链路为：

```text
task YAML
-> Load / randomize
-> Navigate skill
-> local A* path
-> phased waypoint control
-> G1 body-command translation
-> Python decoupled-WBC
-> Isaac physics and rendering
-> LMDB episode store
```

当前实现包括：

- 29-DOF free-base Unitree G1；
- 仅调用原项目已有的 `Navigate` skill；
- 静态地图与 A\* 路径规划；
- 面向 G1 的转向、短时横移校正和直走三个互斥控制阶段；
- NVIDIA decoupled-WBC Balance/Walk ONNX 策略；
- 200 Hz physics、50 Hz WBC、50 Hz rendering 和 50 FPS 视频；
- 第一人称和全局 RGB 视频；
- G1 proprioception、navigation request、WBC target 和 next-state label；
- LMDB 到 LeRobot v2.1 的 G1 转换入口。

当前流程不安装或运行完整 GEAR-SONIC。SONIC planner、encoder、decoder、C++ runtime 和
TensorRT deployment 均不是运行依赖；仍需下载 NVIDIA decoupled-WBC 的 Balance/Walk
两个 ONNX 权重。

## 2. 代码结构

| 边界       | 文件                                                                                       | 作用                                                            |
| ---------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| Robot      | `workflows/simbox/core/robots/unitree_g1.py`                                               | 加载 G1、设置初始状态、读取本体状态并写入 WBC joint target      |
| Platform   | `workflows/simbox/core/mobile/platforms/unitree_g1_platform.py`                            | 通过原 mobile platform registry 选择 G1 driver                  |
| WBC        | `workflows/simbox/core/mobile/g1_decoupled_wbc.py`                                         | 加载 Balance/Walk ONNX 并执行 50 Hz 推理                        |
| Lifecycle  | `workflows/simbox/core/mobile/g1_locomotion_driver.py`                                     | 管理 bootstrap、episode warmup、速度请求和 joint target         |
| Navigation | `workflows/simbox/core/skills/local_navigation.py`                                         | A\* 路径及分阶段 waypoint controller                            |
| Skill      | `workflows/simbox/core/skills/navigate.py`                                                 | 保持原`Navigate` 生命周期，通过 controller factory 选择平台适配 |
| Logger     | `workflows/simbox/core/loggers/utils.py`                                                   | 记录 G1 state、navigation request 和 WBC output                 |
| Task       | `workflows/simbox/core/configs/tasks/navigation/unitree_g1/`                               | A/B 代表性配置与 Scene 4 kitchen 端到端样例                      |
| Engine     | `configs/de_plan_with_render_unitree_g1_navigation_validation.yaml`                        | 200/50 Hz Isaac 与`plan_with_render` 配置                       |
| Converter  | `policy/lmdb2lerobotv21/lmdb2lerobot_unitree_g1_a1.py`                                     | G1 LMDB 到 LeRobot v2.1                                         |

G1 初始化没有建立独立于原项目的新任务流程。Robot 初始化、controller/skill 构造、
reset warmup 和 episode 执行仍由 `SimBoxDualWorkFlow` 管理。

## 3. 时序

| 层级          |            频率 | 行为                                      |
| ------------- | --------------: | ----------------------------------------- |
| Isaac physics |          200 Hz | 每 5 ms 更新动力学和机器人测量状态        |
| decoupled-WBC |           50 Hz | 每 20 ms 根据最新测量状态生成一次 target  |
| Target reuse  | 4 physics steps | 一个 WBC target 连续作用四个 physics step |
| Rendering     |           50 Hz | `rendering_dt: 1/50`                      |
| Episode video |          50 FPS | ego/global 每个记录 step 各保存一帧       |

任务配置中的 `video_fps: 50` 与运行时
`INTERNDATA_VIDEO_FPS=50` 保持一致。当前正式 G1 episode 已验证为 H.264、
1280x720、50 FPS。

## 4. 安装

### 4.1 宿主机工具

Ubuntu/Debian 可安装：

```bash
sudo apt-get update
sudo apt-get install -y curl git git-lfs p7zip-full rsync
git lfs install
```

此外需要：

- Docker 与 Docker Compose；
- NVIDIA driver 和 NVIDIA Container Toolkit；
- `uv`；
- 可运行 Isaac Sim 4.1 的 NVIDIA GPU。

检查主要边界：

```bash
docker compose version
nvidia-smi
uv --version
git lfs version
7z i | head
```

### 4.2 Python 环境

创建与 Isaac Sim 4.1 匹配的 Python 3.10 环境：

```bash
uv venv --python 3.10 .venv

UV_CACHE_DIR=tmp/g1_install/uv_cache uv pip install \
  --python .venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements.txt
```

G1 所需的 ONNX Runtime 依赖已经合并到根目录 `requirements.txt`，不维护第二份
G1 专用 requirements 文件。

### 4.3 下载资源

运行：

```bash
bash scripts/download_unitree_g1_navigation_assets.sh
```

脚本只安装 Unitree G1 USD 和 NVIDIA decoupled-WBC 的 Balance/Walk 两个 ONNX
权重，不下载完整 GEAR-SONIC。SONIC planner、encoder、decoder、observation
configuration、C++ runtime 和 TensorRT deployment 均不是当前运行依赖。

#### Unitree G1 USD

- Dataset: [https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds](https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds)
- Code: [https://github.com/unitreerobotics/unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)
- Pinned revision: `394cf2448f8a9ed815c77c701a761f3d1ff1c8fb`
- Source directory: `assets/robots/g1-29dof_wholebody_dex3/`

运行时使用 free-base `g1_29dof_with_dex3_rev_1_0.usd`，walking task 不应使用
`*_base_fix.usd`。下载脚本通过 `hf-mirror.com` 获取数据；镜像只作为传输端点，
Unitree dataset 仍是权威来源。

#### NVIDIA decoupled-WBC

- Repository: [https://github.com/NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
- Pinned revision: `a0732b642c0333077e127a2f56ab0014c196bca4`
- Source directory: `decoupled_wbc/sim2mujoco/resources/robots/g1/policy/`
- Required models:
  - `GR00T-WholeBodyControl-Balance.onnx`
  - `GR00T-WholeBodyControl-Walk.onnx`

上游通过 Git LFS 保存模型。下载脚本跳过无关 LFS 内容，只拉取上述两个文件，并由
项目内 Python ONNX Runtime 直接加载。

安装后的仓库相对目录为：

```text
InternDataAssets/assets/unitree_g1_sonic/
├── g1/
│   ├── config.yaml
│   ├── g1_29dof_with_dex3_rev_1_0.usd
│   └── configuration/
└── wbc/
    ├── GR00T-WholeBodyControl-Balance.onnx
    └── GR00T-WholeBodyControl-Walk.onnx
```

目录名 `unitree_g1_sonic` 是当前配置保留的兼容路径，不代表完整 SONIC runtime
已经安装。下载缓存和上游 checkout 位于 Git 忽略的 `tmp/g1_install/`。

检查三个必需文件：

```bash
test -s InternDataAssets/assets/unitree_g1_sonic/g1/g1_29dof_with_dex3_rev_1_0.usd
test -s InternDataAssets/assets/unitree_g1_sonic/wbc/GR00T-WholeBodyControl-Balance.onnx
test -s InternDataAssets/assets/unitree_g1_sonic/wbc/GR00T-WholeBodyControl-Walk.onnx
```

Unitree 仓库和 dataset 标记为 Apache-2.0。NVIDIA 仓库源代码采用 Apache-2.0，
模型权重采用 NVIDIA Open Model License。下载的 USD 和 ONNX 不应加入 Git；公开发布
时应保留相应上游 notice 和 attribution。

### 4.4 构建容器

项目继续使用 Isaac Sim 4.1/CUDA 11.8/CuRobo 镜像：

```bash
INTERNDATA_LAUNCHER_CONFIG=configs/de_plan_with_render_unitree_g1_navigation_validation.yaml \
  docker compose -f docker/docker-compose.yml build isaac
```

容器只使用项目 Python 环境中的 ONNX Runtime/CUDA 库，不部署 SONIC 服务或 SONIC
C++ runtime。

## 5. 运行代表性 navigation episode

仓库保留六个 A/B 代表性配置，而不是提交全部 A0-A9、B1-B5 的近重复 YAML；此外保留一个
Scene 4 kitchen 端到端样例：

| Case | 配置文件                                      | 覆盖内容               |
| ---- | --------------------------------------------- | ---------------------- |
| A0   | `navigate_empty_validation.yaml`              | 空场景直走             |
| A4   | `navigate_diagonal_validation.yaml`           | 斜向目标与转向后直走   |
| A6   | `navigate_lateral_validation.yaml`            | 横向目标与横移校正     |
| B1   | `navigate_single_obstacle_validation.yaml`    | 单障碍 A\* 绕行        |
| B4   | `navigate_switchback_validation.yaml`         | 双墙 S 型连续反向转弯  |
| B5   | `navigate_narrow_corridor_validation.yaml`    | 1.25 m 窄走廊边界      |
| Scene 4 | `navigate_scene4_kitchen_validation.yaml` | 冰箱、架子和储物箱三段导航 |

这些配置均包含相同的 G1 第一人称相机挂载，并根据目标范围分别设置全局相机位置。任务的
navigation 目标和障碍物定义来自固定 seed `42` 的代表性验证；运行超时以各 YAML 中的
`runtime_timeout_sec` 为准。Scene 4 使用任务专属 Arena 配置
`workflows/simbox/core/configs/arenas/unitree_g1_scene4_kitchen_navigation_arena.yaml`。它基于
`workflows/simbox/assets/custom/scene_4/01_kitchen/assets/basic/kitchen_apple_to_tray/simbox_arena.yaml`，
唯一语义变化是将缺失的 floor texture library 路径改为已有的 `textures/floor.png` 所在
目录；源 Scene 4 资产不变。

推荐使用项目现有 wrapper：

```bash
CASE_CONFIG=navigate_empty_validation
RUN_ROOT="tmp/g1_navigation_examples/${CASE_CONFIG}"
mkdir -p "${RUN_ROOT}"

TASK_CONFIG="workflows/simbox/core/configs/tasks/navigation/unitree_g1/${CASE_CONFIG}.yaml" \
LAUNCH_TEMPLATE=configs/de_plan_with_render_unitree_g1_navigation_validation.yaml \
RUN_NAME="g1_${CASE_CONFIG}" \
GPU_ID=0 \
RANDOM_NUM=1 \
RANDOM_SEED=42 \
INTERNDATA_ACTION_FPS=50 \
INTERNDATA_VIDEO_FPS=50 \
OUTPUT_DIR="${RUN_ROOT}/output" \
INTERNDATA_EPISODE_EVENT_PATH="${RUN_ROOT}/episode_events.jsonl" \
INTERNDATA_DOCKER_METADATA_PATH="${RUN_ROOT}/docker_runtime.json" \
bash scripts/docker/run_simbox_task.sh 2>&1 | tee "${RUN_ROOT}/run.log"
```

将 `CASE_CONFIG` 替换为表中的文件名（不含 `.yaml`）即可运行其他代表性任务。批量运行
六个配置时可使用：

```bash
set -o pipefail
for case_config in \
  navigate_empty_validation \
  navigate_diagonal_validation \
  navigate_lateral_validation \
  navigate_single_obstacle_validation \
  navigate_switchback_validation \
  navigate_narrow_corridor_validation; do
  run_root="tmp/g1_navigation_examples/${case_config}"
  mkdir -p "${run_root}"
  TASK_CONFIG="workflows/simbox/core/configs/tasks/navigation/unitree_g1/${case_config}.yaml" \
  LAUNCH_TEMPLATE=configs/de_plan_with_render_unitree_g1_navigation_validation.yaml \
  RUN_NAME="g1_${case_config}" \
  GPU_ID=0 RANDOM_NUM=1 RANDOM_SEED=42 \
  INTERNDATA_ACTION_FPS=50 INTERNDATA_VIDEO_FPS=50 \
  OUTPUT_DIR="${run_root}/output" \
  INTERNDATA_EPISODE_EVENT_PATH="${run_root}/episode_events.jsonl" \
  INTERNDATA_DOCKER_METADATA_PATH="${run_root}/docker_runtime.json" \
  bash scripts/docker/run_simbox_task.sh 2>&1 | tee "${run_root}/run.log"
done
```

成功运行必须同时满足：

```bash
rg -n "Task is successful|Episode failed" "${RUN_ROOT}/run.log"

find "${RUN_ROOT}/output" \
  \( -name meta_info.pkl -o -name data.mdb -o -name demo.mp4 \) -print
```

判定条件：

- 日志包含 `Task is successful, mode=plan_with_render`；
- 日志不包含 `[LmdbLogger] Episode failed`；
- `meta_info.pkl`、`lmdb/data.mdb` 和两路 RGB 视频均存在；
- episode event 的 `status` 为 `success`。

失败 episode 不应进入训练数据白名单。

### 5.1 Scene 4 kitchen 三段导航实例

该样例只串联项目已有的三个 `Navigate` skill，不新增任务执行流程。机器人从冰箱前的安全
站位出发，依次到达架子、储物箱，再返回冰箱：

| 阶段 | 目标 `x, y, yaw` | 场景语义 |
| ---- | ---------------- | -------- |
| 起点 | `[1.15, 1.60, 0.0]` | 冰箱前安全站位 |
| Leg 1 | `[1.60, 2.00, pi/2]` | 冰箱到架子 |
| Leg 2 | `[3.05, 1.30, -pi/2]` | 架子到储物箱 |
| Leg 3 | `[1.15, 1.60, pi]` | 储物箱返回冰箱 |

目标坐标是家具前方的可通行站位，不是家具几何中心。global 相机设为
`[2.0, 1.5, 4.5]`，ego 相机继续挂载在 G1 `d435_link`。

运行命令：

```bash
RUN_ROOT=tmp/g1_scene4_kitchen_validation/reproduction
mkdir -p "${RUN_ROOT}"

TASK_CONFIG=workflows/simbox/core/configs/tasks/navigation/unitree_g1/navigate_scene4_kitchen_validation.yaml \
LAUNCH_TEMPLATE=configs/de_plan_with_render_unitree_g1_navigation_validation.yaml \
RUN_NAME=g1_scene4_kitchen_three_leg \
GPU_ID=0 \
RANDOM_NUM=1 \
RANDOM_SEED=42 \
INTERNDATA_ACTION_FPS=50 \
INTERNDATA_VIDEO_FPS=50 \
OUTPUT_DIR="${RUN_ROOT}/output" \
INTERNDATA_EPISODE_EVENT_PATH="${RUN_ROOT}/episode_events.jsonl" \
INTERNDATA_DOCKER_METADATA_PATH="${RUN_ROOT}/docker_runtime.json" \
bash scripts/docker/run_simbox_task.sh 2>&1 | tee "${RUN_ROOT}/run.log"
```

原始 episode 已直接包含 `images.rgb.global/demo.mp4` 和
`images.rgb.ego/demo.mp4`。如需生成 global 在上、ego 在下的诊断视频：

```bash
EPISODE_META="$(find "${RUN_ROOT}/output" -name meta_info.pkl -print | sort | tail -n 1)"
EPISODE_DIR="${EPISODE_META%/meta_info.pkl}"

ffmpeg -y \
  -i "${EPISODE_DIR}/images.rgb.global/demo.mp4" \
  -i "${EPISODE_DIR}/images.rgb.ego/demo.mp4" \
  -filter_complex "[0:v][1:v]vstack=inputs=2[stacked]" \
  -map "[stacked]" -c:v libx264 -pix_fmt yuv420p -r 50 \
  "${RUN_ROOT}/global_top_ego_bottom.mp4"
```

转换该次运行的成功 episode：

```bash
uv run --project policy/openpi-InternData-A1 \
  python policy/lmdb2lerobotv21/lmdb2lerobot_unitree_g1_a1.py \
  --src-path "${RUN_ROOT}/output" \
  --save-path "${RUN_ROOT}/lerobot_v21" \
  --repo-id local/unitree-g1-scene4-kitchen \
  --fps 50 \
  --num-episodes 1
```

固定 seed `42` 的一次真实验证成功完成三段 `Navigate`，生成 `2038` 帧、`40.76 s`、
`50 FPS` 的 LMDB episode 和两路 H.264 RGB 视频。对应 LeRobot v2.1 输出包含 `1` 个
episode、`2038` 帧、`1` 个 task 和 `2` 路视频。这是单 seed 流程验证，不代表多 seed
导航成功率。

三段均满足配置中的 `xy_goal_tolerance: 0.30 m` 和
`yaw_goal_tolerance: 0.12 rad`。Leg 1/2 使用下一段开始规划时记录的真实 base pose 核验，
Leg 3 使用 episode 最后一帧的 `states.base.position/orientation` 核验：

| 阶段 | 实测 `x, y, yaw` | 位置误差 | yaw 误差 | 判定 |
| ---- | ---------------- | -------- | -------- | ---- |
| Leg 1 | `[1.7762, 1.7969, 1.5609]` | `0.2688 m` | `0.0099 rad` | 容差内 |
| Leg 2 | `[3.1160, 1.5517, -1.5834]` | `0.2602 m` | `0.0126 rad` | 容差内 |
| Leg 3 | `[1.2825, 1.5553, 3.1318]` | `0.1399 m` | `0.0098 rad` | 容差内 |

因此该运行证明三段 controller 都进入目标容差并完成静止判定；它不表示机器人精确停在
YAML 给出的浮点坐标上。若下游任务要求更靠近家具，应单独收紧位置容差并重新验证稳定性。

## 6. Episode 内容

### 6.1 元数据

每个成功 episode 包含：

- language instruction 和 detailed language instruction；
- collect info 与版本；
- episode step 数；
- LMDB key 分组；
- RGB 图像对应的 step id；
- `lmdb/info.json`；
- `safety_events.jsonl`。

### 6.2 G1 state

| Key                          | 每步 shape | 含义                                        |
| ---------------------------- | ---------: | ------------------------------------------- |
| `states.body_joint.position` |     `[29]` | 29 个 body joint position                   |
| `states.body_joint.velocity` |     `[29]` | 29 个 body joint velocity                   |
| `states.base.position`       |      `[3]` | 世界坐标 base position                      |
| `states.base.orientation`    |      `[4]` | base quaternion                             |
| `qvel`                       |     `[32]` | base angular velocity 与 29D joint velocity |

### 6.3 Navigation 与动作候选

| Key                                  | 每步 shape | 含义                              |
| ------------------------------------ | ---------: | --------------------------------- |
| `base_actions.vx_body`               |     scalar | body-frame forward request        |
| `base_actions.vy_body`               |     scalar | body-frame lateral request        |
| `base_actions.wz_body`               |     scalar | body-frame yaw-rate request       |
| `base_actions.locomotion_mode`       |     scalar | `0=balance`、`1=walk`             |
| `master_actions.body_joint.position` |     `[29]` | WBC joint position target         |
| `master_actions.body_joint.velocity` |     `[29]` | WBC joint velocity target         |
| `master_actions.body_joint.effort`   |     `[29]` | WBC joint effort                  |
| `actions.body_joint.position`        |     `[29]` | logger next-state position label  |
| `actions.body_joint.velocity`        |     `[29]` | logger next-state velocity label  |
| `actions.base.position`              |      `[3]` | next-state base position label    |
| `actions.base.orientation`           |      `[4]` | next-state base orientation label |

数据同时保留 navigation request、WBC output 和 next-state label。训练配置应根据模型任务
选择监督层级；采集阶段不会把三类信号合并成含义不明确的统一 action。

### 6.4 相机

当前任务包含：

- `images.rgb.ego`：挂在 G1 `d435_link` 上的第一人称 RGB；
- `images.rgb.global`：固定全局 RGB；
- `camera2env_pose.ego` / `camera2env_pose.global`：每步 `[4,4]` 相机位姿。

当前任务没有记录 depth、segmentation 或 bbox。Logger 支持相关类型不代表当前 G1
episode 已包含这些标注。

## 7. 转换为 LeRobot v2.1

转换器沿用原项目 `policy/lmdb2lerobotv21/` 目录：

```bash
uv run --project policy/openpi-InternData-A1 \
  python policy/lmdb2lerobotv21/lmdb2lerobot_unitree_g1_a1.py \
  --src-path <包含成功-episode-目录的输入根目录> \
  --save-path tmp/g1_lerobot_conversion/unitree_g1_navigation_v21 \
  --repo-id local/unitree-g1-navigation \
  --fps 50 \
  --num-episodes 1
```

输入 episode 必须同时包含 LMDB、`meta_info.pkl` 和 ego/global RGB 视频。输出目录已存在时
转换器默认拒绝覆盖；只有明确传入 `--overwrite` 才会替换。

转换结果保留：

- 29D joint state 和 base state；
- navigation `vx/vy/wz` request；
- 29D WBC target/velocity/effort；
- logger next-state label；
- ego/global RGB；
- 展平为 16D 的两路 `camera2env` pose。

当前验证覆盖 converter schema/shape 测试和真实 G1 LMDB 读取；尚未把具体训练策略的 action
定义固化到转换器中。

## 8. 已验证样例

固定 seed `42` 的集成验证包括 A0-A9 和 B1-B5：

- A 类：直走、原地转向、斜向目标和横向目标；
- B 类：单障碍绕行、镜像绕行、墙端转向、S 型双墙路径和窄走廊。

代表性结果：

| Case | 内容              | 结果    | 关键指标                                                            |
| ---- | ----------------- | ------- | ------------------------------------------------------------------- |
| A0   | 直走到`(1,0,0)`   | success | 349 step；ego/global H.264 1280x720 50 FPS                          |
| A4   | 斜向`(1,1,pi/4)`  | success | final goal distance`0.0411 m`                                       |
| A6   | 横向目标`(0,1,0)` | success | min goal distance`0.0797 m`                                         |
| B1   | 单箱左侧绕行      | success | min goal distance`0.0043 m`；mean/max cross-track `0.0425/0.1036 m` |
| B4   | 两墙 S 路径       | success | min goal distance`0.0066 m`；mean/max cross-track `0.0446/0.1465 m` |
| B5   | 1.25 m 走廊       | success | min goal distance`0.0246 m`                                         |
| Scene 4 | 冰箱到架子、储物箱并返回 | success | 2038 step；40.76 s；两路 50 FPS 视频；LeRobot v2.1 转换成功 |

完整 A0-A9、B1-B5 matrix 的批量 runner、视频和日志属于开发期诊断产物，保存在 Git 忽略
的 `tmp/` 中，不是公开 API。仓库永久保留上表六个代表性 task YAML；对外可复现入口是
第 5 节的 `scripts/docker/run_simbox_task.sh`。Scene 4 的复现命令和数据转换方式见第 5.1 节。

这些结果只覆盖一组固定 seed，能够证明当前集成链路和代表性路径可运行，但不能替代多
seed、随机场景和大规模 dataset 的稳定性评估。

## 9. 当前限制

- 当前示例只覆盖简单 navigation，不包含 CuRobo 上肢 motion planning；
- G1 视觉仅包含 RGB，不包含 depth、segmentation 或 bbox；
- 训练目标尚未在采集层固定，应由具体模型选择 navigation request、WBC target 或
  next-state label；
- 正式数据生产前需要加载InternRobotics的环境资产进行仿真验证。
