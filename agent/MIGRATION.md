# InternDataEngine Agent、SimBox 与 CuRobo 能力迁移合并指南

## 1. 这份指南解决什么问题

这份指南面向新安装或重新下载 InternDataEngine 的维护者，用于把当前开发仓库中已经形成的能力，安全地迁移到一个干净仓库中。

迁移目标不是只让 `python -m agent` 能启动，而是恢复下面这条完整链路：

```text
自然语言任务
→ Agent 选择场景、对象、机器人和手臂
→ Workspace 生成底座候选
→ CuRobo planning-only 验证
→ 编译确定性的 SimBox task.yaml
→ Physics-schema Pick/Place 执行
→ CuRobo 轨迹和 Skill 目标可视化
→ LMDB、视频、运行事件和失败证据保存
→ 可选 Docker 多 GPU 并行数据生成
```

因此，不能只复制 `agent/`，也不能把整个旧工作区无选择地覆盖到新仓库。正确做法是按依赖顺序迁移以下五组能力：

1. CuRobo 本地改动；
2. SimBox 确定性规划与执行底层；
3. Workspace 与 Agent；
4. CuRobo 轨迹和 Skill 目标可视化；
5. Docker 多 GPU 并行生成。

## 2. 本次迁移范围

### 2.1 必须迁移

- `agent/` 中的 Agent 代码、Prompt、Registry、默认配置和本指南；
- Workspace 离线候选生成、候选写回和 CuRobo Probe；
- Physics-schema 碰撞世界、Pick/Place phase、执行监督与失败事件；
- CuRobo 的精确 collider 读取、GPU device 和 Piper 碰撞球改动；
- CuRobo 轨迹与 Pick/Place 目标可视化；
- LMDB、MP4、失败 episode 和结构化诊断保存；
- Docker 镜像、Compose、V2 并行调度器及其配置入口。

### 2.2 不作为代码迁移内容

- `test/`；
- `docs/` 下的历史分析、实验报告和阶段性说明；
- `output/`、`outputs/`、旧 run、视频和 LMDB；
- `configs/simbox/**/generated*` 等可重新生成的配置；
- `.docker/` 中的 Isaac cache；
- `.claude/`、`.codex/`、`.env` 和用户凭据；
- 与当前 Agent 无关的 Eval、StarVLA、邮件、Docker 之外的部署代码。

这里的“不迁移文档”不包括 `agent/prompts/`、`agent/registry/`、`agent/README.md` 和本文件。它们会直接影响 Agent 决策或是 Agent 的随附使用说明，属于运行组件。

### 2.3 必须另行准备的资产

以下目录不是普通 Git 代码，但运行时必须存在：

```text
InternDataAssets/Bench_2.1_isaacsim/scene_4/
InternDataAssets/curobo/
```

首批 20 个任务依赖 Bench2.1 `scene_4` 中的 USD、抓取姿态、对象配置、`attach_prim_path_children`、`container_regions` 和 Physics collider。资产应通过数据集下载、`rsync` 或独立制品包迁移，不要强制加入主仓库 Git。

## 3. 推荐的版本与提交组织

### 3.1 源仓库先整理成提交

源工作区如果仍有未提交修改，先建立专用分支，并按能力拆成提交：

```text
commit A: feat(curobo): CuRobo physics-schema 与 Piper 修复
commit B: feat(simbox): 确定性 Pick/Place、诊断和可视化底层
commit C: feat(agent): Workspace 与配置驱动 Agent
commit D: feat(parallel): Docker 多 GPU 并行生成
```

CuRobo 位于 `InternDataAssets/curobo`，它是独立 Git 仓库；commit A 必须在该仓库内创建。commit B、C、D 属于 InternDataEngine 主仓库。

不要执行：

```bash
git add -A
git add -f InternDataAssets
```

每次只 stage 当前提交的路径，并在 commit 前执行：

```bash
git diff --cached --name-status
git diff --cached --stat
git diff --cached --check
```

### 3.2 目标仓库建立集成分支

在新安装的目标仓库中，不要直接在 `main` 上覆盖文件：

```bash
cd <TARGET_REPO>
git status --short --branch
git switch -c integrate-agent-runtime
```

如果源代码已经整理成 Git 提交，优先使用 `cherry-pick`：

```bash
git remote add migration-source <SOURCE_REPO_OR_REMOTE>
git fetch migration-source

git cherry-pick <SIMBOX_COMMIT>
git cherry-pick <AGENT_COMMIT>
git cherry-pick <PARALLEL_COMMIT>
```

如果目标仓库版本比源仓库新，发生冲突时不要简单选择 `ours` 或 `theirs`。应按照第 10 节的职责边界进行三方合并。

## 4. 第一步：迁移 CuRobo 本地改动

### 4.1 文件清单

在 `InternDataAssets/curobo` 独立仓库中迁移：

```text
src/curobo/types/base.py
src/curobo/util/usd_helper.py
src/curobo/content/configs/robot/piper100_left_arm.yml
src/curobo/content/configs/robot/piper100_right_arm.yml
src/curobo/content/configs/robot/spheres/piper100_collision_audited_20260720.yml
```

这些改动分别负责：

- `types/base.py`：默认使用 PyTorch 当前 CUDA device，避免容器或 `CUDA_VISIBLE_DEVICES` 隔离后仍写死 `cuda:0` 的物理含义；
- `usd_helper.py`：从明确的 Physics collider Prim 列表构建 CuRobo world，并支持非三角面 CPU triangulation；
- `piper100_*_arm.yml`：使用审计后的 Piper 碰撞球，并把 attached object sphere 容量设为当前 Pick/Place 契约需要的值；
- `piper100_collision_audited_20260720.yml`：Piper 左右臂共用的碰撞球模型。

### 4.2 合并方式

如果有 CuRobo commit 或 bundle：

```bash
cd <TARGET_REPO>/InternDataAssets/curobo

git fetch <CUROBO_REMOTE_OR_BUNDLE> <CUROBO_BRANCH>
git switch -c internrobotics-physics-schema-v1
git cherry-pick <CUROBO_COMMIT>
```

如果只能复制文件，应先确认目标 CuRobo 的版本与源仓库兼容，再逐文件合并；不要直接用一个来源不明的新 `usd_helper.py` 覆盖目标版本。

### 4.3 恢复软链接和 editable 安装

`workflows/simbox/curobo` 应使用相对链接，不能复制旧机器上的绝对软链接：

```bash
cd <TARGET_REPO>

if [ -L workflows/simbox/curobo ]; then
  unlink workflows/simbox/curobo
fi
ln -s ../../InternDataAssets/curobo workflows/simbox/curobo

conda run -n interndata python -m pip install -e InternDataAssets/curobo
```

确认解释器实际加载的是目标仓库中的 CuRobo：

```bash
conda run -n interndata python -c \
  'from pathlib import Path; import curobo; print(Path(curobo.__file__).resolve())'
```

## 5. 第二步：迁移 SimBox 确定性底层

这一层负责把 Agent 生成的高层 Pick/Place 配置变成安全、可诊断的真实执行。它不是可选优化；缺少其中任何一块，都可能出现“配置能生成但执行语义已经退回旧版”的情况。

### 5.1 新增目录

```text
workflows/simbox/core/planning/
workflows/simbox/core/execution/
workflows/simbox/core/visualization/
```

职责如下：

- `planning/`：Physics/CuRobo 碰撞世界、物体状态机、`MotionPhaseCommand`、规划契约和抓取联合路径评估；
- `execution/`：关节/EE 跟踪、底座漂移、接触、携物滑移、动态 world 和有界重规划；
- `visualization/`：最终 CuRobo 轨迹、Pick/Place 目标、pre-grasp 和 pre-place 可视化。

### 5.2 新增工具模块

```text
workflows/simbox/core/utils/asset_path_utils.py
workflows/simbox/core/utils/attach_collision_utils.py
workflows/simbox/core/utils/episode_event_writer.py
workflows/simbox/core/utils/joint_index_resolver.py
```

### 5.3 必须三方合并的现有文件

```text
configs/simbox/de_plan_with_render_template.yaml
nimbus_extension/components/load/env_loader.py
nimbus_extension/components/store/env_writer.py
workflows/base.py
workflows/simbox_dual_workflow.py
workflows/simbox/core/cameras/custom_camera.py
workflows/simbox/core/configs/robots/split_aloha.yaml
workflows/simbox/core/controllers/splitaloha_controller.py
workflows/simbox/core/controllers/template_controller.py
workflows/simbox/core/loggers/__init__.py
workflows/simbox/core/loggers/lmdb_logger.py
workflows/simbox/core/objects/plane_object.py
workflows/simbox/core/objects/rigid_object.py
workflows/simbox/core/robots/template_robot.py
workflows/simbox/core/skills/__init__.py
workflows/simbox/core/skills/base_skill.py
workflows/simbox/core/skills/pick.py
workflows/simbox/core/skills/place.py
workflows/simbox/core/tasks/banana.py
workflows/simbox/core/utils/plan_utils.py
```

### 5.4 新增 Skill

```text
workflows/simbox/core/skills/observe_hold.py
workflows/simbox/core/skills/pick_plan_probe.py
```

当前 Agent 正式可编排的机器人执行 Skill 仍只有 `pick` 和 `place`。`pick_plan_probe` 是 Workspace 验证内部使用的 planning-only Skill；`observe_hold` 是观察/调试基础设施，不应被 LLM 当成普通任务 Skill 自由选择。

### 5.5 合并后必须保留的语义

合并冲突时至少保留以下行为：

1. `physics_schema` 是 Agent 生成任务的碰撞世界；
2. CuRobo world 来自真实 enabled Physics collider，而不是资产名称过滤；
3. world collider 与 attach proxy 是两个不同契约；
4. Pick 按 `pre-grasp → terminal grasp → contact → attach → lift` 执行；
5. Place 按 `pre-place → descent → release → detach/settle → retreat` 执行；
6. 轨迹执行结束不能单独代表机械臂真实到位；
7. SplitAloha 操作时锁定移动底座自由度；
8. 失败 episode 可以保存结构化证据；
9. 物理 GPU 与进程内逻辑 CUDA device 分开处理。

## 6. 第三步：迁移 Workspace 与 Agent

### 6.1 Agent 目录

迁移整个 `agent/`，但不要带入缓存、凭据或运行输出：

```text
agent/
├── README.md
├── MIGRATION.md
├── __init__.py
├── __main__.py
├── compiler.py
├── config.yaml
├── contracts.py
├── evidence.py
├── inventory.py
├── orchestrator.py
├── resolver.py
├── retention.py
├── settings.py
├── prompts/
├── registry/
├── experience/
└── task_ir/
```

不要迁移旧版入口：

```text
agent/task_generator.py
agent/read.md
agent/prompts/system.txt
agent/prompts/user_template.txt
```

`agent/prompts/` 和 `agent/registry/` 不是普通说明文档：规划规则、中心物品选择、左右臂顺序、Pick/Place 参数合法性和失败码都由这些文件约束，必须随代码迁移。

### 6.2 Workspace 文件

```text
workflows/simbox/core/workspace/
workflows/simbox/core/utils/workspace_planner.py
scripts/simbox/plan_workspace_layout.py
scripts/simbox/validate_workspace_candidates.py
scripts/docker/run_simbox_task.sh
scripts/docker/prepare_simbox_run.py
configs/simbox/de_workspace_probe_template.yaml
```

Agent 对它们的依赖顺序是：

```text
TaskPlan 中先确定每个 Subtask 的 left/right arm
→ 为每个中心物品生成 annulus 候选
→ 几何碰撞过滤
→ 寻找服务所有中心物品的共同底座位姿
→ 用预先指定的手臂执行 CuRobo planning-only Probe
→ 通过后编译最终 SimBox YAML
```

当前没有 Nav。多个中心物品找不到共同底座位姿时必须停止，不能静默插入开环底盘动作。

### 6.3 可选的 TaskIR 工具

如果需要继续维护任务/资产知识聚合，再迁移：

```text
scripts/task_ir_aggregate_assets.py
scripts/task_ir_aggregate_skills.py
scripts/task_ir_batch_roundtrip.py
scripts/task_ir_coverage_matrix.py
scripts/task_ir_export.sh
scripts/task_ir_roundtrip.sh
scripts/task_ir_to_yaml.sh
```

这些工具不是 `python -m agent run` 的启动前提。

## 7. CuRobo 轨迹和 Skill 目标可视化

### 7.1 代码组成

```text
workflows/simbox/core/visualization/curobo_trajectory.py
workflows/simbox/core/visualization/trajectory_math.py
workflows/simbox/core/visualization/skill_targets.py
workflows/simbox/core/visualization/skill_target_math.py
```

还依赖：

```text
workflows/base.py
workflows/simbox_dual_workflow.py
workflows/simbox/core/controllers/template_controller.py
workflows/simbox/core/skills/base_skill.py
workflows/simbox/core/skills/pick.py
workflows/simbox/core/skills/place.py
```

只复制 `core/visualization/` 不会生效，因为可视化器必须由 Workflow 创建、由 Controller/Skill 写入，并由异步保存快照导出。

### 7.2 YAML 开关

任务 YAML 需要包含：

```yaml
visualization:
  curobo_trajectory:
    enabled: true
    export_usd: true
    accumulate_within_episode: true
    show_ee_path: true
    show_robot_spheres: false
    ee_sample_count: 64
    robot_pose_sample_count: 8
    ee_radius_m: 0.02
    ee_min_center_spacing_m: 0.06
    ee_color: [1.0, 0.35, 0.0, 1.0]
    robot_color: [0.1, 0.85, 0.25, 0.28]
    robot_radius_scale: 1.0

  skill_targets:
    enabled: true
    export_usd: true
    retain_completed: true
    completed_opacity_scale: 0.25
    pick:
      enabled: true
      show_pregrasp: true
      line_width_m: 0.006
    place:
      enabled: true
      show_preplace: true
      line_width_m: 0.006
      plane_normal_offset_m: 0.003
      min_display_extent_m: 0.08
```

episode 保存后，启用的可视化会产生：

```text
trajectory_debug.usda
skill_targets_debug.usda
```

`trajectory_debug.usda` 只记录最终被 CuRobo 接受并交给 Controller 的轨迹。如果日志是 `pre=0`，说明不存在成功的 pre-grasp 路径，因此不会凭空出现一条橙色轨迹；这类失败需要结合目标可视化和 CuRobo status 诊断。

### 7.3 当前 Agent 接入状态

当前 Compiler 会复制源任务 YAML，所以源任务带有 `visualization` 时会保留；但是 `agent/config.yaml` 里的统一可视化默认值尚未自动注入所有 Agent 生成配置。

若目标安装要求“每次 Agent run 默认可视化”，应在后续独立提交中完成：

1. 在 `agent/config.yaml` 增加 compiler-owned `visualization` 默认值；
2. 在 `agent/compiler.py` 中把该配置合并到生成的 `attempts/<n>/task.yaml`；
3. 不允许 LLM 任意修改调试 Prim 名称、碰撞排除或物理开关。

在这项接入完成前，不要把“底层支持可视化”描述成“Agent 已对所有任务默认开启可视化”。

## 8. Docker 多 GPU 并行生成

### 8.1 设计位置

Docker 并行生成与 Agent 是并列的上层入口：

```text
Agent：一个自然语言任务 → 一个确定性 YAML → 单 GPU episode → 失败闭环
Docker Parallel V2：一组已有 YAML → 多 GPU/多容器调度 → 数据与统计汇总
```

当前不要让 Agent 直接承担多 GPU 调度。先由 Agent 生成并验证 YAML，再把稳定 YAML 放入 Parallel V2 的 `jobs`。

### 8.2 必须迁移的 Docker 文件

```text
.dockerignore
docker/docker-compose.simbox.yml
docker/isaac/Dockerfile
docker/isaac/entrypoint.sh
docker/isaac/requirements.isaac.txt
```

### 8.3 必须迁移的调度代码

```text
launcher.py
scripts/simbox/simbox_parallel_v2.py
scripts/simbox/simbox_parallel_generate.sh
scripts/simbox/simbox_parallel_api.py
scripts/simbox/simbox_dataset_stats.py
scripts/simbox/simbox_docker_smoke.sh
scripts/simbox/simbox_plan_with_render_multi_gpu.sh
```

其中 `launcher.py` 必须三方合并：保留原来的普通 DataEngine 启动路径，同时根据顶层 `launcher_type: simbox_parallel_v2` 分发到并行调度器。不要用 Parallel 版本覆盖掉普通启动逻辑。

### 8.4 配置文件

必须迁移：

```text
configs/simbox/parallel_generate_v2.yaml
configs/simbox/de_plan_with_render_template.yaml
```

以下是示例配置，可按实际需要迁移：

```text
configs/simbox/parallel_franka_4jobs_20.yaml
configs/simbox/parallel_franka_12jobs_10h.yaml
configs/simbox/parallel_franka_open_microwave_smoke_1.yaml
```

### 8.5 Docker 前置条件

宿主机必须满足：

- NVIDIA Driver 正常；
- `nvidia-smi` 能看到目标 GPU；
- Docker daemon 正常；
- Docker Compose plugin 可用；
- NVIDIA Container Toolkit 已配置；
- 当前用户能够访问 `/var/run/docker.sock`；
- `InternDataAssets/curobo` 已经合并本指南第 4 节的改动。

Dockerfile 在构建时执行 `COPY InternDataAssets/curobo /opt/curobo`。因此 CuRobo 改动后必须重新构建镜像；仅修改宿主机 CuRobo 不会更新旧镜像。

构建：

```bash
cd <TARGET_REPO>
docker compose -f docker/docker-compose.simbox.yml build isaac
```

### 8.6 推荐运行方式

先复制一份本机配置：

```bash
cp configs/simbox/parallel_generate_v2.yaml \
  configs/simbox/parallel_local.yaml
```

最小配置结构：

```yaml
launcher_type: simbox_parallel_v2
name: local_parallel_run

parallel:
  backend: docker
  gpus: [0, 1, 2, 3]
  workers_per_gpu: 1
  run_id: null
  compose_file: docker/docker-compose.simbox.yml
  task_timeout_sec: 0
  stats_after_run: true
  dry_run: false

defaults:
  de_config: configs/simbox/de_plan_with_render_template.yaml
  random_num: 10
  dataset_root: null
  seed_base: 1000
  shard_random_num: false

jobs:
  - id: task_gpu0
    task: workflows/simbox/core/configs/tasks/<task>.yaml
    gpu: 0
```

先 dry-run：

```bash
python3 launcher.py \
  --config configs/simbox/parallel_local.yaml \
  --parallel.dry_run=true \
  --monitor.enabled=false \
  --parallel.run_id=dryrun_check
```

查看：

```bash
cat output/_parallel_runs/dryrun_check/run_report.md
cat output/_parallel_runs/dryrun_check/manifest.jsonl
```

正式执行：

```bash
python3 launcher.py --config configs/simbox/parallel_local.yaml
```

一个 YAML 拆到多张 GPU：

```yaml
jobs:
  - id: task_sharded
    task: workflows/simbox/core/configs/tasks/<task>.yaml
    allowed_gpus: [0, 1, 2, 3]
    random_num: 40
    shard_random_num: true
```

建议先使用 `workers_per_gpu: 1`。一张 GPU 同时运行多个 Isaac 容器会显著增加显存、启动时间和 IO 压力。

### 8.7 输出与恢复

训练数据和调度记录是两套目录：

```text
<dataset_root>/...                       # LMDB、meta_info.pkl、MP4
output/_parallel_runs/<run_id>/...       # 状态、日志、统计、报告
```

主要状态文件：

```text
status.json
manifest.jsonl
episode_events.jsonl
job_summary.csv
gpu_samples.csv
run_report.md
run_report.json
```

异常退出后恢复汇总并清理残留容器：

```bash
python3 scripts/simbox/simbox_parallel_v2.py \
  --config configs/simbox/parallel_local.yaml \
  --recover-run-id <run_id>
```

## 9. 新安装后的 Agent 使用方法

### 9.1 前置检查

```bash
cd <TARGET_REPO>

codex --version
conda run -n interndata python -c 'import isaacsim; import curobo; import scipy; print(scipy.__version__)'
```

Codex CLI 必须已登录并能执行 `codex exec`。Agent 会把 Codex 当作只读的结构化决策后端；场景写入、参数校验和 YAML 编译仍由确定性代码完成。

### 9.2 建立 inventory

首次使用或场景、任务、资产变化后执行：

```bash
conda run -n interndata python -m agent index
```

`agent/config.yaml` 中的 `scene_roots` 必须指向目标安装中真实存在的场景目录。

### 9.3 只规划

```bash
conda run -n interndata python -m agent plan \
  --prompt "把白色杯子放到托盘里"
```

这一步调用 Codex、生成 TaskPlan 和几何 Workspace 候选，但不启动完整 SimBox episode。

### 9.4 规划并执行、默认生成数据

```bash
conda run -n interndata python -m agent run \
  --prompt "把白色杯子放到托盘里" \
  --gpu 0 \
  --max-revisions 2
```

默认值由 `agent/config.yaml` 控制：

```yaml
generation:
  enabled: true
  random_num: 1
```

数据保存在：

```text
output/agent_runs/<run_id>/attempts/<attempt>/data/
```

### 9.5 恢复和重新诊断

```bash
conda run -n interndata python -m agent resume \
  --run-id <run_id> \
  --gpu 0

conda run -n interndata python -m agent diagnose \
  --run-dir output/agent_runs/<run_id>
```

## 10. 合并冲突处理规则

### 10.1 `launcher.py`

必须同时保留：

- 普通 DataEngine 配置的原启动流程；
- 读取顶层 `launcher_type`；
- `simbox_parallel_v2` 分发；
- `--random_seed` 和额外 dot-override 参数。

### 10.2 `env_loader.py` 与 GPU

必须保留：

```text
active_gpu：Isaac/渲染使用的物理 GPU
physics_gpu：进程可见空间内的 PhysX device
cuda_device：进程可见空间内的 PyTorch/CuRobo device
```

Docker 通过 `CUDA_VISIBLE_DEVICES` 让容器只看到一张卡后，`cuda_device` 通常应为 `0`，不能继续使用宿主机物理编号。

### 10.3 `template_controller.py`

这是高冲突文件。合并时不能只保留函数名，必须保留以下完整契约：

- Physics-schema world 初始化和动态同步；
- `MotionPhaseCommand`；
- chained pre-grasp → grasp 规划；
- selected trajectory 可视化；
- attach/detach 与 carried-object spheres；
- 实际关节和 EE 到位判定；
- replanning 所需的当前状态入口。

### 10.4 `simbox_dual_workflow.py`

必须保留：

- CollisionSceneManager 初始化；
- SafetyMonitor 与 ExecutionSupervisor；
- 每步 safety precheck；
- 失败 episode 保存；
- `episode_events.jsonl`；
- trajectory/skill-target/audit/safety 文件导出；
- workflow close，避免视频 writer 或 USD layer 泄漏。

### 10.5 `pick.py` 与 `place.py`

不要把新版 phase 路径和旧 tuple 路径混成一套模糊逻辑。`physics_schema` 走新版结构化命令，legacy 兼容路径只服务明确声明的旧任务。

### 10.6 `agent/compiler.py`

必须保留“源任务只读、最终 YAML 写入 run 目录”的原则。LLM 只提供合法的语义选择和显式覆盖，不允许直接控制：

- Physics collider；
- attach Prim；
- `test_mode`；
- 任意 `ignore_substring`；
- 文件系统输出路径；
- 未注册 Skill。

## 11. 不迁移测试代码时的最小验收

本指南不要求把 `test/` 复制到目标安装，但合并完成后仍应执行下面的最小运行检查。

### 11.1 静态入口检查

```bash
conda run -n interndata python -m compileall -q agent \
  workflows/simbox/core/planning \
  workflows/simbox/core/execution \
  workflows/simbox/core/workspace \
  workflows/simbox/core/visualization

conda run -n interndata python -m agent --help
python3 launcher.py --config configs/simbox/parallel_generate_v2.yaml \
  --parallel.dry_run=true \
  --monitor.enabled=false \
  --parallel.run_id=migration_dryrun
```

### 11.2 Agent 规划检查

```bash
conda run -n interndata python -m agent index

conda run -n interndata python -m agent plan \
  --prompt "把白色杯子放到托盘里"
```

应至少产生：

```text
output/agent_inventory.json
output/agent_runs/<run_id>/request.json
output/agent_runs/<run_id>/selection.json
output/agent_runs/<run_id>/task_plan.json
output/agent_runs/<run_id>/workspace_manifests.json
```

### 11.3 完整运行检查

完整运行可能因为任务几何或 CuRobo 可达性而安全失败；迁移验收首先检查链路是否完整，而不是强制要求每个任务成功。至少确认：

- Codex 结构化返回成功；
- Workspace 脚本能启动；
- CuRobo 来自目标仓库；
- 指定 GPU 没有 invalid device ordinal；
- `task.yaml` 能生成；
- episode 成功或失败时都有 `episode_events.jsonl`；
- 进入保存阶段后，LMDB/MP4/诊断文件写入预期目录；
- 开启可视化且存在成功轨迹时导出 debug USD。

## 12. 当前能力边界

- 首批 inventory 是 Bench2.1 四个场景、20 个任务，但架构不能按这 20 个任务写专属分支；
- 当前只开放标准 Pick 和 Place；
- `dual_arm_simultaneous` 显式关闭；
- 多个 Subtask 可以左右臂先后执行并编译进同一个 YAML；
- 当前没有 Nav，没有共同底座位姿时停止；
- 数据生成默认开启；
- Agent 失败闭环已有结构化证据和有界修订，但部分 CuRobo 失败仍需要更细的 per-goal status；
- Agent 的 Workspace Probe 和正式任务只通过 `scripts/docker/` 单栈流程执行，
  每个 attempt 使用独立 Compose project、ROS domain 和 Isaac/Nav2 容器；
- Docker Parallel V2 仍是已有 YAML 的批量调度入口，不由 Agent orchestrator 调用；
- CuRobo 轨迹可视化底层可用，Agent 统一默认注入仍应作为独立改动完成。

## 13. 最终交付检查表

### Git 代码

- [ ] InternDataEngine 主仓库包含 SimBox 确定性底层提交；
- [ ] 主仓库包含 Agent/Workspace 提交；
- [ ] 主仓库包含 Docker Parallel 提交；
- [ ] CuRobo 独立仓库包含 Physics-schema/Piper 提交；
- [ ] `git diff <target-base>..HEAD` 中没有历史输出和大体积资产。

### 资产和环境

- [ ] `InternDataAssets/Bench_2.1_isaacsim/scene_4` 存在；
- [ ] `InternDataAssets/curobo` 存在；
- [ ] `workflows/simbox/curobo` 是指向当前仓库资产的相对链接；
- [ ] `interndata` 环境中的 `curobo` editable install 指向当前仓库；
- [ ] Codex CLI 已安装和登录；
- [ ] Docker/NVIDIA Container Toolkit 可用；
- [ ] CuRobo 改动后 Docker Isaac 镜像已重新构建。

### 使用入口

- [ ] `python -m agent --help` 正常；
- [ ] `python -m agent index` 正常；
- [ ] `python -m agent plan` 能生成 TaskPlan 和 Workspace manifest；
- [ ] `python -m agent run` 能进入 required-arm CuRobo Probe；
- [ ] Parallel V2 dry-run 能生成 run report；
- [ ] 数据目录与调度报告目录职责清楚；
- [ ] 可视化开关和当前 Agent 默认接入边界已向使用者说明。
