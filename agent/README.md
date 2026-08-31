# InternDataEngine 通用任务 Agent

新安装仓库需要迁移或合并 Agent、Workspace、SimBox/CuRobo 底层和 Docker 并行生成能力时，先阅读 [MIGRATION.md](./MIGRATION.md)。

## 1. 项目定位

这个目录实现一个配置驱动的机器人任务 Agent。用户输入自然语言任务，Agent 负责选择现有场景和资产、确定机器人与工作位姿、拆分对象子任务、编排机器人 Skill、生成 SimBox 配置、启动数据生成，并根据运行证据完成有界失败闭环。

系统的长期目标不是逐个适配场景，而是：

```text
任意合适的任务需求
→ 检索现有场景、资产和Skill
→ 复用现有场景，或规划如何组合新场景
→ 确定配置、机器人、位姿、手臂和Skill顺序
→ 执行任务并生成数据
→ 根据证据修订和积累经验
```

当前 Bench2.1 的 20 个任务只是首批验证集。代码中不得出现按这 20 个任务名称切换行为的专属适配逻辑。新增场景、任务和资产后，应通过重新建立 inventory 进入 Agent 的候选空间。

## 2. 始终有效的开发约束

下面的约束是开发期间必须持续遵守的项目规则，不因具体任务和后续对话而消失。

### 2.1 Agent 的主要产物是配置

Agent 的主要职责是生成或修订配置，而不是无限制编写策略代码。配置必须明确：

- 待操作对象、目标对象和空间关系；
- 选择的场景、资产、机器人类型、机器人位姿和左右臂；
- 任务包含哪些逐对象子任务；
- 每个子任务包含哪些阶段；
- 每个阶段使用 1–2 个可执行机器人 Skill；
- Skill 使用顺序、输入参数、前置条件和成功标准；
- 选择 Skill、顺序和参数的决策依据；
- Place 到哪里、为何选择该位置以及用什么 predicate 判断成功。

LLM 可以利用语义能力选择 Skill 和总体顺序，但对象引用、位姿、几何、参数范围、配置写入和合法性检查必须交给确定性代码。

### 2.2 Skill 指机器人执行技能

本项目中的 Skill 是机器人运行时执行能力，例如 Pick、Place、感知、移动和未来积累的新 Pick/Place 变体；它不是 Codex 或其他 Agent 平台的系统级 Skill。

Agent 编排 Pick/Place 等高层 Skill。Skill 内部的接近、闭合、接触确认、attach、下降、释放、detach、settle、retreat 和安全重规划由 SimBox 负责，不能在每个任务 YAML 中重复实现。

### 2.3 碰撞世界和执行安全是确定性基础设施

标准任务使用 `physics_schema`：

- PhysX 中启用的 Physics collider 是碰撞事实来源；
- CuRobo world 必须和真实 collider 一致；
- 操作对象必须区分刚体根、全部 world collider 和 attach proxy；
- Agent 不根据 `sink/table/counter` 等名字删除障碍物；
- 必要排除只能使用完整 Prim path 和非空 reason；
- Pick/Place 使用结构化对象身份和运动阶段；
- 轨迹执行、接触、底座漂移、物体滑移和动态 world 由 SimBox 安全监督器处理。

Agent 不能为了提高成功率隐藏真实障碍、关闭安全检查或静默回退到 legacy 模式。

### 2.4 失败必须形成闭环

失败诊断必须能区分至少以下情况：

- 配置字段、对象引用、路径或资产契约错误；
- 没有合适的机器人 workspace candidate；
- 没有抓取候选；
- 有抓取候选但没有 CuRobo/IK 联合路径；
- 规划成功但轨迹执行失败；
- 规划轨迹与真实关节或 EE 轨迹不一致；
- 机械臂或携物碰到障碍；
- 夹爪没有碰到目标、抓漏、没有抬起或抓起后掉落；
- Place 目标区域无效、pre-place/terminal plan 失败；
- 释放、detach、settle 或最终位置/姿态 predicate 失败；
- 运行超时、证据缺失和基础设施错误。

诊断优先读取 `collision_world_audit.json`、`object_state_events.jsonl`、`safety_events.jsonl`、episode failure reason、LMDB 和任务 predicate。视频和关键帧用于补充视觉类或证据冲突问题，不能覆盖明确的数值事实。

SimBox phase 内部重规划与 Agent 配置修订是两层预算。phase 内部恢复不消耗 Agent 修订次数；一个对象默认首次执行加最多两次 Agent 修订。每次修订只验证一个主要根因，不能同时无边界修改多个参数。

### 2.5 经验必须判断是否值得保留

运行经验由 Agent 选择以下一种形式：

1. `playbook`：跨场景成立的排障方法、证据链或决策原则；
2. `debug_tool`：可重复使用的测试脚本、检查函数或模块，必须有清晰接口和说明；
3. `robot_skill`：真正改变机器人执行行为的新 Skill，按 pick/place/perception 等类别保存并使用语义化名称；
4. `none`：证据不足、随机性太强或没有复用价值。

例如，观察到橙色 CuRobo 轨迹与机器人实际轨迹不一致时，应先形成关节/EE/base tracking 的通用排查方法和调试工具。只有改进后的执行方法在来源任务和第二任务上稳定成功、并且原 Skill 不回退时，才能晋升为新的 Pick-family Skill。

候选机器人 Skill 的晋升门槛是：

- schema、单元和接口测试通过；
- 来源任务跨 seed 3/3 成功；
- 第二对象或任务至少 2/3 成功；
- 原始 Skill 基线无回退。

### 2.6 控制项目规模

本项目借鉴 CaP-Agent0 的多轮证据闭环、成功经验提取和可恢复运行，但不复制任意 Python 策略搜索、RL、Web UI、多仿真器和大规模并行 Agent 系统。

实现必须保持“做一点就能用一点”：

- 优先复用 SimBox 已有 task parser、workspace planner、Pick/Place、诊断产物和启动器；
- 文件解耦但不过度拆分；
- 新代码必须能跨任务复用；
- 不因一次任务失败无限扩大基础设施；
- 配置生成是主目标，代码只服务于调试工具、确定性模块和新机器人 Skill。

## 3. 运行链路

```text
TaskRequest
→ Inventory / Normalize 得到 SceneSpec 与 SupportGraph
→ LLM 只生成与机器人型号无关的 SemanticTaskPlan
→ EmbodimentResolver 绑定 profile、instance 和每个 Subtask 的手臂
→ PlacementAdapter 生成机器人候选位姿
→ 必要时 SceneLayout 对派生场景做 K=8、G=5、四队列演化
→ 确定性编译与 schema / support / spawn / collision 检查
→ required-arm Pick+Place CuRobo gate
→ 单机器人、单 episode 顺序执行全部 Subtask
→ Structured Trace / Evidence / Failure Classification
→ typed layout、Skill 或 next-candidate 修复（最多2次）
→ debug seed 候选与 held-out qualification 分离
```

机器人按放置方式走两套几何适配器，而不是按型号写主流程分支：

- `floor_standing`：SplitAloha、Lift2。机器人 region 相对 floor，候选围绕目标工作面采样，并检查底盘 footprint、上层 collision layer 和臂基座偏移；操作期间底盘和升降自由度锁定。
- `support_mounted`：FR3、FrankaRobotiq85。机器人 region 相对真实安装支撑面，底座 footprint 必须完整落在实测支撑多边形内；支撑面仍保留在 Physics/CuRobo 碰撞世界中。

具体手臂、相机、CuRobo、数据 schema 和数值几何全部来自 canonical robot profile。同一 SemanticTaskPlan 可以派生多个 profile variant；只有 admission 状态允许的 variant 才能进入 episode，`implemented` 不等于已支持，`qualified` 必须有 held-out 原始 artifact 证据。

没有现有场景覆盖任务时，Agent 输出 `SceneCompositionRequest`，记录建议基础场景、所需资产、空间关系、机器人能力和验证要求。v1 不自动装配和运行新场景。已有场景出现布局、支撑、遮挡或共同工作空间失败时，Orchestrator 调用受类型约束的 SceneLayout 搜索，生成派生 scene revision，依次经过静态检查、Workspace 和 Pick+Place CuRobo gate；没有通过 gate 的候选才以明确 failure code 阻断。

跨 workspace 自动导航不属于当前能力。可靠 Nav 接入后，无共同位姿的分支才改为“执行前一 Subtask → Nav 到下一已验证点位 → 继续执行”；接入前不使用开环底盘动作伪装成 Nav。

### 3.1 已确认的能力边界与待实现项

- `dual_arm_simultaneous` 继续关闭，但保留 TaskPlan 表达和显式报错；未来完成双臂并发碰撞世界、执行和失败恢复验证后再开放。
- 语义计划使用 `any_single_arm` 或用户明确的左右臂约束；确定性 `ExecutionVariant` 在 Workspace 之前绑定 profile、instance 和每个 Subtask 的具体手臂。多个中心物品在同一 variant 中共用一个已验证底座位姿并顺序执行。
- 数据生成默认开启，由 `config.yaml` 的 `generation.enabled` 控制。Agent 仅在用户明确要求生成或不生成时写布尔覆盖，否则返回 `null`，由 Orchestrator 使用默认值。
- 严格成功要求 finalized task predicate、零 safety event、Physics/CuRobo exact audit；请求数据生成时还要求 action/state、相机视频、LMDB 与 metadata 完整。确定性 failure router 先于受限 LLM diagnosis。

## 4. 目录职责

- `contracts.py`：稳定 Pydantic 数据契约；
- `inventory.py`：扫描场景、任务、资产和已有 Skill；
- `resolver.py`：候选过滤、Codex 结构化选择和计划；
- `compiler.py`：TaskPlan、workspace candidate 到 SimBox YAML；
- `orchestrator.py`：共同位姿选择、单 YAML 执行、有界修订和恢复状态机；
- `evidence.py`：归一化 SimBox 诊断产物并确定主失败；
- `retention.py`：经验分流、候选记录和晋升门槛；
- `task_ir/`：对原生任务 YAML 的紧凑、无损表示，供知识聚合脚本复用；
- `workflow/`：语义规划策略与结构化模板；
- `tools/`：scene ingest、SceneSpec/SupportGraph、SceneLayout、演化调度、trace 和 failure routing；
- `robot_skills/`：Agent 可编排的机器人 Skill 契约和 qualification admission；
- `registry/`：语义别名和失败码；
- `experience/`：通用 playbook、调试工具候选和经验索引。

规划阶段加载 `workflow/templates/plan.md`、`workflow/task_planning_policy.md` 和
`workflow/object_role_policy.md`。物体/机器人布局不在 Prompt 中计算，由 `tools/scene_layout/` 处理。
参数的机器可读类型、枚举、范围和所有权以 `robot_skills/contracts.yaml` 为准，
`resolver.py` 必须在配置编译前强制校验，不能只依赖 LLM 遵守文字说明。

### 4.1 配置分工和覆盖顺序

为保持简单，当前不把默认值拆成许多小文件，而是采用四个职责明确的入口：

- `config.yaml`：Agent 全局运行策略、执行预算、数据生成、共同位姿阈值、Physics Schema 和 Skill 默认参数；
- canonical robot YAML：机器人 profile、放置族、臂、相机、数据 adapter 和 capability 的唯一数值真源；
- `robot_skills/contracts.yaml` 与 `registry/robot_admissions.yaml`：参数所有权与可执行 qualification 状态；
- `SceneCapabilityManifest` 与源任务 YAML：场景实例、对象、区域和资产能力；
- `TaskPlan`：对象、关系、Skill 顺序和机器人 capability 需求；`ExecutionVariant` 保存 profile、instance 和 arm binding。

最终 YAML 一律由 `compiler.py` 生成。普通 Skill 参数采用“全局默认 → relation 默认 → Agent 明确覆盖”的顺序；`test_mode`、碰撞世界和资产 Prim path 等 compiler/asset-owned 字段不能被 Agent 覆盖。这样 Agent 不直接编辑源 YAML，也不会把默认参数重复写进每个 TaskPlan。

## 5. 使用方法

Agent 控制进程可继续在服务器仓库根目录的轻量 `interndata` Python 环境中运行。
依赖 Isaac Sim 和 CuRobo 的仿真、规划与验证默认通过
`scripts/docker/up_simbox_isaac.sh` 启动独立 Docker stack；也可以显式选择
`execution.simulator_backend: conda`，由
`scripts/simbox/run_simbox_task.sh` 在完整的 `interndata-isaac6` 环境中运行。
两条路径都固定使用 `InternDataAssets/curobov2`。

首次使用或场景/资产变化后建立 inventory：

```bash
conda run -n interndata python -m agent index
```

两步走战略：

```bash
# Step 1: 只规划（指定 run-id 方便后续引用）
conda run -n interndata python -m agent plan \
  --prompt "把白色杯子放到托盘里" \
  --run-id cup_to_tray

# Step 2: 复用规划结果，直接启动仿真
conda run -n interndata python -m agent resume \
  --run-id 20260725_133748_168506 \
  --gpu 0 \
  --max-revisions 2
```



规划并执行：

```bash
conda run -n interndata python -m agent run \
  --prompt "使用左臂抓取面包片，将面包片完整、稳定地放入金属托盘内部；使用右臂抓取橙子，将橙子完整、稳定地放入同一个金属托盘内部。" \
  --gpu 0 \
  --max-revisions 2
```

使用本机 Conda 仿真后端：

```bash
conda run -n interndata python -m agent run \
  --prompt "把白色杯子放到托盘里" \
  --gpu 0 \
  --simulator-backend conda \
  --conda-env interndata-isaac6
```

`--conda-env` 指定的是完整仿真环境，不要求与启动 Agent 控制进程的环境相同。

恢复中断运行：

```bash
conda run -n interndata python -m agent resume --run-id <run_id> --gpu 0
```

重新诊断最近一次 Agent evidence：

```bash
conda run -n interndata python -m agent diagnose \
  --run-dir output/agent_runs/<run_id>
```

对冻结的 held-out `HeldOutVariantArtifact` 列表执行 qualification：

```bash
conda run -n interndata python -m agent qualify \
  --artifacts output/agent_runs/<run_id>/heldout_artifacts.json \
  --output-dir output/agent_runs/<run_id>/qualification
```

输入文件必须是 JSON 数组，每项只包含由 episode 固化的 `identity` 和
`artifact_manifest_path`。`identity` 包含
`run_id/variant_id/seed/profile_id/profile_hash/source_hash/scene_revision/world_revision`；
每个 attempt 的 `trace.jsonl` 事件另带与目录名一致的 `attempt_id`。
qualification 会重算 manifest 中每个原始成员的大小和 SHA-256，并绑定 manifest、
evaluation、evidence 的 identity、attempt 和 `variant_signature`；evaluation/evidence
路径必须属于 manifest 声明的同一个 attempt，不接受调用者另行指定。该命令只生成
`qualification_summary.json`，不会修改 `registry/robot_admissions.yaml`。

### 5.1 场景观察和 CuRobo 规划诊断

这两个入口属于确定性 `debug_tool`，不是 Pick/Place 一类机器人 Skill，也不需要先
打开 Codex。Codex、Claude Code 和其他 Agent 只需调用相同命令。

如果只想快速核对场景平面布局和候选坐标，不启动 Isaac：

```bash
conda run -n interndata python -m agent view \
  --task path/to/task.yaml \
  --output-dir output/debug/layout \
  --mode layout
```

输出包含 `layout.png`、`layout_manifest.json`、`stdout.log` 和
`view_summary.json`。物理材质、真实碰撞和机器人模型需要使用
下面默认的 `--mode physics`，或直接使用 `agent probe` 的运行时截图。

只加载任务并生成独立的机器人观察视角：

```bash
conda run -n interndata python -m agent view \
  --task path/to/task.yaml \
  --output-dir output/debug/view \
  --camera-height-m 1.2 \
  --gpu 2
```

默认由确定性 `robot_target_overhead_v1` 模板读取仿真中的机器人和中心物品实际位姿，
换算成独立世界相机；相机不会挂在机器人 Prim 下，也不会随着机械臂运动。模板参数位于
`config.yaml` 的 `debug.topdown_template_params`：`height_m` 控制相对高度，
`look_fraction` 控制画面中心在机器人与物品之间的位置。默认局部正俯视用于减少墙、
上柜和台面遮挡；需要斜视时可把模板改为 `robot_target_diagonal_v1`，再使用它的
`behind_m/side_m`。分辨率和焦距由 `topdown_resolution`、
`topdown_focal_length_mm` 控制；`topdown_eye/topdown_target` 保持 `null` 时使用模板。
命令行的 `--camera-height-m`、`--camera-look-fraction`、
`--camera-look-height-m` 可做单次相对构图调整；斜视模板还支持
`--camera-behind-m` 和 `--camera-side-m`。这些参数只影响本次调用，不会改写配置文件。
一次性绝对坐标覆盖参数：

```bash
conda run -n interndata python -m agent view \
  --task path/to/task.yaml \
  --output-dir output/debug/close_view \
  --eye 0.5 0.5 2.2 \
  --target 0.0 0.0 0.8 \
  --focal-length-mm 24 \
  --width 1280 --height 960 \
  --gpu 2
```

输出主图是 `debug_overview/rgb_0000.png`，同时保留 `stdout.log`、
`visualization_manifest.json` 和 `view_summary.json`。这是透视相机：将 `eye` 沿着“相机到
target”的方向靠近目标会看得更近，增大焦距也会缩小视野；提高分辨率只增加细节，
不会改变构图。`--view topdown` 是严格正交俯视图，正交投影的覆盖范围由房间尺寸决定，
单纯降低相机高度不会放大物体；需要近看时使用默认的 `debug_overview`。

对已有 Workspace manifest 运行一次 Pick planning-only CuRobo Probe：

```bash
conda run -n interndata python -m agent probe \
  --manifest output/agent_runs/<run_id>/variants/<variant_id>/subtasks/<subtask_id>/workspace/candidates.json \
  --output-dir output/debug/<case> \
  --candidate-id <candidate_id> \
  --gpu 2
```

如果手上只有单个 SimBox task，可先生成同一格式的 Workspace manifest：

```bash
conda run -n interndata python scripts/simbox/plan_support_mounted_workspace.py \
  --task path/to/task.yaml \
  --target <object_name> \
  --arm left \
  --output-dir output/debug/workspace
```

然后把 `output/debug/workspace/candidates.json` 传给 `agent probe`。在完整 Agent
流程中，这个 manifest 已由 Workspace 阶段生成，不需要重复执行脚本。

工具从 manifest 推导目标对象、required arm 和 attach collision path，复制 manifest 后
调用标准 Workspace validator；不会修改原 manifest。默认在 Probe 发布结果前输出：

```text
probe_summary.json
candidates.json
probes/<candidate_id>/seed_<seed>/
  probe_task.yaml
  stdout.log
  results/<candidate_id>.<arm>.json
  diagnostics/overview.png
  diagnostics/trajectory_debug.usda  # 至少存在一段成功规划路径时才生成
```

`trajectory_debug.usda` 中橙色点是 CuRobo 规划关节路径经正运动学得到的末端轨迹；
它是非物理可视化，不是第二条规划轨迹。Probe 默认关闭容易遮挡画面的绿色机器人碰撞球，
只有任务显式启用 `visualization.curobo_trajectory.show_robot_spheres` 时才同时写入。
若 terminal grasp 失败但 pregrasp 成功，工具仍可导出 pregrasp 段；两段都没有时结果会
明确写 `no_selected_path`，不会生成假的轨迹。碰撞隔离实验可追加
`--collision-world target-only|empty`、`--disable-collision-entity <name>` 或精确 Prim
path 参数；这些选项只用于 A/B 定位，标准运行仍使用完整 Physics/CuRobo 世界。

四个阶段的边界如下：

| 入口 | 是否启动 Isaac/PhysX/CuRobo | 是否移动机械臂 | 是否生成正式数据 |
|---|---:|---:|---:|
| `agent plan` | 否 | 否 | 否 |
| `agent probe`（默认 Pick gate） | 是 | 否，只规划并保持当前位姿 | 否 |
| validator 不带 `--planning-only` | 是 | 是，只做有界 Pick 验证 | 是，但不是完整任务 episode |
| `agent run/resume` 通过 gate 后 | 是 | 是，顺序执行全部 Subtask | 由 `generation.enabled` 决定，默认是 |

也可以在 Codex 对话框中说：

> 使用 InternDataEngine Agent，执行“把白色杯子放到托盘里”，GPU 0，最多修订两次。

对话入口负责调用上述命令，真正的场景选择、配置生成和失败闭环仍由本目录代码完成。

## 6. 运行产物

每次运行保存在 `output/agent_runs/<run_id>/`：

```text
request.json
selection.json
source_snapshot.json
scene_spec.json
semantic_task_plan.json
execution_variants.json
state.json
trace.jsonl
workspace_manifests.json
variants/<variant_id>/
  parent.json
  robot_profile.snapshot.yaml
  base_task.yaml
  subtasks/<subtask_id>/workspace/candidates.json
  subtasks/<subtask_id>/workspace/probes/<candidate_id>/seed_<seed>/
    results/<candidate_id>.<arm>.json
  subtasks/<subtask_id>/workspace/place_probes/<candidate_id>/seed_<seed>/
    results/<candidate_id>.<arm>.json
  workspace_selection/position_selection.json
  attempts/<attempt>/
    task.yaml
    semantic_task_plan.json
    command.json
    trace.jsonl
    stdout.log
    episode_events.jsonl
    evidence.json
    evaluation.json
    artifact_manifest.json
    diagnosis.json
    repair.json
    screenshots/
      topdown.png
    data/<episode_dir>/
      collision_world_audit.json
      object_state_events.jsonl
      safety_events.jsonl
      images.rgb.<camera>/demo.mp4
run_report.json
run_report.md
retention.json
```

`artifact_manifest.json` 不复制运行产物，而是按真实位置索引 compiled task、
workspace/static validation、spawn settle、collision audit、Pick/Place probe、
trace、日志、截图、视频、数据、evaluation 和 evidence。每项记录
`required/present/path/sha256`；必需产物缺失时写入明确的
`ARTIFACT_*_MISSING`，不会创建空 JSON 使检查误通过。多文件项同时记录成员路径与
成员哈希；manifest 本身也记录 episode identity 和 compiled task signature。

Collision audit、视频和 episode 数据的真实目录来自 `evidence.json` 的
`episode_dir`；Pick/Place probe 路径来自 workspace manifest 的
`planning_probe_artifacts`。启用 `debug.topdown_check` 时，episode 结束后的俯视截图
写入当前 `attempts/<attempt>/screenshots/topdown.png` 并由 manifest 索引。Pick/Place
前后等固定阶段截图尚未接入 runtime；截图保持为可选证据，不能把普通相机视频当成
阶段截图。

源任务、场景和资产保持只读。Agent 自动生成的机器人 Skill 只以 candidate 形式进入 `workflows/simbox/core/skills/generated/<category>/candidates/`，通过晋升门槛前不会进入 active registry。
