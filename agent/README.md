# InternDataEngine 通用任务 Agent

新安装仓库需要迁移或合并 Agent、Workspace、SimBox/CuRobo 底层和 Docker 并行生成能力时，先阅读 `[MIGRATION.md](./MIGRATION.md)`。

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
→ inventory硬约束过滤和语义选择
→ 选择现有SceneCapabilityManifest
→ 拆分Subtask并先为每个Subtask确定左/右臂
→ 为所有中心物品分别生成workspace annulus候选
→ 查找一个能服务全部中心物品的共同底座位姿
→ 使用每个Subtask已指定手臂做CuRobo planning-only验证
→ 全部Subtask编译进一个physics-schema SimBox YAML
→ 单GPU、单episode顺序执行
→ 读取结构化诊断证据
→ 单一原因修订（最多2次）
→ 经验留存
```

没有现有场景覆盖任务时，Agent 输出 `SceneCompositionRequest`，记录建议基础场景、所需资产、空间关系、机器人能力和验证要求。v1 不自动装配和运行新场景。多个中心物品没有共同底座位姿时，当前返回 `NO_COMMON_WORKSPACE_CANDIDATE` 或 `NO_COMMON_CUROBO_WORKSPACE_CANDIDATE` 并停止。

跨 workspace 自动导航不属于当前能力。可靠 Nav 接入后，无共同位姿的分支才改为“执行前一 Subtask → Nav 到下一已验证点位 → 继续执行”；接入前不使用开环底盘动作伪装成 Nav。

### 3.1 已确认的能力边界与待实现项

- `dual_arm_simultaneous` 继续关闭，但保留 TaskPlan 表达和显式报错；未来完成双臂并发碰撞世界、执行和失败恢复验证后再开放。
- 每个可执行 Subtask 必须先确定 `arm: left|right`，之后才为所有中心物品生成候选。多个独立中心物品共用一个已验证底座位姿、生成一个 SimBox YAML，并在同一 episode 内按 subtask/stage/phase 顺序执行；例如左臂 `pick cup -> place tray`，再右臂 `pick spoon -> place tray`。
- 数据生成默认开启，由 `config.yaml` 的 `generation.enabled` 控制。Agent 仅在用户明确要求生成或不生成时写布尔覆盖，否则返回 `null`，由 Orchestrator 使用默认值。
- 任务成功判定与失败反馈需要独立设计规范，覆盖 Skill、phase、subtask 和 task 四层判定、证据优先级、失败码、是否可重试和对应修订动作。当前已有部分 evidence/failure-code 基础设施，但统一规范和多 subtask 聚合成功语义仍为待定项。

## 4. 目录职责

- `contracts.py`：稳定 Pydantic 数据契约；
- `inventory.py`：扫描场景、任务、资产和已有 Skill；
- `resolver.py`：候选过滤、Codex 结构化选择和计划；
- `compiler.py`：TaskPlan、workspace candidate 到 SimBox YAML；
- `orchestrator.py`：共同位姿选择、单 YAML 执行、有界修订和恢复状态机；
- `evidence.py`：归一化 SimBox 诊断产物并确定主失败；
- `retention.py`：经验分流、候选记录和晋升门槛；
- `task_ir/`：对原生任务 YAML 的紧凑、无损表示，供知识聚合脚本复用；
- `registry/`：Skill 契约、语义别名和失败码；
- `experience/`：通用 playbook、调试工具候选和经验索引。

规划阶段实际加载三类约束：`prompts/plan.md` 提供结构化输出要求；
`prompts/Agent任务规划与Skill编排规范.md` 提供 subtask、手臂、执行模式以及 Pick/Place 参数含义；
`prompts/Agent中心物品选择与机器人初始位姿生成规范.md` 提供中心物品识别、先选手臂、再生成候选以及共同位姿规则。
参数的机器可读类型、枚举、范围和所有权以 `registry/skill_contracts.yaml` 为准，
`resolver.py` 必须在配置编译前强制校验，不能只依赖 LLM 遵守文字说明。

### 4.1 配置分工和覆盖顺序

为保持简单，当前不把默认值拆成许多小文件，而是采用四个职责明确的入口：

- `config.yaml`：Agent 全局运行策略和确定性默认值，包括 backend、路径、执行预算、数据生成、机器人 profile/默认手臂、共同位姿阈值、Physics Schema 和 Pick/Place 默认参数；
- `registry/skill_contracts.yaml`：Agent 能写哪些 Skill 参数，以及每个参数的类型、范围和所有权；
- `SceneCapabilityManifest` 与源任务 YAML：场景中真实存在的机器人、对象、区域、资产能力和基础场景配置；
- `TaskPlan`：本次请求的语义选择，只保存对象、Subtask、手臂、Skill 顺序和用户明确给出的合法覆盖值。

最终 YAML 一律由 `compiler.py` 生成。普通 Skill 参数采用“全局默认 → relation 默认 → Agent 明确覆盖”的顺序；`test_mode`、碰撞世界和资产 Prim path 等 compiler/asset-owned 字段不能被 Agent 覆盖。这样 Agent 不直接编辑源 YAML，也不会把默认参数重复写进每个 TaskPlan。

## 5. 使用方法

在仓库根目录和 `interndata` 环境中运行。

首次使用或场景/资产变化后建立 inventory：

```bash
conda run -n interndata python -m agent index
```

两步走战略:

```bash
conda run -n interndata python -m agent plan \
  --prompt " 抓取面包片，将面包片完整、稳定地放入金属托盘内部；抓取橙子，将橙子完整、稳定地放入同一个金属托盘内部。" 2>/dev/null
用franka把苹果放进金属托盘

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
conda plan -n interndata python -m agent run \
  --prompt "使用左臂抓取面包片，将面包片完整、稳定地放入金属托盘内部；使用右臂抓取橙子，将橙子完整、稳定地放入同一个金属托盘内部。" \
  --gpu 0 \
  --max-revisions 2
```

恢复中断运行：

```bash
conda run -n interndata python -m agent resume --run-id <run_id> --gpu 0
```

重新诊断最近一次 Agent evidence：

```bash
conda run -n interndata python -m agent diagnose \
  --run-dir output/agent_runs/<run_id>
```

也可以在 Codex 对话框中说：

> 使用 InternDataEngine Agent，执行“把白色杯子放到托盘里”，GPU 0，最多修订两次。

对话入口负责调用上述命令，真正的场景选择、配置生成和失败闭环仍由本目录代码完成。

## 6. 运行产物

每次运行保存在 `output/agent_runs/<run_id>/`：

```text
request.json
selection.json
task_plan.json
state.json
workspace_manifests.json
subtasks/<subtask_id>/workspace/
workspace_selection/position_selection.json
attempts/<attempt>/
  task.yaml
  task_plan.json
  command.json
  stdout.log
  episode_events.jsonl
  evidence.json
  diagnosis.json
run_report.json
run_report.md
retention.json
```

源任务、场景和资产保持只读。Agent 自动生成的机器人 Skill 只以 candidate 形式进入 `workflows/simbox/core/skills/generated/<category>/candidates/`，通过晋升门槛前不会进入 active registry。