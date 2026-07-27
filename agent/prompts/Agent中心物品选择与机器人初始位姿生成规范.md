# Agent 中心物品选择与机器人初始位姿生成规范

## 1. 目的与边界

本规范用于让 Agent 根据任务名称、任务描述和场景资产，选出当前操作阶段的中心物品，并以该物品为参考计算机器人的初始工作位姿。

```text
任务描述与场景资产
  → 确定中心物品的精确实例名
  → 为每个 Subtask 确定使用左臂或右臂
  → 生成机器人候选位姿
  → 选择预期位姿
  → 写回任务配置
  → 保存选位日志
```

本阶段只负责中心物品、Subtask 手臂和机器人初始点位。Skill 名称和顺序由任务规划阶段决定；本阶段不选择抓取姿态、设置 Place 区域或修改碰撞、相机和数据记录配置。

## 2. 输入与信息来源

Agent 的输入包括：

- 任务名称或任务描述；
- 原始 `simbox_task.yaml`；
- 该任务引用的场景资产和 `simbox_arena.yaml`。

任务描述用于理解“要操作什么”；任务配置和场景资产用于确认“场景中具体是哪一个物体”。最终输入选位脚本的必须是 `objects` 中真实存在的精确实例名，例如：

```text
white_mug_a_0_id9000
```

不能只输入 `mug`、`cup`、`apple` 等语义概括名称，也不能只根据任务名称中的单词直接拼接物体名。

`delivery_active_objects`、已有 Skill 的 `objects`、物体的 `description` 和 `asset_category` 可以作为匹配线索，但 Agent 必须回到任务配置的 `objects` 列表确认最终实例。

## 3. 中心物品选择规则

中心物品是当前子任务中将被机器人直接操作的物体，而不是操作目的地、支撑面或附近最显眼的资产。

选中的物体必须：

- 出现在任务配置的 `objects` 中；
- 是可移动、可操作的刚体；
- 具有有效的物理刚体与碰撞配置；
- 具有选位脚本所需的位置区域信息；
- 不是桌子、柜台、墙、地板、支撑平面等固定资产。

例如，“把苹果放入托盘”的中心物品是苹果，不是托盘；托盘只描述任务目的地，不用于本阶段的初始点位中心选择。

如果同一语义对应多个物体实例，Agent 应结合任务描述中的空间关系、角色和任务配置中的 active object 信息确定具体实例，不得只取名称最相似或列表中的第一个对象。

## 4. 单中心物品流程

确定中心物品后，必须先根据任务分工确定该 Subtask 使用 `left` 还是 `right`。随后将原始任务配置、物体精确名称和已确定的手臂交给选位脚本：

```bash
cd /home/bld/ykqin/InternDataEngine

python scripts/simbox/plan_workspace_layout.py \
  --task <原始或待写回的 simbox_task.yaml> \
  --target <中心物品精确名称> \
  --arm <left|right> \
  --output-dir <配置所在目录>/pos_log/<中心物品精确名称>
```

脚本会从目标物体的 region 读取世界位置，从 `simbox_arena.yaml` 读取地板和环境几何，在目标周围生成机器人候选位姿，并把结果写入：

```text
<配置所在目录>/pos_log/<中心物品精确名称>/candidates.json
```

候选必须同时满足环境几何约束，并通过该 Subtask 已指定手臂的 CuRobo planning-only Probe；不得先对左右臂同时试探再回填手臂。选中的位姿必须包含：

```text
world_xy = [x, y]
yaw_deg = yaw
candidate_id = ...
```

随后使用 `core.workspace.planner.apply_candidate_to_document()` 将位姿写回对应任务配置。必须通过该统一入口同步更新机器人朝向、`robot_initial_region` 和运行时 robot region，不能只手改其中一个字段。

除机器人初始位姿和对应的选位来源记录外，任务配置中的其他内容必须保持不变。

## 5. 多中心物品与多子任务

如果任务包含多个子任务，并且不同子任务操作不同物体，应先按顺序列出所有 `[中心物品, 指定手臂]`，然后再为全部中心物品分别生成 Workspace 候选。现有脚本每次只处理一个目标物体；每个物体使用独立的 `pos_log/<object_name>/`，避免结果相互覆盖。

随后检查是否存在一个共同机器人位姿，可以同时高质量地服务多个中心物品。共同位姿的判断对象是同一个 `[x, y, yaw]`，不是相同的 `candidate_id`。同一位姿必须对每个中心物品分别满足：

- 机器人底盘不碰撞且位于有效地面内；
- 与目标物体的距离适合操作；
- 机器人朝向和后续操作空间合理；
- 每个中心物品都能由对应 Subtask 已指定的手臂生成 CuRobo/IK 联合路径。

如果存在这样的共同位姿，则将它作为唯一机器人初始位姿，将全部 Subtask 按顺序写入同一个 SimBox 任务 YAML，只加载一次场景并在同一 episode 内执行。

如果不存在共同高质量位姿，当前必须返回 `NO_COMMON_WORKSPACE_CANDIDATE`并停止，不生成一份无法完整执行的任务配置，也不使用开环底盘移动伪装 Nav。

当前记录形式：

```yaml
metadata:
  robot_position_plan:
    initial:
      targets: [<第一个或共同中心物品>]
      world_xy: [x, y]
      yaw_deg: yaw
      candidate_id: annulus_xxx
    subtasks:
      - subtask_id: <subtask_a>
        target: <中心物品 A>
        arm: left
      - subtask_id: <subtask_b>
        target: <中心物品 B>
        arm: right
```

未来接入 Nav 后，可在“无共同位姿”时改为先执行第一个 Subtask，再 Nav 到后续中心物品的已验证点位继续执行；在 Nav 真正接入前不写入或执行该分支。

## 6. 输出要求

完成后应得到：

```text
<任务目录>/
├── simbox_task.yaml
└── pos_log/
    ├── <中心物品 A>/
    │   └── candidates.json
    ├── <中心物品 B>/
    │   └── candidates.json
    └── position_selection.yaml
```

`simbox_task.yaml` 保存最终选中的机器人初始位姿及必要的多目标点位记录；`pos_log/` 保存所有候选、选择依据和备用点位，便于后续复查和重新选择。

Agent 完成该阶段时，只需报告：

- 识别出的子任务和中心物品精确名称；
- 每个 Subtask 指定的左/右臂；
- 是否找到通过全部指定手臂 Probe 的共同点位；
- 写回的初始 `[x, y, yaw]`；
- `pos_log/` 保存位置。

## 7. 相关实现

- 候选生成入口：[`scripts/simbox/plan_workspace_layout.py`](../../scripts/simbox/plan_workspace_layout.py)
- 目标解析与候选计算：[`workflows/simbox/core/workspace/planner.py`](../../workflows/simbox/core/workspace/planner.py)
- 位姿写回入口：[`apply_candidate_to_document()`](../../workflows/simbox/core/workspace/planner.py#L303)
