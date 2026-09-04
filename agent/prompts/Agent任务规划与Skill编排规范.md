# Agent 任务规划与 Skill 编排规范

## 1. 目的与边界

本规范只约束“任务规划”阶段。Agent 根据用户需求和选中任务的真实 manifest：

```text
任务需求与场景 manifest
  → 划分 object subtask
  → 为每个 subtask 选择执行模式
  → 选择已有 runtime Skill、执行顺序和机械臂
  → 填写允许由 Agent 决定的 Skill 参数
  → 返回结构化 TaskPlan
```

Agent 不直接编辑 SimBox YAML，不生成 `pick_plan_probe`，也不展开接近、闭合、attach、detach、
碰撞世界更新、安全重规划等 Skill 内部动作。TaskPlan 通过校验后，由确定性编译器写入源任务配置的副本。

Physics Schema 不再按 `pick` / `place` 做 Skill 白名单。选中源任务已经存在的 runtime Skill 可以写入
TaskPlan；注册表中的 Skill 继续提供参数和对象数量的确定性校验，未注册 Skill 则保留源配置并由
SimBox runtime 负责最终 Skill 实例化校验。

## 2. Subtask 划分

一个 object subtask 对应一个当前被直接操作的中心物品：

- `manipulated_object` 必须是 manifest 中的精确实例名；
- `target_object` 必须是 manifest 中的精确目标实例名；Pick-only 时可以为空；
- 每个可执行 subtask 必须在 Workspace 候选生成前明确填写 `arm: left` 或 `arm: right`；
- 用户、任务分工或已知资产约束已给出手臂依据时使用该依据；单中心物品且没有可靠依据时使用 `config.yaml` 的 `robot.default_arm`，不得伪造几何理由；
- 两个互不依赖的物体应拆成两个 subtask，不能为了使用双臂而强行合并；
- `relation` 只能是 `on`、`inside`、`left_of`、`right_of`、`next_to`、`hang`、`insert` 或 `none`；
- `inside` 的目标必须拥有 manifest 中声明的可接收容器区域；
- `hang` 和 `insert` 必须有明确的资产 affordance 和方向轴依据，否则写入 `unresolved`，不能猜测。

## 3. 每个 Stage 的四种执行模式

Agent 必须为每个 Stage 明确填写 `execution_mode`，并使 `skills` 的数量、顺序和 `arm` 与该模式一致。

### 3.1 `single_arm_single_skill`

只使用一条 Skill。例如 Pick-only：

```json
{
  "arm": "left",
  "stages": [{
    "execution_mode": "single_arm_single_skill",
    "skills": [{"name": "pick", "arm": "auto", "objects": ["cup"], "params": {}}]
  }]
}
```

规则：恰好一条 Skill；只能操作 subtask 已指定的手臂。Skill 可写 `arm: auto` 继承 subtask.arm，但 subtask.arm 本身不能是 `auto`。

### 3.2 `single_arm_sequential`

同一条手臂先后执行两条 Skill。当前最常见写法是同臂 `pick → place`：

```json
{
  "arm": "left",
  "stages": [{
    "execution_mode": "single_arm_sequential",
    "skills": [
      {"name": "pick", "arm": "auto", "objects": ["cup"], "params": {}},
      {"name": "place", "arm": "auto", "objects": ["cup", "tray"], "params": {}}
    ]
  }]
}
```

规则：当前恰好两条 Skill；两条 Skill 必须解析到同一手臂；Place 必须跟在同臂 Pick 后面。

### 3.3 `dual_arm_sequential`（预留，当前不进入执行）

左右臂先后执行，前一条手臂完成自己的全部 Skill 后，另一条手臂才开始：

```json
{
  "arm": "both",
  "stages": [{
    "execution_mode": "dual_arm_sequential",
    "skills": [
      "<left arm Skill 1>",
      "<left arm Skill 2, optional>",
      "<right arm Skill 1>",
      "<right arm Skill 2, optional>"
    ]
  }]
}
```

该模式保留了 TaskPlan 和 YAML 表达位置，但当前 Workspace 流程要求每个
可执行 Subtask 在生成候选前只选定一条手臂。如果任务必须在同一 Subtask 中使用两臂，
Agent 应如实表达需求并写入 `unresolved`，不得进入 Workspace 生成。

若任务涉及两个独立中心物品，必须拆成两个 subtask，分别指定 `left` 或 `right`。Orchestrator 会在所有手臂确认后，再为所有中心物品生成候选，优先找一个能同时服务全部中心物品和指定手臂的共同底座位姿，再将全部 subtask 顺序编译到同一个 SimBox YAML 并执行一次。无共同位姿时必须停止，不得生成 Nav 或拆成多次仿真。

### 3.4 `dual_arm_simultaneous`

左右臂同时开始，各自内部仍按列表顺序执行。SimBox YAML 能表达该模式，但当前 Agent 使用的
`physics_schema` 尚未开放双臂并发 manipulation，并已确认在当前阶段继续关闭。Agent 如果判断任务必须使用该模式，应：

1. 使用 `execution_mode: dual_arm_simultaneous` 表达真实需求；
2. 显式填写左右臂；
3. 在 `unresolved` 中记录“需要双臂并发 Physics Schema 能力”；
4. 不把它伪装为单臂或双臂先后执行。

确定性编译器当前会阻止这种计划进入运行。该位置是未来完成双臂碰撞世界和执行验证后的能力开关。

## 4. Skill 顺序、对象和语义关系

- `pick.objects` 必须严格为 `[manipulated_object]`；
- `place.objects` 必须严格为 `[manipulated_object, target_object]`，顺序不能颠倒；
- Place 必须跟在同一条手臂的 Pick 后；
- `inside` 是 subtask 的语义关系，不是 Place 的 `success_mode`；
- `inside`/`on` 通常使用 `success_mode: xybbox` 或 `3diou`；
- `left_of` 必须使用 `success_mode: left`；
- `right_of` 必须使用 `success_mode: right`；
- `next_to` 可根据明确方向使用 `left` 或 `right`；
- `place_direction: horizontal` 时必须同时提供 `align_place_obj_axis` 和 `offset_place_obj_axis`；
- 参数没有任务、资产几何或既有配置依据时应省略，让确定性编译器使用受控默认值，不能为了“填满字段”而猜测。

## 5. TaskPlan 返回前自检

Agent 返回 JSON 前必须逐项确认：

1. 所有对象、机器人和资产引用均来自选中 manifest；
2. 每个 Stage 的 `execution_mode` 与 Skill 数量、手臂和顺序一致；
3. 每个可执行 subtask 的 `arm` 已明确为 `left` 或 `right`，其单臂 Skill 与之一致；任何必须双臂的 Subtask 都已写入 `unresolved`；
4. 只使用 manifest/source task 中真实存在或 runtime 已注册的 Skill，不编造 Skill 名称；
5. 已登记 Skill 的 `params` 字段都出现在下方对应参数表中；未登记但来自 source task 的 Skill 使用源配置或明确的 runtime 参数；
6. 参数值满足类型、枚举、长度、单位和依赖条件；
7. `owner` 不是 `agent` 的参数不写入返回；
8. 不使用 `ignore_substring`，不通过隐藏障碍物换取规划成功；
9. 无法从证据确定的决策写入 `unresolved`，不猜测。

## 6. Pick 可用参数

`objects`、`name` 和 `arm` 是 SkillStep 的固定字段，不放入 `params`。

| 参数 | 合法类型或取值 | 含义 |
|---|---|---|
| `filter_x_dir` | `[direction, angle]` 或三元素；方向为 `forward/backward/upward/downward`，角度 0–180° | 按 EE X 轴方向过滤抓取候选 |
| `filter_y_dir` | 同上 | 按 EE Y 轴方向过滤抓取候选 |
| `filter_z_dir` | 同上 | 按 EE Z 轴方向过滤抓取候选 |
| `fixed_orientation` | 4 个数字 `[w,x,y,z]` | 固定末端四元数；仅有明确依据时使用 |
| `npy_name` | 不含路径的 `.npy` 文件名 | 使用资产中已存在的抓取标注；不得编造 |
| `grasp_scale` | 0.01–10 的数字 | 抓取标注位移缩放系数 |
| `tcp_offset` | 0–1 m | 覆盖机器人 TCP 偏移 |
| `direction_to_obj` | `left` 或 `right` | 按候选相对物体的位置进一步过滤 |
| `pre_grasp_offset` | 0–1 m | 预抓取点到抓取点的接近距离 |
| `gripper_change_steps` | 1–1000 的整数 | 夹爪闭合保持步数 |
| `post_grasp_offset_min` | 0–1 m | 抓取后抬升距离下界 |
| `post_grasp_offset_max` | 0–1 m，且不小于下界 | 抓取后抬升距离上界 |
| `return_to_pregrasp` | boolean | 抬升后是否返回预抓取位姿 |
| `lift_th` | 0–1 m | Pick 成功所需的最小抬升高度 |
| `grasp_contact_threshold_n` | 非负数字 | 夹爪接触确认力阈值，单位 N |
| `t_eps` | 0–1 m 的正数 | 平移完成容差 |
| `o_eps` | 0–3.2 rad 的正数 | 旋转完成容差 |
| `test_mode` | 仅 `forward`，owner=`compiler` | Physics Schema 下由编译器写入；Agent 不得设置 |

以下 Pick 内部字段不向 Agent 开放：`ignore_substring`、`constraints`、`final_gripper_state`、
`pre_grasp_hold_vec_weight`、`process_valid` 和 `debug`。其中部分会改变安全检查或依赖无法验证的内部结构。

## 7. Place 可用参数

| 参数 | 合法类型或取值 | 含义 |
|---|---|---|
| `place_direction` | `vertical` / `horizontal` | 从上方放置或水平插入/悬挂 |
| `filter_x_dir` | 二或三元素方向过滤；方向还可为 `leftward/rightward`，角度 0–180° | 按 EE X 轴过滤姿态 |
| `filter_y_dir` | 同上 | 按 EE Y 轴过滤姿态 |
| `filter_z_dir` | 同上 | 按 EE Z 轴过滤姿态 |
| `x_ratio_range` | 两个递增数字 `[min,max]` | 目标包围盒 X 轴采样比例 |
| `y_ratio_range` | 两个递增数字 `[min,max]` | 目标包围盒 Y 轴采样比例 |
| `z_ratio_range` | 两个递增数字 `[min,max]` | 水平放置时 Z 轴采样比例 |
| `pre_place_z_offset` | -2–2 m | 垂直预放置点相对目标顶面高度 |
| `place_z_offset` | -2–2 m | 垂直最终放置点相对目标顶面高度 |
| `pre_place_align` | -2–2 m | 水平预放置点沿对齐轴偏移 |
| `place_align` | -2–2 m | 水平最终点沿对齐轴偏移 |
| `pre_place_offset` | -2–2 m | 水平预放置点沿辅助轴偏移 |
| `place_offset` | -2–2 m | 水平最终点沿辅助轴偏移 |
| `post_place_vector` | 3 个数字 `[x,y,z]` | 释放后的末端撤退向量，单位 m |
| `position_constraint` | `object` / `gripper` | 目标位置约束到被拿物体或夹爪；不能填 region 名称 |
| `success_mode` | `3diou/height/xybbox/left/right/flower/cup` | Place 成功判定方式；必须与 relation 一致，`inside` 不是合法值 |
| `success_th` | 0–1 | `3diou/flower/cup` 使用的阈值 |
| `threshold` | 非负数字，单位 m | `left/right` 使用的水平距离阈值 |
| `place_part_prim_path` | owner=`asset` | 资产内部已有 Prim 子路径；Agent 不得设置或编造 |
| `align_pick_obj_axis` | 3 个数字 | 被拿物体需要对齐的局部轴 |
| `align_place_obj_axis` | 3 个数字 | 目标容器对齐轴；horizontal 必填 |
| `offset_place_obj_axis` | 3 个数字 | 目标容器辅助偏移轴；horizontal 必填 |
| `align_obj_tol` | 0–180° | 物体轴对齐角度容差 |
| `align_plane_x_axis` | 3 个数字 | EE X 轴平面约束法向量 |
| `align_plane_y_axis` | 3 个数字 | EE Y 轴平面约束法向量 |
| `gripper_change_steps` | 1–1000 的整数 | 夹爪打开保持步数 |
| `hesitate_steps` | 0–1000 的整数 | 释放前停留步数 |
| `preserve_attached_orientation` | boolean | 是否保持当前被拿物体的附着朝向 |
| `t_eps` | 0–1 m 的正数 | 平移完成容差 |
| `o_eps` | 0–3.2 rad 的正数 | 旋转完成容差 |
| `test_mode` | 仅 `forward`，owner=`compiler` | Physics Schema 下由编译器写入；Agent 不得设置 |

以下 Place 内部字段不向 Agent 开放：`ignore_substring`、`pre_place_hold_vec_weight`、
`post_place_hold_vec_weight` 和内部调试字段。

## 8. 配置写入责任

Agent 的最终输出是符合 `TaskPlan` schema 的 JSON。确定性编译器负责：

- 把 execution mode 转成 SimBox `skills → robot → phase → arm → Skill list` 嵌套；
- 把 Skill 的 `arm: auto` 替换为 Subtask 在 Workspace 候选生成前已选定的机械臂；
- 把 `relation` 转成受控 Place 默认参数；
- 写入 `test_mode: forward`、Physics Schema 和执行安全配置；
- 校验参数、对象数量、Skill 顺序和并发能力；
- 只写到 Agent run 目录中的配置副本，不修改源任务文件。

官方格式参考：

- <https://internrobotics.github.io/InternDataEngine-Docs/custom/task.html>
- <https://internrobotics.github.io/InternDataEngine-Docs/concepts/skills/pick.html>
- <https://internrobotics.github.io/InternDataEngine-Docs/concepts/skills/place.html>
