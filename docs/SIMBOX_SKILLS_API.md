# SimBox Skill API 说明

> 代码基线：`mobile-base` 分支，提交 `fb3d914`，核对日期 2026-07-14。
>
> 本文以 `workflows/simbox/core/skills/`、`workflows/simbox_dual_workflow.py` 和
> `nav2/runtime/` 的当前实现为准。表中的“必填”表示代码直接用 `cfg[...]` 读取，
> 缺失会抛出异常；“条件必填”表示只在启用某个模式时必填；“否”表示代码提供默认值。

## 1. Skill 是什么

Skill 是 SimBox 中可配置的最小执行节点。Task YAML 把 Skill 放在以下层级中：

```text
skills[] -> robot name -> controller name -> skill item
```

运行时，`SimBoxDualWorkflow` 根据 `skill item.name` 从注册表取出 Python 类，并以
`(robot, controller, task, skill_cfg, world=..., workflow=..., draw=...)` 构造实例。
Skill 通常先由 `simple_generate_manip_cmds()` 生成命令，再循环执行 `update()`、
`is_done()`、`is_success()` 和 `is_feasible()`。

注册名由类名自动转换：大写字母前插入 `_`，再转小写。类名本身含下划线时会产生双下划线，
例如 `Heuristic_Skill -> heuristic__skill`、`Goto_Pose -> goto__pose`。

## 2. 如何调用

### 2.1 Legacy 顺序模式

```yaml
skills:
  - panda_omron:
      - left:
          - name: pick
            objects: [apple]
          - name: place
            objects: [apple, tray]
      - right:
          - name: wait
            objects: [tray]
            success_threshold: 0.005
```

- `panda_omron` 必须匹配 `robots[].name`。
- `left`、`right`、`base` 必须匹配该机器人的 controller 名。
- 同一 controller 列表内按顺序执行。
- 同一个 sequence 中不同 controller 的首个 Skill 可并行推进。
- 外层多个 phase 按顺序执行。

### 2.2 DAG 依赖模式

只要任意节点出现 `id` 或 `depends_on`，整份 Task 都进入 DAG 模式：

```yaml
skills:
  - panda_omron:
      - base:
          - id: nav_to_pick
            name: navigate
            depends_on: []
            goal: nav_to_pick
      - left:
          - id: pick_apple
            name: pick
            depends_on: [nav_to_pick]
            objects: [apple]
          - id: home_left
            name: heuristic__skill
            depends_on: [pick_apple]
            mode: home
```

| 字段 | 类型 | 必填 | 约束 |
| --- | --- | --- | --- |
| `name` | `str` | 是 | 必须是第 4 节中的注册名，否则查表时 `KeyError`。 |
| `id` | `str` | DAG 模式是 | 全 Task 唯一；缺失、重复均报错。 |
| `depends_on` | `list[str]` | 否 | 默认 `[]`；必须是列表；引用必须存在；依赖图不能有环。 |

一个节点只有在全部依赖成功后才启动。节点结束但 `is_success()` 为 false 时，episode 失败，
其下游不会运行。DAG 模式允许互不依赖的节点同时处于 running 状态。

### 2.3 对象引用

- `objects` 中的名称通常引用 `task._task_objects`；该集合包含 Task objects、Arena fixtures、robots 和 cameras。
- 部分旧 Skill 使用 `task.objects` 而不是 `_task_objects`，这类 Skill 只能引用该字典中实际存在的对象。
- 数组顺序有语义。例如 `place.objects = [被放对象, 目标容器]`。

## 3. 类型、单位与范围约定

代码没有统一 schema，也不会拒绝所有未知字段。本文给出的范围分为三类：

1. **硬约束**：代码显式检查，违反即异常。
2. **结构约束**：代码按固定长度或索引读取，长度不符会异常或计算错误。
3. **语义范围**：代码未检查，但超出范围会使几何、采样或成功判定失去意义。

| 类型 | 约定 |
| --- | --- |
| 位置、距离、offset、threshold | meter，通常应为有限 `float`；距离/容差通常应 `>= 0`。 |
| 欧拉角、`angle_deg`、方向过滤角 | degree。 |
| quaternion | `[w, x, y, z]`，应为有限且非零的四元数。 |
| yaw | `navigate` 使用 rad；命名 `positions[].yaw` 也是 rad。 |
| 轴向量 | `[x, y, z]`，应长度 3 且非零；多数实现会归一化但不检查零向量。 |
| 比例范围 | `[min, max]`；bbox 内采样通常建议 `0 <= min <= max <= 1`。代码通常只要求可解包。 |
| 采样范围 | `[min, max]`；必须 `min <= max`，否则 NumPy 采样可能报错。 |
| step/count | `int`；循环或插值步数通常应 `> 0`，等待步数可为 `0`。 |
| `t_eps` | EE 平移或关节范数阈值，通常 `> 0`。 |
| `o_eps` | 四元数角距离阈值，rad，通常在 `[0, pi]`。 |
| `ignore_substring` | `list[str]`，追加到 controller 的碰撞忽略名单。 |
| `hold_vec_weight` | 通常为 6 元数组，对应位姿代价权重；也可为 `null`。 |

### 3.1 通用方向过滤

`filter_x_dir`、`filter_y_dir`、`filter_z_dir` 有两种格式：

```yaml
filter_z_dir: [downward, 140]
filter_z_dir: [downward, 100, 160]
```

- `[direction, angle]`：单侧角度阈值。
- `[direction, value1, value2]`：直接按两个余弦边界筛选，并不是统一的 `min_angle/max_angle` API。正号方向执行 `element >= cos(value1) and element <= cos(value2)`，在 `[0,180]` 内通常需要 `value1 >= value2` 才非空；负号方向执行 `element <= cos(value1) and element >= cos(value2)`，通常需要 `value1 <= value2`。新增配置优先使用二元格式；三元格式必须按矩阵谓词验证。
- `pick`、`manualpick`、`dynamicpick`、`fail_pick` 支持 `forward`、`backward`、`upward`、`downward`。
- `place`、`goto__pose` 额外支持 `leftward`、`rightward`。
- 写入不支持的 direction 会触发 `KeyError`。

建议起始值：

| 使用位置 | `filter_x_dir` | `filter_y_dir` | `filter_z_dir` |
| --- | --- | --- | --- |
| `pick` | `[forward, 90]` | 省略 | `[downward, 140]` |
| `manualpick` / `dynamicpick` / `fail_pick` | 可先参考 `[forward, 90]` | 省略 | 可先参考 `[downward, 140]` |
| `place` | `[forward, 45]` | 省略 | `[downward, 150]` |
| `goto__pose` 姿态采样 | 可先参考 `[forward, 45]` | 省略 | 可先参考 `[downward, 150]` |

这些值适合作为 Panda 风格顶部抓取和顶部放置的起点，不是所有机器人、物体和 EE 轴定义下的通用最优值。

### 3.2 常用枚举

| 字段 | 有效值 | 说明 |
| --- | --- | --- |
| `test_mode` | `forward`, `ik` | 分别用 controller 规划或 IK 检查候选；其他值没有有效分支。 |
| `gripper_state` | `1` / `1.0`, `-1` / `-1.0` | 约定 open / close；个别 Skill 把所有非 1 值都当 close。 |
| `direction_to_obj` | `left`, `right` | EE y 在对象 y 的正侧 / 非正侧。 |
| `position_constraint` | `gripper`, `object` | 约束夹爪目标位置，或先约束物体位置再反推夹爪。 |

## 4. 当前注册的 25 个 Skill

| 注册名 | 实现类 | 主要用途 | 当前 Task YAML 中出现 |
| --- | --- | --- | --- |
| `pick` | `Pick` | 标准静态抓取 | 是 |
| `manualpick` | `Manualpick` | 手工修正抓取姿态 | 是 |
| `dexpick` | `Dexpick` | 离散预定义姿态抓取 | 是 |
| `dynamicpick` | `Dynamicpick` | 运动目标预测抓取 | 是 |
| `fail_pick` | `FailPick` | 故意偏移的失败样本 | 是 |
| `place` | `Place` | 通用放置 | 是 |
| `dexplace` | `Dexplace` | 几何范围放置 | 是 |
| `move` | `Move` | 推动对象靠近目标 | 是 |
| `goto__pose` | `Goto_Pose` | EE 到指定位姿 | 是 |
| `track` | `Track` | 随机路点追踪 | 是 |
| `scan` | `Scan` | 扫描/观测动作 | 是 |
| `wait` | `Wait` | 保持 EE 位姿等待 | 是 |
| `gripper__action` | `Gripper_Action` | 单独控制夹爪 | 是 |
| `heuristic__skill` | `Heuristic_Skill` | home/关节/相对 EE 运动 | 是 |
| `home` | `Home` | 旧式固定 home 动作 | 否 |
| `joint__ctrl` | `Joint_Ctrl` | 关节角直接控制 | 是 |
| `navigate` | `Navigate` | Nav2 固定/动态目标导航 | 是 |
| `open` | `Open` | 打开 articulated object | 是 |
| `close` | `Close` | 关闭 articulated object | 是 |
| `artpreplan` | `Artpreplan` | articulation 预规划 | 是 |
| `rotate` | `Rotate` | articulation 旋转 | 否 |
| `rotate__obj` | `Rotate_Obj` | 旋转已抓物体 | 是 |
| `approach__rotate` | `Approach_Rotate` | 接近对象并可选旋转 | 是 |
| `flip` | `Flip` | 翻转对象 | 是 |
| `pour__water__succ` | `Pour_Water_Succ` | 倒水粒子成功判定 | 是 |

## 5. 抓取类 Skill

### 5.1 基于抓取标注的标准静态抓取：`pick`

`objects[0]` 是被抓对象。对象必须有可加载的抓取标注；默认从所选 USD 同目录读取
`Aligned_grasp_sparse.npy`。

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | 长度至少 1；代码读取 `[0]`。 |
| `npy_name` | `str` | 否 | `Aligned_grasp_sparse.npy` | USD 同目录下的抓取标注文件名。 |
| `grasp_scale` | `float` | 否 | `1` | 正数；传入抓取姿态后处理。 |
| `tcp_offset` | `float` | 否 | `robot.tcp_offset` | meter；TCP 补偿。 |
| `constraints` | `[axis,min_ratio,max_ratio]` / `null` | 否 | `null` | axis 仅 `x/y/z`；按 TCP 位置在该轴的整体 min/max 做比例截取，建议 `0 <= min_ratio <= max_ratio <= 1`。 |
| `final_gripper_state` | `1` 或 `-1` | 否 | `-1` | `1` 生成 open，`-1` 生成 close；其他值无有效分支。 |
| `fixed_orientation` | `[w,x,y,z]` | 否 | `null` | 只覆盖最终选中候选的 pre/grasp 姿态，且发生在可达性检查之后，不会重新规划验证；代码不归一化。 |
| `pre_grasp_offset` | `float` | 否 | `0.1` | 建议 `>= 0`；沿末端接近轴退让。 |
| `grasp_guard_offset` | `float` | 否 | `min(pre_grasp_offset, 0.04)` | 建议 `[0, pre_grasp_offset]`；抓取前保护路点。 |
| `post_grasp_offset_min/max` | `float` | 否 | `0.05/0.05` | meter，要求 min `<=` max；只控制抓取后的 z 向抬升。 |
| `return_to_pregrasp` | `bool` | 否 | `false` | 抬升后是否回到 pre-grasp 位姿。 |
| `gripper_change_steps` | `int` | 否 | `40` | 代码强制至少 1；渐变闭合/打开步数。 |
| `grasp_open_hold_steps` | `int` | 否 | `8` | `>= 0`；到达抓取点后保持张开。 |
| `post_close_hold_steps` | `int` | 否 | `12` | `>= 0`；闭合后保持。 |
| `grasp_t_eps` | `float` | 否 | `0.008` | 实际抓取点平移阈值取 `min(t_eps, grasp_t_eps)`。 |
| `grasp_o_eps` | `float` | 否 | `0.2` | 实际抓取点姿态阈值取 `min(o_eps, grasp_o_eps)`。 |
| `pre_grasp_hold_vec_weight` | 6 元数组 / `null` | 否 | `null` | pre-grasp 到达后更新 pose cost。 |
| `ignore_substring` | `list[str]` | 否 | `[]` | 抓取规划附加碰撞忽略项。 |
| `filter_*_dir` | 2/3 元数组 | 否 | 无 | 建议先用 `x: [forward,90]`、`z: [downward,140]`，省略 y；见 3.1。 |
| `direction_to_obj` | `left` / `right` | 否 | 无 | 硬筛选候选所在侧。 |
| `test_mode` | `forward` / `ik` | 否 | `forward` | 候选可达性检查。 |
| `target_grasp_z` | `float` | 否 | `0.12` | 仅 batch 最终排序使用的对象相对抓取高度，meter。 |
| `target_grasp_orientation` | quaternion | 否 | `null` | 仅 batch 排序使用，必须非零；apple 名称有代码内置姿态。 |
| `target_grasp_orientation_weight` | `float` | 否 | `1.0` | 仅 batch 排序使用，建议 `>= 0`。 |
| `grasp_side_preference` | `toward_arm` / `away_from_arm` | 否 | 无 | 所有路径都做硬筛选；batch 路径还参与排序。对象 XY 不能与 arm base 重合。 |
| `grasp_side_weight` | `float` | 否 | `2.0` | 仅 batch 侧面排序权重，建议 `>= 0`。 |
| `t_eps` / `o_eps` | `float` | 否 | `1e-3` / `5e-3` | 子命令完成阈值。 |
| `process_valid` | `bool` | 否 | `true` | 要求机器人关节速度和对象线速度绝对值最大值 `< 5`。 |
| `lift_th` | `float` | 否 | `0.0` | `> 0` 时要求对象 z 抬升严格大于该值；不改变规划抬升高度。 |
| `output_root` | `str` | 否 | `output/ros_bridge/skills` | 调试快照根目录。 |
| `execution_trace_write_stride` | `int` | 否 | `250` | `> 0` 时按间隔写盘，`<= 0` 禁用周期写盘。 |
| `execution_trace_max_steps` | `int` | 否 | `500` | 代码至少保留 1 步，`<= 1` 都按 1 处理。 |
| `stalled_command_step_limit` | `int` | 否 | `450` | `> 0` 启用停滞判定，`<= 0` 禁用。 |

调试产物包括 `pick_plan_snapshot.json`、`pick_execution_trace.json`、
`pick_success_check_snapshot.json` 和失败时的 `pick_runtime_failure_snapshot.json`。
`pick` 同时从 `task.objects[objects[0]]` 和 `task.cfg.objects` 查对象路径，因此不能直接抓只存在于 Arena fixture
中的对象。抓取标注路径通过字符串替换 `Aligned_obj.usd -> npy_name` 得到；USD 文件名不匹配时不会正确换路径。
普通方向过滤若筛到 0 个候选，代码会退回未过滤的前若干姿态；但配置了 `grasp_side_preference` 时不会退回，
而是返回空候选并使规划失败。

```yaml
- name: pick
  objects: [apple]
  filter_x_dir: [forward, 90]
  filter_z_dir: [downward, 140]
  pre_grasp_offset: 0.12
  post_grasp_offset_min: 0.26
  post_grasp_offset_max: 0.28
  gripper_change_steps: 20
  t_eps: 0.025
  o_eps: 1.0
  process_valid: true
  lift_th: 0.02
```

### 5.2 带人工位姿修正的抓取：`manualpick`

沿用标准抓取的标注、过滤、pre/post offset、`test_mode` 和 `gripper_change_steps` 语义，
并增加人工姿态修正：

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | 读取 `[0]`。 |
| `npy_name` | `str` | 否 | `Aligned_grasp_sparse.npy` | 抓取标注文件。 |
| `grasp_scale` | `float` | 否 | `1` | 正数。 |
| `hold_vec_weight` | 6 元数组 / `null` | 否 | `null` | 初始 pose cost。 |
| `start_lr_skill` | `bool` | 否 | `false` | 先插入一次无约束更新。 |
| `adjust_ori` | `[pose_axis, base_axis, judge]` | 否 | 无 | 轴为 `x/y/z`；只有 Piper/R5A 分支会执行。judge 只有 `min` 走最小值，其他值都按 max。 |
| `adjust_rotate_axis` | `x/y/z` | 否 | `x` | 自动修正的旋转轴。 |
| `adjust_angle_list_cfg` | `[min,max,count]` | 否 | `[-15,15,7]` | degree；count 应为正整数。 |
| `manual_adjust_ori` | `list[[axis,angle]]` | 否 | 无 | 固定欧拉旋转，angle 为 degree；当前代码会把每一项连续应用两遍，实际旋转效果为重复复合。 |
| `adjust_trans_offset` | `[x,y,z]` | 否 | `[0,0,0]` | 抓取 pose 平移修正。 |
| `pre_grasp_offset_manual` | `[x,y,z]` | 否 | 无 | pre-grasp 额外平移。 |
| `pre_grasp_offset` | `float` | 否 | `0.1` | 建议 `>= 0`。 |
| `post_grasp_offset_min/max` | `float` | 否 | `0.05/0.05` | min `<=` max。 |
| `gripper_change_steps` | `int` | 否 | `40` | 建议 `> 0`。 |
| `filter_*_dir` | 2/3 元数组 | 否 | 无 | 可先参考 `x: [forward,90]`、`z: [downward,140]`，省略 y。 |
| `direction_to_obj` | `left` / `right` | 否 | 无 | 候选侧面硬筛选。 |
| `test_mode` | `forward` / `ik` | 否 | `forward` | 候选可达性检查。 |
| `ignore_substring` | `list[str]` | 否 | `[]` | 碰撞忽略项。 |

`manualpick.is_success()` 只检查 gripper-object contact，不读取 `process_valid` 或 `lift_th`；
它的 `is_done()` 也使用代码固定的 `t_eps=1e-3`、`o_eps=5e-3`。
当前 post-grasp 分支还引用未初始化的 `self.gripper_cmd`；默认 post offset 为 `0.05`，因此该 Skill
存在运行时 `AttributeError` 风险，不能作为新任务的推荐抓取 Skill。

### 5.3 使用预定义离散姿态的抓取：`dexpick`

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | 被抓对象。 |
| `pick_pose_idx` | `int` | 否 | `0` | 必须落在 `dexpick_pose.yaml` 的姿态索引内。 |
| `pre_grasp_offset` | `float` | 否 | `0.1` | 建议 `>= 0`。 |
| `post_grasp_offset_min/max` | `float` | 否 | `0.05/0.05` | 要求 min `<=` max。 |
| `gripper_change_steps` | `int` | 否 | `40` | 建议 `> 0`。 |
| `t_eps` / `o_eps` | `float` | 否 | `1e-3` / `5e-3` | 完成阈值。 |
| `ignore_substring` | `list[str]` | 否 | `[]` | 碰撞忽略项。 |
| `process_valid` | `bool` | 否 | `true` | 是否检查速度。 |
| `lift_th` | `float` | 否 | `0.0` | `> 0` 时启用抬升检查。 |

`post_grasp_offset` 单数字段不会被读取，必须写 min/max。
对象同目录必须存在 `dexpick_pose.yaml` 且包含非空 `pick_poses`；文件缺失时构造函数不会设置
`pose_ee2o`，后续生成命令会失败。

### 5.4 对运动目标进行预测抓取：`dynamicpick`

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | 动态目标。 |
| `pick_range` | `[min,max]` | 是 | 无 | 从初始 EE world x 加上的目标会合 x offset，meter；min `<=` max。 |
| `pos_adjust_z` | `[min,max]` | 条件必填 | 无 | 提供 `pivot_angle_z` 时代码会直接索引；z 平移扰动，meter。 |
| `pivot_angle_z` | `[min,max]` | 否 | 无 | 每个候选绕局部 z 的旋转扰动，degree。 |
| `tcp_offset` | `float` | 否 | `0.125` | meter。 |
| `grasp_scale` | `float` | 否 | `1` | 正数。 |
| `time_bias` | `float` | 否 | `0` | 加到预测规划耗时，时间单位与 controller `cmd_time` 一致。 |
| `pick_bias` | `float` | 否 | `0` | 加到启动抓取的 x 阈值，meter。 |
| `pre_grasp_offset` | `float` | 否 | `0.1` | 建议 `>= 0`。 |
| `post_grasp_offset_min/max` | `float` | 否 | `0.05/0.05` | min `<=` max。 |
| `gripper_change_steps` | `int` | 否 | `40` | 建议 `> 0`。 |
| `filter_*_dir`, `direction_to_obj`, `test_mode` | 同 `pick` | 否 | 同 `pick` | filter 可先参考 `x: [forward,90]`、`z: [downward,140]`，省略 y。 |
| `process_valid`, `lift_th`, `t_eps`, `o_eps` | 同 `pick` | 否 | 同 `pick` | 完成/成功判定。 |

该 Skill 要求 Task 提供 `conveyor_velocity`，并假设目标主要沿 world x 运动；`is_ready()` 会根据速度方向、
当前 object x、预测命令耗时和上述偏置决定何时真正启动。

### 5.5 生成故意偏移抓取动作：`fail_pick`

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | 目标对象。 |
| `grasp_x_offset_min` / `grasp_x_offset_max` | `float` | 否 | `0.05/0.1` | base x 的偏移绝对值范围，正负号随机。 |
| `grasp_y_offset_min` / `grasp_y_offset_max` | `float` | 否 | `0.05/0.1` | base y 的偏移绝对值范围，正负号随机。 |
| `post_grasp_offset_min/max` | `float` | 否 | `0.05/0.05` | 抓取后抬升范围。 |
| `filter_*_dir` | 2/3 元数组 | 否 | 无 | 偏移前仍会过滤；可先参考 `x: [forward,90]`、`z: [downward,140]`。 |
| `direction_to_obj` | `left` / `right` | 否 | 无 | 候选侧面过滤。 |
| `test_mode` | `forward` / `ik` | 否 | `forward` | 可达性检查。 |
| `gripper_change_steps` | `int` | 否 | `10` | 建议 `> 0`。 |
| `t_eps` / `o_eps` | `float` | 否 | `1e-3/5e-3` | 完成阈值。 |
| `ignore_substring` | `list[str]` | 否 | `[]` | 碰撞忽略项。 |

`fail_pick.is_success()` 固定返回 true；“失败”来自故意偏移的动作数据，而不是 Skill 状态。
post-grasp pose 从未偏移的原始候选复制，只增加 z，因此它会从偏抓点返回原候选 XY 再抬升。

## 6. 放置与物体操作 Skill

### 6.1 按目标几何和约束进行通用放置：`place`

`objects = [被放对象, 放置目标]`。被放对象通常应已被同一手臂抓住。

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | 长度至少 2。 |
| `offset_place_obj_axis` | `[x,y,z]` | horizontal 是 | 无 | horizontal 模式的第二条局部轴。 |
| `place_direction` | `vertical` / `horizontal` | 否 | `vertical` | 其他值没有初始化目标点的有效分支。 |
| `position_constraint` | `gripper` / `object` | 否 | `gripper` | 见 3.2。 |
| `align_pick_obj_axis`, `align_place_obj_axis` | `[x,y,z]` | 否 | 无 | 物体对齐轴；horizontal 模式要求 `align_place_obj_axis`。 |
| `align_obj_tol` | `float` | 否 | 无 | degree，建议 `[0,180]`。 |
| `x_ratio_range`, `y_ratio_range` | `[min,max]` | 否 | `[0.4,0.6]` | 建议 `0 <= min <= max <= 1`。 |
| `z_ratio_range` | `[min,max]` | 否 | `[0.4,0.6]` | horizontal 目标点采样使用；同样建议在 `[0,1]`。 |
| `pre_place_z_offset` | `float` | 否 | `0.2` | vertical 模式的预放置高度。 |
| `place_z_offset` | `float` | 否 | `0.1` | vertical 模式的最终高度。 |
| `pre_place_align`, `place_align` | `float` | 否 | `0.2/0.1` | 只在 horizontal + `position_constraint: object` 时沿 align 轴使用。 |
| `pre_place_offset`, `place_offset` | `float` | 否 | `0.2/0.1` | 只在 horizontal + `position_constraint: object` 时沿 offset 轴使用。 |
| `filter_*_dir` | 2/3 元数组 | 否 | 无 | 建议先用 `x: [forward,45]`、`z: [downward,150]`，省略 y。 |
| `test_mode` | `forward` / `ik` | 否 | `forward` | 候选可达性检查。 |
| `pre_grasp_offset` | `float` | 否 | `0.1` | 仅在非 batch 候选检查中作为“是否同时检查最终 place pose”的开关；`> 0` 时检查。 |
| `success_mode` | 见下表 | 否 | `3diou` | 选择成功判定。 |
| `success_xy_margin` | `float` | 否 | `0.015` | `xybbox` 内缩 margin，建议 `>= 0`。 |
| `threshold` | `float` | 否 | `0.03` | `left/right` 的额外距离。 |
| `success_th` | `float` | 否 | `0.0` | 只用于 `flower`、`cup` 的 IoU 阈值，语义范围 `[0,1]`；不作用于 `3diou`。 |
| `place_part_prim_path` | `str` | 否 | 无 | 用目标对象的子 prim 计算 bbox。 |
| `pre_place_hold_vec_weight` | 6 元数组 | 否 | 无 | 在当前 EE pose、生成 place 轨迹前更新 pose cost。 |
| `post_place_hold_vec_weight` | 6 元数组 | 否 | 无 | 名称虽为 post，实际在到达 pre-place 后、最终 place 前更新 pose cost。 |
| `hesitate_steps` | `int` | 否 | `0` | `>= 0`；松爪前保持。 |
| `gripper_change_steps` | `int` | 否 | `10` | 建议 `> 0`；松爪命令次数。 |
| `post_place_vector` | `[x,y,z]` | 否 | 无 | 松爪后的 EE 局部坐标 retreat 向量，代码用 place rotation 变换到 base。 |
| `t_eps` / `o_eps` | `float` | 否 | `1e-3/5e-3` | 子命令完成阈值。 |
| `output_root` | `str` | 否 | `output/ros_bridge/skills` | `place_success_check_snapshot.json` 输出根目录。 |
| `place_align_axis`, `pick_align_axis` | `[x,y,z]` | 否 | 无 | 当前只在构造函数保存，后续不读取。 |
| `constraint_gripper_x` | `bool` | 否 | `false` | 当前只保存，后续不读取。 |
| `align_plane_x_axis`, `align_plane_y_axis` | `[x,y,z]` | 否 | 无 | 当前只保存，后续不读取。 |

| `success_mode` | 判定 |
| --- | --- |
| `3diou` | IoU 大于 `is_success(th)` 的函数参数；workflow 无参数调用，当前实际阈值固定为 `0.0`。 |
| `height` | 对象相对 base 的高度低于 place EE 高度条件。 |
| `xybbox` | 对象中心 XY 位于目标 bbox 内，并扣除 `success_xy_margin`。 |
| `left` / `right` | 对象 x 位于目标 bbox 左/右侧并超过 `threshold`。 |
| `flower` | XY 在 bbox 内且 3D IoU 大于 `success_th`。 |
| `cup` | 杯底高度满足条件且 IoU 大于 `success_th`。 |

姿态过滤无候选时，当前实现不会失败，而是退回未过滤随机旋转中的前 `CUROBO_BATCH_SIZE` 个；
因此看到“Warning: No matrix satisfies constraints”时，不能认为过滤约束仍然生效。
`align_pick_obj_axis`、`align_place_obj_axis`、`align_obj_tol` 只有三者同时存在才启用对齐过滤。
horizontal + `position_constraint: gripper` 不读取 `pre_place_align/place_align`，而是沿 align 轴使用
`pre_place_z_offset/place_z_offset + ee2o_distance`。

```yaml
- name: place
  objects: [apple, tray]
  position_constraint: object
  place_direction: vertical
  x_ratio_range: [0.35, 0.65]
  y_ratio_range: [0.35, 0.65]
  pre_place_z_offset: 0.1
  place_z_offset: 0.1
  gripper_change_steps: 20
  success_mode: xybbox
```

### 6.2 按对象放置范围进行几何放置：`dexplace`

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | `[被放对象, 目标]`。 |
| `gripper_axis` | `[x,y,z]` | 否 | `null` | 当前只保存、不读取；实际轴由容器位置减初始 EE 位置计算。 |
| `camera_axis_filter` | `[{direction: [x,y,z]}, {degree: [min,max]}]` | 否 | 无 | direction 非零；degree min `<=` max。 |
| `place_part_prim_path` | `str` | 否 | 无 | 目标子 prim。 |
| `gripper_change_steps` | `int` | 否 | `10` | 建议 `> 0`。 |
| `t_eps` / `o_eps` | `float` | 否 | `1e-3/5e-3` | 完成阈值。 |

若被放对象旁有 `place_range.yaml`，代码读取其中 `x_range/y_range`；否则均为 `[0.4,0.6]`。
成功判定要求对象世界位置的 x/y 位于目标 bbox 内，且 z 不低于 bbox 底面；不检查 bbox 顶面。

### 6.3 推动对象接近另一对象：`move`

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | `[被移动对象, 目标对象]`。 |
| `success_threshold` | `float` | 是 | 无 | 建议 `> 0`；同时用于 EE 到位和两对象距离。 |
| `delta_trans` | `list[[x,y,z]]` | 否 | `[[0,0,0]]` | 至少一个三元向量；最后一个决定最终 EE 目标。 |
| `hold_vec_weight` | 6 元数组 | 否 | `[0,0,0,0,0,0]` | pose cost。 |
| `ignore_substring` | `list[str]` | 否 | `[]` | 碰撞忽略项。 |
| `invisible_object` | `list[str]` | 否 | 无 | 代码读取 `[0]`，开始时显示，成功后隐藏。 |

`move` 只计算世界 XY 平移，主动把 z 方向移动清零。
`delta_trans` 必须至少包含一个向量，否则读取 `delta_trans[-1]` 会失败。

### 6.4 保持抓取关系旋转对象：`rotate__obj`

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | `objects[0]` 为被旋转对象。 |
| `success_threshold_move` | `float` | 是 | 无 | EE 平移阈值，建议 `> 0`。 |
| `success_threshold_rotate` | `float` | 是 | 无 | 姿态角阈值，rad，建议 `(0,pi]`。 |
| `rotate_obj_euler_delta` | `[[min_xyz],[max_xyz]]` | 否 | 全零范围 | degree，各维 min `<=` max；缺省表示不改变姿态。 |
| `trans_offset` | `[x,y,z]` | 否 | 无 | 最终目标平移。 |
| `first_motion` | `move` / `rotate` | 否 | 无 | 先平移或先旋转。 |
| `move_offset`, `rotate_offset` | `[x,y,z]` | 否 | `[0,0,0]` | 第一阶段分支偏移。 |
| `rotate_only` | `bool` | 否 | `false` | 只做姿态变化。 |
| `obj_axis_offset` | `list[[axis,offset]]` | 否 | 无 | axis 仅 `x/y/z`；沿对象局部轴偏移。 |
| `gripper_state` | `float` | 否 | `-1.0` | 执行期间夹爪状态。 |
| `ignore_substring` | `list[str]` | 否 | `[]` | 碰撞忽略项。 |
| `dummy_forward` | `dict` | 否 | 无 | 可含 `num_steps`, `gripper_state`。 |
| `ctrl_list` | `list[[index,degree,mode]]` | 否 | `[]` | 供 dummy target joint 计算，mode 为 `abs/delta`。 |

`gripper_state` 只有精确等于 `-1.0` 时 close，其他值都走 open。成功判定同时要求最终位置和姿态达标。

### 6.5 携带对象接近目标并可调整朝向：`approach__rotate`

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | `[被移动对象, 被接近对象]`。 |
| `success_threshold` | `float` | 是 | 无 | 平移成功阈值，建议 `> 0`。 |
| `rotate` | `dict` / `null` | 否 | `null` | 可选旋转子配置，见下表。 |
| `distance` | `float` | 否 | `0.1` | 距被接近对象的距离，建议 `>= 0`。 |
| `approach_axis` | `+x/-x/+y/-y/+z/-z` | 否 | `+x` | 被移动对象的局部接近轴。 |
| `obj_yaw_offset` | `float` | 否 | `0` | degree，绕世界 z 修正接近方向。 |
| `z_offset` | `float` | 否 | `0.0` | 最终 EE z 偏移。 |
| `obj_axis_offset` | `list[[axis,offset]]` | 否 | 无 | axis 仅 `x/y/z`。 |
| `hold_vec_weight` | 6 元数组 / `null` | 否 | `null` | pose cost。 |
| `ignore_substring` | `list[str]` | 否 | `[]` | 碰撞忽略项。 |
| `dummy_forward` | `dict` | 否 | 无 | **当前不可用**：启用会调用直接抛出 `NotImplementedError` 的 `get_tgt_js()`。 |

| `rotate.*` | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `type` | `random` / `towards` | 是 | 随机目标姿态，或朝向另一对象。 |
| `success_threshold` | `float` | 是 | 姿态成功阈值。 |
| `objects` | `list[str]` | towards 是 | 代码读取 `[1]` 作为朝向目标。 |
| `rotate_obj_euler` | `[[min_xyz],[max_xyz]]` | random 否 | 默认全零，degree。 |

### 6.6 移动、释放并翻转对象：`flip`

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | 被翻转对象。 |
| `gripper_axis` | `[x,y,z]` | 实际是 | `false` | 必须提供非零三元向量；默认 false 会导致数组归一化失败。 |
| `open_wait_steps` | `int` | 否 | `20` | `>= 0`。 |
| `t_eps` / `o_eps` | `float` | 否 | `1e-3/5e-3` | 完成阈值。 |

`flip` 的运动位置包含代码内固定的随机范围，YAML 不能覆盖。`ee_axis` 即使出现在旧 YAML 中也不会读取。

## 7. 位姿、轨迹和基础控制 Skill

### 7.1 将末端执行器移动到目标位姿：`goto__pose`

有两种姿态来源：直接给定 `quaternion/euler`；或不给姿态，由参考对象和对齐轴批量采样。

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `position` | `[x,y,z]` | 是 | 无 | 目标位置。 |
| `objects` | `list[str]` | 未给姿态时是 | 无 | 代码读取 `[0]`。 |
| `align_obj_axis`, `align_ref_axis` | `[x,y,z]` | 未给姿态时是 | 无 | 非零轴。 |
| `align_obj_tol` | `float` | 未给姿态时是 | 无 | degree。 |
| `quaternion` | `[w,x,y,z]` | 否 | 无 | 与 `euler` 二选一；同时给出时由 `get_orientation` 决定优先级。 |
| `euler` | `[r,p,y]` | 否 | 无 | degree。 |
| `position_constraint` | `gripper` / `object` | 否 | `gripper` | 位置约束对象。 |
| `filter_*_dir` | 2/3 元数组 | 否 | 无 | 姿态采样可先参考 `x: [forward,45]`、`z: [downward,150]`，省略 y。 |
| `test_mode` | `forward` / `ik` | 否 | `forward` | 采样姿态检查。 |
| `interp_nums` | `int` | 否 | `1` | 应 `>= 1`。 |
| `gripper_state` | 数值 | 否 | `1` | 等于 1 时 open，否则 close。 |
| `max_noise_m` / `max_noise_deg` | `float` | 否 | `0.0/0` | 建议 `>= 0`。 |
| `ignore_substring` | `list[str]` | 否 | `[]` | 碰撞忽略。 |
| `frame` | `str` | 否 | `robot` | 当前只保存、不读取；不会执行坐标转换，`position` 直接作为 controller/base 坐标。 |

`goto__pose.is_success()` 当前使用“位置满足或姿态满足”，固定阈值为 `0.005 m / 0.087 rad`。
`is_done()` 阈值也固定为 `1e-3/5e-3`，YAML 中的 `t_eps/o_eps` 不会读取。

### 7.2 依次跟踪随机采样路点：`track`

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `way_points_trans.min/max` | `[x,y,z]` | 是 | 无 | 各维 min `<=` max。 |
| `way_points_ori` | quaternion | 是 | 无 | 基准姿态。 |
| `frame` | `str` | 否 | `robot` | 只有 `robot` 有执行分支；其他值抛 `NotImplementedError`。 |
| `way_points_num` | `int` | 否 | `1` | 应 `> 0`。 |
| `max_noise_deg` | `float` | 否 | `5` | 建议 `>= 0`。 |
| `T_tcp_2_ee` | 4x4 matrix | 否 | 单位阵 | TCP 到 EE 变换。 |
| `target` | `str` | 否 | 无 | 当前代码完全不读取。 |

`track` 强依赖 `task.fixtures['table']`；`way_points_num` 必须 `> 0`。成功判定同样是位置或姿态任一满足，
固定阈值为 `0.005 m / 0.087 rad`。

### 7.3 移动到固定扫描或观察姿态：`scan`

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | 读取 `[0]` 作为关联对象。 |
| `t_eps` / `o_eps` | `float` | 否 | `1e-3/5e-3` | 完成阈值。 |
| `process_valid` | `bool` | 否 | `true` | 是否要求机器人和对象速度稳定。 |

扫描路径和姿态主要由实现固定，并非通用可配置扫描器。

### 7.4 保持当前末端位姿并等待：`wait`

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | 读取 `[0]`；仅用于关联对象。 |
| `success_threshold` | `float` | 是 | 无 | EE 与开始位置距离阈值，建议 `> 0`。 |
| `wait_steps` | `int` | 否 | `50` | `>= 0`；为 0 时只执行两条更新命令。 |
| `gripper_state` | 数值 | 否 | `-1.0` | 精确等于 `-1.0` close，否则 open。 |
| `ignore_substring` | `list[str]` | 否 | `[]` | 碰撞忽略。 |

### 7.5 单独执行夹爪开合：`gripper__action`

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `gripper_state` | `int` | 是 | 无 | 只能 `1` 或 `-1`，否则 `NotImplementedError`。 |
| `vel` | `float` / `null` | 否 | `null` | 传给 controller 的夹爪速度。 |
| `wait_steps` | `int` | 否 | `10` | `>= 0`；重复命令次数。 |

该 Skill 的 `is_success()` 固定 true；`post_action`、`post_action_offset` 不会读取。

### 7.6 执行 home、关节目标或相对末端运动：`heuristic__skill`

推荐用它替代旧 `home`。

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `mode` | 枚举 | 否 | `home` | `home`, `abs_qpos`, `rel_qpos`, `rel_ee`；其他值 `ValueError`。 |
| `value` | 数组 / 4x4 matrix | 否 | 见下表 | 目标值；各 mode 都有代码默认值。 |
| `move_steps` | `int` | 否 | `50` | 必须 `> 0`，否则除零。 |
| `t_eps` | `float` | 否 | `0.088` | 关节向量范数阈值。 |
| `gripper_state` | `float` | 否 | 机器人当前值 | 建议 `1/-1`。 |

| mode | `value` 语义 | 缺省值 |
| --- | --- | --- |
| `home` | 忽略 `value`，目标为 arm home joints。 | home joints |
| `abs_qpos` | 与 arm joint 数量相同的绝对关节数组。 | home joints |
| `rel_qpos` | 名称虽表示相对关节，但当前没有加当前关节值，实际直接把 `value` 当绝对目标。 | 全零绝对目标 |
| `rel_ee` | 当前计算 `T_target = value @ T_current`，再通过 controller plan 求关节。 | 单位阵 |

`rel_ee` 不支持 batch controller。插值系数被乘以 1.25，因此轨迹会越过目标，成功时再提前清空命令。

### 7.7 使用旧式固定轨迹返回 home：`home`

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `gripper_state` | `float` | 否 | 机器人当前值 | 建议 `1/-1`。 |

其余行为不可配置：固定生成 50 步，分母为 40，完成阈值固定 `0.088`。因此同样存在 25% 越过目标的轨迹，
且当前 Task YAML 未使用它。

### 7.8 按关节索引直接设置或增量控制：`joint__ctrl`

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `ctrl_list` | `list[[index,angle_deg,mode]]` | 否 | `[]` | index 必须在 arm joint 数组内；mode 仅 `abs/delta`。 |
| `num_steps` | `int` | 否 | `10` | 应 `> 0`。 |
| `gripper_state` | `float` | 否 | `1.0` | dummy forward 期间状态。 |
| `success_threshold_js` | `float` | 否 | `5e-3` | 关节向量范数阈值。 |
| `objects` | `list[str]` | 否 | 无 | 当前控制逻辑不读取。 |

`abs` 把 degree 转 rad 后覆盖关节，`delta` 在当前值上累加。Piper 机器人还会对特定关节做代码内 clamp。

## 8. 通过 Nav2 移动底盘到固定或动态目标：`navigate`

`navigate` 不使用 arm controller 规划命令，而是通过 workflow 的持久 Nav2 session manager 阻塞等待底盘到达。
目标来源优先级为：`approach` 动态目标 > `goal` 命名位置 > `goal_x/y/yaw` 直接坐标。
FollowPath 固定使用 `RotationShimController -> MPPIController`；Skill YAML 不提供 controller 启停或替换选项。

### 8.1 固定目标参数

| 字段 | 类型 | 必填 | 默认值 | 硬约束与语义 |
| --- | --- | --- | --- | --- |
| `goal` | `str` | 三选一 | 无 | 从 Task 顶层 `positions[goal]` 读取 `x/y/yaw`。 |
| `goal_x`, `goal_y`, `goal_yaw` | `float` | 三选一 | 无 | 直接 world/map 坐标；yaw 会 wrap 到 `[-pi,pi)`。 |
| `xy_goal_tolerance` | `float` | 否 | `0.10` | skill 位置容差，建议 `> 0`。 |
| `yaw_goal_tolerance` | `float` | 否 | `0.10` | skill yaw 容差，rad，建议 `(0,pi]`。 |
| `skill_xy_goal_tolerance`, `skill_yaw_goal_tolerance` | `float` | 否 | 同上 | 兼容别名；显式非 skill 前缀字段优先。 |
| `startup_timeout_sec` | `float` | 否 | `60.0` | 应 `> 0`。 |
| `runtime_timeout_sec` | `float` | 否 | `240.0` | 应 `> 0`。 |
| `scene_name` | `str` | 否 | task name | 地图/会话标识。 |
| `output_root` | `str` | 否 | `output/ros_bridge/skills` | 当前只保存在 Skill；持久 manager 已由 workflow 初始化，本字段不会覆盖其输出目录。 |

命名 `positions` 是相对 floor frame 的。代码读取 `task.fixtures['floor']` 的世界位姿，并完整应用 floor yaw：

```text
world_x = floor_x + local_x*cos(floor_yaw) - local_y*sin(floor_yaw)
world_y = floor_y + local_x*sin(floor_yaw) + local_y*cos(floor_yaw)
world_yaw = wrap(floor_yaw + local_yaw)
```

### 8.2 动态 `approach` 参数

| 字段 | 类型 | 必填 | 默认值 | 硬约束与语义 |
| --- | --- | --- | --- | --- |
| `approach` | `str` | 动态模式是 | 无 | 目标 object/fixture 名。非空即启用动态模式。 |
| `approach_min_distance` | `float` | 否 | `0.45` | **必须 `> 0`**。 |
| `approach_max_distance` | `float` | 否 | `1.15` | **必须 `>= min_distance`**。 |
| `approach_sample_count` | `int` | 否 | `128` | **必须 `> 0`**。 |
| `approach_footprint_padding` | `float` / `null` | 否 | `null` | 非 null 时**必须 `>= 0`**。 |
| `approach_sampling_random` | `bool` 或可转换标量 | 否 | `false` | bool/数值按真假转换；字符串仅 `1/true/yes/on` 视为 true。false 用 golden-angle，true 用随机采样。 |
| `approach_sampling_seed` | `int` / `null` | 否 | `null` | 随机模式未给 seed 时从 `os.urandom` 生成。 |
| `approach_arm` | `left` / `right` / `null` | 否 | `null` | 用 arm-base 可达性上下文调整 yaw。 |
| `approach_object_armbase_xy` | `[x,y]` / `null` | 否 | `null` | 长度必须 2；提供时 `approach_arm` 条件必填。 |

候选点会先做静态 footprint 碰撞检查，再请求 Nav2 `ComputePathToPose`，最后从静态可行且路径可达的
候选中排序选择。`ApproachConfig` 数据类中的 `static_free_value_min=250` 当前不能通过 Skill YAML 覆盖。

### 8.3 地图和 Nav2 覆盖

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `map_output_dir` | `str` | 否 | `output/nav2_maps` | 地图输出目录。 |
| `map_resolution` | `float` | 否 | `0.02` | meter/cell，应 `> 0`。 |
| `map_z_min`, `map_z_max` | `float` | 否 | `0.0/0.35` | 高度过滤，语义要求 min `<=` max。 |
| `map_include_visual_wall_geometry` | `bool` | 否 | `true` | 是否纳入 visual wall geometry。 |
| `nav2_skill` | `dict` | 否 | `{}` | 深层覆盖 robot 的 Nav2 skill 配置。 |

Skill 层 `xy_goal_tolerance/yaw_goal_tolerance` 会被写入 `nav2_skill.controller_server.goal_checker`；
而 robot `base_cfg` 中最终 goal checker 值也会参与运行时容差计算。调试时应同时查看 skill 快照中的
`world_dist`、`nav_dist`、yaw error 和最终生效配置。

```yaml
positions:
  nav_to_pick: {x: -0.08, y: -0.72, yaw: 1.5707963267948966}

skills:
  - panda_omron:
      - base:
          - id: nav_to_pick
            name: navigate
            depends_on: []
            goal: nav_to_pick
            xy_goal_tolerance: 0.1
            yaw_goal_tolerance: 0.1
```

## 9. Articulation Skill

`open`、`close`、`artpreplan`、`rotate` 均要求 `objects[0]` 指向 `ArticulatedObject`，并通过
`KPAMPlanner` 消费 `planner_setting`。这不是一个只含少量开关的配置：keypoint 名必须与对象的
articulated info 和 planner 实现一致。

当前 `KPAMPlanner` 只为 robot name 包含 `franka`、`split_aloha` 或 `lift2` 的分支设置 EE/DOF；
其他机器人会 `NotImplementedError`，不能仅靠 YAML planner_setting 扩展。

### 9.1 通用参数

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | 长度至少 1；`objects[0]` 为 articulation。 |
| `planner_setting` | `dict` | 是 | 无 | KPAM planner 配置。 |
| `obj_info_path` | `str` | 否 | 对象原配置 | 开始规划前调用 `update_articulated_info()`。 |
| `ignore_substring` | `list[str]` | 否 | `[]` | `open/close` 的碰撞忽略项。 |
| `collision_valid` | `bool` | 否 | `true` | `open/close` 是否把禁碰撞接触纳入成功判定。 |
| `process_valid` | `bool` | 否 | `true` | `open/close` 是否检查机器人和 articulation 速度。 |

### 9.2 `planner_setting`

| 字段 | 类型 | 必填 | 有效范围与语义 |
| --- | --- | --- | --- |
| `constraint_list` | `list[dict]` | 是 | planner 直接索引并逐项复制，见 9.3。 |
| `pre_actuation_motions` | `list[[motion,value]]` | 是 | 可为空列表；不提供时 planner 不会初始化 actuation 属性。motion 为 `translate_x/y/z` 或 `rotate`。 |
| `keypose_random_range.position` | `{x_min,x_max,y_min,y_max,z_min,z_max}` | 是 | meter，各 min `<=` max。 |
| `keypose_random_range.orientation` | 同上 | 是 | degree，各 min `<=` max。 |
| `contact_pose_index` | `int` | open/close/artpreplan 是 | `open/close/rotate` 必须是 keypose 有效索引；`artpreplan` 直接读取但当前不使用；`rotate` 虽用 `.get()`，缺失后仍会在算术比较时报错。 |
| `success_threshold` | `float` | open/close/artpreplan 是 | `open/close` 的 joint 阈值；`artpreplan` 只保存不使用；`rotate` 缺省 `0.785`。revolute 通常以 rad 表示。 |
| `pre_actuation_times`, `post_actuation_times` | `list[float]` | `actuation_time` 存在时是 | 相对当前仿真时间的 waypoint 时间。 |
| `post_actuation_motions` | `list[[motion,value]]` | 否 | 未提供时只有在 `pre_actuation_motions` 分支才会自动设为空。 |
| `actuation_time` | `float` | 否 | 提供时还必须提供 `pre_actuation_times`、`post_actuation_times` 以及 pre/post motions。 |
| `modify_actuation_motion` | `[motion,float]` | 否 | 修改求得的 actuation pose；motion 同上。value 必须是 Python float 语义。 |
| `success_mode` | `str` | 否 | open: `abs/normal`；close: `zero/dis_to_init`。 |
| `update_art_joint` | `bool` | 否 | 默认 false；`open` 每次 update 都保持当前 joint target，`close/artpreplan` 只在各自成功时同步。 |
| `additional_labels` | `dict` | 否 | 仅 `rotate` 使用；key 是 `art_obj.asset_relative_path`，value 直接是 `[motion,float]`，不是再嵌套一层。 |
| `task_name` | `str` | 否 | 兼容/描述字段；当前 `KPAMPlanner` 不读取。 |
| `category_name` | `str` | 否 | 兼容/描述字段；当前 `KPAMPlanner` 不读取。 |
| `tool_keypoint_name_list` | `list[str]` | 否 | 当前 planner 不读取；实际工具 keypoint 来自 robot 配置。 |
| `object_keypoint_name_list` | `list[str]` | 否 | 当前 planner 不读取；实际对象 keypoint 来自 articulated info。 |

### 9.3 `constraint_list[]`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `name` | `str` | **真正的运行时分发键**，只能使用下表列出的精确名称；未知名称直接报 `undefined constraint`。 |
| `type` | `str` | 当前 `solve_ik_kpam()` 不读取，仅保留作描述/兼容；不能靠它选择约束实现。 |
| `tolerance` | `float` | 非负容差。 |
| `keypoint_name`, `target_keypoint_name` | `str` | 工具/对象 keypoint。 |
| `axis_from_keypoint_name`, `axis_to_keypoint_name` | `str` | 工具两 keypoint 定义的轴。 |
| `target_axis`, `target_axis_frame` | `str` | articulated info 中的轴和坐标系。 |
| `target_axis_from_keypoint_name`, `target_axis_to_keypoint_name` | `str` | 对象 keypoint 定义的目标轴。 |
| `cross_target_axis1_from_keypoint_name`, `cross_target_axis1_to_keypoint_name` | `str` | cross-axis 的第一目标轴。 |
| `target_inner_product` | `float` | 轴内积目标，几何有效范围 `[-1,1]`。 |

| `name` 精确值 | 主要必需字段 | 当前行为 |
| --- | --- | --- |
| `fingers_contact_with_link0` | `keypoint_name`, `target_keypoint_name`, `tolerance` | 工具点与对象点的位置约束。 |
| `fingers_orthogonal_to_link0` | `axis_from/to_keypoint_name`, 两个 `cross_target_axis1_*`, `target_axis`, `target_inner_product`, `tolerance` | 工具轴与两条对象轴叉乘所得方向的角度约束。 |
| `hand_parallel_to_link0_edge` | `axis_from/to_keypoint_name`, `target_axis`, `target_inner_product`, `tolerance` | 工具轴与 articulated axis 的角度约束。 |
| `hand_parallel_to_axis_computed_by_keypoints` | `axis_from/to_keypoint_name`, `target_axis_from/to_keypoint_name`, `target_inner_product`, `tolerance` | 工具轴与对象两 keypoint 构成轴的角度约束。 |
| `hand_parallel_to_link0_move_axis` | 与 `hand_parallel_to_link0_edge` 相同 | 使用对象 move axis。 |
| `hand_parallel_to_link0_edge_door` | `axis_from/to_keypoint_name`, 两个 `cross_target_axis1_*`, `target_inner_product` | 当前分支在使用 `tol` 前没有本地赋值，存在复用旧变量或未定义错误风险，不建议新配置使用。 |

`target_axis` 字符串在进入 IK 前只对以下值做对象轴替换：`link0_contact_axis`、
`object_link0_move_axis`、`object_link0_contact_axis`、`object_link0_vertical_axis`。其他字符串会原样传下去，
随后通常无法参与矩阵计算。

### 9.4 规划并打开 articulated object：`open`

- `planner_setting.contact_pose_index`、`success_threshold` 必填。
- Skill 层 `output_root` 默认 `output/ros_bridge/skills`，用于 `open_plan_failure_snapshot.json`。
- `success_mode` 默认 `abs`。
- `abs`：`abs(current_joint - initial_joint) >= abs(success_threshold)`。
- `normal`：`current_joint - initial_joint >= abs(success_threshold)`，只接受正向运动。
- `update_art_joint` 默认 false；true 时每次 update 都把当前 articulation joint 写成 position target。
- planner 无 keyframe 时返回空命令，workflow 会以 `empty_manip_list` 判失败。

### 9.5 规划并关闭 articulated object：`close`

- `planner_setting.contact_pose_index`、`success_threshold` 必填。
- `success_mode` 默认 `zero`。
- `zero`：`abs(current_joint) <= success_threshold`，因此阈值必须 `>= 0`。
- `dis_to_init`：`abs(current_joint - initial_joint) >= abs(success_threshold)`。
- `update_art_joint` 默认 false；只有 `is_success()` 已成立时才同步当前 joint target。

### 9.6 预先验证 articulation 的 KPAM 规划：`artpreplan`

- `planner_setting.contact_pose_index`、`success_threshold` 因构造函数直接读取而必填。
- `update_art_joint` 默认 false。
- 用于确认 KPAM 能生成 keypose，但不会执行这些 keypose，只生成一条保持当前 EE 的更新命令。
- `contact_pose_index` 和 `success_threshold` 当前不参与成功判定。
- 成功判定是 EE 位置或姿态任一接近规划开始时的 pose，固定阈值 `0.005 m / 0.087 rad`。

### 9.7 操作 articulation 关节完成旋转：`rotate`

- `planner_setting.success_threshold` 默认 `0.785`，即约 45 degree 对应的 rad。
- 成功条件为 articulation joint 相对初始位置的绝对变化达到阈值。
- `additional_labels[asset_relative_path]` 可覆盖 planner 的 `modify_actuation_motion`。
- 当前 Task YAML 未使用，运行前需要针对目标资产验证 contact index 和 keypose。

## 10. 根据流体粒子和容器姿态判断倒水成功：`pour__water__succ`

该 Skill 的主要作用是成功判定，不负责生成完整倒水轨迹。它只生成一条保持当前关节的
`dummy_forward` 命令，然后统计目标容器 XY 圆柱投影内的粒子数量。

| 字段 | 类型 | 必填 | 默认值 | 有效范围与语义 |
| --- | --- | --- | --- | --- |
| `container_name` | `str` | 否 | `cup` | `task.objects` 中的目标容器。 |
| `container_radius` | `float` | 否 | `0.025` | 必须 `> 0` 才有实际统计区域。 |
| `particle_num_th_min` | `int` | 否 | `50` | 成功要求粒子数严格 `>` 此值。 |
| `particle_num_th_max` | `int` | 否 | `300` | 成功要求粒子数严格 `<` 此值；应大于 min。 |
| `container_up` | `list[[name,axis,threshold]]` | 否 | `[]` | axis 为 `x/y/z`；旋转矩阵对应分量必须 `> threshold`，阈值语义范围 `[-1,1]`。 |
| `translation` | `[x,y,z]` | 否 | 当前 EE | 会在构造时解析和扰动，但当前命令生成不会移动到该目标。 |
| `quaternion`, `euler` | quaternion / Euler | 否 | 当前 EE | 同样会解析，但当前命令生成不会使用目标位姿。 |
| `frame` | `str` | 否 | `robot` | 当前不影响粒子成功判定。 |
| `max_noise_m`, `max_noise_deg` | `float` | 否 | `0.05/5` | 只影响上述当前未执行的目标。 |
| `gripper_state` | `float` | 否 | `1.0` | dummy forward 时的夹爪状态。 |

`gripper: close` 等旧字段不会被读取。

## 11. 配置时必须知道的实现边界

### 11.1 未知字段不一定报错

Skill 没有统一的严格 schema。一个拼错的可选字段可能被静默忽略并回退默认值。因此：

- 不要把“YAML 能加载”当作字段生效证明。
- 修改参数后应从对应 Skill 的运行快照、命令序列或成功判定确认。
- 本文明确标为“不读取”的字段不应继续用于新配置。

### 11.2 空命令的含义

- 普通 Skill 规划后产生空 `manip_list` 且 `is_ready()` 为 true，workflow 判为 `empty_manip_list` 失败。
- `navigate` 特意以空 manip list 启动异步 Nav2，会通过 `is_ready()`/`update()` 走独立路径。
- `dynamicpick` 也可能先等待目标进入窗口，不能仅凭初始空命令判定配置错误。

### 11.3 成功与完成不同

`is_done()` 只表示命令耗尽或提前终止；workflow 随后还会检查 `is_success()`。
例如 `pick` 需要接触、可选速度稳定和可选抬升；`place` 需要几何关系；`navigate` 需要 manager 返回成功。
`gripper__action` 和 `fail_pick` 则固定 success，不能用于证明真实物理结果。

### 11.4 当前无效或高风险配置

| 项目 | 当前状态 |
| --- | --- |
| `pick.close_wait_steps` | 不读取；使用 `gripper_change_steps`、`post_close_hold_steps`。 |
| `dexpick.post_grasp_offset` | 不读取；使用 min/max。 |
| `manualpick.update_pose_cost_metric_none` | 不读取。 |
| `manualpick.manual_adjust_ori` | 每个旋转被重复应用两遍。 |
| `manualpick` post-grasp | 引用未初始化的 `self.gripper_cmd`，默认路径存在 `AttributeError` 风险。 |
| `gripper__action.post_action`, `post_action_offset` | 不读取。 |
| `flip.ee_axis` | 不读取。 |
| `pour__water__succ.gripper` | 不读取。 |
| `approach__rotate.dummy_forward` | 会进入未实现函数，当前不可用。 |
| `home` | 固定步数且过冲；优先用 `heuristic__skill mode: home`，但后者也有 1.25 插值系数。 |
| `rotate` | 已注册但当前 Task YAML 无使用样例，需资产级 runtime 验证。 |
| `place` 无合法姿态候选 | 会回退到未过滤随机姿态，不会因过滤为空自动失败。 |
| `goto__pose` / `track` / `artpreplan` | 成功判定使用位置或姿态任一满足，而不是同时满足。 |
| `heuristic__skill.rel_qpos` | 当前实际按绝对关节目标执行。 |
| `dexplace.gripper_axis` | 只保存、不读取。 |
| `place.place_align_axis`, `pick_align_axis`, `constraint_gripper_x` | 只保存、不读取。 |
| `navigate.output_root` | 不会覆盖 workflow 已创建的持久 Nav2 manager 输出根目录。 |

## 12. 标准移动操作样例：apple 放入 tray

以下使用 apple → tray 的实际 5 节点 Skill 序列。示例只省略与 Skill 无关的对象、相机和场景配置：

```yaml
skills:
  - panda_omron:
      - base:
          - name: navigate
            id: nav_to_pick
            depends_on: []
            approach: apple_0_id9008
            approach_arm: left
            approach_object_armbase_xy: [0.5, 0.0]
            xy_goal_tolerance: 0.1
            yaw_goal_tolerance: 0.1
          - name: navigate
            id: nav_to_place
            depends_on: [pick_apple_0_id9008]
            approach: metal_tray_0_id9016
            approach_arm: left
            approach_object_armbase_xy: [0.5, 0.0]
            xy_goal_tolerance: 0.1
            yaw_goal_tolerance: 0.1
        left:
          - name: pick
            id: pick_apple_0_id9008
            depends_on: [nav_to_pick]
            objects: [apple_0_id9008]
            filter_x_dir: [forward, 90]
            filter_z_dir: [downward, 140]
            grasp_side_preference: toward_arm
            gripper_change_steps: 20
            post_grasp_offset_min: 0.26
            post_grasp_offset_max: 0.28
            lift_th: 0.02
          - name: place
            id: place_apple_0_id9008
            depends_on: [nav_to_place]
            objects: [apple_0_id9008, metal_tray_0_id9016]
            position_constraint: object
            success_mode: xybbox
            filter_x_dir: [forward, 45]
            filter_z_dir: [downward, 150]
            x_ratio_range: [0.35, 0.65]
            y_ratio_range: [0.35, 0.65]
            pre_place_z_offset: 0.1
            place_z_offset: 0.1
            gripper_change_steps: 20
          - name: heuristic__skill
            id: home_left
            depends_on: [place_apple_0_id9008]
            mode: home
            gripper_state: 1.0
```

### 12.1 对象前提

- `apple_0_id9008` 是 `RigidObject`、`role: task_object`、`rigidbody: true`，并配置 `prim_path_child: Aligned`。
- apple USD 同目录的 `Aligned_grasp_sparse.npy` 应可加载；当前样例资产已核验为 `(256, 17) float32`，全部为有限值。
- `metal_tray_0_id9016` 是静态 `GeometryObject`，但已合并进 `task._task_objects`，所以可被 `navigate.approach`
  和 `place.objects[1]` 引用；它不是 pick target。
- robot 使用 `panda_omron_virtual.yaml`，当前 `tcp_offset=0.1043`、`ee_axis=z`、左臂 7 个 arm joints。

### 12.2 没有显式写出但实际生效的默认值

| 节点 | 参数 | 实际值 | 来源与影响 |
| --- | --- | --- | --- |
| 两个 `navigate` | `approach_min_distance` | `0.45 m` | 原 YAML 里的 `0.48` 被注释，不生效。 |
| 两个 `navigate` | `approach_max_distance` | `1.15 m` | 原 YAML 里的 `0.7` 被注释，不生效。 |
| 两个 `navigate` | `approach_sample_count` | `128` | 动态候选数。 |
| `pick` | `pre_grasp_offset` | `0.1 m` | YAML 中的 `0.12` 被注释。 |
| `pick` | `tcp_offset` | `0.1043 m` | 来自 robot config。 |
| `pick` | `t_eps/o_eps` | `1e-3 / 5e-3` | 通用子命令阈值；抓取点还受 `grasp_t_eps/o_eps` 上限约束。 |
| `pick` | `process_valid` | `true` | 启用速度稳定检查。 |
| `place` | `z_ratio_range` | `[0.4,0.6]` | vertical 模式计算了该随机值，但不会用于 vertical 位置。 |
| `place` | `success_xy_margin` | `0.015 m` | `xybbox` 把 tray bbox 每边内缩 1.5 cm。 |
| `heuristic__skill` | `move_steps/t_eps` | `50 / 0.088` | home 关节插值与成功阈值。 |

该样例的 robot 配置为 `use_batch: false`，因此 `target_grasp_z/orientation` 一类 batch 排序参数不会参与
候选选择；`grasp_side_preference: toward_arm` 仍会作为硬筛选生效。

### 12.3 样例参数的实际含义

- `approach_object_armbase_xy: [0.5, 0.0]` 不是“导航半径 0.5 m”，而是希望目标物体最终位于左 arm base
  坐标系前方 0.5 m、侧向 0 m；runtime 用它修正候选底盘 yaw 和排序分数。
- `filter_z_dir: [downward, 140]` 对 pick 表示相关旋转矩阵元素 `<= cos(140°)`，即 local z 与 base -Z
  的夹角不超过约 40°。
- place 的 `[forward,45] + [downward,150]` 约束 local x 朝 base +X 45°以内，同时 local z 朝 base -Z
  30°以内。`filter_y_dir` 有意省略，避免过约束。
- `post_grasp_offset_min/max: 0.26/0.28` 只决定抓取后的 z 抬升范围；`lift_th: 0.02` 只负责成功检查。
- `position_constraint: object` 表示先生成 apple 的目标姿态/位置，再根据当前 apple-to-EE 关系反推 EE pose。
- `x_ratio_range/y_ratio_range` 决定 tray bbox 内的落点区域，不控制放置姿态。

这份样例用于展示标准结构，不等价于“所有参数均已通过本轮 Isaac/Nav2 实跑”。真实成功仍需结合
导航、pick、place 快照和 episode 结果验证。

## 13. 调试产物索引

| Skill | 关键证据 |
| --- | --- |
| `pick` | `pick_plan_snapshot.json`, `pick_execution_trace.json`, `pick_success_check_snapshot.json`, `pick_runtime_failure_snapshot.json` |
| `place` | `place_success_check_snapshot.json` |
| `navigate` | `success_snapshot.json`, `failure_snapshot.json`, `bridge_command_history.json`, dynamic approach candidate report |
| `open/close` | planner keyframe 输出、contact views、articulation joint position/velocity |

严格 episode 成功仍应以 workflow 输出中的 `Task is successful, mode=plan_with_render` 且没有
`[LmdbLogger] Episode failed` 为准，不能只看视频或没有 traceback。

## 14. 源码索引

- 注册表与基类：`workflows/simbox/core/skills/base_skill.py`、`workflows/simbox/core/skills/__init__.py`
- 构造与调度：`workflows/simbox_dual_workflow.py`
- Skill 实现：`workflows/simbox/core/skills/*.py`
- 动态导航目标：`nav2/runtime/dynamic_goal.py`
- Nav2 会话执行：`nav2/runtime/runtime.py`
- Arena / Task 其余字段：`docs/SIMBOX_ARENA_TASK_YAML_API.md`
