# SimBox Skill 用法与参数速查

本文整理 SimBox 当前 skill 的实际用法。读它时可以把 skill 理解成“机器人动作的大函数”：task yaml 只写 `name` 和参数，运行时由 SimBox 找到对应 Python 类，把参数转换成一串底层控制命令，再由 controller 执行。

参考来源：

- 官方概念文档：<https://internrobotics.github.io/InternDataEngine-Docs/concepts/skills>
- 官方 skill overview：<https://internrobotics.github.io/InternDataEngine-Docs/concepts/skills/overview.html>
- 官方 pick/place/articulation 文档：
  <https://internrobotics.github.io/InternDataEngine-Docs/concepts/skills/pick.html>,
  <https://internrobotics.github.io/InternDataEngine-Docs/concepts/skills/place.html>,
  <https://internrobotics.github.io/InternDataEngine-Docs/concepts/skills/articulation.html>
- 自定义 skill 文档：<https://internrobotics.github.io/InternDataEngine-Docs/custom/skill.html>
- 本地实现目录：`workflows/simbox/core/skills/`

最后核对日期：2026-06-21。

## 目录

- [1. 先看全局：skill 在任务里处于哪一层](#1-先看全局skill-在任务里处于哪一层)
- [2. YAML 里 skill 的编排模式](#2-yaml-里-skill-的编排模式)
- [3. 先把几何参数讲透](#3-先把几何参数讲透)
  - [3.1 坐标系和轴：先分清是谁的 x/y/z](#geom-frames)
  - [3.2 bbox ratio：`x_ratio_range` 到底是什么意思](#geom-ratio)
  - [3.3 方向滤波：`filter_*_dir` 到底在滤什么](#geom-filter)
  - [3.4 轴对齐：`align_*_axis` 是什么](#geom-axis)
  - [3.5 offset：沿哪个方向偏移](#geom-offset)
  - [3.6 成功阈值：动作完成不等于任务成功](#geom-success)
  - [3.7 参数值通常怎么确定](#geom-tuning)
- [4. Skill 总览](#4-skill-总览)
- [5. 抓取类](#5-抓取类)
- [6. 放置类](#6-放置类)
- [7. 关节/开合类 articulation skill](#7-关节开合类-articulation-skill)
- [8. 位姿移动和物体调整类](#8-位姿移动和物体调整类)
- [9. 关节、夹爪、等待和回 home](#9-关节夹爪等待和回-home)
- [10. 移动底盘类](#10-移动底盘类)
- [11. 液体/特化成功判定类](#11-液体特化成功判定类)
- [12. 新任务如何选 skill](#12-新任务如何选-skill)
- [13. 新增 skill 需要满足什么接口](#13-新增-skill-需要满足什么接口)
- [14. 查参数的实用方法](#14-查参数的实用方法)
- [15. 常见误区](#15-常见误区)

## 1. 先看全局：skill 在任务里处于哪一层

一个 SimBox task 大致分三层：

1. `arena`/`objects` 决定场景里有什么、物体放在哪里。
2. `skills` 决定机器人按什么顺序做什么动作。
3. 每个 skill 的 Python 实现负责把 YAML 参数变成 `manip_list`，也就是一串底层动作命令。

所以 YAML 里的 skill 不是直接“魔法执行”。它只是声明：

```yaml
- name: pick
  objects: [cup]
  filter_z_dir: ["forward", 80]
  pre_grasp_offset: 0.05
```

运行时流程是：

```text
YAML name
  -> get_skill_cls(name)
  -> 实例化 Python skill 类
  -> simple_generate_manip_cmds()
  -> manip_list
  -> controller.forward(...)
  -> is_done() / is_success()
```

关键文件：

- 注册机制：`workflows/simbox/core/skills/base_skill.py`
- skill 导出和查询：`workflows/simbox/core/skills/__init__.py`
- 各个 skill 实现：`workflows/simbox/core/skills/*.py`

### 1.1 `name` 是怎么对应到 Python 类的

Python 类通过 `@register_skill` 注册。注册名由类名自动转小写并插入下划线：

| YAML `name` | Python 文件 | Python 类 |
|---|---|---|
| `pick` | `pick.py` | `Pick` |
| `place` | `place.py` | `Place` |
| `heuristic__skill` | `heuristic_skill.py` | `Heuristic_Skill` |
| `goto__pose` | `goto_pose.py` | `Goto_Pose` |
| `joint__ctrl` | `joint_ctrl.py` | `Joint_Ctrl` |
| `gripper__action` | `gripper_action.py` | `Gripper_Action` |
| `rotate__obj` | `rotate_obj.py` | `Rotate_Obj` |
| `approach__rotate` | `approach_rotate.py` | `Approach_Rotate` |
| `mobile__translate` | `mobile_translate.py` | `Mobile_Translate` |
| `mobile__rotate` | `mobile_rotate.py` | `Mobile_Rotate` |
| `pour__water__succ` | `pour_water_succ.py` | `Pour_Water_Succ` |

注意：类名里本来有 `_` 时，注册名里经常会出现双下划线，比如 `Goto_Pose -> goto__pose`。写 YAML 时要用注册名，不是文件名。

### 1.2 `manip_list` 是 skill 的真实输出

官方文档里把 skill 输出描述为 manipulation command sequence。本地实现里就是 `manip_list`。常见元素形如：

```python
(p_base_ee_tgt, q_base_ee_tgt, function_name, params)
```

含义：

- `p_base_ee_tgt`：目标末端位置，通常在 robot base frame 下。
- `q_base_ee_tgt`：目标末端姿态 quaternion。
- `function_name`：底层动作，例如 `open_gripper`、`close_gripper`、`attach_obj`、`detach_obj`、`update_specific`、`dummy_forward`。
- `params`：传给 controller 的额外参数，例如碰撞忽略列表、关节动作、约束权重。

因此，一个 `pick` skill 内部其实会拆成：更新碰撞配置、到达 pre-grasp、到达 grasp、夹爪闭合、attach object、抬起等多个小命令。

## 2. YAML 里 skill 的编排模式

task yaml 里的 `skills` 决定单臂、双臂并行、双臂顺序。直观理解：

### 2.1 单臂顺序

一个 arm 下是一串 list，按 list 顺序执行。

```yaml
skills:
  - right:
      - name: pick
        objects: [cup]
      - name: place
        objects: [cup, coaster]
```

含义：右臂先 `pick`，完成后再 `place`。

### 2.2 双臂并行

同一层 step 里同时出现 `left` 和 `right`，两个臂各自执行自己的 skill 序列。它们属于同一个阶段。

```yaml
skills:
  - left:
      - name: pick
        objects: [fork]
    right:
      - name: pick
        objects: [spoon]
```

含义：左臂拿 fork，右臂拿 spoon，两个序列在同一阶段并行推进。

### 2.3 双臂顺序

把多个 step 写成 list 的多个元素。前一个 step 完成后，才进入下一个 step。

```yaml
skills:
  - right:
      - name: pick
        objects: [plate]
      - name: place
        objects: [plate, table]
  - left:
      - name: pick
        objects: [fork]
    right:
      - name: pick
        objects: [spoon]
```

含义：先让右臂摆盘子；盘子摆好后，左臂和右臂再并行摆 fork/spoon。

一句话判断：`skills` 外层 list 的不同元素是“阶段顺序”，同一个元素里的 `left/right` 是“该阶段内并行”。

## 3. 先把几何参数讲透

SimBox skill 的难点不在字段数量，而在“这个数值到底属于哪个坐标系”。如果坐标系没分清，`0.5`、`[1, 0, 0]`、`["downward", 150]` 这些值看起来都像经验玄学；分清以后，它们其实就是几何约束。

这一节先讲所有 skill 共用的几何语言。后面的每个 skill 章节会回到具体字段。

<a id="geom-frames"></a>

### 3.1 坐标系和轴：先分清是谁的 x/y/z

文档里会反复出现四类轴：

| 名称 | 你可以怎么理解 | 常见字段 |
|---|---|---|
| world/task frame | 整个场景的全局坐标。物体 `get_world_pose()` 在这里。 | 场景放置、物体姿态、USD stage |
| robot/base frame | 机器人底座坐标。多数 `manip_list` 里的目标末端位姿在这里。 | `p_base_ee_tgt`, `q_base_ee_tgt` |
| end-effector frame | 夹爪/末端自己的坐标。`filter_x_dir` 里的 `x` 就是 EE 的 x 轴。 | `filter_x_dir`, `post_place_vector` |
| object local frame | 物体自己的局部坐标。`[1,0,0]` 表示物体局部 x 轴，不是世界 x 轴。 | `align_pick_obj_axis`, `align_place_obj_axis`, `obj_axis_offset` |

判断一个字段属于哪个坐标系，先看字段名：

- `filter_x_dir`：`x` 是 EE 的 x 轴，`dir` 是它要朝向 robot/base frame 的哪个方向。
- `align_pick_obj_axis`：这是被拿物体的局部轴。
- `align_place_obj_axis`：这是目标物体的局部轴。
- `obj_axis_offset`：沿物体局部轴偏移。
- `post_place_vector`：源码里用 `T_base_ee_places[index][:3, :3] @ post_vec`，所以它是 EE 局部坐标里的撤退向量。
- `x_ratio_range`：不是某个坐标轴姿态，而是目标物体 world bbox 的 x 范围比例。

一个最实用的心法：

```text
ratio 决定“点在哪里”
filter 决定“夹爪朝哪里”
align 决定“物体的哪个轴要对着目标的哪个轴”
offset 决定“在某个方向上再挪多少米”
success 决定“最后怎么算成功”
```

<a id="geom-ratio"></a>

### 3.2 bbox ratio：`x_ratio_range` 到底是什么意思

`place` 里最常见的几何字段是：

```yaml
x_ratio_range: [0.4, 0.6]
y_ratio_range: [0.4, 0.6]
z_ratio_range: [0.4, 0.6]
```

它们不是世界坐标，也不是物体局部坐标，而是“目标物体 bounding box 上的归一化采样比例”。源码里对 `vertical` 放置的核心计算是：

```text
x = bbox_min.x + x_ratio * (bbox_max.x - bbox_min.x)
y = bbox_min.y + y_ratio * (bbox_max.y - bbox_min.y)
pre_place_z = bbox_max.z + pre_place_z_offset
place_z     = bbox_max.z + place_z_offset
```

也就是说：

| ratio 值 | 几何含义 |
|---|---|
| `0.0` | bbox 最小边界。 |
| `0.5` | bbox 中心。 |
| `1.0` | bbox 最大边界。 |
| `< 0.0` | bbox 外侧，靠近 min 那一边。 |
| `> 1.0` | bbox 外侧，靠近 max 那一边。 |

举个具体例子。假设目标盘子的 bbox x 范围是 `[0.20, 0.40]`：

| `x_ratio` | 算出来的 x | 含义 |
|---|---:|---|
| `0.0` | `0.20` | 盘子左边界。 |
| `0.5` | `0.30` | 盘子中心。 |
| `1.0` | `0.40` | 盘子右边界。 |
| `1.25` | `0.45` | 盘子右外侧。 |
| `-0.25` | `0.15` | 盘子左外侧。 |

因此下面这个配置：

```yaml
x_ratio_range: [1.15, 1.25]
y_ratio_range: [0.30, 0.70]
success_mode: right
```

不是“放到目标内部”，而是“放到目标右侧一条带状区域”。这就是餐具任务里把 spoon 放到 plate 右边的常见写法。

固定点和随机区域也不一样：

```yaml
x_ratio_range: [0.5, 0.5]   # 固定在目标 bbox 中心 x
y_ratio_range: [0.3, 0.7]   # 在目标 bbox 中部一条带内随机
```

`horizontal` 放置也用 ratio，但含义多一步：先在目标 bbox 内算出 `tmp_pos_w`，再沿 `align_place_obj_axis` 和 `offset_place_obj_axis` 偏移，形成 pre-place/place 两个点。挂杯子这类任务常见：

```yaml
place_direction: horizontal
position_constraint: object
x_ratio_range: [0.5, 0.5]
y_ratio_range: [0.5, 0.5]
z_ratio_range: [0.73, 0.73]
align_place_obj_axis: [1, 0, 0]
offset_place_obj_axis: [0, 0, -1]
pre_place_align: -0.15
place_align: -0.07
pre_place_offset: 0.027
place_offset: 0.027
```

读法是：先选中架子 bbox 内一个点，再沿架子的局部 x 轴靠近/插入，同时沿架子的局部 -z 轴做小偏移，让杯子挂到合适位置。

<a id="geom-filter"></a>

### 3.3 方向滤波：`filter_*_dir` 到底在滤什么

`filter_x_dir`、`filter_y_dir`、`filter_z_dir` 过滤的是 EE 末端坐标轴的朝向。这里的 `x/y/z` 是 EE 的轴，不是世界点坐标。

例如：

```yaml
filter_z_dir: ["forward", 50]
```

含义：只保留“EE 的 z 轴大致朝 robot/base frame 的 forward 方向”的候选姿态。

方向词在源码里的大致对应是：

| 方向词 | robot/base frame 方向 | `pick` 支持 | `place/goto__pose` 支持 |
|---|---|---|---|
| `forward` | `+x` | yes | yes |
| `backward` | `-x` | yes | yes |
| `leftward` | `+y` | no | yes |
| `rightward` | `-y` | no | yes |
| `upward` | `+z` | yes | yes |
| `downward` | `-z` | yes | yes |

两种格式：

```yaml
filter_z_dir: ["forward", 80]
filter_x_dir: ["downward", 120, 150]
```

第一种 `["forward", 80]` 是一个锥形约束：EE 某个轴和 `forward` 的夹角不能太偏。对 `forward/upward/leftward` 这种正方向，数值越小越严格。

第二种三元素格式是角度区间。它常用于避免“太正”或“太斜”的姿态，例如：

```yaml
filter_x_dir: ["downward", 120, 150]
```

直觉读法：EE 的 x 轴需要朝向下方，但不是随便朝下，而是在一个角度带内。注意源码实际比较的是 rotation matrix 元素和 `cos(angle)`，所以负方向经常用 `120/150` 这种大于 90 的角度。不要把它简单理解成“离 downward 120 度”，更准确地说是“EE 轴在 base 坐标下的矩阵元素落在对应 cos 区间”。

现有配置里常见模式：

```yaml
# 夹爪某个轴接近向下，常用于从上方拿/放扁平物体
filter_x_dir: ["downward", 150]

# 夹爪 z 轴朝前，常用于侧向伸手或挂放
filter_z_dir: ["forward", 50]

# 只允许一个中等倾角范围，而不是完全竖直
filter_x_dir: ["downward", 120, 150]
```

调参建议：

1. 如果 planner 经常找不到候选姿态，先放宽角度，例如 `50 -> 80` 或减少一个 filter。
2. 如果动作姿态太怪，先加 `filter_*_dir`，再调 offset。
3. 对 `pick`，滤波作用在抓取标注候选上；对 `place/goto__pose`，滤波作用在随机采样出来的目标姿态上。
4. 三元素格式对顺序敏感，最好参考已有 YAML 的写法，例如 positive 方向常见 `[60, 30]`，negative 方向常见 `[120, 150]`。

<a id="geom-axis"></a>

### 3.4 轴对齐：`align_*_axis` 是什么

轴对齐解决的问题是：不仅要把物体放到某个点，还要让物体的某个方向和目标的某个方向对齐。

最典型的是挂杯子：

```yaml
align_pick_obj_axis: [1, 0, 0]
align_place_obj_axis: [1, 0, 0]
align_obj_tol: 10
```

读法：

- `align_pick_obj_axis: [1, 0, 0]`：杯子局部 x 轴。
- `align_place_obj_axis: [1, 0, 0]`：架子局部 x 轴。
- `align_obj_tol: 10`：两根轴最终夹角要小于 10 度。

源码里不是直接比较 `[1,0,0]` 和 `[1,0,0]`。它会先把“物体局部轴”乘上物体姿态矩阵，转到 base/world 方向，再比较夹角：

```text
物体局部轴 -> 经过物体当前/目标姿态旋转 -> base/world 方向
目标局部轴 -> 经过目标物体姿态旋转 -> base/world 方向
比较二者夹角 < align_obj_tol
```

### 3.4.1 如何确定一个物体的轴

最可靠的方式有三种：

1. 看 USD/Isaac 里的 local transform gizmo：红/绿/蓝通常对应 local x/y/z。
2. 看物体 bbox 和模型形状：长条物体的最长方向常常是某个 local axis，但这只是经验，要用可视化确认。
3. 在已有成功任务里反推：例如同一个杯架资产已经用 `align_place_obj_axis: [1,0,0]`，说明这个资产里 x 轴就是挂放方向。

注意两个坑：

- task yaml 里的 `euler` 会旋转物体，所以“USD 原始轴”和“仿真里当前轴”不一定一致。skill 里用的是运行时物体姿态下的 local axis。
- `[1,0,0]` 只表示“局部 x 正方向”，如果实际需要反方向，要用 `[-1,0,0]`。

简单判断模板：

```text
要让杯柄朝向架子钩子：
  找杯子上“杯柄方向”对应 local axis -> align_pick_obj_axis
  找架子上“钩子伸出方向”对应 local axis -> align_place_obj_axis
  允许误差 -> align_obj_tol
```

<a id="geom-offset"></a>

### 3.5 offset：沿哪个方向偏移

offset 的单位通常是米，但方向由字段决定：

| 字段 | 方向所在坐标系 | 直觉 |
|---|---|---|
| `pre_grasp_offset` | EE 接近轴，`r5a` 用 EE x，其他常用 EE z | 抓之前退一点，形成 pre-grasp。 |
| `post_grasp_offset_min/max` | 多数实现直接加 base/world z | 抓住后向上抬起。 |
| `pre_place_z_offset` | world/base z | 放之前在目标上方停一下。 |
| `place_z_offset` | world/base z | 真正释放高度。 |
| `pre_place_align/place_align` | `align_place_obj_axis` 转到 world 后的方向 | 水平放置时沿目标轴靠近/插入。 |
| `pre_place_offset/place_offset` | `offset_place_obj_axis` 转到 world 后的方向 | 水平放置时做侧向/上下偏移。 |
| `post_place_vector` | EE 局部坐标 | 松开后沿夹爪自身坐标撤退。 |
| `obj_axis_offset` | object local frame | 先把物体参考点沿局部轴挪一下。 |
| `trans_offset` / `delta_trans` | 通常是 base/world 线性偏移 | 对算出的目标点做最后修正。 |

所以不要只看数字大小，还要看“沿谁的轴”。同样的 `0.05`，沿 world z 是抬高 5cm，沿 EE z 可能是沿夹爪方向后退 5cm。

<a id="geom-success"></a>

### 3.6 成功阈值：动作完成不等于任务成功

每个 skill 都有两类判断：

1. `is_done()`：`manip_list` 里的动作命令执行完了没有。
2. `is_success()`：物体/关节/粒子状态是否真的满足任务目标。

这两个不能混用。比如 `place` 已经打开夹爪并完成撤退，只能说明动作结束；杯子是否真的在杯垫上，还要看 `success_mode`。再比如 `fail_pick` 的 `is_success()` 源码里基本返回 True，但它的语义是“失败动作执行完”，不是抓取成功。

常见成功阈值：

| 字段 | 常见单位/语义 |
|---|---|
| `lift_th` | 米，物体抬升高度。 |
| `success_threshold` | 可能是米，也可能是 articulation joint displacement。看 skill。 |
| `success_threshold_move` | 米，末端位置误差。 |
| `success_threshold_rotate` | 弧度，姿态误差。 |
| `success_th` | 常用于 IoU 或 task-specific 阈值。 |
| `threshold` | 常用于 left/right 等相对位置判定，单位多为米。 |

<a id="geom-tuning"></a>

### 3.7 参数值通常怎么确定

参数不是凭空生成的。一般来自四类依据：

1. 物体资产标注：例如 `Aligned_grasp_sparse.npy`、`dexpick_pose.yaml`、`place_range.yaml`、articulation contact/keypoint 配置。
2. 几何关系：目标物体 bbox、物体局部轴、末端轴、目标表面高度。
3. 机器人约束：可达空间、夹爪开合、碰撞、IK/planner 是否可解。
4. 仿真调试：看失败样本后微调 offset、ratio、filter、success threshold。

一个新任务的最小调参顺序建议：

```text
1. 确认 objects 名字和资产路径
2. 确认 pick 标注能用
3. 用最宽松的 pick/place 跑通
4. 收紧 filter，让姿态合理
5. 调 ratio/offset，让落点合理
6. 调 success_mode/threshold，让成功判定合理
```

不同 skill 参数很多，但背后只有几类问题：选哪个物体、从哪个方向接近、放到哪里、夹爪怎么开合、怎样判定成功。

| 参数 | 常见位置 | 含义 |
|---|---|---|
| `objects` | 几乎所有物体相关 skill | 参与动作的物体名。`pick` 通常是 `[被抓物体]`，`place` 通常是 `[被放物体, 目标物体]`。 |
| `ignore_substring` | pick/place/open/close 等 | 规划时额外忽略碰撞的 prim 名称片段。用于避免把目标物体或支架误判成不可碰撞障碍。 |
| `filter_x_dir` / `filter_y_dir` / `filter_z_dir` | pick/place/goto_pose | 对候选末端姿态做方向过滤。见 [3.3 方向滤波](#geom-filter)。 |
| `pre_grasp_offset` | pick 系 | 抓取前沿夹爪接近轴后退多少米，形成 pre-grasp 点。 |
| `post_grasp_offset_min/max` | pick 系 | 抓住后向上或沿指定方向抬起的距离范围。 |
| `pre_place_z_offset` | place | 放置前在目标点上方多少米。 |
| `place_z_offset` | place | 真正放置点相对目标表面的高度偏移。 |
| `x_ratio_range` / `y_ratio_range` / `z_ratio_range` | place | 在目标物体 bbox 上采样位置的归一化比例。见 [3.2 bbox ratio](#geom-ratio)。 |
| `align_pick_obj_axis` / `align_place_obj_axis` / `align_ref_axis` | place/goto_pose | 物体局部轴对齐约束。见 [3.4 轴对齐](#geom-axis)。 |
| `gripper_change_steps` | pick/place/gripper | 夹爪开合动作重复多少步。步数太少可能没夹紧或没松开。 |
| `gripper_state` | gripper/home/heuristic/wait 等 | 一般 `1.0` 表示开，`-1.0` 表示关。 |
| `t_eps` / `o_eps` | 多数位姿型 skill | 判断“到达当前 waypoint”的位置/姿态容差。 |
| `success_mode` | place/open/close | 成功判定方式。不同 skill 含义不同，不能跨 skill 乱套。 |
| `success_threshold` / `threshold` / `success_th` | 多数成功判定 | 成功阈值，单位可能是米、弧度、IoU 或 joint displacement，必须看对应 skill。 |
| `test_mode` | pick/place/goto_pose | 候选位姿测试方式，常见为 `forward` 或 `ik`。 |
| `process_valid` | pick/open/close 等 | 是否检查过程中速度、碰撞等稳定性。 |

## 4. Skill 总览

下面按用途分类。`必填` 是当前源码里直接索引 `cfg[...]` 或实际运行必需的字段；`常用参数` 是最值得调的字段；`成功判定/注意` 是调 task 时最容易踩坑的地方。

## 5. 抓取类

### 5.1 `pick`

用途：最常用的标准抓取。它从物体旁边的 `Aligned_grasp_sparse.npy` 读取候选抓取姿态，按方向过滤后选一个可达姿态执行抓取。

最小配置：

```yaml
- name: pick
  objects: [cup]
```

常用参数：

| 参数 | 含义 |
|---|---|
| `objects: [obj]` | 被抓物体。 |
| `npy_name` | 抓取候选文件名，默认 `Aligned_grasp_sparse.npy`。 |
| `grasp_scale` | 对 grasp pose 的尺度修正。 |
| `tcp_offset` | TCP 到末端的偏移，默认用 robot 的 `tcp_offset`。 |
| `constraints` | 传给 grasp pose 采样/过滤的额外约束。 |
| `filter_x_dir/y_dir/z_dir` | 过滤夹爪姿态。 |
| `direction_to_obj` | 要求某个方向朝向物体。 |
| `pre_grasp_offset` | 抓取前后退距离，默认约 `0.1`。 |
| `pre_grasp_hold_vec_weight` | pre-grasp 规划时保持某些方向的权重。 |
| `gripper_change_steps` | 夹爪闭合步数，默认偏大以保证夹紧。 |
| `final_gripper_state` | 默认 `-1` 关爪；设成 `1` 可做打开式动作。 |
| `post_grasp_offset_min/max` | 抓取后抬升距离范围。 |
| `return_to_pregrasp` | 抓取后是否回到 pre-grasp。 |
| `fixed_orientation` | 强制使用固定末端姿态。 |
| `lift_th` | 要求物体被抬起超过该高度才算成功。 |
| `process_valid` | 是否检查速度等过程稳定性。 |

成功判定：通常要求夹爪和物体有接触；若设置 `lift_th`，还要求物体相对初始高度被抬起足够多。

注意：没有抓取标注文件时，`pick` 很难正常工作。新资产完成“标注”时，最要紧的就是确认抓取 npy 是否存在、路径是否能被 task 加载。

参数怎么读：

- `pick` 的候选姿态来自物体资产旁边的 grasp npy，不是 YAML 现场生成的。YAML 主要是在这些候选里筛选、偏移、测试可达性。
- `filter_*_dir` 对候选 EE 姿态生效。比如 `filter_z_dir: ["forward", 50]` 是从所有 grasp pose 里挑出 EE z 轴朝前的候选。
- `pre_grasp_offset` 决定“先停在抓取点外多远”。太小容易直接撞物体，太大可能 planner 绕不过去。
- `post_grasp_offset_min/max` 通常是抓住后向上抬起的高度。窄范围如 `[0.10, 0.10]` 是固定抬 10cm；宽范围会引入数据多样性。
- `direction_to_obj: right/left` 是额外位置过滤，用来指定夹爪从物体的哪一侧接近。

例子：

```yaml
- name: pick
  objects: [oo3d_object1]
  pre_grasp_offset: 0.05
  filter_y_dir: ["upward", 60]
  filter_z_dir: ["forward", 50]
  pre_grasp_hold_vec_weight: [1, 1, 1, 0, 0, 0]
  post_grasp_offset_min: 0.10
  post_grasp_offset_max: 0.10
```

这段可以读成：抓 `oo3d_object1`，只接受“EE y 轴比较朝上、EE z 轴比较朝前”的抓取姿态；抓取前退 5cm，抓住后抬 10cm。

调参顺序：

1. 先不加或少加 filter，确认 grasp npy 能产生可达抓取。
2. 如果姿态太怪，再逐个加 `filter_y_dir`、`filter_z_dir`。
3. 如果碰撞，先调 `pre_grasp_offset`；如果抓起后刮蹭，调 `post_grasp_offset_min/max`。
4. 如果抓住但成功判定失败，检查 contact view 和 `lift_th`，不要只调抓取姿态。

### 5.2 `manualpick`

用途：在标准抓取候选基础上做手工姿态/位置修正。适合候选 grasp 大体可用，但需要整体旋转或平移一点才能稳定抓取的物体。

最小配置：

```yaml
- name: manualpick
  objects: [obj]
```

常用参数：

| 参数 | 含义 |
|---|---|
| `adjust_ori` | 是否启用姿态调整。 |
| `adjust_rotate_axis` | 调整绕哪个轴，常见 `x/y/z`。 |
| `adjust_angle_list_cfg` | 角度搜索范围，例如 `[min, max, num]`。 |
| `manual_adjust_ori` | 直接给定姿态调整。 |
| `adjust_trans_offset` | 对抓取点做平移修正。 |
| `pre_grasp_offset_manual` | 单独修正 pre-grasp 距离。 |
| 其他 | 基本继承 `pick` 的过滤、offset、夹爪参数。 |

成功判定：与 `pick` 类似，看接触和过程有效性。

注意：它是“人工补偿版 pick”，不是新抓取算法。参数要结合仿真观察调。

什么时候用：

- 标准 `pick` 能找到候选，但末端总是有固定角度偏差。
- 某个物体需要绕固定轴多转几度才能夹稳。
- 你不想重新生成抓取标注，只想在 YAML 层做小范围补偿。

例子：

```yaml
- name: manualpick
  objects: [tool]
  adjust_ori: true
  adjust_rotate_axis: z
  adjust_angle_list_cfg: [-20, 20, 9]
  adjust_trans_offset: [0.0, 0.0, 0.01]
```

读法：在原始 grasp pose 周围，绕 z 轴搜索 `-20` 到 `20` 度的姿态，并把抓取点上移 1cm。

### 5.3 `dexpick`

用途：使用 `dexpick_pose.yaml` 中的确定性抓取 pose，而不是从 `Aligned_grasp_sparse.npy` 采样。

最小配置：

```yaml
- name: dexpick
  objects: [obj]
  pick_pose_idx: 0
```

常用参数：

| 参数 | 含义 |
|---|---|
| `pick_pose_idx` | 使用 `dexpick_pose.yaml` 里的第几个抓取 pose。 |
| `pre_grasp_offset` | 抓取前后退距离。 |
| `gripper_change_steps` | 闭合夹爪步数。 |
| `post_grasp_offset_min/max` | 抓取后抬升范围。 |
| `lift_th` | 抬升成功阈值。 |
| `process_valid` | 是否检查过程稳定性。 |

成功判定：夹爪和物体接触，并通过过程有效性/抬升检查。

注意：它依赖物体资产旁边存在 `dexpick_pose.yaml`。如果没有这个文件，不能把它当普通 `pick` 用。

和 `pick` 的区别：

- `pick` 是“从一堆 grasp npy 候选里筛选”。
- `dexpick` 是“直接用 `dexpick_pose.yaml` 中的指定姿态”。

所以 `dexpick` 更稳定、更可控，但前提是标注文件质量足够好。调参重点通常不是 filter，而是 `pick_pose_idx`、`pre_grasp_offset`、`post_grasp_offset_min/max`。

### 5.4 `dynamicpick`

用途：抓取运动中的物体，例如 conveyor 场景。它会结合物体速度、预测时间和 `pick_range` 计算抓取点。

最小配置：

```yaml
- name: dynamicpick
  objects: [obj]
  pick_range: [0.0, 0.2]
```

常用参数：

| 参数 | 含义 |
|---|---|
| `pick_range` | 沿运动方向随机选择抓取偏移范围。 |
| `time_bias` | 对预测抓取时间做偏置。 |
| `pick_bias` | 对抓取位置做额外偏置。 |
| `pivot_angle_z` | 对抓取姿态绕 z 做随机旋转。 |
| `pos_adjust_z` | 对抓取位置 z 做随机修正。 |
| 其他 | 基本继承 `pick` 的 grasp、filter、offset、夹爪参数。 |

成功判定：类似 `pick`，但场景还需要提供 conveyor velocity 等动态信息。

注意：这是动态任务专用 skill。普通静态桌面抓取优先用 `pick`。

参数怎么读：

- `pick_range` 决定沿动态物体运动方向选择哪个截获点。
- `time_bias` 是对预测到达时间的修正，常用于补偿 planner/执行延迟。
- `pick_bias` 是对预测位置的修正。
- `pivot_angle_z` 和 `pos_adjust_z` 是给抓取姿态/高度加随机或经验补偿。

调参直觉：如果机器人总是抓晚了，调时间；如果总是抓偏了，调位置；如果总是姿态不稳，再调姿态扰动。

### 5.5 `fail_pick`

用途：故意偏离正确抓取点，生成失败或负样本行为。

最小配置：

```yaml
- name: fail_pick
  objects: [obj]
```

常用参数：

| 参数 | 含义 |
|---|---|
| `grasp_x_offset_min/max` | 沿 x 偏离正确抓取点的范围。 |
| `grasp_y_offset_min/max` | 沿 y 偏离正确抓取点的范围。 |
| `filter_x_dir/y_dir/z_dir` | 姿态过滤。 |
| `gripper_change_steps` | 夹爪闭合步数。 |
| `post_grasp_offset_min/max` | 假抓后抬升距离。 |

成功判定/注意：源码里的 `is_success()` 基本返回 True，它的语义不是“任务成功”，而是“这段失败动作执行完”。不要把它用于正常成功数据生成。

使用边界：

- 适合做 failure demo、负样本、恢复策略研究。
- 不适合做普通 task 的 fallback。因为它不会告诉你“真实抓取成功”，只会按失败动作执行完。

## 6. 放置类

### 6.1 `place`

用途：最常用的标准放置。它根据目标物体 bbox 采样放置点，再结合当前抓住的物体和夹爪姿态生成放置轨迹。

最小配置：

```yaml
- name: place
  objects: [cup, coaster]
```

常用参数：

| 参数 | 含义 |
|---|---|
| `objects: [held, target]` | 第一个是被放物体，第二个是目标物体。 |
| `place_part_prim_path` | 如果只想放到目标物体的某个子 prim 上，用这个字段指定。 |
| `place_direction` | `vertical` 表示从上往下放；`horizontal` 表示水平插入/挂放。默认 `vertical`。 |
| `position_constraint` | `gripper` 或 `object`。决定采样点约束的是夹爪还是被拿物体。 |
| `x_ratio_range/y_ratio_range/z_ratio_range` | 在目标 bbox 上采样放置点。可超出 `[0,1]` 表示目标旁边。 |
| `pre_place_z_offset` | 到目标点上方的预放置高度。 |
| `place_z_offset` | 真正释放高度。 |
| `filter_x_dir/y_dir/z_dir` | 过滤末端姿态。 |
| `align_pick_obj_axis` | 被放物体的哪个轴参与对齐。 |
| `align_place_obj_axis` | 目标物体的哪个轴参与对齐。 |
| `align_obj_tol` | 对齐容忍度。 |
| `align_plane_x_axis/y_axis` | 平面对齐辅助轴。 |
| `pre_place_hold_vec_weight` | pre-place 阶段姿态/方向保持权重。 |
| `post_place_hold_vec_weight` | post-place 阶段姿态/方向保持权重。 |
| `hesitate_steps` | 到达释放点后停顿多少步。 |
| `gripper_change_steps` | 打开夹爪步数。 |
| `post_place_vector` | 释放后的撤退方向，通常在末端局部坐标系里表达。 |
| `success_mode` | 成功判定方式，见下表。 |
| `success_th` / `threshold` | 成功阈值，取决于 `success_mode`。 |

`place_direction: horizontal` 还常用：

| 参数 | 含义 |
|---|---|
| `align_place_obj_axis` | 水平放置时沿哪个目标轴对齐。 |
| `offset_place_obj_axis` | 沿哪个目标轴偏移。 |
| `pre_place_align/pre_place_offset` | 预放置阶段的对齐/偏移距离。 |
| `place_align/place_offset` | 真正放置阶段的对齐/偏移距离。 |

常见 `success_mode`：

| 模式 | 大意 |
|---|---|
| `3diou` | 被放物体和目标区域三维 IoU 足够。默认模式。 |
| `height` | 高度满足要求。 |
| `xybbox` | x-y 平面落在目标 bbox 内。 |
| `left` / `right` | 被放物体在目标物体左/右侧。常用于餐具摆放。 |
| `flower` | 花瓶/插花类任务特化判定。 |
| `cup` | 杯子/架子类任务特化判定。 |

注意：

- `place` 参数最容易和场景几何绑定。新任务里通常先调 `x/y_ratio_range`、`place_z_offset`、`filter_*_dir`、`success_mode`。
- `objects` 顺序不能反：`[cup, coaster]` 是把 cup 放到 coaster，不是反过来。

### 6.1.1 `place` 的执行逻辑拆开看

`place` 实际上在做四件事：

1. 找目标区域：对 `objects[1]` 或 `place_part_prim_path` 对应 prim 计算 bbox。
2. 采样放置点：用 `x/y/z_ratio_range` 在 bbox 上取点。
3. 采样/过滤放置姿态：用 `filter_*_dir` 和 `align_*_axis` 约束 EE/物体方向。
4. 执行释放和撤退：到 pre-place、到 place、打开夹爪、detach object、可选 `post_place_vector` 撤退。

所以 `place` 参数可以按功能分组看：

| 功能 | 主要字段 |
|---|---|
| 目标点在哪里 | `x_ratio_range`, `y_ratio_range`, `z_ratio_range`, `pre_place_z_offset`, `place_z_offset` |
| 目标点绑定谁 | `position_constraint` |
| 姿态朝哪里 | `filter_x_dir/y_dir/z_dir` |
| 物体轴怎么对齐 | `align_pick_obj_axis`, `align_place_obj_axis`, `align_obj_tol` |
| 水平插入/挂放 | `place_direction: horizontal`, `align_place_obj_axis`, `offset_place_obj_axis`, `pre_place_align`, `place_align`, `pre_place_offset`, `place_offset` |
| 放完怎么退 | `post_place_vector`, `post_place_hold_vec_weight` |
| 怎么算成功 | `success_mode`, `success_th`, `threshold` |

### 6.1.2 `position_constraint`: `gripper` 和 `object` 的区别

这是 `place` 里很关键、也很容易忽略的字段。

```yaml
position_constraint: gripper
```

表示 `x/y/z_ratio_range` 采样出来的点是“夹爪 EE 要到达的位置”。这适合普通从上往下放：只要夹爪到某个位置，物体自然跟着过去。

```yaml
position_constraint: object
```

表示采样出来的点是“被放物体目标位置”。源码会用当前 `T_obj_ee` 关系反推出 EE 应该去哪里。这适合挂杯子、插物体、对齐物体中心等场景，因为真正关心的是杯子/工具的位置，不是夹爪的位置。

选择规则：

- 桌面放置、盘子上叠东西：通常 `gripper` 就够。
- 需要物体的某个位置精确对上目标：优先 `object`。
- 发现夹爪到了点但物体没对上：考虑切换到 `object`。

### 6.1.3 `vertical` 放置例子：杯子放杯垫

对 `livingroom_mug_to_coaster` 这种任务，可以从最小链条开始：

```yaml
- name: place
  objects: [livingroom_mug_0_id9005, round_coaster_a_0_id9006]
  place_direction: vertical
  position_constraint: object
  x_ratio_range: [0.5, 0.5]
  y_ratio_range: [0.5, 0.5]
  pre_place_z_offset: 0.15
  place_z_offset: 0.04
  filter_x_dir: ["downward", 140]
  success_mode: xybbox
```

读法：

- 目标是 coaster 的 bbox 中心。
- `position_constraint: object` 表示希望 mug 的参考位置对准 coaster 中心。
- `pre_place_z_offset: 0.15` 先到杯垫上方 15cm。
- `place_z_offset: 0.04` 释放时让 mug 底部/参考点略高于 coaster 顶部。
- `filter_x_dir: ["downward", 140]` 让夹爪姿态有“从上往下放”的倾向。
- `success_mode: xybbox` 只看 mug 的 x-y 是否落在 coaster bbox 内，适合先跑通。

真实任务里 `place_z_offset` 要根据物体参考点和 bbox 位置微调。杯子如果悬空，就降一点；如果插进杯垫/桌面，就升一点。

### 6.1.4 `horizontal` 放置例子：杯子挂架子

来自 `hang_the_cup_on_rack_part0.yaml` 的核心配置：

```yaml
- name: place
  objects: [oo3d_object1, oo3d_object2]
  place_direction: horizontal
  filter_y_dir: ["upward", 30]
  align_pick_obj_axis: [1, 0, 0]
  align_place_obj_axis: [1, 0, 0]
  offset_place_obj_axis: [0, 0, -1]
  align_obj_tol: 10
  pre_place_align: -0.15
  place_align: -0.07
  pre_place_offset: 0.027
  place_offset: 0.027
  position_constraint: object
  x_ratio_range: [0.5, 0.5]
  y_ratio_range: [0.5, 0.5]
  z_ratio_range: [0.73, 0.73]
  success_mode: cup
```

这段可以拆成：

- `x/y/z_ratio_range`：先选架子 bbox 里的一个挂放参考点。
- `align_pick_obj_axis` 和 `align_place_obj_axis`：让杯子的局部 x 轴和架子的局部 x 轴对齐，误差小于 10 度。
- `pre_place_align/place_align`：沿架子 x 轴先在外侧预对齐，再更靠近放置点。
- `offset_place_obj_axis: [0,0,-1]` 加上 `place_offset`：沿架子局部 -z 方向做 2.7cm 偏移。
- `position_constraint: object`：保证上面这些点是杯子目标点，而不是夹爪目标点。

这就是为什么挂放任务参数看起来比普通 place 多：它不是只解决“放在哪里”，还要解决“以哪个姿态、沿哪个轴插过去”。

### 6.1.5 `place` 调参顺序

推荐顺序：

1. 先固定位置：把 `x/y/z_ratio_range` 都设成 `[0.5, 0.5]`，减少随机性。
2. 先跑通高度：调 `pre_place_z_offset` 和 `place_z_offset`，让物体不撞、不悬空。
3. 再调姿态：加 `filter_*_dir`，让夹爪方向合理。
4. 最后调轴对齐：加 `align_*_axis` 和 `align_obj_tol`，解决挂放/插入/朝向问题。
5. 成功判定单独调：先用宽松 `xybbox` 或低阈值，再换成更严格的 `3diou/cup/flower`。

常见失败和对应参数：

| 现象 | 优先检查 |
|---|---|
| 物体落点偏左/偏右 | `x_ratio_range`, `y_ratio_range` |
| 物体悬空 | `place_z_offset` 太大，或物体参考点不是底部 |
| 物体压进桌面/目标 | `place_z_offset` 太小 |
| 夹爪从奇怪方向放 | `filter_*_dir` |
| 挂不上/插不进去 | `align_*_axis`, `pre_place_align/place_align`, `offset_place_obj_axis` |
| 放完撞到目标 | `post_place_vector`, `ignore_substring` |
| 动作看起来成功但判失败 | `success_mode`, `success_th`, `threshold` |

### 6.2 `dexplace`

用途：Dex 风格放置。它根据目标 bbox 和可选的 `place_range.yaml` 生成放置点，末端方向更多由目标容器方向和相机/夹爪轴约束决定。

最小配置：

```yaml
- name: dexplace
  objects: [obj, container]
```

常用参数：

| 参数 | 含义 |
|---|---|
| `objects: [held, target]` | 被放物体和目标物体。 |
| `place_part_prim_path` | 目标子 prim。 |
| `gripper_axis` | 夹爪轴设置。 |
| `camera_axis_filter` | 对相机/末端观察轴做过滤。 |
| `gripper_change_steps` | 松开夹爪步数。 |

成功判定：被放物体当前位置落入目标 bbox 边界内。

注意：如果被放物体资产旁边有 `place_range.yaml`，会读取其中 `x_range/y_range`；否则默认 `[0.4, 0.6]`。

和 `place` 的区别：

- `place` 的位置范围主要从 task YAML 的 `x/y/z_ratio_range` 来。
- `dexplace` 会优先读资产旁边的 `place_range.yaml`，更像“资产自带可放置区域”。
- `camera_axis_filter` 会参与生成 EE 姿态，适合希望相机/夹爪观察方向也满足一定约束的场景。

如果你只是做普通“杯子放杯垫”，先用 `place` 更直接；如果资产已经有 dex place 标注，再考虑 `dexplace`。

## 7. 关节/开合类 articulation skill

这类 skill 用于抽屉、柜门、旋钮等 articulated object。它们一般依赖 `planner_setting` 和 articulation 标注，不是只靠 bbox 就能跑。

### 7.1 `open`

用途：打开 articulated object，例如拉开抽屉、打开门。

最小配置：

```yaml
- name: open
  objects: [drawer]
  planner_setting:
    contact_pose_index: 0
    success_threshold: 0.1
    constraint_list: [...]
```

常用参数：

| 参数 | 含义 |
|---|---|
| `objects: [art_obj]` | 被操作的关节物体。 |
| `obj_info_path` | 额外 articulation 信息文件。 |
| `planner_setting.contact_pose_index` | 使用哪个接触/keypoint pose。 |
| `planner_setting.success_threshold` | 打开幅度成功阈值。 |
| `planner_setting.success_mode` | 默认 `abs`，按关节位移绝对值判断。 |
| `planner_setting.update_art_joint` | 是否更新 articulation joint 信息。 |
| `planner_setting.constraint_list` | KPAM 约束列表。 |
| `collision_valid` | 是否检查碰撞有效性。 |
| `process_valid` | 是否检查过程速度稳定性。 |

成功判定：关节位移达到阈值，并通过碰撞/速度检查。

参数怎么读：

- `contact_pose_index` 不是随便编号，它指向 articulation 标注里的某个接触姿态/关键点。换把手、换抽屉、换门板，index 可能就要变。
- `constraint_list` 是 KPAM planner 的核心约束，决定末端怎样贴住把手、沿哪个方向拉/推。
- `success_threshold` 的单位不是米，而是关节位移。对 prismatic joint 更像线性位移，对 revolute joint 更像角位移。
- `ignore_substring` 常用于忽略被操作物体自身或附近静态结构，否则 planner 可能因为接触目标而判碰撞。

调参顺序：

1. 先确认 articulation 物体的 joint 信息和 contact/keypoint 标注存在。
2. 再选 `contact_pose_index`，保证机器人手能到达把手/接触点。
3. 再调 `success_threshold`，避免“轻微动了一下就成功”或“已经打开但判失败”。
4. 最后处理碰撞和过程稳定性：`collision_valid`、`process_valid`、`ignore_substring`。

### 7.2 `close`

用途：关闭 articulated object，例如推回抽屉、关门。

最小配置类似 `open`：

```yaml
- name: close
  objects: [drawer]
  planner_setting:
    contact_pose_index: 0
    success_threshold: 0.05
    constraint_list: [...]
```

常用参数：

| 参数 | 含义 |
|---|---|
| `planner_setting.success_mode` | 默认 `zero`，表示关节接近 0；也支持按与初始距离变化判断。 |
| 其他 | 与 `open` 基本一致。 |

成功判定：关节接近关闭状态或满足配置阈值，并通过碰撞/过程检查。

和 `open` 的主要区别在成功判定：

- `open` 默认看关节是否离初始/关闭状态足够远。
- `close` 默认 `success_mode: zero`，更像“关节回到接近 0 的关闭状态”。

如果抽屉初始状态不是标准 0，或者关节定义方向和预期相反，优先检查 `success_mode` 和 `success_threshold`，不要先改运动参数。

### 7.3 `rotate`

用途：旋转 articulated object，例如旋钮、可转动部件。

最小配置：

```yaml
- name: rotate
  objects: [knob]
  planner_setting:
    contact_pose_index: 0
    success_threshold: 0.785
```

常用参数：

| 参数 | 含义 |
|---|---|
| `objects: [art_obj]` | 被旋转关节物体。 |
| `planner_setting.contact_pose_index` | 接触 pose index。 |
| `planner_setting.success_threshold` | 旋转成功阈值，默认约 `0.785` rad。 |
| `planner_setting.additional_labels` | 额外 keypoint/label 配置。 |
| `obj_info_path` | articulation 信息文件。 |

成功判定：关节旋转量达到阈值。

参数怎么读：

- `success_threshold: 0.785` 大约是 45 度，适合旋钮类任务。
- `additional_labels` 通常和具体 articulation/keypoint 标注有关，不同资产之间不要盲目复用。
- `rotate` 操作的是 articulated joint，不是已经抓住的普通刚体。普通刚体旋转应看 [`rotate__obj`](#83-rotate__obj)。

### 7.4 `artpreplan`

用途：articulation 动作前的预规划/预处理。它主要让 KPAM planner 根据当前 articulation 状态生成接触/约束信息，为后续 `open/close/rotate` 铺路。

最小配置：

```yaml
- name: artpreplan
  objects: [drawer]
  planner_setting:
    contact_pose_index: 0
    success_threshold: 0.1
```

常用参数：

| 参数 | 含义 |
|---|---|
| `objects: [art_obj]` | 目标关节物体。 |
| `planner_setting.contact_pose_index` | 接触 pose index。 |
| `planner_setting.success_threshold` | 后续规划的成功阈值参考。 |
| `planner_setting.update_art_joint` | 是否更新关节信息。 |
| `obj_info_path` | articulation 信息文件。 |

注意：它不是通用移动 skill，而是 articulation pipeline 的辅助 skill。

什么时候加：

- `open/close/rotate` 直接跑不稳，需要先更新 articulation 状态或约束。
- 同一个任务里要连续操作 articulated object，希望后续 skill 使用更可靠的接触/关节信息。

什么时候不加：

- 普通 rigid object 的 pick/place。
- 没有 articulation 标注的物体。

## 8. 位姿移动和物体调整类

### 8.1 `goto__pose`

用途：把末端移动到指定位置/姿态。适合做过渡动作、预定位、到达固定观察位。

最小配置有两种。

直接给姿态：

```yaml
- name: goto__pose
  position: [0.4, 0.0, 0.3]
  euler: [0, 0, 0]
```

或给位置，再通过物体轴对齐采样姿态：

```yaml
- name: goto__pose
  position: [0.4, 0.0, 0.3]
  objects: [box]
  align_obj_axis: [1, 0, 0]
  align_ref_axis: [0, 0, 1]
  align_obj_tol: 0.2
```

常用参数：

| 参数 | 含义 |
|---|---|
| `position` | 目标末端位置。 |
| `quaternion` / `euler` | 目标末端姿态。二选一即可。 |
| `frame` | 默认 `robot`。 |
| `gripper_state` | 到达时夹爪状态。 |
| `position_constraint` | `gripper` 或 `object`。 |
| `filter_x_dir/y_dir/z_dir` | 姿态过滤。 |
| `max_noise_m` / `max_noise_deg` | 位置/姿态随机扰动。 |
| `interp_nums` | 插值 waypoint 数。 |

注意：如果不直接给姿态，就必须提供对象轴对齐相关参数，否则无法生成姿态候选。

两种用法的区别：

1. 明确目标姿态：你已经知道 EE 应该以什么姿态到达，用 `euler` 或 `quaternion`。
2. 只知道几何约束：你不知道具体姿态，但知道“物体某个轴应该朝上/朝前”，就用 `objects + align_obj_axis + align_ref_axis + align_obj_tol` 让代码采样可行姿态。

例子：让被拿物体的局部 z 轴尽量朝 robot/base 的上方：

```yaml
- name: goto__pose
  position: [0.4, 0.0, 0.35]
  objects: [held_obj]
  position_constraint: object
  align_obj_axis: [0, 0, 1]
  align_ref_axis: [0, 0, 1]
  align_obj_tol: 15
  filter_x_dir: ["forward", 70]
```

读法：

- `position_constraint: object`：目标点是物体要到的位置。
- `align_obj_axis: [0,0,1]`：物体局部 z 轴。
- `align_ref_axis: [0,0,1]`：base frame 的上方。
- `align_obj_tol: 15`：两者夹角小于 15 度。
- `filter_x_dir` 再约束 EE x 轴，避免手腕姿态太离谱。

调参建议：如果只是移动到固定过渡位，直接给 `euler/quaternion` 更可控；如果要保持物体朝向，才用轴对齐采样。

### 8.2 `move`

用途：拿着某个物体向另一个目标物体移动，常见于擦拭、推动、靠近目标等任务。

最小配置：

```yaml
- name: move
  objects: [sponge, stain]
  success_threshold: 0.02
```

常用参数：

| 参数 | 含义 |
|---|---|
| `objects: [move_obj, target_obj]` | 当前操作物体和目标物体。 |
| `invisible_object` | 用不可见辅助物体作为目标参考。 |
| `delta_trans` | 对目标位置额外平移。 |
| `hold_vec_weight` | 姿态/方向保持权重。 |
| `success_threshold` | 末端到目标的距离阈值。 |

成功判定：末端到计算出的目标位置距离小于 `success_threshold`。

参数怎么读：

- `objects[0]` 是当前被控制/被拿的工具或物体。
- `objects[1]` 是目标参考物体。
- `delta_trans` 是对目标点的额外平移，适合“擦到污渍旁边一点”“推到目标后一段距离”。
- `invisible_object` 可用隐藏辅助点作为目标，避免真实目标 bbox 不适合作为动作终点。

例子：

```yaml
- name: move
  objects: [sponge, dirt]
  delta_trans: [[0.02, 0.0, 0.0]]
  success_threshold: 0.015
```

读法：拿着 sponge 移动到 dirt 附近，并在 x 方向再偏 2cm；末端离目标 1.5cm 内算成功。

### 8.3 `rotate__obj`

用途：旋转一个被夹住的刚体物体。它会根据“物体-夹爪”的相对变换，反推夹爪应该到达的目标位姿。

最小配置：

```yaml
- name: rotate__obj
  objects: [obj]
  success_threshold_move: 0.02
  success_threshold_rotate: 0.1
```

常用参数：

| 参数 | 含义 |
|---|---|
| `rotate_obj_euler_delta` | 物体目标旋转增量范围。 |
| `first_motion` | 先移动还是先旋转，可用 `move` / `rotate`。 |
| `move_offset` | 移动偏移。 |
| `rotate_offset` | 旋转阶段位置偏移。 |
| `rotate_only` | 只旋转，不做完整移动。 |
| `gripper_state` | 执行时夹爪状态。 |
| `ctrl_list` | 可叠加关节控制。 |
| `obj_axis_offset` | 按物体坐标轴做偏移。 |
| `trans_offset` | 平移修正。 |

成功判定：目标位置和目标姿态都达到阈值。

参数怎么读：

- `rotate_obj_euler_delta` 是物体目标姿态的欧拉角增量范围，单位 degree。
- `first_motion: move` 表示先把 EE 移到目标位置附近，再旋转到目标姿态。
- `first_motion: rotate` 表示先调整姿态，再移动到最终位置。
- `obj_axis_offset` 是先把物体参考点沿物体局部轴偏移，适合物体几何中心不是你真正想控制的点。
- `trans_offset` 是最后对 EE 目标位置做线性微调。

例子：

```yaml
- name: rotate__obj
  objects: [box]
  rotate_obj_euler_delta: [[0, 0, 80], [0, 0, 100]]
  first_motion: rotate
  rotate_only: true
  success_threshold_move: 0.02
  success_threshold_rotate: 0.1
```

读法：把已经抓住的 box 绕局部/当前姿态组合后的 z 方向旋转大约 90 度，主要检查姿态误差小于 0.1 rad。

容易混淆：[`rotate`](#73-rotate) 是 articulated joint 旋转；`rotate__obj` 是 rigid object 被夹住后重定向。

### 8.4 `approach__rotate`

用途：让当前物体靠近另一个物体，并可选地先旋转当前物体。适合“对准后靠近”的任务。

最小配置：

```yaml
- name: approach__rotate
  objects: [held_obj, target_obj]
  success_threshold: 0.02
```

常用参数：

| 参数 | 含义 |
|---|---|
| `objects: [move_obj, approach_obj]` | 被移动物体和靠近目标。 |
| `approach_axis` | 从目标哪个轴向靠近，例如 `+x`。 |
| `distance` | 靠近时保持的距离。 |
| `z_offset` | z 方向修正。 |
| `obj_yaw_offset` | 绕 yaw 做修正。 |
| `obj_axis_offset` | 按物体轴做偏移。 |
| `rotate` | 可选旋转子配置。 |
| `hold_vec_weight` | 姿态保持权重。 |

`rotate` 子配置常见字段：

```yaml
rotate:
  type: random
  success_threshold: 0.1
  rotate_obj_euler: [[0, 0, -30], [0, 0, 30]]
```

注意：`approach_rotate` 中的 `dummy_forward` 已弃用。旧配置仍可加载，但该参数会被忽略；普通任务应使用正常的 EE 位姿规划路径。

参数怎么读：

- `approach_axis` 是 `move_obj` 的局部轴，支持 `+x/+y/+z/-x/-y/-z`。
- `distance` 是让 `move_obj` 停在 `approach_obj` 前方多远。
- `obj_yaw_offset` 是绕 world z 额外转一角度，用来修正物体朝向。
- `rotate.type: random` 表示随机采样目标朝向。
- `rotate.type: towards` 表示让物体朝向另一个目标物体。

例子：

```yaml
- name: approach__rotate
  objects: [held_tool, target_part]
  approach_axis: +x
  distance: 0.08
  z_offset: 0.02
  success_threshold: 0.02
  rotate:
    type: random
    success_threshold: 0.15
    rotate_obj_euler: [[0, 0, -20], [0, 0, 20]]
```

读法：让 held_tool 的局部 +x 方向对准 target_part，保持 8cm 距离并抬高 2cm；同时允许物体 yaw 在正负 20 度内随机。

### 8.5 `flip`

用途：翻转已抓住的物体。内部轨迹比较硬编码，属于任务/机器人特化 skill。

最小配置：

```yaml
- name: flip
  objects: [obj]
  gripper_axis: x
```

常用参数：

| 参数 | 含义 |
|---|---|
| `objects: [obj]` | 被翻转物体。 |
| `gripper_axis` | 翻转时参考的夹爪轴。 |
| `open_wait_steps` | 打开夹爪后等待步数。 |
| `t_eps/o_eps` | waypoint 到达阈值。 |

成功判定：物体姿态和位移满足翻转后的几何条件。

注意：新任务优先考虑 `rotate__obj`；只有确实需要翻转轨迹时再用 `flip`。

调参重点：

- `gripper_axis` 必须和当前 robot/夹爪定义一致，否则翻转方向会错。
- `open_wait_steps` 影响释放后的等待时间。太短可能还没稳定就进入下一步。
- 因为内部轨迹更硬编码，跨 robot 或跨物体复用前最好先小批量测试。

### 8.6 `scan`

用途：将持有物体移动到扫描/观察姿态。实现中包含比较固定的轨迹假设。

最小配置：

```yaml
- name: scan
  objects: [obj]
```

常用参数：

| 参数 | 含义 |
|---|---|
| `objects: [obj]` | 被扫描物体。 |
| `process_valid` | 是否检查过程稳定性。 |
| `t_eps/o_eps` | waypoint 到达阈值。 |

成功判定：夹爪仍和物体接触，并且过程稳定。

注意：这是特化 skill，不是通用相机扫描系统。

使用边界：

- 适合已有任务里“拿起后移动到观察位/扫描位”的固定流程。
- 不适合替代 `goto__pose` 做任意位姿移动。新任务如果只是要到一个观察点，优先写 `goto__pose`。

### 8.7 `track`

用途：跟踪随机采样的 waypoint。适合生成“移动到某段轨迹/点位”的动作。

最小配置：

```yaml
- name: track
  way_points_trans:
    min: [0.2, -0.1, 0.2]
    max: [0.4, 0.1, 0.4]
  way_points_ori: [1, 0, 0, 0]
```

常用参数：

| 参数 | 含义 |
|---|---|
| `way_points_num` | waypoint 数量，默认 `1`。 |
| `way_points_trans.min/max` | 位置采样范围。 |
| `way_points_ori` | 姿态基准。 |
| `frame` | 默认 `robot`。 |
| `T_tcp_2_ee` | TCP 到末端变换。 |
| `max_noise_deg` | 姿态扰动。 |

注意：当前实现里有对 `table` 物体/参考的假设，换场景时要先检查是否适配。

参数怎么读：

- `way_points_trans.min/max` 是位置采样盒子。
- `way_points_ori` 是姿态基准，会加 `max_noise_deg` 扰动。
- `way_points_num` 增加后，会生成多段 waypoint。

例子：

```yaml
- name: track
  way_points_num: 3
  way_points_trans:
    min: [0.25, -0.10, 0.25]
    max: [0.45, 0.10, 0.35]
  way_points_ori: [1, 0, 0, 0]
  max_noise_deg: 5
```

读法：在一个 3D 盒子里随机生成 3 个点，姿态围绕 `[1,0,0,0]` 做小扰动。

## 9. 关节、夹爪、等待和回 home

### 9.1 `heuristic__skill`

用途：通用的启发式移动，最常见用法是让手臂回 home。很多 task 在一段 pick/place 之后会加它，把夹爪移回安全姿态。

最小配置：

```yaml
- name: heuristic__skill
  mode: home
  gripper_state: 1.0
```

支持模式：

| `mode` | 含义 |
|---|---|
| `home` | 回到 robot 配置里的 arm home joints。最常用。 |
| `abs_qpos` | `value` 直接作为目标关节位置。 |
| `rel_qpos` | 源码当前把 `value` 当目标关节位置使用，不是严格“当前关节 + delta”。 |
| `rel_ee` | `value` 是相对 EE 的 4x4 transform，需要 controller plan 求解目标关节；batch controller 下未实现。 |

常用参数：

| 参数 | 含义 |
|---|---|
| `move_steps` | 插值步数，默认 `50`。 |
| `gripper_state` | 移动过程中夹爪状态。 |
| `t_eps` | 关节距离成功阈值，默认较宽。 |
| `value` | `abs_qpos/rel_qpos/rel_ee` 的目标值。 |

成功判定：当前关节接近目标关节。

例子：

```yaml
- name: heuristic__skill
  mode: home
  gripper_state: 1.0
  move_steps: 50
```

读法：保持夹爪打开，把当前臂插值移动回 robot 配置里的 home joints。

注意：

- `home` 是最可靠、最常用的模式。
- `abs_qpos` 的 `value` 应该是目标关节数组，不是末端位姿。
- 源码里的 `rel_qpos` 当前实现更接近“直接使用 value 作为目标关节”，不要按“当前关节 + delta”去理解。
- `rel_ee` 需要 controller plan 求解目标关节，batch controller 下不可用。

### 9.2 `home`

用途：旧版/简单版回 home。功能与 `heuristic__skill mode: home` 类似。

最小配置：

```yaml
- name: home
  gripper_state: 1.0
```

注意：新配置更建议用 `heuristic__skill`，因为它模式更明确、可扩展。

什么时候还会看到它：旧任务或早期配置。新写 YAML 时优先：

```yaml
- name: heuristic__skill
  mode: home
```

### 9.3 `joint__ctrl`

用途：直接控制 arm joints，适合绕过笛卡尔规划做小范围关节动作。

最小配置：

```yaml
- name: joint__ctrl
  ctrl_list:
    - [2, -10, "delta"]
  num_steps: 10
```

常用参数：

| 参数 | 含义 |
|---|---|
| `ctrl_list` | 每项为 `[joint_index, angle_deg, mode]`。 |
| `mode` | `abs` 表示设为绝对角度；`delta` 表示在当前角度上加增量。 |
| `num_steps` | 关节插值步数。 |
| `gripper_state` | 移动时夹爪状态。 |
| `success_threshold_js` | 关节距离成功阈值。 |

成功判定：当前 arm joints 接近目标 joints。

例子：

```yaml
- name: joint__ctrl
  ctrl_list:
    - [0, 10, "delta"]
    - [2, -30, "abs"]
  num_steps: 20
  gripper_state: -1.0
  success_threshold_js: 0.01
```

读法：

- 第 0 个 arm joint 在当前角度上加 10 度。
- 第 2 个 arm joint 直接设为 -30 度。
- 用 20 步插值过去，过程中保持夹爪关闭。

使用边界：它绕过了笛卡尔目标位姿的直观语义，直接动关节。适合手腕调整、规避某些 planner 问题；不适合描述“把物体放到某个地方”这种任务目标。

### 9.4 `gripper__action`

用途：只开关夹爪，不移动末端。

最小配置：

```yaml
- name: gripper__action
  gripper_state: 1.0
```

常用参数：

| 参数 | 含义 |
|---|---|
| `gripper_state` | `1` 打开，`-1` 关闭。 |
| `vel` | 可选夹爪速度。 |
| `wait_steps` | 重复开/关动作的步数，默认 `10`。 |

成功判定：动作序列执行完即成功。

例子：

```yaml
- name: gripper__action
  gripper_state: -1.0
  wait_steps: 20
```

读法：在当前 EE 位姿不动的情况下，重复 20 步关闭夹爪。

注意：它不会 attach/detach object。要完成物理抓取，通常还是用 `pick`；单独 `gripper__action` 只适合补动作或测试夹爪状态。

### 9.5 `wait`

用途：保持当前末端位姿和夹爪状态等待若干步。它不是“空等”，而是会持续发保持当前 pose 的命令。

最小配置：

```yaml
- name: wait
  objects: [obj]
  success_threshold: 0.01
```

常用参数：

| 参数 | 含义 |
|---|---|
| `objects: [obj]` | 参与等待判定/上下文的物体。 |
| `success_threshold` | 当前 EE 与目标保持 pose 的距离阈值。 |
| `wait_steps` | 等待步数，默认 `50`。 |
| `gripper_state` | 等待时夹爪状态，默认关闭。 |
| `ignore_substring` | 等待阶段碰撞忽略补充。 |

成功判定：等待命令执行完，且末端保持在目标附近。

例子：

```yaml
- name: wait
  objects: [cup]
  wait_steps: 60
  gripper_state: -1.0
  success_threshold: 0.01
```

读法：保持当前末端 pose 和关闭夹爪等待 60 步；末端偏离保持目标小于 1cm 时认为成功。

使用场景：

- 等物体稳定。
- 等液体/动态效果完成。
- 在双臂并行任务中做时间对齐。

## 10. 移动底盘类

### 10.1 `mobile__translate`

用途：控制移动底盘平移关节 `mobile_translate_x/y`。

最小配置：

```yaml
- name: mobile__translate
  target: [0.2, 0.0]
```

参数：

| 参数 | 含义 |
|---|---|
| `target` | 底盘平移目标 `[x, y]`，默认 `[0.0, 0.0]`。 |

成功判定：源码中 `is_success()` 返回 True；完成条件主要看底盘关节位移达到 target。

注意：

- `target` 是底盘关节位移，不是世界导航目标点。
- 它适合简单移动底盘，不等同于完整导航 planner。
- 如果场景里没有 `mobile_translate_x/y` 这些 dof，会直接不适配。

### 10.2 `mobile__rotate`

用途：控制移动底盘旋转关节 `mobile_rotate`。

最小配置：

```yaml
- name: mobile__rotate
  target: 0.785
```

参数：

| 参数 | 含义 |
|---|---|
| `target` | 旋转目标，单位 rad，默认约 `0.785`。 |

成功判定：源码中 `is_success()` 返回 True；完成条件主要看底盘旋转达到 target。

注意：

- `target` 单位是 rad。`0.785` 大约 45 度。
- 和 `mobile__translate` 一样，它依赖 robot USD/配置中存在对应 mobile dof。

## 11. 液体/特化成功判定类

### 11.1 `pour__water__succ`

用途：倒水任务的成功判定/辅助位姿 skill。它会统计 container 内粒子数量，不是通用 pick/place 动作。

最小配置通常依赖具体倒水任务：

```yaml
- name: pour__water__succ
  container_name: cup
  container_radius: 0.025
  particle_num_th_min: 50
  particle_num_th_max: 300
```

常用参数：

| 参数 | 含义 |
|---|---|
| `translation` | 可选末端目标位置。 |
| `euler` / `quaternion` | 可选末端目标姿态。 |
| `frame` | 默认 `robot`。 |
| `max_noise_m/max_noise_deg` | 目标位姿扰动。 |
| `gripper_state` | 执行时夹爪状态。 |
| `container_name` | 容器物体名，默认 `cup`。 |
| `container_radius` | 判断粒子是否在容器内的半径。 |
| `particle_num_th_min/max` | 容器内粒子数量成功范围。 |
| `container_up` | 容器朝上方向约束。 |

成功判定：落入容器半径内的粒子数量在阈值范围内，并可选检查容器朝向。

注意：它需要 task 里有 fluid particles。普通固体搬运任务不要用它。

参数怎么读：

- `container_name` 决定统计哪个容器附近的粒子。
- `container_radius` 决定“粒子算在容器内”的空间半径。
- `particle_num_th_min/max` 防止两类错误：倒太少不成功，粒子数量异常太多也不认为成功。
- `container_up` 用来约束容器姿态，例如容器不能翻倒。

例子：

```yaml
- name: pour__water__succ
  container_name: cup
  container_radius: 0.03
  particle_num_th_min: 60
  particle_num_th_max: 260
  container_up:
    - [0, 0, 1]
    - 25
```

读法：统计 cup 半径 3cm 内的粒子数量，60 到 260 之间算成功，同时要求容器朝上方向在 25 度容忍范围内。

## 12. 新任务如何选 skill

可以按任务意图从下面这张“决策表”开始：

| 任务意图 | 推荐 skill chain |
|---|---|
| 拿起物体 | `pick` |
| 把物体放到桌面/垫子/盘子上 | `pick -> place(vertical)` |
| 把物体放到另一个物体左/右边 | `pick -> place`，调 `x/y_ratio_range` 和 `success_mode: left/right` |
| 把杯子挂到架子、插入槽位 | `pick -> place(horizontal)` 或任务特化 `place` 参数 |
| 摆餐具 | 每个物体一组 `pick -> place -> heuristic__skill(home)`；多臂可并行 |
| 打开/关闭抽屉或门 | `artpreplan -> open/close`，或直接 `open/close` |
| 旋转旋钮/关节件 | `rotate` |
| 旋转已抓住的刚体 | `pick -> rotate__obj -> place` |
| 只开关夹爪 | `gripper__action` |
| 回安全位 | `heuristic__skill mode: home` |
| 移动底盘 | `mobile__translate` / `mobile__rotate` |
| 倒水成功判断 | `pour__water__succ` |

以 `livingroom_mug_to_coaster` 这种“杯子到杯垫”任务为例，最自然的链条是：

```text
pick(mug) -> place(mug, coaster) -> heuristic__skill(home)
```

最先看这些字段：

1. `objects` 名称是否和 arena/task 里的物体名一致。
2. mug 旁边是否有抓取标注，例如 `Aligned_grasp_sparse.npy`。
3. coaster 的 bbox 是否能代表真实放置区域。
4. `place` 的 `x_ratio_range/y_ratio_range/place_z_offset` 是否让 mug 落在 coaster 上。
5. `success_mode` 是否能正确判断“杯子在杯垫上”，通常优先看 `xybbox` 或默认 `3diou` 是否合适。

一个可执行的起步版本可以长这样：

```yaml
skills:
  - split_aloha:
      - right:
          - name: pick
            objects: [livingroom_mug_0_id9005]
            pre_grasp_offset: 0.06
            filter_z_dir: ["forward", 80]
            post_grasp_offset_min: 0.08
            post_grasp_offset_max: 0.12
          - name: place
            objects: [livingroom_mug_0_id9005, round_coaster_a_0_id9006]
            place_direction: vertical
            position_constraint: object
            x_ratio_range: [0.5, 0.5]
            y_ratio_range: [0.5, 0.5]
            pre_place_z_offset: 0.15
            place_z_offset: 0.04
            filter_x_dir: ["downward", 140]
            success_mode: xybbox
          - name: heuristic__skill
            mode: home
            gripper_state: 1.0
```

这不是保证一次成功的最终参数，而是一个合理的 debug 起点。先让链条跑起来，再根据现象调整：

| 现象 | 下一步 |
|---|---|
| `pick` 找不到可达姿态 | 放宽/删除 `filter_z_dir`，检查 `Aligned_grasp_sparse.npy`。 |
| 抓到了但抬不起来 | 增大 `gripper_change_steps` 或检查 contact/collider。 |
| mug 放偏 | 调 `x/y_ratio_range`，先固定 `[0.5,0.5]` 再扩范围。 |
| mug 悬空/穿透 coaster | 调 `place_z_offset`。 |
| 放置姿态扭曲 | 加/改 `filter_x/y/z_dir`。 |
| 动作成功但判失败 | 先用 `xybbox`，再考虑 `3diou` 或自定义 predicate。 |

### 12.1 新场景确定 skill 和参数的 checklist

拿到一个新场景/新 YAML，按这个顺序看：

1. 看任务动词：拿起、放到、挂上、插入、打开、关闭、旋转、倒入。
2. 看物体类型：rigid object、articulated object、fluid/container、mobile base。
3. 看目标关系：目标在 bbox 内、目标旁边、目标轴方向、关节位移、粒子数量。
4. 看资产标注：grasp npy、dex pose、place range、articulation contact/keypoint。
5. 选最小 skill chain：先少参数跑通，再逐渐加约束。
6. 分开调动作和成功判定：动作看 `filter/ratio/offset`，判定看 `success_mode/threshold`。

对应关系：

| 看到的任务语义 | 先考虑 |
|---|---|
| "pick up / lift" | `pick` + `lift_th` |
| "put A on B" | `pick -> place(vertical)` |
| "put A beside B" | `pick -> place` + ratio 超出 `[0,1]` + `left/right` |
| "hang / insert / place into slot" | `pick -> place(horizontal)` + `align_*_axis` |
| "open / close drawer" | `open/close` + articulation 标注 |
| "turn knob" | `rotate` |
| "reorient held object" | `rotate__obj` |
| "move held tool over target" | `move` 或 `track` |
| "navigate base a little" | `mobile__translate/rotate` |
| "pour water" | 运动 skill + `pour__water__succ` 判定 |

### 12.2 如何从现有 YAML 反推参数依据

不要孤立看单个数字，要看它解决的几何问题。

例子：

```yaml
x_ratio_range: [-0.25, -0.15]
success_mode: left
threshold: 0.01
```

这通常表示“放到目标左边”，不是“错误的负数”。

```yaml
align_pick_obj_axis: [1, 0, 0]
align_place_obj_axis: [1, 0, 0]
align_obj_tol: 10
```

这表示两个物体的局部 x 轴要对齐，常用于挂、插、朝向一致。

```yaml
filter_x_dir: ["downward", 120, 150]
```

这表示 EE x 轴需要落在一个向下的角度带里，常用于控制从上方接近但不完全垂直。

## 13. 新增 skill 需要满足什么接口

自定义 skill 至少要做四件事：

1. 新建一个 Python 类，继承 `BaseSkill`。
2. 用 `@register_skill` 注册。
3. 在 `simple_generate_manip_cmds()` 里生成 `self.manip_list`。
4. 实现 `is_done()` 和 `is_success()`。

骨架：

```python
from core.skills.base_skill import BaseSkill, register_skill

@register_skill
class My_Skill(BaseSkill):
    def __init__(self, robot, controller, task, cfg, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg
        self.manip_list = []

    def simple_generate_manip_cmds(self):
        # 生成 (p, q, function_name, params)
        self.manip_list = [...]

    def is_done(self):
        ...

    def is_success(self):
        ...
```

如果类名是 `My_Skill`，YAML 里的名字会是：

```yaml
- name: my__skill
```

新增后还要确保 `workflows/simbox/core/skills/__init__.py` import 了这个类，否则不会注册进 `SKILL_DICT`。

## 14. 查参数的实用方法

当你不确定某个参数是不是必须时，看源码最直接：

```bash
rg -n "skill_cfg\\[|skill_cfg\\.get|cfg\\[|cfg\\.get" workflows/simbox/core/skills/place.py
```

判断规则：

- `cfg["xxx"]` 或 `skill_cfg["xxx"]`：通常是必填，缺了会直接报错。
- `cfg.get("xxx", default)`：可选，有默认值。
- 参数在 `is_success()` 里出现：影响成功率和数据是否被记录。
- 参数在 `simple_generate_manip_cmds()` 里出现：影响动作轨迹本身。

查现有例子：

```bash
rg -n "name: place|name: pick|name: open|name: rotate" workflows/simbox/core/configs/tasks -g "*.yaml"
```

对新场景最实用的做法不是先写复杂参数，而是先找相似任务，复制最小 chain，再根据新物体的 bbox、抓取标注、成功条件逐个收紧参数。

## 15. 常见误区

1. `objects` 不是自然语言描述，必须和 task/arena 里的 object name 对上。
2. `pick` 依赖抓取标注；只有 USD 模型不等于能抓。
3. `place` 的 ratio 是目标 bbox 的比例，不是世界坐标。
4. `success_mode` 是每个 skill 自己解释的，同名字段不能跨 skill 泛化。
5. 双臂并行不是写两个外层 list，而是在同一个外层 step 里同时写 `left` 和 `right`。
6. `heuristic__skill mode: home` 很常见，它不是任务目标本身，而是把手臂收回安全位，减少下一段碰撞。
7. 特化 skill 如 `scan`、`flip`、`pour__water__succ`、`dynamicpick` 不要当通用模板；先确认场景、robot 和 task 里需要的额外对象/状态都存在。
