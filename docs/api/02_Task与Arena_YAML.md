# SimBox Task 与 Arena YAML API

> 核对日期：2026-08-13；代码基线：527f31b

本文是 SimBox `arena` 与 `task` YAML 的 API 参考。先给数据模型与加载流程，再逐项列出字段的类型、必填性、默认值和运行语义。skill 参数不在此重复，见 [01_Skill_API.md](01_Skill_API.md)；agent 运行时配置见 [03_配置参考.md](03_配置参考.md)。

## 适用范围

| 范围 | 说明 |
| --- | --- |
| `workflows/simbox/core/configs/arenas/**/*.yaml` | Arena YAML。 |
| `workflows/simbox/core/configs/tasks/**/*.yaml` | Task YAML。 |
| 排除 `download` 子目录 | 包括 arenas 与 tasks 下的 `download`。 |

当前统计（核对日期：2026-08-13）：

| 类型 | 文件数 |
| --- | ---: |
| Arena YAML | 31 |
| Task YAML | 1145 |
| task 顶层唯一 key | 19 |
| task-only 全层级唯一 key | 629（叶子 563） |
| task + arena 全层级唯一 key | 654 |
| 含顶层 `planning` key 的 task YAML | 0（运行时注入，见 A3） |

## 术语和约定

### 路径

| 类型 | 示例 | 解析方式 |
| --- | --- | --- |
| 配置文件路径 | `workflows/simbox/core/configs/arenas/base_arena.yaml` | 相对仓库根目录读取。 |
| 资源路径 | `table0/instance.usd` | 相对 task 的 `asset_root` 拼接。 |
| texture 路径 | `floor_textures` | 相对 `asset_root`，代码读取目录内文件。 |
| USD prim path | `/World/Objects/foo` 或 `root` | stage 内部路径或子 prim 名。 |

### 位姿

| 字段 | 格式 | 单位 |
| --- | --- | --- |
| `translation` | `[x, y, z]` | 米 |
| `euler` | `[roll, pitch, yaw]` | 角度制 degree |
| `quaternion` / `orientation` | `[w, x, y, z]` | scalar-first |
| `scale` | `[sx, sy, sz]` | 无量纲 |

### 名称引用

以下字段是命名引用，不是固定枚举：

- `regions[].object` / `regions[].target`：当前 task 已加载的 object、fixture、robot 或 camera 名。
- `skills[].<robot_name>`：`robots[].name`。
- `skills[].<controller_name>`：当前 robot 可建立的 controller 名，常见 `left`、`right`、`base`。
- `skills[].*.objects[]`：skill 需要的 task object 或 arena fixture 名。
- `navigate.goal`：task 顶层 `positions` 中的 key。

## 加载流程

入口 `workflows/simbox/utils/task_config_parser.py::TaskConfigParser.parse_tasks` 用 `OmegaConf.load` 读 Task YAML，要求顶层存在 `tasks`，逐个转成 dict 返回。

`simbox_dual_workflow.py::SimBoxDualWorkFlow` 随后：

1. `_merge_robot_configs()`：读 `robots[].robot_config_file` 合并，task 内字段覆盖基础配置。
2. `_merge_base_configs()`：读 `base.base_config_file` 与 `base.local_navigation_config_file` 递归合并，task/robot 的 `base` 最后覆盖。
3. `reset()`：读 `arena_file` 指向的 Arena YAML 写入 `task_cfg["arena"]`，删除 `arena_file`、`camera_file`、`logger_file` 等运行前引用字段。
4. 创建 `BananaBaseTask`，加载 arena fixtures、objects、robots、cameras、regions、skills。

解析器不删除未知字段；未知字段只有在后续代码显式读取时才有运行语义。主链路未读取的字段见 Part D。

---

# Part A. Task YAML

## A1. 文档结构

```yaml
tasks:
  - name: example_task
    asset_root: workflows/simbox/assets
    task: BananaBaseTask
    task_id: 0
    offset: null
    render: true
    arena_file: workflows/simbox/core/configs/arenas/base_arena.yaml
    env_map: {}
    robots: []
    objects: []
    regions: []
    cameras: []
    skills: []
    data: {}
```

## A2. `tasks`

| 属性 | 内容 |
| --- | --- |
| 类型 | `list[dict]` |
| 必填 | 是 |
| 消费代码 | `TaskConfigParser.parse_tasks()` |
| 当前观测 | 1145 个文件均有 `tasks`，每文件 1 个 task |

## A3. Task Object

```yaml
name: <str>
asset_root: <str>
task: BananaBaseTask
task_id: <int>
offset: null
render: <bool>
arena_file: <path>

env_map: <EnvMap>
robots: [<Robot>, ...]
objects: [<Object>, ...]
regions: [<Region>, ...]
cameras: [<Camera>, ...]
skills: [<SkillPhase>, ...]
data: <Data>

neglect_collision_names: [<str>, ...]      # optional
random_region_list: [<RandomRegion>, ...]  # optional
positions: {<name>: {x, y, yaw}}           # optional
distractors: <Distractors>                 # optional
fluid: <Fluid>                             # optional
```

### 必填字段

| 字段 | 类型 | 必填 | 允许值 / 当前观测 | 运行语义 |
| --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 任意非空字符串。 | task 实例名，传给 `BaseTask(name=...)`。 |
| `asset_root` | `str` | 是 | 当前全部为 `workflows/simbox/assets`。 | 所有 object、fixture、texture、distractor、envmap 资源的根目录。 |
| `task` | `str` | 是 | 当前全部为 `BananaBaseTask`。 | `get_task_cls(task)` 取任务类，写错会 KeyError。 |
| `task_id` | `int` | 是 | 当前全部为 `0`。 | 构造 USD root prim：`/World/task_<task_id>`。 |
| `offset` | `null` 或 `list[float]` | 是 | 当前全部为 `null`。 | 传给 Isaac `BaseTask`，当前任务类不额外解释。 |
| `render` | `bool` | 是 | 当前 1144 个 `true`、1 个 `false`。 | `false` 时 `get_observations()` 不读 cameras。 |
| `arena_file` | `str` | 是 | 指向 Arena YAML。 | workflow reset 时读取，随后从 task cfg 删除。 |
| `env_map` | `dict` | 是 | 见 A4。 | DomeLight 环境光配置。 |
| `robots` / `objects` / `regions` / `cameras` / `skills` | `list[dict]` | 是 | 可为空。 | 见 A5-A11 与 Part B。 |
| `data` | `dict` | 是 | 见 A11。 | logger、语言、episode 长度配置。 |

### 可选字段

| 字段 | 类型 | 默认值 | 允许值 / 当前观测 | 运行语义 |
| --- | --- | --- | --- | --- |
| `neglect_collision_names` | `list[str]` | `[]` | 常见 `["table"]`。 | 名称含这些子串的 object/fixture 从 global collision 集合移除。 |
| `random_region_list` | `list[dict]` | `[]` | 部分任务预设随机区域池。 | 仅当 `regions[].priority` 存在时使用。 |
| `positions` | `dict[str, {x, y, yaw}]` | 无 | `nav_to_pick`、`nav_to_place` 等。 | `navigate.goal` 的命名目标表。 |
| `distractors` | `dict` | 无 | 见 A9。 | 生成并摆放视觉干扰物。 |
| `fluid` | `dict` | 无 | 见 A10。 | 创建 PhysX 粒子流体；存在时强制 GPU physics。 |

### 使用要点

- scene-only task 可以把 `robots`、`objects`、`regions`、`cameras`、`skills` 都写成空列表。
- arena fixture 的 `path` 依赖当前 task 的 `asset_root`，arena 不能独立解析资源。
- `positions` 的 key 是开放命名，与 `navigate.goal` 一致即可。

### `planning`（运行时注入，不在 task YAML 中书写）

实测 0 个 task YAML 含顶层 `planning` key。该块由 `agent/compiler.py::compile_task_config` 编译时注入（`agent/compiler.py:209-219`）：合并 `agent/config.yaml` 的 `planning` 与源 task 的 `planning`（task 可覆盖 `pick_place` / `execution_safety` 项），但 `collision_world` 整体取自 agent 配置、task 无法削弱；随后以 `collision_world.mode` 调用 `config_contract.py::validate_planning_contract` 校验，写回 `task["planning"]`。

#### `collision_world`

| 字段 | 类型 | 默认值 | 语义 |
| --- | --- | --- | --- |
| `mode` | `str` | `physics_schema` | 操作类规划唯一使用 Physics schema；旧 `legacy_stage_scan` / `hybrid` 值只产生弃用警告并被规范化为 `physics_schema`。skill 级解析由 `resolve_skill_collision_world_mode` 完成：`navigate`、`observe_hold` 等非操作 skill 为 `passthrough`，其他操作 skill 为 `physics_schema`。`home` Skill 与 `heuristic__skill(mode: home)` 另走 `direct_execution`，不创建或切换碰撞规划世界。 |
| `strict` | `bool` | `true` | `CollisionSceneManager` 读取（`collision_scene_manager.py:132`）。`true` 时遇到缺失/不支持碰撞体直接报错；`false` 时记入 `schema_exclusions` 降级。 |
| `exact_exclusions` | `list[dict]` | `[]` | 精确排除项，每项必须含 `prim_path` 与 `reason`；路径必须是完整绝对 Prim 路径、`reason` 非空、不可重复（`validate_exact_exclusions`，`collision_scene_manager.py:99`）。 |

`mode` 说明：

| 值 | 语义 |
| --- | --- |
| `physics_schema` | 按 Physics 刚体精确映射 CuRobo 世界；`pick` / `place` / `pick_plan_probe` 使用该规划契约。 |
| `legacy_stage_scan` / `hybrid` | 已废弃；不会选择旧运行时世界，规范化时仅产生弃用警告。 |
| `passthrough` | 仅用于 skill 级解析（导航/观测类），不是 task 级模式。 |

#### `pick_place`

| 字段 | 类型 | 默认值 | 语义 |
| --- | --- | --- | --- |
| `terminal_step_m` | `float` | `0.005` | pick 末端到位步长，注入 `pick.terminal_step_m`。 |
| `max_terminal_distance_m` | `float` | `0.10` | pick 末端最大到位距离，注入 `pick.max_terminal_distance_m`。 |
| `place_continuous_descent` | `bool` | `true` | place 是否连续下降，注入 `place.place_continuous_descent`。 |
| `place_terminal_step_m` | `float` | `0.01` | place 末端到位步长，注入 `place.place_terminal_step_m`。 |
| `place_terminal_tolerance_m` | `float` | `0.005` | place 末端到位容差，注入 `place.place_terminal_tolerance_m`；`place.py` 实际取 `min(0.005, place_terminal_step_m)`。 |
| `place_terminal_max_path_length_ratio` | `float` | `1.5` | place 末端路径长度比上限。 |
| `place_terminal_max_path_deviation_m` | `float` | `0.01` | place 末端路径偏差上限。 |
| `place_settle_steps` | `int` | `10` | place 放置后稳定步数，注入 `place.place_settle_steps`。 |

#### `execution_safety`

`simbox_dual_workflow.py:416-464` 读取，构造 `SafetyMonitor` 与 `ExecutionSupervisor`；`_execution_safety_precheck`（`simbox_dual_workflow.py:1889`）每个 sim 步前调用。

| 字段 | 类型 | 默认值 | 语义 |
| --- | --- | --- | --- |
| `enabled` | `bool` | `true` | 预检总开关；缺省时等于 `task_uses_physics_schema(mode)`。 |
| `max_waypoint_stride` | `int` | `2` | 计划路点最大步长（`curobo/controller.py`）。 |
| `max_replans_per_phase` | `int` | `2` | 每个 phase 允许的重规划次数上限（`execution_supervisor.py:20`）。 |
| `hold_steps_before_replan` | `int` | `5` | 重规划前保持步数（`execution_supervisor.py:21`）。 |
| `dynamic_sync_interval_steps` | `int` | `5` | 动态障碍物位姿同步间隔（`sync_dynamic_poses`，`simbox_dual_workflow.py:1909`）。 |

`CollisionSceneManager` 还从 `execution_safety` 读取两个可选键：`dynamic_translation_replan_m`（默认 `0.01`）、`dynamic_rotation_replan_deg`（默认 `3.0`），动态对象位姿变化超过阈值时触发重规划（`collision_scene_manager.py:137-141`）。

#### Skill 分组（`config_contract.py`）

| 集合 | 内容 | 校验规则 |
| --- | --- | --- |
| `PHYSICS_SCHEMA_SKILLS` | `{pick, place}` | pick 要求恰好 1 个 object，place 恰好 2 个。 |
| `VALIDATION_ONLY_SKILLS` | `{pick_plan_probe}` | 仅校验/探测；要求 `metadata.workspace_probe` 且恰好 1 个 object。 |
| `NON_MANIPULATION_SKILLS` | `{navigate, observe_hold}` | 不参与操作类校验，skill 级模式固定 `passthrough`。 |
| `DIRECT_EXECUTION_MODE` | `direct_execution` | `home` 类 Skill 的直接关节动作模式；不创建或切换碰撞规划世界。 |

`validate_planning_contract` 校验 Physics-schema 操作的对象数量和 DAG 约束；`direct_execution` 是 Skill 的执行模式，不属于 collision-world mode。

---

## A4. `env_map`

由 `BananaBaseTask._set_envmap()` 消费。

```yaml
env_map:
  envmap_lib: envmap_lib
  apply_randomization: true
  intensity_range: [5000, 5000]
  rotation_range: [0, 0]
  light_type: DomeLight  # optional
```

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `envmap_lib` | `str` | 是 | 无 | 当前全部为 `envmap_lib`。 | 相对 `asset_root` 的 HDR 目录；代码读取 `asset_root/envmap_lib/*.hdr` 并排序。 |
| `apply_randomization` | `bool` | 是 | 无 | 当前 1128 个 `true`、17 个 `false`。 | `true` 随机选 HDR 并采样 intensity/rotation；`false` 固定用第 0 个 HDR、强度 `1000.0`、旋转 `[0,0,0]`。 |
| `intensity_range` | `[min, max]` | 是 | 无 | 通常 `[5000, 5000]`。 | 随机模式下 `random.uniform(min, max)`。 |
| `rotation_range` | `[min, max]` | 是 | 无 | 通常 `[0, 0]`。 | 随机模式下三轴同范围采样，degree。 |
| `light_type` | `str` | 否 | `DomeLight` | 代码仅实现 `DomeLight`。 | 在 task root 下创建/复用 DomeLight；其他值不进入环境光分支。 |

---

## A5. `robots[]`

task 内机器人配置与 `robot_config_file` 指向的基础配置合并（`_merge_robot_configs`），task 内同名字段优先级更高。

```yaml
robots:
  - name: split_aloha
    robot_config_file: workflows/simbox/core/configs/robots/split_aloha.yaml
    translation: [0.0, 0.0, 0.0]
    euler: [0.0, 0.0, 90.0]
    use_batch: true
```

### 字段

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | `split_aloha`、`lift2`、`lift2_0`、`lift2_1`、`genie1`、`franka`、`frankarobotiq`。 | 实例名，skills 外层 robot key 必须匹配。 |
| `robot_config_file` | `str` | 常用 | 无 | 完整候选见 E1。 | 基础配置路径；workflow 先读它再按 task 内字段覆盖。 |
| `target_class` | `str` | 合并后必填 | 无 | 完整候选见 E1。 | 机器人类注册名，传给 `get_robot_cls()`；通常来自 `robot_config_file`。 |
| `path` | `str` | 合并后必填 | 无 | 如 `split_aloha_mid_360/robot.usd`。 | 相对 `asset_root` 的机器人 USD。 |
| `translation` | `[x,y,z]` | 是 | 无 | 均显式配置。 | robot root prim 初始位置。 |
| `euler` | `[r,p,y]` | `euler`/`quaternion` 二选一 | 无 | 常用。 | 初始朝向，degree。 |
| `quaternion` | `[w,x,y,z]` | `euler`/`quaternion` 二选一 | 无 | 少见。 | 初始朝向；同写 `euler` 时 `get_orientation()` 优先 `euler`。 |
| `scale` | `[sx,sy,sz]` | 否 | `[1,1,1]` | 很少写。 | root prim 缩放。 |
| `robot_file` | `str` 或 `list[str]` | 操作类 skill 时必填 | 无 | 基础配置提供。 | controller 配置路径；含 `left` 的映射到 `left` controller，含 `right` 的映射到 `right`。 |
| `constrain_grasp_approach` | `bool` | 否 | `false` | 少量 `true`。 | 约束抓取接近方向。 |
| `collision_activation_distance` | `float` | 否 | `0.03` | 常见 `0.05`。 | 控制器碰撞激活距离。 |
| `ignore_substring` | `list[str]` | 否 | `["material","Plane","conveyor","scene","table"]` | 任意字符串。 | 规划时忽略名称含这些子串的 prim，子串匹配。 |
| `use_batch` | `bool` | 否 | `false` | 部分 `true`。 | 是否用 batch 规划。 |
| `left_joint_home` / `right_joint_home` | `list[float]` | 合并后必填 | 无 | 基础配置提供。 | 左右臂 home 关节位。 |
| `left_joint_home_std` / `right_joint_home_std` | `list[float]` | 否 | 长度匹配的 0 列表 | 大量显式配置。 | reset/home 关节噪声标准差。 |
| `left_gripper_home` / `right_gripper_home` | `list[float]` | 合并后必填 | 无 | 基础配置提供。 | home 时夹爪关节位置。 |
| `tcp_offset` | `float` | 合并后常用 | 基础配置内定义 | `0.115`、`0.12`、`0.135` 等。 | pick 类 skill 默认 TCP 偏移。 |
| `base` | `dict` | 移动底盘可选 | 无 | 见下。 | 底盘与本地导航配置。 |

### `base`

`_merge_base_configs`（`simbox_dual_workflow.py:217-244`）合并顺序：`base_config_file` 为基础 → `local_navigation_config_file` 深合并覆盖 → task/robot 的 `base` 最后覆盖。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `base_config_file` | `str` | 无 | 底盘基础参数 YAML，候选见 E1。 |
| `local_navigation_config_file` | `str` 或 `list[str]` | 无 | ROS-free 本地导航参数（A* + waypoint 控制器），例如 `workflows/simbox/core/configs/navigation/default_local_navigation.yaml`。 |
| `steering_joint_names` | `list[str]` | 基础配置提供 | 底盘转向关节名。 |
| `wheel_joint_names` | `list[str]` | 基础配置提供 | 车轮关节名。 |
| `wheel_base` / `track_width` / `wheel_radius` | `float` | 基础配置提供 | 轴距 / 轮距 / 轮半径。 |
| `steering_limit` / `wheel_velocity_limit` | `float` | 基础配置提供 | 最大转向角 / 车轮速度上限。 |
| `platform.profile` | `str` | 基础配置提供 | `virtual_base`、`differential_drive`、`ranger_mini_v3`。 |
| `local_navigation.settle` | `dict` | 基础配置提供 | `linear_speed_tolerance`、`angular_speed_tolerance`、`consecutive_steps`。 |
| `local_navigation.footprint_points` | `list` | 基础配置提供 | 底盘 footprint，A* 膨胀与避障使用。 |
| `local_navigation.inflation_radius_m` | `float` | 基础配置提供 | 障碍膨胀半径。 |
| `local_navigation.controller_hard_limits` | `dict` | 基础配置提供 | `max_velocity`、`min_velocity`、`max_accel`、`max_decel`。 |

历史遗留的 ROS/Nav2 字段（`ros`、`nav2_skill`、`nav_config_file`）已从配置中移除，本地导航链路不消费。

---

## A6. `objects[]`

由 `_load_obj()` 按 `target_class` 分发到对象类。

```yaml
objects:
  - name: pick_object
    target_class: RigidObject
    path: pick_and_place/pre-train-pick/assets/omniobject3d-cup/xxx/Aligned_obj.usd
    prim_path_child: Aligned
    translation: [0.0, 0.0, 0.0]
    euler: [0.0, 0.0, 0.0]
    scale: [1.0, 1.0, 1.0]
    category: cup
    dataset: oo3d
    apply_randomization: false
```

### 通用字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | 实例名，regions 与 skills 引用。 |
| `target_class` | `str` | 是 | 无 | 候选见 E2。 |
| `path` | `str` | 资源型对象必填 | 无 | 相对 `asset_root` 的 USD。 |
| `translation` | `[x,y,z]` | 常用 | 无 | 初始位置，可能被 `regions` 覆盖。 |
| `euler` | `[r,p,y]` | `euler`/`quaternion` 二选一 | 无 | 初始朝向，degree。 |
| `quaternion` | `[w,x,y,z]` | `euler`/`quaternion` 二选一 | 无 | 初始朝向。 |
| `scale` | `[sx,sy,sz]` | 否 | `[1,1,1]` | 初始缩放。 |
| `visible` | `bool` | 否 | `true` | 初始可见性。 |
| `category` | `str` | 语义任务常用 | 无 | 语言替换、随机化、distractor 排除时使用。 |
| `dataset` | `str` | 否 | 无 | 数据集元信息；主加载链路不分支。 |
| `apply_randomization` | `bool` | 否 | `false` | 对 `RigidObject`/`ArticulatedObject` 启用资源随机化。 |
| `texture` | `dict` | 否 | 无 | 见下方 Texture Block。 |
| `mass` | `float` | 否 | `None` | 传给 `RigidPrim`。 |
| `color` | `[r,g,b]` | 否 | ShapeObject 默认 `[1,0,0]` | 程序几何颜色。 |
| `physics` / `source_physics` / `collision` / `collider` | `dict` | 否 | `{}` / `{}` / 无 / `""` | 物理与碰撞配置，`collision_scene_manager.py` 读取。 |
| `collision_enabled` | `bool` | 否 | 无 | 对象碰撞开关，`collision_scene_manager.py` 与 `GeometryObject` 读取。 |
| `scene_prim_path` / `scene_reference` | `str` | 否 | 无 | 原始 scene 元数据，主加载链路不读取。 |
| `optimize_2d_layout` | `bool` | 否 | 无 | 布局优化辅助逻辑使用。 |
| `filter_collision` | `bool` | 否 | 无 | 主加载链路不读取。 |

### Target Classes

#### `RigidObject`

`workflows/simbox/core/objects/rigid_object.py::RigidObject`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `path` | `str` | 是 | 无 | 相对 `asset_root` 的 USD。 |
| `prim_path_child` | `str` | 是 | 无 | 实际刚体子 prim，常见 `Aligned`、少量 `root`。 |
| `init_translation` | `[x,y,z]` | 否 | 无 | 与 `init_orientation`、`init_parent` 一起把对象放到相对位置。 |
| `init_orientation` | quaternion | 否 | 无 | 同上。 |
| `init_parent` | `str` | 否 | 无 | 相对 task root 的父 prim 路径，可含 OmegaConf 插值。 |
| `gap` | 任意 | 否 | 无 | 容器摆放间隙候选；随机化时可自动写入。 |
| `mass` | `float` | 否 | `None` | 传给 `RigidPrim`。 |

#### `GeometryObject`

`workflows/simbox/core/objects/geometry_object.py::GeometryObject`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `path` | `str` | 是 | 无 | 相对 `asset_root` 的 USD。 |
| `prim_path_child` | `str` | 否 | 无 | 若提供，拼到对象 root 下。 |
| `collision_enabled` | `bool` | 否 | `false` | 是否生成 bbox collision proxy。 |
| `collision_approximation` | `str` | 否 | `bbox` | 只允许 `bbox`，其他值抛 `ValueError`。 |
| `collision_visible` | `bool` | 否 | `false` | 是否显示 collision proxy。 |

#### `ArticulatedObject`

`workflows/simbox/core/objects/articulated_object.py::ArticulatedObject`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `path` | `str` | 是 | 无 | 必须指向 `instance.usd`，代码替换成 `Kps/<info_name>/info.json` 读关键点。 |
| `info_name` | `str` | 是 | 无 | 关键点信息目录名，如 `open_v_inner`、`close_h_left`。 |
| `category` | `str` | 是 | 无 | articulation 类别。 |
| `fix_base` | `bool` | 否 | `false` | true 时创建 FixedJoint 固定 base。 |
| `joint_position_range` | `[min,max]` | 否 | 无 | 初始化随机采样 joint position。 |
| `strict_init.joint_positions` / `strict_init.joint_indices` | `list` | 否 | 无 | 严格设置目标关节位置/索引。 |
| `art_cat` | `str` | 随机化时需要 | 无 | reset 中 articulation 随机资产目录。 |
| `obj_info_path` | `str` | 随机化后写入 | 无 | 信息文件相对路径；skill 也可覆盖。 |

#### `PlaneObject` / `ConveyorObject` / `BoxObject` / `ShapeObject` / `XFormObject`

| 类型 | 必填字段 | 主要用途 |
| --- | --- | --- |
| `PlaneObject` | `size`；`collision_enabled`、`collision_thickness`（默认 `0.02`）、`collision_visible` 可选 | 程序生成平面；当前 task objects 未用，arena fixtures 使用。 |
| `ConveyorObject` | `path`、`linear_velocity`、`linear_track_list`、`angular_velocity`、`angular_track_list` | 传送带，轨道名 `World/<track>/node_`、`World/<track>/validate_obj`；当前仅在 arena fixtures 出现。 |
| `BoxObject` | `name`、`target_class` | 程序 cube，`scale` 控大小，可选 `color`、`collision_enabled`。 |
| `ShapeObject` | `name`、`target_class` | 低层 `GeometryPrim` 包装，当前无完整 shape 创建逻辑。 |
| `XFormObject` | `name`、`target_class`、`path` | 加载 USD 为 xform，可选 `parent_obj`。 |

### 随机化字段

| 字段 | 类型 | 允许值 | 说明 |
| --- | --- | --- | --- |
| `apply_randomization` | `bool` | `true` / `false` | 总开关。 |
| `randomization_scope` | `str` 或 `list[str]` | `category`、`full`、类别列表 | 随机采样资源范围，候选见 E3。 |
| `orientation_mode` | `str` | `keep`、`suggested`、`random` | 姿态策略，语义见 E3。 |
| `scale_mode` | `str` | `keep`、`suggested` | 缩放策略，语义见 E3。 |
| `gap` | 任意 | `false` 或 list | 容器/间隙元数据。 |

### Texture Block

```yaml
texture:
  texture_lib: floor_textures
  apply_randomization: true
  texture_id: 1
  texture_scale: [1.0, 1.0]
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `texture_lib` | `str` | 是 | 无 | 相对 `asset_root` 的纹理目录，候选如 `floor_textures`、`table_textures`、`val2017`。 |
| `apply_randomization` | `bool` | 是 | 无 | true 随机选目录内文件；false 用 `texture_id`。 |
| `texture_id` | `int` | 非随机必需 | 无 | 排序后文件索引，越界会失败。 |
| `texture_scale` | `list[float]` 或 `float` | 否 | `None` | 传给 OmniPBR。 |
| `target_prim_path` | `str` | 否 | 无 | 仅 `GeometryObject.apply_texture()` 支持，从该子 prim 递归绑定。 |

---

## A7. `regions[]`

由 `BananaBaseTask._set_regions()` 与 `RandomRegionSampler` 消费。

```yaml
regions:
  - object: pick_object
    target: table
    random_type: A_on_B_region_sampler
    random_config:
      pos_range:
        - [-0.1, -0.1, 0.0]
        - [0.1, 0.1, 0.0]
      yaw_rotation: [-180.0, 180.0]
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `object` | `str` | 是 | 无 | 被移动和设置 pose 的对象。 |
| `target` | `str` | 是 | 无 | 采样参考对象。 |
| `random_type` | `str` | 普通分支必填 | 无 | 采样算法，候选见下与 E4。 |
| `random_config` | `dict` | 普通分支必填 | 无 | 以关键字参数传给 sampler。 |
| `priority` | `list[int]` 或 `bool` | 否 | 无 | 存在时使用 `random_region_list`；真取指定索引，假随机选索引。 |
| `container` | `str` | 否 | 无 | 特殊分支：放到 container pose 上并用 `container.gap` 修正 x。 |
| `z_init` | `float` | `container` 分支必填 | 无 | container 分支中 object 的 z 偏移（当前 `0.06`）。 |
| `sub_tgt_prim` | `str` | 否 | 无 | 用 `target.prim_path + sub_tgt_prim` 作采样目标（当前 `/World`）。 |
| `target2` | `str` | `A_along_B_C_circle_sampler` 必填 | 无 | 双目标 sampler 的第二目标。 |
| `visible` | `bool` | 否 | 无 | 摆放代码不读取，配置元数据。 |

### Samplers

#### `A_on_B_region_sampler`

把 object 放到 target 上表面附近（杯子放桌上、物体放托盘上）。

| 参数 | 类型 | 必填 | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `pos_range` | `[[min_x,min_y,min_z],[max_x,max_y,max_z]]` | 是 | 米 | 以 target bbox 中心为 x/y 基准、顶面为 z 基准，均匀采样并加上 `dx/dy/dz`。 |
| `yaw_rotation` | `[min_deg,max_deg]` | 是 | degree | 均匀采样 yaw，叠加到 object 原姿态上（不重置 roll/pitch）。 |

结果：`translation.x/y = target_bbox_center.x/y + sampled`；`z = target_bbox.max.z + (object 中心到底面距离) + 0.001 + sampled_z`；`orientation = Rz(yaw) * current`。

#### `A_in_B_region_sampler`

把 object 放到 target 中心上沿，z 自动对齐 target 顶面与 object 底面。

| 参数 | 类型 | 默认值 | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `x_bias` / `y_bias` | `float` | `0` | 米 | target local pose 上的固定偏移。 |
| `z_bias` | `float` | `0` | 米 | 自动对齐高度上的额外偏移。 |

结果：`x/y = target_local_translation + bias`；`z = target_bbox.max.z + (object 中心到底面距离) - 0.005 + z_bias`；保留原朝向，不加随机 yaw。

#### `A_by_B_region_sampler`

把 object 放在 target 旁边矩形区域内，z 用两者 bbox 半高差让底部近似对齐。

| 参数 | 类型 | 必填 | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `pos_range` | 同上 | 是 | 米 | 只使用采样 x/y。 |
| `yaw_rotation` | `[min,max]` | 是 | degree | 叠加到 object 姿态。 |

结果：`x/y = target_bbox_center + sampled`；`z = target_bbox_center.z + obj_half_h - tgt_half_h`。

#### `A_by_B_circle_sampler`

把 object 放在 target 周围圆环/扇形区域。

| 参数 | 类型 | 必填 | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `r_range` | `[min,max]` | 是 | 米 | 均匀采样半径。 |
| `theta_range` | `[min_deg,max_deg]` | 是 | degree | 极角；`0` 对应 +x、`90` 对应 +y。 |
| `yaw_rotation` | `[min,max]` | 是 | degree | 叠加到 object 姿态。 |

结果：`x/y = target_bbox_center + [r*cosθ, r*sinθ]`；`z` 同 `A_by_B_region_sampler`。

#### `A_face_B_circle_sampler`

把 object 放在 target 当前朝向指向的射线上。

| 参数 | 类型 | 必填 | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `r_range` | `[min,max]` | 是 | 米 | 沿 target 当前 yaw 方向采样距离。 |
| `yaw_rotation` | `[min,max]` | 是 | degree | 叠加到 object 姿态。 |

结果：`x/y = target_bbox_center + r * [cos(target_yaw), sin(target_yaw)]`；`z` 同半高差对齐。

#### `A_along_B_C_circle_sampler`

把 object 沿 target B 的朝向放到越过 target2 C 之后；外层 `target` 是 B，`target2` 是 C。

| 参数 | 类型 | 必填 | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `target2` | `str` | 是 | 名称引用 | 写在 region 外层，读 B 与 C 的 world pose 距离。 |
| `r_range` | `[min,max]` | 是 | 米 | 最终半径 = `distance(B,C) + uniform(r_range)`。 |
| `yaw_rotation` | `[min,max]` | 是 | degree | 叠加到 object 姿态。 |

结果：`x/y = B_bbox_center + (distance(B,C)+sampled_r) * [cos(B_yaw), sin(B_yaw)]`；`z` 用 B 半高差对齐。

---

## A8. `cameras[]`

由 `BananaBaseTask._load_camera()` 与 `CustomCamera` 消费。

```yaml
cameras:
  - name: split_aloha_head
    camera_file: workflows/simbox/core/configs/cameras/realsense_d455_v3.yaml
    parent: split_aloha/.../top_camera_link
    translation: [0.0, -0.00818, 0.1]
    orientation: [0.658, 0.259, -0.282, -0.648]
    camera_axes: usd
    apply_randomization: false
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | camera dict key 与 observation key。 |
| `camera_file` | `str` | 是 | 无 | 外部内参/分辨率配置，候选见 E1。 |
| `parent` | `str` | 是 | 空字符串表示 world-mounted | 非空时 mount 创建在该 parent 下，相对路径拼到 task root。 |
| `translation` | `[x,y,z]` | 是 | 无 | 局部位置。 |
| `orientation` | `[w,x,y,z]` | 是 | 无 | 局部朝向。 |
| `camera_axes` | `str` | 是 | 无 | 当前全部为 `usd`，传给 `set_local_pose()`。 |
| `apply_randomization` | `bool` | 否 | `false` | true 时每次 randomize 扰动外参。 |
| `max_translation_noise` | `float` | 否 | `0.05` | 外参扰动最大平移。 |
| `max_orientation_noise` | `float` | 否 | `10.0` | 外参扰动最大旋转，degree。 |
| `output_mode` | `str` | 否 | `rgb` | `rgb` 取 RGBA 前三通道；`diffuse_albedo` 挂 albedo annotator。当前目标 YAML 未观测。 |

### Camera File 字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `camera_params` | `[fx,fy,cx,cy]` | 是 | 无 | 内参。 |
| `resolution_width` / `resolution_height` | `int` | 是 | 无 | 渲染分辨率。 |
| `pixel_size` | `float` | 是 | 无 | 像素尺寸，micron。 |
| `f_number` | `float` | 是 | 无 | 光圈。 |
| `focus_distance` | `float` | 是 | 无 | 对焦距离。 |
| `with_distance` | `bool` | 否 | `true` | 输出 distance-to-image-plane。 |
| `with_semantic` / `with_bbox2d` / `with_bbox3d` / `with_motion_vector` | `bool` | 否 | `false` | 各渲染通道开关。 |
| `depth` | `bool` | 否 | `false` | depth 标志。 |
| `camera_type` / `frequency` | `str` / `int` | 否 | 无 | 元信息，`CustomCamera` 不读取。 |

---

## A9. `distractors`

由 `_create_distractor_cfg()` 与 `visual_distractor.set_distractors()` 消费。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `path` | `str` | 是 | 无 | 相对 `asset_root` 的干扰物资产根，搜索 `asset_root/path/*/*/*.usd`。 |
| `min_num` / `max_num` | `int` | 是 | 无 | 每次随机采样数量范围。 |
| `target` | `str` | 是 | 无 | 摆放目标对象/fixture 名。 |
| `scale` | `[sx,sy,sz]` | 否 | `[1,1,1]` | 生成 object 缩放。 |
| `pos_range` | `[[x_min,y_min],[x_max,y_max]]` | 否 | 不限制 | 相对 target 中心的 XY 限制。 |
| `min_object_distance` | `float` | 否 | `0.03` | 与主 objects 的最小 XY 距离。 |
| `distractor_buffer` | `float` | 否 | `0.03` | distractor 之间 buffer。 |
| `max_attempts` | `int` | 否 | `10` | 放置尝试次数。 |
| `fallback_z` | `float` | 否 | `-5.0` | 失败时移出视野的 z。 |
| `exclude_categories` | `list[str]` | 否 | `[]` | 精确排除类别。 |
| `exclude_keywords` | `list[str]` | 否 | `[]` | 按类别名子串排除，大小写不敏感。 |
| `target_class` | `str` | 否 | `RigidObject` | `_rebuild_distractors()` 只支持 `RigidObject`。 |
| `prim_path_child` | `str` | 否 | `Aligned` | 生成 RigidObject 子 prim。 |
| `translation` | `[x,y,z]` | 否 | `[0,0,0]` | 生成 object 初始位置。 |

---

## A10. `fluid`

由 `_set_fluid()` 消费。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `container_name` | `str` | 是 | 无 | 初始粒子网格中心使用的对象名。 |
| `particleContactOffset` | `float` | 否 | `0.005` | 粒子 contact offset。 |
| `spacing_scale` | `float` | 否 | `1.2` | 粒子间距倍率。 |
| `numParticlesX` / `numParticlesY` / `numParticlesZ` | `int` | 否 | `7` / `7` / `450` | 粒子网格尺寸；当前 Z 常为 200 或 400。 |
| `center_z` | `bool` | 否 | `false` | 粒子网格 z 是否围绕容器中心。 |
| `z_offset` | `float` | 否 | `0.0` | 粒子网格 z 偏移。 |
| `max_velocity` | `float` | 否 | `0.8` | 粒子系统最大速度。 |
| `mass` / `density` | `float` | 否 | `0.0` | 粒子质量 / 密度。 |
| `color` | `[r,g,b]` | 否 | `[1,1,1]` | 粒子材质颜色。 |
| `emissiveColor` | `[r,g,b]` | 否 | `[0,0,0]` | 自发光颜色。 |
| `opacity` | `float` | 否 | `1.0` | 透明度。 |

---

## A11. `data`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `task_dir` | `str` | 是 | 无 | logger 输出任务目录。 |
| `language_instruction` | `str` | 是 | 内部 fallback 为 pick 模板 | 高层指令，可 `;` 分隔多条。 |
| `detailed_language_instruction` | `str` | 是 | 内部 fallback 为 grasp 模板 | 详细指令，可 `;` 分隔多条。 |
| `collect_info` | `str` | 是 | 无 | 采集元信息，articulation 随机化时可能被覆盖。 |
| `version` | `str` | 否 | `v1.0` | 数据版本。 |
| `update` | `bool` | 是 | 无 | 任务更新标记。 |
| `max_episode_length` | `int` | 是 | 无 | episode 最大步数。 |
| `log_motion_vectors` | `bool` | 否 | `false` | 是否记录 motion vectors。 |

---

# Part B. Skill 编排结构（task 侧）

skill 各注册名的参数、枚举与成功判定详见 [01_Skill_API.md](01_Skill_API.md)。本文件只描述 task 侧如何编排 skill。

## B1. `skills[]` 层级（Legacy 顺序模式）

```yaml
skills:
  - split_aloha:
      - left:
          - name: pick
            objects: [pick_object]
      - right:
          - name: place
            objects: [pick_object, tray]
```

| 层级 | 类型 | 说明 |
| --- | --- | --- |
| `skills[]` | `dict` | 一个 phase，legacy 模式按顺序执行。 |
| `<robot_name>` | `list[dict]` | key 必须匹配 `robots[].name`。 |
| `<controller_name>` | `list[dict]` | 常见 `left`、`right`、`base`。 |
| skill item | `dict` | 至少含 `name`；大多数还需要 `objects`。 |

## B2. DAG 模式

任意 skill item 含 `id` 或 `depends_on` 时，整个 task 进入 DAG 模式：

| 字段 | 类型 | 规则 |
| --- | --- | --- |
| `id` | `str` | 可选。显式 ID 必须唯一；legacy 嵌套 skill 省略时由 workflow 按源位置生成稳定 `legacy:` ID。 |
| `depends_on` | `list[str]` | 可选，默认空列表；依赖的 skill id，若出现则必须是 list。 |

运行规则：workflow 对全部 skill 拓扑排序；依赖全部成功后节点进入 running；依赖 id 不存在或成环会报错。legacy 节点的隐式 phase/sequence 边只补给编译生成 ID 的节点，并在可能形成环时跳过；显式 DAG 依赖保持原语义。scene-4 基础任务的典型 5-skill DAG 形态见 AGENTS.md 与 `output/scene4_nav_skill_generation/scene4_nav_skill_generation_summary.json`。

---

# Part C. Arena YAML

## C1. 文档结构

```yaml
name: base_arena
fixtures:
  - name: floor
    target_class: PlaneObject
    size: [5.0, 5.0]
    translation: [0.0, 0.0, 0.0]
```

## C2. 顶层字段

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | 任意 arena 名。 | arena 标识。 |
| `fixtures` | `list[dict]` | 是 | 无 | 见 C3。 | 静态场景元素列表。 |
| `update_freq` | `int` | 否 | `10` | 当前 `5000`。 | `individual_reset()` 中 scene 随机化刷新周期。 |
| `involved_scenes` | `str` | 否 | 无 | `dining_room_scene_info` 等。 | scene-pair 随机化读取 `<name>.json`，逗号分隔多个前缀。 |

## C3. `fixtures[]`

fixture 用 object 注册表加载，字段与 task `objects[]` 基本一致；当前 arena 只用 `GeometryObject`、`PlaneObject`、`ConveyorObject`。

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | 任意唯一名。 | 可被 `regions[].target` 或 skill object 引用。 |
| `target_class` | `str` | 是 | 无 | `GeometryObject`、`PlaneObject`、`ConveyorObject`。 | 候选见 E2。 |
| `path` | `str` | 资源型 fixture 必填 | 无 | 几何或传送带 USD。 | 相对引用 task 的 `asset_root`。 |
| `translation` | `[x,y,z]` | 是 | 无 | 均显式配置。 | 初始位置。 |
| `euler` | `[r,p,y]` | `euler`/`quaternion` 二选一 | 无 | 常用。 | 初始朝向，degree。 |
| `quaternion` | `[w,x,y,z]` | `euler`/`quaternion` 二选一 | 无 | 少量。 | 初始朝向。 |
| `scale` | `[sx,sy,sz]` | 资源型常用 | `[1,1,1]` | 多数资源型显式配置。 | 缩放。 |
| `size` | `[width,length]` | `PlaneObject` 必填 | 无 | floor/background 使用。 | 平面大小。 |
| `texture` | `dict` | 否 | 无 | 见 A6 Texture Block。 | 加载后应用材质。 |
| `apply_randomization` | `bool` | 否 | `false` | 少量 `true`。 | scene-pair / hearth 随机化开关，不是通用位姿随机化。 |
| `visible` | `bool` | 否 | `true` | 少量 `false`。 | 初始可见性。 |
| `collision_enabled` | `bool` | 否 | class-specific | `true`、`false`。 | 生成/启用碰撞。 |
| `collision_thickness` | `float` | `PlaneObject` 可选 | `0.02` | 当前 `0.02`。 | 平面碰撞厚度。 |
| `collision_approximation` | `str` | `GeometryObject` 可选 | `bbox` | 当前 `bbox`。 | 只支持 `bbox`。 |
| `collision_visible` | `bool` | 否 | `false` | 少量。 | 是否显示碰撞 proxy。 |
| `linear_velocity` / `linear_track_list` | `list` | `ConveyorObject` 必填 | 无 | 2 个传送带 fixture。 | 线速度与线性轨道名。 |
| `angular_velocity` / `angular_track_list` | `list` | `ConveyorObject` 必填 | 无 | 2 个传送带 fixture。 | 角速度与角轨道名。 |

## C4. Scene 随机化

| 条件 | 行为 |
| --- | --- |
| arena 恰好 2 个 fixture 且都 `apply_randomization: true` | `update_scene_pair()` 读 `involved_scenes` JSON，替换 table/scene fixture 的 `path`、`target_class`、`scale`、`translation`、`euler`。 |
| task objects 含 `hearth` 且 arena 名为 `scene` 的 fixture `apply_randomization: true` | `update_hearths()` 从 `HEARTH_KITCHENS` 随机替换。 |

确定性 arena 不设 `involved_scenes`，也不给 fixture 设 `apply_randomization: true`。

---

# Part D. 未读取或元数据字段

以下字段在目标 YAML 中出现，但当前主链路不读取，或只作为生成器/语义元数据保留：

| 字段 | 位置 | 状态 | 说明 |
| --- | --- | --- | --- |
| `scene_prim_path` / `scene_reference` | `objects[]` | 元数据 | `assets_addition` 原始 scene 路径/reference。 |
| `filter_collision` | `objects[]` | 未读取 | 主加载链路不读取。 |
| `regions[].visible` | `regions[]` | 未读取 | 摆放逻辑不读取。 |
| `dexpick.post_grasp_offset` | `skills[]` | 未读取 | `Dexpick` 读 `post_grasp_offset_min/max`。 |
| `flip.ee_axis` | `skills[]` | 未读取 | `Flip` 读 `gripper_axis`。 |
| `gripper__action.post_action` / `post_action_offset` | `skills[]` | 未读取 | `Gripper_Action` 不读取。 |
| `manualpick.update_pose_cost_metric_none` | `skills[]` | 未读取 | `Manualpick` 不读取。 |
| `pour__water__succ.gripper` | `skills[]` | 未读取 | `Pour_Water_Succ` 不读取。 |
| `track.target` | `skills[]` | 语义字段 | 主轨迹生成不使用。 |
| `camera_file.camera_type` / `frequency` | camera config | 元数据 | `CustomCamera` 不读取。 |
| `startup_timeout_sec` | 导航 task | 历史字段 | `navigate.py` / `local_navigation.py` 已无消费，仅残留在少量导航 task YAML。 |

---

# Part E. 候选值索引

对象名、fixture 名、region 名、skill `id`、`depends_on`、`positions` key、`scene_name` 属于用户自定义标识符，不在此当作枚举；约束是“必须能在当前 task 的命名表里解析到”。

## E1. 配置文件路径候选值

### `robots[].robot_config_file`

| 当前出现值 | 实例名 | 说明 |
| --- | --- | --- |
| `workflows/simbox/core/configs/robots/genie1.yaml` | `genie1` | Genie1 基础配置。 |
| `workflows/simbox/core/configs/robots/lift2.yaml` | `lift2`、`lift2_0`、`lift2_1` | Lift2 基础配置，可多实例化。 |
| `workflows/simbox/core/configs/robots/split_aloha.yaml` | `split_aloha` | Split Aloha 基础配置，常带移动底盘。 |
| `workflows/simbox/core/configs/robots/fr3.yaml` | `franka` | FR3 / Franka 机械臂。 |
| `workflows/simbox/core/configs/robots/franka_robotiq85.yaml` | `frankarobotiq` | Franka + Robotiq85。 |

同目录还有 `panda_omron.yaml`、`panda_omron_virtual.yaml`、`split_aloha_actual.yaml`、`tracer2_franka.yaml`，当前目标 task YAML 未出现，供移动操作类配置扩展。

### 机器人 `target_class`

`target_class` 通常来自 `robot_config_file`，决定创建的 robot Python 类。当前注册类：

| 值 | 关键字段 |
| --- | --- |
| `SplitAloha` | 双臂，`left`/`right` controller，可通过 `base` 用移动底盘。 |
| `Lift2` | 多实例化（`lift2`、`lift2_0`、`lift2_1`）。 |
| `Genie1` | Genie1 资产与对应关节/home/gripper。 |
| `FR3` | Franka Research 3，实例名常写 `franka`。 |
| `FrankaRobotiq85` | Franka + Robotiq85。 |
| `TemplateRobot` | 模板类，供新 robot 扩展，目标 YAML 未直接使用。 |
| `PandaOmron` | Panda 臂 + Omron 差速底盘（`panda_omron.yaml`）。`left_joint_indices: [14..20]`、`left_gripper_indices: [21]`、`gripper_max_width: 0.08`、`tcp_offset: 0.1034`；含 `mobile_base_path`、`fl_ee_path`、`fl_base_path`、`fl_base_mount_translation: [0,0,0.7]`、`fl_gripper_keypoints`、`fl_filter_paths`、`fl_forbid_collision_paths`；`base_config_file` 指向 `bases/omron_diff_drive.yaml`，`local_navigation_config_file` 指向 `navigation/default_local_navigation.yaml`（`map_z_max: 1.25`）。 |
| `PandaOmronVirtual` | Panda 臂 + Omron 虚拟底盘（`panda_omron_virtual.yaml`）。结构与 `PandaOmron` 相同；`left_joint_indices: [3..9]`、`left_gripper_indices: [10]`；`base_config_file` 指向 `bases/omron_virtual_base.yaml`；虚拟轮 `mobilebase0_joint_mobile_forward/side/yaw`，含 `base_velocity_joint_names`、`base_velocity_command_signs`。 |
| `SplitAlohaActual` | 物理 4WIS Split Aloha（`split_aloha_actual.yaml`，标记 `deprecated: true`）。`left_joint_indices: [1,8,15,18,20,22]`、`right_joint_indices: [2,9,16,19,21,23]`、`left_gripper_indices: [24]`、`right_gripper_indices: [26]`；`base_config_file` 指向 `bases/ranger_mini_v3.yaml`；`steering_joint_names: [fl_steering_joint, fr_steering_joint, rl_steering_joint, rr_steering_joint]`、`wheel_joint_names: [fl_wheel, fr_wheel, rl_wheel, rr_wheel]`。 |
| `Tracer2Franka` | Tracer 2 底盘 + Franka 臂（`tracer2_franka.yaml`）。`robot_type: mobile_manipulator`、`tcp_offset: 0.1043`、`body_indices: [0,1,2]`、`left_joint_indices: [3..9]`、`left_gripper_indices: [10,11]`、`body_home: [0,0,0]`；含 `mobile_base_path`、`fl_ee_path`、`fl_base_path`、`fl_gripper_keypoints`。 |

### `base_config_file` 候选值

| 完整值 | 说明 |
| --- | --- |
| `workflows/simbox/core/configs/bases/omron_virtual_base.yaml` | `platform.profile: virtual_base`；`local_navigation.settle`（`linear_speed_tolerance: 0.005`、`angular_speed_tolerance: 0.005`、`consecutive_steps: 8`）、`footprint_points`、`controller_hard_limits`、`inflation_radius_m: 0.34`。 |
| `workflows/simbox/core/configs/bases/omron_diff_drive.yaml` | `platform.profile: differential_drive`、`wheel_radius: 0.085`、`footprint_points`、`controller_hard_limits`。 |
| `workflows/simbox/core/configs/bases/ranger_mini_v3.yaml` | `platform.profile: ranger_mini_v3`、`ackermann_split_steering: true`、`ackermann_split_wheel_speeds: true`、`footprint_points`。 |

本地导航默认配置：`workflows/simbox/core/configs/navigation/default_local_navigation.yaml`（`runtime_timeout_sec: 180.0`、`planner.*`、`controller.*`、`map.*`；`waypoint_tolerance_m: 0.25`、`position_tolerance_m: 0.10`、`yaw_tolerance_rad: 0.10`）。

### `cameras[].camera_file`

| 当前出现值 | 说明 |
| --- | --- |
| `workflows/simbox/core/configs/cameras/realsense_d455_v3.yaml` | D455 v3，使用最多。 |
| `workflows/simbox/core/configs/cameras/realsense_d405_v2.yaml` / `realsense_d405.yaml` | D405。 |
| `workflows/simbox/core/configs/cameras/astra.yaml` | Astra。 |
| `workflows/simbox/core/configs/cameras/realsense_d435i.yaml` | D435i。 |

## E2. Object / Arena 类名候选值

| 值 | 可用于 | 说明 |
| --- | --- | --- |
| `RigidObject` | `objects[].target_class`，代码支持 fixture | 把 `prim_path_child` 包装成 `RigidPrim`，具刚体/质量/碰撞语义。 |
| `GeometryObject` | `objects[].target_class`、`fixtures[].target_class` | 静态几何，可选 bbox 碰撞代理。 |
| `ArticulatedObject` | `objects[].target_class`，代码支持 fixture | 读 `Kps/<info_name>/info.json`，支持 `open`/`close`/`artpreplan`。 |
| `PlaneObject` | `fixtures[].target_class`，代码支持 task object | 程序生成平面。 |
| `ConveyorObject` | `fixtures[].target_class`，代码支持 task object | 传送带，轨道加线/角速度。 |
| `BoxObject` / `ShapeObject` / `XFormObject` | 代码已注册 | 程序 cube / 低层 GeometryPrim / xform 容器；目标 YAML 未出现。 |

资源元信息：`texture_lib` 候选如 `background_textures`、`floor_textures`、`table_textures`、`val2017`、`light_table_textures`、`dark_table_textures`；`objects[].dataset` 候选 `assets_addition`、`oo3d`、`gso`、`pm`、`arcode`、`grutopia`、`gr`；`prim_path_child` 常见 `Aligned`、`root`。

## E3. 随机化候选值

| 字段 | 值 | 语义 |
| --- | --- | --- |
| `randomization_scope` | `category` | 当前对象所在类别目录内随机，保持语义类别不变。 |
| `randomization_scope` | `full` | 资源库全类别随机，随机后 `category` 被替换为新类别名。 |
| `randomization_scope` | 类别列表 | 只在列表目录内随机，元素必须是实际资产目录名（如 `omniobject3d-bottle`、`phocal-spoon`）。 |
| `orientation_mode` | `keep` | 保留配置 `euler`；缺失会触发断言。 |
| `orientation_mode` | `suggested` | 按类别查推荐姿态，未命中回退 `[0,0,0]` 并 warning。 |
| `orientation_mode` | `random` | `[-180,180]` degree 内独立采样 roll/pitch/yaw。 |
| `scale_mode` | `keep` | 保留配置 `scale`；缺失触发断言。 |
| `scale_mode` | `suggested` | 类别推荐缩放，对象级覆盖类别级。 |

## E4. Region Sampler 候选值

| `random_type` | 必要参数 | 说明 |
| --- | --- | --- |
| `A_on_B_region_sampler` | `pos_range`、`yaw_rotation` | A 放 B 上。 |
| `A_in_B_region_sampler` | `x_bias`/`y_bias`/`z_bias` 可选 | A 放 B 中心上沿。 |
| `A_by_B_region_sampler` | `pos_range`、`yaw_rotation` | A 放 B 旁矩形区域。 |
| `A_by_B_circle_sampler` | `r_range`、`theta_range`、`yaw_rotation` | A 放 B 周围圆环/扇形。 |
| `A_face_B_circle_sampler` | `r_range`、`yaw_rotation` | A 放 B 正面方向射线上。 |
| `A_along_B_C_circle_sampler` | 外层 `target2`、`r_range`、`yaw_rotation` | A 沿 B 朝向放于越过 C 之后。 |

各 sampler 的完整计算方式见 A7。

---

# Part F. 最小完整示例

## F1. Arena

```yaml
name: my_floor_arena
fixtures:
  - name: floor
    target_class: PlaneObject
    size: [5.0, 5.0]
    translation: [0.0, 0.0, 0.0]
    euler: [0.0, 0.0, 0.0]
    collision_enabled: true
    collision_thickness: 0.02
```

## F2. Scene-only Task

```yaml
tasks:
  - name: my_scene_only_task
    asset_root: workflows/simbox/assets
    task: BananaBaseTask
    task_id: 0
    offset: null
    render: false
    arena_file: workflows/simbox/core/configs/arenas/my_floor_arena.yaml

    env_map:
      envmap_lib: envmap_lib
      apply_randomization: false
      intensity_range: [5000, 5000]
      rotation_range: [0, 0]

    robots: []
    objects: []
    regions: []
    cameras: []
    skills: []

    data:
      task_dir: debug/my_scene_only_task
      language_instruction: Load the scene.
      detailed_language_instruction: Load the scene.
      collect_info: my_scene_only_task
      version: v1.0
      update: true
      max_episode_length: 100
```

## F3. Pick Object On Table

```yaml
objects:
  - name: pick_object
    target_class: RigidObject
    path: pick_and_place/pre-train-pick/assets/omniobject3d-cup/example/Aligned_obj.usd
    prim_path_child: Aligned
    translation: [0.0, 0.0, 0.0]
    euler: [0.0, 0.0, 0.0]
    scale: [1.0, 1.0, 1.0]
    category: cup
    dataset: oo3d
    apply_randomization: false

regions:
  - object: pick_object
    target: table
    random_type: A_on_B_region_sampler
    random_config:
      pos_range:
        - [-0.1, -0.1, 0.0]
        - [0.1, 0.1, 0.0]
      yaw_rotation: [-180.0, 180.0]
```
