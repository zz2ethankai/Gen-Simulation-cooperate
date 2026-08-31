# SimBox Arena / Task YAML API Reference

本文是 SimBox `arena` 与 `task` YAML 的 API 参考手册。它按 Wiki/API 文档风格组织：先给出数据模型和加载流程，再逐项列出字段的类型、必填性、默认值、允许值、当前配置中观测到的值、运行语义、使用方式和注意事项。

## 适用范围

本文覆盖以下 YAML：

| 范围 | 说明 |
| --- | --- |
| `workflows/simbox/core/configs/arenas/**/*.yaml` | Arena YAML。 |
| `workflows/simbox/core/configs/tasks/**/*.yaml` | Task YAML。 |
| 排除 `download` 子目录 | 包括 `workflows/simbox/core/configs/arenas/download` 和 `workflows/simbox/core/configs/tasks/download`。 |

当前统计结果：

| 类型 | 文件数 | 说明 |
| --- | ---: | --- |
| Arena YAML | 31 | 顶层为 arena object。 |
| Task YAML | 1145 | 顶层为 `tasks: [...]`。 |
| 实际出现的唯一 key | 295 | 本文逐字覆盖全部 key。 |
| 实际出现的 skill 注册名 | 23 | 本文逐项覆盖全部 skill。 |

## 术语和约定

### 路径

| 路径类型 | 示例 | 解析方式 |
| --- | --- | --- |
| 配置文件路径 | `workflows/simbox/core/configs/arenas/base_arena.yaml` | 相对仓库根目录或当前运行目录读取。 |
| 资源路径 | `table0/instance.usd` | 相对 task 的 `asset_root` 拼接。 |
| texture 路径 | `floor_textures` | 相对 task 的 `asset_root`，代码会读取目录内文件。 |
| USD prim path | `/World/Objects/foo` 或 `root` | USD stage 内部路径或子 prim 名。 |

### 位姿

| 字段 | 格式 | 单位 / 约定 |
| --- | --- | --- |
| `translation` | `[x, y, z]` | 米。 |
| `euler` | `[roll, pitch, yaw]` | 角度制 degree。 |
| `quaternion` / `orientation` | `[w, x, y, z]` | scalar-first。 |
| `scale` | `[sx, sy, sz]` | 无量纲缩放。 |

### 名称引用

以下字段不是固定枚举，而是“命名引用”：

| 字段 | 必须引用什么 |
| --- | --- |
| `regions[].object` | 当前 task 已加载的 object、fixture、robot 或 camera 名。 |
| `regions[].target` | 当前 task 已加载的 object、fixture、robot 或 camera 名。 |
| `skills[].<robot_name>` | `robots[].name`。 |
| `skills[].<controller_name>` | 当前 robot 可建立的 controller 名，常见为 `left`、`right`、`base`。 |
| `skills[].*.objects[]` | skill 需要的 task object 或 arena fixture 名。 |
| `navigate.goal` | task 顶层 `positions` 中的 key。 |

## 加载流程

### Task 解析

入口：`workflows/simbox/utils/task_config_parser.py::TaskConfigParser.parse_tasks`

1. 使用 `OmegaConf.load(task_cfg_path)` 读取 Task YAML。
2. 要求顶层存在 `tasks`。
3. 遍历 `tasks[]`，把每个 task 转成普通 dict / list。
4. 返回 task cfg 列表。

### Workflow 合并

入口：`workflows/simbox_dual_workflow.py::SimBoxDualWorkFlow`

1. `_merge_robot_configs()` 读取 `robots[].robot_config_file` 并合并。
2. `_merge_base_configs()` 读取 `base.base_config_file` 和 `base.nav_config_file` 并递归合并。
3. `reset()` 读取 `arena_file` 指向的 Arena YAML。
4. arena 内容写入 `task_cfg["arena"]`。
5. 删除 `arena_file`、`camera_file`、`logger_file` 等运行前引用字段。
6. 创建 `BananaBaseTask`，加载 arena fixtures、objects、robots、cameras、regions、skills。

### 未知字段

解析器不会删除未知字段。未知字段只有在后续代码显式读取时才有运行语义。本文在“未读取字段”章节列出当前 YAML 中存在但主链路未读取的字段。

---

# Part A. Task YAML

## A1. 文档结构

Task YAML 顶层必须是 mapping，且必须包含 `tasks`：

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

### Schema

```yaml
tasks:
  - <task object>
```

### Reference

| 属性 | 内容 |
| --- | --- |
| 类型 | `list[dict]` |
| 必填 | 是 |
| 默认值 | 无 |
| 允许值 | list 中每个元素必须是一个 task object。 |
| 当前观测 | 1145 个 task YAML 均包含 `tasks`，且每个文件中当前均有 1 个 task。 |
| 消费代码 | `TaskConfigParser.parse_tasks()` |
| 使用方式 | 一个文件可放多个 task；workflow 会逐个返回。当前仓库基本使用单 task 文件。 |

---

## A3. Task Object

### Schema

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
random_region_list: [<RandomRegion>, ...] # optional
positions: {<name>: {x, y, yaw}}           # optional
distractors: <Distractors>                 # optional
fluid: <Fluid>                             # optional
```

### Required Fields

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 运行语义 |
| --- | --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | 任意非空字符串。 | task 实例名，传给 `BaseTask(name=...)`，也用于日志和语言上下文。 |
| `asset_root` | `str` | 是 | 无 | 当前全部为 `workflows/simbox/assets`。 | 所有 object、fixture、texture、distractor、envmap 资源的根目录。 |
| `task` | `str` | 是 | 无 | 代码注册表值；当前全部为 `BananaBaseTask`。 | `get_task_cls(task)` 取任务类。写错会 KeyError。 |
| `task_id` | `int` | 是 | 无 | 当前全部为 `0`。 | 构造 USD root prim：`/World/task_<task_id>`。 |
| `offset` | `null` 或 `list[float]` | 是 | 无 | 当前全部为 `null`。 | 传给 Isaac `BaseTask` 的 offset。当前任务类没有额外解释。 |
| `render` | `bool` | 是 | 无 | 当前 1144 个 `true`，1 个 `false`。 | 为 `false` 时 `get_observations()` 不读取 cameras。 |
| `arena_file` | `str` | 是 | 无 | 指向 Arena YAML。 | workflow reset 时读取，读取后从 task cfg 删除。 |
| `env_map` | `dict` | 是 | 无 | 见 A4。 | 配置 DomeLight 环境光。 |
| `robots` | `list[dict]` | 是 | 无 | 可为空。 | 机器人实例列表。 |
| `objects` | `list[dict]` | 是 | 无 | 可为空。 | 任务物体列表。 |
| `regions` | `list[dict]` | 是 | 无 | 可为空。 | 初始摆放/随机摆放规则。 |
| `cameras` | `list[dict]` | 是 | 无 | 可为空。 | 相机实例列表。 |
| `skills` | `list[dict]` | 是 | 无 | 可为空。 | 技能执行流程。 |
| `data` | `dict` | 是 | 无 | 见 A11。 | logger、语言、episode 长度等配置。 |

### Optional Fields

| 字段 | 类型 | 默认值 | 允许值 / 当前观测 | 运行语义 |
| --- | --- | --- | --- | --- |
| `neglect_collision_names` | `list[str]` | `[]` | 当前常见 `["table"]`。 | workflow 构建 collision filter 时，名称包含这些子串的 object / fixture 会被加入 `prim_paths`，从 global collision 集合移除。 |
| `random_region_list` | `list[dict]` | `[]` | 当前用于部分任务的预设随机区域池。 | 仅当 `regions[].priority` 存在时使用。 |
| `positions` | `dict[str, {x, y, yaw}]` | 无 | 当前有 `nav_to_pick`、`nav_to_place`、`nav_to_living_room_east`。 | `navigate.goal` 的命名目标表。 |
| `distractors` | `dict` | 无 | 当前大量 pick/place 任务使用。 | 生成并摆放视觉干扰物。 |
| `fluid` | `dict` | 无 | 当前倒水/浇花任务使用。 | 创建 PhysX 粒子流体。存在时 workflow 强制 GPU physics。 |

### Usage Notes

- scene-only task 可以把 `robots`、`objects`、`regions`、`cameras`、`skills` 都写为空列表。
- `arena_file` 指向的 arena 不能独立解析资源；arena fixture 的 `path` 仍然依赖当前 task 的 `asset_root`。
- `positions` 的 key 是开放命名；只要 `navigate.goal` 与它一致即可。

---

## A4. `env_map`

环境光配置。由 `BananaBaseTask._set_envmap()` 消费。

### Schema

```yaml
env_map:
  envmap_lib: envmap_lib
  apply_randomization: true
  intensity_range: [5000, 5000]
  rotation_range: [0, 0]
  light_type: DomeLight  # optional
```

### Fields

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `envmap_lib` | `str` | 是 | 无 | 当前全部为 `envmap_lib`。 | 相对 `asset_root` 的 HDR 文件目录。代码读取 `asset_root/envmap_lib/*.hdr` 并排序。 |
| `apply_randomization` | `bool` | 是 | `false` 分支由代码处理 | 当前 1128 个 `true`，17 个 `false`。 | `true` 时随机选择 HDR、随机采样 intensity 和 rotation；`false` 时固定使用第 0 个 HDR、强度 `1000.0`、旋转 `[0,0,0]`。 |
| `intensity_range` | `[min, max]` | 是 | `apply_randomization: false` 时不参与计算 | 当前通常 `[5000, 5000]`。 | 随机模式下调用 `random.uniform(min, max)`。两个值相同表示固定强度。 |
| `rotation_range` | `[min, max]` | 是 | `apply_randomization: false` 时不参与计算 | 当前通常 `[0, 0]`。 | 随机模式下 x/y/z 三轴分别从同一范围采样，单位 degree。 |
| `light_type` | `str` | 否 | `DomeLight` | 当前代码仅实现 `DomeLight`。 | `DomeLight` 表示在 task root 下创建/复用 USD DomeLight，用 HDR env map 作为环境贴图并设置强度和旋转。其他值不会进入环境光创建分支，等同于本段配置不生效。 |

### Example

```yaml
env_map:
  envmap_lib: envmap_lib
  apply_randomization: false
  intensity_range: [5000, 5000]
  rotation_range: [0, 0]
```

---

## A5. `robots[]`

机器人实例配置。task 内机器人配置会与 `robot_config_file` 指向的机器人基础配置合并。

### Schema

```yaml
robots:
  - name: split_aloha
    robot_config_file: workflows/simbox/core/configs/robots/split_aloha.yaml
    translation: [0.0, 0.0, 0.0]
    euler: [0.0, 0.0, 90.0]
    use_batch: true
```

### Fields

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | 当前观测：`split_aloha`、`lift2`、`lift2_0`、`lift2_1`、`genie1`、`franka`、`frankarobotiq`。 | 机器人实例名。skills 外层 robot key 必须匹配它。 |
| `robot_config_file` | `str` | 常用 | 无 | 当前观测完整路径见 E1。 | 外部机器人配置路径。workflow 先读它，再用 task 内字段覆盖；task 中同名字段优先级更高。 |
| `target_class` | `str` | 合并后必填 | 无 | 注册类：`SplitAloha`、`Lift2`、`Genie1`、`FR3`、`FrankaRobotiq85`、`TemplateRobot`。 | 机器人类注册名，传给 `get_robot_cls()` 创建实例。通常来自 `robot_config_file`；task 内只有在需要覆盖基础机器人类型时才直接写。 |
| `path` | `str` | 合并后必填 | 无 | 资源路径，例如 `split_aloha_mid_360/robot.usd`。 | 相对 `asset_root` 的机器人 USD。 |
| `translation` | `[x,y,z]` | 是 | `[0,0,0]` 仅在局部代码 fallback | 当前均显式配置。 | robot root prim 初始位置。 |
| `euler` | `[r,p,y]` | `euler` / `quaternion` 二选一 | 无 | 当前大量使用。 | robot root prim 初始朝向，degree。 |
| `quaternion` | `[w,x,y,z]` | `euler` / `quaternion` 二选一 | 无 | 当前 robot YAML 中少见。 | robot root prim 初始朝向。若同时写 `euler`，`get_orientation()` 优先走 `euler`。 |
| `scale` | `[sx,sy,sz]` | 否 | `[1,1,1]` | 当前很少写。 | robot root prim 缩放。 |
| `robot_file` | `str` 或 `list[str]` | 需要操作类 skill 时必填 | 无 | 基础 robot YAML 中通常提供左右 CuRobo 文件。 | controller 配置路径。列表中包含 `left` 的路径映射到 `left` controller，包含 `right` 的路径映射到 `right` controller。 |
| `constrain_grasp_approach` | `bool` | 否 | `false` | 当前少量任务为 `true`。 | 传给 controller，用于约束抓取接近方向。 |
| `collision_activation_distance` | `float` | 否 | `0.03` | 当前常见 `0.05`。 | 传给 controller 的碰撞激活距离。 |
| `ignore_substring` | `list[str]` | 否 | `["material","Plane","conveyor","scene","table"]` | 任意字符串列表。当前 YAML 中还出现 `material`、`table`、`plate` 等业务名。 | controller 规划时忽略名称含这些子串的 prim；匹配是子串匹配，不要求完整 prim 名一致。 |
| `use_batch` | `bool` | 否 | `false` | 当前部分任务为 `true`。 | 是否使用 batch 规划。 |
| `left_joint_home` / `right_joint_home` | `list[float]` | 合并后必填 | 无 | 基础 robot YAML 提供。 | 左右臂 home 关节位。 |
| `left_joint_home_std` / `right_joint_home_std` | `list[float]` | 否 | 长度匹配的 0 列表 | 当前大量显式配置。 | reset/home 时使用的关节噪声标准差。 |
| `left_gripper_home` / `right_gripper_home` | `list[float]` | 合并后必填 | 无 | 基础 robot YAML 提供。 | home 时夹爪关节位置。 |
| `tcp_offset` | `float` | 合并后常用 | robot YAML 内定义 | 当前有 `0.115`、`0.12`、`0.135` 等。 | pick 类 skill 默认 TCP 偏移。 |
| `base` | `dict` | 移动底盘可选 | 无 | `SplitAloha` 基础配置含该字段。 | 移动底盘和 Nav2 配置，见下文。 |

### `base`

`base` 先读取 `base_config_file`，再读取 `nav_config_file`，最后由 task 内 `base` 覆盖。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `base_config_file` | `str` | 无 | 移动底盘基础参数 YAML，例如 `configs/bases/ranger_mini_v3.yaml`。 |
| `nav_config_file` | `str` | 无 | Nav2 参数 YAML，例如 `nav2/config/default_nav.yaml`。 |
| `steering_joint_names` | `list[str]` | 无 | 底盘转向关节名。 |
| `wheel_joint_names` | `list[str]` | 无 | 车轮关节名。 |
| `wheel_base` | `float` | 基础配置提供 | 轴距。 |
| `track_width` | `float` | 基础配置提供 | 轮距。 |
| `wheel_radius` | `float` | 基础配置提供 | 轮半径。 |
| `steering_limit` | `float` | 基础配置提供 | 最大转向角。 |
| `wheel_velocity_limit` | `float` | 基础配置提供 | 车轮速度上限。 |
| `ros` | `dict` | 基础配置提供 | ROS topic、frame、Nav2 runtime 配置。 |
| `nav2_skill` | `dict` | nav 配置提供 | planner/controller/costmap 等 Nav2 参数。 |

---

## A6. `objects[]`

任务物体。由 `_load_obj()` 根据 `target_class` 分发到对象类。

### Common Schema

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

### Common Fields

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | 任意唯一名称。 | 对象实例名。regions 和 skills 通过它引用。 |
| `target_class` | `str` | 是 | 无 | 当前 task objects：`GeometryObject`、`RigidObject`、`ArticulatedObject`。注册表另支持 `PlaneObject`、`ConveyorObject`、`BoxObject`、`ShapeObject`、`XFormObject`。 | 对象类注册名，决定 `_load_obj()` 创建哪种对象包装、需要哪些字段、是否具备刚体/articulation/程序几何语义；每个值的含义见 E2。 |
| `path` | `str` | 资源型对象必填 | 无 | USD 路径，相对 `asset_root`。 | `RigidObject`、`GeometryObject`、`ArticulatedObject`、`ConveyorObject`、`XFormObject` 需要。 |
| `translation` | `[x,y,z]` | 常用 | 无 | 当前多数显式配置。 | 初始位置；之后可能被 `regions` 覆盖。 |
| `euler` | `[r,p,y]` | `euler` / `quaternion` 二选一 | 无 | 当前多数显式配置。 | 初始朝向，degree。 |
| `quaternion` | `[w,x,y,z]` | `euler` / `quaternion` 二选一 | 无 | 当前少量对象使用。 | 初始朝向。 |
| `scale` | `[sx,sy,sz]` | 否 | `[1,1,1]` | 当前几乎都显式配置。 | 初始缩放。 |
| `visible` | `bool` | 否 | `true` | 当前少量隐藏对象使用。 | 初始可见性。 |
| `category` | `str` | 语义任务常用 | 无 | 例如 `bottle`、`box`、`assets_addition` 资源类别。 | 语言替换、随机化、distractor 排除时使用。 |
| `dataset` | `str` | 否 | 无 | 当前观测：`assets_addition`、`oo3d`、`gso`、`pm`、`arcode`、`grutopia`、`gr`。 | 数据集元信息；主加载链路不按该字段分支。 |
| `apply_randomization` | `bool` | 否 | `false` | 当前大量为 `true` 或 `false`。 | 对 `RigidObject` 和 `ArticulatedObject` 启用资源随机化。 |
| `texture` | `dict` | 否 | 无 | 见 A6.5。 | 加载后调用 `apply_texture()`。 |
| `mass` | `float` | 否 | `None` | 少量刚体使用。 | 传给 Isaac `RigidPrim`。 |
| `color` | `[r,g,b]` | 否 | 对 ShapeObject 默认 `[1,0,0]` | 目标 YAML 中少见。 | 程序几何颜色。 |
| `filter_collision` | `bool` | 否 | 无 | 当前 4 处为 `true`。 | 当前主加载链路未读取。 |
| `scene_prim_path` | `str` | 否 | 无 | `assets_addition` 任务大量出现。 | 原始 scene USD prim path 元数据；主加载链路不读取。 |
| `scene_reference` | `str` | 否 | 无 | `assets_addition` 任务大量出现。 | 原始 scene reference 元数据；主加载链路不读取。 |
| `optimize_2d_layout` | `bool` | 否 | 无 | 当前部分 task 使用。 | 供布局优化辅助逻辑使用。 |

### Target Classes

#### `RigidObject`

刚体物体。构造函数：`workflows/simbox/core/objects/rigid_object.py::RigidObject`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | 对象名。 |
| `path` | `str` | 是 | 无 | 相对 `asset_root` 的 USD。 |
| `prim_path_child` | `str` | 是 | 无 | USD 中实际刚体 prim 的子路径。当前观测值主要为 `Aligned`，少量为 `root`。 |
| `init_translation` | `[x,y,z]` | 否 | 无 | 若同时存在 `init_orientation` 和 `init_parent`，workflow `_init_static_objects()` 会把对象放到 `init_parent` 相对位置。 |
| `init_orientation` | quaternion | 否 | 无 | 与 `init_translation` 一起使用。 |
| `init_parent` | `str` | 否 | 无 | 相对 task root 的父 prim 路径，可包含 OmegaConf 插值。 |
| `gap` | 任意 | 否 | 无 | 容器摆放时可作为间隙候选。随机化时可由 `gap.yaml` 自动写入。 |
| `mass` | `float` | 否 | `None` | 传给 `RigidPrim`。 |

使用方式：

```yaml
- name: cup
  target_class: RigidObject
  path: pick_and_place/pre-train-pick/assets/omniobject3d-cup/foo/Aligned_obj.usd
  prim_path_child: Aligned
  translation: [0.0, 0.0, 0.0]
  euler: [0.0, 0.0, 0.0]
  scale: [1.0, 1.0, 1.0]
```

#### `GeometryObject`

静态几何或可碰撞几何。构造函数：`workflows/simbox/core/objects/geometry_object.py::GeometryObject`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | 对象名。 |
| `path` | `str` | 是 | 无 | 相对 `asset_root` 的 USD。 |
| `prim_path_child` | `str` | 否 | 无 | 若提供，最终 prim path 会拼到对象 root 下。 |
| `collision_enabled` | `bool` | 否 | `false` | 是否生成 bbox collision proxy。 |
| `collision_approximation` | `str` | 否 | `bbox` | 当前只允许 `bbox`；其他值会抛 `ValueError`。 |
| `collision_visible` | `bool` | 否 | `false` | 是否显示 collision proxy。 |

#### `ArticulatedObject`

带关节物体。构造函数：`workflows/simbox/core/objects/articulated_object.py::ArticulatedObject`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | 对象名。 |
| `path` | `str` | 是 | 无 | 必须指向 `instance.usd`，代码会把它替换成 `Kps/<info_name>/info.json` 读取关键点信息。 |
| `info_name` | `str` | 是 | 无 | articulation 关键点信息目录名，例如 `open_v_inner`、`close_h_left`。 |
| `category` | `str` | 是 | 无 | articulation 类别。 |
| `fix_base` | `bool` | 否 | `false` | 若 true，创建 FixedJoint 固定 base。 |
| `joint_position_range` | `[min,max]` | 否 | 无 | 初始化时随机采样 articulation joint position。 |
| `strict_init.joint_positions` | `list[float]` | 否 | 无 | 严格设置目标关节位置。 |
| `strict_init.joint_indices` | `list[int]` | 否 | 无 | 严格设置目标关节索引。 |
| `art_cat` | `str` | 随机化时需要 | 无 | workflow reset 中 articulation 随机资产目录。 |
| `obj_info_path` | `str` | 随机化后写入 | 无 | articulation 信息文件相对路径。skill 也可用它覆盖。 |

#### `PlaneObject`

程序生成平面。当前目标 task objects 未使用，但 arena fixtures 使用。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | 平面名。 |
| `size` | `[width,length]` | 是 | 无 | 平面宽和长。 |
| `collision_enabled` | `bool` | 否 | `false` | 是否添加薄 box 碰撞体。 |
| `collision_thickness` | `float` | 否 | `0.02` | 碰撞体厚度；必须大于 0。 |
| `collision_visible` | `bool` | 否 | `false` | 是否显示碰撞体。 |

#### `ConveyorObject`

传送带对象。当前只在 arena fixtures 中出现。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `path` | `str` | 是 | 无 | 传送带 USD。 |
| `linear_velocity` | `[vx,vy,vz]` | 是 | 无 | 线速度。 |
| `linear_track_list` | `list[str]` | 是 | 无 | 线性轨道名。代码查找 `<conveyor>/World/<track>/node_`。 |
| `angular_velocity` | `[wx,wy,wz]` | 是 | 无 | 角速度。 |
| `angular_track_list` | `list[str]` | 是 | 无 | 角轨道名。代码查找 `<conveyor>/World/<track>/validate_obj`。 |

#### `BoxObject`, `ShapeObject`, `XFormObject`

这些类型已注册但当前目标 YAML 未使用。

| 类型 | 必填字段 | 主要用途 |
| --- | --- | --- |
| `BoxObject` | `name`, `target_class` | 程序生成 cube，可用 `scale` 控制大小，可选 `color`、`collision_enabled`。 |
| `ShapeObject` | `name`, `target_class` | 低层 `GeometryPrim` 包装；当前没有完整 shape 创建逻辑。 |
| `XFormObject` | `name`, `target_class`, `path` | 加载 USD 为 xform；可选 `parent_obj`。 |

### Randomization Fields

| 字段 | 类型 | 允许值 / 当前观测 | 说明 |
| --- | --- | --- | --- |
| `apply_randomization` | `bool` | `true` / `false` | 总开关。只有对应对象类型的随机化代码会读取。 |
| `randomization_scope` | `str` 或 `list[str]` | 允许 `category`、`full` 或类别列表；当前 YAML 使用的类别列表见 E3。 | `RigidObject` 随机采样资源范围。`category` 表示当前对象所在类别内随机，`full` 表示当前资源库全类别随机，list 表示只从列出的类别目录随机。 |
| `orientation_mode` | `str` | `keep`、`suggested`、`random`；当前观测三者都有。 | 随机化后姿态策略。`keep` 保留配置中的 `euler`；`suggested` 按类别查推荐欧拉角；`random` 在 `[-180, 180]` degree 中独立采样 roll/pitch/yaw。 |
| `scale_mode` | `str` | `keep`、`suggested`；当前 YAML 只观测到 `suggested`。 | 随机化后缩放策略。`keep` 保留配置中的 `scale`；`suggested` 先按类别查推荐缩放，再用对象级推荐缩放覆盖。 |
| `gap` | 任意 | 当前有 `false` 和 list。 | 容器/间隙相关元数据。 |

### Texture Block

```yaml
texture:
  texture_lib: floor_textures
  apply_randomization: true
  texture_id: 1
  texture_scale: [1.0, 1.0]
```

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `texture_lib` | `str` | 是 | 无 | 当前 fixture 观测：`background_textures`、`floor_textures`、`table_textures`、`val2017`、`light_table_textures`、`dark_table_textures`。 | 相对 `asset_root` 的纹理目录。 |
| `apply_randomization` | `bool` | 是 | 无 | 当前 fixture 多数为 `true`，少量 `false`。 | true 时随机选择目录内文件；false 时使用 `texture_id`。 |
| `texture_id` | `int` | 非随机必需 | 无 | 当前常见 0、1、2。 | 排序后的文件索引。越界会在取 list 时失败。 |
| `texture_scale` | `list[float]` 或 `float` | 否 | `None` | 当前常见长度 2 list。 | 传给 OmniPBR。 |
| `target_prim_path` | `str` | 否 | 无 | 仅 `GeometryObject.apply_texture()` 支持。 | 从指定子 prim 开始递归绑定材质。 |

---

## A7. `regions[]`

用于设置对象初始摆放位置。由 `BananaBaseTask._set_regions()` 和 `RandomRegionSampler` 消费。

### Schema

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

### Fields

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `object` | `str` | 是 | 无 | 任意已加载 task object / fixture / robot / camera 名。 | 被移动和设置 pose 的对象。 |
| `target` | `str` | 是 | 无 | 任意已加载 task object / fixture / robot / camera 名。 | 采样参考对象。 |
| `random_type` | `str` | 普通分支必填 | 无 | 当前观测：`A_on_B_region_sampler`、`A_in_B_region_sampler`；完整候选值见下文和 E5。 | 选择 region 采样算法。它决定 object 的最终 `translation` 如何相对 target 计算，以及是否在原始姿态上追加随机 yaw。 |
| `random_config` | `dict` | 普通分支必填 | 无 | 根据 `random_type` 不同。 | 以关键字参数传给 sampler。 |
| `priority` | `list[int]` 或 `bool` | 否 | 无 | 当前观测为 `false` 或 list。 | 若存在，使用 `random_region_list` 中的候选区域。为真时从 list 中选索引；为假时随机选任意索引。 |
| `container` | `str` | 否 | 无 | 当前少量 sandwich 任务使用。 | 特殊分支：把 object 放到 container pose 上，并使用 container.gap 修正 x。 |
| `z_init` | `float` | `container` 分支必填 | 无 | 当前观测 `0.06`。 | container 分支中 object 的 z 偏移。 |
| `sub_tgt_prim` | `str` | 否 | 无 | 当前观测 `/World`。 | 使用 `target.prim_path + sub_tgt_prim` 作为采样目标。 |
| `target2` | `str` | 双目标 sampler 必填 | 无 | 代码支持，当前目标 YAML 未观测。 | `A_along_B_C_circle_sampler` 使用的第二目标。 |
| `visible` | `bool` | 否 | 无 | 当前观测 `true` / `false`。 | 当前主摆放代码不读取，属于配置元数据。 |

### Samplers

#### `A_on_B_region_sampler`

把 object 放到 target 的上表面附近，适合“杯子放桌上”“物体放托盘上”这类 A on B 场景。

| 参数 | 类型 | 必填 | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `pos_range` | `[[min_x,min_y,min_z], [max_x,max_y,max_z]]` | 是 | 米 | 先取 target bbox 中心和 target bbox 顶面高度，再从该范围均匀采样 `[dx,dy,dz]` 加到结果位置上。`dx/dy` 控制在 target 表面附近的水平随机范围；`dz` 控制额外高度偏移。 |
| `yaw_rotation` | `[min_deg,max_deg]` | 是 | degree | 从该范围均匀采样 yaw，并在 object 当前姿态基础上追加绕 z 轴旋转。它不会重置 roll/pitch，只是在原姿态上叠加 yaw。 |

计算结果：

| 输出 | 计算方式 |
| --- | --- |
| `translation.x/y` | `target_bbox_center.x/y + sampled_pos_range.x/y`。 |
| `translation.z` | `target_bbox.max.z + object当前中心到object bbox底面的距离 + 0.001 + sampled_pos_range.z`。`0.001` 用于减少穿模。 |
| `orientation` | `Rz(sampled_yaw) * object_current_orientation`。 |

#### `A_in_B_region_sampler`

把 object 放到 target 的中心位置上方，z 高度按 target 顶面和 object 底面自动对齐，适合“把 A 放进/放在容器 B 内部上沿附近”的简单场景。它不是严格几何包含检测，只是按 target 的 local pose 中心加偏移。

| 参数 | 类型 | 默认值 | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `x_bias` | `float` | `0` | 米 | 在 target local pose 的 x 坐标上追加固定偏移。 |
| `y_bias` | `float` | `0` | 米 | 在 target local pose 的 y 坐标上追加固定偏移。 |
| `z_bias` | `float` | `0` | 米 | 在自动对齐高度上追加固定 z 偏移。 |

计算结果：

| 输出 | 计算方式 |
| --- | --- |
| `translation.x/y` | `target_local_translation.x/y + x_bias/y_bias`。 |
| `translation.z` | `target_bbox.max.z + object当前中心到object bbox底面的距离 - 0.005 + z_bias`。`-0.005` 会让 object 略微贴近或压入 target 顶面。 |
| `orientation` | 保留 object 当前 local orientation，不追加随机 yaw。 |

#### `A_by_B_region_sampler`

把 object 放在 target 旁边的矩形区域内，适合“物体在桌子/容器旁边某个平面范围内”的场景。与 `A_on_B_region_sampler` 的区别是：z 不是放到 target 顶面，而是用 object 和 target 的 bbox 半高差，让两者 bbox 中心高度按底部近似对齐。

| 参数 | 类型 | 必填 | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `pos_range` | `[[min_x,min_y,min_z], [max_x,max_y,max_z]]` | 是 | 米 | 代码只使用采样结果的 x/y；z 分量即使写入也不参与当前位置计算。 |
| `yaw_rotation` | `[min_deg,max_deg]` | 是 | degree | 从该范围均匀采样 yaw，并叠加到 object 当前姿态上。 |

计算结果：

| 输出 | 计算方式 |
| --- | --- |
| `translation.x/y` | `target_bbox_center.x/y + sampled_pos_range.x/y`。 |
| `translation.z` | `target_bbox_center.z + object_bbox_half_height.z - target_bbox_half_height.z`。 |
| `orientation` | `Rz(sampled_yaw) * object_current_orientation`。 |

#### `A_by_B_circle_sampler`

把 object 放在 target 周围的圆环或扇形区域内，适合“围绕 B 随机放置 A”的场景。

| 参数 | 类型 | 必填 | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `r_range` | `[min,max]` | 是 | 米 | 从该范围均匀采样半径 `r`。 |
| `theta_range` | `[min_deg,max_deg]` | 是 | degree | 从该范围均匀采样极角 `theta`。`0` degree 对应 +x 方向，`90` degree 对应 +y 方向。 |
| `yaw_rotation` | `[min_deg,max_deg]` | 是 | degree | 从该范围均匀采样 yaw，并叠加到 object 当前姿态上。 |

计算结果：

| 输出 | 计算方式 |
| --- | --- |
| `translation.x/y` | `target_bbox_center + [r*cos(theta), r*sin(theta)]`。 |
| `translation.z` | `target_bbox_center.z + object_bbox_half_height.z - target_bbox_half_height.z`。 |
| `orientation` | `Rz(sampled_yaw) * object_current_orientation`。 |

#### `A_face_B_circle_sampler`

把 object 放在 target 当前朝向指向的一条射线上，距离由 `r_range` 随机采样。适合“把 A 放在 B 正前方/面向方向上”的场景。这里的“face”来自 target 的当前世界 yaw，而不是 object 自己的朝向。

| 参数 | 类型 | 必填 | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `r_range` | `[min,max]` | 是 | 米 | 沿 target 当前 yaw 方向采样距离。 |
| `yaw_rotation` | `[min_deg,max_deg]` | 是 | degree | 从该范围均匀采样 yaw，并叠加到 object 当前姿态上。 |

计算结果：

| 输出 | 计算方式 |
| --- | --- |
| `translation.x/y` | `target_bbox_center + r * [cos(target_yaw), sin(target_yaw)]`。 |
| `translation.z` | `target_bbox_center.z + object_bbox_half_height.z - target_bbox_half_height.z`。 |
| `orientation` | `Rz(sampled_yaw) * object_current_orientation`。 |

#### `A_along_B_C_circle_sampler`

把 object 放在 target B 当前朝向上，并把 B 到 C 的距离作为基础距离，再额外加上 `r_range` 采样值。适合“沿 B 指向的方向，把 A 放在越过 C 之后一段距离”的场景。外层 `target` 是 B，外层 `target2` 是 C。

| 参数 | 类型 | 必填 | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `target2` | `str` | 是 | 名称引用 | 写在 region 外层，必须引用已加载对象/fixture。代码读取 B 和 C 的 world pose 距离。 |
| `r_range` | `[min,max]` | 是 | 米 | 在 `distance(B,C)` 基础上额外增加的距离范围。最终半径为 `distance(B,C) + uniform(r_range)`。 |
| `yaw_rotation` | `[min_deg,max_deg]` | 是 | degree | 从该范围均匀采样 yaw，并叠加到 object 当前姿态上。 |

计算结果：

| 输出 | 计算方式 |
| --- | --- |
| `translation.x/y` | `B_bbox_center + (distance(B,C)+sampled_r) * [cos(B_yaw), sin(B_yaw)]`。 |
| `translation.z` | `B_bbox_center.z + object_bbox_half_height.z - B_bbox_half_height.z`。 |
| `orientation` | `Rz(sampled_yaw) * object_current_orientation`。 |

---

## A8. `cameras[]`

相机实例配置。由 `BananaBaseTask._load_camera()` 和 `CustomCamera` 消费。

### Schema

```yaml
cameras:
  - name: split_aloha_head
    camera_file: workflows/simbox/core/configs/cameras/realsense_d455_v3.yaml
    parent: split_aloha/split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper/top_camera_link
    translation: [0.0, -0.00818, 0.1]
    orientation: [0.658, 0.259, -0.282, -0.648]
    camera_axes: usd
    apply_randomization: false
```

### Fields

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | 任意唯一相机名。 | camera dict key 和 observation key。 |
| `camera_file` | `str` | 是 | 无 | 当前观测完整路径见 E1。 | 外部相机内参/分辨率配置。读取后合并为 `params` 传给 `CustomCamera`。 |
| `parent` | `str` | 是 | 空字符串表示 world-mounted | 当前有 robot link 相对路径、OmegaConf 插值路径和空字符串。 | 非空时相机 mount 创建在该 parent 下；相对路径会拼到 task root。 |
| `translation` | `[x,y,z]` | 是 | 无 | 当前均显式配置。 | 相机局部位置。 |
| `orientation` | `[w,x,y,z]` | 是 | 无 | 当前均显式配置。 | 相机局部朝向。 |
| `camera_axes` | `str` | 是 | 无 | 当前全部为 `usd`。 | 传给 Isaac camera `set_local_pose()`。 |
| `apply_randomization` | `bool` | 否 | `false` | 当前大量显式配置。 | true 时每次 randomize 扰动外参。 |
| `max_translation_noise` | `float` | 否 | `0.05` | 当前常见 `0.02`、`0.03`。 | 外参扰动最大平移距离。 |
| `max_orientation_noise` | `float` | 否 | `10.0` | 当前常见 `2.5`、`5.0`。 | 外参扰动最大旋转角，degree。 |
| `output_mode` | `str` | 否 | `rgb` | 代码支持 `rgb`、`diffuse_albedo`。 | `rgb` 表示观测图像来自 camera RGBA 的前三通道；`diffuse_albedo` 表示挂载 diffuse_albedo annotator 并输出去掉光照影响的反照率颜色。当前目标 YAML 未观测。 |

### Camera File Fields

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `camera_type` | `str` | 否 | 无 | 元信息；`CustomCamera` 不读取。 |
| `camera_params` | `[fx,fy,cx,cy]` | 是 | 无 | 内参。 |
| `resolution_width` / `resolution_height` | `int` | 是 | 无 | 渲染分辨率。 |
| `frequency` | `int` | 否 | 无 | 元信息；当前 `CustomCamera` 不读取。 |
| `pixel_size` | `float` | 是 | 无 | 像素尺寸，micron。 |
| `f_number` | `float` | 是 | 无 | 光圈，用于 lens aperture。 |
| `focus_distance` | `float` | 是 | 无 | 对焦距离。 |
| `with_distance` | `bool` | 否 | `true` | 是否输出 distance-to-image-plane。 |
| `with_semantic` | `bool` | 否 | `false` | 是否输出语义分割。 |
| `with_bbox2d` | `bool` | 否 | `false` | 是否输出 2D bbox。 |
| `with_bbox3d` | `bool` | 否 | `false` | 是否输出 3D bbox。 |
| `with_motion_vector` | `bool` | 否 | `false` | 是否输出 motion vectors。 |
| `depth` | `bool` | 否 | `false` | 是否启用 depth 标志。 |

---

## A9. `distractors`

视觉干扰物配置。由 `_create_distractor_cfg()` 和 `visual_distractor.set_distractors()` 消费。

### Schema

```yaml
distractors:
  path: pick_and_place/pre-train-pick/assets
  min_num: 5
  max_num: 10
  target: table
  scale: [0.001, 0.001, 0.001]
  pos_range:
    - [-0.6, -0.2]
    - [0.6, 0.2]
  min_object_distance: 0.03
  exclude_keywords: [shoe, book]
```

### Fields

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `path` | `str` | 是 | 无 | 相对 `asset_root` 的干扰物资产根。代码搜索 `asset_root/path/*/*/*.usd`。 |
| `min_num` / `max_num` | `int` | 是 | 无 | 每次随机采样的干扰物数量范围。 |
| `target` | `str` | 是 | 无 | 摆放目标对象/fixture 名，通常是 `table`。 |
| `scale` | `[sx,sy,sz]` | 否 | `[1,1,1]` | 生成的 distractor object 缩放。 |
| `pos_range` | `[[x_min,y_min], [x_max,y_max]]` | 否 | 不限制 | 相对 target 中心的 XY 限制区域。 |
| `min_object_distance` | `float` | 否 | `0.03` | distractor 与主 objects 的最小 XY 距离。 |
| `distractor_buffer` | `float` | 否 | `0.03` | distractor 之间的 buffer。 |
| `max_attempts` | `int` | 否 | `10` | 每个 distractor 放置尝试次数。 |
| `fallback_z` | `float` | 否 | `-5.0` | 放置失败时移出视野的 z。 |
| `exclude_categories` | `list[str]` | 否 | `[]` | 精确排除类别。 |
| `exclude_keywords` | `list[str]` | 否 | `[]` | 按类别名子串排除，大小写不敏感。 |
| `target_class` | `str` | 否 | `RigidObject` | `_rebuild_distractors()` 当前只支持 `RigidObject`。 |
| `prim_path_child` | `str` | 否 | `Aligned` | 生成的 RigidObject 子 prim。 |
| `translation` | `[x,y,z]` | 否 | `[0,0,0]` | 生成 object 初始位置。 |

---

## A10. `fluid`

PhysX 粒子流体配置。由 `_set_fluid()` 消费。

### Fields

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `container_name` | `str` | 是 | 无 | 初始粒子网格中心使用的对象名。 |
| `particleContactOffset` | `float` | 否 | `0.005` | 粒子 contact offset。 |
| `spacing_scale` | `float` | 否 | `1.2` | 粒子间距倍率。 |
| `numParticlesX` / `numParticlesY` / `numParticlesZ` | `int` | 否 | `7` / `7` / `450` | 粒子网格尺寸。当前 YAML 中 `Z` 常为 200 或 400。 |
| `center_z` | `bool` | 否 | `false` | 是否让粒子网格在 z 方向围绕容器中心。 |
| `z_offset` | `float` | 否 | `0.0` | 粒子网格 z 偏移。 |
| `max_velocity` | `float` | 否 | `0.8` | 粒子系统最大速度。 |
| `mass` | `float` | 否 | `0.0` | 粒子质量。 |
| `density` | `float` | 否 | `0.0` | 粒子密度。 |
| `color` | `[r,g,b]` | 否 | `[1,1,1]` | 粒子材质颜色。 |
| `emissiveColor` | `[r,g,b]` | 否 | `[0,0,0]` | 粒子自发光颜色。 |
| `opacity` | `float` | 否 | `1.0` | 粒子材质透明度。 |

---

## A11. `data`

采集和语言配置。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `task_dir` | `str` | 是 | 无 | logger 输出任务目录。 |
| `language_instruction` | `str` | 是 | 内部 fallback 为 pick 模板 | 高层语言指令。可用 `;` 分隔多条。 |
| `detailed_language_instruction` | `str` | 是 | 内部 fallback 为 grasp 模板 | 详细语言指令。可用 `;` 分隔多条。 |
| `collect_info` | `str` | 是 | 无 | 采集元信息。articulation 随机化时可能被覆盖。 |
| `version` | `str` | 否 | `v1.0` | 数据版本。 |
| `update` | `bool` | 是 | 无 | 任务更新标记。 |
| `max_episode_length` | `int` | 是 | 无 | episode 最大步数。 |
| `log_motion_vectors` | `bool` | 否 | `false` | 是否记录 motion vectors。 |

---

# Part B. Skills API

## B1. Skill Phase Schema

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
| `skills[]` | `dict` | 一个 phase。legacy 模式下 phase 顺序执行。 |
| `<robot_name>` | `list[dict]` | key 必须匹配 `robots[].name`。 |
| `<controller_name>` | `list[dict]` | 常见 `left`、`right`、`base`。 |
| skill item | `dict` | 至少包含 `name`；大多数还需要 `objects`。 |

## B2. DAG Mode

如果任意 skill item 包含 `id` 或 `depends_on`，整个 task 进入 DAG 模式。

| 字段 | 类型 | 规则 |
| --- | --- | --- |
| `id` | `str` | DAG 模式下每个 skill 都必须有唯一 id。 |
| `depends_on` | `list[str]` | 依赖的 skill id；必须是 list。 |

运行规则：

1. workflow 对全部 skill 做拓扑排序。
2. 依赖全部成功后，节点进入 running。
3. 任意依赖 id 不存在会报错。
4. 有环会报错。

## B3. Skill 注册名

注册名来自类名转小写并在大写字母前插入下划线。源码类名中已有 `_` 时，注册名可能出现双下划线。

| 注册名 | 类 | 当前是否出现 | 功能 |
| --- | --- | --- | --- |
| `pick` | `Pick` | 是 | 标准抓取。 |
| `manualpick` | `Manualpick` | 是 | 手工修正抓取。 |
| `dexpick` | `Dexpick` | 是 | 离散姿态抓取。 |
| `dynamicpick` | `Dynamicpick` | 是 | 动态目标抓取。 |
| `fail_pick` | `FailPick` | 是 | 故意失败抓取。 |
| `place` | `Place` | 是 | 通用放置。 |
| `dexplace` | `Dexplace` | 是 | 几何范围放置。 |
| `move` | `Move` | 是 | 移动物体。 |
| `goto__pose` | `Goto_Pose` | 是 | 到指定 EE 位姿。 |
| `track` | `Track` | 是 | 路点追踪。 |
| `scan` | `Scan` | 是 | 扫描/观测姿态。 |
| `wait` | `Wait` | 是 | 保持位姿等待。 |
| `gripper__action` | `Gripper_Action` | 是 | 夹爪开合。 |
| `heuristic__skill` | `Heuristic_Skill` | 是 | 启发式运动。 |
| `joint__ctrl` | `Joint_Ctrl` | 是 | 关节控制。 |
| `navigate` | `Navigate` | 是 | Nav2 导航。 |
| `open` | `Open` | 是 | 打开 articulation。 |
| `close` | `Close` | 是 | 关闭 articulation。 |
| `artpreplan` | `Artpreplan` | 是 | articulation 预规划。 |
| `rotate` | `Rotate` | 否 | articulation 旋转；注册支持。 |
| `rotate__obj` | `Rotate_Obj` | 是 | 旋转被抓物体。 |
| `approach__rotate` | `Approach_Rotate` | 是 | 接近并旋转。 |
| `flip` | `Flip` | 是 | 翻转物体。 |
| `pour__water__succ` | `Pour_Water_Succ` | 是 | 倒水成功检测。 |
| `home` | `Home` | 否 | 关节回 home；注册支持。 |

## B4. Common Skill Parameters

这些字段可能被多个 skill 使用。

| 字段 | 类型 | 默认值 | 允许值 / 说明 |
| --- | --- | --- | --- |
| `name` | `str` | 无 | 必须是注册名。 |
| `objects` | `list[str]` | 无 | skill 需要的对象引用。长度和语义由具体 skill 定义。 |
| `ignore_substring` | `list[str]` | `[]` | 附加到 controller 的碰撞忽略名单。 |
| `t_eps` | `float` | 多数 skill 默认 `1e-3` 或配置值 | EE 位置完成阈值。 |
| `o_eps` | `float` | 多数 skill 默认 `5e-3` 或配置值 | EE 姿态完成阈值。 |
| `filter_x_dir` / `filter_y_dir` / `filter_z_dir` | `list` | 无 | 姿态筛选。格式为 `[direction, angle]` 或 `[direction, min_angle, max_angle]`；direction 候选值和语义见 E4。 |
| `test_mode` | `str` | `forward` | 允许 `forward`、`ik`。`forward` 用 controller 前向规划结果筛选，`ik` 用 IK 可达性筛选。 |
| `gripper_state` | `int` 或 `float` | skill-specific | 夹爪目标状态。约定 `1` / `1.0` 为 open，`-1` / `-1.0` 为 close；`gripper__action` 只接受整数语义的 `1` 和 `-1`。 |
| `gripper_change_steps` | `int` | skill-specific | 夹爪动作重复步数。 |
| `process_valid` | `bool` | `true` | 成功判定是否检查机器人/物体速度稳定。 |
| `collision_valid` | `bool` | `true` | articulation skill 是否检查禁碰撞。 |

---

## B5. `pick`

标准抓取 skill。实现：`workflows/simbox/core/skills/pick.py`

### Parameters

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 用法 |
| --- | --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | 必须为 `pick`。 | 选择 `Pick` 类。 |
| `objects` | `list[str]` | 是 | 无 | 长度 1。 | `objects[0]` 是被抓取对象，必须是 task object。 |
| `output_root` | `str` | 否 | `output/ros_bridge/skills` | 任意路径。 | 保存抓取调试输出。 |
| `npy_name` | `str` | 否 | `Aligned_grasp_sparse.npy` | 当前观测 `Aligned_grasp_sparse_left6.npy`、`Aligned_grasp_sparse_right6.npy`。 | 从对象 USD 旁替换得到抓取姿态 npy。 |
| `grasp_scale` | `float` | 否 | `1` | 当前观测 `0.35`、`0.7`、`0.8`。 | 传给 grasp pose 后处理函数。 |
| `tcp_offset` | `float` | 否 | `robot.tcp_offset` | 当前观测 `0.11`、`0.115`、`0.125`、`0.145`、`0.155`。 | TCP 到末端的偏移补偿。 |
| `constraints` | `list` | 否 | `None` | 当前格式如 `["y", 0.2, 0.3]`。 | 抓取姿态后处理约束。 |
| `final_gripper_state` | `int` | 否 | `-1` | 当前观测 `1`。 | 生成抓取后夹爪命令；`1` open，`-1` close。 |
| `fixed_orientation` | quaternion | 否 | `None` | `[w,x,y,z]`。 | 若提供，覆盖采样抓取姿态的 orientation。 |
| `pre_grasp_offset` | `float` | 否 | `0.1` | 当前常见 `0.0`、`0.05`、`0.1`。 | 沿抓取姿态 x 或 z 轴退让，构造 pre-grasp。 |
| `pre_grasp_hold_vec_weight` | `list[float]` | 否 | `None` | 通常长度 6。 | pre-grasp 阶段传给 controller 的 pose cost metric。 |
| `gripper_change_steps` | `int` | 否 | `40` | 当前观测 `10`、`20`。 | 夹爪闭合命令重复步数。 |
| `post_grasp_offset_min` | `float` | 否 | `0.05` | 当前常见 `0.0`、`0.05`、`0.1`。 | 抓取后 z 向上抬距离采样下限。 |
| `post_grasp_offset_max` | `float` | 否 | `0.05` | 当前常见 `0.0`、`0.1`、`0.2`。 | 抓取后 z 向上抬距离采样上限。 |
| `return_to_pregrasp` | `bool` | 否 | `false` | 当前观测 `true`。 | 抓取后是否回到 pre-grasp pose。 |
| `filter_x_dir` / `filter_y_dir` / `filter_z_dir` | `list` | 否 | 无 | direction 常见 `forward`、`downward`、`upward`。 | 过滤抓取姿态方向。 |
| `direction_to_obj` | `str` | 否 | 无 | `left`、`right`。 | 限制抓取点在对象左/右侧。 |
| `test_mode` | `str` | 否 | `forward` | 当前观测 `forward`、`ik`。 | 候选姿态可达性检查方式。 |
| `t_eps` | `float` | 否 | `1e-3` | 当前常见 `0.01`、`0.025`。 | 子命令完成位置阈值。 |
| `o_eps` | `float` | 否 | `5e-3` | 当前观测 `1`。 | 子命令完成姿态阈值。 |
| `process_valid` | `bool` | 否 | `true` | 当前观测 `true`。 | 成功判定是否要求机器人和对象速度稳定。 |
| `lift_th` | `float` | 否 | `0.0` | 当前常见 `0.0`、`0.02`。 | 成功判定时要求对象相对初始高度至少抬升该值。 |
| `close_wait_steps` | `int` | 否 | 不读取 | 当前观测 `10`。 | 当前 `Pick` 代码不读取；闭合等待由 `gripper_change_steps` 控制。 |

### Example

```yaml
- name: pick
  objects: [pick_object]
  pre_grasp_offset: 0.05
  post_grasp_offset_min: 0.05
  post_grasp_offset_max: 0.2
  gripper_change_steps: 10
  filter_z_dir: [forward, 60]
  test_mode: forward
  process_valid: true
  lift_th: 0.02
```

---

## B6. `manualpick`

手工修正抓取。实现：`manualpick.py`

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 用法 |
| --- | --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | `manualpick`。 | 选择 skill。 |
| `objects` | `list[str]` | 是 | 无 | 长度 1。 | 被抓取对象。 |
| `npy_name` | `str` | 否 | `Aligned_grasp_sparse.npy` | 任意 npy 文件名。 | 抓取姿态文件。 |
| `grasp_scale` | `float` | 否 | `1` | 当前 `0.01`、`1`。 | 抓取姿态缩放。 |
| `hold_vec_weight` | `list[float]` | 否 | `None` | 通常长度 6。 | 初始 controller pose cost metric。 |
| `start_lr_skill` | `bool` | 否 | `false` | true / false。 | 先插入一次无约束 update。 |
| `adjust_ori` | `[pose_axis, base_axis, judge_flag]` | 否 | 无 | 当前如 `[z,z,max]`。 | 自动旋转候选抓取姿态并选择更优姿态。 |
| `adjust_rotate_axis` | `str` | 否 | `x` | 当前 `x`、`z`。 | 自动修正时绕哪个轴旋转。 |
| `adjust_angle_list_cfg` | `[min,max,count]` | 否 | `[-15,15,7]` | 当前如 `[-30,30,21]`。 | 自动修正角度采样列表。 |
| `manual_adjust_ori` | `list[[axis, angle]]` | 否 | 无 | 当前如 `[[y,-15]]`。 | 对抓取姿态追加固定旋转。 |
| `adjust_trans_offset` | `[x,y,z]` | 否 | `[0,0,0]` | 当前如 `[0,0.03,0.1]`。 | 对抓取 pose 追加世界/局部平移修正。 |
| `pre_grasp_offset` | `float` | 否 | `0.1` | 当前全部 `0.0`。 | pre-grasp 退让距离。 |
| `pre_grasp_offset_manual` | `[x,y,z]` | 否 | 无 | 当前如 `[-0.1,0,0]`。 | 对 pre-grasp 追加手工偏移。 |
| `post_grasp_offset_min` / `post_grasp_offset_max` | `float` | 否 | `0.05` | 当前 `0.0`、`0.008`。 | 抓取后上抬距离范围。 |
| `test_mode` | `str` | 否 | `forward` | 当前全部 `ik`。 | 可达性检查方式。 |
| `filter_x_dir` / `filter_y_dir` / `filter_z_dir` | `list` | 否 | 无 | 方向过滤。 | 与 `pick` 相同。 |
| `direction_to_obj` | `str` | 否 | 无 | `left` / `right`。 | 左右侧抓取约束。 |
| `update_pose_cost_metric_none` | `bool` | 否 | 不读取 | 当前全部 `true`。 | 当前 `Manualpick` 代码不读取。 |

---

## B7. `dexpick`

离散姿态抓取。实现：`dexpick.py`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | `objects[0]` 为被抓取对象。 |
| `pick_pose_idx` | `int` | 否 | `0` | 选择 `dexpick_pose.yaml` 中的抓取姿态索引。当前全部为 `0`。 |
| `pre_grasp_offset` | `float` | 否 | `0.1` | pre-grasp 退让距离。当前全部 `0.0`。 |
| `post_grasp_offset_min` / `post_grasp_offset_max` | `float` | 否 | `0.05` | 代码实际读取的抓取后上抬范围。 |
| `post_grasp_offset` | `float` | 否 | 不读取 | 当前 YAML 写了 `0.1`，但代码不读取。 |
| `gripper_change_steps` | `int` | 否 | `40` | 闭合夹爪步数。 |
| `process_valid` | `bool` | 否 | `true` | 是否检查速度稳定。 |
| `lift_th` | `float` | 否 | `0.0` | 最小抬升高度。 |

---

## B8. `dynamicpick`

动态目标抓取。实现：`dynamicpick.py`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | `objects[0]` 为动态目标。 |
| `pick_range` | `[min,max]` | 是 | 无 | 沿运动方向的抓取窗口。 |
| `tcp_offset` | `float` | 否 | `0.125` | 当前 `0.1`、`0.12`。 |
| `grasp_scale` | `float` | 否 | `1` | 抓取姿态缩放。 |
| `time_bias` | `float` | 否 | `0` | 预测时间偏置。 |
| `pick_bias` | `float` | 否 | `0` | 抓取位置偏置。 |
| `pivot_angle_z` | `[min,max]` | 否 | 无 | 抓取姿态 z 轴旋转扰动。 |
| `pos_adjust_z` | `[min,max]` | 否 | 无 | 抓取点 z 偏移采样范围。 |
| `pre_grasp_offset` | `float` | 否 | `0.1` | pre-grasp 退让。 |
| `post_grasp_offset_min` / `post_grasp_offset_max` | `float` | 否 | `0.05` | post-grasp 上抬范围。 |
| `filter_x_dir` / `filter_y_dir` / `filter_z_dir` | `list` | 否 | 无 | 抓取姿态方向过滤；格式和 direction 候选值见 E4。 |
| `direction_to_obj` | `str` | 否 | 无 | 左右侧抓取约束；`left` 要求 EE 在物体 y 正侧，`right` 要求 EE 在物体 y 负侧或相等。 |
| `process_valid` | `bool` | 否 | `true` | 是否检查速度稳定。 |
| `lift_th` | `float` | 否 | `0.0` | 最小抬升高度。 |

---

## B9. `fail_pick`

故意偏抓。实现：`failpick.py`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | `objects[0]` 为目标对象。 |
| `grasp_x_offset_min` / `grasp_x_offset_max` | `float` | 否 | `0.05` / `0.1` | x 方向故意偏移范围。 |
| `grasp_y_offset_min` / `grasp_y_offset_max` | `float` | 否 | `0.05` / `0.1` | y 方向故意偏移范围。 |
| `post_grasp_offset_min` / `post_grasp_offset_max` | `float` | 否 | `0.05` | 抓取后上抬范围。 |
| `filter_x_dir` / `filter_y_dir` / `filter_z_dir` | `list` | 否 | 无 | 故意偏抓前仍可筛选抓取姿态方向；格式和 direction 候选值见 E4。 |

注意：`fail_pick.is_success()` 固定返回 `true`。失败语义来自动作偏移，不来自 episode success。

---

## B10. `place`

通用放置。实现：`place.py`

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 用法 |
| --- | --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | 长度 2。 | `objects[0]` 被放置对象，`objects[1]` 放置目标。 |
| `place_part_prim_path` | `str` | 否 | 无 | 当前 `contact_link1`、`base`。 | 使用目标对象子 prim 作为放置几何。 |
| `place_align_axis` / `pick_align_axis` | `[x,y,z]` | 否 | 无 | 任意轴向量。 | 末端/物体对齐轴。 |
| `constraint_gripper_x` | `bool` | 否 | `false` | true / false。 | 是否约束 gripper x。 |
| `align_pick_obj_axis` / `align_place_obj_axis` | `[x,y,z]` | 否 | 无 | 当前常见 `[1,0,0]`。 | 水平放置或对齐时使用。 |
| `align_plane_x_axis` / `align_plane_y_axis` | `[x,y,z]` | 否 | 无 | 任意轴。 | 平面对齐约束。 |
| `align_obj_tol` | `float` | 否 | 无 | 当前 3、5、10、15、25、30。 | 对齐容差，degree。 |
| `pre_place_hold_vec_weight` / `post_place_hold_vec_weight` | `list[float]` | 否 | 无 | 通常长度 6。 | pre/post 阶段 pose cost 权重。 |
| `hesitate_steps` | `int` | 否 | `0` | 任意非负整数。 | 到达 place pose 后保持步数。 |
| `gripper_change_steps` | `int` | 否 | `10` | 当前 `10`、`20`。 | 松开夹爪步数。 |
| `post_place_vector` | `[x,y,z]` | 否 | 无 | 任意向量。 | 松爪后沿该向量 retreat。 |
| `x_ratio_range` / `y_ratio_range` / `z_ratio_range` | `[min,max]` | 否 | x/y 默认 `[0.3,0.7]`，z 默认无 | 当前广泛配置。 | 在目标 bbox 内按比例采样放置点。 |
| `place_direction` | `str` | 否 | `vertical` | 代码有效分支：`vertical`、`horizontal`；当前观测二者都有。 | 决定放置目标点的生成方式。`vertical` 从目标 bbox 上方下放；`horizontal` 沿目标对象局部轴侧向接近/放置。详细语义见 E4。 |
| `position_constraint` | `str` | 否 | `gripper` | 当前观测 `object`。 | `object` 表示让物体到目标位置；`gripper` 表示让夹爪到目标位置。 |
| `pre_place_z_offset` | `float` | 否 | `0.2` | 当前多个值。 | 垂直放置 pre-place 高度。 |
| `place_z_offset` | `float` | 否 | `0.1` | 当前多个值。 | 垂直放置最终高度。 |
| `offset_place_obj_axis` | `[x,y,z]` | 否 | 无 | 当前 `[0,0,-1]`。 | 水平放置沿目标轴偏移。 |
| `pre_place_align` / `place_align` | `float` | 否 | `0.2` / `0.1` | 当前正负值。 | 水平放置沿对齐轴偏移。 |
| `pre_place_offset` / `place_offset` | `float` | 否 | `0.2` / `0.1` | 当前 `0.027`。 | 水平放置沿 offset 轴偏移。 |
| `filter_x_dir` / `filter_y_dir` / `filter_z_dir` | `list` | 否 | 无 | 姿态方向筛选条件，作用于随机生成的 place EE 姿态；格式和 direction 候选值见 E4。 | 筛选 IK 候选姿态。 |
| `test_mode` | `str` | 否 | `forward` | `forward`、`ik`。 | 可达性检查；`forward` 用 controller 前向规划筛选，`ik` 用 IK 筛选。 |
| `success_mode` | `str` | 否 | `3diou` | 代码支持 `3diou`、`height`、`xybbox`、`left`、`right`、`flower`、`cup`。当前观测除 `3diou` 外都有。 | 放置成功判定模式，决定 `is_success()` 使用 IoU、高度、bbox 范围或左右关系检查；每个值的含义见 E4。 |
| `threshold` | `float` | 否 | `0.03` | 当前 `0.01`。 | `left/right` 成功判定 margin。 |
| `success_th` | `float` | 否 | `0.0` | 当前 `0.05`、`0.1`、`0.15`。 | IoU / flower / cup 模式阈值。 |

---

## B11. `dexplace`

几何范围放置。实现：`dexplace.py`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | 长度 2。`objects[0]` 被放置对象，`objects[1]` 目标。 |
| `gripper_axis` | `[x,y,z]` | 否 | 由目标方向计算 | 指定夹爪方向。 |
| `camera_axis_filter` | `list[dict]` | 否 | 无 | 当前格式 `[{direction: [...]}, {degree: [...]}]`，用于姿态选择。 |
| `place_part_prim_path` | `str` | 否 | 无 | 放置目标子 prim。 |
| `gripper_change_steps` | `int` | 否 | `10` | 松爪步数。 |
| `t_eps` / `o_eps` | `float` | 否 | `1e-3` / `5e-3` | 子命令完成阈值。 |

---

## B12. `move`

推动/移动对象。实现：`move.py`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | 长度 2。`objects[0]` 被移动对象，`objects[1]` 目标对象。 |
| `success_threshold` | `float` 或字符串数值 | 是 | 无 | EE 到位和对象靠近目标的阈值。当前有 `0.04`、`0.05`、`"5e-3"`。 |
| `delta_trans` | `list[[x,y,z]]` | 否 | `[[0,0,0]]` | 主目标外的追加平移序列。 |
| `hold_vec_weight` | `list[float]` | 否 | `[0,0,0,0,0,0]` | pose cost 权重。 |
| `invisible_object` | `list[str]` | 否 | 无 | 执行时临时显示/隐藏的对象。 |

---

## B13. `goto__pose`

直接移动 EE 到指定位姿。实现：`goto_pose.py`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `position` | `[x,y,z]` | 是 | 无 | 目标 EE 位置。 |
| `quaternion` | `[w,x,y,z]` | 否 | 无 | 目标 EE 姿态。 |
| `euler` | `[r,p,y]` | 否 | 无 | 目标 EE 姿态，degree。 |
| `frame` | `str` | 否 | `robot` | 目标坐标系。当前观测 `robot`。 |
| `gripper_state` | `1` 或其他 | 否 | `1` | `1` open，否则 close。当前多为 `-1`。 |
| `max_noise_m` | `float` | 否 | `0.0` | 位置噪声。 |
| `max_noise_deg` | `float` | 否 | `0` | 姿态噪声。 |
| `objects` | `list[str]` | 未给姿态时必填 | 无 | 用于姿态采样的参考对象。 |
| `align_obj_axis` / `align_ref_axis` | `[x,y,z]` | 未给姿态时必填 | 无 | 姿态对齐轴。 |
| `align_obj_tol` | `float` | 未给姿态时必填 | 无 | 对齐容差。 |
| `position_constraint` | `str` | 否 | `gripper` | 当前观测 `object`。`gripper` 表示目标 position 约束夹爪；`object` 表示目标 position 约束被操作对象。 |
| `interp_nums` | `int` | 否 | `1` | 插值路点数。当前 `1`、`4`。 |
| `filter_x_dir` / `filter_y_dir` / `filter_z_dir` | `list` | 否 | 无 | 目标 EE 姿态方向过滤；`goto__pose` 支持 `leftward/rightward`，格式和 direction 候选值见 E4。 |

---

## B14. `track`

随机路点追踪。实现：`track.py`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `target` | `str` | 是 | 无 | 当前观测 `target_left`、`target_right`；当前主轨迹生成逻辑不读取它。 |
| `T_tcp_2_ee` | `4x4 matrix` | 否 | 单位阵 | TCP 到 EE 的变换。 |
| `way_points_num` | `int` | 否 | `1` | 路点数。当前为 `3`。 |
| `way_points_trans.min` / `way_points_trans.max` | `[x,y,z]` | 是 | 无 | 路点位置采样范围。 |
| `way_points_ori` | quaternion | 是 | 无 | 路点基准姿态。 |
| `max_noise_deg` | `float` | 否 | `5` | 姿态噪声。当前 `180`。 |
| `ignore_substring` | `list[str]` | 否 | `[]` | 碰撞忽略名单。 |

---

## B15. `scan`

扫描/观测姿态。实现：`scan.py`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | `objects[0]` 是关联对象。 |
| `t_eps` / `o_eps` | `float` | 否 | `1e-3` / `5e-3` | 子命令完成阈值。 |
| `process_valid` | `bool` | 否 | `true` | 成功判定是否检查速度稳定。 |

---

## B16. `wait`

保持当前 EE 位姿等待。实现：`wait.py`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | 关联对象。 |
| `wait_steps` | `int` | 否 | `50` | 当前观测 `10`、`15`、`20`、`80`。 |
| `success_threshold` | `float` 或字符串数值 | 是 | 无 | 等待结束时 EE 与起始 pose 的距离阈值。当前 `"5e-3"`。 |
| `gripper_state` | `-1` 或 `1` | 否 | `-1` | 等待期间夹爪状态。 |
| `ignore_substring` | `list[str]` | 否 | `[]` | 碰撞忽略名单。 |

---

## B17. `gripper__action`

仅控制夹爪。实现：`gripper_action.py`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `gripper_state` | `1` 或 `-1` | 是 | 无 | `1` open，`-1` close。其他值会 `NotImplementedError`。 |
| `vel` | `float` | 否 | `None` | 可选夹爪速度参数。 |
| `wait_steps` | `int` | 否 | `10` | 重复发送夹爪命令步数。 |
| `post_action` | `bool` | 否 | 不读取 | 当前 YAML 中出现，但代码不读取。 |
| `post_action_offset` | `float` | 否 | 不读取 | 当前 YAML 中出现，但代码不读取。 |

---

## B18. `heuristic__skill`

启发式运动。实现：`heuristic_skill.py`

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `mode` | `str` | 否 | `home` | 代码支持 `home`、`abs_qpos`、`rel_qpos`、`rel_ee`；当前观测 `home`、`rel_ee`。 | 启发式动作模式，决定 `value` 被解释成 home、绝对关节、相对关节还是相对 EE 变换；每个值的含义见 E4。 |
| `gripper_state` | `float` | 否 | controller 当前值 | 当前 `1.0`、`-1.0`、`1`。 | 目标夹爪状态。 |
| `move_steps` | `int` | 否 | `50` | 任意正整数。 | 插值步数。 |
| `t_eps` | `float` | 否 | `0.088` | 任意正数。 | 完成阈值。 |
| `value` | `list` 或 `4x4 matrix` | 取决于 mode | mode 默认值 | `rel_ee` 下为 4x4 矩阵。 | `abs_qpos` / `rel_qpos` 下作为目标关节数组；`rel_ee` 下作为 EE 相对变换。 |

---

## B19. `joint__ctrl`

直接关节控制。实现：`joint_ctrl.py`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | 关联对象；当前逻辑不直接使用对象控制。 |
| `ctrl_list` | `list[[joint_index, angle_deg, mode]]` | 是 | 无 | `joint_index` 是受控 arm joint 数组下标；`angle_deg` 用 degree 表示；`mode` 候选为 `abs`、`delta`，`abs` 把该关节设为指定角度，`delta` 在当前角度上叠加。 |
| `num_steps` | `int` | 否 | `10` | 当前观测 `30`、`40`、`60`、`80`、`200`。 |
| `gripper_state` | `float` | 否 | `1.0` | dummy forward 时保持的夹爪状态。 |
| `success_threshold_js` | `float` 或字符串数值 | 否 | `5e-3` | 关节成功阈值。 |

---

## B20. `navigate`

Nav2 移动底盘导航。实现：`navigate.py`

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `goal` | `str` | `goal_x/y/yaw` 未写时必填 | 无 | 当前 `nav_to_pick`、`nav_to_place`、`nav_to_living_room_east`。 | 命名目标；从 task 顶层 `positions[goal]` 读取。优先级高于直接坐标。 |
| `goal_x` / `goal_y` / `goal_yaw` | `float` | `goal` 未写时必填 | 无 | 当前直接坐标任务使用。 | 直接导航目标，world/map 坐标。 |
| `xy_goal_tolerance` | `float` | 否 | `0.10` | 当前 `0.05`、`0.1`、`0.11`。 | legacy 位置容差；可能被 robot base Nav2 goal checker 覆盖。 |
| `yaw_goal_tolerance` | `float` | 否 | `0.10` | 当前 `0.05`、`0.1`。 | legacy 朝向容差；可能被 Nav2 goal checker 覆盖。 |
| `skill_xy_goal_tolerance` / `skill_yaw_goal_tolerance` | `float` | 否 | 同上 | 代码支持。 | legacy 别名。 |
| `startup_timeout_sec` | `float` | 否 | `60.0` | 当前全部 `60.0`。 | Nav2 启动超时。 |
| `runtime_timeout_sec` | `float` | 否 | `240.0` | 当前 `180`、`240`、`300`。 | 单次导航运行超时。 |
| `output_root` | `str` | 否 | `output/ros_bridge/skills` | 当前全部该值。 | skill 输出目录。 |
| `scene_name` | `str` | 否 | task name | 当前多个导航场景名。 | 地图/会话标识。 |
| `map_output_dir` | `str` | 否 | `output/nav2_maps` | 当前少量覆盖。 | 地图输出目录。 |
| `map_resolution` | `float` | 否 | `0.02` | 代码支持。 | 建图分辨率。 |
| `map_z_min` / `map_z_max` | `float` | 否 | `0.0` / `0.35` | 当前 `0.05` / `0.35`。 | 地图生成高度过滤。 |
| `map_include_visual_wall_geometry` | `bool` | 否 | `true` | 当前 `true`。 | 建图是否包含 visual wall geometry。 |

### Example

```yaml
positions:
  nav_to_pick:
    x: -0.08
    y: -0.72
    yaw: 1.5707963267948966

skills:
  - split_aloha:
      - base:
          - id: nav_to_pick
            name: navigate
            depends_on: []
            goal: nav_to_pick
```

---

## B21. Articulation Skills: `open`, `close`, `artpreplan`, `rotate`

这些 skill 都需要 `planner_setting`，并要求 `objects[0]` 指向 `ArticulatedObject`。

### Common Parameters

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | `objects[0]` 为 articulation 对象名。 |
| `planner_setting` | `dict` | 是 | 无 | KPAM planner 配置。 |
| `obj_info_path` | `str` | 否 | 无 | 若提供，skill 开始前调用 `art_obj.update_articulated_info()`。 |
| `collision_valid` | `bool` | 否 | `true` | `open` / `close` 成功判定是否检查禁碰撞。 |
| `process_valid` | `bool` | 否 | `true` | `open` / `close` 成功判定是否检查速度稳定。 |

### `planner_setting`

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `task_name` | `str` | 是 | KPAM 任务名，例如 `OpenBox`、`CloseBox`。 |
| `category_name` | `str` | 是 | 类别名，当前常见 `Articulated`。 |
| `tool_keypoint_name_list` | `list[str]` | 是 | 机器人/工具关键点，如 `tool_head`、`tool_tail`、`tool_side`。 |
| `object_keypoint_name_list` | `list[str]` | 是 | articulation 对象关键点。 |
| `constraint_list` | `list[dict]` | 是 | KPAM 约束列表。 |
| `contact_pose_index` | `int` | 是 | 接触位姿索引。 |
| `pre_actuation_motions` | `list[[motion,value]]` | 否 | 接触前动作。 |
| `post_actuation_motions` | `list[[motion,value]]` | 否 | 接触后动作。 |
| `keypose_random_range.position` | `dict` | 是 | `x_min/x_max/y_min/y_max/z_min/z_max`。 |
| `keypose_random_range.orientation` | `dict` | 是 | `x_min/x_max/y_min/y_max/z_min/z_max`，degree。 |
| `success_threshold` | `float` | 是 | 成功阈值。 |
| `success_mode` | `str` | 否 | `open` 默认 `abs`，还支持 `normal`；`close` 默认 `zero`，还支持 `dis_to_init`。 |
| `update_art_joint` | `bool` | 否 | 是否同步 articulation joint target。 |
| `additional_labels` | `dict` | 否 | `rotate` 用于特定资产覆盖。 |
| `additional_labels.<asset>.modify_actuation_motion` | `[motion,value]` | 否 | 覆盖 planner 的 actuation motion。 |

### `constraint_list[]`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `name` | `str` | 约束名，仅用于标识。 |
| `type` | `point2point_constraint` / `frame_axis_parallel` | KPAM 约束类型。 |
| `tolerance` | `float` | 容差。 |
| `keypoint_name` | `str` | 工具关键点。 |
| `target_keypoint_name` | `str` | 对象关键点。 |
| `axis_from_keypoint_name` / `axis_to_keypoint_name` | `str` | 由工具关键点定义轴。 |
| `target_axis` | `str` | articulation info 中的目标轴名。 |
| `target_axis_frame` | `str` | 目标轴坐标系。 |
| `target_axis_from_keypoint_name` / `target_axis_to_keypoint_name` | `str` | 由对象关键点定义目标轴。 |
| `cross_target_axis1_from_keypoint_name` / `cross_target_axis1_to_keypoint_name` | `str` | cross-axis 约束中的第一条目标轴。 |
| `target_inner_product` | `float` | 轴向内积目标，常见 `1` 或 `-1`。 |

### Skill-specific Notes

| skill | 特殊语义 |
| --- | --- |
| `open` | 成功判定检查 articulation joint 相对初始位置的位移；`success_mode` 默认 `abs`。 |
| `close` | 成功判定可检查 joint 接近 0 或远离初始位置；`success_mode` 默认 `zero`。 |
| `artpreplan` | 主要初始化 planner 和 key pose，不执行完整开关动作。 |
| `rotate` | `planner_setting.success_threshold` 默认 `0.785` rad；可通过 `additional_labels` 改 actuation motion。 |

---

## B22. `rotate__obj`

旋转被抓物体。实现：`rotate_obj.py`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | `objects[0]` 为被旋转对象。 |
| `rotate_obj_euler_delta` | `[[min_xyz],[max_xyz]]` | 是 | 无 | 物体欧拉角增量采样范围，degree。 |
| `success_threshold_move` | `float` 或字符串数值 | 是 | 无 | EE 平移成功阈值。当前 `"5e-3"`。 |
| `success_threshold_rotate` | `float` | 是 | 无 | EE 姿态成功阈值。当前 `0.087`、`0.15`、`0.16`。 |
| `trans_offset` | `[x,y,z]` | 否 | 无 | 最终平移偏移。 |
| `first_motion` | `move` / `rotate` | 否 | 无 | 先平移还是先旋转；`move` 先使用目标位置和当前姿态，`rotate` 先使用当前位置和目标姿态。 |
| `obj_axis_offset` | `list[[axis,value]]` | 否 | 无 | 沿物体局部轴追加偏移；`axis` 候选为 `x`、`y`、`z`。 |
| `move_offset` / `rotate_offset` | `[x,y,z]` | 否 | `[0,0,0]` | 分支偏移。 |
| `rotate_only` | `bool` | 否 | `false` | true 时只旋转，不保持平移跟随。 |
| `dummy_forward` | `dict` | 否 | 无 | 预插值关节动作，含 `ctrl_list`、`num_steps`、`gripper_state`。 |
| `ctrl_list` | `list[[joint_index, angle_deg, mode]]` | `dummy_forward` 分支可用 | `[]` | 用于计算 dummy target joint；格式和 `joint__ctrl.ctrl_list` 相同。 |

---

## B23. `approach__rotate`

接近对象并可选旋转。实现：`approach_rotate.py`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | `objects[0]` 被移动对象，`objects[1]` 被接近对象。 |
| `success_threshold` | `float` 或字符串数值 | 是 | 无 | 平移成功阈值。当前 `"5e-3"`。 |
| `rotate` | `dict` | 是 | 无 | 旋转子配置。 |
| `distance` | `float` | 否 | `0.1` | 距离目标对象的接近距离。当前 `0.05`、`0.16`、`0.2`。 |
| `obj_yaw_offset` | `float` | 否 | `0` | yaw 偏移，degree。当前 `-90`、`0`、`90`、`180`。 |
| `hold_vec_weight` | `list[float]` | 否 | `None` | pose cost 权重。 |
| `approach_axis` | `+x/-x/+y/-y/+z/-z` | 否 | `+x` | 当前观测 `-x`。 |
| `z_offset` | `float` | 否 | `0.0` | 最终 z 偏移。 |
| `obj_axis_offset` | `list[[axis,value]]` | 否 | 无 | 在计算目标前沿被移动对象局部 `x`、`y`、`z` 轴偏移。 |
| `dummy_forward` | `dict` | 否 | 无 | 已弃用；为兼容旧配置允许出现，但会被忽略并发出弃用提示。 |

### `rotate` 子配置

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `type` | `random` / `towards` | `random` 从欧拉角范围采样；`towards` 朝向另一个对象。其他值不会赋值目标姿态，属于无效配置。 |
| `objects` | `list[str]` | `towards` 模式的参照对象列表；代码读取 `objects[1]`。 |
| `rotate_obj_euler` | `[[min_xyz],[max_xyz]]` | `random` 模式欧拉角采样范围，degree；未写时退化为 `[[0,0,0],[0,0,0]]`。 |
| `success_threshold` | `float` | 旋转成功阈值。 |

---

## B24. `flip`

翻转对象。实现：`flip.py`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `objects` | `list[str]` | 是 | 无 | `objects[0]` 为被翻转对象。 |
| `gripper_axis` | `[x,y,z]` | 代码需要 | `false` | 用于构造翻转目标姿态。 |
| `ee_axis` | `[x,y,z]` | 否 | 不读取 | 当前 YAML 中出现，但 `Flip` 代码不读取。 |
| `open_wait_steps` | `int` | 否 | `20` | 张开夹爪保持步数。 |
| `t_eps` / `o_eps` | `float` | 否 | `1e-3` / `5e-3` | 子命令完成阈值。 |

---

## B25. `pour__water__succ`

倒水成功检测。实现：`pour_water_succ.py`

| 字段 | 类型 | 必填 | 默认值 | 当前观测 / 说明 |
| --- | --- | --- | --- | --- |
| `container_name` | `str` | 否 | `cup` | 当前 `redwine_glass`、`plant`、`liquors_glass`、`bowl`、`mug`、`cup`。统计粒子的目标容器。 |
| `particle_num_th_min` / `particle_num_th_max` | `int` | 否 | `50` / `300` | 当前 `150/4000` 或 `50/300`。成功粒子数量范围。 |
| `container_radius` | `float` | 否 | `0.025` | 当前 `0.02`、`0.025`、`0.04`。统计容器 XY 半径。 |
| `container_up` | `list[[container,axis,threshold]]` | 否 | `[]` | 额外朝向约束，axis 为 `x`、`y`、`z`。 |
| `translation` | `[x,y,z]` | 否 | 当前 EE 位置 | 目标位置；当前实现只生成 dummy forward，不完成倒水轨迹。 |
| `quaternion` / `euler` | quaternion / euler | 否 | 当前 EE 姿态 | 目标姿态。 |
| `max_noise_m` / `max_noise_deg` | `float` | 否 | `0.05` / `5` | 目标扰动。 |
| `gripper_state` | `float` | 否 | `1.0` | dummy forward 阶段夹爪状态。 |
| `gripper` | `str` | 否 | 不读取 | 当前 YAML 中有 `close`，代码不读取。 |

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
| `update_freq` | `int` | 否 | `10` | 当前观测 `5000`。 | `individual_reset()` 中 scene 随机化刷新周期。 |
| `involved_scenes` | `str` | 否 | 无 | 当前 `dining_room_scene_info`、`study_room_scene_info`、`living_room_scene_info`。 | scene-pair 随机化读取 `<name>.json`。逗号分隔多个前缀。 |

## C3. `fixtures[]`

fixture 使用 object 注册表加载，字段基本与 task `objects[]` 一致。当前目标 arena 只使用 `GeometryObject`、`PlaneObject`、`ConveyorObject`。

| 字段 | 类型 | 必填 | 默认值 | 允许值 / 当前观测 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | 任意唯一 fixture 名。 | 可被 `regions[].target` 或 skill object 引用。 |
| `target_class` | `str` | 是 | 无 | 当前 `GeometryObject`、`PlaneObject`、`ConveyorObject`。 | fixture 对象类注册名，决定 arena 元素是加载静态几何、生成程序平面还是加载传送带；每个值的含义见 E2。 |
| `path` | `str` | 资源型 fixture 必填 | 无 | 几何或传送带 USD。 | 相对引用 task 的 `asset_root`。 |
| `translation` | `[x,y,z]` | 是 | 无 | 当前全部显式配置。 | 初始位置。 |
| `euler` | `[r,p,y]` | `euler` / `quaternion` 二选一 | 无 | 当前大量使用。 | 初始朝向，degree。 |
| `quaternion` | `[w,x,y,z]` | `euler` / `quaternion` 二选一 | 无 | 当前少量使用。 | 初始朝向。 |
| `scale` | `[sx,sy,sz]` | 资源型常用 | `[1,1,1]` | 当前多数资源型 fixture 显式配置。 | 缩放。 |
| `size` | `[width,length]` | `PlaneObject` 必填 | 无 | 当前 floor/background 使用。 | 平面大小。 |
| `texture` | `dict` | 否 | 无 | 见 A6 Texture Block。 | 加载后应用材质。 |
| `apply_randomization` | `bool` | 否 | `false` | 当前少量 fixture 为 `true`。 | scene-pair / hearth 随机化开关，不是通用位姿随机化。 |
| `visible` | `bool` | 否 | `true` | 当前少量 `false`。 | 初始可见性。 |
| `collision_enabled` | `bool` | 否 | class-specific | 当前 `true`、`false`。 | 生成/启用碰撞。 |
| `collision_thickness` | `float` | `PlaneObject` 可选 | `0.02` | 当前 `0.02`。 | 平面碰撞厚度。 |
| `collision_approximation` | `str` | `GeometryObject` 可选 | `bbox` | 当前生成器常用 `bbox`。 | 当前只支持 `bbox`。 |
| `collision_visible` | `bool` | 否 | `false` | 当前少量使用。 | 是否显示碰撞 proxy。 |
| `linear_velocity` | `[vx,vy,vz]` | `ConveyorObject` 必填 | 无 | 当前 2 个传送带 fixture。 | 线速度。 |
| `linear_track_list` | `list[str]` | `ConveyorObject` 必填 | 无 | 当前如 `track_01`、`track_03`。 | 线性轨道名。 |
| `angular_velocity` | `[wx,wy,wz]` | `ConveyorObject` 必填 | 无 | 当前 2 个传送带 fixture。 | 角速度。 |
| `angular_track_list` | `list[str]` | `ConveyorObject` 必填 | 无 | 当前如 `track_02`、`track_04`。 | 角轨道名。 |

## C4. Scene 随机化

`arena.apply_randomization` 的含义只限于旧 scene 随机化逻辑：

| 条件 | 行为 |
| --- | --- |
| arena 有且只有 2 个 fixture，且两个 fixture 都 `apply_randomization: true` | `update_scene_pair()` 读取 `involved_scenes` 对应 JSON，替换 table 和 scene fixture 的 `path`、`target_class`、`scale`、`translation`、`euler`。 |
| task objects 中有名称包含 `hearth` 的 object，arena 中名为 `scene` 的 fixture 设置 `apply_randomization: true` | `update_hearths()` 从 `HEARTH_KITCHENS` 随机替换 scene fixture。 |

确定性 arena 不应设置 `involved_scenes`，也不应给 fixture 设置 `apply_randomization: true`。

---

# Part D. 未读取或元数据字段

以下字段在目标 YAML 中出现，但当前主运行链路不读取，或只作为生成器/语义元数据保留：

| 字段 | 位置 | 状态 | 说明 |
| --- | --- | --- | --- |
| `scene_prim_path` | `objects[]` | 元数据 | `assets_addition` 原始 scene prim path。 |
| `scene_reference` | `objects[]` | 元数据 | `assets_addition` 原始 scene reference。 |
| `filter_collision` | `objects[]` | 未读取 | 当前主加载链路不读取。 |
| `regions[].visible` | `regions[]` | 未读取 | 当前摆放逻辑不读取。 |
| `dexpick.post_grasp_offset` | `skills[]` | 未读取 | `Dexpick` 读取 `post_grasp_offset_min/max`，不读取该字段。 |
| `flip.ee_axis` | `skills[]` | 未读取 | `Flip` 读取 `gripper_axis`。 |
| `gripper__action.post_action` | `skills[]` | 未读取 | 当前 `Gripper_Action` 不读取。 |
| `gripper__action.post_action_offset` | `skills[]` | 未读取 | 当前 `Gripper_Action` 不读取。 |
| `manualpick.update_pose_cost_metric_none` | `skills[]` | 未读取 | 当前 `Manualpick` 不读取。 |
| `pour__water__succ.gripper` | `skills[]` | 未读取 | 当前 `Pour_Water_Succ` 不读取。 |
| `track.target` | `skills[]` | 语义字段 | 当前 `Track` 主轨迹生成不使用它。 |
| `camera_file.camera_type` | camera config | 元数据 | `CustomCamera` 不读取。 |
| `camera_file.frequency` | camera config | 元数据 | `CustomCamera` 不读取。 |

---

# Part E. 候选值索引

本节集中列出“有限候选值”。对象名、fixture 名、region 名、skill `id`、`depends_on`、`positions` key、`scene_name` 等属于用户自定义标识符，不在这里当作枚举列出；它们的约束是“必须能在当前 task 的命名表里解析到”。

## E1. 配置文件路径候选值

### `robots[].robot_config_file`

| 当前 YAML 中出现的完整值 | 对应实例名 | 说明 |
| --- | --- | --- |
| `workflows/simbox/core/configs/robots/genie1.yaml` | `genie1` | Genie1 机器人基础配置。 |
| `workflows/simbox/core/configs/robots/lift2.yaml` | `lift2`、`lift2_0`、`lift2_1` | Lift2 机器人基础配置，可实例化多个命名机器人。 |
| `workflows/simbox/core/configs/robots/split_aloha.yaml` | `split_aloha` | Split Aloha 机器人基础配置，常带移动底盘。 |
| `workflows/simbox/core/configs/robots/fr3.yaml` | `franka` | FR3 / Franka 类机械臂配置。 |
| `workflows/simbox/core/configs/robots/franka_robotiq85.yaml` | `frankarobotiq` | Franka + Robotiq85 夹爪配置。 |

同目录还存在 `tracer2_franka.yaml`，但在本文目标 task YAML 中没有出现。

### 机器人 `target_class`

`target_class` 通常来自 `robot_config_file`，task 内一般不直接写。它决定创建哪个 robot Python 类，以及 robot 对象提供哪些 arm、gripper、base 接口。

| 值 | 详细含义 |
| --- | --- |
| `SplitAloha` | Split Aloha 双臂机器人类，通常带 `left` / `right` 两个机械臂 controller，也可通过 `base` 配置使用移动底盘。 |
| `Lift2` | Lift2 机器人类，当前 task 中可用 `lift2`、`lift2_0`、`lift2_1` 等实例名创建多个 Lift2 实例。 |
| `Genie1` | Genie1 机器人类，使用 Genie1 资产和对应关节/home/gripper 配置。 |
| `FR3` | Franka Research 3 / Franka 类机械臂配置，当前实例名常写作 `franka`。 |
| `FrankaRobotiq85` | Franka 机械臂加 Robotiq85 夹爪的组合机器人类。 |
| `TemplateRobot` | 模板机器人类，供新 robot 配置扩展使用；目标 task YAML 中未观测到直接使用。 |

### `cameras[].camera_file`

| 当前 YAML 中出现的完整值 | 说明 |
| --- | --- |
| `workflows/simbox/core/configs/cameras/realsense_d455_v3.yaml` | D455 v3 相机参数，当前使用最多。 |
| `workflows/simbox/core/configs/cameras/realsense_d405_v2.yaml` | D405 v2 相机参数。 |
| `workflows/simbox/core/configs/cameras/realsense_d405.yaml` | D405 相机参数。 |
| `workflows/simbox/core/configs/cameras/astra.yaml` | Astra 相机参数。 |
| `workflows/simbox/core/configs/cameras/realsense_d435i.yaml` | D435i 相机参数。 |

同目录还存在 `realsense_d455.yaml`、`realsense_d455_v2.yaml`，但在本文目标 task YAML 中没有出现。

## E2. Object / Arena 类名候选值

| 值 | 可用于 | 详细含义 |
| --- | --- | --- |
| `RigidObject` | `objects[].target_class`，代码也支持 fixture | 加载 USD 后把 `prim_path_child` 指向的子 prim 包装成 Isaac `RigidPrim`，具有刚体状态、质量、碰撞和可抓取/可移动语义。适合杯子、瓶子、盒子等需要物理交互的物体。 |
| `GeometryObject` | `objects[].target_class`、`fixtures[].target_class` | 加载 USD 为静态/几何 prim，可选 `collision_enabled: true` 生成 bbox 碰撞代理。适合桌子、背景场景、静态障碍物和视觉几何。 |
| `ArticulatedObject` | `objects[].target_class`，代码也支持 fixture | 加载带关节的 USD，并读取 `Kps/<info_name>/info.json` 作为 articulation 关键点信息。适合抽屉、柜门、微波炉、纸箱盖等需要 `open`/`close`/`artpreplan` 操作的对象。 |
| `PlaneObject` | `fixtures[].target_class`，代码也支持 task object | 程序生成 USD 平面，不依赖外部 USD；`size` 控制宽和长，可选生成薄 box 碰撞体。适合 floor、background plane。 |
| `ConveyorObject` | `fixtures[].target_class`，代码也支持 task object | 加载传送带 USD，并按 `linear_track_list` / `angular_track_list` 给轨道对象施加线速度或角速度。适合动态输送场景。 |
| `BoxObject` | 代码已注册，目标 YAML 未出现 | 程序生成 cube，主要用于简单 box primitive；通过 `scale` 控制尺寸，通过 `color` 控制颜色。 |
| `ShapeObject` | 代码已注册，目标 YAML 未出现 | 低层 `GeometryPrim` 包装；当前创建逻辑较薄，通常不作为 arena/task 配置首选。 |
| `XFormObject` | 代码已注册，目标 YAML 未出现 | 加载 USD 为 xform 容器，不按刚体或 articulation 管理；可用 `parent_obj` 挂到已有对象下。 |

### 资源元信息值

| 字段 | 当前值 / 候选值 | 详细含义 |
| --- | --- | --- |
| `texture.texture_lib` | `background_textures`、`floor_textures`、`table_textures`、`val2017`、`light_table_textures`、`dark_table_textures` | 相对 `asset_root` 的纹理目录名。随机纹理会在该目录下排序后的文件列表中采样；固定纹理用 `texture_id` 取排序后的第 N 个文件。 |
| `objects[].dataset` | `assets_addition`、`oo3d`、`gso`、`pm`、`arcode`、`grutopia`、`gr` | 数据来源标签。主加载链路不按它分支，但它用于资产来源追踪、生成器元数据和人工理解类别来源。 |
| `objects[].prim_path_child` | `Aligned`、`root`，也可写其他 USD 子 prim 名 | 指向对象 USD 内实际被包装的子 prim。`Aligned` 通常表示对齐后的 mesh/rigid prim；`root` 通常表示 USD 根下名为 root 的子 prim。写错会导致 `get_prim_at_path()` 找不到目标。 |

## E3. 随机化候选值

### `randomization_scope`

| 值 | 类型 | 语义 |
| --- | --- | --- |
| `category` | `str` | 在当前对象所在类别目录内随机选择资产。例如当前 `path` 属于 `omniobject3d-cup`，则只在该类别的实例目录里选 USD。用于保持语义类别不变，只换具体实例。 |
| `full` | `str` | 在当前资源库的所有类别下随机选择资产。用于语义类别也可变化的强随机化；随机后 `category` 会被替换成新类别的人类可读名称。 |
| `<category list>` | `list[str]` | 只在列表列出的类别目录中随机选择资产。用于把随机范围限制在一组允许类别内，例如只在若干食材/日用品之间随机。列表元素必须是实际资产目录名。 |

当前目标 YAML 中出现的 `<category list>` 完整内容如下：

```yaml
randomization_scope:
  - omniobject3d-bamboo_shoots
  - omniobject3d-banana
  - omniobject3d-bottle
  - omniobject3d-carrot
  - omniobject3d-chili
  - omniobject3d-corn
  - omniobject3d-cucumber
  - omniobject3d-pen
  - omniobject3d-shampoo
  - omniobject3d-scissor
  - omniobject3d-sweet_potato
  - omniobject3d-tooth_brush
  - omniobject3d-tooth_paste
  - phocal-bottle
  - phocal-spoon
```

### 姿态和缩放模式

| 字段 | 值 | 语义 |
| --- | --- | --- |
| `orientation_mode` | `keep` | 保留 object 配置中的 `euler`。缺少 `euler` 会触发断言。 |
| `orientation_mode` | `suggested` | 通过类别查推荐姿态；类别未命中时回退到 `[0,0,0]` 并打印 warning。 |
| `orientation_mode` | `random` | 在 `[-180,180]` degree 内独立采样 roll/pitch/yaw。 |
| `scale_mode` | `keep` | 保留 object 配置中的 `scale`。缺少 `scale` 会触发断言。 |
| `scale_mode` | `suggested` | 使用类别推荐缩放，若对象级推荐存在则对象级覆盖类别级。 |

## E4. Skill 通用枚举

### `filter_x_dir` / `filter_y_dir` / `filter_z_dir`

这些字段用于筛选候选 EE 姿态。格式有两种：

| 格式 | 语义 |
| --- | --- |
| `[direction, angle]` | 要求指定 EE 轴与指定世界方向的夹角落在单侧阈值内。 |
| `[direction, min_angle, max_angle]` | 要求夹角落在一个角度区间内。 |

direction 候选值由 skill 实现决定：

| skill | `x` 轴候选 direction | `y` / `z` 轴候选 direction |
| --- | --- | --- |
| `pick`、`manualpick`、`dynamicpick`、`fail_pick` | `forward`、`backward`、`upward`、`downward` | `forward`、`backward`、`upward`、`downward` |
| `place`、`goto__pose` | `forward`、`backward`、`leftward`、`rightward`、`upward`、`downward` | `forward`、`backward`、`leftward`、`rightward`、`upward`、`downward` |

方向含义是相对 base/world 旋转矩阵元素的约束：`forward/backward` 对应 x 方向正/负，`leftward/rightward` 对应 y 方向正/负，`upward/downward` 对应 z 方向正/负。不是所有 skill 都支持 `leftward/rightward`；在不支持的 skill 中写入会 KeyError。

### 其他常用枚举

| 字段 | 候选值 | 语义 |
| --- | --- | --- |
| `test_mode` | `forward` | 对候选 EE pose 调用 controller 的前向规划/可执行性检查。更接近实际运动规划，但通常比纯 IK 筛选更重。 |
| `test_mode` | `ik` | 对候选 EE pose 做 IK 可达性筛选。适合只想快速排除不可达姿态的配置。 |
| `direction_to_obj` | `left` | 要求 EE 的 y 坐标在物体 y 坐标的正侧，代码判定为 `EE_y > obj_y`。 |
| `direction_to_obj` | `right` | 要求 EE 的 y 坐标在物体 y 坐标的负侧或相等，代码判定为 `EE_y <= obj_y`。 |
| `gripper_state` / `final_gripper_state` | `1` / `1.0` | 打开夹爪，生成 `open_gripper` 命令。 |
| `gripper_state` / `final_gripper_state` | `-1` / `-1.0` | 闭合夹爪，生成 `close_gripper` 命令。 |
| `heuristic__skill.mode` | `home` | 目标关节位为当前手臂的 home joint。常用于阶段开始/结束时回到安全姿态。 |
| `heuristic__skill.mode` | `abs_qpos` | `value` 解释为目标关节数组，直接作为绝对关节目标。 |
| `heuristic__skill.mode` | `rel_qpos` | `value` 解释为相对关节增量数组，在当前关节位基础上叠加。 |
| `heuristic__skill.mode` | `rel_ee` | `value` 解释为 4x4 EE 相对变换矩阵，先算目标 EE pose，再通过 controller plan 求目标关节。 |
| `joint__ctrl.ctrl_list[].mode` | `abs` | `angle_deg` 是目标绝对关节角，代码会转换成 rad 后覆盖该关节。 |
| `joint__ctrl.ctrl_list[].mode` | `delta` | `angle_deg` 是相对角度增量，代码会转换成 rad 后加到当前关节角上。 |
| `place.place_direction` | `vertical` | 从目标 bbox 顶面上方采样 pre-place/place 点：x/y 由比例范围确定，z 为 `bbox.max.z + pre_place_z_offset/place_z_offset`。适合从上往下放。 |
| `place.place_direction` | `horizontal` | 在目标 bbox 内采样一个基准点，再沿 `align_place_obj_axis` 和 `offset_place_obj_axis` 做水平接近/放置偏移。适合侧向插入、横向摆放。 |
| `place.position_constraint` | `gripper` | 把采样点当作夹爪目标位置，物体位置由当前 grasp 关系间接决定。 |
| `place.position_constraint` | `object` | 把采样点当作被放物体的目标位置，再反推夹爪 pose。适合要求物体本体落到精确位置的放置。 |
| `place.success_mode` | `3diou` | 计算被放物体 bbox 与目标 bbox 的 3D IoU，要求 IoU 大于阈值。 |
| `place.success_mode` | `height` | 判断物体相对 robot base 的 z 是否低于 place EE 高度一定距离，偏向“已放下/高度降低”的判定。 |
| `place.success_mode` | `xybbox` | 判断被放物体中心 x/y 是否落在目标 bbox 的 x/y 范围内，并留出 0.015 m 边距。 |
| `place.success_mode` | `left` | 判断被放物体 x 是否小于目标 bbox 左边界减 `threshold`，用于“放在左侧”。 |
| `place.success_mode` | `right` | 判断被放物体 x 是否大于目标 bbox 右边界加 `threshold`，用于“放在右侧”。 |
| `place.success_mode` | `flower` | 先要求物体中心 x/y 在目标 bbox 内，再要求 3D IoU 大于 `success_th`。 |
| `place.success_mode` | `cup` | 要求杯类物体底部 z 高于目标底部 `0.05` m，且 IoU 大于 `success_th`。 |
| `open.planner_setting.success_mode` | `abs` | 打开成功条件为 articulation joint 相对初始值的绝对位移大于等于 `success_threshold`。正负方向都算成功。 |
| `open.planner_setting.success_mode` | `normal` | 打开成功条件为 `current_joint - initial_joint >= abs(success_threshold)`，只接受正方向打开。 |
| `close.planner_setting.success_mode` | `zero` | 关闭成功条件为当前 articulation joint 绝对值小于等于 `success_threshold`，即接近 0 位。 |
| `close.planner_setting.success_mode` | `dis_to_init` | 关闭成功条件为 articulation joint 相对初始值的绝对位移大于等于 `success_threshold`，适合“远离初始状态”式判定。 |
| `rotate__obj.first_motion` | `move` | 先移动到目标平移位置但保持当前姿态，再执行最终目标 pose。 |
| `rotate__obj.first_motion` | `rotate` | 先在当前位置附近旋到目标姿态，再执行最终目标 pose。 |
| `rotate__obj.obj_axis_offset[].axis` | `x`、`y`、`z` | 沿被旋转对象局部坐标轴偏移目标物体坐标系；常用于让旋转围绕对象上某个偏置点发生。 |
| `approach__rotate.approach_axis` | `+x`、`-x`、`+y`、`-y`、`+z`、`-z` | 指定被移动对象的哪条局部轴作为接近方向。`+x` 表示沿对象局部 x 正向，`-x` 表示反向；其他轴同理。 |
| `approach__rotate.rotate.type` | `random` | 从 `rotate_obj_euler` 范围采样目标欧拉角，作为被移动对象的新朝向。 |
| `approach__rotate.rotate.type` | `towards` | 读取 `rotate.objects[1]` 指向的对象位置，计算一个让被移动对象朝向该对象的 yaw。 |
| `container_up[].axis` | `x`、`y`、`z` | 倒水成功检测中的容器朝上轴；代码用该轴和阈值判断容器姿态是否满足约束。 |
| `camera_axes` | `usd` | 使用 Isaac Camera 的 USD 坐标轴约定设置相机 pose。当前目标 YAML 只使用该值。 |
| `cameras[].output_mode` | `rgb` | 输出普通 RGB 图像，即 camera RGBA frame 的前三个通道。 |
| `cameras[].output_mode` | `diffuse_albedo` | 输出 diffuse albedo annotator 图像，颜色更接近材质反照率，弱化光照影响。 |
| `env_map.light_type` | `DomeLight` | 创建或复用 task root 下的 USD DomeLight，并设置 HDR texture、强度和旋转。其他值不会创建 DomeLight。 |

## E5. Region Sampler 候选值

| `random_type` | 当前 YAML 是否出现 | 必要参数 | 详细含义 |
| --- | --- | --- | --- |
| `A_on_B_region_sampler` | 是 | `pos_range`、`yaw_rotation` | “A 放在 B 上”。以 B 的 bbox 中心作为 x/y 基准，以 B 的 bbox 顶面作为 z 基准，再补偿 A 的底面高度，最后加上 `pos_range` 采样偏移；适合桌面、托盘、架子上放物。 |
| `A_in_B_region_sampler` | 是 | `x_bias`、`y_bias`、`z_bias` 可选 | “A 放进/放到 B 的中心上沿”。x/y 直接取 B 的 local pose 中心再加 bias，z 对齐到 B bbox 顶面附近；不做随机 yaw，也不检查 A 是否真的完全落在 B 内。 |
| `A_by_B_region_sampler` | 否 | `pos_range`、`yaw_rotation` | “A 放在 B 旁边的矩形区域”。以 B 的 bbox 中心为原点，在 `pos_range` 的 x/y 范围内随机偏移；z 用 A/B bbox 半高差让底部近似对齐。适合地面或同高度平台上的邻近摆放。 |
| `A_by_B_circle_sampler` | 否 | `r_range`、`theta_range`、`yaw_rotation` | “A 放在 B 周围的圆环/扇形区域”。从 `r_range` 采样半径、从 `theta_range` 采样极角，计算 B 周围的 x/y；z 与 `A_by_B_region_sampler` 同样按半高差对齐。 |
| `A_face_B_circle_sampler` | 否 | `r_range`、`yaw_rotation` | “A 放在 B 正面方向上”。读取 B 当前世界 yaw，以该方向为射线，从 `r_range` 采样距离；适合需要跟随 B 朝向的前方摆放。 |
| `A_along_B_C_circle_sampler` | 否 | `r_range`、`yaw_rotation`，外层 `target2` | “A 沿 B 的朝向，放在越过 C 之后”。先计算 B 到 C 的世界距离，再加上 `r_range` 采样距离，方向仍使用 B 当前 yaw；适合把 A 放在 B-C 连线尺度之外的延长位置。 |

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
