# SimBox 配置参考

本文把 YAML 配置项映射到实际消费它们的代码路径。

## 阅读说明

- 字段名、类名、文件路径保持源码原文，便于直接全局搜索。
- 文档按三层组织：
  - `Task YAML`：任务侧配置，包括对象、相机、skills、语言和数据输出。
  - `Arena YAML`：场景静态元素和贴图/场景随机化。
  - `configs/de*.yaml`：数据引擎管线配置。
- 对 skill 章节，统一按照“注册名 / 实现 / 功能 / 参数”说明。
- 即使某些字段当前没有被主链路触发，只要源码中存在读取逻辑，本文仍会保留。

## 1. Task YAML

Task YAML 由 `workflows/simbox/utils/task_config_parser.py` 解析，并由 `workflows/simbox_dual_workflow.py` 和 `workflows/simbox/core/tasks/banana.py` 执行。

文件顶层必须是：

```yaml
tasks:
  - name: my_task
    ...
```

`TaskConfigParser.parse_tasks()` 会逐个返回 `tasks[]` 内的 task dict。解析器不会过滤未知字段；未知字段会保留在 `task.cfg` 中，但只有对应代码分支读取时才有运行语义。要写出能直接运行的 task，需要同时满足：

- `arena_file` 指向一个可读取的 Arena YAML。
- `asset_root` 能解析所有 object、fixture、texture、camera 文件中的相对路径。
- `robots`、`objects`、`regions`、`cameras`、`skills`、`data` 这些键存在；scene-only task 可以使用空列表。
- 如果 `skills` 非空，skill 引用的 robot、controller、object 名必须在当前 task/arena 中存在。
- 如果 `regions` 非空，`object` 和 `target` 必须能在 task objects、arena fixtures、robots 或 cameras 中找到。

### 顶层键

| Key | 值类型 / 候选值 | 消费位置 | 说明 |
| --- | --- | --- | --- |
| `name` | `str` | `BananaBaseTask`、日志、输出命名 | 必填。 |
| `asset_root` | `str` | 物体/相机加载、资源查找、distractor | 必填。 |
| `task` | `str`, currently `BananaBaseTask` | `get_task_cls(task)` | 必填，必须存在于任务注册表中。 |
| `task_id` | `int` or `str` | root prim path、场景命名空间 | 必填；当前配置通常用整数。 |
| `offset` | `null` or `list[float]`, len=3 | `BaseTask.__init__` | 当前任务类要求必填；常见为 `null`。 |
| `render` | `bool` | `BananaBaseTask._render` | 是否启用相机观测。 |
| `arena_file` | `str` | `SimBoxDualWorkFlow.reset()` | 必填，会加载到 `cfg["arena"]`。 |
| `neglect_collision_names` | `list[str]` | `SimBoxDualWorkFlow.reset()` | 可选，碰撞过滤。 |
| `env_map` | `dict` | `BananaBaseTask._set_envmap()` | 可选。 |
| `robots` | `list[dict]` | `SimBoxDualWorkFlow._initialize_controllers()`、`BananaBaseTask._load_robot()` | 必填键；scene-only task 可为空列表。 |
| `objects` | `list[dict]` | `update_rigid_objs()`、`update_articulated_objs()`、`_load_obj()` | 必填键；只加载 arena 的 scene-only task 可为空列表。 |
| `regions` | `list[dict]` | `BananaBaseTask._set_regions()`、`optimize_2d_manip_layout()` | 必填键；无需随机摆放时可为空列表。 |
| `random_region_list` | `list[dict]` | `BananaBaseTask._set_regions()` | 可选，供 `regions[].priority` 复用的随机区域候选池。 |
| `positions` | `dict[str, dict{x:float, y:float, yaw:float}]` | `navigate` skill | 可选，命名导航点位表；每个点通常含 `x/y/yaw`。 |
| `cameras` | `list[dict]` | `BananaBaseTask._load_camera()`、`_set_camera_poses()` | 必填键；不渲染或 scene-only task 可为空列表。 |
| `distractors` | `dict` | `BananaBaseTask._create_distractor_cfg()`、`set_distractors()` | 可选。 |
| `fluid` | `dict` | `BananaBaseTask._set_fluid()`、`SimBoxDualWorkFlow` 重置钩子 | 可选。 |
| `skills` | `list[dict]` | `SimBoxDualWorkFlow._initialize_skills()` | 必填键；只加载场景不执行技能时可为空列表。 |
| `data` | `dict` | logger、语言、episode 长度、collect-info | 必填。 |

### 最小可运行 Task 模板

这是 scene-only 模板，只加载 arena 和光照，不加载机器人、不执行技能。它适合检查 arena 是否能被加载。`arena_file`、`asset_root` 和 `env_map.envmap_lib` 必须按实际项目路径调整。

```yaml
tasks:
  - name: my_scene_only_task
    asset_root: workflows/simbox/assets
    task: BananaBaseTask
    task_id: 0
    offset: null
    render: False
    arena_file: workflows/simbox/core/configs/arenas/my_arena.yaml

    env_map:
      envmap_lib: envmap_lib
      apply_randomization: False
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
      update: True
      max_episode_length: 100
```

带机器人、物体、区域、相机和技能的 task 至少应在此基础上补齐 `robots[]`、`objects[]`、`regions[]`、`cameras[]`、`skills[]`。常见的移动操作写法可参考 `workflows/simbox/core/configs/tasks/mobile_manipulation/split_aloha/navigate_pick_place_boxed_beverage.yaml`。

### 可运行 Task / Arena 对照示例

下面是一组最小“有 arena + 有 object + 有 region”的配置。它不执行技能，但会加载 table、floor 和一个刚体 object，并把 object 放到 table 上。保存为两个文件后，把 task 文件路径传给数据引擎的 `scene_loader.args.cfg_path` 即可进入正常加载流程。

Arena 文件，例如 `workflows/simbox/core/configs/arenas/my_pick_arena.yaml`：

```yaml
name: my_pick_arena
fixtures:
  - name: table
    path: table0/instance.usd
    target_class: GeometryObject
    translation: [0.0, 0.0, 0.375]
    euler: [0.0, 0.0, 0.0]
    scale: [0.001, 0.001, 0.001]
  - name: floor
    target_class: PlaneObject
    size: [5.0, 5.0]
    translation: [0.0, 0.0, 0.0]
    collision_enabled: true
    collision_thickness: 0.02
```

Task 文件，例如 `workflows/simbox/core/configs/tasks/example/my_pick_scene.yaml`：

```yaml
tasks:
  - name: my_pick_scene_task
    asset_root: workflows/simbox/example_assets
    task: BananaBaseTask
    task_id: 0
    offset: null
    render: False
    arena_file: workflows/simbox/core/configs/arenas/my_pick_arena.yaml

    env_map:
      envmap_lib: envmap_lib
      apply_randomization: False
      intensity_range: [5000, 5000]
      rotation_range: [0, 0]

    robots: []
    objects:
      - name: pick_object
        path: task/sort_the_rubbish/non_recyclable_garbage/obj_1/Aligned_obj.usd
        target_class: RigidObject
        dataset: oo3d
        category: bottle
        prim_path_child: Aligned
        translation: [0.0, 0.0, 0.0]
        euler: [0.0, 0.0, 0.0]
        scale: [1.0, 1.0, 1.0]
        apply_randomization: False

    regions:
      - object: pick_object
        target: table
        random_type: A_on_B_region_sampler
        random_config:
          pos_range:
            - [0.0, 0.0, 0.0]
            - [0.0, 0.0, 0.0]
          yaw_rotation: [0.0, 0.0]

    cameras: []
    skills: []

    data:
      task_dir: debug/my_pick_scene_task
      language_instruction: Load one object on the table.
      detailed_language_instruction: Load one object on the table.
      collect_info: my_pick_scene_task
      version: v1.0
      update: True
      max_episode_length: 100
```

这个例子里的名字绑定关系是实际运行会检查的关系：`arena_file` 指向 arena 文件；`regions[].target: table` 必须匹配 arena fixture `name: table`；`regions[].object: pick_object` 必须匹配 task object `name: pick_object`；`RigidObject.prim_path_child: Aligned` 必须存在于该 USD 内。

### `env_map`

由 `BananaBaseTask._set_envmap()` 消费。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `envmap_lib` | `str` | `asset_root` 下的 HDR 目录。 |
| `apply_randomization` | `bool` | 为 true 时随机采样 HDR / intensity / rotation，否则走确定性默认值。 |
| `intensity_range` | `list[float]`, len=2 | 随机强度范围。 |
| `rotation_range` | `list[float]`, len=2 | 随机欧拉角范围；当前实现会对 xyz 三轴分别从同一范围采样。 |
| `light_type` | `str`, currently `DomeLight` | 可选，默认 `DomeLight`。 |

### `robots`

由 `SimBoxDualWorkFlow._initialize_controllers()` 和 `BananaBaseTask._load_robot()` 消费。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `name` | `str` | 机器人实例名。 |
| `target_class` | `str`, one of `TemplateRobot` / `FR3` / `FrankaRobotiq85` / `Genie1` / `Lift2` / `SplitAloha` | 注册表中的机器人类 key。 |
| `robot_config_file` | `str` | 外部机器人 YAML，会合并进任务 robot 配置。 |
| `robot_file` | `str` or `list[str]` | 每个 controller 使用的 CuRobo/机器人配置文件。双臂配置里常为左右各一个路径。 |
| `path` | `str` | 可选，覆盖机器人 USD 资源。 |
| `translation` | `list[float]`, len=3 | 初始位姿平移。 |
| `euler` | `list[float]`, len=3, degrees | 初始朝向。 |
| `quaternion` | `list[float]`, quaternion `[w, x, y, z]` | 初始朝向。 |
| `scale` | `list[float]`, len=3 | 初始缩放。 |
| `constrain_grasp_approach` | `bool` | 传给 controller 构造函数。 |
| `collision_activation_distance` | `float` | 传给 controller 构造函数。 |
| `ignore_substring` | `list[str]` | 传给 controller 构造函数。 |
| `use_batch` | `bool` | 传给 controller 构造函数。 |
| `left_joint_home` / `right_joint_home` | `list[float]` | `TemplateRobot` 使用的初始关节位。 |
| `left_joint_home_std` / `right_joint_home_std` | `list[float]` | 关节初始噪声。 |

### `objects`

由 `update_rigid_objs()`、`update_articulated_objs()`、`_load_obj()`、`optimize_2d_manip_layout()` 和 contact-view 构建逻辑消费。

通用字段：

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `name` | `str` | 物体实例名。 |
| `path` | `str` | 相对于 `asset_root` 的资源路径。 |
| `target_class` | `str`, one of `ArticulatedObject` / `BoxObject` / `ConveyorObject` / `GeometryObject` / `PlaneObject` / `RigidObject` / `ShapeObject` / `XFormObject` | 对象注册表中的类名。 |
| `dataset` | `str`, common values `oo3d` / `gso` / `phocal` / `real` | 仅元数据，除非下游工具额外使用。 |
| `category` | `str` | 会被随机化工具和语言替换逻辑更新。 |
| `prim_path_child` | `str` | 刚体资源的子 prim path。 |
| `translation` | `list[float]`, len=3 | 初始位姿平移。 |
| `euler` | `list[float]`, len=3, degrees | 初始朝向。 |
| `quaternion` | `list[float]`, quaternion `[w, x, y, z]` | 初始朝向。 |
| `scale` | `list[float]`, len=3 | 初始缩放。 |
| `visible` | `bool` | 初始可见性。 |
| `texture` | `dict` | 传给 `apply_texture()`。 |
| `apply_randomization` | `bool` | 启动物体类型对应的随机化逻辑。 |
| `mass` | `float` | 可选，仅刚体类对象消费。 |
| `color` | `list[float]`, len=3 | 可选，`ShapeObject` 使用的 RGB 颜色。 |
| `parent_obj` | `str` | 可选，`XFormObject` 挂载到父对象 prim 下。 |
| `scene_prim_path` | `str` | 可选元数据，`assets_addition` 生成器记录原始 `scene.usd` prim path；当前运行时不直接消费。 |
| `scene_reference` | `str` | 可选元数据，`assets_addition` 生成器记录原始 scene reference；当前运行时不直接消费。 |
| `optimize_2d_layout` | `bool` | 可选，供 `optimize_2d_manip_layout()` 使用的布局开关。 |

随机化相关字段：

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `orientation_mode` | `str`, `keep` / `suggested` / `random` | 刚体随机化朝向模式。 |
| `scale_mode` | `str`, `keep` / `suggested` | 刚体缩放模式。 |
| `randomization_scope` | `str` or `list[str]`, `category` / `full` / 分类列表 | 刚体随机化采样范围。 |
| `gap` | `list[float]` or `float` | 可选 gap 元数据，从 `gap.yaml` 读取。 |
| `info_name` | `str` | articulation 信息查询必填。 |
| `obj_info_path` | `str` | articulation 物体会自动设置。 |
| `joint_position_range` | `list[float]`, len=2 | 若存在，`ArticulatedObject.initialize()` 会采样关节位置。 |
| `strict_init` | `dict` | articulation 严格初始化配置，含 `joint_positions` 和 `joint_indices`。 |
| `fix_base` | `bool` | 固定 articulation 基座。 |

按 `target_class` 的硬要求：

| `target_class` | 必填字段 | 说明 |
| --- | --- | --- |
| `RigidObject` | `name`, `path`, `target_class`, `prim_path_child` | `prim_path_child` 必须指向 USD 内的刚体 prim；常见为 `Aligned`。 |
| `GeometryObject` | `name`, `path`, `target_class` | `prim_path_child` 可选；若提供，最终对象 prim 会挂到该子 prim。 |
| `PlaneObject` | `name`, `target_class`, `size` | 程序生成平面，不需要 `path`。 |
| `BoxObject` | `name`, `target_class` | 程序生成 cube，可用 `scale` 控制尺寸。 |
| `XFormObject` | `name`, `path`, `target_class` | 可用 `parent_obj` 挂到已有 object prim 下。 |
| `ConveyorObject` | `name`, `path`, `target_class`, `linear_velocity`, `linear_track_list`, `angular_velocity`, `angular_track_list` | 通常放在 arena fixtures，也可作为 object 加载。 |
| `ArticulatedObject` | `name`, `path`, `target_class` | 随机化时通常还需要 `info_name`；初始化会读取关节信息。 |

`translation`、`euler`/`quaternion`、`scale` 在 `_load_obj()` 中统一应用；未写 `scale` 时默认 `[1.0, 1.0, 1.0]`，未写 `visible` 时默认 `true`。除 `PlaneObject`、`BoxObject` 外，资源型对象的 `path` 均相对 `asset_root`。

### `cameras`

由 `BananaBaseTask._load_camera()`、`CustomCamera`、`_perturb_camera()`、`_set_camera_poses()` 消费。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `name` | `str` | 相机实例名。 |
| `camera_file` | `str` | 外部相机参数 YAML，包含内参和分辨率。 |
| `parent` | `str` | 可选父 prim path；空字符串表示挂到任务根下的 `cameras/` 树。 |
| `translation` | `list[float]`, len=3 | 安装位姿平移。 |
| `orientation` | `list[float]`, quaternion `[w, x, y, z]` | 安装位姿四元数。 |
| `camera_axes` | `str`, current configs use `usd` | 传给 Isaac camera API。 |
| `apply_randomization` | `bool` | 为 true 时对相机位姿做噪声扰动。 |
| `max_translation_noise` | `float` | 最大平移噪声幅度。 |
| `max_orientation_noise` | `float`, degrees | 最大角度噪声，单位度。 |

注意：`camera_file` 提供内参和渲染分辨率；`apply_randomization` 只扰动外参。

### `regions`

由 `BananaBaseTask._set_regions()` 和 `optimize_2d_manip_layout()` 消费。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `object` | `str` | 目标物体名。 |
| `target` | `str` | 参考物体名。 |
| `random_type` | `str`, one of `A_in_B_region_sampler` / `A_on_B_region_sampler` / `A_by_B_circle_sampler` / `A_by_B_region_sampler` / `A_face_B_circle_sampler` / `A_along_B_C_circle_sampler` | `RandomRegionSampler` 上的方法名。 |
| `random_config` | `dict` | 采样器参数。 |
| `priority` | `list[int]` | 可选，从 `random_region_list` 里选索引。空列表时随机选任意索引。 |
| `container` | `str` | 容器类任务的特殊放置分支。 |
| `z_init` | `float` | `container` 分支里相对容器中心的初始 z 偏移。 |
| `target2` | `str` | 需要双支撑时的第二目标。 |
| `sub_tgt_prim` | `str` | 目标上的可选子 prim path。 |

`random_config` 常见字段：

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `pos_range` | `list[list[float]]`, shape `2x3` | 平移采样范围；常用于 `A_on_B_region_sampler` / `A_by_B_region_sampler`。 |
| `yaw_rotation` | `list[float]`, len=2, degrees | 绕 z 轴随机旋转范围。 |
| `r_range` | `list[float]`, len=2 | 圆环/半径采样范围。 |
| `theta_range` | `list[float]`, len=2, degrees | 角度采样范围。 |
| `x_bias` / `y_bias` / `z_bias` | `float` | 偏移量；`A_in_B_region_sampler` 使用。 |

### `distractors`

由 `_create_distractor_cfg()` 和 `_rebuild_distractors()` 消费。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `path` | `str` | 资源搜索根目录。 |
| `min_num` / `max_num` | `int` | 采样数量上下限。 |
| `scale` | `list[float]`, len=3 | distractor 默认缩放。 |
| `target` | `str` | 放置目标。 |
| `pos_range` | `list[list[float]]`, shape `2x3` | 放置范围。 |
| `min_object_distance` | `float` | XY 间距约束。 |
| `distractor_buffer` | `float` | distractor 间缓冲距离。 |
| `exclude_categories` | `list[str]` | 可选排除分类。 |
| `exclude_keywords` | `list[str]` | 可选排除子串。 |
| `target_class` | `str`, currently usually `RigidObject` | 默认 `RigidObject`。 |
| `prim_path_child` | `str`, currently usually `Aligned` | 默认 `Aligned`。 |

### `fluid`

由 `_set_fluid()` 消费。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `container_name` | `str` | 必填，用于放置粒子体积。 |
| `particleContactOffset` | `float` | 粒子间距基值。 |
| `spacing_scale` | `float` | 粒子间距倍率。 |
| `numParticlesX` / `numParticlesY` / `numParticlesZ` | `int` | 粒子网格维度。 |
| `center_z` | `bool` | 是否在 Z 方向居中。 |
| `z_offset` | `float` | Z 偏移。 |
| `max_velocity` | `float` | 粒子系统速度上限。 |
| `mass` | `float` | 粒子质量。 |
| `density` | `float` | 粒子密度。 |
| `color` / `emissiveColor` | `list[float]`, len=3 | 视觉材质颜色。 |
| `opacity` | `float` | 视觉材质透明度。 |

### `skills`

由 `SimBoxDualWorkFlow._initialize_skills()` 和技能执行循环消费。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| robot name | `str` | 外层 key，必须匹配某个 robot 的 `name`。 |
| controller name (`left` / `right` / etc.) | `str`, commonly `left` / `right` | 下一层嵌套。 |
| `name` | `str` | 注册表中的技能名。 |
| `objects` | `list[str]` | 技能对象参数。 |
| `depends_on` | `list[str]` | DAG 模式依赖列表。 |
| `id` | `str` or `int` | DAG 节点 id / 旧版内部 id。 |
| skill-specific fields | `Any` | 原样传给技能构造函数。 |

`skills` 的 YAML 嵌套结构固定为：

```yaml
skills:
  - <robot_name>:
      - <controller_name>:
          - name: <skill_name>
            objects: [...]
```

含义：

- `skills[]` 的每个元素是一个 phase。legacy 模式下，后一个 phase 会等前一个 phase 全部完成后才开始。
- `<robot_name>` 必须匹配 `robots[].name`。
- `<controller_name>` 必须匹配 workflow 中给该 robot 建出的 controller key。双臂常见为 `left`、`right`；移动底盘导航常用 `base`。
- 同一 controller 下的 list 是顺序队列；前一个 skill 完成后才进入下一个。
- 同一个 phase 内多个 robot/controller 可以并行推进，实际动作会在 workflow 中合并。

有任意 skill 写了 `id` 或 `depends_on` 时，整个 task 进入 DAG 模式。DAG 模式要求每个 skill 都必须有唯一 `id`，`depends_on` 必须是 list；所有依赖成功后该 skill 才会启动。

Legacy 顺序示例：

```yaml
skills:
  - split_aloha:
      - left:
          - name: pick
            objects: [pick_object]
          - name: place
            objects: [pick_object, place_target]
```

DAG 示例：

```yaml
skills:
  - split_aloha:
      - base:
          - id: nav_to_pick
            name: navigate
            depends_on: []
            goal: pick_pose
      - left:
          - id: pick_left
            name: pick
            depends_on: [nav_to_pick]
            objects: [pick_object]
```

如果只是加载 scene，不执行技能，写 `skills: []`。

### `data`

由 logger / language / episode 控制逻辑消费。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `task_dir` | `str` | 输出子目录。 |
| `language_instruction` | `str` | 高层语言指令模板。 |
| `detailed_language_instruction` | `str` | 详细语言指令模板。 |
| `collect_info` | `str` or `dict` | 记录/采集信息；articulation 随机化时可能被代码自动覆盖。 |
| `version` | `str` | 记录的数据集版本字符串。 |
| `update` | `bool` | 下游 workflow 使用的任务开关。 |
| `max_episode_length` | `int` | episode 截断长度。 |
| `log_motion_vectors` | `bool` | 是否记录 motion vector。 |

### `skills` 详细参数

`skills` 由 `workflows/simbox/core/skills/__init__.py` 注册，实际注册名来自类名转小写并以下划线分词，例如 `Goto_Pose -> goto_pose`、`Pour_Water_Succ -> pour_water_succ`。

所有 skill 共享的基础行为来自 `workflows/simbox/core/skills/base_skill.py`：

| 公共点 | 说明 |
| --- | --- |
| `name` | 大多数 skill 会读取，用于日志或内部标识。 |
| `objects` | 大多数操作类 skill 会读取，用于绑定任务中的对象。 |
| `is_ready/is_done/is_success/is_feasible/is_record` | 由各 skill 自己实现。 |
| `controller.num_plan_failed` | 几乎所有操作类 skill 用它判断 `is_feasible()`。 |
| `controller.num_last_cmd > 10` | 很多 skill 用它把“指令已推进”视为当前子命令完成。 |

下面按 skill 列出代码实际读取的参数。这里的“功能”和“实现特性”都按真实执行路径写，不按 skill 名字推断。

#### 如何理解 skill 参数

同一个 skill 配置里的字段，代码中的作用并不相同。阅读下面每个 skill 时，建议按这三类理解：

| 类型 | 说明 |
| --- | --- |
| 轨迹生成参数 | 直接影响 `simple_generate_manip_cmds()` / `predict_manip_cmds()`，决定会生成什么动作。 |
| 成功判定参数 | 只影响 `is_success()`，不会改变轨迹本身。 |
| 调试/辅助参数 | 影响日志、调试输出、可视化或过滤逻辑，不直接改变主轨迹。 |

### Skill 一览

| 注册名 | 主要功能 |
| --- | --- |
| `pick` | 基于抓取标注执行标准抓取，并支持调试快照输出。 |
| `manualpick` | 在标准抓取上增加手工姿态/位移修正。 |
| `dexpick` | 使用 `dexpick_pose.yaml` 里的离散抓取位姿执行抓取。 |
| `dynamicpick` | 面向动态目标的抓取，带时间偏置和运动预测。 |
| `failpick` | 故意偏移抓取点，生成失败抓取轨迹。 |
| `place` | 通用放置 skill，支持垂直/水平放置和多种成功判定。 |
| `dexplace` | 基于物体/容器几何范围的放置。 |
| `move` | 驱动末端带动物体向参考物体移动。 |
| `goto_pose` | 直接到指定末端位姿。 |
| `track` | 采样一组路点，驱动末端按路点追踪。 |
| `scan` | 把末端移到固定观测/扫描姿态。 |
| `wait` | 保持当前位置和夹爪状态等待若干步。 |
| `gripper_action` | 单独控制夹爪开合。 |
| `home` | 关节回 home 位。 |
| `heuristic_skill` | 基于启发式模式执行关节或相对 EE 运动。 |
| `joint_ctrl` | 直接按关节指令列表插值控制。 |
| `navigate` | 通过 Nav2 驱动移动底盘导航到目标点。 |
| `open` | 打开 articulation 对象。 |
| `close` | 关闭 articulation 对象。 |
| `artpreplan` | 为 articulation 操作做 KPAM 预规划准备。 |
| `rotate` | 沿 articulation 关节方向旋转对象。 |
| `rotate_obj` | 直接旋转被抓物体的姿态。 |
| `approach_rotate` | 先接近目标，再可选旋转被抓物体。 |
| `flip` | 将物体翻转到指定朝向。 |
| `pour_water_succ` | 用粒子统计和容器朝向判定倒水是否成功。 |

#### `pick`

注册名：`pick`  
实现：`workflows/simbox/core/skills/pick.py`  
功能：从抓取候选中筛选可达姿态，执行 pre-grasp、grasp、attach、post-grasp，并输出调试快照。
实现特性：即使没有筛到完全成功的候选，也会选一个 fallback 候选继续生成轨迹；成功判定依赖接触、速度稳定和可选抬升高度，而不是仅依赖轨迹执行完。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `objects[0]` | `str` | 被抓取对象名。 |
| `output_root` | `str` | 调试输出目录，默认 `output/ros_bridge/skills`。 |
| `npy_name` | `str` | 抓取姿态文件名，默认 `Aligned_grasp_sparse.npy`。 |
| `grasp_scale` | `float` | 抓取姿态缩放。 |
| `tcp_offset` | `float` | 传给 `pose_post_process_fn` 的 TCP 偏移。 |
| `constraints` | `list[axis:str, min_ratio:float, max_ratio:float]` | 抓取姿态后处理约束。 |
| `final_gripper_state` | `int`, `1` or `-1` | 抓取后夹爪状态，`1=open`，`-1=close`。 |
| `fixed_orientation` | `list[float]`, quaternion `[w, x, y, z]` | 若提供，则强制使用该末端姿态四元数。 |
| `ignore_substring` | `list[str]` | 更新碰撞过滤时附加忽略名单。 |
| `pre_grasp_offset` | `float` | 预抓取相对抓取点的退让距离。 |
| `pre_grasp_hold_vec_weight` | `list[float]`, usually len=6 | 预抓取阶段的 pose cost metric。 |
| `gripper_change_steps` | `int` | 开合夹爪重复步数。 |
| `post_grasp_offset_min` / `post_grasp_offset_max` | `float` | 抓取后抬升距离采样范围。 |
| `return_to_pregrasp` | `bool` | 抓取后是否返回 pre-grasp。 |
| `test_mode` | `str`, `forward` or `ik` | 候选姿态可达性检查方式。 |
| `filter_x_dir` / `filter_y_dir` / `filter_z_dir` | `list[str, float]` or `list[str, float, float]` | 抓取候选姿态方向过滤。 |
| `direction_to_obj` | `str`, `left` or `right` | 约束抓取点在物体左/右侧。 |
| `t_eps` / `o_eps` | `float` | 子命令完成判定阈值。 |
| `process_valid` | `bool` | 成功判定时是否检查机器人/物体速度稳定。 |
| `lift_th` | `float` | 成功判定时要求物体相对初始高度抬升量。 |

#### `manualpick`

注册名：`manualpick`  
实现：`workflows/simbox/core/skills/manualpick.py`  
功能：在标准抓取流程上增加显式的姿态旋转、平移补偿和人工修正，更适合难抓或标注不稳定物体。
实现特性：整体流程仍然是标准抓取，只是额外在抓取候选上做旋转和位移修正；成功判定只看接触，不检查 `process_valid` 或抬升高度。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `objects[0]` | `str` | 被抓取对象名。 |
| `npy_name` | `str` | 抓取姿态文件名。 |
| `grasp_scale` | `float` | 抓取姿态缩放。 |
| `hold_vec_weight` | `list[float]`, usually len=6 | 初始 pose cost metric。 |
| `ignore_substring` | `list[str]` | 碰撞过滤附加忽略名单。 |
| `start_lr_skill` | `bool` | 若为 true，开始阶段插入一次 `hold_vec_weight=None` 的更新。 |
| `adjust_ori` | `list[pose_axis:str, base_axis:str, judge_flag:str]` | 自动姿态调整配置：比较指定轴投影并择优旋转。 |
| `adjust_rotate_axis` | `str`, `x` / `y` / `z` | `adjust_ori` 使用的旋转轴。 |
| `adjust_angle_list_cfg` | `list[min_deg:float, max_deg:float, count:int]` | 自动姿态调整角度采样。 |
| `manual_adjust_ori` | `list[list[axis:str, angle_deg:float]]` | 手工附加旋转列表。 |
| `adjust_trans_offset` | `list[float]`, len=3 | 对所有抓取候选追加平移偏移。 |
| `pre_grasp_offset` | `float` | pre-grasp 退让距离。 |
| `pre_grasp_offset_manual` | `list[float]`, len=3 | 对 pre-grasp 再追加平移偏移。 |
| `test_mode` | `str`, `forward` or `ik` | 可达性检查方式。 |
| `gripper_change_steps` | `int` | 夹爪切换保持步数。 |
| `post_grasp_offset_min` / `post_grasp_offset_max` | `float` | 抓取后抬升距离范围。 |
| `filter_x_dir` / `filter_y_dir` / `filter_z_dir` | `list[str, float]` or `list[str, float, float]` | 姿态方向过滤。 |
| `direction_to_obj` | `str`, `left` or `right` | 左/右侧抓取约束。 |
| `t_eps` / `o_eps` | `float` | 子命令完成阈值。 |

#### `dexpick`

注册名：`dexpick`  
实现：`workflows/simbox/core/skills/dexpick.py`  
功能：从 `dexpick_pose.yaml` 读取离散抓取位姿，按固定抓取模板执行抓取。
实现特性：抓取位姿不是在线采样，而是直接读取 `dexpick_pose.yaml`；如果该文件不存在，代码里不会补默认姿态，后续可能在使用 `self.pose_ee2o` 时失败。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `objects[0]` | `str` | 被抓取对象名。 |
| `pick_pose_idx` | `int` | `dexpick_pose.yaml` 中选择的抓取姿态索引。 |
| `ignore_substring` | `list[str]` | 碰撞过滤附加忽略名单。 |
| `pre_grasp_offset` | `float` | pre-grasp 退让距离。 |
| `gripper_change_steps` | `int` | 夹爪闭合步数。 |
| `post_grasp_offset_min` / `post_grasp_offset_max` | `float` | 抓取后上抬距离。 |
| `t_eps` / `o_eps` | `float` | 子命令完成阈值。 |
| `process_valid` | `bool` | 成功判定时是否检查速度稳定。 |
| `lift_th` | `float` | 成功判定时最小抬升高度。 |

#### `dynamicpick`

注册名：`dynamicpick`  
实现：`workflows/simbox/core/skills/dynamicpick.py`  
功能：针对动态目标做抓取预测和时间补偿，适合传送带或移动物体。
实现特性：`simple_generate_manip_cmds()` 是空实现，真正的轨迹生成发生在 `is_ready()` 内部触发的 `predict_manip_cmds()`；只有目标物进入窗口后才开始预测和发起抓取。注意：这不是未实现 skill，而是依赖调度器持续轮询 `is_ready()` 的特殊实现方式。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `objects[0]` | `str` | 被抓取对象名。 |
| `grasp_scale` | `float` | 抓取姿态缩放。 |
| `tcp_offset` | `float` | TCP 偏移。 |
| `pick_range` | `list[min_x:float, max_x:float]` | 动态抓取时沿输送方向的采样区间。 |
| `time_bias` | `float` | 额外时间偏置。 |
| `pick_bias` | `float` | 抓取偏置。 |
| `pivot_angle_z` | `list[min_deg:float, max_deg:float]` | 对抓取姿态绕 z 轴的随机旋转范围。 |
| `pos_adjust_z` | `list[min_z:float, max_z:float]` | 对抓取点 z 的额外随机平移范围。 |
| `pre_grasp_offset` | `float` | pre-grasp 退让距离。 |
| `test_mode` | `str`, `forward` or `ik` | 可达性检查方式。 |
| `gripper_change_steps` | `int` | 夹爪闭合步数。 |
| `post_grasp_offset_min` / `post_grasp_offset_max` | `float` | 抓取后上抬距离。 |
| `filter_x_dir` / `filter_y_dir` / `filter_z_dir` | `list[str, float]` or `list[str, float, float]` | 抓取姿态过滤。 |
| `direction_to_obj` | `str`, `left` or `right` | 左/右侧抓取约束。 |
| `t_eps` / `o_eps` | `float` | 子命令完成阈值。 |
| `process_valid` | `bool` | 成功判定时是否检查速度稳定。 |
| `lift_th` | `float` | 成功判定时最小抬升高度。 |

#### `failpick`

注册名：`failpick`  
实现：`workflows/simbox/core/skills/failpick.py`  
功能：故意在抓取点上加偏移，构造失败抓取示例。
实现特性：它的 `is_success()` 永远返回 `True`，所以“失败”只体现在轨迹本身是故意偏抓，不体现在 episode 结果上；另外 `is_record()` 会在末段控制录像保留范围。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `objects[0]` | `str` | 目标对象。 |
| `ignore_substring` | `list[str]` | 碰撞过滤附加忽略名单。 |
| `grasp_x_offset_min` / `grasp_x_offset_max` | `float` | 故意偏移抓取点的 x 范围。 |
| `grasp_y_offset_min` / `grasp_y_offset_max` | `float` | 故意偏移抓取点的 y 范围。 |
| `test_mode` | `str`, `forward` or `ik` | 可达性检查方式。 |
| `gripper_change_steps` | `int` | 夹爪闭合步数。 |
| `post_grasp_offset_min` / `post_grasp_offset_max` | `float` | 抬升距离范围。 |
| `filter_x_dir` / `filter_y_dir` / `filter_z_dir` | `list[str, float]` or `list[str, float, float]` | 姿态过滤。 |
| `direction_to_obj` | `str`, `left` or `right` | 左/右侧约束。 |
| `t_eps` / `o_eps` | `float` | 子命令完成阈值。 |

#### `place`

注册名：`place`  
实现：`workflows/simbox/core/skills/place.py`  
功能：通用放置 skill，支持不同放置方向、对齐策略、姿态约束和多种成功判定模式。
实现特性：成功判定完全由几何关系决定，比如 IoU、bbox、左右关系、高度等；`is_done()` 只负责消耗 `manip_list`，不会因为成功而主动清空队列。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `objects[0]` | `str` | 被放置对象。 |
| `objects[1]` | `str` | 放置目标对象/容器。 |
| `place_align_axis` | `list[float]`, len=3 | 放置时 gripper 朝向约束。 |
| `pick_align_axis` | `list[float]`, len=3 | 与抓取物体相关的朝向约束。 |
| `constraint_gripper_x` | `bool` | 是否约束 gripper x 轴。 |
| `place_part_prim_path` | `str` | 放置目标上的局部 prim path。 |
| `align_pick_obj_axis` | `list[float]`, len=3 | 对抓取物体的对齐轴。 |
| `align_place_obj_axis` | `list[float]`, len=3 | 对放置目标的对齐轴。 |
| `align_plane_x_axis` / `align_plane_y_axis` | `list[float]`, len=3 | 平面对齐轴。 |
| `align_obj_tol` | `float`, degree | 对齐容差。 |
| `ignore_substring` | `list[str]` | 碰撞过滤附加忽略名单。 |
| `pre_place_hold_vec_weight` | `list[float]`, usually len=6 | pre-place 的 pose cost metric。 |
| `post_place_hold_vec_weight` | `list[float]`, usually len=6 | post-place 的 pose cost metric。 |
| `hesitate_steps` | `int` | 到达放置位后原地保持步数。 |
| `gripper_change_steps` | `int` | 松夹爪步数。 |
| `post_place_vector` | `list[float]`, len=3 | 放置后退让向量。 |
| `x_ratio_range` / `y_ratio_range` / `z_ratio_range` | `list[min:float, max:float]` | 在目标 bbox 内采样放置点的比例区间。 |
| `place_direction` | `str`, `vertical` or `horizontal` | 放置方向。 |
| `position_constraint` | `str`, `gripper` or `object` | 位置约束对象。 |
| `pre_place_z_offset` | `float` | 垂直放置时 pre-place 高度偏移。 |
| `place_z_offset` | `float` | 垂直放置时最终高度偏移。 |
| `offset_place_obj_axis` | `list[float]`, len=3 | 水平放置时偏移轴。 |
| `pre_place_align` / `pre_place_offset` | `float` | 水平放置 pre-place 的对齐/偏移量。 |
| `place_align` / `place_offset` | `float` | 水平放置最终点的对齐/偏移量。 |
| `pre_grasp_offset` | `float` | 某些分支里用于复用测试逻辑。 |
| `test_mode` | `str`, `forward` or `ik` | 可达性检查方式。 |
| `filter_x_dir` / `filter_y_dir` / `filter_z_dir` | `list[str, float]` or `list[str, float, float]` | 姿态过滤。 |
| `t_eps` / `o_eps` | `float` | 子命令完成阈值。 |
| `success_mode` | `str`, one of `3diou` / `height` / `xybbox` / `left` / `right` / `flower` / `cup` | 成功判定模式。 |
| `threshold` | `float` | `left/right` 模式阈值。 |
| `success_th` | `float` | `3diou/flower/cup` 等模式阈值。 |

#### `dexplace`

注册名：`dexplace`  
实现：`workflows/simbox/core/skills/dexplace.py`  
功能：基于几何包围盒和范围标注采样放置点，生成较轻量的放置轨迹。
实现特性：`place_range.yaml` 是从“被放置对象”的资源目录读，不是从容器/目标对象读；成功判定只检查物体世界坐标是否落在目标边界范围内。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `objects[0]` | `str` | 被放置对象。 |
| `objects[1]` | `str` | 放置目标对象。 |
| `gripper_axis` | `list[float]`, len=3 | 构造末端朝向时使用的夹爪方向。 |
| `camera_axis_filter` | `list[float]`, len=3 | 相机/法向过滤，影响姿态构造。 |
| `place_part_prim_path` | `str` | 放置目标的局部 prim path。 |
| `gripper_change_steps` | `int` | 松夹爪步数。 |
| `t_eps` / `o_eps` | `float` | 子命令完成阈值。 |

#### `move`

注册名：`move`  
实现：`workflows/simbox/core/skills/move.py`  
功能：让末端带着当前抓持状态沿目标物体方向移动，常用于推、挪、对位。
实现特性：真实移动量是“目标对象位置减被移动对象位置”再投到 arm base 平面，且 z 分量会被强制清零；成功同时检查 EE 到位和物体是否贴近目标对象。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `objects[0]` | `str` | 被移动对象。 |
| `objects[1]` | `str` | 目标参照对象。 |
| `invisible_object` | `list[str]` | 可选，skill 执行期间临时显示，成功后隐藏。 |
| `success_threshold` | `float` | EE 和对象移动是否成功的距离阈值。 |
| `delta_trans` | `list[list[float]]`, each len=3 | 在主目标平移外追加的偏移序列。 |
| `hold_vec_weight` | `list[float]`, usually len=6 | pose cost metric。 |
| `ignore_substring` | `list[str]` | 碰撞过滤附加忽略名单。 |

#### `goto_pose`

注册名：`goto_pose`  
实现：`workflows/simbox/core/skills/goto_pose.py`  
功能：直接到指定末端位置/姿态；若未给定姿态，可按对象对齐约束自动采样可行朝向。
实现特性：如果显式给了 `quaternion/euler`，不会做姿态采样；如果 `interp_nums >= 2`，当前实现插入的是一串 `update_specific` 指令而不是普通抓取/开合命令，这一点和名字直觉不一致。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `frame` | `str`, currently `robot` | 目标位姿所在坐标系。 |
| `gripper_state` | `int` or `float`, `1=open`, else close | 目标夹爪状态。 |
| `position` | `list[float]`, len=3 | 必填，目标位置。 |
| `quaternion` | `list[float]`, quaternion `[w, x, y, z]` | 可选目标朝向。 |
| `euler` | `list[float]`, len=3, degree | 可选目标朝向；若都不提供，会走约束采样。 |
| `max_noise_m` | `float` | 位置噪声。 |
| `max_noise_deg` | `float` | 姿态噪声。 |
| `objects[0]` | `str` | 当未显式给朝向时，用于构造与对象对齐约束。 |
| `align_obj_axis` / `align_ref_axis` | `list[float]`, len=3 | 朝向约束参数。 |
| `align_obj_tol` | `float`, degree | 朝向对齐容差。 |
| `position_constraint` | `str`, `gripper` or `object` | 目标位置约束对象。 |
| `ignore_substring` | `list[str]` | 碰撞过滤附加忽略名单。 |
| `test_mode` | `str`, `forward` or `ik` | 可达性检查方式。 |
| `interp_nums` | `int` | 插值路点数，`>=2` 时分段走。 |
| `filter_x_dir` / `filter_y_dir` / `filter_z_dir` | `list[str, float]` or `list[str, float, float]` | 姿态方向过滤。 |

#### `track`

注册名：`track`  
实现：`workflows/simbox/core/skills/track.py`  
功能：随机采样一系列目标路点，让末端依次跟踪，同时在场景中可视化 TCP 轴。
实现特性：路点会持续采样直到 IK 通过；成功判定只看最终末端位姿是否接近最后一个 waypoint，不检查轨迹是否完整覆盖中间点。已知问题：`is_success()` 当前使用“位置接近或姿态接近”即成功的宽松判定，不是“位置和姿态都满足”。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `frame` | `str`, currently `robot` | 路点坐标系。 |
| `T_tcp_2_ee` | `list[list[float]]`, shape `4x4` | TCP 相对 EE 的变换，用于可视化。 |
| `way_points_num` | `int` | 路点数量。 |
| `way_points_trans.min/max` | `list[float]`, len=3 | 采样路点平移范围。 |
| `way_points_ori` | `list[float]`, quaternion `[w, x, y, z]` | 路点基准姿态。 |
| `max_noise_deg` | `float` | 路点姿态噪声。 |

#### `scan`

注册名：`scan`  
实现：`workflows/simbox/core/skills/scan.py`  
功能：把末端移动到预设观察姿态，常用于扫描、俯视或接触前观察。
实现特性：虽然名字叫 scan，但 `is_success()` 实际上看的是接触传感器和速度稳定，而不是“是否到达一个观察姿态”或“是否获得观测结果”。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `name` | `str` | skill 名。 |
| `objects[0]` | `str` | 被扫描/接触检查对象。 |
| `t_eps` / `o_eps` | `float` | 子命令完成阈值。 |
| `process_valid` | `bool` | 成功判定时是否检查速度稳定。 |

#### `wait`

注册名：`wait`  
实现：`workflows/simbox/core/skills/wait.py`  
功能：保持当前位置和夹爪状态等待指定步数，常用于时序同步或让物体稳定。
实现特性：它不是空转 world，而是反复发送同一条开/合夹爪命令并保持当前 EE 位姿；成功判定是等待结束后 EE 仍然接近起始位姿。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `name` | `str` | skill 名。 |
| `objects[0]` | `str` | 关联对象。 |
| `success_threshold` | `float` | 等待结束时 EE 与目标点的距离阈值。 |
| `ignore_substring` | `list[str]` | 碰撞过滤附加忽略名单。 |
| `gripper_state` | `float` or `int`, `-1` / `1` | 等待期间保持的夹爪状态，默认 close。 |
| `wait_steps` | `int` | 等待步数。 |

#### `gripper_action`

注册名：`gripper_action`  
实现：`workflows/simbox/core/skills/gripper_action.py`  
功能：只执行夹爪开合，不移动末端。
实现特性：所有命令都在当前 EE 位姿上重复执行；真正变化的只有夹爪动作名和可选速度参数。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `gripper_state` | `int`, `1=open`, `-1=close` | 必填夹爪状态。 |
| `vel` | `float` | 可选夹爪速度。 |
| `wait_steps` | `int` | 重复执行步数。 |

#### `home`

注册名：`home`  
实现：`workflows/simbox/core/skills/home.py`  
功能：将当前手臂关节插值回 home 位，可选覆盖夹爪状态。
实现特性：轨迹长度硬编码为 50 步，插值比例按 `1/40` 递增，所以后 10 步会超过 1.0；也就是说按代码它会有轻微“过冲”趋势，但通常在成功后会被提前清空。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `gripper_state` | `float` or `int`, usually `-1` / `1` | 可选，覆盖机器人默认 home 时的夹爪状态。 |

#### `heuristic_skill`

注册名：`heuristic_skill`  
实现：`workflows/simbox/core/skills/heuristic_skill.py`  
功能：以启发式模式执行预定义关节动作、相对关节偏移或相对 EE 运动，适合作为轻量动作原语。
实现特性：`rel_qpos` 按名字像“当前关节增量”，但实际代码直接把 `value` 作为目标关节数组使用，并没有和当前关节相加；同时插值阶段用了 `1.25` 的放大系数，存在过冲倾向。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `gripper_state` | `float` or `int`, usually `-1` / `1` | 可选夹爪状态。 |
| `mode` | `str`, one of `home` / `abs_qpos` / `rel_qpos` / `rel_ee` | 动作模式。 |
| `move_steps` | `int` | 关节插值步数。 |
| `t_eps` | `float` | 成功和完成判定的关节阈值。 |
| `value` | `list[float]` or `4x4 matrix` | 与 `mode` 对应的目标值：绝对关节、相对关节或相对 EE 变换。 |

#### `joint_ctrl`

注册名：`joint_ctrl`  
实现：`workflows/simbox/core/skills/joint_ctrl.py`  
功能：直接按关节控制列表构造目标关节角并插值执行，适合底层调试和特定关节动作。
实现特性：当前生成的都是 `dummy_forward` 命令，不会生成 `joint_ctrl` 命令，因此 `is_subtask_done()` 里针对 `joint_ctrl` 分支的逻辑在主路径下实际上走不到。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `name` | `str` | skill 名。 |
| `ctrl_list` | `list[list[int, float, str]]`, mode in `abs` / `delta` | 关节控制列表。 |
| `num_steps` | `int` | 插值步数。 |
| `gripper_state` | `float` or `int`, usually `-1` / `1` | `dummy_forward` 时保持的夹爪状态。 |
| `success_threshold_js` | `float` | 关节成功阈值。 |

#### `navigate`

注册名：`navigate`  
实现：`workflows/simbox/core/skills/navigate.py`  
功能：通过 Nav2 会话管理器驱动移动底盘到指定世界坐标目标。
实现特性：这是一个事件驱动 skill。`simple_generate_manip_cmds()` 不生成 `manip_list`，真正推进发生在 `update()` 里；成功与失败完全来自 runtime manager，而不是来自机械臂控制器。支持两种目标写法：直接写 `goal_x/goal_y/goal_yaw`，或在 task 顶层 `positions` 里定义命名点位后用 `goal` 引用；若两者同时存在，优先使用 `goal`。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `goal` | `str` | 可选，命名导航目标。若提供，则从 task 顶层 `positions[goal]` 读取 `x/y/yaw`。 |
| `goal_x` / `goal_y` / `goal_yaw` | `float` | 可选，世界坐标系导航目标；当 `goal` 未提供时使用。 |
| `xy_goal_tolerance` / `skill_xy_goal_tolerance` | `float` | 旧版位置容差，若机器人 base 配置里有 Nav2 goal_checker，则会被后者覆盖。 |
| `yaw_goal_tolerance` / `skill_yaw_goal_tolerance` | `float` | 旧版朝向容差。 |
| `startup_timeout_sec` | `float` | Nav2 启动超时。 |
| `runtime_timeout_sec` | `float` | 导航运行超时。 |
| `output_root` | `str` | 技能相关输出目录。 |
| `scene_name` | `str` | 导航场景名。 |
| `map_output_dir` | `str` | 地图输出目录。 |
| `map_resolution` | `float` | 导航地图分辨率。 |
| `map_z_min` / `map_z_max` | `float` | 建图高度过滤范围。 |

task 顶层 `positions` 示例：

```yaml
positions:
  nav_to_pick:
    x: -0.08
    y: -0.72
    yaw: 1.5707963267948966
```

skill 引用示例：

```yaml
- name: navigate
  goal: nav_to_pick
```

#### `open`

注册名：`open`  
实现：`workflows/simbox/core/skills/open.py`  
功能：对 articulation 物体执行打开动作，轨迹来自 KPAM 关键位姿规划。
实现特性：在 `contact_pose_index` 之前保持 `open_gripper`，到接触位后会额外插入一段 40 步的 `close_gripper` 保持；成功判定依赖 articulation 关节位移、碰撞合法性和速度稳定。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `name` | `str` | skill 名。 |
| `objects[0]` | `str` | articulation 对象名。 |
| `planner_setting` | `dict` / `omegaconf.DictConfig` | KPAM 规划配置。 |
| `planner_setting.contact_pose_index` | `int` | 接触位姿索引。 |
| `planner_setting.success_threshold` | `float` | articulation 关节位移成功阈值。 |
| `planner_setting.success_mode` | `str`, `normal` or `abs` | 成功判定模式。`normal` 检查带符号位移，`abs` 检查绝对位移。 |
| `planner_setting.update_art_joint` | `bool` | 是否在执行中同步 articulation joint target。 |
| `obj_info_path` | `str` | 若提供，先更新 articulation 信息。 |
| `ignore_substring` | `list[str]` | 碰撞过滤附加忽略名单。 |
| `collision_valid` | `bool` | 成功判定时是否检查禁碰撞。 |
| `process_valid` | `bool` | 成功判定时是否检查速度稳定。 |

#### `close`

注册名：`close`  
实现：`workflows/simbox/core/skills/close.py`  
功能：对 articulation 物体执行关闭动作，轨迹来自 KPAM 关键位姿规划。
实现特性：执行中会在接触位附近更新 `ignore_substring`，把 articulation 父节点临时加入忽略列表；成功判定支持“绝对接近零位”和“相对初始位移足够大”两种模式。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `name` | `str` | skill 名。 |
| `objects[0]` | `str` | articulation 对象名。 |
| `planner_setting` | `dict` / `omegaconf.DictConfig` | KPAM 规划配置。 |
| `planner_setting.contact_pose_index` | `int` | 接触位姿索引。 |
| `planner_setting.success_threshold` | `float` | articulation 关节成功阈值。 |
| `planner_setting.success_mode` | `str`, `zero` or `dis_to_init` | 成功判定模式。`zero` 检查关节是否接近 0，`dis_to_init` 检查相对初始位移。 |
| `planner_setting.update_art_joint` | `bool` | 是否同步 articulation joint target。 |
| `obj_info_path` | `str` | articulation 信息文件。 |
| `ignore_substring` | `list[str]` | 碰撞过滤附加忽略名单。 |
| `collision_valid` | `bool` | 是否检查禁碰撞。 |
| `process_valid` | `bool` | 是否检查速度稳定。 |

#### `artpreplan`

注册名：`artpreplan`  
实现：`workflows/simbox/core/skills/artpreplan.py`  
功能：为 articulation 类技能提前建立 KPAM 规划器和关键接触位姿，通常作为 open/close 之前的准备步骤。
实现特性：它并不会真正执行长轨迹，只会初始化规划器、取关键位姿、做一次 `update_specific`，更像“预热/预规划 skill”。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `name` | `str` | skill 名。 |
| `objects[0]` | `str` | articulation 对象名。 |
| `planner_setting` | `dict` / `omegaconf.DictConfig` | KPAM 规划配置。 |
| `planner_setting.contact_pose_index` | `int` | 接触位姿索引。 |
| `planner_setting.success_threshold` | `float` | 预规划配置中的阈值字段；当前实现初始化时会读取，但 `is_success()` 实际不使用它。 |
| `planner_setting.update_art_joint` | `bool` | 是否同步 articulation joint target。 |
| `obj_info_path` | `str` | articulation 信息文件。 |

#### `rotate`

注册名：`rotate`  
实现：`workflows/simbox/core/skills/rotate.py`  
功能：沿 articulation 目标关节执行旋转类操作，也支持对特定资产覆盖 actuation 逻辑。
实现特性：它本质上仍是 KPAM articulation 技能；`hearth` 类对象会走特殊分支，额外插入一串 `in_plane_rotation` 指令。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `name` | `str` | skill 名。 |
| `objects[0]` | `str` | articulation 对象名。 |
| `planner_setting` | `dict` / `omegaconf.DictConfig` | KPAM 规划配置。 |
| `planner_setting.contact_pose_index` | `int` | 接触位姿索引。 |
| `planner_setting.success_threshold` | `float`, radians | articulation 关节旋转成功阈值，默认约 `0.785`。 |
| `planner_setting.additional_labels` | `dict[str, Any]` | 针对特定资产路径覆盖 `planner.modify_actuation_motion`。 |
| `obj_info_path` | `str` | articulation 信息文件。 |

#### `rotate_obj`

注册名：`rotate_obj`  
实现：`workflows/simbox/core/skills/rotate_obj.py`  
功能：在保持抓持关系的前提下，直接改变被抓物体姿态，并同步更新末端目标位姿。
实现特性：成功判定只比较 EE 是否到目标平移/姿态阈值内，不直接检查物体是否真的转到了目标朝向。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `name` | `str` | skill 名。 |
| `objects[0]` | `str` | 被旋转物体名。 |
| `success_threshold_move` | `float` | EE 平移成功阈值。 |
| `success_threshold_rotate` | `float`, radians | EE 姿态成功阈值。 |
| `ignore_substring` | `list[str]` | 碰撞过滤附加忽略名单。 |
| `dummy_forward` | `dict` | 预插值关节运动配置。 |
| `dummy_forward.num_steps` | `int` | 插值步数。 |
| `dummy_forward.gripper_state` | `float` or `int`, usually `-1` / `1` | 插值阶段夹爪状态。 |
| `first_motion` | `str`, `move` or `rotate` | 决定先平移还是先旋转。 |
| `gripper_state` | `float` or `int`, usually `-1` / `1` | 主命令夹爪状态。 |
| `move_offset` | `list[float]`, len=3 | `first_motion=move` 时的平移偏移。 |
| `rotate_offset` | `list[float]`, len=3 | `first_motion=rotate` 时的平移偏移。 |
| `rotate_only` | `bool` | 若为 true，旋转后位置不跟随目标。 |
| `obj_axis_offset` | `list[list[str, float]]` | 物体局部轴上的位姿偏移，如 `[[\"x\", 0.01], [\"z\", -0.02]]`。 |
| `trans_offset` | `list[float]`, len=3 | 末端最终平移偏移。 |
| `rotate_obj_euler_delta` | `list[list[float]]`, shape `2x3`, degrees | 物体旋转欧拉角增量采样范围。 |
| `ctrl_list` | `list[list[int, float, str]]`, mode in `abs` / `delta` | `dummy_forward` 求目标关节时使用的关节指令列表。 |

#### `approach_rotate`

注册名：`approach_rotate`  
实现：`workflows/simbox/core/skills/approach_rotate.py`  
功能：先沿指定轴接近目标对象，再根据可选子配置对被抓物体做旋转。
实现特性：目标位姿是通过“当前 EE 相对被抓物体的变换”映射出来的，所以 success 本质上也是 EE 位姿成功，而不是直接检查接近对象的距离或接触。当前状态：此 skill 未完全实现，`get_tgt_js()` 仍为 `NotImplementedError`；如果配置启用 `dummy_forward`，运行时会直接报错。仓库内当前也没有任务在使用它，建议视为未完成 skill。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `name` | `str` | skill 名。 |
| `objects[0]` | `str` | 被移动/旋转对象。 |
| `objects[1]` | `str` | 被接近对象。 |
| `success_threshold` | `float` | 平移成功阈值。 |
| `hold_vec_weight` | `list[float]`, usually len=6 | pose cost metric。 |
| `ignore_substring` | `list[str]` | 碰撞过滤附加忽略名单。 |
| `dummy_forward` | `dict` | 预插值关节运动配置。注意当前若启用会调用未实现的 `get_tgt_js()`。 |
| `dummy_forward.num_steps` | `int` | 插值步数。 |
| `dummy_forward.gripper_state` | `float` or `int`, usually `-1` / `1` | 插值阶段夹爪状态。 |
| `obj_axis_offset` | `list[list[str, float]]` | 目标物体局部轴偏移。 |
| `z_offset` | `float` | 最终末端 z 偏移。 |
| `approach_axis` | `str`, one of `+x` / `-x` / `+y` / `-y` / `+z` / `-z` | 以目标物体哪个局部轴作为接近方向。 |
| `obj_yaw_offset` | `float`, degrees | 对接近方向额外绕世界 z 轴旋转。 |
| `distance` | `float` | 距离目标的接近距离。 |
| `rotate` | `dict` | 可选旋转子配置。 |
| `rotate.type` | `str`, `random` or `towards` | 旋转模式。 |
| `rotate.success_threshold` | `float`, radians | 旋转成功阈值。 |
| `rotate.rotate_obj_euler` | `list[list[float]]`, shape `2x3`, degrees | `random` 模式欧拉角采样范围。 |
| `rotate.objects[1]` | `str` | `towards` 模式下参照对象。 |

#### `flip`

注册名：`flip`  
实现：`workflows/simbox/core/skills/flip.py`  
功能：通过预设的中间姿态和释放动作，把物体翻转到目标朝向。
实现特性：中间位姿和最终位姿大部分是硬编码采样区间，不是从目标对象几何推出来的；成功判定看的是物体局部 y 轴与世界 z 轴夹角，以及相对初始 EE 的 y 向位移。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `name` | `str` | skill 名。 |
| `objects[0]` | `str` | 被翻转对象。 |
| `gripper_axis` | `list[float]`, len=3 | 用于构造翻转姿态的夹爪主轴。 |
| `open_wait_steps` | `int` | 到位后张开夹爪保持步数。 |
| `t_eps` / `o_eps` | `float` | 子命令完成阈值。 |

#### `pour_water_succ`

注册名：`pour_water_succ`  
实现：`workflows/simbox/core/skills/pour_water_succ.py`  
功能：主要用于成功判定，通过粒子落入容器数量和容器朝向判断倒水是否成功。
实现特性：按当前实现，它并不会根据 `translation/quaternion` 真正生成倒水动作轨迹，`simple_generate_manip_cmds()` 只是把当前关节状态包装成一个 `dummy_forward`；因此它更像“倒水成功检测 skill”，不是完整的倒水动作 skill。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `frame` | `str`, currently `robot` | 坐标系，默认 `robot`。 |
| `translation` | `list[float]`, len=3 | 可选目标位置。当前实现只用于记录/扰动目标，不直接生成倒水轨迹。 |
| `quaternion` | `list[float]`, quaternion `[w, x, y, z]` | 可选目标朝向。 |
| `euler` | `list[float]`, len=3, degrees | 可选目标朝向。 |
| `max_noise_m` | `float` | 位置噪声。 |
| `max_noise_deg` | `float` | 朝向噪声。 |
| `gripper_state` | `float` or `int`, usually `-1` / `1` | `dummy_forward` 阶段夹爪状态。 |
| `container_name` | `str` | 成功判定时统计粒子的容器名，默认 `cup`。 |
| `container_radius` | `float` | 粒子统计的 XY 半径。 |
| `particle_num_th_min` / `particle_num_th_max` | `int` | 成功判定的粒子数量范围。 |
| `container_up` | `list[list[str, str, float]]` | 额外朝向约束列表，每项为 `(container_name, axis, threshold)`，其中 `axis` 目前为 `x/y/z`。 |


## 2. Arena YAML

Arena 文件由 `task_cfg["arena_file"]` 读取，并存入 `cfg["arena"]`。

Arena YAML 只描述静态场景底座和 fixture，不描述任务动作。`objects`、`regions`、`robots`、`cameras`、`skills` 仍然属于 Task YAML。一个 arena 能否被加载，取决于引用它的 task 中的 `asset_root` 是否能解析 fixture 的 `path` 和 texture 目录。

### 顶层键

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `name` | `str` | arena 名称。 |
| `fixtures` | `list[dict]` | 必填，静态场景元素列表。 |
| `update_freq` | `int` | `BananaBaseTask.individual_reset()` 用于 scene-pair 刷新。 |
| `involved_scenes` | `str`, comma-separated scene prefixes | 当两个 fixture 都随机化时，`update_scene_pair()` 使用；当前实现期待逗号分隔字符串。 |

### 最小可运行 Arena 模板

```yaml
name: my_arena
fixtures:
  - name: floor
    target_class: PlaneObject
    size: [5.0, 5.0]
    translation: [0.0, 0.0, 0.0]
    collision_enabled: true
    collision_thickness: 0.02
    visible: true
```

这个模板不依赖 USD 资产。若要加载外部 USD fixture，使用 `GeometryObject`：

```yaml
name: my_usd_arena
fixtures:
  - name: table
    path: table0/instance.usd
    target_class: GeometryObject
    translation: [0.0, 0.0, 0.375]
    euler: [0.0, 0.0, 0.0]
    scale: [0.001, 0.001, 0.001]
```

### `fixtures`

由 `BananaBaseTask.set_up_scene()`、`_set_fixture_textures()`、`update_scenes()`、`update_hearths()`、`update_scene_pair()` 消费。

通用字段：

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `name` | `str` | fixture 名称。 |
| `target_class` | `str`, commonly `GeometryObject` / `PlaneObject` / `ConveyorObject` / `XFormObject` | fixture 注册类名。 |
| `path` | `str` | 相对于 `asset_root` 的资源路径。 |
| `translation` | `list[float]`, len=3 | 初始位姿平移。 |
| `euler` | `list[float]`, len=3, degrees | 初始朝向。 |
| `quaternion` | `list[float]`, quaternion `[w, x, y, z]` | 初始朝向。 |
| `scale` | `list[float]`, len=3 | 初始缩放。 |
| `size` | `list[float]`, len=2 | `PlaneObject` 的平面尺寸。 |
| `texture` | `dict` | 可选纹理规格。 |
| `apply_randomization` | `bool` | 启用纹理或 scene 切换逻辑。 |
| `visible` | `bool` | 初始可见性；默认 `true`。 |
| `linear_velocity` | `list[float]`, len=3 | 传送带类 fixture 使用。 |
| `linear_track_list` | `list[str]` | `ConveyorObject` 线速度轨道名；代码会查找 `<conveyor>/World/<track>/node_`。 |
| `angular_velocity` | `list[float]`, len=3 | `ConveyorObject` 角速度。 |
| `angular_track_list` | `list[str]` | `ConveyorObject` 角速度轨道名；代码会查找 `<conveyor>/World/<track>/validate_obj`。 |
| `collision_enabled` | `bool` | 可选平面碰撞开关。 |
| `collision_visible` | `bool` | 若启用平面碰撞，是否显示碰撞体。 |
| `collision_thickness` | `float` | 可选平面碰撞厚度。 |

按 fixture `target_class` 的硬要求：

| `target_class` | 必填字段 | 常见用途 |
| --- | --- | --- |
| `PlaneObject` | `name`, `target_class`, `size` | 地面、背景板、导航边界。 |
| `GeometryObject` | `name`, `target_class`, `path` | 桌子、房间、整场景 USD、assets_addition 静态家具。 |
| `XFormObject` | `name`, `target_class`, `path` | 加载可作为 xform 控制的 USD。 |
| `ConveyorObject` | `name`, `target_class`, `path`, `linear_velocity`, `linear_track_list`, `angular_velocity`, `angular_track_list` | 传送带。 |
| `BoxObject` | `name`, `target_class` | 程序生成碰撞/可视 box。 |

fixture 会通过和 task objects 相同的 `_load_obj()` 加载，因此同样支持 `translation`、`euler`/`quaternion`、`scale`、`visible`、`texture`。`PlaneObject` 和 `BoxObject` 不需要 `path`；其他资源型 fixture 的 `path` 相对 task 的 `asset_root`。

纹理块：

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `texture_lib` | `str` | 资源库目录。 |
| `apply_randomization` | `bool` | 随机纹理选择。 |
| `texture_id` | `int` | 不随机时的确定性纹理索引。 |
| `texture_scale` | `list[float]` or `float` | 传给材质。 |
| `target_prim_path` | `str` | 仅 `GeometryObject.apply_texture()` 支持；相对 fixture prim 的子 prim 路径，用于指定递归绑定材质的起点。 |

`apply_randomization: false` 时必须保证 `texture_id` 在 `asset_root/texture_lib/*` 排序后的列表范围内；`true` 时会随机选择该目录下的文件。

### scene-pair / hearth 随机化

Arena 顶层 `involved_scenes` 和 fixture 级 `apply_randomization` 用于旧的 table+scene 随机化路径：

- 当 arena 只有两个 fixture，且两个 fixture 都 `apply_randomization: true` 时，`update_scene_pair()` 会读取 `involved_scenes` 指向的 JSON 列表，并替换 table/scene fixture 的 `path`、`target_class`、`scale`、`translation`、`euler`。
- 当 task objects 中存在名字包含 `hearth` 的 object 时，`update_hearths()` 会随机替换名为 `scene` 且 `apply_randomization: true` 的 fixture。

如果只是写一个确定性可运行 arena，不需要 `involved_scenes`，也不要给 fixture 写 `apply_randomization: true`。

### `assets_addition` split 场景写法

`assets_addition` 生成的 split arena 通常把不可动物体写成 `GeometryObject` fixtures，把可作为 task object 操作/检查的物体写入 Task YAML 的 `objects`。arena 形态通常是：

```yaml
name: assets_addition_file_2_arena
fixtures:
  - name: empty_room
    path: assets_addition/file_2/empty_room.usd
    target_class: GeometryObject
    translation: [0.0, 0.0, 0.0]
    euler: [0.0, 0.0, 0.0]
    scale: [1.0, 1.0, 1.0]
  - name: file_2_sofa
    path: assets_addition/file_2/assets/arena/file_2__sofa.usd
    target_class: GeometryObject
    translation: [0.0, 0.0, 0.0]
    euler: [0.0, 0.0, 0.0]
    scale: [1.0, 1.0, 1.0]
  - name: floor
    target_class: PlaneObject
    size: [30.0, 30.0]
    translation: [0.0, 0.0, 0.0]
    collision_enabled: false
    visible: false
```

对应 task 可把 task objects 的 `scene_prim_path` 和 `scene_reference` 保留为元数据；运行时真正加载的是 `path` 指向的 `Aligned_obj.usd`。

## 3. `configs/de*.yaml`

这些是 Nimbus pipeline 配置，由 `launcher.py` -> `ConfigProcessor` -> `run_data_engine()` 解析。

### 常见顶层键

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `name` | `str` | 实验名。 |
| `load_stage` | `dict` | 创建 world 和 workflow 的 stage。 |
| `layout_random_generator` | `dict` | 随机化或加载 layout 的 stage。 |
| `plan_stage` | `dict` | 规划 stage。 |
| `plan_with_render_stage` | `dict` | 规划 + 渲染合并 stage。 |
| `dump_stage` | `dict` | 序列化 stage。 |
| `dedump_stage` | `dict` | 反序列化 stage。 |
| `render_stage` | `dict` | 仅渲染 stage。 |
| `store_stage` | `dict` | 输出写盘 stage。 |
| `stage_pipe` | `dict` | 仅 `de_pipe` 使用的并行管线配置。 |

### `load_stage.scene_loader.args.simulator`

由 `nimbus_extension/components/load/env_loader.py` 消费。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `physics_dt` | `str` fraction or `float` | 物理时间步。 |
| `rendering_dt` | `str` fraction or `float` | 渲染时间步。 |
| `stage_units_in_meters` | `float` | Isaac stage 单位尺度。 |
| `headless` | `bool` | 是否无头启动 `SimulationApp`。 |
| `experience` | `str` | 可选 Kit experience 路径。 |
| `anti_aliasing` | `int` | `SimulationApp` 抗锯齿模式。 |
| `multi_gpu` | `bool` | `SimulationApp` 多 GPU 开关。 |
| `active_gpu` | `int` | CUDA 设备索引。 |
| `physics_gpu` | `int` | 物理 GPU 索引。 |
| `max_gpu_count` | `int` | `SimulationApp` 限制。 |
| `renderer` | `str`, commonly `RayTracedLighting` / `PathTracing` | 渲染器类型。 |
| `width` / `height` | `int` | 渲染分辨率。 |
| `resolution` | `Any` | `env_loader` 不消费，应该用 `width` / `height`。 |
| `denoiser` | `bool` | Path tracing 降噪开关。 |
| `samples_per_pixel_per_frame` | `int` | 每帧采样数。 |
| `max_bounces` | `int` | 光线反弹上限。 |
| `max_specular_transmission_bounces` | `int` | 光线反弹上限。 |
| `max_volume_bounces` | `int` | 光线反弹上限。 |
| `subdiv_refinement_level` | `int` | 网格细分等级。 |
| `planning_step_render` | `bool` | 传给 `SimBoxDualWorkFlow`。 |
| `rendering_interval` | `int` | 渲染进程里的渲染节奏。 |
| `disable_viewport_updates` | `bool` | headless 启动时是否禁用 viewport 更新。 |
| `total_spp` | `int` | RTX pathtracing 设置。 |
| `adaptive_sampling` | `bool` | Path tracing 设置。 |
| `adaptive_sampling_target_error` | `float` | Path tracing 设置。 |
| `optix_denoiser` | `bool` | Path tracing 设置。 |
| `optix_temporal_denoiser` | `bool` | Path tracing 设置。 |
| `denoise_aovs` | `bool` | Path tracing 设置。 |
| `firefly_filter` | `bool` | Path tracing 设置。 |
| `firefly_max_intensity_glossy` | `float` | Path tracing 设置。 |
| `firefly_max_intensity_diffuse` | `float` | Path tracing 设置。 |
| `reset_pt_accum_on_time_change` | `bool` | Path tracing 设置。 |
| `fractional_cutout_opacity` | `bool` | Path tracing 设置。 |

重要：`physics_dt = 1/30` 和 `rendering_dt = 1/30` 表示仿真按 30 Hz 前进，但导出视频 FPS 由别处控制：

| 导出路径 | FPS 来源 |
| --- | --- |
| `nimbus/components/data/observation.py` | `flush_to_disk(..., video_fps=10)` 默认值 |
| `workflows/simbox/core/loggers/lmdb_logger.py` | 当前默认由 `physics_dt` 推导，`1/30 -> 30fps` |
| `scripts/simbox/record_collaborate_topdown_mp4.py` | CLI `--fps`，默认 `20` |

因此，导出的 MP4 不是天然 30fps，除非调用方传入 30，或者默认链路按 `physics_dt` 推导得到 30。

### `layout_random_generator.args`

由 `nimbus_extension` 的 layout randomizer 消费。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `random_num` | `int` | 生成样本数。 |
| `strict_mode` | `bool` | 强制输出条数等于 `random_num`。 |
| `input_dir` | `str` | 仅渲染模板使用，用于读取 plan 输出。 |

### `plan_with_render_stage.plan_with_render.args`

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `emit_obs_on_failure` | `bool` | 规划失败时是否输出兜底观测。 |
| `failure_obs_length` | `int` | 兜底观测长度。 |

### `store_stage.writer.args`

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `batch_async` | `bool` | 异步批量写入。 |
| `output_dir` | `str` | 输出根目录。 |

### `stage_pipe`

由 `nimbus/scheduler/sches.py` 和 `nimbus/data_engine.py` 消费。

| Key | 值类型 / 候选值 | 含义 |
| --- | --- | --- |
| `stage_num` | `list[int]` | 每个 pipe 段里的 stage 数量。 |
| `stage_dev` | `list[str]`, e.g. `cpu` / `cuda:0` / `cuda:1` | 每个 pipe 段的设备。 |
| `worker_num` | `list[int]` | 每个 pipe 段的 worker 数。 |
| `worker_schedule` | `bool` | 是否启用动态 worker 调度。 |
| `safe_threshold` | `int` | 队列限流阈值。 |
| `status_timeouts.idle` | `int` or `float` | Idle 超时，单位秒。 |
| `status_timeouts.ready` | `int` or `float` | Ready 超时，单位秒。 |
| `status_timeouts.running` | `int` or `float` | Running 超时，单位秒。 |
| `monitor_check_interval` | `int` | 健康检查间隔，单位秒。 |

## 4. 说明

- 即使某些字段不会被代码直接使用，只要它们出现在发布版配置里，这里也保留。
- 有些字段只对特定任务或物体类生效；未被支持的字段会被忽略，除非对应代码分支消费它们。
