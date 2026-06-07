# Download Scene YAML Change Log

本文记录从 `scripts/convert_download_scene_configs.py` 开始，为了把 `download/<scene>/arena.yaml` 和 `download/<scene>/task.yaml` 接入当前 SimBox 系统，YAML 需要做的全部结构转换和运行期修正。

## 范围

输入文件保持不改：

- `download/<scene>/arena.yaml`
- `download/<scene>/task.yaml`
- `download/<scene>/interndata_scene/task.yaml`

生成文件：

- `download/<scene>/simbox_arena.yaml`
- `download/<scene>/simbox_task.yaml`
- `workflows/simbox/core/configs/arenas/download/<scene>_arena.yaml`
- `workflows/simbox/core/configs/tasks/download/<scene>_task.yaml`

当前已覆盖的场景：

- `01_kitchen`
- `02_bookroom`
- `03_livingroom`
- `04_bedroom`

约束：

- 不覆盖原始下载的 `arena.yaml` 和 `task.yaml`。
- 源目录生成文件和 `workflows/simbox/core/configs/.../download/` 下的系统配置文件需要保持一致。
- 生成后的路径必须能从 `asset_root` 解析到真实资产。

## Arena YAML

### 顶层结构

下载格式包含 `format`、`version`、`coordinate_frame`、`texture_libs`、`fixtures` 等字段。当前 SimBox arena 只保留运行需要的结构：

```yaml
name: asset_01_kitchen
fixtures:
  - ...
```

修改规则：

- `name` 使用场景名清洗后生成，例如 `01_kitchen` 变成 `asset_01_kitchen`。
- 只输出 `fixtures`，不输出下载格式的 `format`、`version`、`coordinate_frame`、`texture_libs`。

### 名称清洗

所有对象和 fixture 名称都要清洗成 SimBox 可直接引用的名字：

- 非字母、数字、下划线字符替换成 `_`。
- 连续 `_` 合并。
- 首字符是数字时补 `asset_` 前缀。

技能、positions、regions 后续只能引用清洗后的名字。

### 坐标轴转换

下载 YAML 中的 layout 坐标语义是：

```text
layout [x, y/up, z] -> SimBox [x, y, z/up]
```

所以所有三维 translation 需要按下面方式转换：

```text
[x, y, z] -> [x, z, y]
```

适用位置：

- arena fixtures 的 `translation`
- task objects 的 `translation`
- source regions 的 center 派生值

`euler` 或 `rotation` 不做轴交换，按三维欧拉角写入 `euler`。

### PlaneObject

下载的 `PlaneObject` 需要转换成当前 SimBox `PlaneObject`：

```yaml
- name: floor
  target_class: PlaneObject
  size:
  - 4.0
  - 3.0
  translation:
  - 2.0
  - 1.5
  - 0.0
```

修改规则：

- `size` 只取前两个维度。
- `translation` 执行坐标轴转换。
- 源字段 `rotation` 或 `euler` 输出为 `euler`。
- 保留 `role`、`support_surface`、`static` 等运行可能使用的元数据字段。

### PlaneObject 纹理路径

下载 arena 中的纹理通常写成逻辑库名：

```yaml
texture:
  texture_lib: floor_textures
  apply_randomization: false
  texture_id: 1
```

SimBox 运行时需要能按 `asset_root` 解析到真实目录，因此需要改成：

```yaml
texture:
  texture_lib: texture_libs/floor_textures
  apply_randomization: false
  texture_id: 0
```

规则：

- 如果 `download/<scene>/texture_libs/<texture_lib>` 存在，`texture_lib` 改为相对 `asset_root` 的路径。
- `apply_randomization: false` 时，`texture_id` 固定为 `0`。当前目录中虽然图片名是 `1.png`，但加载器按列表下标取图，`0` 是第一个可用纹理。

### PlaneObject 物理碰撞

必须给所有 `PlaneObject` 补物理碰撞，包括 floor 和 wall：

```yaml
collision_enabled: true
collision_thickness: 0.02
```

原因：

- `workflows/simbox/core/objects/plane_object.py` 只有在 `collision_enabled` 为 true 时才会创建 PlaneObject 碰撞。
- 之前未补该字段时，`split_aloha` 会穿过地面下落。
- 墙面也作为大型障碍参与物理碰撞，避免机器人或物体穿出场景边界。
- Nav2 的 `Robot is out of bounds of the costmap!` 是下落后的后续症状，不是 costmap 本身的根因。

### 非 PlaneObject fixture

下载格式中的 fixture 类别可能是 `FixtureObject`、`GeometryObject`、`TaskObject`、`RigidObject`、`XFormObject`。作为 arena fixture 接入时统一转为静态几何：

```yaml
target_class: GeometryObject
```

规则：

- `path` 或 `usd_path` 必须存在，并转换为相对 `asset_root` 的路径。
- `translation` 执行坐标轴转换。
- `rotation` 或 `euler` 输出为 `euler`，缺省为 `[0.0, 0.0, 0.0]`。
- `scale` 缺省为 `[1.0, 1.0, 1.0]`。
- 保留 `category`、`asset_source_mode`、`support_surface`、`static`、`metadata`。
- 大型 fixture 默认补静态 bbox 碰撞代理：

```yaml
collision_enabled: true
collision_approximation: bbox
collision_visible: false
```

原因：

- 下载资产的 fixture USD 可能只有视觉 mesh，没有 `PhysicsCollisionAPI`。
- `support_surface: true` 的台面、柜面、床头柜等必须能承载 small object。
- 墙、柜子、沙发、床、桌子等大型障碍也需要参与物理碰撞，避免穿模。
- 当前运行时由 `workflows/simbox/core/objects/geometry_object.py` 根据上述字段创建隐藏 `collision_proxy` cube。
- 先使用 bbox 代理，避免直接用复杂 mesh collision 带来的性能和稳定性问题。

## Task YAML

### 顶层结构

当前 SimBox task 必须是：

```yaml
tasks:
- name: asset_01_kitchen_task
  asset_root: download/01_kitchen
  task: BananaBaseTask
  task_id: 0
  offset: null
  render: true
  arena_file: download/01_kitchen/simbox_arena.yaml
```

规则：

- `asset_root` 默认使用单个场景目录，例如 `download/01_kitchen`。
- `arena_file` 指向生成出的 `simbox_arena.yaml`。
- `task` 使用当前系统已有的 `BananaBaseTask`。
- `task_id` 固定为 `0`。
- `offset` 固定为 `null`。
- `render` 固定为 `true`。

### Env Map

必须使用当前仓库内可解析的 envmap 目录：

```yaml
env_map:
  envmap_lib: ../../InternDataAssets/assets/envmap_lib
  apply_randomization: false
  intensity_range:
  - 5000
  - 5000
  rotation_range:
  - 0
  - 0
```

原因：

- 之前 envmap 路径或 id 不匹配时，`banana.py::_set_envmap()` 中 `envmap_hdr_path_list[envmap_id]` 会触发 `IndexError: list index out of range`。
- 当前 `asset_root` 是 `download/<scene>`，所以 `envmap_lib` 要写成从该目录回到仓库资产目录的相对路径。

### Robot

必须使用 `split_aloha`，不能使用 `genie1`：

```yaml
robots:
- name: split_aloha
  robot_config_file: workflows/simbox/core/configs/robots/split_aloha.yaml
  path: ../../workflows/simbox/assets/split_aloha_mid_360/robot.usd
  euler:
  - 0.0
  - 0.0
  - 90.0
  ignore_substring:
  - material
  - table
  use_batch: true
  collision_activation_distance: 0.05
```

原因：

- Nav2 需要移动底盘和 `/odom`。
- `genie1` 不是当前 mobile manipulation 流程使用的底盘配置，运行时会出现等待 `/odom` 超时。

路径规则：

- `robot_config_file` 使用仓库相对路径。
- `path` 要相对 `asset_root`，所以在 `asset_root: download/<scene>` 下写成 `../../workflows/simbox/assets/split_aloha_mid_360/robot.usd`。

### Task Objects

下载 `task.yaml` 中的 `task_objects` 或 `objects` 转成 SimBox `objects`。

用于后续 pick/place 的对象必须是：

```yaml
target_class: RigidObject
apply_randomization: false
prim_path_child: Aligned/Normalize/Source/base_link
```

规则：

- `path` 或 `usd_path` 必须存在，并转换为相对 `asset_root` 的路径。
- `translation` 执行坐标轴转换。
- `rotation` 或 `euler` 输出为 `euler`，缺省为 `[0.0, 0.0, 0.0]`。
- `scale` 缺省为 `[1.0, 1.0, 1.0]`。
- `object-mode=rigid` 时输出 `RigidObject`。
- `object-mode=geometry` 只适合加载场景，不适合直接生成 pick 技能。
- `RigidObject` 必须有 `prim_path_child`。
- 转换脚本会读取 USD 中的 `RigidBodyAPI` 自动推断 `prim_path_child`：
  - 唯一 `RigidBodyAPI`：直接使用该 prim 相对 default prim 的路径。
  - 多个 `RigidBodyAPI` 且唯一末级名为 `base_link`：使用 `base_link` 路径。
  - 没有或仍然不唯一：保留默认 `Aligned` 并打印 warning，后续需单独处理资产物理标注。
- 保留 `category`、`asset_source_mode`、`metadata`、`parent_fixture`、`spawn_region`。
- 如果下载源包含 `mass_kg`，转成 `mass`。

### Regions

当前机械转换不直接把下载 object regions 放入 SimBox `regions`，而是保存到：

```yaml
source_regions:
  - ...
```

原因：

- 下载 regions 是源数据语义，不完全等同于当前 SimBox `regions` 的 random placement schema。
- 直接塞入 `regions` 容易出现 `object` 或 `target` 引用不匹配。

当前运行 task 中：

```yaml
regions:
- object: split_aloha
  target: floor
  random_type: A_on_B_region_sampler
  random_config:
    pos_range:
    - [1.45, -0.36, 0.0]
    - [1.45, -0.36, 0.0]
    yaw_rotation: [90.0, 90.0]
source_regions:
  - ...
```

`regions` 必须至少包含 robot 初始化 region：

- `object` 使用当前 robot 名，默认 `split_aloha`。
- `target` 使用 arena 中的 `floor`。
- `random_type` 使用成功导航样本同款 `A_on_B_region_sampler`。
- `pos_range` 使用第一个 robot waypoint 相对 floor bbox center 的偏移。
- `yaw_rotation` 使用第一个 waypoint yaw 减去 robot 默认 yaw 90 度。

当前 4 个场景已生成的 robot 初始化 region：

| scene | pos_range | yaw_rotation |
| --- | --- | --- |
| `01_kitchen` | `[[1.45, -0.36, 0.0], [1.45, -0.36, 0.0]]` | `[90.0, 90.0]` |
| `02_bookroom` | `[[1.35, 0.0, 0.0], [1.35, 0.0, 0.0]]` | `[180.0, 180.0]` |
| `03_livingroom` | `[[0.0, -0.95, 0.0], [0.0, -0.95, 0.0]]` | `[90.0, 90.0]` |
| `04_bedroom` | `[[-1.15, -0.8, 0.0], [-1.15, -0.8, 0.0]]` | `[90.0, 90.0]` |

原因：

- 成功样本 `navigate_asset_switchback_narrow.yaml` 通过 `A_on_B_region_sampler` 把 `split_aloha` 初始化到 `floor` 上。
- 如果 `regions` 为空，`BananaBaseTask._load_robot()` 会用默认 `translation: [0.0, 0.0, 0.0]`。
- 对下载场景，floor 往往是正坐标房间，默认原点经常位于地板角点或墙角，不是安全起点，底盘可能部分悬空并掉落。
- `01_kitchen` 的初始点已按地图手动覆盖到 prep island 东侧通道，估算世界坐标是 `(3.45, 1.14)`，最终朝向是 `180` 度；该点按 `split_aloha` 完整 USD bbox 加 `0.05m` margin 后不与 fixture bbox 相交。

如果后续 API 生成了可用的 SimBox regions，才写入 `regions`。
API 生成的 regions 只能追加，不能覆盖 robot 初始化 region。
对应脚本行为在 `build_robot_spawn_regions()` 和 `merge_llm_fragment()` 中实现。

### Positions

机械转换从下载 task 的 robot waypoints 生成 `positions`：

```yaml
positions:
  wp_k_1_sink_counter_front:
    x: 1.0
    y: 1.1
    yaw: -3.141592653589793
```

规则：

- 来源字段是 `robot.waypoints[*].pose_xy_yaw`。
- `x` 和 `y` 直接使用源 waypoint 的平面坐标。
- `yaw` 从角度转弧度，并统一归一化到 `[-pi, pi)`。
- 例如源角度 `180` 输出为 `-3.141592653589793`，`270` 输出为 `-1.5707963267948966`。
- 名字必须清洗，供 navigate skill 的 `goal` 引用。
- API 生成自然语言技能时可以增补 positions，但 navigate 的 `goal` 必须存在。
- `01_kitchen` 当前不是纯机械输出：已基于地图手动修正起点和导航点，避免 `split_aloha` footprint 压进 prep island、pantry、tray support、cooktop 等 fixture。
- 手动修正后的 `01_kitchen` positions：

```yaml
positions:
  wp_k_1_sink_counter_front:
    x: 3.45
    y: 1.14
    yaw: -3.141592653589793
  wp_k_2_prep_counter_front:
    x: 3.45
    y: 1.14
    yaw: -3.141592653589793
  wp_k_3_stove_tray_front:
    x: 3.17
    y: 1.34
    yaw: 0.0
  wp_k_4_storage_counter_front:
    x: 3.45
    y: 1.14
    yaw: -3.141592653589793
  wp_k_5_fridge_counter_front:
    x: 3.45
    y: 0.8
    yaw: -1.5707963267948966
```

### Cameras

下载场景生成的 task 不能保留 `cameras: []`，否则 pipe render 阶段没有可写出的 RGB camera，成功的 Nav2 轮次只会留下 `output/ros_bridge/skills/...` 下的轨迹 JSON 和调试快照，不会生成 `demo.mp4`。

当前按可工作的 `workflows/simbox/core/configs/tasks/navigation/split_aloha/navigate_asset_switchback_narrow.yaml` 补齐四个 camera：

```yaml
cameras:
- name: navigate_global
  translation: [2.0, 1.5, 6.0]
  orientation: [1.0, 0.0, 0.0, 0.0]
  camera_axes: usd
  camera_file: workflows/simbox/core/configs/cameras/realsense_d455_v3.yaml
  parent: ""
  apply_randomization: false
- name: split_aloha_hand_left
  translation: [0.0, 0.08, 0.05]
  orientation: [0.0, 0.0, 0.965, 0.259]
  camera_axes: usd
  camera_file: workflows/simbox/core/configs/cameras/astra.yaml
  parent: split_aloha/split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper/fl/link6
  apply_randomization: false
- name: split_aloha_hand_right
  translation: [0.0, 0.08, 0.04]
  orientation: [0.0, 0.0, 0.972, 0.233]
  camera_axes: usd
  camera_file: workflows/simbox/core/configs/cameras/astra.yaml
  parent: split_aloha/split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper/fr/link6
  apply_randomization: false
- name: split_aloha_head
  translation: [0.0, -0.00818, 0.1]
  orientation: [0.658, 0.259, -0.282, -0.648]
  camera_axes: usd
  camera_file: workflows/simbox/core/configs/cameras/realsense_d455_v3.yaml
  parent: split_aloha/split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper/top_camera_link
  apply_randomization: false
```

- `navigate_global` 是场景全局相机。`01_kitchen` 的 floor 是 `4.0 x 3.0`，中心在 `(2.0, 1.5)`，所以全局相机放在 `[2.0, 1.5, 6.0]`。
- 左右手相机和头部相机沿用 nav2 成功样本的 parent path 和外参。
- 两份输出都要同步补齐：
  - `download/<scene>/simbox_task.yaml`
  - `workflows/simbox/core/configs/tasks/download/<scene>_task.yaml`

### Skills

机械转换阶段先生成空技能骨架：

```yaml
skills:
- split_aloha:
  - base: []
    left: []
    right: []
```

自然语言任务通过 `--generate-skills-with-api` 调用 Codex 配置中的 API 生成 `positions`、`skills`，不是调用 Codex CLI。

技能结构必须保持当前 SimBox 格式：

```yaml
skills:
- split_aloha:
  - base:
    - name: navigate
      id: nav_to_pick
      depends_on: []
      goal: some_position
    left:
    - name: pick
      id: pick_object
      depends_on:
      - nav_to_pick
      objects:
      - object_name
    - name: place
      id: place_object
      depends_on:
      - nav_to_place
      objects:
      - object_name
      - target_object_or_fixture
    right: []
```

规则：

- 顶层是 list，每个 phase 是 `{robot_name: [queue_mapping]}`。
- robot key 必须是 `split_aloha`。
- `base`、`left`、`right` 三个队列都要存在。
- `navigate` 只放 `base` 队列。
- `pick`、`place`、`heuristic__skill` 放 arm 队列，当前默认 `left`。
- 使用 DAG 时，每个技能必须有唯一 `id`，`depends_on` 必须是 list。
- `navigate.goal` 必须引用 `positions` 中已有 key。
- `pick.objects[0]` 必须引用 `tasks.objects` 中的 `RigidObject`。
- `place.objects[1]` 可以引用 task object，也可以引用 arena fixture。
- 生成技能允许包含 `pick`，即使当前对象还没有 grasp 标注；grasp 标注后续补。

### Source Tasks

下载自然语言任务保留在：

```yaml
source_tasks:
  - ...
```

来源：

- `download/<scene>/interndata_scene/task.yaml`

用途：

- 作为 API 生成 skills 的上下文。
- 保留原始任务语义，便于人工复查生成技能是否语义一致。

### Data

生成 task 需要补齐当前系统常用的 `data` 字段：

```yaml
data:
  task_dir: download/01_kitchen
  language_instruction: ...
  detailed_language_instruction: ...
  collect_info: download_01_kitchen
  version: v1.0
  update: true
  max_episode_length: 1000
```

规则：

- 机械转换时说明该配置来自下载场景。
- API 生成 skills 后，`language_instruction` 和 `detailed_language_instruction` 改成描述技能来自 source task 转换。
- `max_episode_length` 当前默认 `1000`。

## 运行期问题对应的 YAML 修正

### Envmap IndexError

现象：

```text
IndexError: list index out of range
banana.py::_set_envmap()
```

原因：

- `env_map.envmap_lib` 指向的目录不对，或列表为空但仍按 id 取值。

修正：

- task YAML 中使用 `../../InternDataAssets/assets/envmap_lib`。
- `apply_randomization: false`。
- 固定 `intensity_range` 和 `rotation_range`。

### Nav2 等待 odom 超时

现象：

```text
timed out waiting for 3 odom messages on /odom
```

原因：

- 生成 task 曾使用 `genie1`，该 robot 不适配当前 Nav2 mobile manipulation 流程。

修正：

- task YAML 中 robot 改为 `split_aloha`。
- 使用 `workflows/simbox/core/configs/robots/split_aloha.yaml`。
- 使用 `../../workflows/simbox/assets/split_aloha_mid_360/robot.usd`。

### Robot out of bounds of costmap

现象：

```text
Robot is out of bounds of the costmap!
```

确认过的根因：

- robot 掉穿 floor，位姿 z 持续变成大负数，x/y 也漂移到地图外。

修正：

- arena YAML 的所有 PlaneObject 补：

```yaml
collision_enabled: true
collision_thickness: 0.02
```

- 现在该规则扩展到所有 `PlaneObject`，墙面也补同样字段。
- 所有 arena `GeometryObject` fixture 补：

```yaml
collision_enabled: true
collision_approximation: bbox
collision_visible: false
```

- 运行时 `GeometryObject` 会据此创建隐藏 bbox collision proxy。

- task YAML 的 `regions` 必须补 robot-on-floor 初始化：

```yaml
- object: split_aloha
  target: floor
  random_type: A_on_B_region_sampler
  random_config:
    pos_range:
    - [1.45, -0.36, 0.0]
    - [1.45, -0.36, 0.0]
    yaw_rotation: [90.0, 90.0]
```

- 不要把下载源里的 robot 初始位置直接写成 `robots[].translation`。
- `split_aloha` 的 robot asset 本地 bbox 最低点低于 root，直接写 `translation: [x, y, 0.0]` 容易让底盘穿进 floor；严重时会在 `get_local_pose()` 链路上触发 `Found zero norm quaternions in quat`。
- `A_on_B_region_sampler` 会按 floor bbox 和 robot bbox 自动计算安全 z，高度语义与现有可工作导航 YAML 一致。

示例：

```yaml
robots:
- name: split_aloha
  euler: [0.0, 0.0, 90.0]
regions:
- object: split_aloha
  target: floor
  random_type: A_on_B_region_sampler
  random_config:
    pos_range:
    - [1.45, -0.36, 0.0]
    - [1.45, -0.36, 0.0]
    yaw_rotation: [90.0, 90.0]
```

### Navigate waypoint 落入 fixture

现象：

```text
GridBased: failed to create plan, no valid path found.
Planning algorithm GridBased failed to generate a valid path to (2.00, 1.45)
```

排查结论：

- 以 `01_kitchen` 为例，`wp_k_2_prep_counter_front: (2.0, 1.45)` 落在 `prep_island_0_id4` 的实体 bbox 内。
- 加上 fixture collision 后，Nav2 静态地图将该点标记为 occupied，planner 在 `compute_path_to_pose` 阶段直接失败，不会发出 `cmd_vel`。

修正：

- 转换脚本会检查 `positions` 是否落入带 collision 的 `GeometryObject` fixture bbox。
- 如果落入，则按机械规则推到该 fixture bbox 外侧，并在多个可选侧面中选择离其它 fixture bbox 更宽的一侧。
- 仅检查中心点还不够，`split_aloha` 的 Nav2 footprint 可能在中心点合法时仍与 fixture bbox 相交。
- `01_kitchen` 当前已按地图手动修正为：

```yaml
wp_k_2_prep_counter_front:
  x: 3.45
  y: 1.14
  yaw: -3.141592653589793
```

同时手动修正了 robot 初始化点和 `wp_k_1/wp_k_3/wp_k_4/wp_k_5`。最新修正不再只看 Nav2 footprint，而是使用 `split_aloha` 完整 USD bbox 加 `0.05m` margin 做离线检查；所有点均未与 floor 边界或 fixture bbox 相交。

### Nav2 yaw 表示归一化

现象：

```text
nav2 skill finished: ... reason=bridge_aborted message=goal finished with status_code=6
```

排查结论：

- 该问题的直接原因不是 `+pi` 无法解析。失败记录中 goal yaw 是 `3.141592653589793`，成功记录中 Nav2 最终 yaw 为 `-3.074...` 时仍能按环绕角算出小误差。
- 但为了避免不同模块对 `+pi`、`3pi/2` 或更大角度的边界处理不一致，生成的 YAML 统一使用 `[-pi, pi)` 范围。

修正：

- `positions.*.yaw` 统一归一化到 `[-pi, pi)`。
- `navigate` skill 运行时解析 `goal_yaw` 或 `positions[goal].yaw` 后也会再次归一化。
- 已同步更新：
  - `download/<scene>/simbox_task.yaml`
  - `workflows/simbox/core/configs/tasks/download/<scene>_task.yaml`

## 生成和同步命令

仅做机械转换：

```bash
/home/dyf/miniconda3/envs/anygrasp/bin/python scripts/convert_download_scene_configs.py --all --object-mode rigid --overwrite
```

机械转换并通过 Codex 配置中的 API 生成自然语言 skills：

```bash
/home/dyf/miniconda3/envs/anygrasp/bin/python scripts/convert_download_scene_configs.py --all --object-mode rigid --generate-skills-with-api --overwrite
```

写入系统配置目录时使用：

```bash
/home/dyf/miniconda3/envs/anygrasp/bin/python scripts/convert_download_scene_configs.py --all --object-mode rigid --output-arena-dir workflows/simbox/core/configs/arenas/download --output-task-dir workflows/simbox/core/configs/tasks/download --overwrite
```

注意：

- `--generate-skills-with-codex` 只是兼容旧参数名，实际走 API，不启动 Codex CLI。
- 不带 `--generate-skills-with-api` 的机械重跑默认会保留已存在输出 YAML 中的 `positions`、`skills` 和自然语言 `data` 字段，避免清空已经生成的技能；如果需要重置，显式加 `--reset-generated-skills`。
- 如果需要同时更新源目录和系统配置目录，需要分别生成或同步。

## 校验项

转换脚本当前至少校验：

- arena fixture 的 `target_class` 是否是 SimBox 支持的类。
- arena fixture 的 `path` 是否存在。
- 所有 PlaneObject 是否有 `collision_enabled: true`。
- 所有 PlaneObject 是否有正数 `collision_thickness`。
- 所有 GeometryObject fixture 是否有 `collision_enabled: true`。
- 所有 GeometryObject fixture 是否使用 `collision_approximation: bbox`。
- task object 的 `target_class` 是否支持。
- task object 的 `path` 是否存在。
- `RigidObject` 是否有 `prim_path_child`。
- skills 引用的 object 或 fixture 是否存在。
- pick 是否引用 `RigidObject`。
- navigate goal 是否存在于 `positions`。
- DAG skill id 是否唯一，`depends_on` 是否引用已存在 id。

建议每次改完 YAML 后跑：

```bash
/home/dyf/miniconda3/envs/anygrasp/bin/python -m py_compile scripts/convert_download_scene_configs.py
```

并至少对一个场景 dry-run：

```bash
/home/dyf/miniconda3/envs/anygrasp/bin/python scripts/convert_download_scene_configs.py --scene-dir download/01_kitchen --object-mode rigid --dry-run
```

## 2026-05-31 interndata_scene 版本切换

问题现象：

- 视频中多个小物体一开始就在地下或明显偏离桌面。
- 排查发现外层 `download/01_kitchen/arena.yaml` / `task.yaml` 与 `download/01_kitchen/interndata_scene/arena.yaml` / `task.yaml` 不是同一套 task-ready 布局。
- 外层 task 转换后，多个物体与 interdata task 坐标相差超过 1m，例如水杯、盐瓶、托盘、面包片等；因此直接使用外层 YAML 会导致物体不在对应支撑面上。

修正：

- `scripts/convert_download_scene_configs.py` 现在优先使用 `download/<scene>/interndata_scene/arena.yaml` 和 `download/<scene>/interndata_scene/task.yaml` 作为转换源。
- 如果 `interndata_scene` 不存在，则仍 fallback 到外层 `arena.yaml` / `task.yaml`，保持旧场景兼容。
- `interndata_scene` 坐标已经是 SimBox `[x, y, z]`，不会再按外层 `layout [x,y,z] -> usd [x,z,y]` 交换轴。
- 外层下载 schema 仍继续使用原来的轴转换逻辑。
- `interndata_scene` 中缺失的非支撑门窗资产会被跳过，例如 `kitchen_window_0_id8212` 和 `kitchen_entry_door_0_id8214`；支撑面资产缺失仍会报错。
- `interndata_scene` 的 floor 原点表示房间左下角，转换时将 floor 平面中心修正到 `[room_width / 2, room_depth / 2, z]`，当前 01_kitchen 为 `[2.0, 1.5, 0.0]`。
- `output/main_scenes/.../task_ready_assets/assets/...` 路径会映射回当前下载目录中的 `download/<scene>/assets/...` 实际资产。
- 保留已生成 YAML 中的手工内容：`regions`、`cameras`、`positions`、`skills` 以及自然语言 `data` 字段，不因重新机械转换而重置。

本次重新生成：

```bash
/home/dyf/miniconda3/envs/anygrasp/bin/python scripts/convert_download_scene_configs.py --scene-dir download/01_kitchen --object-mode rigid --overwrite
/home/dyf/miniconda3/envs/anygrasp/bin/python scripts/convert_download_scene_configs.py --scene-dir download/01_kitchen --object-mode rigid --output-arena-dir workflows/simbox/core/configs/arenas/download --output-task-dir workflows/simbox/core/configs/tasks/download --overwrite
```

检查结果：

- `download/01_kitchen/simbox_task.yaml` 与 `workflows/simbox/core/configs/tasks/download/01_kitchen_task.yaml` 的 object 坐标已来自 `interndata_scene/task.yaml`。
- 关键物体已回到对应支撑面附近：
  - `apple_0_id9008`: `[2.0, 1.1125, 0.946]`, parent `prep_island_0_id8204`
  - `water_glass_0_id9002`: `[0.5525, 2.475, 0.966]`, parent `sink_counter_base_0_id8201`
  - `salt_bottle_0_id9011`: `[3.47, 1.75, 0.966]`, parent `storage_counter_base_0_id8203`
  - `bread_slice_a_0_id9014`: `[2.2025, 2.52, 0.916]`, parent `cooktop_counter_base_0_id8202`
  - `metal_tray_0_id9016`: `[0.38, 0.8886, 0.766]`, parent `tray_support_0_id8205`
- 所有 PlaneObject / GeometryObject fixture 仍保留 collision 配置。
- `regions`、`cameras`、`positions`、`skills` 按各自输出文件中的已有内容保留；源目录版本和 workflows 版本原本存在差异，因此保留后仍可能不同。

## 2026-05-31 手动按 interndata waypoints 修正导航点

原因：

- `interndata_scene/task.yaml` 的 `waypoints` 是任务语义中给出的导航参考点。
- SimBox `positions` 可以直接使用这些 waypoint 的平面坐标，但 `yaw` 必须由角度转弧度并归一化到 `[-pi, pi)`。
- SimBox `regions[*].random_config.pos_range` 不能直接写全局坐标；`A_on_B_region_sampler` 会把它作为相对 `target` bbox center 的 offset。
- 因此 robot 初始 region 使用第一个 waypoint `(1.0, 1.1, 180deg)` 时，应写成相对 floor center `(2.0, 1.5)` 的 offset `[-1.0, -0.4, 0.0]`，`yaw_rotation` 为 `180 - 90 = 90deg`。

手动检查：

- 原始 WP2 `(2.0, 1.45)` 位于 `prep_island_0_id8204` 的 collision bbox 内。
- 原始 WP4 `(3.35, 2.1)` 位于 `storage_counter_base_0_id8203` 的 collision bbox 内。
- WP5 `(3.45, 0.8)` 距离 pantry/fridge 较近，机器人完整 footprint 有贴边风险。
- 所以没有机械照抄全部 waypoint，而是按语义手动把压入实体或贴边的点挪到相邻通道。

当前手动结果：

```yaml
regions:
- object: split_aloha
  target: floor
  random_type: A_on_B_region_sampler
  random_config:
    pos_range:
    - [-1.0, -0.4, 0.0]
    - [-1.0, -0.4, 0.0]
    yaw_rotation: [90.0, 90.0]

positions:
  wp_k_1_sink_counter_front:
    x: 1.0
    y: 1.1
    yaw: -3.141592653589793
  wp_k_2_prep_counter_front:
    x: 1.0
    y: 1.45
    yaw: 0.0
  wp_k_3_stove_tray_front:
    x: 3.0
    y: 1.15
    yaw: -3.141592653589793
  wp_k_4_storage_counter_front:
    x: 3.0
    y: 2.1
    yaw: 0.0
  wp_k_5_fridge_counter_front:
    x: 3.1
    y: 0.8
    yaw: -1.5707963267948966
```

验证：

- 已同步修改 `download/01_kitchen/simbox_task.yaml` 和 `workflows/simbox/core/configs/tasks/download/01_kitchen_task.yaml`。
- YAML 可解析。
- 当前 navigate skill 引用的 goal 均存在。
- 所有当前 `positions` 都在 floor `(0.0, 0.0) - (4.0, 3.0)` 内。
- 按当前 fixture bbox 检查，所有当前 `positions` 的中心点都没有落入带 collision 的 GeometryObject fixture bbox。
