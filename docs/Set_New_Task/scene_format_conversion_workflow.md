# InterData ↔ SimBox 场景 YAML 格式转换工作流

## 1. 目标与范围

将任意场景在以下两种格式之间互换：

| 格式 | 场景输入 | 作用 |
|---|---|---|
| InterData | `interdata/task.yaml` + `interdata/arena.yaml` | 场景与任务语义的中间格式 |
| SimBox | `simbox_task.yaml` + `simbox_arena.yaml` | 最终仿真运行格式 |

当前目录中的具体例子：

```text
runs/memory/s04_map04/interdata/task.yaml
runs/memory/s04_map04/interdata/arena.yaml

runs/memory/base_scenes/s04_map04/simbox_task.yaml
runs/memory/base_scenes/s04_map04/simbox_arena.yaml
```

`engine.yaml` 只负责启动引擎和指向 `task.yaml`，不属于场景语义格式转换；需要时单独生成或更新。

本流程保留场景资产和布局信息，但暂不生成可执行的机器人位姿、导航 waypoint、skill/substep 编排。最终仿真输入仍然只有 `simbox_task.yaml` 和 `simbox_arena.yaml`。

同一个 `SceneIR` 可以反复导出多个任务变体：例如厨房的 fixture、支撑区域和相机保持不变，只替换 apple/banana 资产、贴图或局部物体布局。

## 2. 转换原则

1. **先规范化，再导出**：不在两种 YAML 之间逐字段硬拷贝，先转成统一的 `SceneIR`。
2. **场景与执行计划分离**：fixture、object、region、材质和物理属性属于场景；robot pose、waypoint、skill 和 substep 属于执行计划。
3. **路径只解析一次**：先依据源文件位置解析为真实文件，再按目标格式重写相对路径，禁止直接复制绝对路径。
4. **坐标显式转换**：每个对象只允许经过一次坐标变换，禁止根据文件名猜测坐标系或直接交换 Euler 分量。
5. **源文件只读**：先写入 staging 目录，验证通过后再发布到目标目录；失败时删除 staging，不能破坏源场景。
6. **未知字段不丢失**：目标格式没有对应字段时放入 `source_*` 或转换 manifest，不能静默删除。

## 3. 总体流程

```text
发现输入 pair
    ↓
读取 YAML、检测格式和路径根
    ↓
规范化为 SceneIR
    ↓
路径解析 + 坐标统一 + 字段别名统一
    ↓
清空 robot pose / waypoint / skill plan（保留 cameras）
    ↓
导出目标格式的 task.yaml + arena.yaml
    ↓
结构校验、引用校验、物理字段校验
    ↓
往返转换检查（A→B→A 或 B→A→B）
    ↓
发布两个最终 YAML 文件
```

## 4. 统一中间表示 SceneIR

转换器内部只处理一个与格式无关的对象：

```yaml
SceneIR:
  scene_id: <唯一场景名>
  coordinate_frame: <canonical world XYZ, Z-up, meter>
  fixtures: []       # 房间、地面、墙、柜台等固定结构
  objects: []        # USD 资产、变换、物理和支撑关系
  regions: []        # 物体候选区域、支撑区域、容器内部区域
  textures: {}
  environment: {}
  cameras: []
  provenance: {}     # 源文件、未映射字段、转换版本
  robot_plan: null   # 暂不导出到可执行字段
  skill_plan: null   # 暂不导出到可执行字段
```

每个 `object` 至少保留：`name`、资产路径、`target_class`、`translation/euler/scale`、`parent_fixture`、`spawn_region`、`role`、碰撞/刚体/摩擦、质量、grasp annotation 和 placement metadata。每个 `fixture` 至少保留：`name`、尺寸、变换、材质、碰撞和物理属性。

## 5. 规范化规则

### 5.1 路径

- `InterData.objects[].usd_path` ↔ `SimBox.objects[].path`。
- `InterData.arena` ↔ `SimBox.tasks[].arena_file`。
- 先将 `usd_path`、纹理和相机文件解析为真实文件；输出时改写成目标文件约定的相对路径。
- 目标路径必须能从 `asset_root` 或目标 YAML 所在目录解析；文件不存在直接报错，不生成“看似可运行”的 YAML。
- 纹理库名称、纹理文件和环境贴图保持同一相对路径根；必要时同时写入 `source_texture_lib` 供追溯。

### 5.2 坐标

SceneIR 统一采用：

```text
translation: [x, y, z]，单位 meter
up axis: +Z
地面: XY 平面
rotation: Euler XYZ，单位 degree
```

- InterData 的 `isaac_sim_world_xyz` 已是上述规范坐标，可直接进入 SceneIR。
- 如果 SimBox `coordinate_frame` 声明字段来自 legacy layout，且存在 `layout [x,y,z] -> usd [x,z,y]`，则位置转换为 `[x_c, y_c, z_c] = [x_layout, z_layout, y_layout]`。
- 如果 SimBox 字段已经声明为 world/Isaac 坐标，不得再次应用该映射。
- 旋转必须通过坐标系变换的矩阵或四元数转换后再提取 Euler；不能简单把 `euler[1]` 和 `euler[2]` 对调。存在 `asset_yaw_rule` 时，以该规则为准。
- 转换器写入 `coordinate_frame.source_frame`、`target_frame` 和 `transform_applied`，用于防止重复变换。

### 5.3 字段别名与物理属性

- `mass_kg` ↔ `mass`。
- `rigidbody`、`kinematic`、`collision_enabled`、`collider`、`friction` 统一到 SceneIR 的 physics block，再按目标格式展开。
- `initial_pose` 只在确实表达物体初始姿态时合并；不能把机器人初始位姿误写成物体姿态。
- 目标格式没有对应字段时保留在 `source_physics`、`source_metadata` 或 manifest 中。

## 6. InterData → SimBox

### 6.1 读取

1. 读取 `interdata/task.yaml`。
2. 根据 `task.arena` 读取同目录 `arena.yaml`。
3. 检查 `format: task/arena`、版本、资产路径和对象唯一性。

### 6.2 生成 `simbox_arena.yaml`

- `arena.fixtures` 映射为 SimBox `fixtures`，字段名 `usd_path` 改为 `path`。
- 保留尺寸、translation、euler、texture、collision、rigidbody、kinematic、static、physics 等字段。
- 复制并规范化 `coordinate_frame`、材质库、环境信息和导航约束。
- InterData 的 scene-level regions 如目标 SimBox schema 不接收，则转移到 `simbox_task.yaml` 的 `regions`；不得丢失或重复创建。
- 由于机器人位姿暂留空，写入：

```yaml
robot_waypoints: []
```

### 6.3 生成 `simbox_task.yaml`

根结构保持 SimBox 运行格式：

```yaml
tasks:
- name: <scene_id>
  arena_file: <relative path to simbox_arena.yaml>
  objects: [<normalized objects>]
  regions: [<object/support/container regions>]
  robots: []
  skills: []
  source_tasks: []
  waypoints: []
  positions: {}
```

具体映射：

- `objects[].usd_path` → `objects[].path`，其他资产、变换、支撑和物理字段保留。
- `task.regions` 与 `arena.regions` 合并去重；对象候选区域和支撑关系优先保留 task 侧定义。
- `environment.env_map`、`cameras`、`max_episode_length` 和非执行类 metadata 复制并规范化。
- `cameras` 属于场景采集配置，不属于 robot execution plan。即使 `skills: []`、`waypoints: []`，也必须保留固定相机和机器人挂载相机；只有源场景确实无相机或用户明确要求无相机时才可输出 `cameras: []`。
- SimBox 最终运行文件中的 `camera_file` 使用仓库根相对路径。SplitAloha 相机标定统一引用 `workflows/simbox/core/configs/cameras/*.yaml`，例如手部相机使用 `astra.yaml`、全局/头部相机使用 `realsense_d455_v3.yaml`；不得保留仅在转换 staging 或 InterData 包内有效的 scene-local 路径。
- 机器人挂载相机必须保留校准后的 `translation`、`orientation`、`camera_axes`、完整 `parent` prim 路径和 `apply_randomization`，并与所选 canonical robot profile 校验一致。
- `role`、`delivery_active_objects`、`container_regions` 等任务-场景绑定可保留，它们描述场景语义，不等于 skill 编排。
- `robot` 的名称、配置和位姿不写入 `robots`；如需追溯，放入 manifest 的 `source.robot`。
- InterData `tasks[].substeps`、导航 waypoint 和 positions 不写入可执行字段；原文放入 manifest 的 `source.tasks`。

## 7. SimBox → InterData

### 7.1 读取

1. 读取 `simbox_task.yaml`，要求根节点存在 `tasks`。
2. 读取 task 中的 `arena_file` 指向的 `simbox_arena.yaml`。
3. 若 `arena_file` 为相对路径，按 SimBox 的 `asset_root`/YAML 所在目录解析后再进入 SceneIR。

### 7.2 生成 `arena.yaml`

- `simbox_arena.fixtures` → InterData `fixtures`，`path` 改回 `usd_path`。
- 保留 `coordinate_frame`、房间边界、texture library、collision/physics 和 navigation metadata。
- `robot_waypoints` 属于执行计划，输出为 `robot_waypoints: []` 或放入 provenance，不生成可执行 robot 数据。
- SimBox task 中只描述支撑/物体的 regions 写入 InterData task；真正属于 arena 的固定结构信息写入 InterData arena。

### 7.3 生成 `task.yaml`

```yaml
format: task
format_version: 1
arena: arena.yaml
objects: [<normalized objects>]
regions: [<object/support/container regions>]
robot: null
waypoints: []
positions: {}
tasks: []
cameras: [<scene cameras, if independent of robot execution>]
environment: {<normalized environment>}
```

具体映射：

- `objects[].path` → `objects[].usd_path`。
- `regions`、`source_regions` 和 `container_regions` 统一为 SceneIR regions；同名但内容冲突时报错，不静默覆盖。
- `env_map`、`cameras`、`max_episode_length` 和 provenance metadata 复制。
- `robots` 不转换为 `robot`；输出 `robot: null`。若目标 schema 不接受 `null`，使用空 mapping `{}`，但不得填入默认位姿。
- `skills`、`source_tasks` 不转换为 `tasks`；输出 `tasks: []`，原始内容只写入 manifest。

## 8. 机器人与 skill 的留空契约

转换阶段必须满足以下不变量：

| 字段 | SimBox 输出 | InterData 输出 |
|---|---|---|
| 机器人实例/位姿 | `robots: []` | `robot: null`（或 schema 允许的 `{}`） |
| 导航路线 | `robot_waypoints: []`、`waypoints: []` | `waypoints: []` |
| 机器人派生位置 | `positions: {}` | `positions: {}` |
| skill 编排 | `skills: []` | `tasks: []` |
| 相机采集配置 | 保留并规范化 `cameras` | 保留并规范化 `cameras` |
| 原始信息 | `conversion_manifest.source` | `conversion_manifest.source` |

对象的 translation、region、support surface 和物理属性不能因为机器人字段留空而删除；它们是场景可复用性的核心。

相机也不能因为机器人执行字段留空而删除。`clear_execution_plan` 只清理动作计划，不清理观测传感器；否则 SimBox 虽可执行，但 LMDB 中不会产生预期的 RGB 流和 `demo.mp4`。

若下游 schema 必须知道机器人型号，可保留 `name` 和 `robot_config_file`，但将 `translation/euler` 设为 `null` 或省略，并标记该输出为 `scene-only`；不得自动填充默认位姿。

## 9. Agent 在工作流中的职责

Agent 不直接手改 YAML，而是调用带 schema 的转换工具：

```text
detect_format
→ load_pair
→ normalize_to_scene_ir
→ convert_direction
→ clear_execution_plan
→ validate_scene
→ round_trip_check
→ promote_outputs
```

Agent 需要：

- 自动识别输入是 InterData pair 还是 SimBox pair；
- 记录每个字段的映射、丢失原因和 warning；
- 发现路径、坐标、region 引用或物理约束错误时终止发布；
- 将失败结果回滚到 staging，并返回结构化报告；
- 不在本阶段猜测机器人型号、起始位姿或 skill 顺序。

建议命令接口（作为目标工作流，不代表当前 CLI 已全部实现）：

```bash
python scripts/convert_scene_format.py \
  --input runs/memory/s04_map04/interdata \
  --output runs/memory/base_scenes/s04_map04 \
  --direction interdata-to-simbox \
  --clear-robot --clear-skills \
  --path-mode repo-relative \
  --validate --round-trip-check
```

反向转换只需改为：

```bash
--direction simbox-to-interdata
```

## 10. 校验与往返测试

### 10.1 发布前校验

- YAML 根结构、`format/version` 和必需字段正确。
- 所有 object、fixture、region 名称唯一，`parent_fixture`、`A/B/target` 引用可解析。
- USD、纹理、相机和 arena 引用文件存在，且没有未解析的绝对路径。
- 源相机列表非空时，目标相机列表仍非空；相机名称唯一，`camera_file` 可从目标仓库根解析，机器人挂载相机的 `parent` 与 canonical robot profile 一致。
- translation/euler/scale 为有限数值，scale 大于零，单位为 meter。
- collision、rigidbody、kinematic、mass 和 friction 之间没有明显矛盾。
- `robots/robot`、waypoints、positions、skills/tasks 符合第 8 节的留空契约。
- SimBox task 引用的 `simbox_arena.yaml` 与实际输出文件一致。

### 10.2 往返不变量

执行以下测试：

```text
InterData → SimBox → InterData
SimBox → InterData → SimBox
```

在容差内应保持：

- fixture/object/region 的 ID、资产路径、尺寸、scale 和变换；
- parent/support 关系、材质和物理属性；
- 坐标系声明以及路径解析结果。

允许的有意差异：字段别名、文件布局、缓存字段，以及机器人位姿、waypoint、skill/substep 等被明确清空的执行信息。建议位置误差阈值为 `1e-6 m`，角度误差阈值由坐标转换器统一设定。

## 11. 输出目录与非目标

推荐先生成：

```text
<scene>/conversion_staging/<direction>/
  simbox_task.yaml
  simbox_arena.yaml
  conversion_manifest.yaml
  validation_report.json
```

验证通过后，将对应的两个 YAML 发布到目标目录；`conversion_manifest.yaml` 和 `validation_report.json` 仅用于追溯，不作为 SimBox 仿真输入。

生成场景的最终 SimBox pair 应发布到 `runs/<suite>/base_scenes/<scene_id>/`。例如本场景的服务器运行文件是 `/data1/zepeng/Gen-Simulation-merge-debug/runs/memory/base_scenes/s04_map04/simbox_task.yaml`；生成任务根目录和 `conversion_staging` 只作为源与中间产物，不能被 launcher 当作最终 task config。

本工作流不负责：

- 生成或修改 USD 几何、碰撞网格和 grasp annotation；
- 求解机器人初始位姿、可达性或导航路线；
- 生成 skill、substep 或完整 task plan；
- 代替 Isaac Sim 做最终物理运行验证。

这些内容由后续 robot placement、skill planner 和仿真验证阶段填充。
