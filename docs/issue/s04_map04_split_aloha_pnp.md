# s04_map04 / Split Aloha PnP：Task YAML 生成 Agent 约束

> 本文面向负责生成 TaskPlan 或其编译结果 simbox_task.yaml 的 Agent。
> 它是 s04_map04 的场景特化约束，不是运行时修复记录。
> 通用 TaskPlan、Skill 和 YAML 规则分别以 agent/prompts/、docs/api/ 和当前代码为准。

## 0. 生成原则

Agent 的目标是生成一份可验证的移动操作任务，而不是把任务 YAML 填满。无法由
manifest、资产几何或 Probe 证实的字段必须省略或写入 unresolved，不能猜测。

优先级如下：

1. 当前代码和实际 manifest；
2. 本文的场景约束；
3. scene8 等历史任务的参数写法。

历史运行日志只能说明失败模式，不能直接当作坐标、碰撞体或成功证据。

当前 Agent 架构中，Agent 返回 TaskPlan，确定性 compiler 写入运行目录中的 YAML 副本；
不要修改源场景资产或源任务文件。若使用旧的直接 YAML 生成入口，也必须遵守同一组约束。

本文只描述 Split Aloha 版本。若 manifest 选择的是 PandaOmron 或其他机器人，必须使用该
机器人的真实 config、USD、EE/base path 和已验证手臂，不能混用 Split Aloha 的机器人字段。

## 1. 唯一输入与资源路径

场景入口和机器人资源：

~~~text
场景根目录：InternDataAssets/assets/custom/s04_map04
任务入口：  InternDataAssets/assets/custom/s04_map04/simbox_task.yaml
Arena：     InternDataAssets/assets/custom/s04_map04/simbox_arena.yaml
机器人配置：workflows/simbox/core/configs/robots/split_aloha.yaml
机器人 USD：InternDataAssets/robots/split_aloha_mid_360_virtual/robot.usd
~~~

生成前必须从任务入口读取 asset_root、arena_file、robots、objects、regions 和
skills；不要根据目录名、source_name 或自然语言名称自行拼接运行时对象名。

- objects[].name、regions[].object/target/A/B、Skill objects[] 使用运行时精确名称。
- source_name 只保留为来源元数据，不能用于运行时引用。
- arena_file、机器人 USD、对象 USD 和纹理路径都按 asset_root/仓库根目录的实际解析规则检查。
- Arena 的地面/墙纹理使用场景包内的 interdata/texture_libs/floor_textures、
  interdata/texture_libs/wall_textures；不要改成机器上的绝对路径。

## 2. 场景资产不变量

生成 PnP 任务时，桌子、固定设施和 9 个场景对象的原始
translation/euler/scale/path 是只读输入，不得为了让规划通过而搬动、缩放或替换资产。

本任务的操作关系固定为：

~~~text
pick：  red_cube_block_in_the_tray_as_part_of_the_matching_work_set
place：compact_placement_board_with_three_empty_slots
~~~

其余 8 个对象只是场景上下文或支撑物：

- low_rectangular_cover_shell_sized_to_fully_occlude_the_reference_row
- red_cube_block_for_the_reference_row
- green_cube_block_for_the_reference_row
- yellow_cube_block_for_the_reference_row
- green_cube_block_in_the_tray_as_part_of_the_matching_work_set
- yellow_cube_block_in_the_tray_as_part_of_the_matching_work_set
- shallow_open_tray_holding_the_matching_work_set_blocks
- compact_placement_board_with_three_empty_slots（Place 的目标，同时是静态支撑物）

运行版可以将未操作对象加载为 GeometryObject、static: true、kinematic: true，以避免
背景刚体在 reset 或执行期间下落并触发动态重规划；但必须保留其原始 USD、姿态、尺度和
碰撞几何。被抓取的红色方块必须保持可抓取的刚体和启用的碰撞，不能把它改成静态物体。

固定物体通常使用 apply_randomization: false。不要用删除/重建对象来“修复”重试状态。

## 3. 碰撞与附着契约

被抓取红色方块的附着碰撞路径必须完整引用以下实际 Prim：

~~~text
Aligned/Collision/component_000_hull_000
...
Aligned/Collision/component_000_hull_023
~~~

即共 24 个路径。生成前检查每个路径在对象 USD/运行时 Prim 下存在并启用。

必须遵守：

- 不能只保留 4 个路径以适配旧的 attached_object 容量；这会丢失对象碰撞几何。
- 若选定 Split Aloha 右臂的 CuRobo 配置容量小于 24，应报告
  ATTACH_CAPACITY_EXCEEDED 并停止生成；不能通过改 Task YAML、删路径或添加新 box 规避。
- 不得用新造的 collision_proxy 替代源资产的 Native CollisionAPI mesh、PhysX shape 或
  CuRobo 几何。
- table 必须保留在 CuRobo 和 PhysX 的碰撞世界中；不能为了规划通过把桌子加入忽略列表。
- 不能把视频中不可见、审计字段缺失或单层 proxy 缺失直接判定为“没有碰撞体”；应分别检查
  USD CollisionAPI、PhysX shape、CuRobo world 和附着路径。

本场景允许的任务级 CuRobo 排除名单（仅用于非桌面大型家具）为：

~~~yaml
ignore_substring:
  - material
  - floor
  - wall
  - scene
  - chair
  - bench
  - floor_lamp
  - floor_plant
  - cabinet
  - bookcase
  - bookshelf
  - shelf
  - cart
~~~

它只能写在 tasks[].robots[].ignore_substring，不写到历史的
planning.collision_world.ignore_substring。它只过滤 CuRobo 的规划输入，不关闭 USD/PhysX
碰撞，也不影响附着碰撞。TaskPlan Agent 不应自行发明新的排除项。

## 4. 区域和坐标规则

场景当前有 9 个对象区域加 1 个机器人起始区域。每个 A_on_B_region_sampler 必须满足：

- 对象区域的 A/object 是实际对象名；
- B/target/parent_fixture 是实际支撑物名，9 个桌面对象区域的 target 为
  wide_oak_work_table；
- 机器人区域的 target 为 floor，A/object 为实际机器人名；
- objects[].spawn_region 和 placement.spawn_region 能在 regions[].name 中找到；
- random_config 只传 sampler 支持的参数，通常为 pos_range 和 yaw_rotation。
  support_surface_z 是区域元数据，不要作为未确认的 sampler keyword 传入。

区域拥有最终运行位姿时，translation: [0, 0, 0] 不能被解释成物体的世界真值。应读取
region 的 runtime_placement、支撑物世界位姿和 sampler 结果，得到最终 world pose。
不能把桌面局部坐标、父物体坐标、floor-center 坐标重复相加。

若使用 scene-4 的 positions 生成导航目标，必须遵循：

~~~text
world_x = floor_center_x + x
world_y = floor_center_y + y
~~~

positions 是 floor-center relative；不要另加一个 reference-frame 字段，也不要把物体的
局部 [0, 0] 当成导航目标。导航 approach 字段直接引用真实对象名，而不是桌子、region
名或 [0, 0] 占位值。

## 5. Split Aloha 初始位姿与工作区

Split Aloha 使用：

~~~yaml
robot_config_file: workflows/simbox/core/configs/robots/split_aloha.yaml
path: InternDataAssets/robots/split_aloha_mid_360_virtual/robot.usd
~~~

本场景已有的参考起始位姿是：

~~~yaml
translation: [-1.0, -0.05, 0.0]
euler: [0.0, 0.0, 0.0]
~~~

机器人和 robot_start_region 必须完全同步：区域的 center/world_translation、固定
random_config.pos_range、world_euler/yaw_range 以及 robots[].euler 不能各写一套。
区域应为 target: floor、placement_mode: fixed_from_robot_start_position。

上述位姿是已知参考点，不替代验证。生成 Agent 必须以当前任务的中心物体和指定手臂运行
工作区流程：

1. Geometry：底盘 footprint 不与桌子、墙和家具相交，且在 floor 边界内；
2. CuRobo Probe：用指定手臂检查 pre-grasp/grasp 联合可达性，并带上完整附着碰撞路径；
3. Pick 验证：在真实运行中确认抓取接触和抬升。

候选点必须由 scripts/simbox/plan_workspace_layout.py 和现有 Probe/validator 产生。
仅凭“桌边界减去机身长度”的间隙估算，或仅凭无碰撞二维 overlay，不能证明工作点可用。
0.65 等导航距离不是通用安全阈值；导航点还必须满足底盘稳定性、携物后的碰撞余量和
机械臂可达性。

如果没有一个通过指定手臂 Probe 的候选点，返回 NO_COMMON_WORKSPACE_CANDIDATE 或
相应 blocked 状态，不生成一份依赖猜测坐标的任务。

## 6. 当前运行时要求的 Skill DAG

Scene-4 移动 PnP 不能只生成 pick → place 两个节点。当前基本任务使用 5 节点 DAG：

~~~text
nav_to_pick → pick_red_cube_from_tray
pick_red_cube_from_tray → nav_to_place
nav_to_place → place_red_cube_in_left_slot
place_red_cube_in_left_slot → home_right
~~~

Split Aloha 本参考任务使用右臂；每个 Skill 的 arm/controller 必须与该选择一致。结构示意：

~~~yaml
skills:
- split_aloha:
  - base:
    - id: nav_to_pick
      name: navigate
      depends_on: []
      approach: red_cube_block_in_the_tray_as_part_of_the_matching_work_set
      approach_arm: right
      approach_object_armbase_xy: [0.5, 0.0]
      xy_goal_tolerance: 0.1
      yaw_goal_tolerance: 0.1
    - id: nav_to_place
      name: navigate
      depends_on: [pick_red_cube_from_tray]
      approach: compact_placement_board_with_three_empty_slots
      approach_arm: right
      approach_object_armbase_xy: [0.5, 0.0]
      xy_goal_tolerance: 0.1
      yaw_goal_tolerance: 0.1
  - right:
    - id: pick_red_cube_from_tray
      name: pick
      depends_on: [nav_to_pick]
      objects: [red_cube_block_in_the_tray_as_part_of_the_matching_work_set]
    - id: place_red_cube_in_left_slot
      name: place
      depends_on: [nav_to_place]
      objects:
      - red_cube_block_in_the_tray_as_part_of_the_matching_work_set
      - compact_placement_board_with_three_empty_slots
    - id: home_right
      name: heuristic__skill
      depends_on: [place_red_cube_in_left_slot]
      mode: home
      gripper_state: 1.0
  - left: []
~~~

导航使用 ROS-free local A* 和 waypoint controller。不要重新引入 Nav2/ROS，也不要在 YAML
里添加不存在的 heading-controller 开关。导航成功只表示底盘到达目标附近，不表示后续 arm
可达或 Place 一定成功。

## 7. Pick 参数基线

对 Split Aloha 右臂，只有在资产/Probe 支持时才覆盖默认值；可复用的参考组合为：

~~~yaml
filter_x_dir: [forward, 90]
filter_z_dir: [downward, 140]
grasp_side_preference: toward_arm
pre_grasp_offset: 0.05
gripper_change_steps: 20
t_eps: 0.025
o_eps: 1
process_valid: true
lift_th: 0.02
post_grasp_offset_min: 0.10
post_grasp_offset_max: 0.15
~~~

注意：

- objects 必须只有一个精确的被抓取对象名；
- lift_th 只控制 Pick 成功判定，不改变抬升目标；
- post_grasp_offset_min/max 控制计划抬升距离，不能在代码中另加隐藏上限；
- 不要为了“增加候选”同时随意添加 filter_y_dir、fixed_orientation 或 direction_to_obj；
  每次只使用有几何/运行证据的约束。

## 8. Place 参数和零候选防护

左槽目标的几何比例来自 placement board 的真实 bbox，参考范围为：

~~~yaml
place_direction: vertical
position_constraint: object
success_mode: xybbox
x_ratio_range: [0.1, 0.3]
y_ratio_range: [0.25, 0.75]
pre_place_z_offset: 0.15
place_z_offset: 0.10
success_xy_margin: 0.015
~~~

姿态过滤的语义必须按当前实现理解：filter_*_dir 约束 EE 局部轴对应的旋转矩阵列，
方向是基座轴；负方向使用 element <= cos(angle)。例如：

~~~text
filter_z_dir: [downward, 150]
~~~

实际表示 EE Z 轴进入向下约 30° 的圆锥（180° - 150°），不是“向下 ±150°”。

不要把 scene8 中的以下组合当成无条件可用模板：

~~~yaml
filter_x_dir: [forward, 45]
filter_z_dir: [downward, 150]
~~~

这两个条件在当前 3000 个随机旋转样本下有效区域很稀疏，可能得到 0 个候选；问题是
采样/过滤约束不足以产生候选，不应靠随机重试或修改 transformation_utils.py 掩盖。
垂直放置至少先用已验证的 filter_z_dir，只有在离线采样和 Probe 确认候选数大于 0 后
才增加 filter_x_dir 或物体轴对齐约束。

align_pick_obj_axis、align_place_obj_axis、align_obj_tol 只有在资产轴向有明确依据时
才写入。place.objects 的顺序必须是 [被拿物体, 目标物体]。

当前 Place 运行时在候选姿态为空时会记录 NO_VALID_PLACE_ORIENTATION，由现有失败路径
输出 failure_reason 和 has_target: false，不能让原始 np.stack 异常逃出 episode。
生成 Agent 仍必须在运行前发现并拒绝这种 YAML；捕获异常不是把无效任务当成成功。

## 9. 生成前检查清单

生成并提交运行副本前，逐项确认：

- [ ] 所有 object、fixture、robot、region 和 camera 引用都能在 manifest/YAML 中闭环。
- [ ] 9 个对象区域的桌面 target 为 wide_oak_work_table，机器人区域 target 为 floor。
- [ ] spawn_region 和 placement.spawn_region 都引用真实 region 名。
- [ ] 24 个红方块 attach Prim 全部存在；选定 CuRobo 配置容量不少于 24。
- [ ] 未操作对象的静态/运动学设置没有改变其 USD、姿态、尺度或原生碰撞几何。
- [ ] table 未被加入 ignore_substring；忽略列表若存在，只是任务级 CuRobo 输入过滤。
- [ ] 机器人位姿、robot_start_region 和 positions 没有相互矛盾的坐标系或重复偏移。
- [ ] DAG 为 nav_to_pick → pick → nav_to_place → place → home，依赖无环，Skill 对象顺序正确。
- [ ] 每个可执行 Subtask 已明确指定 left 或 right，不能用 auto 代替。
- [ ] Place 姿态过滤离线得到至少一个候选；不接受 NO_VALID_PLACE_ORIENTATION。
- [ ] 导航点通过 Geometry、指定手臂 Probe 和必要的 Pick 验证；二维 overlay 只能作设计证据。

## 10. 运行验证与失败归因

真实验证必须使用项目 wrapper 的单任务运行。成功标准同时包括：

~~~text
Task is successful, mode=plan_with_render
且无 [LmdbLogger] Episode failed
且无 Traceback
~~~

有视频或场景加载成功不等于任务成功。优先检查第一个失败阶段：

| 现象 | Agent 应检查 | 不要做的事 |
|---|---|---|
| ATTACH_PRIM_NOT_IN_CUROBO_WORLD / attach 容量不足 | 24 个精确路径、机器人 CuRobo 容量、Physics/CuRobo 映射 | 截断 attach 列表、删除原碰撞体 |
| NO_JOINT_GRASP_PLAN | 指定手臂、工作点、EE 轴、抓取标注和 Pick 过滤 | 只改任务名称或盲目放宽所有过滤 |
| 动态物体持续位移、重规划后中止 | 未操作对象是否仍是动态刚体、reset 是否恢复原位 | 关闭桌子或隐藏碰撞 |
| no_reachable_approach_goal | world/nav 坐标、底盘 footprint、携物余量、arm reach | 把 0.65 当成所有任务的固定距离 |
| NO_VALID_PLACE_ORIENTATION | filter_*_dir、对齐约束和候选数量 | 修改 np.stack 或把失败当成功 |
| NO_COLLISION_FREE_PLACE_PLAN | 目标 bbox、pre/place 高度、携物完整碰撞世界 | 误判为 Pick 失败 |
| Pick 已抬升但 Place 失败 | Pick trace 中物体 z、Place 快照和最终 XY/Z | 把下游 Place 失败归因给 Pick |

Agent 只能根据日志、快照、Probe 和视频共同给出成功/失败结论；缺少关键证据时必须保留
blocked/unresolved 状态。

## 11. 输出报告

每次生成至少记录：

- 精确的 pick object、place object 和选定手臂；
- 机器人初始 [world_x, world_y, yaw]、坐标来源和 workspace candidate id；
- attach Prim 数量及容量检查结果；
- Pick/Place 姿态过滤和离线候选数量；
- YAML、workspace manifest、Probe 结果和运行日志的路径；
- 未解决的几何、资产或运行问题。

只有通过严格运行判据后，才可以把任务标记为成功或把配置推广给其他场景。

相关参考：

- [agent/prompts/Agent任务规划与Skill编排规范.md](../../agent/prompts/Agent任务规划与Skill编排规范.md)
- [agent/prompts/Agent中心物品选择与机器人初始位姿生成规范.md](../../agent/prompts/Agent中心物品选择与机器人初始位姿生成规范.md)
- [docs/api/02_Task与Arena_YAML.md](../api/02_Task与Arena_YAML.md)
- [docs/api/01_Skill_API.md](../api/01_Skill_API.md)
- [docs/development/04_PickPlace稳定性与姿态过滤.md](../development/04_PickPlace稳定性与姿态过滤.md)
