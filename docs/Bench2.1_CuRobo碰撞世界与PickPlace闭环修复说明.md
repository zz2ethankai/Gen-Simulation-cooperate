# Bench2.1 CuRobo 碰撞世界与 Pick/Place 闭环修复说明

## 1. 全局视角：这次修复解决的是什么

这不是单纯给水槽补一个障碍物，也不是再增加一组 `ignore_substring`。问题位于整个操作链的中间契约：PhysX 决定 Stage 中什么东西真的会碰撞，CuRobo 决定规划时“看见”什么，Skill 决定什么时候允许目标接触，Controller 决定如何消费轨迹。四者只要有一处对同一物体的身份理解不一致，就可能出现“规划成功，但执行时撞到”“目标 A 被忽略后连带忽略 B”“轨迹消费完但机械臂没有到位却被判完成”等现象。

新链路把职责拆成四层：

1. `CollisionSceneManager` 从真实 Physics schema 建立唯一碰撞清单，并维护物体在 world、terminal contact、attached、placed 等状态之间的身份切换。
2. `Pick` 和 `Place` 只产生带语义的 `MotionPhaseCommand`，不再用无语义 tuple 暗中修改碰撞世界。
3. `TemplateController` 负责执行单个 phase，只有轨迹消费完且实际 EE 到位才允许完成。
4. `SafetyMonitor` 每个物理步在下一动作应用前检查跟踪、底座、接触、物体滑移和动态世界变化，决定继续、保持并重规划或中止。

Agent 运行时默认值在 [`agent/config.yaml:38`](../agent/config.yaml#L38)，编译器会把它写入最终任务配置，并阻止源任务或 LLM 降低碰撞契约。`physics_schema` 是新默认；未迁移 Skill 必须显式使用 `legacy_stage_scan`，禁止无声回退。

## 2. 问题、根因和修改总表

| 编号 | 症状 | 根因 | 新入口 |
|---|---|---|---|
| WORLD-001 | visual mesh、相机或调试几何可能进入 CuRobo world | 旧扫描遍历全部 `UsdGeom`，没有以 Physics collider 为边界 | [`CollisionSceneManager._discover():269`](../workflows/simbox/core/planning/collision_scene_manager.py#L269) |
| WORLD-002 | `sink/table/counter` 等关键词可能误删真实障碍，未来 Agent 生成新名称后规则失效 | `ignore_substring` 把资产名字当作物理语义 | [`get_obstacles_from_collision_prims():546`](../workflows/simbox/curobo/src/curobo/util/usd_helper.py#L546) |
| STATE-001 | Pick 过程中目标可能从 pre-grasp world 提前消失 | 同一个忽略目标的 world 同时验证 transit 和 terminal | [`Pick._physics_schema_generate_manip_cmds():105`](../workflows/simbox/core/skills/pick.py#L105) |
| STATE-002 | attach/detach 分散，可能出现 world 和 attached 双重存在或两边都不存在 | 没有统一物体状态和不变量 | [`CollisionObjectState:30`](../workflows/simbox/core/planning/collision_scene_manager.py#L30)、[`assert_invariants():887`](../workflows/simbox/core/planning/collision_scene_manager.py#L887) |
| STATE-003 | 盐瓶路径执行到 attach 后返回 `max_spheres: 4 n_objects: 40`，或把容量粗暴增至 128 后单进程显存接近 24 GiB | 把“world 中 40 个精确 collider”误当成“40 个独立 attach 输入”，混淆了碰撞注册与挂接代理 | [`attach_target():708`](../workflows/simbox/core/planning/collision_scene_manager.py#L708)、[`attach_objects():833`](../workflows/simbox/core/controllers/template_controller.py#L833) |
| PLACE-001 | Place 没有“只允许物体接触支撑面”的有限接触阶段 | 整个 Place 共用一种碰撞策略 | [`begin_placement_descent():743`](../workflows/simbox/core/planning/collision_scene_manager.py#L743) |
| EXEC-001 | CuRobo 路径与实际运动偏离后仍继续消费旧轨迹 | 原执行循环是开环发送 | [`_execution_safety_precheck():733`](../workflows/simbox_dual_workflow.py#L733) |
| EXEC-002 | 等待十余帧后，即使没有到位也弹出命令 | 旧完成条件是 `pose_flag OR num_last_cmd > 10` | [`is_phase_command_complete():621`](../workflows/simbox/core/controllers/template_controller.py#L621) |
| EXEC-003 | 橙色轨迹留在原位置，机器人底座和双臂整体被碰撞推走 | SplitAloha 的三个平面移动 joint 在交付 USD 中 drive gain 为 0，而 CuRobo 路径使用 arm-base 坐标系 | [`enable_manipulation_base_hold():242`](../workflows/simbox/core/robots/template_robot.py#L242)、[`manipulation_base_hold:38`](../workflows/simbox/core/configs/robots/split_aloha.yaml#L38) |
| CONFIG-001 | 临时 YAML 依赖资产名称，不能扩展到 Agent 自动创作配置 | 配置承担了本应由 Stage Physics schema 表达的事实 | [`validate_planning_contract():11`](../workflows/simbox/core/planning/config_contract.py#L11) |
| ROBOT-001 | 环境建模正确时，机械臂仍可能用未被球覆盖的表面撞击环境 | Piper 原模型仅 15 个 collision sphere，最大漏覆盖约 88.26 mm | [`audit_piper_collision_spheres.py:90`](../scripts/simbox/audit_piper_collision_spheres.py#L90) |

## 3. WORLD-001 / WORLD-002：碰撞世界如何建立

### 3.1 唯一事实来源

[`CollisionSceneManager`](../workflows/simbox/core/planning/collision_scene_manager.py#L118) 只遍历 `task.fixtures`、`task.objects` 和 `task.distractors`。机器人、相机和调试 Prim 不在遍历入口内。每个实体内部只注册满足以下条件的 Prim：

- 带启用的 `UsdPhysics.CollisionAPI`；
- 类型是 `Mesh/Cube/Sphere/Cylinder/Capsule`；
- 没有出现在带理由的精确 `exact_exclusions` 中。

可移动性由 `RigidBodyAPI`、`kinematicEnabled` 和 `ArticulationRootAPI` 推导，只决定 pose 如何同步，不决定“是否是障碍物”。因此 B 现在即使以后也要被抓取，在抓 A 时仍然是障碍物；等 B 自己成为 active target 时，才只在 B 的 terminal approach 中对 owner controller 临时禁用。

显式声明为 `collision_enabled: false` / `collider: none` 的视觉实体（例如书房地毯）允许没有 Stage collider，并以固定理由进入 `schema_exclusions`。这项配置只用于区分“明确的视觉实体”和“声称可碰撞但资产缺 schema 的错误”：它绝不会根据配置凭空注册障碍；真正进入 CuRobo 的几何仍必须来自 Stage 中启用的 `CollisionAPI`。

Bench 资产中存在一种真实结构：父级 `Xform` 也被错误施加了 `CollisionAPI`，但真正的碰撞几何位于后代 Mesh。非几何父级不会送给 CuRobo；只有当它确实拥有已注册的后代 collider 时，才以 `schema_exclusions` 的固定理由记入审计。这不是名字过滤，也不会吞掉后代碰撞。

### 3.2 精确转换和审计

[`UsdHelper.get_obstacles_from_collision_prims()`](../workflows/simbox/curobo/src/curobo/util/usd_helper.py#L546) 接受完整 Prim path 列表和 arm-base 参考系，拒绝重复路径、缺失 Prim、disabled collider、无 `CollisionAPI` 的 Prim 和不支持的类型。它不接受 `ignore_substring`。

旧 [`get_obstacles_from_stage()`](../workflows/simbox/curobo/src/curobo/util/usd_helper.py#L478) 完整保留并标记为 `LEGACY_STAGE_SCAN`。新模式不会调用它；Controller 中旧入口保留在 [`_legacy_update():421`](../workflows/simbox/core/controllers/template_controller.py#L421) 和 [`_legacy_update_specific():906`](../workflows/simbox/core/controllers/template_controller.py#L906)。

初始化和 episode 导出都会比较 Physics collider path 与每个 Controller 的 CuRobo `world_model.objects`。缺失或多余任一方都属于硬错误。

## 4. STATE-001 / STATE-002：统一物体身份状态机

状态定义见 [`motion_command.py`](../workflows/simbox/core/planning/motion_command.py#L12) 和 [`collision_scene_manager.py`](../workflows/simbox/core/planning/collision_scene_manager.py#L30)。核心状态语义如下：

```text
WORLD_OBSTACLE / PLACED_WORLD
  -> ACTIVE_TARGET_TRANSIT       所有 controller 中仍启用
  -> ACTIVE_TARGET_APPROACH      只在 owner controller 中禁用目标
  -> ATTACHED                    owner world 禁用，attached spheres 启用
  -> PLACEMENT_CONTACT           attached 保持，owner 暂时禁用 support
  -> PLACED_WORLD                settle 后读取实际 pose 并恢复 world
```

关键不变量：

- 普通物体对所有 controller 都是 enabled world obstacle；
- terminal grasp 只允许 owner 看不见 active target，另一只臂仍把它当障碍物；
- attached 时 owner 的 world obstacle 必须关闭；
- world 碰撞注册与 attach 几何是两份不同但都精确的契约：`collision_prim_paths` 包含实体全部 Physics collider，`attach_prim_paths` 必须来自显式 `attach_prim_path_children`，并且必须是前者的子集。attach 时关闭全部 world collider，只用配置的合并碰撞代理拟合 32 个 carried-object spheres；禁止回退到 `children[0]`；
- detach 不是立刻“假定已放好”：[`detach_target():783`](../workflows/simbox/core/planning/collision_scene_manager.py#L783) 先移除 attached spheres，等待物理 settle；[`finalize_detach_target():795`](../workflows/simbox/core/planning/collision_scene_manager.py#L795) 再读取真实 Stage pose、更新 CuRobo 并重新启用；
- 检测到另一只臂同时拥有 active/attached 物体时返回 `UNSUPPORTED_CONCURRENT_MANIPULATION`。

动态物体每 5 步检查一次；平移或旋转超过配置阈值时增加 world revision，并触发执行监督器在旧轨迹继续消费前保持和重规划。实现见 [`sync_dynamic_poses():859`](../workflows/simbox/core/planning/collision_scene_manager.py#L859)。

## 5. Pick / Place 的新定义

### 5.1 为什么必须拆阶段

“抓取”不是一种统一碰撞策略。远距离 transit 必须完全无接触；最后几厘米接近目标时必须允许手指接触目标，但仍禁止手腕/前臂碰目标和环境；attach 后目标又从 world obstacle 变成机器人携带碰撞体。Place 同理。因此新命令使用 [`MotionPhaseCommand`](../workflows/simbox/core/planning/motion_command.py#L36) 显式携带 active object、support、允许接触类型、是否可重规划和完成容差。

### 5.2 Pick

标准 Pick 新入口为 [`pick.py:105`](../workflows/simbox/core/skills/pick.py#L105)：

1. `SYNC_WORLD`：同步 A、B 和环境，全部启用。
2. `TRANSIT_PREGRASP`：在包含 A 的完整 world 中规划。
3. 对每个候选，从它对应的 pre-grasp 末端关节状态重新规划 terminal path，而不是从机器人初始状态再规划一次。入口是 [`test_batch_forward_from_paths():938`](../workflows/simbox/core/controllers/template_controller.py#L938)。
4. [`measure_cartesian_path():957`](../workflows/simbox/core/controllers/template_controller.py#L957) 通过 FK 检查 terminal path：路径长度不超过直线 1.5 倍，最大偏离不超过 1 cm。
5. 只执行最终共同通过的 pre-grasp 和 terminal path；terminal 中 owner 暂时禁用 A，B 和环境保持启用。
6. `GRIPPER_CLOSE` dwell 完成后先读取目标 contact view；没有目标-手指接触时返回 `GRASP_CONTACT_MISSING`、恢复完整 world，并禁止进入 `ATTACH`。
7. 接触确认后 attach；抬升时 A 由 attached spheres 表示，并继续和 B、台面、水槽等碰撞检查。Physics-schema 路径不再沿用 legacy 的 1 cm world-frame attach 偏移，attached spheres 对齐物体真实 Stage pose。

旧 tuple 生成器完整保留在 [`_legacy_simple_generate_manip_cmds():301`](../workflows/simbox/core/skills/pick.py#L301)，只有 legacy 模式可达。

### 5.3 Place

标准 Place 新入口为 [`place.py:85`](../workflows/simbox/core/skills/place.py#L85)：

1. 到 pre-place 时 A 保持 attached，support、B 和环境全部启用。
2. 最后最多 10 cm 为 `TERMINAL_PLACE_DESCENT`；owner CuRobo world 只临时禁用 support，PhysX 碰撞不关闭。
3. active object 的 contact view 将 A-support 力与 A-其他环境力分开。首次合法 A-support 接触会停止剩余下降命令；机器人 link-support 或 A-其他环境接触仍进入安全判定。
4. 打开夹爪后进入 `DETACH_AND_SETTLE`，等待配置的 10 个物理步，再把真实落点同步回 CuRobo。
5. retreat 是单独短程阶段，只在 retreat 临时禁用 A；结束后恢复完整 world。
6. 最终成功仍使用实际物体 bbox/落点与原任务区域约束判断，不把“命令发完”当作成功。

旧 Place 生成器保留在 [`_legacy_simple_generate_manip_cmds():208`](../workflows/simbox/core/skills/place.py#L208)。

## 6. EXEC-001 / EXEC-002：执行安全闭环

[`SafetyMonitor`](../workflows/simbox/core/execution/safety_monitor.py#L85) 将每步测量映射为三种决策：`CONTINUE`、`HOLD_AND_REPLAN`、`ABORT`。工作流在应用下一动作之前调用 [`_execution_safety_precheck()`](../workflows/simbox_dual_workflow.py#L733)，因此触发安全事件后不会再多消费一个旧轨迹点。

每步检查：

- command joint 与实际 joint 的最大误差；
- command joint FK 与实际 EE 的位置/姿态误差；
- 当前 phase 起点以来的 base 平移和旋转；
- 非手指机器人 link 与 Physics 环境接触；
- attached object 与 support/其他环境接触；
- object 相对 EE 的 attach 后滑移；
- NaN、关节越界和异常速度；
- 动态物体是否跨过重规划阈值；
- collision state 不变量、规划失败和轨迹结束但未实际到位。

软异常连续 3 帧才触发，动态障碍移动、规划失败和“轨迹结束未到位”立即触发。恢复流程为：清空 `cmd_plan`、用当前关节保持 5 步、同步真实 joint/EE/base/object pose、从真实状态重规划同一个 phase。每 phase 最多重规划 2 次；之后转为 abort。硬阈值、强碰撞、物体脱落或非法状态立即 abort。hard abort 不是只清空 Python 变量：执行循环在同一个物理步把实测关节写成 hold target，避免 PhysX drive 再追逐上一个轨迹点。

新完成判定见 [`is_phase_command_complete():621`](../workflows/simbox/core/controllers/template_controller.py#L621)：运动命令必须同时满足“轨迹已消费完”和“实际 EE 在容差内”。旧 `pose_flag OR num_last_cmd > 10` 仅保留在 Pick [`_legacy_is_subtask_done():612`](../workflows/simbox/core/skills/pick.py#L612) 与 Place [`_legacy_is_subtask_done():609`](../workflows/simbox/core/skills/place.py#L609)。

CuRobo 插值为 100 Hz、PhysX 约为 60 Hz。新模式根据两者 `dt` 自动设置 `ds_ratio`（当前为 2），避免把 100 Hz 的每个插值点都按 60 Hz 慢速发送；legacy 模式仍固定为 1。

### 6.1 EXEC-003：为什么还要锁定移动底座

先前视频中“橙色轨迹和机器人实际轨迹完全不同”的直接物理原因，不只是 arm joint 索引：SplitAloha 的 `mobile_translate_x`、`mobile_translate_y`、`mobile_rotate` 为导航保留，交付 USD 中 stiffness/damping/max-force 为 0。机械臂碰到水槽后，反作用力可以推动整个移动底座；CuRobo 仍在原 arm-base frame 中输出正确轨迹，所以画面表现为轨迹留在桌面旁、机器人整体漂走。

[`TemplateRobot.enable_manipulation_base_hold():242`](../workflows/simbox/core/robots/template_robot.py#L242) 只按机器人配置中的完整 joint name 解析这三个 DOF，在 Pick/Place 物理第一步之前读取当前位置并施加 position hold。它不通过 `mobile/base` 关键词猜测 joint，也不修改导航动作定义。每次 episode reset 后重新捕获实际布局姿态；运行日志中的 reset 前后 target 必须一致。base 的软/硬漂移阈值仍由 SafetyMonitor 独立检查，drive hold 不是监控的替代品。

## 7. ROBOT-001：Piper collision sphere 修复

[`fit_piper_collision_spheres.py`](../scripts/simbox/fit_piper_collision_spheres.py#L95) 使用 URDF collision mesh 的确定性表面采样、farthest seed 和 Lloyd 聚类生成保守球集。左右臂共用新的 `spheres/piper100_collision_audited_20260720.yml`，每只臂 144 个球。

原 15 球配置没有删除，在 `piper100_left_arm.yml` 和 `piper100_right_arm.yml` 中以 `ROBOT-001 LEGACY_BEGIN/END` 注释完整保留。独立随机表面采样审计见 [`audit_piper_collision_spheres.py`](../scripts/simbox/audit_piper_collision_spheres.py#L90)：30,000 点/link 的验证中，两侧最大未覆盖距离均为 `0.004887619 m`，低于 5 mm 标准。

球覆盖通过只说明“不会漏掉真实机器人表面”；更多球会让规划更保守并增加碰撞计算量，后续可以在保持 5 mm 上界的前提下继续做球数压缩。

`attached_object` 的预留球数与上述 144 个机械臂表面球是不同用途。旧值 4 以注释保留，新值 32 用来拟合配置明确指定的单个合并 attach proxy；不能再把实体几十个 world collider 逐个送入 attach，否则 CuRobo 会平均分配容量，或者因扩大优化图导致显存不可接受。

## 8. 配置契约与 Future Agent 责任

公共默认值见 [`agent/config.yaml:38`](../agent/config.yaml#L38)，确定性编译入口见 [`compiler.py:200`](../agent/compiler.py#L200)。编译器合并任务级 planning 参数后，会重新写入系统持有的 `collision_world` 契约，Agent 不能通过返回内容关闭或放宽它。

Future Agent 的责任是：

- 为物理上会碰撞的几何正确施加并启用 `CollisionAPI`；
- 为可动物体正确施加 `RigidBodyAPI/kinematic`；
- 为可抓取物体给出精确 `attach_prim_path_children`；
- 标准 Pick/Place 使用结构化对象身份。

Future Agent 不再负责猜测 `sink/table/counter` 等环境关键词。若确需排除 collider，必须提供完整 Prim path 和非空 reason；排除路径不在已注册实体内会被拒绝。物理 schema 模式遇到 DynamicPick、ManualPick、DexPick/DexPlace、Open/Close 等未迁移 Skill 会在启动前报错，并要求显式 legacy mode。

## 9. 诊断产物

[`workflow.save():1062`](../workflows/simbox_dual_workflow.py#L1062) 在每个成功或可保存失败 episode 中导出：

- `collision_world_audit.json`：实体、mobility、collision Prim、精确排除、schema 排除、world revision，以及每个 controller 的 Physics/CuRobo 差集；
- `object_state_events.jsonl`：每次状态迁移、owner、step 和 revision；
- `safety_events.jsonl`：触发值、阈值、phase、plan id、replan index 和最终决策；
- 已有 `trajectory_debug.usda` 与 `skill_targets_debug.usda`。

这些文件和视频/LMDB 一起保存。安全失败不是无信息丢弃：只要配置 `save_failed`，仍关闭 MP4 writer、保存 episode 并导出诊断。

## 10. 测试证据与当前边界

### 10.1 离线与单元测试已通过

- Physics-schema 内存 Stage：enabled collider 收集、visual/disabled 排除、名称无关、非几何父级审计、unsupported strict error、精确排除校验；
- A/B 状态机的非法转换和双臂并发拒绝；
- MotionPhaseCommand 参数和语义校验；
- SafetyMonitor 软阈值防抖、硬阈值、动态变化、非法状态、object collision 和 attached slip；
- ExecutionSupervisor 的 hold 步数、同 phase 两次重规划预算和第三次 abort；
- GraspPlanEvaluator 的 full-world/terminal-world 顺序、batch 与 non-batch 链式起点、共同可行候选，以及 terminal 直线性过滤；
- visual-only entity 跳过、声称碰撞但缺 collider 的 strict error，以及 world collider / attach proxy 分离；
- 四份 generated YAML 的公共配置和可视化配置；
- Piper 左右臂独立 30,000 点/link、16 组 joint pose 审计均小于 5 mm。

最终回归为 `71 passed`，无 warning；相关 Python 文件同时通过 `compileall` 与 `pyflakes`。Piper 最终审计产物为 [`piper_collision_sphere_audit_final_20260720.json`](../output/bench21_collision_closed_loop_validation/piper_collision_sphere_audit_final_20260720.json)，两臂最大未覆盖距离均为 `0.004887619 m`。

测试命令：

```bash
pytest -q \
  test/unit/test_collision_scene_manager.py \
  test/unit/test_motion_phase_command.py \
  test/unit/test_safety_monitor.py \
  test/unit/test_execution_supervisor.py \
  test/unit/test_grasp_plan_evaluator.py \
  test/unit/test_planning_config_contract.py \
  test/unit/test_bench21_pick_debug_config.py \
  test/unit/test_curobo_trajectory_visualization.py \
  test/unit/test_skill_target_visualization.py \
  test/unit/test_bench21_joint_tracking_audit.py \
  test/unit/test_attach_collision_utils.py \
  test/unit/test_joint_index_resolver.py

python scripts/simbox/audit_piper_collision_spheres.py \
  --samples-per-link 30000 \
  --output output/bench21_collision_closed_loop_validation/piper_collision_sphere_audit_final_20260720.json
```

### 10.2 Isaac Sim 实跑结果

四份 generated YAML 的 seed 0 最终回归见 [`run_report.md`](../output/bench21_collision_closed_loop_validation/four_cases_final_20260720/run_report.md) 和 [`run_report.json`](../output/bench21_collision_closed_loop_validation/four_cases_final_20260720/run_report.json)：

| 场景 | 规划/执行结果 | 安全含义 | 保存结果 |
|---|---|---|---|
| kitchen salt bottle | Pick、目标接触门控、单 attach proxy、lift 全部通过；task success | joint tracking mean/p95/max 为 `0.0109/0.0446/0.0598 rad`，无安全中止 | LMDB、7 路 MP4、两个 debug USD、三类诊断均存在 |
| kitchen white cup | `NO_JOINT_GRASP_PLAN` | 完整 collider world 中没有联合可行路径，未执行碰撞轨迹 | 失败 episode 与全部诊断正常保存 |
| bedroom hand cream | `NO_JOINT_GRASP_PLAN` | 同上 | 同上 |
| bookroom cup | `NO_JOINT_GRASP_PLAN` | 同上；visual-only rug 只做可审计 schema exclusion | 同上 |

这里的 `1/4 task success` 不是闭环只覆盖了一个场景：其余三个是新定义下的预期“安全规划失败”，没有继续消费伪成功路径。四个 episode 的 `physics_curobo_difference` 都为空。

为了实际进入 Place，新建了不覆盖正式 generated YAML 的验证副本 [`validation_config.yaml`](../output/bench21_collision_closed_loop_validation/vertical_place_final_20260720/validation_config.yaml)：在已通过的 salt Pick 后串接 `vertical Place(salt_bottle -> cutting_board)`。实跑结果为 Pick/attach 成功，Place 在携物完整 world 中返回 `NO_COLLISION_FREE_PREPLACE_PLAN`；[`object_state_events.jsonl`](../output/bench21_collision_closed_loop_validation/vertical_place_final_20260720/data/BananaBaseTask/split_aloha/runs/kitchen_salt_bottle_placement/kitchen_salt_bottle_placement/fail_2026-07-20_06_31_22_643946/object_state_events.jsonl) 显示物体停留在合法 `ATTACHED` 状态，没有错误 detach 或继续撞击。

仓库旧 horizontal Place 配置 [`hang_the_cup_on_rack_part1.yaml:132`](../workflows/simbox/core/configs/tasks/basic/split_aloha/hang_the_cup_on_rack/hang_the_cup_on_rack_part1.yaml#L132) 的首次 strict 实跑在初始化时返回 `physics_schema discovered no collision entities`。这说明该旧资产没有可注册的 `CollisionAPI`，不能用 visual mesh 冒充障碍继续验收。它必须先补齐 Physics schema，或在未迁移前显式使用 `legacy_stage_scan`；本轮没有伪造 horizontal Place 的物理成功结论。

### 10.3 明确边界

- 首版只迁移标准 Pick/Place，并禁止两臂同时操作；
- 不自动为缺少 Physics schema 的资产猜测碰撞几何；strict 模式会直接指出缺失；
- contact view 依赖操作物体具有真实 `RigidBodyAPI`；缺失时启动失败，不用 visual mesh 代替；
- 当前碰撞球模型优先保证不漏碰，规划性能仍可继续优化；
- vertical Place 已验证到“携物 pre-place 安全规划失败”，尚未在当前 Bench 固定点位走到 descent/contact/detach；horizontal 旧资产尚未完成 Physics schema 迁移；
- A/B 状态转换、接触矩阵和两次重规划预算已有离线测试，但本轮未另造一个 Isaac A/B 阻挡场景，也未向真实机器人注入强碰撞、底座漂移或物体脱落；
- 因此当前可以确认 Physics world、标准 Pick、attach、tracking 与安全失败保存链路；不能把尚未进入的 Place terminal 和真实故障注入写成已物理通过。
