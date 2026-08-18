# 碰撞规划与 Pick/Place 闭环

> 覆盖时间：2026-08-02 ~ 2026-08-08
> 涉及提交：baa37d2, 98e558a, 432c004, b2c943f
> 涉及代码：workflows/simbox/core/planning/collision_scene_manager.py, workflows/simbox/core/planning/config_contract.py, workflows/simbox/core/planning/motion_command.py, workflows/simbox/core/execution/safety_monitor.py, workflows/simbox/core/controllers/template_controller.py, workflows/simbox/core/skills/pick.py, workflows/simbox/core/skills/place.py, workflows/simbox/core/skills/heuristic_skill.py, workflows/simbox_dual_workflow.py, workflows/simbox/core/utils/json_utils.py, workflows/simbox/core/utils/attach_collision_utils.py, agent/config.yaml

## 背景

PhysX、CuRobo、Skill、Controller 四层对同一物体身份理解不一致，产生几类现象：visual mesh 进入 CuRobo 世界（WORLD-001）、关键词误删真实障碍（WORLD-002）、Pick 过程中目标从 pre-grasp world 提前消失（STATE-001）、attach/detach 分散导致双重存在或两边都不存在（STATE-002）、把 world 中 40 个精确 collider 误当成 40 个独立 attach 输入（STATE-003）、Place 没有只允许接触支撑面的有限阶段（PLACE-001）、轨迹消费完但实际未到位被判完成（EXEC-002）。修复把职责拆成四层。

### 四层职责

- **CollisionSceneManager**（collision_scene_manager.py:119）：以启用 CollisionAPI 的 Prim 为唯一事实源。_discover()（:370）建立 records / path_to_entity / attach_prim_paths；build_world_config()（:547）按精确 Prim path 构建 CuRobo world；bind_controller()（:680）注册 controller 并 audit；维护物体状态机（_transition :1126 / assert_invariants :1391）与动态位姿同步（sync_dynamic_poses :1355，默认每 5 步）。导出 collision_world_audit.json 与 object_state_events.jsonl（export :1536）。
- **Pick / Place**（pick.py / place.py）：只产生带语义的 MotionPhaseCommand（motion_command.py:37），不直接改 world。phase 序列为 SYNC_WORLD → TRANSIT_PREGRASP → TERMINAL_GRASP_APPROACH → GRIPPER_CLOSE → ATTACH → CARRY_HOME（motion_command.py:19，98e558a 新增）→ TRANSIT → TERMINAL_PLACE_DESCENT → DETACH_AND_SETTLE → TERMINAL_RETREAT → RESTORE_WORLD。调试产物 pick_plan_snapshot.json / pick_execution_trace.json 经 json_ready() 转换后写入（pick.py:200 _write_debug_artifact）。
- **TemplateController**（template_controller.py）：forward_phase_command()（:1297）逐 phase 执行，_begin_phase_command()（:1271）触发 manager 状态迁移；ee_forward()（:1619）把规划结果按 ds_ratio 步长消费，轨迹消费完（cmd_plan 置空、_phase_plan_finished）且 is_phase_command_complete()（:1509）用实测 EE 位姿误差在容差内才算完成（EXEC-002）。
- **SafetyMonitor**（safety_monitor.py:88）：每个物理步在下一动作应用前 evaluate()（:174）把测量换算成 continue / hold_and_replan / abort 决策，事件写入 safety_events.jsonl（export :274）。

agent/config.yaml 的 planning.collision_world.mode 默认 physics_schema，编译器把契约写入最终任务配置（agent/compiler.py:218-264），未迁移 Skill 必须显式使用 legacy_stage_scan。

## 碰撞世界模式

模式常量定义在 config_contract.py:18-21，四种枚举的语义与使用场景：

| 模式 | 定义 | 使用场景 |
|------|------|----------|
| physics_schema | PHYSICS_SCHEMA_MODE（:18） | 默认模式。CollisionSceneManager 以精确 CollisionAPI Prim 建 world，只放行已迁移 Skill（pick / place / pick_plan_probe）；CollisionSceneManager 构造时非此模式直接抛 ValueError（collision_scene_manager.py:134） |
| legacy_stage_scan | LEGACY_STAGE_SCAN_MODE（:19） | 旧关键词扫描 world（template_controller.py _legacy_update :935，usd_parser.get_obstacles_from_stage + ignore_substring）。未迁移 Skill 在 physics_schema 模式下被 validate_planning_contract() 拒绝，必须显式声明此模式（config_contract.py:175-178） |
| hybrid | HYBRID_MODE（:20） | resolve_collision_world_mode()（:100）在 auto 下检测到 legacy skill 时回落；未迁移 skill 仍走 legacy world，已迁移 skill 走 physics world |
| passthrough | PASSTHROUGH_MODE（:21） | 非操作 Skill（navigate / observe_hold）直接放行，不切换 controller world（resolve_skill_collision_world_mode :44-45） |

- **默认值**：agent/config.yaml:60-63，planning.collision_world.mode: physics_schema、strict: true、exact_exclusions: []。
- **Skill 集合**（config_contract.py）：PHYSICS_SCHEMA_SKILLS = {pick, place}（:8）；VALIDATION_ONLY_SKILLS = {pick_plan_probe}（:9，要求 metadata.workspace_probe 且恰 1 个 object，:162-171）；NON_MANIPULATION_SKILLS = {navigate, observe_hold}（:13-16）；ATTACHED_PHYSICS_SCHEMA_SKILL_MODES = {("heuristic__skill", "home")}（:10-12，运行时携带 attached 物体时保持 physics world 直到 detach，resolve_runtime_skill_collision_world_mode :53）。
- **逐 skill 决议**：resolve_skill_collision_world_mode()（:37）按 NON_MANIPULATION→passthrough、显式 legacy→legacy、physics 集合→physics_schema、其余回落 legacy 的顺序；resolve_collision_world_mode()（:100）在 auto 下先判 task 是否含 physics skill，再按 legacy 集合是否为空选 hybrid 或 physics_schema。
- **test_mode 强制**：resolve_skill_test_mode()（:87）在 physics_schema/hybrid 下强制 forward，test_mode: ik 只保留给显式 legacy_stage_scan Skill；agent/registry/skill_contracts.yaml 中 pick/place 的 collision_world_modes: [physics_schema]，test_mode owner: compiler 且只允许 forward。
- **契约校验**：validate_planning_contract()（:129）拒绝双臂并发（UNSUPPORTED_CONCURRENT_MANIPULATION，:151-156）与 pick/place 物体数不符（pick=1、place=2，:179-185）。

## 物体状态机

CollisionObjectState（collision_scene_manager.py:30-37）覆盖物体从世界障碍到被操作再到落地的完整身份：

- WORLD_OBSTACLE：纯世界障碍，物理与 CuRobo world 均启用。
- ACTIVE_TARGET_TRANSIT：Pick 目标，transit 阶段对所有 controller world 保持启用（begin_target_transit :1190）。
- ACTIVE_TARGET_APPROACH：进入终末抓取，owner 的 world 中目标禁用，其它 controller 仍启用（begin_target_approach :1197）。
- ATTACHED：attach_target()（:1206）先恢复 owner 的 world enabled，调 controller.attach_objects()（template_controller.py:1920，native attachment_manager.attach :1945，输入显式 attach_collision_prim_paths + native mesh），成功后显式禁用该实体全部精确 world collider 并记录 _attached_relative_pose；CuRobo attach 失败时回滚为 APPROACH 不变式再抛错（:1215-1225）。
- PLACEMENT_CONTACT：begin_placement_descent()（:1243）只对 owner 的 CuRobo world 临时禁用 support collider（_temporary_disabled），PhysX 碰撞保持开启（PLACE-001）；是否接触支撑面由 SafetyMonitor 判。
- PLACED_WORLD：detach_target()（:1283）先 detach_obj() 移出 attached spheres，进 _pending_detach 等待物理 settle 窗口，finalize_detach_target()（:1295）读 Stage 实测 pose 后恢复 world；restore_world()（:1308）按原状态恢复为 WORLD_OBSTACLE 或 PLACED_WORLD。
- DISABLED：仅用于诊断分组场景，只能回到 WORLD_OBSTACLE。

迁移合法性由 _ALLOWED_TRANSITIONS（:58-88）约束，_transition()（:1126）同时校验 owner 一致性：任何 ACTIVE/ATTACHED 状态必须有 owner，且不允许第二个手臂同时持有活动物体（UNSUPPORTED_CONCURRENT_MANIPULATION，:1147-1163）。

assert_invariants()（:1391）在每个迁移与 reset 后调用：

- ATTACHED / PLACEMENT_CONTACT：owner 的 controller_enabled 中该实体路径必须全部关闭，且（非 pending_detach 时）controller.has_attached_collision_spheres() 必须为真（:1394-1406）。
- 其它非 APPROACH / DISABLED 状态：所有 physics controller 中该实体的 world collider 不得被意外禁用（_temporary_disabled 例外，:1407-1424）。

## SafetyMonitor 检查项

SafetyMeasurements（safety_monitor.py:21-41）按类分五组：

- **跟踪**：joint_error_rad、ee_position_error_m、ee_orientation_error_rad（阈值 joint_error 0.10/0.25 rad、ee_position 0.03/0.06 m、ee_orientation 0.15/0.30 rad，DEFAULT_THRESHOLDS :67-85）。
- **底座**：base_translation_m、base_rotation_deg（soft 0.01 m / 2.0°，hard 0.03 m / 5.0°）。
- **接触**：unexpected_contact_n（机器人/环境，数据源 get_unexpected_robot_contact_force :966）、unexpected_object_contact_n 与 allowed_object_support_contact_n（物体/环境，get_object_environment_contact_forces :1000，按 parent_fixture 支持面拆分）、手指接触走 get_finger_environment_contact_forces（:1049）。soft 5 N、hard 20 N。
- **滑移**：attached_slip_translation_m、attached_slip_rotation_deg（get_attached_object_slip :1077，相对 attach 时刻的 EE 相对位姿差分）。平移滑移 > 0.02 m 是硬中止条件（:136-137）；旋转滑移 b2c943f 起只记录不再 abort（:138-144，碰撞场景 pose 路径的 3x3 块含 USD scale，0.001 缩放的 wine-glass 资产会产生约 120° 伪漂移）。
- **动态世界变化**：dynamic_obstacle_changed（sync_dynamic_poses :1355 检测到物体位移超 dynamic_translation_replan_m 0.01 m 或旋转超 3.0°）、plan_failed、tracking_completion_failed；这三项作为软触发单步即 HOLD_AND_REPLAN（:214-219），其余软触发需连续 soft_trigger_consecutive_steps（默认 3）步（:210-213）。

硬触发（_hard_trigger :109，nan / joint_limit / abnormal_velocity / illegal_object_state / attached_object_dropped + 硬阈值超限）直接 ABORT；evaluate() 在 replan_allowed 时把软触发转 HOLD_AND_REPLAN，超过 max_replans_per_phase 后转 ABORT（workflow 侧控制）。

## 时间线

| 日期 | 提交 | 内容 |
|------|------|------|
| 2026-08-02 | baa37d2 | Checkpoint physics-schema hybrid planning：CollisionSceneManager 唯一碰撞清单与物体状态机、配置契约四模式、Pick/Place 的 physics-schema 分支、执行闭环默认值 |
| 2026-08-03 | 98e558a | Implement closed-loop hybrid collision planning：新增 CARRY_HOME 阶段、attached home 运行时 physics 决议、pick_plan_snapshot.json 与 start-state 碰撞诊断 |
| 2026-08-07 | 432c004 | checkpoint local navigation and physics skill updates：visual-only 无 CollisionAPI 实体跳过、刚体 sibling 分支根解析、physics 模式强制 test_mode=forward |
| 2026-08-08 | b2c943f | Update local navigation and physics skills：SafetyMonitor 旋转滑移判定调整、refresh_after_task_reset 重发现 collider 并重建 world |

## 修改记录

### 2026-08-02

#### 2026-08-02 · 碰撞世界唯一清单与物体状态机（baa37d2）
- 改动：CollisionSceneManager 以启用 CollisionAPI 的 Prim 为边界建立唯一清单（_discover，collision_scene_manager.py:370，实体遍历只进 task.fixtures/objects/distractors，机器人、相机、调试 Prim 不在入口内，WORLD-001）；精确 exact_exclusions 取代 ignore_substring 关键词（validate_exact_exclusions :99，WORLD-002）。CollisionObjectState 状态机覆盖 WORLD_OBSTACLE / ACTIVE_TARGET_TRANSIT / ACTIVE_TARGET_APPROACH / ATTACHED / PLACEMENT_CONTACT / PLACED_WORLD，_ALLOWED_TRANSITIONS（:58）约束迁移，并新增 assert_invariants()（:1391）（STATE-002）。attach_target()（:1206）用配置的合并 attach 代理经 controller.attach_objects()（template_controller.py:1920，native attachment_manager，STATE-003）；begin_placement_descent()（:1243）只在 owner 的 CuRobo world 临时禁用 support，PhysX 碰撞不关（PLACE-001）；detach_target()/finalize_detach_target()（:1283/:1295）先移除 attached spheres 等待 settle，再读取真实 Stage pose 恢复 world。build_world_config()（:547）按精确 Prim path 构建 world，sync_dynamic_poses()（:1355）每 5 步同步动态位姿。attach_collision_utils.py 新增 resolve_attach_collision_prims() 等路径解析辅助（attach_collision_utils.py:107）。
- 原因：四层对同一物体身份理解不一致，旧扫描遍历全部 UsdGeom、以资产名当物理语义、attach 与 world 注册混淆。
- 文件：workflows/simbox/core/planning/collision_scene_manager.py, workflows/simbox/core/utils/attach_collision_utils.py
- 验证：test/unit/test_collision_scene_manager.py 增 186 行，新增 test/unit/test_motion_state_continuity.py、test/unit/test_json_utils.py、test/unit/test_attach_collision_utils.py；见后续整体回归。

#### 2026-08-02 · 配置契约四模式（baa37d2）
- 改动：config_contract.py 定义 PHYSICS_SCHEMA_SKILLS={pick, place}（:8）、VALIDATION_ONLY_SKILLS={pick_plan_probe}（:9）、NON_MANIPULATION_SKILLS={navigate, observe_hold}（:13-16）；mode 枚举 physics_schema / legacy_stage_scan / hybrid / passthrough（:18-21）；resolve_skill_collision_world_mode()（:37）逐 skill 决议，physics_schema 模式下未迁移 Skill 报错并要求显式 legacy_stage_scan（:175-178）；resolve_collision_world_mode()（:100）在 auto 下检测存在 legacy skill 时回落 hybrid；validate_planning_contract()（:129）拒绝双臂并发（UNSUPPORTED_CONCURRENT_MANIPULATION）和 pick/place 物体数不符的配置（CONFIG-001）。
- 原因：配置不能再承担本应由 Stage Physics schema 表达的资产语义。
- 文件：workflows/simbox/core/planning/config_contract.py
- 验证：test/unit/test_planning_config_contract.py 增 117 行。

#### 2026-08-02 · Pick/Place 拆阶段与执行闭环接入（baa37d2）
- 改动：pick.py 的 debug 产物改为经 json_utils.json_ready()（json_utils.py:13）转换，_write_debug_artifact()（pick.py:200）对写入失败只告警不打断 episode；template_controller.py 增 _configure_execution_stride()（:366，physics 模式按 physics/interpolation dt 设 ds_ratio，上限 max_waypoint_stride）、activate_collision_world_mode()（:892）；simbox_dual_workflow.py 增 _bind_skill_collision_world_mode()（:740）、_activate_skill_collision_world()（:1569），碰撞 world 按 skill 激活，reset_episode() 同步重置 collision manager 与 contact view（:1152）；SafetyMonitor 默认仅在 physics 模式启用（EXEC-001/EXEC-002 所属闭环在 2026-08-02 之前的 48f1c44 已建立，本轮调整其默认行为）。
- 原因：debug 产物写入失败不能破坏 episode；碰撞契约按 skill 生效而不是全局切换。
- 文件：workflows/simbox/core/skills/pick.py, workflows/simbox/core/controllers/template_controller.py, workflows/simbox_dual_workflow.py, workflows/simbox/core/utils/json_utils.py
- 验证：test/unit/test_json_utils.py、test/unit/test_template_controller_collision_mode.py 新增；compileall 与 pyflakes 通过。

### 2026-08-03

#### 2026-08-03 · CARRY_HOME 阶段与 attached home（98e558a）
- 改动：motion_command.py 新增 MotionPhase.CARRY_HOME（motion_command.py:19）；config_contract.py 新增 ATTACHED_PHYSICS_SCHEMA_SKILL_MODES={("heuristic__skill", "home")}（:10-12）与 resolve_runtime_skill_collision_world_mode()（:53），静态决议把 heuristic__skill 留在 legacy 集合，运行时携带 attached 物体时保持 physics world 直到 detach；heuristic_skill.py 补 attached home 的 CARRY_HOME 命令与 world 恢复逻辑（heuristic_skill.py:213 _physics_schema_generate_manip_cmds）；template_controller.py forward_phase_command() 处理 CARRY_HOME（:1341，要求 preplanned joint path 且经 assert_attached_owner 校验）。
- 原因：携带物体的 home 不能进入 legacy world，否则 attached 物体跨 world 不安全。
- 文件：workflows/simbox/core/planning/motion_command.py, workflows/simbox/core/planning/config_contract.py, workflows/simbox/core/skills/heuristic_skill.py, workflows/simbox/core/controllers/template_controller.py
- 验证：新增 test/unit/test_heuristic_attached_home.py、test/unit/test_hybrid_runtime_activation.py、test/unit/test_nav2_dynamic_approach.py。

#### 2026-08-03 · pick_plan_snapshot.json 与 start-state 碰撞诊断（98e558a）
- 改动：pick.py 在候选评估后写出 pick_plan_snapshot.json（pick.py:616-617），包含 plan_evaluation、sample_debug、geometry_debug、pregrasp/grasp 位姿与 world_collision_diagnostic；当 pregrasp_success_count==0 且不可行时调用 manager.diagnose_controller_world_collision()（collision_scene_manager.py:815，逐实体分组 enable/disable 后 check_current_start_state，定位碰撞实体）记录 start-state 碰撞诊断。
- 原因：需要区分候选生成失败与执行阶段问题，避免把规划失败笼统当点位不可达。
- 文件：workflows/simbox/core/skills/pick.py, workflows/simbox/core/planning/collision_scene_manager.py, workflows/simbox/core/controllers/template_controller.py
- 验证：test/unit/test_collision_scene_manager.py 增 96 行、test_planning_config_contract.py 增 45 行。

### 2026-08-07

#### 2026-08-07 · visual-only 实体跳过与 sibling 分支刚体根（432c004）
- 改动：collision_scene_manager.py 的 _discover()（:370）对整棵子树无 CollisionAPI 且配置未声明 collision 的实体，以固定理由 visual_only_no_collision_api 记入 schema_exclusions（:470）并跳过，不再触发 strict 报错；碰撞 Mesh 位于刚体 sibling 分支（而非子孙）时，_rigid_body_paths()（:335）从对象子树内第一个 RigidBodyAPI Prim 解析刚体根（:351-355），避免误判为不兼容。
- 原因：遗留 arena 会把纯渲染 fixture 根注册进实体集合，与真实障碍混在一起。
- 文件：workflows/simbox/core/planning/collision_scene_manager.py
- 验证：test/unit/test_collision_scene_manager.py 相应用例。

#### 2026-08-07 · physics 模式强制 forward 规划（432c004）
- 改动：config_contract.py 新增 resolve_skill_test_mode()（:87），physics_schema/hybrid 下把 test_mode 强制为 forward，test_mode: ik 只保留给显式 legacy_stage_scan Skill；移除 validate_planning_contract() 中对 test_mode 的直接拒绝；pick.py/place.py 的候选评估改经 resolve_skill_test_mode() 取 test_mode。
- 原因：IK-only 检查不验证碰撞-free 路径，不能让未迁移 Skill 用 legacy 语义绕过 physics 世界。
- 文件：workflows/simbox/core/planning/config_contract.py, workflows/simbox/core/skills/pick.py, workflows/simbox/core/skills/place.py
- 验证：test/unit/test_planning_config_contract.py。

### 2026-08-08

#### 2026-08-08 · SafetyMonitor 旋转滑移判定调整（b2c943f）
- 改动：safety_monitor.py 移除 attached_slip_rotation_deg 的 abort 判定（_hard_trigger :138-144 注释保留说明），旋转漂移仍写入 SafetyMeasurements 但不再是中止条件；原因是当前 collision-scene pose 路径会在 3x3 块中包含 USD scale，把该仿射块喂给旋转角公式会对 0.001 缩放的 wine-glass 资产产生约 120° 的伪漂移，即使物体没有转动。
- 原因：缩放信息混入位姿仿射块导致误报旋转滑移。
- 文件：workflows/simbox/core/execution/safety_monitor.py
- 验证：test/unit/test_collision_scene_manager.py 增 31 行。

#### 2026-08-08 · refresh_after_task_reset 重建碰撞世界（b2c943f）
- 改动：collision_scene_manager.py 新增 refresh_after_task_reset()（:1451），任务重载随机化重建刚体 USD 后：先对当前绑定 controller 清除 attached CuRobo 状态（has_attached_collision_spheres / detach_obj，:1464-1477），清空 schema_exclusions/records/attach_prim_paths/pose/审计等全部按精确 Prim path 键控的结构（:1481-1497），重新 _discover()（:1499），再为每个 physics controller 重建 world（build_world_config + planner/batch_planner.update_world + _make_world_update_signature + audit_controller，:1505-1532）后 world_revision 自增；workflow 的 _reset_controllers() 在 TemplateController.reset() 前调用它（simbox_dual_workflow.py:1074-1080）。
- 原因：随机化 retry 可能删除重建刚体 USD，替换物暴露不同的精确 collider path，旧记录与 CuRobo world 仍含旧路径；普通 reset_episode 只在旧记录上重置，会复现 ATTACH_COLLISION_PRIM_NOT_IN_CUROBO_WORLD。
- 文件：workflows/simbox/core/planning/collision_scene_manager.py, workflows/simbox_dual_workflow.py
- 验证：test/unit/test_collision_scene_manager.py。