# Pick/Place 稳定性与姿态过滤

> 覆盖时间：2026-06-10 ~ 2026-08-06
> 涉及提交：36d7f46, 6f99314, 98e558a, d5b563e（c8095b8 为 d5b563e 在 isaac-sim-6-curobo-v2 分支上的同一 hash）
> 涉及代码：workflows/simbox/core/skills/pick.py, workflows/simbox/core/skills/place.py, workflows/simbox/core/controllers/curobo/controller.py, workflows/simbox/core/utils/constants.py, workflows/simbox/core/configs/robots/panda_omron_virtual.yaml, agent/config.yaml, scripts/visualize_place_orientation_filters.py

## 背景

Pick 失败要区分两个阶段：候选生成阶段看 pick_plan_snapshot.json（candidate、success_found、pregrasp_success_count），执行阶段看 pick_execution_trace.json（命令转换时序）和 pick_runtime_failure_snapshot.json（运行期失败）。Place 失败看 place_success_check_snapshot.json 的 success_mode、bbox 限制、margin 与最终 XY/Z，小偏差属于落点/沉降问题而不是 pick 失败。place 姿态过滤的 filter_*_dir 参数语义常被误解：filter_z_dir: [downward, 150] 的实际向下容差是 180-150=30°，写 downward, 30 反而几乎不约束。

## 时间线

| 日期 | 提交 | 内容 |
|------|------|------|
| 2026-06-10 | 36d7f46 | Checkpoint successful scene4 mobile manipulation fixes：pick_execution_trace.json 产物、lift_th 与 success_xy_margin 等参数引入 |
| 2026-07-09 | 6f99314 | Fix panda omron navigation and manipulation stability：pick_runtime_failure_snapshot.json 产物引入 |
| 2026-08-03 | 98e558a | Implement closed-loop hybrid collision planning：grasp_contact_threshold_n 接触确认与 pick_plan_snapshot.json |
| 2026-08-06 | d5b563e | improve pick skill stability：place 连续下降执行闭环与 terminal 参数解析 |

## 修改记录

### 2026-06-10

#### 2026-06-10 · pick 执行 trace 与成功阈值参数（36d7f46）
- 改动：pick 执行阶段写出 pick_execution_trace.json，记录命令转换时序（pre-grasp/open、close_gripper、attach_obj）；lift_th 默认 0.0，仅在大于 0 时启用抬升高度成功检查；place 的 success_xy_margin 默认值引入，供 xybbox 成功判定使用。
- 原因：命令时序是判断 pick 各阶段是否按序完成的最佳证据。
- 文件：workflows/simbox/core/skills/pick.py, workflows/simbox/core/skills/place.py
- 验证：scene4 成功 checkpoint 实跑。

### 2026-07-09

#### 2026-07-09 · pick 运行期失败快照（6f99314）
- 改动：pick 执行失败时写出 pick_runtime_failure_snapshot.json，与候选快照分离，避免把执行失败误标为候选生成失败。
- 原因：后期 retry 中候选已能生成（success_found: true），旧"无候选"归因不再适用。
- 文件：workflows/simbox/core/skills/pick.py
- 验证：panda omron 稳定性实跑。

### 2026-08-03

#### 2026-08-03 · 抓取接触阈值与候选快照（98e558a）
- 改动：pick.py 引入 grasp_contact_threshold_n（默认 0.0）作为目标-手指接触力的确认阈值，GRIPPER_CLOSE dwell 后无接触返回 GRASP_CONTACT_MISSING 并禁止进入 ATTACH；候选评估后写 pick_plan_snapshot.json（含 success_found、pregrasp_success_count、world_collision_diagnostic）。
- 原因：接触确认先于 attach，区分候选生成与执行失败。
- 文件：workflows/simbox/core/skills/pick.py, workflows/simbox/core/planning/collision_scene_manager.py
- 验证：test/unit/test_collision_scene_manager.py。

### 2026-08-06

#### 2026-08-06 · place 连续下降执行闭环（d5b563e）
- 改动：agent/config.yaml 的 planning.pick_place 新增 place_continuous_descent: true、place_terminal_step_m: 0.01、place_terminal_tolerance_m: 0.005、place_terminal_max_path_length_ratio: 1.5、place_terminal_max_path_deviation_m: 0.01；curobo/controller.py 新增 _validate_continuous_place_plan()，对 TERMINAL_PLACE_DESCENT 的候选计划做 FK 检查（路径长度比、按 ds_ratio 抽样的最大步长、直线最大偏离），不达标即拒绝并计入 num_plan_failed；TERMINAL_PLACE_DESCENT 携带 contact_complete 时停止剩余下降并直接完成，新增 complete_terminal_place_on_contact()；place.py 新增 _resolve_terminal_step()/ _resolve_terminal_tolerance()，tolerance 超过 terminal step 时报错。
- 原因：快速下降计划可能绕远或单帧推进过大；5 mm 目标把完成容差当 plan 触发容差会导致无 plan 无完成，形成无限 hold。
- 文件：agent/config.yaml, workflows/simbox/core/controllers/curobo/controller.py, workflows/simbox/core/skills/place.py
- 验证：新增 test/unit/test_place_continuous_descent.py、test/unit/test_place_terminal_speed.py。

## 调参经验（来自 PANDA_OMRON_PLACE_ORIENTATION_FILTER_GUIDE，无提交日期）

#### filter_*_dir 的运行语义
- 字段选择 EE 本地轴（旋转矩阵列）：filter_x_dir 约束第 0 列、filter_y_dir 第 1 列、filter_z_dir 第 2 列；方向选择基座轴（行）：forward=第 0 行、leftward=第 1 行、upward=第 2 行，负方向用同元素反向比较。
- 正方向条件为 element >= cos(value)，value 是相对正方向的最大偏差；负方向条件为 element <= cos(value)，value 是相对正方向的最小夹角，相对负方向的实际容差是 180° - value。filter_z_dir: [downward, 150] 实际是向下 30° 圆锥。
- 旧配置 filter_x_dir: [backward, 110] + filter_y_dir: [downward, 120] + filter_z_dir: [forward, 70] 把工具轴推向水平：200,000 个随机旋转下工具轴与竖直向下的夹角中位数 90.1°、95 分位 134.6°。
- PandaOmron 的 ee_axis=z（panda_omron_virtual.yaml 中 fl_gripper_keypoints 的 tool_head/tool_tail 只在 z 上不同），控制垂直接近必须约束 filter_z_dir；filter_y_dir: downward 只会让手部侧轴朝下。
- 有效候选为 0 时回退到未过滤旋转的前 20 个并打印 Warning: No matrix satisfies constraints；运行验证必须检查该警告，不能只看 skill 是否继续执行。
- 姿态生成流程：scipy 生成 3000 个随机旋转矩阵 → filter 生成布尔 mask → 从有效集中随机抽 CUROBO_BATCH_SIZE=20 个 → 构造 pre-place/place 目标 → CuRobo 检查 IK、碰撞与可达性。

#### 推荐档位与调整顺序
- 三档保持工具 z 朝下、末端 x 朝前：宽松 x forward 60 / z downward 140（40°），平衡 x forward 45 / z downward 150（30°，每 3000 个候选约 47 个），严格 x forward 30 / z downward 160（20°，约 14 个）。统计使用 200,000 个随机旋转、固定种子 7。
- 调整顺序：先收紧 z（150 → 160），再收紧 x（45 → 30）；CuRobo 不可达时优先放宽水平 x（45 → 60），不要取消 z 向下约束；每次只改一个角度，不要同时约束 x、y、z 三个轴。
- 姿态质量与任务结果分开判定：place_success_check_snapshot.json 只能证明物体中心 XY 是否进入目标 bbox（success_mode: xybbox，margin 默认 0.015），不能证明放置姿态自然。

## pick 调试方法

#### 两阶段分离
- 候选生成在 simple_generate_manip_cmds()（pick.py:544 分发到 physics_schema 路径 pick.py:557 或 legacy 路径 pick.py:799），执行在 update()/is_subtask_done()/is_success()。
- 候选阶段只回答"是否存在无碰撞的 pre-grasp→grasp→post-grasp 路径"，执行阶段回答"机器人是否实际走到这些目标并抓住物体"。两个阶段产物不同，不能互相替代。

#### pick_plan_snapshot.json（候选生成阶段）
- 位置 output/local_navigation/skills/<robot>_pick_<object>_<ts>/，由 _write_debug_artifact()（pick.py:200）写出。
- physics_schema 路径（pick.py:616-633）字段：robot、object、lr_arm、collision_world_mode、plan_evaluation（含 result.to_dict() 的 feasible/pregrasp_success_count/selected_grasp_index/failure_code）、sample_debug（candidate_count、filtered_candidate_count、filter_pass_counts、sampled_indices、sampled_scores）、geometry_debug、pregrasp/grasp positions/orientations、terminal_plan_diagnostics、world_collision_diagnostic。
- legacy 路径（pick.py:1147-1161）另有 success_found、selected_candidate、candidate_results、candidate_rank_debug、manip_command_sequence。
- 判定：result.feasible=false 或 success_found=false 时，失败在候选生成阶段；常见 failure_code 有 NO_COLLISION_FREE_PLAN、no_complete_pick_path、no_grasp_candidates_after_sampling、TERMINAL_DISTANCE_EXCEEDED（pick.py:655-664）。
- 后期 retry 出现 success_found: true，说明旧"无候选"问题不活跃，失败转移到执行阶段。

#### pick_execution_trace.json（执行阶段）
- 每步由 _record_execution_step()（pick.py:1164）追加，按 execution_trace_write_stride（默认 250 步）周期性落盘（pick.py:1240），结束时 _flush_execution_trace()（pick.py:1291）写最终版；内存保留上限 execution_trace_max_steps（默认 500 步）。
- 每步字段：step、remaining_commands、current_command（phase 名或命令名）、target_diff_trans/target_diff_ori、command_age_steps、controller_gripper_state、controller_cmd_plan_active/cmd_idx/num_last_cmd/num_plan_failed、action_arm/action_gripper、actual_arm_position/actual_gripper_position、ee_translation/ee_orientation、object_world_translation/object_world_orientation。
- current_command 就是命令转换时序：SYNC_WORLD → TRANSIT_PREGRASP（open_gripper）→ TERMINAL_GRASP_APPROACH → GRIPPER_CLOSE（close_gripper）→ ATTACH → POST_GRASP_LIFT。
- 同一命令停留过久触发 watchdog：command_age_steps 达到 stalled_command_step_limit（默认 450）且 target_diff_trans 未收敛，置 failure_reason=pick_command_stalled（pick.py:1251-1263）。

#### 成功判定证据链
- is_success()（pick.py:1580）同时检查三块：接触（close_gripper 时接触对数量 >= 1，pick.py:1584-1585）、process_valid（关节/物体线速度 < 5，pick.py:1591-1593）、lift_th（仅当配置值 > 0，pick.py:1599-1601）。
- 证明 pick 成功的直接证据链：pick_execution_trace.json 中 object_world_translation 的 z 在 POST_GRASP_LIFT 期间抬升；close_gripper 之前没有 attach_obj；全程无 pick_runtime_failure_snapshot.json。pick_success_check_snapshot.json 只在失败时写（pick.py:1659-1665），其 lift 字段含 enabled/valid/delta/threshold/initial_position/current_position（pick.py:1639-1646）。
- GRIPPER_CLOSE 完成瞬间 is_subtask_done()（pick.py:1528-1547）调用 get_contact()（pick.py:1493）检查目标-手指接触：接触力超过 grasp_contact_threshold_n 才把 _grasp_contact_verified 置 true；否则 failure_reason=GRASP_CONTACT_MISSING、恢复世界并停留在当前命令，禁止进入 ATTACH。ATTACH 命令的 verify_grasp_contact 参数读取该标志（pick.py:760）。
- 不要把下游 place 失败误标为 pick 失败：place 报错而 pick 阶段 object z 已抬升、无运行期失败快照，应诊断为 place 侧问题。

#### lift_th 与 post_grasp_offset_min/max 的关系
- lift_th 只控制成功判定，不控制计划运动：lift_th > 0 时要求 (物体当前 z - 初始 z) > lift_th（pick.py:1595-1601）；默认 0.0 直接禁用抬升高度检查，不影响 post-grasp 运动目标。
- post_grasp_offset_min/max 控制计划抬升高度：post_offset 从区间均匀采样（pick.py:763-766），非 0 时在 grasp 位置 z 上叠加 post_offset 生成 POST_GRASP_LIFT 目标（pick.py:767-782）；区间两端都为 0 时不生成抬升命令。不要在代码中加隐藏上限覆盖这两个值。
- 调参注意：lift_th 过大在物体正常抬升后仍可能误报失败；post_grasp_offset 过小（如 0.05）抬升不足，物体未脱离支撑面，place 阶段会拖动物体。

## place 调试方法

#### 快照文件
- place_success_check_snapshot.json：仅在成功判定失败时由 _record_success_check_debug()（place.py:198-220）写出；字段 robot、skill、pick_object、place_object、place_prim_path、success_mode、success、failure_reasons、details。成功时只更新内存记录，不落盘。
- place_runtime_failure_snapshot.json：规划失败（num_plan_failed > th）时由 _write_runtime_failure_snapshot()（place.py:146-196）写出；字段 num_plan_failed、failure_reason、error_message、selected_place_target、constraints（position_constraint/place_direction/filter_*_dir/x_ratio_range/y_ratio_range/z_ratio_range/pre_place_z_offset/place_z_offset）、armbase/ee/pick 物体世界位姿、pick_collision_record。

#### success_mode 判定逻辑（place.py is_success() 862-1058）
- 3diou（默认，864-884）：计算 pick 物体与 place prim 两个 bbox 的 3D IoU，iou > th（th 为 is_success 参数，默认 0.0）；失败 reason iou_below_threshold。
- height（885-902）：pick 物体在机器人基座系下 z（T_o2r_trans[2]）< place_ee_trans[2] - 0.4；失败 reason height_not_below_threshold。
- xybbox（903-938）：pick 物体中心 XY（pick_obj.get_local_pose()[0][:2]）严格落在 place bbox XY 内缩 success_xy_margin（默认 0.015）后的 valid_min/valid_max 区间；失败 reason x_outside_place_bbox / y_outside_place_bbox。
- left（939-958）：pick_x < place_xy_min[0] - threshold（threshold 默认 0.03）；right（959-978）：pick_x > place_xy_max[0] + threshold。
- flower（979-1013）：物体中心落在 place bbox 内且 IoU > success_th，两条件同时成立；失败 reason center_outside_place_bbox / iou_below_threshold。
- cup（1014-1051）：pick bbox 的 z_min > place bbox 的 z_min + 0.05 且 IoU > success_th；失败 reason cup_z_below_shelf_threshold / iou_below_threshold。
- 未支持模式统一返回 False，reason unsupported_success_mode（1053-1057）。
- 任务名说"tray"但 place skill 目标是 sink 时，以快照中的 place_object/place_prim_path 为准诊断。

#### xybbox 失败与落点/沉降判断
- 诊断读 details：pick_xy、place_xy_min/max、margin、valid_xy_min/max、distance_to_valid_min/max、pick_local_pose、place_bbox_min/max。
- pick_xy 与 valid 边界差几个毫米、落在 margin 附近时，属于放置目标/沉降问题：物体落地滚动、bbox 采样抖动或 settle 未完成，不是 pick 失败。
- pick_xy 与 valid 区间相差较大时，检查 place 目标姿态与 pre_place_z_offset/place_z_offset（是否提前放低、TERMINAL_PLACE_DESCENT 是否被 contact_complete 提前截断）。
- position_constraint: object 不改变 filter_*_dir 语义：先记录 T_obj_ee，再按 R_base_ee_sampled × inverse(R_obj_ee) 推导物体目标姿态，过滤的仍是最终 EE 姿态。
