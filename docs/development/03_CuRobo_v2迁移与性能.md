# CuRobo v2 原生迁移与性能诊断

> 状态：截至 2026-08-14；原生 CuRobo v2 与 Warp 1.13.0 fork 的本轮修改已落地。retry24 选择了 Apple candidate13，但在 `post_grasp_lift` 发生非手指接触/附着滑移并 abort，未到 Orange Place，严格成功 marker 缺失；retry24 仍失败，当前问题尚未修复。
>
> 当前策略：使用 `InternDataAssets/curobov2` 中维护的原生 CuRobo v2 fork，适配 Isaac 自带 Warp 1.13.0；核心兼容层已取消，Isaac 源码保持不变。CuRobo fork 由用户单独维护；不要把 fork 内部修复回写成 Isaac 源码或用 Docker 版本 pin 替代。
>
> 基线：CuRobo v2 `0.8.0`，基线 commit 为 `4ea77366ca48ee453e7df139e39fa6532af49f3b`；当前工作树包含本轮 fork 修改。
>
> 运行时：当前开发容器使用 Isaac 自带 Warp `1.13.0`。当前方案没有把 Warp 固定为 `1.12.1`。

## 背景

旧运行时依赖 CuRobo v1 的批量语义和兼容 API，普通单目标查询也要经过 batch 路径，重复目标 padding、每次 world/attachment 变化都清空重建，性能与正确性都不满足要求。迁移目标是删除核心 v1 兼容层，改用 vendored native v2。

## 当前调用契约

契约全部落在 workflows/simbox/core/controllers/curobo/controller.py 与 workflows/simbox/core/planning/collision_scene_manager.py，代码位置如下：

- **native MotionPlanner（单目标）**：_init_native_planners()（curobo/controller.py:540）构造 self.planner = MotionPlanner（:597，batch_size=1、trajopt_seeds=12），update_world（:600）后 warmup（:602）。普通 transit、place、home 与单目标 pick 执行走 plan()（:1182）→ planner.plan_pose()（:1188）；joint 目标走 plan_joint_positions()（:1197）→ plan_cspace()（:1224）。
- **native BatchMotionPlanner（候选批量）**：use_batch 时构造 self.batch_planner = BatchMotionPlanner（:611，batch_size=CUROBO_BATCH_SIZE=20、trajopt_seeds=1，workflows/simbox/core/utils/constants.py:331）。plan_batch()（:1142）校验实际候选数在 1..batch_size（:1161-1165）后按实际维度构造状态与目标，→ batch_planner.plan_pose()（:1172）。抓取候选评估入口在 grasp_plan_evaluator.py:230（grasp-only）与 :250（pregrasp→terminal 两段，test_batch_forward_from_paths）。
- **GoalToolPose 5D 形状**：目标位姿 helper 把位姿 reshape 成 `[B,H,L,G,3/4]`——batch_size==1 时 (1,1,1,1,3)/(1,1,1,1,4)，否则 (batch_size,1,1,1,3/4)，再构造 GoalToolPose（tool_frames 取 planner.tool_frames[0]）。
- **attachment_manager**：attach_objects()（:1920）输入显式 attach_collision_prim_paths + native mesh：_native_attachment_geometry()（:1841）把全部 collider mesh 在首个 collider 当前帧合并为 __native_attached_object__，经 self.planner.attachment_manager.attach()（:1945，link_name=attached_object、SphereFitType.VOXEL、world_objects_pose_offset、disable_obstacle_names=paths）。detach_obj()（:2084）→ attachment_manager.detach()（:2086）；reset 清理走 _clear_attached_object_state()（:976）；attach 后校验走 has_attached_collision_spheres()（:2090）。legacy 兼容入口 attach_obj()（:2021）与 test_attached_forward_from_joint_positions()（:1961，评估期瞬时 attach/detach）仍存在，物理路径不经过它们。
- **world 更新**：_update_world_if_changed()（:858）按 _make_world_update_signature 签名变化触发 planner.update_world()/batch_planner.update_world()（:867/:869）；动态障碍位姿同步走 collision_scene_manager.sync_dynamic_poses()（collision_scene_manager.py:1355）→ scene_collision_checker.update_obstacle_pose()（:1349），覆盖 single 与 batch 两个 planner（_native_planners :1107）。
- **legacy_stage_scan**：_legacy_update()（:935）的 get_obstacles_from_stage + ignore_substring 只是显式旧碰撞 world 扫描模式，不是 v1 planner 或兼容 API。

## 时间线

| 时间 | 内容 |
|------|------|
| 迁移阶段 | 删除核心 CuRobo v1 兼容层，vendored native v2 落地到 `InternDataAssets/curobov2/`。 |
| 迁移阶段 | 普通查询改走 native `MotionPlanner`，抓取候选批量评估改走 native `BatchMotionPlanner`。 |
| 迁移阶段 | attachment 改走 native `attachment_manager` 与显式 `attach_collision_prim_paths`。 |
| 迁移阶段 | 完成集成侧的批量语义、world 更新、动态障碍同步和候选结果处理优化。 |
| 2026-08-13 | 在 CuRobo fork 中适配当前 Isaac Warp 1.13.0，并补齐 USD、SceneData、Kinematics、TrajOpt 和 batch pose warmup 相关修复。 |
| 2026-08-14 | 完成真实 attach/runtime deferred post-grasp 校验、terminal batch 的 single-candidate fallback、Place native start-collision/`place_plan_failure_snapshot` 诊断，以及 region `keep_upright` 和刚体速度 reset；retry21 已推进到 orange Place preplace。 |
| 2026-08-14 | physics-schema 侧完成 Pick 物理抓取排序与共享 native `GraspPlanEvaluator` 的接入，以及 Place native single fallback 的 mask/预算/terminal 几何校验修复；retry24 选中 Apple candidate13，但在 `post_grasp_lift` 因非手指接触/附着滑移 abort，未到 Orange Place，严格运行仍失败。 |

## 修改记录

### 集成侧迁移与性能修改

以下是应用层/控制器侧已完成的 native v2 迁移修改，保留作为当前调用路径的背景记录。七条修复按"慢点（原行为）| 改动 | 收益"合并如下：

| 慢点（原行为） | 改动 | 收益 |
|----------------|------|------|
| 普通单目标查询也走 v1 批量语义/兼容 API | 移除 v1 batch 语义扩散，single 查询用 controller.planner 持有的 native MotionPlanner（curobo/controller.py:597），候选评估才用 batch_planner 持有的 native BatchMotionPlanner（:611） | 单目标路径不再承担 batch 开销，plan()/plan_batch() 职责分离 |
| 重复目标 padding，复制目标凑固定容量 | batch 传递实际候选数（plan_batch :1161-1165 按 1..batch_size 校验并构造），不再复制目标凑容量 | 去掉无效候选的 IK/优化浪费，结果按真实候选索引对齐 |
| 每次 world/attachment 变化都清空重建 | 改 native update_world()（_update_world_if_changed :858-872，按 world 签名变化触发）原地更新 SceneData；attachment 用 attachment_manager 的 attach/detach（:1945/:2086） | 不再重建 planner 与 CUDA graph，world 更新只在签名变化时发生 |
| 移动参考系下每帧刷新全部 obstacle pose | activate_collision_world_mode 不再每步刷新（:905-909 注释），改由计划前 _refresh_reference_world_for_planning()（:944）→ refresh_controller_reference_world()（collision_scene_manager.py:790）一次同步 | 每帧全量（约 230 物体）pose 更新降为每次 CuRobo 查询前一次 |
| 动态障碍位姿不同步，规划基于旧世界 | sync_dynamic_poses()（collision_scene_manager.py:1355，默认 5 步间隔）把障碍位姿同步到 single 与 candidate batch 两个 planner 的 scene_collision_checker（:1349） | 动态障碍规划一致，且同步按步数节流 |
| native batch graph seed 按最大 batch reshape 实际请求状态 | graph seed 改用实际 batch 维度 reshape 请求状态 | CUDA graph 输入尺寸与实际批一致，非满 batch 下正确 |
| 失败候选无路径被伪成功继续执行 | 按 native 结果 success [B,S] 逐候选掩码（pick.py:33 _candidate_success_mask、grasp_plan_evaluator.py:254-283），trajectory 为 None 明确视为无路径 | 失败候选不再进入执行或影响路径选择 |

相关路径包括 `workflows/simbox/core/controllers/curobo/controller.py` 和 `workflows/simbox/core/planning/collision_scene_manager.py`。这些修改不改变当前“Isaac 源码保持不变、CuRobo fork 单独维护”的边界。

### 本轮 CuRobo fork 修改（2026-08-13）

本轮修改全部位于 `InternDataAssets/curobov2`，基线为 CuRobo v2 commit `4ea77366ca48ee453e7df139e39fa6532af49f3b`：

1. **使用 public Warp API**：将 CuRobo 代码路径中的 `wp.torch.device_from_torch` 迁移为 `wp.device_from_torch`，适配 Isaac 自带 Warp 1.13.0 已提供的顶层 public API。
2. **USD polygon face fan triangulation**：USD 允许 `faceVertexCounts` 表示多边形面；USD parser 现在按每个面的顶点序列做 face fan 三角化，将 `(v0, v1, v2, ...)` 转成 `(v0, vi, vi+1)` 三角形后交给 CuRobo 碰撞网格。
3. **SceneData 按 name dispatch**：`SceneData` 的 pose/update 和 enable/disable 操作先按 name 查询 cuboid、mesh、voxel 所属集合，再 dispatch 到对应数据对象，避免依次调用并依赖异常来探测类型。
4. **Kinematics singleton `[dof,1]` 归一化**：识别单例 `JointState` 被展开成的 `[dof,1]` 输入，并转换到 fused kernel 使用的规范 batch/horizon/dof 布局，避免把 singleton 误解释成普通二维 batch。
5. **TrajOptSolver 暴露 `attachment_manager`**：增加对 shared solver core 所持 attachment manager 的 property，供 native batch/planner 路径使用统一的 attachment 管理器。
6. **BatchMotionPlanner pose warmup**：warmup 阶段先走完整的 `GoalToolPose`/batch IK pose 路径，避免第一次真实抓取查询因 pose-to-IK kernel 尚未初始化而出现全 0 IK 结果。
7. **保护 CUDA graph goal 类型切换**：当 CUDA graph reset 不可用时，不再在 cspace goal 与 pose goal 之间切换；仅在未启用 CUDA graph 或明确支持 graph reset 时保留 cspace warmup，避免触发不可用的 graph reset。

### 本轮 native v2 闭环修复（2026-08-14）

这些修改仍属于应用侧与 CuRobo fork 的既定边界：使用原生 CuRobo v2；不恢复 v1 兼容层；不改 Isaac 源码。

- **post-grasp native attached validation 改为真实 attach/runtime deferred**：候选生成阶段只保留 pre-grasp 与 terminal-grasp 的 native batch 结果；不再对每个候选做仍处于未闭合夹爪/未更新物体姿态的合成 attached 查询。`post_grasp_validation` 现在明确记录 `mode=deferred_runtime_attach`、`success=null`；权威检查延后到执行期 `POST_GRASP_LIFT`，在真实 `CollisionSceneManager.attach_target` 完成后，使用实际附着几何、物体姿态和当前关节状态查询 native planner。
- **native terminal batch 失败的 single-candidate fallback**：`test_batch_forward_from_paths` 返回全失败、`success` 缺失或 mask 不可用时，只对 pre-grasp 已成功且有路径的候选逐一调用 native single planner；仅保留返回有效路径且 Cartesian path ratio `<=1.5`、最大偏差 `<=0.01 m` 的候选。正常 batch 有效时仍走 batch 快路径，不以 padding 或伪成功替代结果。
- **Place 失败诊断**：physics-schema Place 耗尽所有 pre-place 候选后，调用 native `diagnose_native_start_collision()`，从 live arm state 做一次 FK，并用与 planner 相同的 attached spheres/scene checker 记录总碰撞代价、attached-object 碰撞代价和碰撞 sphere 明细；该诊断不关闭障碍物，也不改变后续状态。同时写出 `place_plan_failure_snapshot.json`，保存候选诊断、Place 约束、bbox/目标姿态、运行时附着状态和 `native_start_collision`；写诊断失败不得遮蔽原始 `NO_COLLISION_FREE_PREPLACE_PLAN`。
- **region `keep_upright` 与刚体速度 reset**：`A_on_B_region_sampler` 在随机摆放时保留对象 yaw 并清除历史 roll/pitch；`sampling.keep_upright` 同步传入随机 `_set_regions` 和确定性 `reset_fixed_rigid_objects` 的 region pose。随机 region 摆放及失败重试的固定刚体 reset 都将 linear/angular velocity 清零，避免上一次失败留下的速度把固定台面物体误判为动态障碍并耗尽 replan budget。
- **physics-schema 抓取排序与 Place single fallback 修复**：将 Pick 已有的物理抓取排序注入共享 native `GraspPlanEvaluator`；候选选择只在 pre-grasp、terminal-grasp 和路径几何均有效的交集内进行。Place native single-candidate fallback 增加 native mask 归一化、总查询预算诊断，并按 native pre-path endpoint 做 terminal 几何校验。

## retry17–24 关键验证发现

以下记录以 native Warp 1.13 fork 的 strict scene run 为准；`de_time_profile` 与 Docker Isaac log 的失败原因保持区分。

| Run | 关键发现 | 首个失败/结论 |
|-----|----------|---------------|
| retry17 | native batch 与 pre-grasp batch 均为 `success_count=20`，pre+final 共有 16 个成功候选；但旧的 post-grasp 预先 attached 检查对候选全部返回 false。 | Pick 失败：`NO_COLLISION_FREE_POST_GRASP_PLAN`。说明未闭合夹爪、未完成真实 attach 时的检查不能作为 post-grasp 权威门槛。证据：`output/curobo_v2_native_validation_warp113_fork_retry17/de_time_profile_20260813_213516_185803.log`。 |
| retry18 | batch/pre-grasp 仍可成功（20/20，pre+final 19 个），但 `test_attached_forward` 加 support-contact mask 后仍对候选返回 0；问题仍在合成 attached 状态与真实执行状态不一致。 | Pick 仍失败：`NO_COLLISION_FREE_POST_GRASP_PLAN`。证据：`output/curobo_v2_native_validation_warp113_fork_retry18/de_time_profile_20260813_220133_657388.log`。 |
| retry19 | 改为 runtime deferred 后，apple 有 17 个 joint-success 候选并进入真实 `transit_pregrasp`；步骤 240、250 的 dynamic-obstacle change 触发两次重规划，步骤 310 再次变化后 abort，尚未 attach。 | 首个执行失败是 `transit_pregrasp` 的动态障碍重规划预算耗尽，不是 CuRobo 初始化或候选 batch 失败。证据：`output/curobo_v2_native_validation_warp113_fork_retry19/de_time_profile_20260813_222832_615098.log`。 |
| retry20 | deferred validation 下 apple 已完成 Pick、native attach、post-grasp lift、carry，并完成 tray Place/restore；随后 orange 的 terminal batch 返回 `success_count=-1`。 | Orange Pick 失败：`NO_JOINT_GRASP_PLAN`。该 run 隔离出 native terminal batch 全失败时需要 single-candidate fallback 的异常路径。证据：`output/curobo_v2_native_validation_warp113_fork_retry20/de_time_profile_20260813_225505_927662.log`。 |
| retry21 | apple 路径继续通过；orange terminal batch 仍返回 `success_count=-1`，随后 single-candidate native fallback 找到可用路径（Pick snapshot 记录 `joint_success_count=12`、selected candidate `2`）。真实执行已通过 orange transit/pre-grasp、terminal grasp、close、`attached=native`、post-grasp lift 和 carry_home。 | **首个失败是 orange Place 的 pre-place planning**：`place_orange_0_id9009` 返回 `NO_COLLISION_FREE_PREPLACE_PLAN`；不是 orange Pick、attach、lift 或 carry 失败。证据：`output/curobo_v2_native_validation_warp113_fork_retry21/de_time_profile_20260813_233856_950804.log`、`output/docker_runtime/simbox_task-2566180/docker_runtime.isaac.log`、`output/local_navigation/skills/panda_omron_pick_orange_0_id9009_1786636281728/pick_plan_snapshot.json` 与 `pick_execution_trace.json`。 |
| retry22 | Apple 的 pre-place transit 已成功；进入 `terminal_place_descent` 后，native planner 三次均返回 `success_count=1`，但连续放置几何校验均拒绝路径：`path_ratio/max_deviation` 分别约为 `2.0500/0.041365`、`3.5538/0.121275`、`1.2431/0.018824`，阈值为 `1.5/0.010000`。前两次触发 replan，第三次仍不满足约束后 abort。 | **首个失败是 Apple 的 `terminal_place_descent`**，不是 native planner 返回失败：`continuous-place-plan valid=False`，随后 `plan_with_render returned 0`。retry22 wrapper 结果为 `application_failed`、exit code `20`，没有 `Task is successful, mode=plan_with_render` 成功 marker。证据：`output/curobo_v2_native_validation_warp113_fork_retry22/de_time_profile_20260814_002251_371820.log`、`output/docker_runtime/simbox_task-2587654/docker_runtime.isaac.log`、`output/docker_runtime/simbox_task-2587654/docker_runtime.json`。 |
| retry23 | Apple Place 的 native batch pre+terminal 选出 candidate `1`；首次 `terminal_place_descent` 的 native planner 查询有结果，但连续放置几何校验因 `max_deviation=0.012081` 超过 `0.010000` 被拒绝，安全重规划后得到 `valid=True`、`path_ratio=1.0211`、`max_deviation=0.003178`，并完成 Apple Place。随后 Orange 已完成 Pick、native attach、post-grasp lift 和 carry_home。 | Orange Place 的 pre-place 20 个候选均无解，最终 `failure=NO_COLLISION_FREE_PREPLACE_PLAN`；`NativeCollisionDebug` 显示 `collision_cost_sum=0`、`collision_spheres=0`、`attached_cost_max=0`。retry23 严格验证仍失败，当前 Orange place preplace 问题尚未修复。证据：`output/curobo_v2_native_validation_warp113_fork_retry23/de_time_profile_20260814_005633_194685.log`、`output/docker_runtime/simbox_task-2605771/docker_runtime.isaac.log`、`output/docker_runtime/simbox_task-2605771/docker_runtime.json`、`output/local_navigation/skills/panda_omron_place_orange_0_id9009_to_metal_tray_0_id9016_1786640928653/place_plan_failure_snapshot.json`。 |
| retry24 | physics-schema 的物理抓取排序选择 Apple candidate `13`；Apple 执行到 `post_grasp_lift` 时发生非手指接触/附着滑移并 abort，未进入 Orange Place。 | **严格运行仍失败**：未到 Orange Place，严格成功 marker 缺失，未出现 `Task is successful, mode=plan_with_render`；不能宣称任务成功。 |

retry21 只证明了橙子 Pick/attach/lift/carry 链路已越过此前的 batch 与合成 attached 检查问题；Place 仍未成功，不能据此宣称完整 task 成功。

retry22 进一步证明：native planner 的 `success_count=1` 只表示生成了候选轨迹，不等于该轨迹可以执行。`continuous-place-plan` 的 `path_ratio` 和 `max_deviation` 是几何安全门槛，任一超阈值都必须拒绝该路径并进入重规划/失败路径；**不能绕过、跳过或仅因 native planner 返回 success 而放宽这项几何安全校验**。

retry23 证明该几何安全门槛可以在安全重规划后正常放行：Apple 首次路径的 `max_deviation=0.012081` 被拒绝，重规划得到 `0.003178` 后才允许执行。与此同时，Orange 的 preplace 失败不是 live native 起始碰撞（诊断 cost 为 0），而是 20 个 preplace 候选均未产生可用解；问题仍待修复，不能用 `NativeCollisionDebug cost=0` 或其他成功阶段结果替代 Place 规划成功。

retry24 证明新的 physics-schema 抓取排序最终选择了 Apple candidate13，但这不等于 Pick 已完成：Apple 在 `post_grasp_lift` 发生非手指接触/附着滑移并 abort，运行没有到达 Orange Place，且严格成功 marker 缺失。因此 retry24 仍是失败运行，不能据此宣称任务成功。

## 本轮验证命令与 retry23/24 结果

retry17–24 使用统一的 SceneBox 单任务验证 wrapper（由 `up_simbox_isaac.sh` 启动 Isaac），命令契约如下：

```bash
GPU_ID=0 \
TASK_CONFIG=InternDataAssets/assets/custom/scene_8/01_kitchen/assets/basic/kitchen_apple_orange_to_tray/simbox_task.yaml \
RUN_NAME=curobo_v2_native_validation_warp113_fork_retry22 \
LAUNCH_TEMPLATE=output/simbox_plan_with_render/de_config.yaml \
RANDOM_NUM=1 RANDOM_SEED=0 \
bash scripts/docker/up_simbox_isaac.sh
```

运行结束后按 wrapper 的严格判据检查 Isaac log；不能只看容器退出、视频或缺少 traceback：

```bash
rg -n 'Task is successful, mode=plan_with_render|\[LmdbLogger\] Episode failed|plan_with_render returned|NO_' \
  output/docker_runtime/<run-dir>/docker_runtime.isaac.log
```

retry21 的运行日志为 `output/docker_runtime/simbox_task-2566180/docker_runtime.isaac.log`；retry22 的日志为 `output/docker_runtime/simbox_task-2587654/docker_runtime.isaac.log`。retry23 已完成，profile 为 `output/curobo_v2_native_validation_warp113_fork_retry23/de_time_profile_20260814_005633_194685.log`，日志为 `output/docker_runtime/simbox_task-2605771/docker_runtime.isaac.log`，metadata 为 `output/docker_runtime/simbox_task-2605771/docker_runtime.json`。retry23 的 Apple Place 在几何安全重规划后通过，但 Orange place preplace 的 20 个候选均无解并以 `NO_COLLISION_FREE_PREPLACE_PLAN` 结束。retry24 进一步在 Apple `post_grasp_lift` 因非手指接触/附着滑移 abort，未到 Orange Place，严格成功 marker 仍缺失；严格验证仍失败，当前问题尚未修复。

## 旧版本描述的更正

旧文档曾把“镜像固定 Warp 1.12.1”和“不修改 vendored CuRobo 源码”写成当前方案。这是本轮 fork 适配前的历史诊断，不再是当前事实，也不应作为当前开发环境或维护边界。

此前确实观察到 Isaac Sim 6.0.1 某基础环境中的 Warp 1.16.0 缺少 `warp.torch` shim（例如 `warp.torch.device_from_torch`），这解释了为什么需要适配 CuRobo fork；它不意味着当前应固定 Warp 1.12.1。当前开发容器的实际基准是 Isaac 自带 Warp 1.13.0，CuRobo 通过 fork 内的 public API 修改与之配合。

## 验证证据

### 已完成的证据

- **Warp/Kit 加载**：Warp 1.13.0 已在 Kit 中成功加载；在 public API 修改后的开发容器路径中，CuRobo native planner 可以完成初始化。相关本轮运行记录位于 `output/curobo_v2_native_validation_warp113_fork_retry5/de_time_profile_20260813_154550_972492.log`。
- **native v2 基础 batch smoke**：20 个目标全部成功。记录中 `test_batch_forward` 和 `test_batch_forward_from_pregrasp` 均为 `success_count=20`，并列出候选索引 `0..19`；证据位于 `output/scene8_kitchen_apple_orange_to_tray_timing/de_time_profile_20260811_184445_807239.log`。

### 严格 scene task 的失败链路

严格 scene task 的修复前 retry5 首个失败不是 Warp 加载失败，而是：

1. `BatchMotionPlanner.plan_pose` 的 batch IK 结果全为 0，因而返回 `None`；
2. 上层抓取规划没有得到联合关节抓取计划，最终报告 `NO_JOINT_GRASP_PLAN`。

随后对 `BatchMotionPlanner` 增加 pose warmup 的第一次验证又暴露出 `CUDA graph reset is not available`。因此才追加了上述 goal-type 切换保护，避免在 reset 不可用的部署中先做 cspace warmup、再切回 pose goal。这个链路说明 warmup 和 graph reset 保护是两个连续的修复点，不能只记录其中一个。

当前 Warp 1.13.0 fork 的 retry5 日志仍以 `NO_JOINT_GRASP_PLAN` 结束，且没有严格成功 marker；它只能证明环境加载、native planner 初始化和失败位置，不能证明完整 Pick/Place task 已成功。

此前已完成的代码级 AST/diff 检查、旧运行时兼容入口扫描，以及 CuRobo v2/Isaac Torch 导入、CUDA `GoalToolPose`/`JointState` 构造和 `success [B,S]`/失败候选路径 smoke check 仍然有效。

## 待继续验证

### 尚未验证项

- **完整 attach/detach task validation**：retry24 的 Apple candidate13 在 `post_grasp_lift` 因非手指接触/附着滑移 abort，未到 Orange Place，严格成功 marker 未出现；完整 Pick/Place 闭环仍未验证成功，当前问题尚未修复。
- **CUDA graph / 显存 / wall-clock 基准与实测加速倍数**：相对 v1 的加速倍数没有实测数据。
- **pose warmup 与 graph-reset 保护的组合**：二者只分别验证过失败链路，组合后在真实部署的行为待验证。
- **非满 batch 的 graph seed 行为**：smoke 只有 20 目标（满 batch）全成功记录，中间规模（如 3~19 个候选）没有验证记录。
- **双 planner world 一致性**：refresh_after_task_reset()（collision_scene_manager.py:1451）中 batch_planner.update_world（:1527）仅在 batch_planner 存在时执行，该 reset 路径的 batch world 一致性待验证。
- **GoalToolPose 5D 形状覆盖**：place/home/transit 单目标与候选 batch 的形状只做过代码级核对，无运行级验证记录。
- **多 controller 隔离**：left/right 双 controller 各自持有 attachment_manager 与 planner，双臂并发的 attachment 隔离没有验证记录。

### 最终判定标准

retry24 的 strict scene run 已结束且失败；它使用最新 CuRobo fork 及本次 physics-schema 修复（包括 Pick 物理抓取排序注入共享 native `GraspPlanEvaluator`、有效候选集合选择，以及 Place native single fallback 的 mask 归一化、总查询预算诊断和基于 native pre-path endpoint 的 terminal 几何校验）。Apple 选择 candidate13 后在 `post_grasp_lift` 因非手指接触/附着滑移 abort，未到 Orange Place，严格成功 marker 缺失；不能用候选选择成功或前序阶段结果替代完整任务成功结论。

最终判定至少需要同时满足：

- Kit 中成功加载 Warp 1.13.0，且 CuRobo 来源和 commit 仍为预期 fork/base commit；
- native v2 基础 batch smoke 的 20 个目标保持成功；
- strict task 日志明确包含 `Task is successful, mode=plan_with_render`；
- 同一严格运行没有 `[LmdbLogger] Episode failed`，并结合 per-task log 与 skill snapshot 确认 Pick/Place/attachment 执行链路没有失败。

在 retry24 日志没有上述严格成功 marker，且运行在 Apple `post_grasp_lift` 因非手指接触/附着滑移 abort、未到 Orange Place；不宣称完整 scene task 或完整 Pick/Place 闭环已成功。retry24 仍失败，几何安全校验必须保留，当前问题尚未修复。

## 本阶段记录：retry25 与目标刚体稳定性（2026-08-14）

本阶段只补充验证日志；Isaac 源码、CuRobo v2 原生调用边界和现有历史记录均保持不变。

### retry25 strict scene 验证

执行命令：

```bash
TASK_CONFIG=InternDataAssets/assets/custom/scene_8/01_kitchen/assets/basic/kitchen_apple_orange_to_tray/simbox_task.yaml \
LAUNCH_TEMPLATE=output/simbox_plan_with_render/de_config.yaml \
RANDOM_NUM=1 RANDOM_SEED=0 \
RUN_NAME=curobo_v2_native_validation_warp113_fork_retry25 \
CUROBO_DEBUG_WORLD_COLLISION=1 \
INTERNDATA_CONTAINER_UID=$(id -u) INTERNDATA_CONTAINER_GID=$(id -g) \
scripts/docker/up_simbox_isaac.sh
```

结果：严格成功 marker 缺失，首个失败发生在 `transit_pregrasp`。Apple `apple_0_id9008` 被 `CollisionSceneManager` 识别为活动目标刚体，其动态位姿在 step 250、310、325 发生变化；step 250 触发 `hold_and_replan` `1/2`，step 310 触发 `hold_and_replan` `2/2`，step 325 再次变化后以 `dynamic_obstacle_changed` abort，随后日志记录 `plan_with_render returned 0`。该阶段没有接触、滑移或关节跟踪异常，不能把失败归因于 CuRobo batch IK 或 native planner 初始化。

证据：

- profile：`output/curobo_v2_native_validation_warp113_fork_retry25/de_time_profile_20260814_020141_638538.log:53-72`
- Isaac log：`output/docker_runtime/simbox_task-2640901/docker_runtime.isaac.log:963-970`
- marker 检查未找到 `Task is successful, mode=plan_with_render`，也未找到 `[LmdbLogger] Episode failed`；可见的终止记录是 `plan_with_render returned 0`。

### 开发容器最小物理复现

使用开发容器 wrapper 启停环境，并在容器内运行临时 Isaac physics probe（该 probe 为现场 inline 调试代码，未写入仓库）：创建/加载苹果刚体，设置为直立初始姿态，逐物理步记录 world pose 与 linear/angular velocity，同时在 reset 后清零速度：

```bash
scripts/docker/isaac_dev.sh start --gpu 0
scripts/docker/isaac_dev.sh status
# 在上一步确认的 /workspace 开发容器内执行上述临时 physics probe
scripts/docker/isaac_dev.sh stop
```

结果是苹果约经过 25 个物理步就翻倒到与 retry25 相同的侧倒姿态；在 reset/摆放后清零 linear/angular velocity 仍不能阻止翻倒。因此，速度清零只能清理历史动量，不能解决当前动态刚体的姿态/接触稳定性问题；`keep_upright` 的初始姿态处理也不能等同于把动态物体固定为 kinematic 或冻结姿态。后续应继续检查苹果资产的刚体根节点、碰撞几何、质心/惯性和支撑接触，而不是继续单独增加 replan 次数。

### 本阶段代码诊断与 Pick 重规划契约

- `CollisionSceneManager._sync_record_poses()`（`workflows/simbox/core/planning/collision_scene_manager.py:1363-1400`）现在在动态位姿超过阈值时记录 `translation_delta_m`、`rotation_delta_deg`、world position 和实时 `linear_velocity`，并继续把最新 pose 同步到 native scene checker；这使 retry25 的位姿变化可与速度/姿态诊断关联，而不是只看到一个无上下文的动态障碍事件。
- Pick 在 native 初始候选规划前保存活动目标 world pose（`workflows/simbox/core/skills/pick.py:740-743`），候选生成完成、第一次执行前再次按当前刚体 pose 增量 retarget 全部待执行目标（`:988-993`）。`_retarget_pick_commands_to_current_object()`（`:213-276`）通过 object world-frame delta 更新 arm-base 下的后续目标位置/姿态。
- 安全重规划回调 `replan_after_safety()`（`workflows/simbox/core/skills/pick.py:278-312`）复用同一 retarget 逻辑；当活动目标变化导致重新规划时，旧 `TERMINAL_GRASP_APPROACH` 的 `preplanned_joint_path`、`path_length_ratio` 和 `path_max_deviation_m` 会被删除，避免从旧目标姿态继续执行失效 terminal path。执行监督器也会先清除缓存的 `preplanned_joint_path`（`workflows/simbox/core/execution/execution_supervisor.py:61-65`）。

### 验证结论

retry25 已证明当前安全监控能够发现活动目标的真实刚体位姿变化，并在重规划预算耗尽后安全终止；开发容器最小物理复现进一步证明“摆放后清零速度”不足以稳定苹果。当前严格 task 仍未满足 `Task is successful, mode=plan_with_render`，下一阶段应优先修复苹果动态刚体的物理稳定性/资产姿态，再重新评估动态位姿同步和 Pick terminal path 的重规划效果。

## 本阶段记录：retry26 与实体级动态障碍审计（2026-08-14）

### retry26 strict scene 验证

retry26 的严格成功 marker 缺失，运行没有到达 attach 或 Place。`transit_pregrasp` 已完成；进入 `terminal_grasp_approach` 后，Apple `apple_0_id9008` 的动态位姿在 step 385 和 step 400 被检测为变化，分别触发 `hold_and_replan` `1/2` 和 `2/2`。每次重规划前，Pick 的 `replan_after_safety()` 都执行了活动目标 retarget，并使当前 terminal path 失效后重新生成；step 415 再次检测到同一 Apple 动态变化，因重规划预算耗尽以 `dynamic_obstacle_changed` abort。该结论只描述 retry26，不能推断尚未完成的 retry27 结果。

两次 terminal 重规划的 `SafetyDebug` 测量中，`unexpected_contact_n=0.0`、`unexpected_object_contact_n=0.0`、`attached_slip_translation_m=0.0`、`attached_slip_rotation_deg=0.0`，关节/末端跟踪误差也保持在很小范围；因此 retry26 的首要失败原因是活动 Apple 刚体持续发生动态位姿变化并耗尽安全重规划预算，不是接触、附着滑移、跟踪失败或 CuRobo native batch 规划失败。日志最终记录 `plan_with_render returned 0`。

证据：

- profile：`output/curobo_v2_native_validation_warp113_fork_retry26/de_time_profile_20260814_112440_720577.log:437-465`
- `terminal_grasp_approach` 的首次/第二次 retarget 与重规划：`:441-445`、`:451-456`
- 最终 abort 与严格运行失败：`:461-465`

### CollisionSceneManager 的实体级动态变化判定

本轮将 `CollisionSceneManager._sync_record_poses()` 的“是否发生显著动态变化”判定从逐 collider 改为按实体的 `tracking_prim_path` 汇总：每个 `CollisionObjectRecord` 只对 tracking prim 保存上一帧矩阵，并据此计算实体级 `translation_delta_m` 与 `rotation_delta_deg`；只有实体级位姿超过动态重规划阈值时，才向执行安全层报告该实体发生变化。这样可以避免一个固定刚体下多个子几何 collider 的相同刚体运动被重复报告，也避免把固定刚体子几何体的局部变换误报成动态障碍变化，减少日志和阈值判定开销。

该优化没有牺牲 CuRobo 碰撞世界的精确同步：`_sync_record_poses()` 仍逐个遍历 `collision_prim_paths`，按每个 collider 的当前 world pose 更新 single/batch native planner 的 `scene_collision_checker`；实体级 tracking 只负责变化判定和对外返回实体名，逐 collider 姿态同步仍然保留。对应实现位置为 `workflows/simbox/core/planning/collision_scene_manager.py:1357-1419`，动态同步入口为 `:1421-1439`。

当前记录的下一步仍是处理 Apple 动态刚体自身的物理稳定性，再重新验证实体级变化判定、Pick retarget 与 terminal grasp 闭环；本节不记录或预判 retry27。

## 本阶段记录：retry27 post_grasp_lift 安全门定位与修复设计（2026-08-14）

### retry27 strict scene 的首个新失败

retry27 已经完成 Apple 的 native attach，说明前序候选选择、抓取接近、夹爪闭合和原生附着链路已通过；但在 `post_grasp_lift` 阶段仍未达到严格成功 marker。安全日志中的 `robot contact detail` 显示 `allowed=None`，`unexpected_contact_n` 约为 7--8 N；同一时段 `attached_slip_translation_m` 约为 `0.0188 m`、`attached_slip_rotation_deg` 约为 `36 deg`，而 `dynamic_obstacle_changed=false`。因此本次首要失败不是动态障碍重规划，也不是 native attach 未发生，而是附着目标与机器人之间的预期接触被安全门当成异常接触处理。

### 根因与代码修复

根因有两层：Pick 生成的两个 `POST_GRASP_LIFT` 命令没有显式声明 `allow_target_robot_contact`；同时，`workflows/simbox_dual_workflow.py` 的安全门只在 `DETACH_AND_SETTLE` 且存在 `pending_detach` 时排除目标实体，导致即使命令设置了该开关，在附着后的抬升阶段也不会真正生效。

本轮代码已将 Pick 的两个 `POST_GRASP_LIFT` 命令显式设置为 `allow_target_robot_contact=True`，并把安全门收窄为：只有命令显式允许目标机器人接触，且目标处于 `ATTACHED`/`PLACEMENT_CONTACT`，或目标处于 `pending_detach` 状态时，才排除该目标的预期接触；其他机器人--环境碰撞仍按异常处理。该设计只放行当前附着/放置流程声明的目标接触，不会把一般环境碰撞变成可忽略事件。

### 验证状态

上述修复后的严格验证尚未运行 retry28。当前不能宣称完整 scene task 成功；最终判定仍必须看到 `Task is successful, mode=plan_with_render`，并确认同一运行没有 `[LmdbLogger] Episode failed`。

## 本阶段记录：retry28 与后续安全/Place 修复（2026-08-14）

### retry28 strict scene 验证

retry28 使用 `allow_target_robot_contact` 修复后，Apple 已依次成功完成 native attach、`post_grasp_lift` 和 `carry_home`。该阶段期间未再出现 retry27 中的 `unexpected_contact` 重规划或 abort，说明附着目标与夹爪之间的预期接触不再被错误判定为异常机器人接触。

严格任务随后在 Place 规划入口失败：`workflows/simbox/core/skills/place.py` 的 `sample_gripper_place_traj` 使用了未定义的 `pick_place_cfg`，抛出 `NameError`。因此本次运行最终缺少 `Task is successful, mode=plan_with_render` 严格成功标记，不能宣称完整 scene task 成功。

### retry28 后续代码修复

- 在 `sample_gripper_place_traj` 函数开头初始化 `pick_place_cfg`，修复 Place 规划入口的 `NameError`。
- 安全测量初始化 `record=None`；目标接触豁免绑定到附着记录的 owner，并将 `pending_detach` 豁免限制在 `DETACH_AND_SETTLE` 阶段，避免无对象命令或其它 controller/phase 获得错误的目标接触放行。
- `CollisionSceneManager` 使用 SVD 极分解从 affine 3x3 矩阵提取纯旋转，恢复 attached rotation slip 的 hard trigger，并新增对应单元测试。

上述后续修复尚未通过严格运行验证；retry29 尚未验证，当前仍不能宣称完整 scene task 成功。

## 本阶段记录：retry29 Place 原生候选执行一致性修复（2026-08-14）

### retry29 暴露的问题

retry29 已经完成 Pick 的 native attach、`post_grasp_lift` 和 `carry_home`，Place 候选评估也能选出 native v2 候选；但进入实际 `TRANSIT_PREPLACE` 执行时，controller 重新调用单步 `ee_forward` 规划并连续失败。审计发现 batch planner 没有同步 Pick 产生的附着物：候选评估使用的 batch collision world 与执行阶段使用的 single planner world 不一致，因此“候选可行”不能证明实际执行规划可行。

同一轮还发现 native 返回路径可能包含 7 个 arm joint 加 2 个 locked finger joint，即 9-DOF；Place 端将该结果直接交给 7-DOF arm FK，触发 `q should have dof = 7, got 9`。这不是 CuRobo 候选本身失败，而是路径关节命名/维度归一化缺失。

### Place 与原生 planner 链路修复

- `TemplateController.attach_objects()` 和 `detach_obj()` 现在对 single planner 与可选 `batch_planner` 使用相同的 active arm joint state、附着 mesh、offset、禁用路径和 sphere 数量；任一 planner 附着失败时，会对已经成功附着的 planner 执行失败回滚，并保持 controller 的附着记录只在全部成功后提交。
- Place 每次重新采样都会清空并更新选中候选的 native v2 路径：保存 `pre-place` path，并在连续下降模式下额外保存 terminal descent path；对应路径写入 `TRANSIT_PREPLACE` 与连续 `TERMINAL_PLACE_DESCENT` 的 command params，由 controller 消费 `preplanned_joint_path`，避免执行阶段无条件重新规划。`legacy_stage_scan` 的 batch 选择路径保持原有行为不变。
- Place 对 native path 使用显式 `joint_names` 归一化：在 FK 和后续 arm 命令前按 active arm joint names 重排并降为 7 个 arm DOF，不再依赖位置切片，也不会把 locked finger 维度误传给 arm-only kinematics。

### 验证状态

开发容器验证环境为 CuRobo 0.8.0 fork（commit `4ea77366ca48ee453e7df139e39fa6532af49f3b`）与 Warp 1.13；相关 Place、controller attachment/path 和安全逻辑离线测试共 `33 passed`，`git diff --check` 通过。严格 retry30 尚待执行；当前不能宣称完整 scene task 已成功，最终仍需确认 `Task is successful, mode=plan_with_render` 且没有 `[LmdbLogger] Episode failed`。

## 本阶段记录：native v2 性能回归诊断与低风险修复（2026-08-14）

### v1/v2 性能差异定位

对照基线 commit `2a0a21f2c836df97c925729084e13d68950b4deb` 的 v1 实现后，当前性能回归首先不是“GPU 没有工作”，而是初始化和查询路径的工作量发生了变化：

| 项目 | v1 基线 | 当前 native v2 | 影响判断 |
|------|---------|----------------|----------|
| planner 数量 | 一个 `MotionGen`，single 与 batch 语义共用同一规划器 | controller 同时持有 single `MotionPlanner` 和候选用 `BatchMotionPlanner` | 两个 planner 各自维护 collision checker、IK、TrajOpt、graph 和 world，初始化/更新成本会叠加 |
| warmup | 单个 `MotionGen` 做一次 warmup | v2 的 pose warmup 会执行完整 pose/IK 路径；此前每个 planner 默认重复 5 次 | 重复完整 pose solve 是初始化 wall-clock 明显增大的高置信来源 |
| `interpolation_dt` | 默认 `0.01` | 迁移初期默认漂移为 `0.025` | 改变轨迹采样与执行步进契约，不能作为隐式 v2 默认值 |
| `time_dilation_factor` | 默认 `1.0` | 迁移初期默认漂移为 `0.8` | 引入未声明的执行时间变化；慢化必须由任务配置显式指定 |

已有 profile 也显示出数量级差异：旧日志中 planner 初始化约 135 秒，而 native v2 retry29/30 的初始化约 410 秒。不过旧日志使用了约 230 个 oriented-bbox proxy，当前 native v2 使用 exact mesh；这组数字只能作为回归证据，不能把全部差异单独归因于双 planner 或某一个 CUDA kernel。当前仍未完成可用于最终结论的严格 scene wall-clock benchmark。

### Controller 侧修复

- `measure_cartesian_path()` 和连续 Place 路径验证现在把整条 `[T, D]` 轨迹交给一次 native v2 FK 调用，FK 完成后只做一次 device-to-host 拷贝；不再对每个 waypoint 单独调用 FK 并同步 CPU。
- 批量 FK 接收路径的显式 `joint_names`，先按 source names 重排到 native planner joint order，再构造 `JointState`；不依赖位置切片，也不会把 locked finger 维度误当成 arm DOF。新增的 FK reorder 测试覆盖了 source/planner 顺序不一致的情况。
- Cartesian 验证是只读计算，批量 FK 使用 `torch.inference_mode()`，避免为每条候选路径保留 autograd graph；这不改变 native v2 FK 的几何结果。
- `pick_place.warmup_iterations` 现在是显式配置项，默认值为 `1`，single 与 batch planner 使用同一配置值。需要额外 warmup 时必须在任务配置中明确指定，而不是由 controller 隐式重复五次。
- controller 的 native v2 默认 `interpolation_dt` 恢复为 `0.01`，默认 `time_dilation_factor` 恢复为 `1.0`；任务可以显式覆盖这两个值。该调整用于对齐 v1 基线，不代表取消任务自身的速度策略。

### CuRobo fork 的 `data_mesh` AABB early-out 边界

本轮 CuRobo fork 在 `InternDataAssets/curobov2/curobo/_src/geom/data/data_mesh.py` 增加了保守的 mesh AABB broad-phase：

- 对于已经能够由 AABB 保守距离判定为“超过本次 query radius 的远处点”，直接返回 Warp miss-distance，跳过 mesh BVH traversal。
- 近 AABB、AABB 内部以及可能命中/穿入 mesh 的查询仍走原有 `wp.mesh_query_point` 和 signed-distance 逻辑；query radius 仍至少保留配置的 `max_dist`，不把 deep-inside 查询裁成 bbox 半径。
- `update_from_warp_id()` 管理的外部 Warp mesh 没有可信的缓存 AABB 时不启用该 early-out，回退到原查询路径。因此外部 Warp mesh 的查询语义不因缓存边界缺失而改变。
- 该优化只修改 vendored CuRobo fork 的 GPU 查询路径，不回写 Isaac 源码，也不修改 Isaac 的 Warp/mesh 数据所有权；CuRobo fork 由用户单独维护。

### 验证状态与边界

- Isaac 开发容器中的 controller 相关 focused unit tests 共 `31 passed in 3.30s`；其中包含新增的批量 FK joint-name reorder 测试，`git diff --check` 通过。
- CuRobo mesh AABB GPU 测试已在开发容器中独立复现：先设置 `export PYTHONPATH=/isaac-sim/extscache/omni.warp.core-1.13.0+lx64:$PYTHONPATH`，再运行 `/isaac-sim/python.sh -m pytest -o addopts="" -q InternDataAssets/curobov2/curobo/tests/_src/geom/sdf/test_mesh_collision.py`，结果为 `2 passed`（Warp 1.13.0，CUDA RTX 4090）。此前关于当前容器无法独立重跑 Warp GPU 测试的记录已由该证据修正，不再适用。
- 已完成 retry32 strict run，但尚未完成可重复的 v1/v2 完整 wall-clock benchmark；retry32 没有拿到完整的 `Task is successful, mode=plan_with_render` 成功 marker，完整 Pick/Place wall-clock 对比仍待执行。

本轮没有修改任何 Isaac 源码或 YAML。应用侧改动仍在 controller/workflows 中，CuRobo 侧改动仅在 `InternDataAssets/curobov2/` fork 内；该 fork 由用户单独维护。

## 本阶段记录：retry32 strict validation（2026-08-14）

### 性能与 native v2 候选结果

retry32 profile：`output/curobo_v2_native_validation_warp113_fork_retry32/de_time_profile_20260814_150132_005695.log`。

- native parser 于 `15:07:31` 完成加载，planner 于 `15:09:32` 初始化完成，初始化约 `122 s`；相较 retry29/30 约 `410 s`，初始化 wall-clock 已显著下降。
- 运行时记录 `interpolation_dt=0.010000`、`ds_ratio=2`，与 v1 基线的执行采样契约对齐。
- batch/pregrasp 的 native 候选规划均记录 `success_count=20`。Pick 后续完成 native attach，`post_grasp_lift` 的 planner 查询也返回 success，说明本次不是 native planner 初始化失败或候选 batch 全失败。

### 严格运行失败位置

严格运行在 `15:13:11` 的 `post_grasp_lift` 阶段因 `attached_object_rotation_slip` abort：`attached_slip_translation_m=0.008345 m`，`attached_slip_rotation_deg=10.4011306 deg`。wrapper exit code 为 `20`，且日志缺少 `Task is successful, mode=plan_with_render`；因此 retry32 不能宣称为成功运行。

该失败应归类为执行安全门/物理附着稳定性问题：native attach 已发生，post-grasp planner 也有成功结果，异常发生在附着目标的实际运动与旋转滑移检查；不能把它归因于 native planner 初始化变慢或 candidate batch 规划失败。
