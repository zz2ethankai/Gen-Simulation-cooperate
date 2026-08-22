# CuRobo controller optimization closed-loop log

本文件记录本轮 controller/Pick/Place 收敛过程。目标是保留 Physics Schema 碰撞语义和直接控制接口，同时让 controller 只承担 CuRobo 规划/执行，Pick/Place 只承担候选与操作阶段语义。

## Stage 1 — placement transaction cleanup

- 日期：2026-08-23
- 基线运行：`output/docker_runtime/grasp-plan-removal-20260823-r2/docker_runtime.isaac.log`
- 基线结果：失败。首个有效错误为 `NO_COLLISION_SAFE_CONTINUOUS_PLACE_PLAN`；上一轮的 `attached -> world_obstacle` 状态机错误已被先前修复，但候选验证结束时仍需要明确清理 placement transaction。
- 修复：
  - placement candidate query 结束后恢复为 `ATTACHED`，不提前 `RESTORE_WORLD`；
  - 只恢复本次 descent 临时禁用的 support collider；
  - 不误恢复永久 planning exclusion；
  - pre-place 与 place 位姿重合时复用 pre-place path，不再发起无位移的连续下降规划。
- 静态检查：`/home/dyf/miniconda3/envs/anygrasp/bin/python -m py_compile ...` 通过；`git diff --check` 通过。
- checkpoint：本阶段提交记录见 git commit `checkpoint: simplify direct pick place runtime`。

后续闭环统一记录官方 wrapper 的 metadata、Isaac 日志和最终 marker：`Task is successful, mode=plan_with_render`，并检查没有 `[LmdbLogger] Episode failed`。

## Stage 2 — candidate accounting and strict validation

- r3 运行：`output/docker_runtime/grasp-plan-removal-20260823-r3/docker_runtime.isaac.log`
- 结果：仍失败。Pick 的 `post_grasp_lift` 规划两次 replan 均未收敛；Place 随后报告 `NO_COLLISION_SAFE_CONTINUOUS_PLACE_PLAN`。
- 修复：Place 不再要求候选 success mask 同时必须带有缓存 trajectory；当 pre-place 与 place 目标重合时，允许 controller 在执行阶段从实测状态规划；新增轻量 `place_plan_snapshot.json` 记录 mask/path 数量和 native 结果摘要。
- 严格配置：`configs/de_plan_with_render_template.yaml` 的 `emit_obs_on_failure` 已设为 `false`。
- 静态检查：py_compile 与 `git diff --check` 通过。
- checkpoint：`d669511 checkpoint: improve placement candidate accounting`。

## Stage 3 — restore batch candidate planning budget

- r4 运行：`output/docker_runtime/grasp-plan-removal-20260823-r4/docker_runtime.isaac.log`。
- 证据：Place 的 20 个 pre-place 候选 native mask 全部为 `success=false`，但大多数位置/姿态误差已经接近 0；问题不是候选路径缓存，而是 batch planner 每个候选只有 1 个 TrajOpt seed。历史 controller 使用 12 个 seed 和 graph-assisted batch planning。
- 修复：保留 batch 并行和 Physics Schema，只将 `NativePlannerFactory` 的 batch TrajOpt seed 从 1 恢复为 12，给每个候选保留可行的碰撞分支搜索预算；`dummy_forward` 未改变。
- 静态检查：修改后执行 py_compile 与 `git diff --check`。
- checkpoint：本阶段提交 `checkpoint: restore batch candidate planning budget`，随后运行 r5 严格闭环。

## Stage 4 — restore candidate-batch semantics in the controller

- r5 运行：`output/docker_runtime/grasp-plan-removal-20260823-r5/docker_runtime.isaac.log`。
- 结果：Pick 成功；`post_grasp_lift` 在第 3 次 controller replan 成功。Place 仍失败，严格错误为 `NO_COLLISION_SAFE_CONTINUOUS_PLACE_PLAN`。`place_plan_snapshot.json` 显示 20 个候选的末端误差大多为 `1e-7` 量级，但 batch mask 全部失败；单次闭环耗时约 307.8 秒。
- 根因收敛：当前 v2 `BatchMotionPlanner` 的 `success_ratio=1.0` 表示必须让整个候选 batch 全部成功，而 Pick/Place 的语义只需要一个可行候选；同时 batch graph seed 被错误地跟随 single-query 的默认关闭配置，历史 batch 路径使用 graph-assisted planning。
- 修复：controller batch pose query 改为 `success_ratio=1/CUROBO_BATCH_SIZE`，并默认开启 batch graph seed；单 query 的 `enable_graph` 配置保持不变。Physics Schema、直接 `dummy_forward` 和 Pick/Place 候选职责不变。
- 静态检查与 checkpoint：本阶段修改后执行 py_compile、`git diff --check`，再提交独立 checkpoint；随后运行 r6 严格闭环。

## Stage 5 — expose the native batch failure boundary

- r6 运行：`output/docker_runtime/grasp-plan-removal-20260823-r6/docker_runtime.isaac.log`。
- 结果：仍在 Place 的 `place_preplace_batch` 失败；CuRobo graph 输出 `Start or End state in collision`。原有 Place 快照只记录末端误差和 success mask，不能说明失败来自起点碰撞、IK feasibility 还是 TrajOpt feasibility。
- 修复：将 native result 的 `feasible/converged/valid_query/solve_time` 等摘要保留在 typed batch metrics；batch 全失败时由 controller runtime 记录一次 `[CuRoboBatchDebug]`，并调用已有 native 起点碰撞审计。诊断不改变 Physics Schema、障碍开关、候选生成或执行逻辑。
- r7 运行：batch planner 仍输出 `Start or End state in collision`，但新增诊断路径因 `runtime.py` 漏定义 `LOGGER` 在写日志时中断；该轮不计入规划成功率。
- 修复：补齐 controller runtime logger，保证诊断失败不会改变 episode 行为；重新以 r8 运行取得有效碰撞证据。
- 目标：用 r8 日志定位真正的碰撞源，再实施最小 controller 修复；本阶段先独立 checkpoint，再做严格闭环。

## Stage 6 — avoid poisoned batch graph seeds

- r8 运行：`output/docker_runtime/grasp-plan-removal-20260823-r8/docker_runtime.isaac.log`。
- 证据：Place 的 batch 仍为 `0/20`，CuRobo 输出 `Start or End state in collision`；同一时刻 native 起点碰撞审计为 `collision_cost_sum=0`、`collision_spheres=0`。候选末端误差大多为 `1e-7` 量级，说明当前失败边界不是 live start state，而是 batch graph 汇总检查中的某个 IK goal。
- 修复：`batch_enable_graph` 默认改为 `false`。批量候选仍保留 Physics Schema、batch IK、12 个 TrajOpt seeds 和 `success_ratio=1/20`；只有任务显式设置 `planning.pick_place.batch_enable_graph: true` 时才启用 graph seed。这样去掉 v2 graph 的全批次失败传播，也减少不必要的 graph 构建/检查开销。
- 约束：单目标规划的 `enable_graph` 语义不变；`dummy_forward` 和 Pick/Place 候选职责不变。
- checkpoint：本阶段代码验证通过后提交独立 checkpoint，再进行 r9 官方闭环。

## Stage 7 — expose the actual failed batch constraints

- r9 运行：`output/docker_runtime/grasp-plan-removal-20260823-r9/docker_runtime.isaac.log`。
- 证据：关闭 batch graph 后仍为 `0/20`，但没有再次出现 graph 的 `Start or End state in collision`；live native 起点审计仍为零。当前 v2 在返回 top-k seed 时清空 `metrics`，所以已有快照只能看到末端误差，不能看到真正拒绝轨迹的碰撞或关节约束。
- 修复：在 `PlannerRuntime` 增加失败诊断开关 `CUROBO_BATCH_DIAGNOSTICS=1`，并由官方 Isaac compose 透传。只在调用方显式开启且 batch 全失败时，使用 native solver 仍持有的 metrics rollout 对返回优化动作做一次只读摘要，记录约束名称、最大值、正值数量和 feasibility；默认关闭，不改变规划结果和正常运行速度。
- Physics Schema、batch 规划、`dummy_forward` 以及 Pick/Place 的候选职责均未改变。
- 静态检查：本阶段修改后执行 py_compile 与 `git diff --check`。
- r10 运行：官方 wrapper 已完成，但由于 compose 尚未透传该开关，未产生原生约束摘要；任务仍以 `place_preplace_batch 0/20` 失败，不能把本轮当作诊断闭环。
- checkpoint：完成静态检查后提交本阶段独立 checkpoint，再以诊断开关运行 r11 官方闭环。

## Stage 8 — restore single-query retry budget

- r11 运行：未到 Place；`post_grasp_lift` 的单目标规划连续 3 次失败，最终由 ExecutionSafety abort。r10 在同一任务和 seed 下第 3 次重试成功，说明失败来自单目标 IK/TrajOpt seed 波动，而不是 live 起点碰撞或 Physics Schema 初始化。
- 修复：controller 的单目标 `max_plan_attempts` 默认从 4 调到 8，接近历史实现的 10；batch 仍通过独立的 `batch_max_plan_attempts` 默认最多 4 次，不让候选 batch 继承单目标预算。
- 约束：不改变碰撞世界、attachment、batch graph 或 `dummy_forward`；任务 YAML 无修改。
- 静态检查：本阶段修改后执行 py_compile 与 `git diff --check`。
- checkpoint：完成静态检查后提交本阶段独立 checkpoint，再运行 r12，确认先能稳定通过 Pick 并重新取得 Place 的约束诊断。

## Stage 9 — keep batch search parallel, recover through the single controller path

- r12 运行：Pick 已通过；Place 的 `place_preplace_batch` 仍为 `0/20`。快照显示 20 个候选的末端位置/姿态误差均接近 `1e-7`，native 起点碰撞审计为零，失败落在整段 Physics Schema 约束判定。诊断开关已经生效，但旧 reshape 把 `[candidate, seed, horizon, dof]` 展平成了 `[candidate*seed, horizon, dof]`，报告 `240 vs 20`，没有得到约束名称。
- 修复：诊断现在只在失败 batch 且显式设置 `CUROBO_BATCH_DIAGNOSTICS=1` 时运行，并只取每个候选的首选 seed 重新计算 metrics；默认运行不增加 GPU rollout。controller 的 batch pose query 在 `0/N` 时最多把 4 个候选交给已有 single planner 重规划，成功后仍返回原有 `BatchPlanResult` 的候选 mask/path，不把规划逻辑重新放回 Pick/Place。
- Physics Schema、attachment、碰撞策略和 `dummy_forward` 均保持不变；单候选 fallback 使用相同的 world revision、collision policy 和 attached geometry。
- 静态检查：py_compile 与 `git diff --check` 通过；下一步运行 r13 官方闭环，记录 fallback 命中率、总耗时和严格成功 marker。

## Stage 10 — align attached-object contact semantics during pre-place transit

- r13 运行：batch 全失败后 controller 的 single fallback 在 candidate `1`、约 `1.2s` 内给出路径；但执行 `transit_preplace` 时连续触发 `unexpected_contact`，最终安全中止。该轮没有进入 place descent，严格成功 marker 缺失，耗时约 `359.8s`。
- 根因：Pick 的 `POST_GRASP_LIFT` 已设置 `allow_target_robot_contact=True`，Place 的 `TRANSIT_PREPLACE` 却只设置了 `allow_target_finger_contact=True`。持物经过手腕/hand 的接触被安全监控当成环境碰撞，和 attached-carry 的 Physics Schema 语义不一致。
- 修复：仅为 Place 的 `TRANSIT_PREPLACE` 补齐 `allow_target_robot_contact=True`；不关闭任何 Physics Schema collider，不改变 planner 的碰撞约束。
- checkpoint：本阶段静态检查后提交独立 checkpoint，再运行 r14 验证是否越过 pre-place transit。

## Stage 11 — keep coincident placement targets on the old release path

- r14 运行：`transit_preplace` 已通过，说明 attached-object 与 robot-link 的接触声明修正有效；但 terminal phase 记录 `continuous-place-plan valid=false`，随后在 `gripper_open` 因 `attached_object_rotation_slip=17.37°` 中止。检查发现本任务的 `pre_place_z_offset` 和 `place_z_offset` 都是 `0.1`，前后目标重合；新逻辑把完整的 pre-place 轨迹重复挂到了 terminal phase，而 Physics Schema 的 release 帧又被当作 attached-carry slip 监控。
- 修复：
  - pre-place/place 目标重合时只保留 `terminal_ok`，不再把 pre-place path 填入 terminal path；执行语义回到旧 PnP 的“到位后保持并开夹”；
  - `GRIPPER_OPEN` 阶段跳过 attached-carry slip 计算，仍保留碰撞、非法状态、掉落和放置支撑接触检查；`DETACH_AND_SETTLE` 后恢复完整的 world/placement 语义。
- 约束：不修改 YAML，不关闭 Physics Schema，不改变 `dummy_forward`；这是 controller/执行边界的最小修正。
- 静态检查与 checkpoint：本阶段修改后执行 py_compile、`git diff --check`，提交独立 checkpoint，再运行 r15 严格闭环。
