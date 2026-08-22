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
