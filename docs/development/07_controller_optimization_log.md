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
- checkpoint：待提交 `checkpoint: improve placement candidate accounting`。
