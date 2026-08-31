# Bench2.1 CuRobo v2 原生迁移与性能诊断记录

## 当前调用契约

SimBox 已删除核心 CuRobo v1 兼容层。vendored native v2 位于
[`InternDataAssets/curobov2/`](../InternDataAssets/curobov2/)；运行时入口是
[`TemplateController`](../workflows/simbox/core/controllers/template_controller.py)。

- 普通 transit、place、home 和单目标查询使用 native `MotionPlanner`。
- 抓取候选批量评估使用 native `BatchMotionPlanner`；普通单目标不走候选 batch。
- 位姿目标使用 native `GoalToolPose` 的 5D 形状 `[B,H,L,G,3/4]`。
- attachment 使用 native planner 的 `attachment_manager`，输入为显式
  `attach_collision_prim_paths` 和 native mesh。
- `legacy_stage_scan` 仅是显式旧碰撞 world 扫描模式，不是 v1 planner 或兼容 API。

## 已修复的慢点

1. 移除旧 v1 batch 语义对普通查询的扩散；single 使用 `MotionPlanner`，候选才使用 `BatchMotionPlanner`。
2. 移除重复目标 padding；batch 传递实际候选数，不再复制目标凑固定容量。
3. 避免每次 world/attachment 变化都清空并重建；使用 native `update_world()`、原地 obstacle pose 更新和 attachment manager 的 attach/detach。
4. 动态障碍位姿同步到 single 与 candidate batch 两个 planner。
5. 修复 native batch graph seed 按最大 batch reshape 实际请求状态的问题，改用实际 batch 维度。
6. 按 native 结果 `success [B,S]` 选择 batch item 的成功 seed；失败时 trajectory 为 `None` 则明确视为无路径，禁止伪成功继续执行。
7. 不修改 vendored CuRobo 源码；镜像固定 Warp `1.12.1`，保留 CuRobo `0.8.0` 所需的 `warp.torch` API。Isaac Sim 6.0.1 基础镜像自带 Warp `1.16.0`，该版本已移除这个 shim，不能直接用于当前不可修改的 CuRobo checkout。

## 验证边界

已完成代码级 AST/diff 检查、旧运行时兼容入口扫描，以及开发容器内的
CuRobo v2/Isaac Torch 导入、CUDA `GoalToolPose`/`JointState` 构造和
`[B,S]` 成功结果/失败候选路径 smoke check。容器实际确认：CuRobo `0.8.0`、
固定 commit `4ea77366ca48ee453e7df139e39fa6532af49f3b`、Isaac Torch
`2.11.0+cu128`、CUDA `12.8`。

旧镜像的标准 task 已完成场景加载并进入 native `MotionPlanner` 初始化，
首个失败点是 Warp `1.16.0` 缺少 `warp.torch.device_from_torch`；已改为在
Docker 镜像中固定兼容版本，仍需重建镜像后复跑。完整 Pick/Place
attach/detach task validation、CUDA graph/显存/wall-clock 基准仍未完成，
当前不能给出实测加速倍数。

## 本轮 CuRobo fork 修改（2026-08-13）

本轮用户授权的原生迁移修改如下：

1. Isaac 环境保持 Warp `1.13`；此前镜像中的 Warp `1.12` pin 已撤销。
2. CuRobo fork 源码将 `wp.torch.device_from_torch` 迁移为
   `wp.device_from_torch`，以匹配当前 Warp API。
3. USD parser 增加对多边形碰撞面的三角化支持。

因此，上文第 7 条关于“镜像固定 Warp `1.12.1`”以及“不修改 vendored
CuRobo 源码”的描述属于本轮修改前的历史诊断，不再是当前方案约束。
