# Bench2.1 目标环形工作点规划与 Pick 验证

## 1. 系统位置与职责边界

工作点规划位于场景布置和 Skill 执行之间：

```text
任务目标与场景几何
  → 目标周围生成底盘候选
  → 离线底盘碰撞/房间边界过滤
  → CuRobo 抓取轨迹过滤
  → Pick 真实执行与数据录制
```

三个阶段分别回答不同问题：

- Geometry：机器人底盘能否安全放入房间；
- CuRobo Probe：当前机器人和抓取标注是否存在联合成功的 pre-grasp/grasp 轨迹；
- Pick：夹爪是否实际接触目标并通过官方 `Pick.is_success()`。

任何阶段的成功都不能代替后续阶段。特别是 CuRobo 成功不表示物体一定能被夹住。

## 2. 对外接口

### 2.1 生成离线候选

```bash
cd /home/bld/ykqin/InternDataEngine

python scripts/simbox/plan_workspace_layout.py \
  --task <simbox_task.yaml> \
  --target <object-name> \
  --output-dir output/<run-name>
```

`--target` 可以省略，解析顺序为：第一个 `Pick.objects`，然后
`delivery_active_objects[0]`。

默认采样参数只在 planner schema 中定义一次：

```yaml
manipulation_workspace:
  planner: target_annulus_v1
  sampling:
    min_radius_m: 0.45
    max_radius_m: 1.05
    candidate_count: 24
    preferred_radius_m: 0.75
    sequence: golden_angle
```

任务没有这一段时使用同样的默认值。命令行可以临时覆盖半径、候选数量和首选半径。

### 2.2 CuRobo 与 Pick 验证

```bash
python scripts/simbox/validate_workspace_candidates.py \
  --manifest output/<run-name>/candidates.json \
  --gpus 0,1,2,3 \
  --max-pick-candidates 3
```

验证器对全部几何候选运行 planning-only Probe，排序后最多对前三名执行 seed 0 Pick。
第一个 seed 0 成功点继续验证 seed 1、2。

### 2.3 单独观察或执行已编译配置

```bash
TASK_CONFIG=<compiled-task.yaml> \
RUN_MODE=observe \
GPU_ID=0 \
bash scripts/simbox/run_bench21_observe.sh
```

`RUN_MODE=observe` 只把 Skill 替换为双臂 `ObserveHold`，保持实测关节且不重新计算或覆盖机器人位置。
`RUN_MODE=skill` 原样运行配置中的 Skill。

## 3. 离线候选生成

### 3.1 目标与工作面

目标 region 必须提供：

- `center`：目标世界 XY 参考点；
- `target`/`B`：支撑面；
- `support_surface_z` 或支撑面高度。

支撑面用于恢复目标的世界位置。可移动刚体位于 `floor` 时仍可生成机器人初始位姿候选；
这一结果只表示底盘几何可放置，不表示地面 Pick 已通过机械臂可达性验证。

### 3.2 确定性环形采样

以目标世界 XY 为圆心，使用黄金角低差异序列：

```text
u_i = (i + 0.5) / N
r_i = sqrt(r_min² + u_i × (r_max² - r_min²))
theta_i = i × golden_angle

base_xy = target_xy + r_i × [cos(theta_i), sin(theta_i)]
yaw = atan2(target_y - base_y, target_x - base_x)
```

对 `r²` 均匀采样使候选在环形面积上均匀分布。候选 ID 固定为
`annulus_000 ... annulus_023`，相同输入会得到完全相同的坐标和排序。

Waypoint、交付初始点、桌面 north/south/east/west 和 interaction edge 不参与候选生成。

### 3.3 几何门禁

每个候选只经过两项低成本检查：

1. 机器人旋转 footprint OBB 不与家具、柜体、床、墙相交；
2. footprint 四个角全部位于 floor 边界内。

通过点按照以下顺序排列：

```text
abs(radius - preferred_radius)
→ candidate_id
```

离线阶段不再使用肩部距离公式推测机械臂可达性。

### 3.4 运行坐标写入

Manifest 保存世界坐标。SimBox robot region 使用 floor-relative shift：

```text
sampler_shift = candidate.world_xy - floor.translation.xy
```

编译器统一写入：

- `robots[0].euler[2]`；
- `source_regions.robot_initial_region`；
- `regions[robot].random_config.pos_range`；
- `metadata.workspace_candidate`。

源 task 和 arena 始终只读。

## 4. CuRobo Probe

`GraspPlanEvaluator` 是唯一的抓取轨迹筛选实现，由 Probe 和 Pick 共同调用。

它对每只手返回：

```yaml
feasible: true
arm: left
grasp_count: 20
pregrasp_success_count: 8
grasp_success_count: 6
joint_success_count: 5
selected_grasp_index: 17
selected_grasp_score: 0.12
attach_prim_valid: true
failure_code: null
```

`joint_success_count` 只统计同一个抓取标注在 pre-grasp 和 grasp 两段都成功的轨迹。
数量为零时直接失败，不允许回退到最后一个或第一个抓取标注继续执行。

Probe 同时检查 `pick_obj.mesh_prim_path` 是否存在于 CuRobo `motion_gen.world_model`。
这是候选无关错误；一旦发现便终止后续点位，避免同一资产错误被重复执行。

每个底盘候选使用独立 Isaac 进程，左右臂在同一进程中分别输出 JSON。Probe 只执行一个
controller bookkeeping 命令，不移动机器人、不写 LMDB 或视频。左右臂 JSON 都以原子方式
落盘后，验证器主动结束该 Isaac 进程，不等待通用数据生成 workflow 重试或录制；writer 所需的
占位目录也位于该候选的 run directory 内。

候选排序为：

```text
joint_success_count 降序
→ selected_grasp_score 升序
→ abs(radius - 0.75) 升序
→ candidate_id
```

## 5. Pick 验收

真实 Pick 使用最小官方写法和 Probe 推荐手臂：

```yaml
- name: pick
  objects: [<target>]
  filter_y_dir: [forward, 90]
  filter_z_dir: [downward, 140]
```

没有新增成功谓词。一次 Pick 成功必须同时满足：

- 日志包含 `Task is successful`；
- episode event 为 `success`；
- episode 目录不是 `fail_*`；
- 本轮产生新的 `meta_info.pkl`；
- 本轮产生新的 `lmdb/data.mdb`。

seed 0 成功后运行 seed 1、2，并标为：

- `3/3 stable`；
- `2/3 partially_stable`；
- `unstable`。

## 6. Manifest 与失败定位

`candidates.json` 是全流程唯一状态文件：

```yaml
version: 3
target: {}
sampling: {}
geometry_candidates: []
curobo_results: []
pick_attempts: []
selected_candidate: null
status: geometry_ready
failure_code: null
```

主要状态：

- `geometry_ready`：存在底盘几何候选；
- `no_geometry_candidate`：全部碰撞或越界；
- `no_curobo_candidate`：没有联合成功抓取轨迹；
- `no_pick_success`：前三个 CuRobo 点位均未通过 Pick；
- `blocked`：资产、抓取标注或 attach prim 等候选无关错误；
- `success`：至少 seed 0 Pick 成功。

## 7. 代码维护约束

- `core.workspace` 必须能在没有 Isaac/CuRobo 的 Python 环境中导入；
- Geometry 不调用 CuRobo，Evaluator 不生成底盘点，Pick 不解析 workspace manifest；
- Probe 和 Pick 不得复制 CuRobo 判断逻辑；
- 配置默认值只能定义一次；
- 不保留旧 edge/waypoint 算法的兼容 mode；
- 修改算法时同步替换测试和本文件，不追加互相冲突的历史章节；
- 运行产物只写入 output，不覆盖源 task/arena。

## 8. 当前验证基线

20 个 Bench2.1 scene_4 任务的离线审计位于：

```text
output/workspace_annulus_v3_audit/
```

其中 19 个任务产生 7–16 个几何候选；`livingroom_toy_blocks_cleanup` 因目标在 floor
上被正确拒绝。书房杯子 Probe 已验证能够在第一次运行中发现 attach prim 契约问题并停止，
没有再进入无界重试。

官方 Split Aloha bottle Pick 对照任务中，Probe 在 seed 1 得到 12 条联合成功轨迹；同一配置的
真实 Pick 随后出现 `Task is successful`，并产生 success event、`meta_info.pkl`、LMDB 和三路视频。
这验证了 Probe 与 Pick 共享 evaluator 后没有破坏官方成功路径。

8 个 Bench2.1 场景的运行时回归结果位于：

```text
output/workspace_annulus_v3_validation_summary.md
```

其中 7 个任务在首批并行 Probe 中暴露 `ATTACH_PRIM_NOT_IN_CUROBO_WORLD` 并提前停止；
`bedroom_hand_cream_to_organizer` 的 `annulus_003` 左臂得到 20 条联合成功轨迹，随后进入真实
Pick，但 seed 0 没有通过内置成功判断，因此最终状态为 `NO_PICK_SUCCESS`。这证明三个阶段的
失败类型能够被分别定位。
