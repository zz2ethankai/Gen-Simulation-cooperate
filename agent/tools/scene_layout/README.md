# SceneLayout 确定性场景工具

SceneLayout 是 Agent 的仿真工具，不是机器人 Skill。它把源 task/arena 规范化成 `SceneSpec + SupportGraph`，
再将受限的布局动作编译为新的 scene revision。源 task、arena 和 USD 始终只读；运行时 active/attached
世界不热改。

## 接口与边界

允许的 mutation 只有：

- `MoveEntityOnSupport(entity, support, delta_xy_m)`；
- `RotateEntityOnSupport(entity, yaw_offset_deg)`；
- `SetSupportHeight(support, world_z_m)`；
- `SetRobotPlacement(instance_name, support, delta_xy_m, yaw_offset_deg)`。

接口不接受 YAML pointer、任意字段名或任意代码。每次编译输出独立的 `simbox_task.yaml`、
`simbox_arena.yaml` 和 `scene_mutations.json`，并记录 source hash 与 `scene_revision`。
XY mutation 是同一支撑面内的位移增量，工具同时平移 sampler range、world center、runtime offset 和机器人
source region，避免把 fixture pivot、运行时 bbox center 与世界坐标混成一个数值。跨支撑移动不属于该接口。

`SetSupportHeight` 在内存中一次性更新 fixture 的平移与声明高度、该支撑上的 task/arena region，以及随
支撑移动的子 fixture；任一引用不合法则不写任何派生文档。支撑面上的固定机器人也通过同一 region 关系
联动。新的高度必须在新 episode 中 settle，并重建 Physics/CuRobo collision world 后才能使用。

## 真实 runtime 语义

Banana runtime 在 `RandomRegionSampler.A_on_B_region_sampler` 中分别对 `pos_range` 和 `yaw_rotation` 调用
`numpy.random.uniform`，其 XY 偏移基准是目标物的运行时 bbox center。因此普通范围不是确定性中点。
SceneLayout 对单个候选写入相同的 min/max，确保该 revision 的布局确定；候选之间的 diversity 由搜索器
显式产生并保留 lineage。

`runtime_placement`、`center` 和 `size` 是规范化、可视化、支撑审计及溯源所需信息，不替代 Banana 的
region sampler。`agent/tools/scene_ingest.py` 必须完整保留这些字段，同时生成 runtime 使用的
`random_config`。

## 搜索与反馈

v1 搜索预算固定为每代 8 个 genome、最多 5 代、4 个逻辑 worker queue。debug seeds 为 0–4；held-out
seeds 100–119 只用于冻结策略后的 qualification。调度器本身不声称 GPU 或仿真成功，只有 evaluator 写入
的 schema、spawn、collision、Pick/Place probe、episode 和 data-integrity 证据可以判定候选。

`SceneLayoutPlanner` 只在 `center/size`、运行时 `pos_range/yaw_rotation` 和支撑面 footprint 可以交叉验证时
生成候选。同一实体在 `source_regions` 中的几何会合入 runtime region；实体别名先通过 object/robot 的
`source_name` 归一化，冲突则直接返回 typed `BLOCK`。每代候选 ID 和 `scene_revision` 只由 source hash 与
mutation payload 决定，不依赖 seed 或临时目录。generation 0 从可用支撑范围生成 8 个 move/rotate genome，
generation 1–4 围绕排序最高的父候选缩小实测范围并保持 8 个唯一签名。

`validate_candidate` 先经 `SceneLayoutCompiler` 写独立 revision，再输出 `static_validation.json`，固定检查
schema、支撑关系、物体 placement envelope 是否完整落在支撑 footprint 内，以及是否与同一支撑上的其他
物体 region envelope 重叠。这些只是便宜的确定性筛选；通过后仍必须由上层 evaluator 执行 settle、
Physics/CuRobo audit、Pick/Place probe 和完整 episode。

硬约束先于排序分数。通过硬约束后，才依次比较语义成功、碰撞/可达余量、遮挡、路径长度和 diversity。
配置、资产、collision-world 和数据完整性错误直接阻断；布局错误进入 typed mutation；规划不可行切换
候选；Skill 执行错误只允许修改 contract 中 Agent-owned 参数；未知原因才交给 LLM 返回单一受限意图。
