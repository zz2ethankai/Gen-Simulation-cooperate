# 文档索引

仓库文档按两类组织：**API 文档**描述接口契约（YAML 字段、skill 参数、配置项），**开发文档**记录代码修改历史（时间戳 + 改动 + 原因 + 验证）。历史分析报告与对外对接材料存放在 `docs/archive/`。

| 目录 | 内容 | 适用场景 |
| --- | --- | --- |
| [`docs/api/`](api/) | 接口契约：skill API、task/arena YAML、配置参考、场景数据格式 | 写 task YAML、调 skill 参数、查配置字段时 |
| [`docs/development/`](development/) | 修改记录：导航、碰撞规划、CuRobo 迁移、资产修复等 8 个主题 | 理解某段代码为什么这么写、查历史改动时 |
| [`docs/archive/`](archive/) | 历史分析报告、旧版文档、对外对接建议 | 需要回溯旧文档原文时 |

## API 文档

| 文档 | 内容 |
| --- | --- |
| [api/01_Skill_API.md](api/01_Skill_API.md) | 27 个 skill 的注册名、参数、默认值、成功判定、调试产物、最小示例；含几何参数讲解与调参经验 |
| [api/02_Task与Arena_YAML.md](api/02_Task与Arena_YAML.md) | arena 与 task YAML 字段：类型、必填性、默认值、运行时注入的 planning 配置 |
| [api/03_配置参考.md](api/03_配置参考.md) | 配置项到消费代码的映射：task/arena/de*.yaml 三层，含 simulator rt_* 字段 |
| [api/04_场景数据格式.md](api/04_场景数据格式.md) | `assets_addition` 场景包的数据格式（package_manifest/asset_optimization_report）与生成脚本 |

skill 参数表只在 01_Skill_API.md 保留，02、03 通过交叉引用指向它，不重复维护。

## 开发文档

按主题组织，每篇自包含（背景 → 时间线 → 修改记录）。

| 文档 | 覆盖范围 |
| --- | --- |
| [development/01_本地导航系统.md](development/01_本地导航系统.md) | 2026-07-27 ~ 08-08，Nav2 移除、本地 A* 导航全历程 |
| [development/02_碰撞规划与PickPlace闭环.md](development/02_碰撞规划与PickPlace闭环.md) | 2026-08-02 ~ 08-08，CollisionSceneManager 四层闭环、physics_schema |
| [development/03_CuRobo_v2迁移与性能.md](development/03_CuRobo_v2迁移与性能.md) | CuRobo v1 兼容层移除、native v2 迁移、7 个性能慢点 |
| [development/04_PickPlace稳定性与姿态过滤.md](development/04_PickPlace稳定性与姿态过滤.md) | pick 稳定性、place 姿态过滤调参 |
| [development/05_资产挂接与USD结构修复.md](development/05_资产挂接与USD结构修复.md) | attach 路径三阶段修复、铰链结构约定、导入修复日志 |
| [development/06_机器人配置与技能计时.md](development/06_机器人配置与技能计时.md) | PandaOmron TCP 偏移、telemetry 技能计时 |
| [development/07_场景4验证与录制.md](development/07_场景4验证与录制.md) | 工作点规划三阶段、观察录制、docker 并行生成 |
| [development/08_开发环境与脚本.md](development/08_开发环境与脚本.md) | Docker 开发容器、验证 wrapper、生命周期脚本 |

## 归档

`docs/archive/` 保留全部旧版文档原文（git 历史可追溯）：

- 被合并的旧文档：`SIMBOX_SKILLS_API.md`、`simbox_skill_reference.md`、`SIMBOX_ARENA_TASK_YAML_API.md`、`SIMBOX_CONFIG_REFERENCE.md`、`ADDITIONAL_SCENE_FORMAT.md`（以上为 api/ 重写前的旧版）、`PANDA_OMRON_PLACE_ORIENTATION_FILTER_GUIDE.md`、`yaml_change.md`、`SCENE7_NEW_ASSET_ADAPTATION_AGENT_GUIDE.md`、`PROJECT_DOCUMENTATION.md`、全部 `Bench2.1_*.md` 修复说明、`NEW_ASSET_IMPORT_FIX_LOG.md`、`铰链修复.md`、`docker并行生成使用说明.md`
- 分析/对接报告：`benchmark1_asset_inventory.md`、`benchmark_vs_standard_usd_stage_dump.md`、`capx_integration_feasibility.md`、`usd_structure_comparison.md`、`Bench2.1_资产问题与制造方对接建议.md`
- 遗留：`ASSETS_ADDITION_TASK_TODO.md`、`MERGE_TODO.md`、`data_generation/`、`images/`

## 约定

- API 文档核对日期与代码基线写在各文档开头；改代码后需要同步更新对应文档。
- 开发文档条目格式：`日期 + commit 短 hash + 改动 + 原因 + 文件 + 验证`。
- 新改动落库时，按主题追加到对应开发文档，不新建文件。
- skill 参数、task/arena 字段、配置映射各归一篇，其他文档只交叉引用不复制。