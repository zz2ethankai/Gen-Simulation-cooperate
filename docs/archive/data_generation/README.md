# 数据生成 README / Quick Start


InternDataEngine 的数据生成流程可以概括为：

```text
任务配置 -> 加载场景与机器人 -> 场景随机化 -> skill 规划与执行 -> 渲染观测 -> 保存 episode
              ^                                                       |
              +------------------ DataEngine 执行配置 -----------------+
```

启动一次任务需要同时指定两类配置：

- **DataEngine 执行配置**决定“怎样生成”，包括采用哪种 Plan/Render 流程、生成数量、仿真参数和输出目录。
- **SimBox 任务配置**决定“生成什么”，包括场景、机器人、物体、相机、摆放区域、skill 序列和数据元信息。

## 1. Quick Start

以下命令使用一份执行配置和一份任务配置快速测试

```bash
cd /home/bld/ykqin/InternDataEngine

conda activate interndata

CUDA_VISIBLE_DEVICES=0 python launcher.py \
  --config configs/simbox/de_plan_with_render_template.yaml \
  --load_stage.scene_loader.args.cfg_path=workflows/simbox/core/configs/tasks/basic/split_aloha/track_the_targets/track_the_targets.yaml
```


命令中的两个配置入口分别是：

```text
--config <DataEngine 执行配置>
--load_stage.scene_loader.args.cfg_path=<SimBox 任务配置>
```

执行配置通常不需要更换,任务配置可以在/home/bld/ykqin/InternDataEngine/workflows/simbox/core/configs/tasks选取即可,注意选取好yaml文件,复制路径时候使用相对路径(从workflows开始)

```bash
find output/quick_start \( -name meta_info.pkl -o -path '*/lmdb/data.mdb' \) -print
```

## 2. 两类配置文件

### 2.1 DataEngine 执行配置

执行配置位于 [`configs/simbox/`](../../configs/simbox/)。它把数据生成组织为 Load、Plan、Render、Store 等 stage，并通过 `load_stage.scene_loader.args.cfg_path` 接入任务配置。

| 配置 | 流程 | 适用场景 |
| --- | --- | --- |
| [`de_plan_with_render_template.yaml`](../../configs/simbox/de_plan_with_render_template.yaml) | 规划、执行和渲染在同一 stage 内完成 | 最直接的完整数据生成；调试和小规模生成首选，也是流体任务所需模式 |
| [`de_plan_and_render_template.yaml`](../../configs/simbox/de_plan_and_render_template.yaml) | 先规划，再在同一进程中顺序渲染 | 分阶段排查规划或回放问题 |
| [`de_plan_template.yaml`](../../configs/simbox/de_plan_template.yaml) | 只生成轨迹，不渲染图像 | 预生成轨迹、降低渲染开销 |
| [`de_render_template.yaml`](../../configs/simbox/de_render_template.yaml) | 读取已有轨迹并重新渲染 | 更换背景、材质或渲染设置后复用轨迹 |
| [`de_pipe_template.yaml`](../../configs/simbox/de_pipe_template.yaml) | 规划与渲染解耦为流水线 worker | 吞吐优先的大规模生产，需要按机器资源调整 worker 数量 |

常用字段包括：

- `load_stage.scene_loader.args.simulator`：物理步长、渲染步长、是否无界面运行和 GPU 设置。
- `load_stage.layout_random_generator.args.random_num`：计划生成的随机化样本数。
- `plan_with_render_stage`、`plan_stage`、`render_stage`：参与本模式的处理阶段。
- `store_stage.writer.args.output_dir` 或 `seq_output_dir`：完整 episode 或仅轨迹的保存位置。
- `stage_pipe`：Pipe 模式的 stage 数、worker 数、调度和超时设置。

配置中的任意字段都可以用点号路径在命令行覆盖。一次性试验可直接覆盖；需要团队复现的设置应保存为新的 YAML。

### 2.2 SimBox 任务配置

任务配置位于 [`workflows/simbox/core/configs/tasks/`](../../workflows/simbox/core/configs/tasks/)，按任务类型组织，并在下一层按机器人平台区分。

| 分类 | 作用 |
| --- | --- |
| `basic/` | 抓取、放置、插入、倾倒、跟踪等基础操作任务 |
| `art/` | 微波炉、抽屉、箱盖等可动关节物体的打开、关闭和推动任务 |
| `pick_and_place/` | 面向大量物体类别的通用抓取与放置任务 |
| `long_horizon/` | 由多个 skill 串联或多机器人协作完成的长时程任务 |
| `example/` | 用于理解配置结构的示例任务，不作为业务分类 |

一份任务 YAML 可以在顶层 `tasks:` 下定义一个或多个任务。每个任务主要包含：

| 字段 | 作用 |
| --- | --- |
| `asset_root`、`arena_file`、`env_map` | 资产根目录、静态场景和环境光照 |
| `robots` | 机器人型号、基础配置和初始位姿 |
| `objects` | 交互物体的 USD、类型、位姿、尺度和随机化方式 |
| `regions` | 机器人和物体相对于桌面、地面等 fixture 的摆放范围 |
| `cameras` | 相机内参文件、挂载位置、外参和随机化范围 |
| `skills` | 按机器人、手臂和执行顺序组织的操作原语 |
| `data` | 输出子目录、语言指令、版本和最大 episode 长度 |

可从 [`sort_the_rubbish.yaml`](../../workflows/simbox/core/configs/tasks/example/sort_the_rubbish.yaml) 查看完整结构。任务文件由 [`TaskConfigParser.parse_tasks()`](../../workflows/simbox/utils/task_config_parser.py#L31-L40) 读取；机器人条目再与 `robot_config_file` 指向的基础配置合并，任务 YAML 中的同名字段优先，合并逻辑见 [`_merge_robot_configs()`](../../workflows/simbox_dual_workflow.py#L58-L72)。

## 3. Docker 并行生成

Docker 并行生成位于单任务生成流程的外层：宿主机负责发现任务、分配 GPU、监控和汇总，每个容器运行一份 DataEngine 执行配置与任务配置。并行 YAML 是额外的**调度配置**，不会替代前述两类配置。

运行前需要 Docker、NVIDIA Container Toolkit、可用的项目镜像，以及 [`docker-compose.simbox.yml`](../../docker/docker-compose.simbox.yml) 中正确的挂载和镜像设置。

建议从 [`parallel_generate_v2.yaml`](../../configs/simbox/parallel_generate_v2.yaml) 复制一份本机配置：

```bash
cp configs/simbox/parallel_generate_v2.yaml configs/simbox/parallel_local.yaml
```

然后至少确认以下字段：

```yaml
launcher_type: simbox_parallel_v2
name: local_parallel_run

parallel:
  backend: docker
  gpus: [0, 1]
  workers_per_gpu: 1

defaults:
  de_config: configs/simbox/de_plan_with_render_template.yaml
  random_num: 10
  dataset_root: null

jobs:
  - id: job_gpu0
    task: workflows/simbox/core/configs/tasks/pick_and_place/franka/single_pick/omniobject3d-ball.yaml
    gpu: 0

  - id: job_gpu1
    task: workflows/simbox/core/configs/tasks/art/franka/open_the_microwave/open_the_microwave_part0.yaml
    gpu: 1
```

先执行 dry-run，检查任务路径、GPU 分配和输出位置，不启动容器：

```bash
python launcher.py \
  --config configs/simbox/parallel_local.yaml \
  --parallel.dry_run=true \
  --monitor.enabled=false
```

检查通过后正式运行：

```bash
python launcher.py --config configs/simbox/parallel_local.yaml
```

训练数据默认写入执行配置的 `output_dir`，运行状态和汇总写入 `output/_parallel_runs/<run_id>/`。两者用途不同：前者是 episode 数据，后者是调度日志、状态和报告。一个稳定的起点是每张 GPU 一个容器；确认显存和 IO 仍有余量后，再提高 `workers_per_gpu`。

完整参数、状态字段、失败恢复、统计和缓存清理见[《SimBox Docker 并行生成使用说明》](../docker并行生成使用说明.md)。

## 4. 替换资产

资产替换的核心是：让任务 YAML 中的逻辑对象继续指向正确的物理资产。只改 USD 路径并不一定足够，资产的尺度、prim 层级、碰撞、抓取标注以及 skill 参数都要与新资产匹配。

### 4.1 替换任务中的可操作物体

在任务 YAML 的 `objects` 中修改目标条目：

```yaml
asset_root: workflows/simbox/assets

objects:
  - name: target_object
    path: custom/my_object/Aligned_obj.usd
    target_class: RigidObject
    prim_path_child: Aligned
    translation: [0.0, 0.0, 0.0]
    euler: [0.0, 0.0, 0.0]
    scale: [1.0, 1.0, 1.0]
```

路径默认按 `asset_root + objects[].path` 解析。本仓库也支持在单个 object 中设置 `asset_root` 覆盖任务级根目录，具体解析逻辑见 [`asset_path_utils.py`](../../workflows/simbox/core/utils/asset_path_utils.py#L10-L21)。

替换时建议保留原来的 `name`，这样 `regions[].object` 和 `skills[].objects` 不需要同步改名；如果修改了名称，所有引用该对象的位置都必须同时更新。`target_class` 和 `prim_path_child` 必须与 USD 的物理类型和 prim 层级一致，否则加载或碰撞绑定会失败。

用于 `pick` 类 skill 的刚体资产通常应包含：

```text
Aligned_obj.usd
Aligned_grasp_sparse.npy
textures/...
```

`Aligned_obj.usd` 应使用真实尺度和稳定朝向，并包含刚体、碰撞体和摩擦属性。抓取 skill 默认从 USD 同目录读取 `Aligned_grasp_sparse.npy`，对应代码见 [`Pick.__init__()`](../../workflows/simbox/core/skills/pick.py#L27-L50)。资产转换、刚体/碰撞设置和抓取位姿生成可参考[官方 New Assets](https://internrobotics.github.io/InternDataEngine-Docs/custom/assets.html)及 [`workflows/simbox/tools/rigid_obj/`](../../workflows/simbox/tools/rigid_obj/) 和 [`workflows/simbox/tools/grasp/`](../../workflows/simbox/tools/grasp/)。

### 4.2 替换场景或静态 fixture

- 更换整套静态场景：创建或选择新的 arena YAML，并修改任务中的 `arena_file`。
- 只更换桌面、地面等 fixture：修改 arena YAML 中对应 `fixtures[].path`、`target_class`、位姿和尺度。
- arena 中的 fixture `name` 会被 `regions[].target` 和碰撞过滤引用；改名时必须同步更新任务配置。

arena 在任务 reset 时载入，并与任务中的 objects、robots、regions 一起构成完整场景，加载位置见 [`SimBoxDualWorkFlow.reset()`](../../workflows/simbox_dual_workflow.py#L74-L125)。因此，`arena_file` 只负责静态场景，不会自动替换任务中的机器人、操作物体、相机和 skill。

可动关节资产还需要稳定的 link、joint、collider 和质量设置，以及与操作类型匹配的 `Kps/<info_name>/info.json`。任务中的 `target_class: ArticulatedObject`、`info_name`、`path`/`art_cat` 和关节范围应与资产包保持一致。

## 5. 替换 skill

### 5.1 使用已有 skill

已有 skill 位于 [`workflows/simbox/core/skills/`](../../workflows/simbox/core/skills/)。替换任务行为时，修改任务 YAML 的 `skills` 段即可：

```yaml
skills:
  - franka:
      - left:
          - name: pick
            objects: [target_object]
            pre_grasp_offset: 0.10
            post_grasp_offset_min: 0.05
            post_grasp_offset_max: 0.10
          - name: place
            objects: [target_object, target_container]
            place_direction: vertical
```

层级含义依次是：执行阶段、机器人名称、手臂、按顺序执行的 skill 列表。以下约束必须保持一致：

- 机器人名称必须存在于 `robots[].name`。
- 手臂名称必须与该机器人控制器提供的手臂一致。
- `objects` 中的名称必须存在于当前任务场景。大多数抓取和关节 skill 只接受 `objects` 中的可操作物体；`place` 的目标还可以是 arena fixture。
- 不同资产的尺寸、抓取方向和可达空间不同，替换资产后通常需要重新调整 offset、方向过滤、容差和成功判定参数。

运行时会按 YAML 中的 `name` 从注册表取得 skill 类并依次实例化，见 [`SimBoxDualWorkFlow._initialize_skills()`](../../workflows/simbox_dual_workflow.py#L206-L240)。现有 skill 的用途和参数可参考[项目 skill 速查](../simbox_skill_reference.md)和[官方 Skills Overview](https://internrobotics.github.io/InternDataEngine-Docs/concepts/skills/overview.html)。

### 5.2 新增自定义 skill

现有 skill 无法表达目标操作时，再新增 Python 实现：

1. 在 `workflows/simbox/core/skills/` 新建 skill 类，继承 `BaseSkill` 并使用 `@register_skill` 注册。
2. 实现命令生成、可行性检查、阶段完成和成功判定等必要接口；可从 [`pick.py`](../../workflows/simbox/core/skills/pick.py#L25-L77) 复制结构。
3. 在 [`skills/__init__.py`](../../workflows/simbox/core/skills/__init__.py) 中导入并导出新类，使模块加载时完成注册。
4. 在任务 YAML 中将 `name` 改为注册名并配置所需字段。注册名由类名转换为小写下划线形式，例如 `NewSkill` 对应 `new_skill`，规则见 [`register_skill()`](../../workflows/simbox/core/skills/base_skill.py#L7-L12)。
5. 先用 `random_num=1`、固定随机种子和 `--debug` 验证，再进入批量或 Docker 并行生成。

自定义 skill 的接口和命令格式可参考[官方 New Skill](https://internrobotics.github.io/InternDataEngine-Docs/custom/skill.html)。

## 6. 推荐工作顺序

```text
选已有任务并单条跑通
  -> 修改 task YAML
  -> 替换资产或 skill
  -> random_num=1 固定 seed 调试
  -> 小批量检查成功率和输出
  -> Docker 多 GPU 并行生成
```

单条任务没有稳定运行前不建议直接扩大并发。并行层只提高吞吐，不会修复资产层级、抓取标注、skill 参数或任务可达性问题。
