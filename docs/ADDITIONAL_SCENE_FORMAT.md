# InternDataAssets Additional 场景格式说明

## 1. 适用范围

本文档说明 `InternDataAssets/assets/additional` 目录下场景资产包的实际组织格式。内容基于当前目录中已存在的数据整理，覆盖：

- `scene/`：场景主数据、USD、纹理、资产清单
- `render_image/`：多视角渲染图
- `render_params/`：渲染脚本与参数说明

当前目录内包含 10 个场景目录：

- `file_2`
- `file_3`
- `file_4`
- `file_5`
- `file_6`
- `file_7`
- `file_8`
- `file_9`
- `file_10`
- `file_11`

## 2. 根目录结构

```text
InternDataAssets/assets/additional/
├── render_image/
│   ├── file_2__hero.png
│   ├── file_2__front_right.png
│   └── ...
├── render_params/
│   ├── README.md
│   ├── render_isaac_scenes.py
│   ├── run_render.sh
│   └── render_multiview.log
└── scene/
    ├── conversion_summary.json
    ├── file_2/
    │   ├── scene.json
    │   ├── scene.usd
    │   ├── manifest.json
    │   ├── repair_conversion_manifest.json
    │   ├── assets/
    │   │   ├── file_2__*.usd
    │   │   └── *_texture.png
    │   └── textures/
    │       ├── floor_*.png
    │       ├── wall_*.png
    │       └── file_2_material_*.png
    └── file_3/
        └── ...
```

## 3. 命名规则

### 3.1 场景目录

- 场景目录命名为 `file_<id>`
- 同一场景的主文件均以该目录名为上下文

示例：

- `scene/file_2/scene.json`
- `scene/file_2/scene.usd`
- `render_image/file_2__hero.png`

### 3.2 渲染图

渲染图命名规则：

```text
file_<id>__<view>.png
```

当前固定 4 个视角：

- `hero`
- `front_right`
- `rear_left`
- `top_oblique`

### 3.3 USD 资产

单物体 USD 一般命名为：

```text
file_<scene_id>__<object_name>.usd
```

示例：

- `file_2__coffee_table_0_id57.usd`
- `file_10__shoe_storage_cabinet_0_id7.usd`

## 4. 单场景目录格式

每个 `scene/file_<id>/` 目录至少包含以下核心内容：

- `scene.json`：场景语义与布局描述
- `scene.usd`：整场景 USD
- `manifest.json`：场景转换与校验清单
- `repair_conversion_manifest.json`：补修复的小资产转换记录
- `assets/`：按物体拆分的 USD 与补纹理
- `textures/`：墙面、地面和材质贴图

其中：

- `scene.json` 用于描述房间、墙、门窗、物体、布局生成痕迹
- `scene.usd` 用于 Isaac/Omniverse 侧直接加载
- `manifest.json` 用于追踪从源场景到 USD 场景的转换结果

## 5. `scene.json` 格式

当前 10 个场景的 `scene.json` 顶层键一致，包含以下字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `doors` | array | 门信息 |
| `metadata` | object | 生成器元数据 |
| `objects` | array | 当前数据中为空，预留对象列表 |
| `proceduralParameters` | object | 程序生成参数 |
| `rooms` | array | 房间定义 |
| `walls` | array | 墙体定义 |
| `windows` | array | 窗户定义 |
| `query` | string | 原始文本场景生成指令 |
| `scene_skill_tags` | array | 场景技能标签 |
| `scene_skill_trace` | object | 场景程序和对象图追踪 |
| `floor_objects` | array | 地面摆放对象 |
| `wall_objects` | array | 墙挂对象，当前数据中通常为空 |
| `raw_floor_plan` | string | 原始文本平面图 |
| `wall_height` | number | 墙高 |
| `raw_doorway_plan` | string | 原始文本门洞规划 |
| `room_pairs` | array | 房间连通对 |
| `open_room_pairs` | array | 开放式连通房间对 |
| `open_walls` | object | 开放墙段定义 |
| `raw_window_plan` | string | 原始文本窗户规划 |
| `asset_bank` | object | 资产候选池，键通常为对象 `assetId` |
| `object_selection_plan` | object | 按房间记录的对象选择计划 |
| `selected_objects` | object | 最终按房间选中的对象 |
| `stairs` | array | 楼梯信息，当前样本中为空 |
| `floor_layout_reports` | array | 每个房间的布置报告 |
| `floor_layout_unresolved` | array | 未完全解决的布置项 |
| `scene_quality_reward` | object | 场景质量评估结果 |
| `scene_quality_ok` | boolean | 质量检查是否通过 |
| `landing_audit` | object | 落地检查结果 |
| `landing_ok` | boolean | 落地检查是否通过 |
| `small_asset_manifest_paths` | array | 小物件修复清单路径 |

### 5.1 `metadata`

`metadata` 当前包含以下键：

- `agent`
- `roomSpecId`
- `schema`
- `warnings`
- `agentPoses`

这部分主要是生成器和场景 schema 元数据，不是几何主体。

### 5.2 `proceduralParameters`

`proceduralParameters` 当前包含以下键：

- `ceilingColor`
- `ceilingMaterial`
- `floorColliderThickness`
- `lights`
- `receptacleHeight`
- `reflections`
- `skyboxId`

这部分描述程序生成和渲染相关的全局环境参数。

### 5.3 `rooms` 数组项格式

每个房间对象包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `Polygon` | array | 房间 3D 边界点，元素为 `{x, y, z}` |
| `id` | string | 房间唯一标识 |
| `roomType` | string | 房间类型 |
| `vertices` | array | 2D 平面轮廓点 |
| `floor_design` | string | 地面材质文本描述 |
| `wall_design` | string | 墙面材质文本描述 |
| `full_vertices` | array | 完整轮廓点 |

### 5.4 `walls` 数组项格式

每个墙体对象包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string | 墙体 ID，常含房间、方向、序号 |
| `roomId` | string | 所属房间 |
| `material` | object | 材质信息，当前含 `name`、`color` |
| `Polygon` | array | 墙体 3D 多边形 |
| `connected_rooms` | array | 与该墙相关的房间 |
| `width` | number | 墙段长度 |
| `height` | number | 墙高 |
| `direction` | string | 朝向，如 `west`、`north` |
| `segment` | array | 2D 线段端点 |

### 5.5 `windows` 数组项格式

每个窗户对象包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `assetId` | string | 资产标识，当前常见为 `random_window` |
| `id` | string | 窗户 ID |
| `room0` | string | 关联房间 0 |
| `room1` | string | 关联房间 1，外墙时可与 `room0` 相同 |
| `wall0` | string | 主墙 ID |
| `wall1` | string | 对应另一侧墙 ID |
| `Polygon` | array | 窗户 3D 边界 |
| `assetPosition` | object | 资产位置 `{x, y, z}` |
| `roomId` | string | 所属房间 |
| `color` | string | 调试/标注颜色 |
| `windowSegment` | array | 2D 窗口线段 |
| `windowBoxes` | array | 窗户两侧盒区域 |

### 5.6 `doors` 数组项格式

每个门对象至少包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `assetId` | null 或 string | 门资产 ID，当前样本中常为空 |
| `id` | string | 门 ID |
| `prompt_door` | null 或 string | 原始门描述 |
| `room0` | string | 连接房间 0 |
| `room1` | string | 连接房间 1 |
| `wall0` | string | 墙体 ID 0 |
| `wall1` | string | 墙体 ID 1 |
| `Polygon` | array | 门洞 3D 边界 |
| `assetPosition` | object | 门位置 `{x, y, z}` |
| `doorBoxes` | array | 门两侧盒区域 |
| `doorSegment` | array | 2D 门洞线段 |

部分门额外包含可开合信息：

- `openable`
- `openness`

### 5.7 `floor_objects` 数组项格式

`floor_objects` 是场景内最重要的对象布局清单。每个对象当前至少包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `assetId` | number | 对应 `asset_bank` 中的资产 ID |
| `id` | string | 对象实例 ID |
| `kinematic` | boolean | 是否按运动学对象处理 |
| `position` | object | 位置 `{x, y, z}` |
| `rotation` | object | 旋转 `{x, y, z}` |
| `material` | object/null | 材质覆盖，当前样本中常为空 |
| `roomId` | string | 所属房间 |
| `size` | object | 尺寸 `{x, y, z}` |
| `vertices` | array | 平面占用轮廓 |
| `object_name` | string | 对象名 |
| `placement_layer` | string | 摆放层，如 `floor` |
| `description` | string | 文本描述 |
| `is_hinged` | boolean | 是否带铰链结构 |
| `is_interactive` | boolean | 是否可交互 |
| `allowed_interactions` | array | 允许的交互动作 |
| `table_object` | array | 放在该对象上的小物件清单 |
| `inside_object` | array | 放在该对象内部的小物件清单 |
| `hinged_parts` | array | 铰链部件定义 |

其中：

- `table_object` 常用于桌面、柜面上的小物件
- `inside_object` 常用于抽屉、柜体、容器内部物件
- `hinged_parts` 用于门板、抽屉等可开合部件

### 5.8 生成与校验相关字段

以下字段更多是“生成过程追踪”而不是“场景运行时主体”：

- `scene_skill_trace`
- `asset_bank`
- `object_selection_plan`
- `selected_objects`
- `floor_layout_reports`
- `floor_layout_unresolved`
- `scene_quality_reward`
- `landing_audit`
- `small_asset_manifest_paths`

如果只需要加载最终场景，通常优先关注：

- `rooms`
- `walls`
- `windows`
- `doors`
- `floor_objects`
- `wall_height`

## 6. `manifest.json` 格式

`manifest.json` 是每个场景的 USD 转换总清单，当前结构稳定，顶层字段如下：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `scene` | string | 场景名，如 `file_2` |
| `source_scene_json` | string | 原始场景 JSON 路径 |
| `scene_usd` | string | 生成后的整场景 USD 路径 |
| `assets_converted` | number | 已转换资产数 |
| `objects_instanced` | number | 已实例化对象数 |
| `missing` | array | 缺失资产列表 |
| `asset_manifest` | array | 单资产转换清单 |
| `small_assets` | object | 小资产修复统计 |
| `camera` | object | 渲染相机信息 |
| `validation` | object | 场景校验结果 |
| `postprocess_tabletop` | object | 桌面小物体后处理摘要 |

### 6.1 `asset_manifest` 数组项格式

每个资产转换记录包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `source_xml` | string | 源 MJCF/XML 路径 |
| `asset_usd` | string | 输出 USD 路径 |
| `mesh_prims` | number | 网格 prim 数 |
| `source_kind` | string | 资产来源类型，当前样本中为 `mjcf` |
| `source_files` | array | 关联源文件列表 |
| `source_extent` | array | 源包围盒尺寸 |
| `scale` | array | 应用缩放 |
| `textured_materials` | number | 带贴图材质数 |

### 6.2 `small_assets`

`small_assets` 当前包含以下键：

- `scene_xml`
- `small_mesh_assets`
- `added`
- `small_textured_materials`
- `missing`

### 6.3 `camera`

`camera` 当前包含以下键：

- `eye`
- `target`
- `extent`

### 6.4 `validation`

`validation` 当前包含以下键：

- `open`
- `prim_count`
- `mesh_count`
- `reference_count`
- `bbox_min`
- `bbox_max`

### 6.5 `postprocess_tabletop`

`postprocess_tabletop` 当前包含以下键：

- `structure`
- `openings`
- `small_assets`

## 7. `repair_conversion_manifest.json` 格式

该文件记录额外修复过的小资产转换结果，顶层字段固定为：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `done` | array | 修复成功的资产 |
| `fail` | array | 修复失败的资产 |
| `skipped` | number | 跳过数量 |

`done` 数组项格式：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `folder` | string | 资产目录名 |
| `object_name` | string | 对象名 |
| `usd` | string | 修复后 USD 路径 |
| `texture` | string | 补充贴图名 |
| `source` | string | 来源，当前样本中常为 `mesh_out` |
| `textured_materials` | number | 带贴图材质数 |

## 8. `conversion_summary.json` 格式

`scene/conversion_summary.json` 是场景级汇总文件，格式如下：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `total_done` | number | 总成功数 |
| `total_fail` | number | 总失败数 |
| `scenes` | object | 按场景统计的成功/失败数 |

`scenes.<scene_name>` 子项格式：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `done` | number | 当前场景修复成功数 |
| `fail` | number | 当前场景修复失败数 |

## 9. 贴图与资产目录约定

### 9.1 `textures/`

`textures/` 目录通常包含三类 PNG：

- `floor_<room_name>.png`：按房间命名的地面贴图
- `wall_<room_name>.png`：按房间命名的墙面贴图
- `file_<id>_material_<n>_<hash>.png`：材质导出的通用贴图

### 9.2 `assets/`

`assets/` 目录包含：

- 按对象拆分的 USD 文件
- 少量修复补充贴图，如 `*_texture.png`

## 10. 渲染参数目录格式

`render_params/` 目录用于记录多视角渲染过程：

- `render_isaac_scenes.py`：Isaac Sim 渲染脚本
- `run_render.sh`：渲染启动脚本
- `render_multiview.log`：渲染日志
- `README.md`：当前渲染参数说明

当前 `README.md` 中记录的固定参数包括：

- 分辨率：`2560 x 1440`
- 视角：`hero`、`front_right`、`rear_left`、`top_oblique`
- 渲染器：`RayTracedLighting` + Path Tracing
- 每像素采样：`256`
- 每视角总采样：`2048`

## 11. 生成 SimBox Arena / Task YAML

仓库提供 `scripts/generate_assets_addition_scene_configs.py`，用于把 `workflows/simbox/assets/assets_addition/file_*` 场景包转换成 SimBox 可加载的 arena yaml 和 scene-only task yaml。

### 11.1 环境要求

该脚本在 split 模式下会解析 `scene.usd` 的 root layer，需要 `pxr` / `usd-core`。推荐使用当前已验证的环境：

```bash
/home/dyf/miniconda3/envs/anygrasp/bin/python scripts/generate_assets_addition_scene_configs.py --help
```

如果用普通 Python 环境运行且没有 `pxr`，脚本会直接报错，不会生成缺失位姿的 YAML。

### 11.2 常用命令

生成单个场景，并覆盖已有文件：

```bash
/home/dyf/miniconda3/envs/anygrasp/bin/python scripts/generate_assets_addition_scene_configs.py \
  --scene-dir workflows/simbox/assets/assets_addition/file_2 \
  --overwrite
```

将 task yaml 输出到 example 目录：

```bash
/home/dyf/miniconda3/envs/anygrasp/bin/python scripts/generate_assets_addition_scene_configs.py \
  --scene-dir workflows/simbox/assets/assets_addition/file_2 \
  --output-task-dir workflows/simbox/core/configs/tasks/addition/example \
  --overwrite
```

批量生成所有 `file_*` 场景：

```bash
/home/dyf/miniconda3/envs/anygrasp/bin/python scripts/generate_assets_addition_scene_configs.py \
  --all \
  --overwrite
```

### 11.3 输出位置

默认输出：

- arena yaml：`workflows/simbox/core/configs/arenas/addition/<scene>_arena.yaml`
- task yaml：`workflows/simbox/core/configs/tasks/addition/<scene>_scene_task.yaml`

可通过以下参数覆盖：

- `--output-arena-dir`
- `--output-task-dir`

### 11.4 split 模式语义

默认 `--mode split` 会生成：

- arena：加载 `empty_room.usd` 和 `assets/arena/*.usd`
- task objects：只从 `scene.usd` 中收集 `assets/task_obj/...` reference

这意味着 arena 中已经包含的不可动物体不会重复进入 task objects。例如 `file_2` 中的 sofa、bookcase、bed frame 等 `assets/arena/...` 会留在 arena yaml 中，而不会被写入 task yaml 的 `objects`。

task object 的位姿来自 `scene.usd` 实例 prim 上的：

- `xformOp:translate` -> YAML `translation`
- `xformOp:transform:yaw` -> YAML `euler[2]`
- `xformOp:scale` -> YAML `scale`

对于很多 `/World/SmallAssets` 小物体，原始 `scene.usd` 的实例 prim 本身可能没有 `xformOp:translate`，位置被烘在对应 `Aligned_obj.usd` 的 mesh 顶点中。这类对象在 YAML 中出现 `translation: [0.0, 0.0, 0.0]` 是对当前资产格式的忠实导出，不代表一定丢失位置。

### 11.5 full-scene 模式语义

`--mode full-scene` 会把 `scene.usd` 作为一个整体 fixture 写入 arena yaml，并生成不含 task objects 的 scene-only task yaml。该模式适合只想加载完整静态场景，不需要单独引用 task objects 的情况。

### 11.6 YAML 布尔值约定

SimBox 当前配置文件约定布尔值使用 Python 风格的大写：

```yaml
apply_randomization: False
render: True
update: True
```

脚本已使用自定义 YAML dumper，生成文件时会输出 `True` / `False`，不会输出小写 `true` / `false`。

## 12. 最小可用集合

如果目标是“加载并使用一个最终场景”，建议至少依赖以下文件：

- `scene/file_<id>/scene.usd`
- `scene/file_<id>/scene.json`
- `scene/file_<id>/textures/`
- `scene/file_<id>/assets/`

如果目标是“追踪转换过程和质量状态”，再额外读取：

- `scene/file_<id>/manifest.json`
- `scene/file_<id>/repair_conversion_manifest.json`
- `scene/conversion_summary.json`

## 13. 结论

`InternDataAssets/assets/additional` 实际上是一套“场景包”格式，而不是单独的场景文件。每个场景包同时包含：

- 可直接加载的 `scene.usd`
- 结构化语义描述 `scene.json`
- 资产转换清单 `manifest.json`
- 小资产修复记录 `repair_conversion_manifest.json`
- 本地纹理与拆分资产目录
- 对应的多视角渲染图

因此后续如果要接入工具链，建议把 `scene/file_<id>/` 视为基本分发单元，而不是只取其中一个 JSON 或一个 USD 文件。
