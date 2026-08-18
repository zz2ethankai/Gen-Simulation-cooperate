# Bench 2.1 资产问题与制造方对接建议

> 用途：与资产制造方进行问题复盘、交付规范确认和下一版本资产验收。
> 范围：`InternDataAssets/Bench_2.1_isaacsim/scene_4` 的 4 个房间、20 个任务，以及其在 SimBox 中的加载、观察录制、材质、光照和碰撞表现。
> 当前统计时间：2026-07-12。

## 1. 会议需要先达成的核心共识

我们的原则是：在资产条件允许时，尽量沿用 InterData 官方的 YAML 语义和运行流程；但不能为了形式一致，忽略 Bench 2.1 与官方资产在目录结构、材质复杂度和随机化资源数量上的实际差异。

目前最核心的判断有五点：

1. **地板和墙壁可以继续采用官方式 `texture` 随机化。**它们都是单一平面，材质替换边界清晰。
2. **家具和任务刚体不能直接把现有 `materials/` 目录当成随机纹理库。**其中大部分图片是同一套 PBR 材质的不同通道或重复副本，不是多种外观。
3. **没有 `texture` 配置时保留 USD 内部材质，是当前最安全的默认行为。**这不是缺少实现，而是避免破坏木材、金属、玻璃、布料等多材质关系。
4. **当前资产真正缺少的是明确的“可随机外观集合”和机器可读的材质清单。**仅增加 `apply_randomization: true` 不能解决这个问题。
5. **下一版资产应交付稳定的规范入口和验证报告。**不能再依赖使用方猜测哪个目录是正式资产、哪张图是 RGB、哪些文件只是语义图或 PBR 通道。

## 2. 当前资产交付的整体结构

### 2.1 官方资产结构

官方任务通常使用一个全局资产根目录：

```yaml
asset_root: workflows/simbox/assets
```

`workflows/simbox/assets` 当前实际指向 `InternDataAssets/assets`。官方将可复用纹理库集中放在全局根目录，例如：

```text
InternDataAssets/assets/
├── floor_textures/
├── background_textures/
├── table_textures/
├── dark_table_textures/
├── light_table_textures/
└── envmap_lib/
```

代码出处：

- 官方任务资产根目录：[`open_the_microwave_left_im.yaml`](../workflows/simbox/core/configs/tasks/art/lift2/open_the_microwave/left/open_the_microwave_left_im.yaml)，第 4 行。
- 官方 arena 的桌子、地板、背景纹理配置：[`pick_randomized_arena.yaml`](../workflows/simbox/core/configs/arenas/pick_randomized_arena.yaml)，第 4-70 行。

### 2.2 Bench 2.1 资产结构

Bench 2.1 以房间作为资产根目录：

```yaml
asset_root: InternDataAssets/Bench_2.1_isaacsim/scene_4/03_livingroom
```

每个房间包含：

```text
03_livingroom/
├── texture_libs/
│   ├── floor_textures/
│   └── wall_textures/
├── shared/canonical assets（通过 assets/basic/03_livingroom 表示）
├── assets/basic/03_livingroom/
│   ├── fixtures/
│   └── small_objects/
├── assets/basic/livingroom_coffee_table_cleanup/
├── assets/basic/livingroom_mug_to_coaster/
└── ...其他任务局部目录
```

这意味着 Bench 同时存在两套容易混淆的资产路径：

- `assets/basic/03_livingroom/...`：房间级规范共享资产，当前 SimBox arena 实际引用这一套。
- `assets/basic/<task_name>/...`：任务局部副本或交付副本，不一定是运行时真正使用的文件。

代码/配置出处：

- Bench 的 `asset_root`：[`simbox_task.yaml`](../InternDataAssets/Bench_2.1_isaacsim/scene_4/03_livingroom/assets/basic/livingroom_coffee_table_cleanup/simbox_task.yaml)，第 3 行。
- arena 中沙发实际引用 `assets/basic/03_livingroom/...`：[`simbox_arena.yaml`](../InternDataAssets/Bench_2.1_isaacsim/scene_4/03_livingroom/assets/basic/livingroom_coffee_table_cleanup/simbox_arena.yaml)，第 191-193 行。
- 当前材质修复脚本只扫描规范共享目录：[`repair_bench21_material_bindings.py`](../scripts/simbox/repair_bench21_material_bindings.py)，第 46-49 行。

### 2.3 这套结构带来的主要风险

1. 修改任务局部副本后，运行画面可能完全不变，因为运行时引用的是房间级规范资产。
2. 同一个家具可能在多个目录出现，无法仅凭文件名判断哪个是主版本。
3. 任务配置、arena 配置和 USD 内部引用的相对路径可能指向不同资产层级。
4. 资产更新后，如果只覆盖一套副本，容易产生任务间视觉不一致。

## 3. 已遇到并确认的关键问题

### 3.1 路径不可直接执行和机器绑定问题

#### 现象

原始任务中的 `asset_root`、`arena_file`、机器人 USD、HDR 目录存在不同层级或机器环境下不可用的问题。相同任务换仓库位置后不能稳定加载。

#### 当前处理

我们通过 [`normalize_bench_paths.py`](../scripts/simbox/normalize_bench_paths.py) 统一 20 个任务的：

- `asset_root`
- `arena_file`
- `envmap_lib`
- `robots[0].path`
- 墙面法线方向
- 当前引擎不支持的 region 参数

关键实现位于该脚本第 35-114 行。

#### 对制造方的要求

- 所有 USD、纹理、HDR、metadata 引用必须使用交付包内的相对路径。
- 禁止写入开发者个人绝对路径，如 `/home/...`、`/data1/...`。
- 每个任务必须声明唯一、可验证的 `asset_root` 和 `arena_file`。
- 交付前应在一个不同绝对路径的干净目录中执行加载测试。

### 3.2 规范共享资产与任务副本不一致

#### 现象

同一房间的任务共用一套规范家具，但任务目录下又保留家具副本。部分规范 USD 与任务副本的材质绑定不同，导致“查看源图片正常、运行结果却不同”。

#### 已确认事实

- 当前 arena 引用 `assets/basic/<room>/...` 的规范共享资产。
- 修改 `assets/basic/<task>/...` 中的同名文件不会影响当前运行。
- 之前电视柜等规范 USD 绑定过 `semantic_fixture.png`，而任务局部资产可能绑定普通 RGB 图。

#### 当前处理

[`repair_bench21_material_bindings.py`](../scripts/simbox/repair_bench21_material_bindings.py) 会检查规范共享 `Aligned_obj.usd`：

- 如果 USD 绑定 `semantic_*.png` 且同目录存在 `texture.png`，改回 `texture.png`。
- 如果没有明确 RGB 图，则跳过，不臆造材质。
- 修改前备份原 USD。

关键实现位于该脚本第 34-91 行。

#### 对制造方的要求

- 明确每个资产的唯一 canonical 目录和唯一主 USD。
- 如果需要任务副本，应声明它是引用、派生物还是独立版本。
- 提供 `asset_manifest.json`，记录 `asset_id`、canonical USD、版本、哈希和被哪些任务引用。
- 同一 `asset_id` 不允许在多个目录中出现内容不同但版本号相同的 USD。

### 3.3 RGB 材质和语义材质混用

#### 现象

部分家具 USD 在 RGB 录制路径中绑定了 `semantic_fixture.png`。源目录虽然存在正常木纹图片，但运行时实际使用的是 USD 已绑定的语义图。

#### 根因

SimBox 的规则是：

- YAML 有 `texture`：调用 `apply_texture()`，显式覆盖 USD 材质。
- YAML 没有 `texture`：保留 USD 内部材质。

代码出处：[`banana.py`](../workflows/simbox/core/tasks/banana.py)，第 302-324 行。

因此，没有 YAML `texture` 的家具是否正常，完全取决于 USD 内部材质是否正确。

#### 对制造方的要求

- RGB、语义分割、实例 ID、法线、深度使用不同的材质层或不同 render purpose。
- RGB 默认 variant/默认 material binding 不得指向 `semantic_*`、`id_*` 图片。
- 语义图不得与普通 RGB 图使用模糊文件名或相同材质槽。
- 每个 USD 应提供自动检查结果：默认渲染材质引用了哪些图片、是否存在缺失路径。

### 3.4 大面积黑色并伴随绿色条纹

#### 现象

多次录制中，某一面墙或地板会变成大面积纯黑，并出现绿色高光条纹。整体环境亮度仍会变化，因此最初容易被误判为 HDR 太暗。

#### 排查结论

通过以下对照实验确认：

1. 四面墙固定使用不同 `texture_id`。
2. 黑面跟随 `wall_textures/2.png` 从南墙移动到北墙。
3. 对同一张图片进行无损 RGB PNG 重编码。
4. 再次固定使用该纹理，黑面消失。
5. 恢复随机纹理后，最终录制正常。

由此可以确认：黑面不是墙的固定几何错误，也不是 HDR 亮度的直接结果；直接触发点是该纹理在 Isaac/Hydra 纹理加载或缓存链路中的兼容异常。由于普通图片查看和像素内容正常，更具体地属于编码/纹理缓存兼容问题，而不是图片本身画成黑色。

#### 当前处理

[`refresh_bench21_plane_textures.py`](../scripts/simbox/refresh_bench21_plane_textures.py) 将四个房间的 24 张有效地板/墙面图片无损重编码为标准 RGB PNG，并备份原图。关键实现位于第 31-48 行。

#### 对制造方的要求

- RGB 纹理统一交付为经过验证的 8-bit sRGB PNG 或 JPG。
- 明确禁止调色板 PNG、异常 alpha、错误色彩空间和损坏 ICC profile。
- 每张图片提供分辨率、通道数、色彩空间、SHA256。
- 交付前在目标 Isaac Sim 版本中完成纹理加载测试，不能只用普通图片查看器验收。
- 纹理更新时必须更新版本或哈希，避免渲染器继续命中旧缓存。

### 3.5 地板和墙壁碰撞缺失或方向错误

#### 现象

原始 `PlaneObject` 主要承担显示功能。没有碰撞时机器人会穿过或下落；墙面旋转方向相同时，相对墙面的法线和碰撞薄盒会偏向错误一侧。

#### 当前处理

我们接入了合作者仓库的静态薄盒碰撞方案，并保留了来源标记：

- `collision_enabled: true` 时创建薄盒碰撞体。
- 碰撞体与渲染平面保留间隙。
- 隐藏碰撞体使用 USD `guide` purpose。
- 四面墙法线统一朝向室内，使碰撞体位于墙外侧。

代码出处：

- [`plane_object.py`](../workflows/simbox/core/objects/plane_object.py)，第 43-77 行。
- [`normalize_bench_paths.py`](../scripts/simbox/normalize_bench_paths.py)，第 18-25、85-114 行。

#### 对制造方的要求

每个场景平面或固定家具必须明确：

- 是否参与碰撞。
- collider 类型。
- 碰撞厚度或近似方式。
- 是否是 support surface。
- 法线方向和正面方向。
- 单位、坐标系和中心点定义。

不应仅交付“看起来位置正确”的可视模型，而缺少用于机器人仿真的物理语义。

### 3.6 相机名称与机器人类型耦合

#### 现象

如果全局相机固定写成 `split_aloha_navigate_global`，更换成 Franka 或其他机器人后，记录器无法按当前机器人名识别该相机。

#### 当前处理

观察配置使用官方动态写法：

```yaml
name: ${tasks.0.robots.0.name}_global
```

代码出处：[`prepare_bench_observe_config.py`](../scripts/simbox/prepare_bench_observe_config.py)，第 12、40-51 行。

#### 对制造方的建议

如果制造方同时交付任务 YAML，应避免把机器人类型写进场景资产或相机逻辑：

- 场景资产只描述世界坐标相机或相机挂载点。
- 机器人相关相机使用动态机器人名或明确的 mount contract。
- 相机外参必须声明相对哪个 Prim 和哪个坐标系。

### 3.7 HDR 路径和光照随机化能力不足

#### 现象

Bench 任务曾被归一化到官方全局 `envmap_lib`，但 Bench 交付本身只提供房间交付包内的共享 HDR。两者混用会让结果依赖仓库外部资源，也无法判断光照来自哪套资产。

#### 当前状态

当前 20 个任务统一使用：

```yaml
envmap_lib: ../shared_assets/envmap_lib
```

该目录当前只有一张：

```text
abandoned_factory_canteen_01_1k.hdr
```

因此即使写了 `apply_randomization: true`，HDR 文件本身也没有选择空间。当前任务的 intensity 和 rotation 还是固定范围时，实际只能作为确定性基线，不构成有效的光照随机化。

光照选择代码出处：[`banana.py`](../workflows/simbox/core/tasks/banana.py)，第 475-516 行。

#### 对制造方的要求

- 提供经过亮度标定的 HDR 集合，而不是只提供一张 HDR。
- 每张 HDR 声明推荐 intensity 范围、白平衡、动态范围和适用场景。
- HDR 应覆盖不同光线方向、色温和室内/室外条件，但避免极端过曝或近黑。
- 提供 HDR 文件清单、许可证、哈希和 Isaac Sim 预览图。

## 4. 当前纹理和资产多样性统计

### 4.1 官方全局纹理库

当前本地官方资产快照中：

| 纹理库 | 文件数量 |
|---|---:|
| `table_textures` | 896 |
| `background_textures` | 101 |
| `floor_textures` | 16 |
| `dark_table_textures` | 7 |
| `light_table_textures` | 5 |
| `envmap_lib` | 87 HDR + 87 EXR |

官方随机化的优势不是 YAML 写法更特殊，而是它已经准备了可以直接随机抽取的纯纹理库和大量同类资产实例。

### 4.2 Bench 平面纹理库

每个房间当前只有：

| 纹理库 | 每个房间数量 |
|---|---:|
| `floor_textures` | 3 |
| `wall_textures` | 3 |

四个房间运行时有效的平面纹理总数是 24 张。目录下另外存在 `scene/texture_libs` 副本，应由制造方说明其用途，避免重复交付。

当前地板和墙壁已经支持随机抽取，但随机空间明显小于官方。

### 4.3 固定家具实例数量

按房间和家具类别统计：

- 共 38 个“房间—固定家具类别”组合。
- 37 个组合只有 1 个资产实例。
- 只有客厅 `toy_block` 类别有 3 个实例。

所以多数固定家具只有一个几何和一套外观，无法通过“同类别换资产”产生有意义的随机化。

### 4.4 交互刚体实例数量

按房间和小物体类别统计：

- 共 58 个“房间—小物体类别”组合。
- 51 个组合只有 1 个实例。
- 只有杯子、面包片、笔、书、积木、杯垫、杂志等少量类别有 2-3 个实例。

官方 `update_rigid_objs()` 支持在同类别或跨类别随机选择整个 USD，代码位于 [`dr.py`](../workflows/simbox/core/utils/dr.py) 第 187-233 行。但 Bench 大部分类别只有一个实例，即使打开配置，随机池也基本为空。

### 4.5 “图片很多”不等于“外观很多”

Bench 交互物体目录经常包含：

```text
xxx-albedo.png
xxx-normal.png
xxx-roughness.png
xxx-metalness.png
xxx-opacity.png
```

这些是同一材质的不同 PBR 通道，不是五种可替换纹理。部分图片还会重复出现在：

```text
materials/
textures/
usd/materials/
```

因此制造方必须明确区分：

- 材质通道集合。
- 外观 variant 集合。
- 语义或实例分割图片。
- fallback 图片。
- 重复导出副本。

## 5. 当前随机化能力边界

| 对象类型 | 当前状态 | 只改 YAML 是否足够 | 主要限制 |
|---|---|---|---|
| 地板、墙壁 | 已随机 | 是 | 纹理只有 3 张，目录必须只含 RGB 图片 |
| 单材质桌面/布面 | 暂未系统接入 | 通常是 | 需要制造方先提供纯 RGB 外观库 |
| 多材质家具 | 保留 USD 材质 | 否 | 当前覆盖方式可能把木材、金属、玻璃全部替换成同一材质 |
| 普通刚体 | 保留 USD PBR 材质 | 简单整物体覆盖时可以 | 现有目录混有 normal、roughness 等通道，不能直接随机 |
| 完整 PBR 外观切换 | 未支持 | 否 | 需要按套选择 albedo/normal/roughness/metallic |
| 同类别更换整个刚体资产 | 引擎支持 | 资产目录符合规则时可以 | Bench 大部分类别只有一个实例 |
| 关节家具 | 保留 USD 材质 | 通常不够 | 需要保持关节结构和多材质槽一致 |

相关代码行为：

- 所有对象有 YAML `texture` 时调用 `apply_texture()`：[`banana.py`](../workflows/simbox/core/tasks/banana.py)，第 302-324 行。
- arena fixture 每次 episode 重新设置纹理：[`banana.py`](../workflows/simbox/core/tasks/banana.py)，第 167-174、458-462 行。
- `PlaneObject` 从目录随机选择一个文件：[`plane_object.py`](../workflows/simbox/core/objects/plane_object.py)，第 82-108 行。
- `GeometryObject` 递归覆盖已有材质绑定：[`geometry_object.py`](../workflows/simbox/core/objects/geometry_object.py)，第 53-89 行。
- `RigidObject` 当前只创建一个单纹理 OmniPBR 材质：[`rigid_object.py`](../workflows/simbox/core/objects/rigid_object.py)，第 62-78 行。
- `ArticulatedObject` 当前只从 `*.jpg` 中选纹理：[`articulated_object.py`](../workflows/simbox/core/objects/articulated_object.py)，第 124-140 行。

## 6. 对下一版资产制造的正式建议

### 6.1 P0：不满足就不能稳定加载或验收

#### A. 唯一规范入口

每个资产必须提供：

```text
asset_id
canonical_usd
asset_version
sha256
category
role: fixture | rigid_object | articulated_object | support_surface
```

不得依赖使用方在多个同名目录中猜测主版本。

#### B. 路径可迁移

- USD 内所有外部引用必须为包内相对路径。
- 在任意绝对目录解压后都能加载。
- 不允许失效引用、个人路径、未交付的 Nucleus 路径。

#### C. 默认 RGB 材质正确

- 默认 variant 必须能直接用于 RGB 渲染。
- 不得默认绑定 semantic、ID 或 debug 材质。
- 所有贴图引用必须存在，且大小写与磁盘文件一致。

#### D. 坐标和单位统一

每个资产需要声明：

- 米制单位。
- up axis。
- forward axis。
- 原点位置。
- bbox。
- 推荐放置高度。
- 关节零位和关节范围。

#### E. 物理语义完整

- collider 类型和近似方式。
- 刚体/静态/运动学状态。
- 质量、摩擦、碰撞开关。
- support surface 位置和范围。

### 6.2 P1：支持有效的纹理和材质随机化

#### A. 平面纹理库

建议每类至少交付 10-20 张经过筛选的 RGB 纹理：

```text
texture_libs/
├── floor/
│   ├── wood/
│   ├── tile/
│   └── carpet/
└── wall/
    ├── paint/
    ├── wallpaper/
    └── concrete/
```

每张纹理应提供：

- 可平铺性说明。
- 真实尺度或推荐 `texture_scale`。
- 色彩空间。
- 分辨率。
- 类别标签。

#### B. 家具外观 variant

不要只提供一张 `texture.png`。对于确实需要随机化的家具，建议提供：

```text
appearances/
├── oak/
│   ├── albedo.png
│   ├── normal.png
│   ├── roughness.png
│   └── metallic.png
├── walnut/
└── painted_white/
```

同时提供机器可读清单：

```yaml
appearance_sets:
  - id: oak
    target_material_slots: [cabinet_wood]
    albedo: appearances/oak/albedo.png
    normal: appearances/oak/normal.png
    roughness: appearances/oak/roughness.png
    metallic: appearances/oak/metallic.png
```

#### C. 多材质部件边界

需要稳定命名材质槽或 Prim：

```text
Fabric
Wood
Metal
Glass
Plastic
Screen
```

制造方应声明哪些槽允许随机，哪些必须保留。否则使用方无法只换沙发布料而保留木腿，也无法只换柜体而保留金属把手。

#### D. 交互刚体外观和实例多样性

对需要训练泛化能力的类别，建议至少提供：

- 3-5 个不同几何实例，或
- 每个实例 3-5 套完整 PBR 外观。

不能把 normal、roughness、metalness 文件数量当作外观数量。

### 6.3 P2：支持高质量数据生成

#### A. HDR 光照库

- 至少提供多种光线方向、色温和亮度条件。
- 为每张 HDR 提供推荐强度范围。
- 在 Isaac Sim 中输出标准预览和亮度统计。

#### B. 可复现随机化元数据

每个生成样本应能记录：

- seed
- asset variant
- appearance variant
- 纹理文件
- HDR 文件
- HDR rotation/intensity
- 相机扰动参数

#### C. 自动质量门禁

建议制造方随资产交付验证脚本或报告，至少检查：

- USD 能否打开。
- 所有外部引用是否存在。
- 默认 RGB 材质是否正常。
- 是否误绑 semantic 图。
- 图片能否被目标 Isaac Sim 版本加载。
- bbox、单位、法线和坐标轴是否合理。
- collider 是否存在且不穿模。
- 关节是否能在合法范围运动。
- 是否出现全黑、全白、紫色缺失纹理或绿色异常条纹。

## 7. 建议的资产交付目录模板

```text
<asset_id>/
├── asset.usd                     # 唯一规范入口
├── manifest.yaml                 # 版本、类别、单位、坐标、哈希
├── geometry/
├── collisions/
├── materials/
│   ├── default/
│   │   ├── albedo.png
│   │   ├── normal.png
│   │   ├── roughness.png
│   │   └── metallic.png
│   └── appearances/
│       ├── variant_01/
│       └── variant_02/
├── semantics/
│   ├── labels.yaml
│   └── semantic_materials.usd
├── metadata/
│   ├── physics.yaml
│   ├── support_surfaces.yaml
│   └── articulation.yaml
└── validation/
    ├── rgb_preview.png
    ├── collision_preview.png
    └── validation_report.json
```

关键原则：RGB 材质、PBR 通道、语义材质和碰撞数据在目录与清单中明确分离，但通过唯一 `asset.usd` 和 manifest 关联。

## 8. 建议的验收标准

### 8.1 单资产验收

每个资产应满足：

1. 从任意绝对路径加载成功。
2. 没有缺失引用和绝对路径。
3. 默认 RGB 预览与制造方参考图一致。
4. semantic 材质不会出现在普通 RGB 渲染中。
5. 多材质槽名称稳定且有说明。
6. collider、质量、摩擦和关节参数完整。
7. 每个 appearance variant 都能单独加载。
8. 所有 PBR 通道尺寸和 UV 对齐。

### 8.2 单场景验收

每个房间应满足：

1. arena 中所有路径可解析。
2. 地面能承载机器人，墙体碰撞方向正确。
3. 固定家具位置、尺度和朝向正确。
4. 随机纹理运行至少 20 次无黑纹理、缺失纹理或明显曝光异常。
5. 不同任务引用同一 canonical 家具时渲染结果一致。
6. 固定 seed 能复现相同资产、纹理和光照选择。

### 8.3 整包验收

整包应满足：

1. 20 个任务全部通过路径和引用检查。
2. 资产 manifest 中没有重复 `asset_id` 或冲突版本。
3. 所有纹理、HDR、USD 都有哈希。
4. 提供变更日志，说明新增、删除、替换了哪些资产。
5. 在目标 Isaac Sim 版本中完成自动 smoke test。

## 9. 会议上需要制造方明确回答的问题

### 资产主版本

1. `assets/basic/<room>` 和 `assets/basic/<task>` 哪个是规范来源？
2. 任务目录中的家具是副本、引用还是独立版本？
3. 后续更新会更新哪一层目录？

### 材质语义

4. `texture.png`、`material_0.png` 和 `*-albedo.png` 各自承担什么角色？
5. `semantic_fixture.png` 为什么会被部分规范 USD 默认绑定？
6. 哪个文件是制造方认可的默认 RGB 外观？
7. 多目录中的重复 PBR 文件是否可以去重？

### 随机化资源

8. 是否计划为地板、墙壁提供更多纹理？目标数量是多少？
9. 哪些固定家具允许随机外观？哪些必须保持原始材质？
10. 是否能为重点交互类别提供多个几何实例或完整 PBR appearance sets？
11. 是否能提供稳定的 material slot/Prim 命名？

### 光照和渲染兼容

12. 为什么当前交付只包含一张 HDR？
13. 纹理是否在目标 Isaac Sim 版本中做过批量加载验证？
14. 能否随交付提供 RGB、语义和碰撞三类标准预览？

### 物理和坐标

15. 地板、墙壁、support surface 的碰撞信息是否属于制造方正式交付范围？
16. 每个资产的坐标轴、原点、单位和推荐放置高度在哪里定义？
17. 关节家具是否提供正式关节范围和零位？

## 10. 建议的会议结论输出

会议结束前，建议双方至少形成以下明确结论：

- 唯一 canonical 资产目录。
- 下一版资产 manifest 格式。
- RGB、PBR、semantic 文件命名和绑定规范。
- 平面纹理库目标数量。
- 首批需要支持 appearance randomization 的家具/刚体清单。
- HDR 扩充数量和标定方式。
- 物理碰撞和 support surface 的责任边界。
- 制造方交付日期、使用方验证日期和问题回传格式。

## 11. 当前使用方已经完成的临时修复

这些修改用于当前版本能够观察和录制，不应替代制造方从源资产解决问题：

| 临时修复 | 文件 | 作用 |
|---|---|---|
| 路径和墙面方向归一化 | `scripts/simbox/normalize_bench_paths.py` | 修复任务路径、HDR/机器人路径和墙面法线 |
| 规范 USD RGB 绑定修复 | `scripts/simbox/repair_bench21_material_bindings.py` | 将错误 semantic 绑定恢复为明确 RGB 图 |
| 平面纹理刷新 | `scripts/simbox/refresh_bench21_plane_textures.py` | 无损重编码，规避 Isaac/Hydra 黑纹理问题 |
| 观察配置生成 | `scripts/simbox/prepare_bench_observe_config.py` | 动态相机名和无破坏 wait 录制配置 |
| 平面碰撞实现 | `workflows/simbox/core/objects/plane_object.py` | 为地板和墙壁增加可选静态薄盒碰撞 |

制造方下一版本如果从源头修正路径、默认材质、纹理编码和物理元数据，我们应逐项取消或简化这些补丁，避免长期维护两套事实来源。
