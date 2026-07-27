# USD Stage 结构证据摘录：标准 SimBox pickable vs benchmark1.0 小物体

> 由 `workflows/simbox/tools/rigid_obj/dump_usd_stage_md.py` 生成；脚本只读检查 USD，不会改写资产文件。

这份文档是证据附录，不是主解释文档。阅读时建议先看每个文件的 `Stage 顶层总览` 和 `本文件的快速读法`，再往下看 Prim 树、物理标记和具体属性。

## `InternDataAssets/assets/art/heat_the_food_in_the_microwave/pick_objs/omniobject3d-bread_090/Aligned_obj.usd`

### Stage 顶层总览

- Default prim（默认入口）: `/World`
- Root layer（根 layer）: `InternDataAssets/assets/art/heat_the_food_in_the_microwave/pick_objs/omniobject3d-bread_090/Aligned_obj.usd`
- Up axis（上方向）: `Y`
- Meters per unit（单位换算）: `0.01`
- Start/end time code（时间范围）: `0.0` / `0.0`
- Root prims（最顶层 prim）: `/World`

### 本文件的快速读法

- 从顶层看，默认入口是 `/World`。运行时代码如果从 default prim 开始找对象，第一步看到的就是这一层。
- `/World` 下面的直接 child 是：`/World/Looks`, `/World/Aligned`, `/World/Physics_Materials`。这决定了硬编码脚本用 `GetAllChildren()[1]` 时能不能取到目标。
- 存在 `/World/Aligned`：这更接近 SimBox 标准 pickable 的组织方式，`Aligned` 是物体主体，刚体、质量和下游 `prim_path_child: Aligned` 都围绕它建立。
- 已经看到刚体相关 API schema，说明至少有某个 prim 被声明为物理刚体主体。
- 已经看到碰撞相关 API schema，但还要继续看它加在 mesh 上还是加在对象主体上，以及是否有 convex decomposition 等细节。

### 同目录旁路文件

这些文件不是 USD stage 的内部 prim，但它们常常决定资产能不能被 SimBox 的抓取、材质、碰撞生成流程继续使用。

- 共 `6` 项。
- `Aligned.mtl`
- `Aligned.obj`
- `Aligned_grasp_sparse.npy`
- `Aligned_obj.usd`
- `Aligned_sim.png`
- `textures`

### Prim 树

| Prim 路径 | 类型 | 启用 | 已定义 | 已应用 API schemas | References 引用 | Payload 负载 |
| --- | --- | --- | --- | --- | --- | --- |
| `/World` | Xform | 是 | 是 | - | - | - |
| &nbsp;&nbsp;`/World/Looks` | Scope | 是 | 是 | - | - | - |
| &nbsp;&nbsp;&nbsp;&nbsp;`/World/Looks/material_texture_001` | Material | 是 | 是 | - | - | - |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`/World/Looks/material_texture_001/material_texture_001` | Shader | 是 | 是 | NodeDefAPI | - | - |
| &nbsp;&nbsp;`/World/Aligned` | Xform | 是 | 是 | PhysicsRigidBodyAPI, PhysicsMassAPI, MaterialBindingAPI | - | - |
| &nbsp;&nbsp;&nbsp;&nbsp;`/World/Aligned/cake_cake_001` | Mesh | 是 | 是 | MaterialBindingAPI, PhysicsCollisionAPI, PhysicsMeshCollisionAPI | - | - |
| &nbsp;&nbsp;`/World/Physics_Materials` | Material | 是 | 是 | PhysicsMaterialAPI | - | - |

### 物理 / 材质 / 碰撞标记

| Prim 路径 | 类型 | 刚体 API | 质量 API | 碰撞 API | Mesh 碰撞 API | 物理材质 API | 材质绑定 API |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `/World` | Xform | - | - | - | - | - | - |
| `/World/Looks` | Scope | - | - | - | - | - | - |
| `/World/Looks/material_texture_001` | Material | - | - | - | - | - | - |
| `/World/Looks/material_texture_001/material_texture_001` | Shader | - | - | - | - | - | - |
| `/World/Aligned` | Xform | 是 | 是 | - | - | - | 是 |
| `/World/Aligned/cake_cake_001` | Mesh | - | - | 是 | 是 | - | 是 |
| `/World/Physics_Materials` | Material | - | - | - | - | 是 | - |

### Mesh 几何摘要

| Mesh prim | 点数 | 面数 | 局部包围盒中心 | 局部包围盒尺寸 | 世界包围盒 min | 世界包围盒 max | 材质目标 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `/World/Aligned/cake_cake_001` | 21880 | 43764 | (0.0, 0.0, 0.0) | (45.753487, 43.930195, 114.177643) | (-0.022877, -0.021965, -0.057089) | (0.022877, 0.021965, 0.057089) | /World/Looks/material_texture_001 |

### Prim 细节

#### `/World`

- 类型: `Xform`
- 已应用 API schemas: `-`
- Prim stack（这个 prim 的来源 layer）:
  - `InternDataAssets/assets/art/heat_the_food_in_the_microwave/pick_objs/omniobject3d-bread_090/Aligned_obj.usd:/World`
- 已写入 metadata:
  - `kind`: `'component'`
- 已写入 attributes:
  - `xformOp:orient` `quatf` = `Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))`
  - `xformOp:scale` `float3` = `[1.0, 1.0, 1.0]`
  - `xformOp:translate` `double3` = `[0.0, 0.0, 0.0]`
  - `xformOpOrder` `token[]` = `['xformOp:translate', 'xformOp:orient', 'xformOp:scale']`

#### `/World/Looks`

- 类型: `Scope`
- 已应用 API schemas: `-`
- Prim stack（这个 prim 的来源 layer）:
  - `InternDataAssets/assets/art/heat_the_food_in_the_microwave/pick_objs/omniobject3d-bread_090/Aligned_obj.usd:/World/Looks`

#### `/World/Looks/material_texture_001`

- 类型: `Material`
- 已应用 API schemas: `-`
- Prim stack（这个 prim 的来源 layer）:
  - `InternDataAssets/assets/art/heat_the_food_in_the_microwave/pick_objs/omniobject3d-bread_090/Aligned_obj.usd:/World/Looks/material_texture_001`

#### `/World/Looks/material_texture_001/material_texture_001`

- 类型: `Shader`
- 已应用 API schemas: `NodeDefAPI`
- Prim stack（这个 prim 的来源 layer）:
  - `InternDataAssets/assets/art/heat_the_food_in_the_microwave/pick_objs/omniobject3d-bread_090/Aligned_obj.usd:/World/Looks/material_texture_001/material_texture_001`
- 已写入 attributes:
  - `inputs:diffuse_color_constant` `color3f` = `[0.800000011920929, 0.800000011920929, 0.800000011920929]`
  - `inputs:diffuse_texture` `asset` = `asset=./textures/Scan.jpg, resolved=/home/bld/ykqin/InternDataEngine/InternDataAssets/assets/art/heat_the_food_in_the_microwave/pick_objs/omniobject3d-bread_090/textures/Scan.jpg`
  - `inputs:emissive_color` `color3f` = `[1.0, 1.0, 1.0]`
  - `inputs:emissive_intensity` `float` = `10000.0`
  - `inputs:enable_emission` `bool` = `False`
  - `inputs:enable_opacity` `bool` = `False`
  - `inputs:enable_opacity_texture` `bool` = `False`
  - `inputs:opacity_constant` `float` = `1.0`
  - `inputs:opacity_mode` `int` = `1`
  - `inputs:opacity_threshold` `float` = `0.0`
  - `inputs:texture_rotate` `float` = `0.0`
  - `inputs:texture_scale` `float2` = `[1.0, 1.0]`
  - `inputs:texture_translate` `float2` = `[0.0, 0.0]`

#### `/World/Aligned`

- 类型: `Xform`
- 已应用 API schemas: `PhysicsRigidBodyAPI, PhysicsMassAPI, MaterialBindingAPI`
- Prim stack（这个 prim 的来源 layer）:
  - `InternDataAssets/assets/art/heat_the_food_in_the_microwave/pick_objs/omniobject3d-bread_090/Aligned_obj.usd:/World/Aligned`
- 已写入 metadata:
  - `apiSchemas`: `explicitItems=['PhysicsRigidBodyAPI', 'PhysxRigidBodyAPI', 'PhysicsMassAPI', 'MaterialBindingAPI']`
- 已写入 relationships:
  - `material:binding:physics` -> `/World/Physics_Materials`
- 已写入 attributes:
  - `physics:density` `float` = `295.00299072265625`
  - `physics:kinematicEnabled` `bool` = `False`
  - `physics:mass` `float` = `0.5`
  - `physics:rigidBodyEnabled` `bool` = `True`
  - `xformOp:orient` `quatf` = `Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))`
  - `xformOp:scale` `float3` = `[0.0010000000474974513, 0.0010000000474974513, 0.0010000000474974513]`
  - `xformOp:translate` `double3` = `[0.0, 0.0, 0.0]`
  - `xformOpOrder` `token[]` = `['xformOp:translate', 'xformOp:orient', 'xformOp:scale']`

#### `/World/Aligned/cake_cake_001`

- 类型: `Mesh`
- 已应用 API schemas: `MaterialBindingAPI, PhysicsCollisionAPI, PhysicsMeshCollisionAPI`
- Prim stack（这个 prim 的来源 layer）:
  - `InternDataAssets/assets/art/heat_the_food_in_the_microwave/pick_objs/omniobject3d-bread_090/Aligned_obj.usd:/World/Aligned/cake_cake_001`
- 已写入 metadata:
  - `apiSchemas`: `explicitItems=['MaterialBindingAPI', 'PhysicsCollisionAPI', 'PhysxCollisionAPI', 'PhysxConvexDecompositionCollisionAPI', 'PhysicsMeshCollisionAPI']`
- 已写入 relationships:
  - `material:binding` -> `/World/Looks/material_texture_001`
- 已写入 attributes:
  - `extent` `float3[]` = `[[-22.87674331665039, -21.965097427368164, -57.08882141113281], [22.87674331665039, 21.965097427368164, 57.08882141113281]]`
  - `faceVertexCounts` `int[]` = `[3, 3, 3, 3, 3, 3, 3, 3, ... （共 43764 项）]`
  - `faceVertexIndices` `int[]` = `[2, 14, 7, 8, 15, 19, 1, 2, ... （共 131292 项）]`
  - `physics:approximation` `token` = `'convexDecomposition'`
  - `physics:collisionEnabled` `bool` = `True`
  - `physxConvexDecompositionCollision:errorPercentage` `float` = `0.10000000149011612`
  - `physxConvexDecompositionCollision:hullVertexLimit` `int` = `64`
  - `physxConvexDecompositionCollision:maxConvexHulls` `int` = `64`
  - `physxConvexDecompositionCollision:minThickness` `float` = `0.0010000000474974513`
  - `physxConvexDecompositionCollision:shrinkWrap` `bool` = `True`
  - `points` `point3f[]` = `[[-20.332443237304688, -20.42137336730957, -41.573020935058594], [21.51070785522461, 9.08806037902832, 42.293399810791016], [21.30767822265625, 9.461625099182129, 42.30196762084961], [21.65158462524414, 9.854989051818848, 41.60628128051758], [-20.69239044189453, 2.145024061203003, -5.000486850738525], [-21.254730224609375, -18.349037170410156, -41.28447723388672], [21.214038848876953, 9.234217643737793, 42.89003372192383], [20.94717025756836, 10.137353897094727, 42.8556022644043], ... （共 21880 项）]`
  - `subdivisionScheme` `token` = `'none'`

#### `/World/Physics_Materials`

- 类型: `Material`
- 已应用 API schemas: `PhysicsMaterialAPI`
- Prim stack（这个 prim 的来源 layer）:
  - `InternDataAssets/assets/art/heat_the_food_in_the_microwave/pick_objs/omniobject3d-bread_090/Aligned_obj.usd:/World/Physics_Materials`
- 已写入 metadata:
  - `apiSchemas`: `explicitItems=['PhysicsMaterialAPI']`
- 已写入 attributes:
  - `physics:dynamicFriction` `float` = `1.0`
  - `physics:staticFriction` `float` = `1.0`

## `InternDataAssets/benchmark1.0/scene/file_4/assets/small_usd/mesh_small_box_0_0000_mesh_0_00.usd`

### Stage 顶层总览

- Default prim（默认入口）: `/Asset`
- Root layer（根 layer）: `InternDataAssets/benchmark1.0/scene/file_4/assets/small_usd/mesh_small_box_0_0000_mesh_0_00.usd`
- Up axis（上方向）: `Y`
- Meters per unit（单位换算）: `0.01`
- Start/end time code（时间范围）: `0.0` / `0.0`
- Root prims（最顶层 prim）: `/Asset`

### 本文件的快速读法

- 从顶层看，默认入口是 `/Asset`。运行时代码如果从 default prim 开始找对象，第一步看到的就是这一层。
- `/Asset` 下面的直接 child 是：`/Asset/Geometry`。这决定了硬编码脚本用 `GetAllChildren()[1]` 时能不能取到目标。
- 存在 `/Asset/Geometry` 但没有 `/World/Aligned`：这更像场景数据集里的单个 mesh 片段，能显示或被场景引用，但还不是 SimBox task loader 期待的 pickable 资产。
- 没有看到刚体相关 API schema；即使 mesh 上有碰撞信息，也不等于整个对象已经会作为动态刚体参与仿真。
- 已经看到碰撞相关 API schema，但还要继续看它加在 mesh 上还是加在对象主体上，以及是否有 convex decomposition 等细节。

### 同目录旁路文件

这些文件不是 USD stage 的内部 prim，但它们常常决定资产能不能被 SimBox 的抓取、材质、碰撞生成流程继续使用。

- 共 `112` 项。
- `mesh_small_alarm_clock_1_0006_mesh_0_00.usd`
- `mesh_small_board_games_1136_0045_mesh_0_00.usd`
- `mesh_small_board_games_1137_0046_mesh_0_00.usd`
- `mesh_small_board_games_1211_0063_mesh_0_00.usd`
- `mesh_small_board_games_1212_0064_mesh_0_00.usd`
- `mesh_small_books_1133_0042_mesh_0_00.usd`
- `mesh_small_books_1134_0043_mesh_0_00.usd`
- `mesh_small_books_1209_0061_mesh_0_00.usd`
- `mesh_small_books_1210_0062_mesh_0_00.usd`
- `mesh_small_books_1213_0065_mesh_0_00.usd`
- `mesh_small_books_1214_0066_mesh_0_00.usd`
- `mesh_small_books_1215_0067_mesh_0_00.usd`
- `mesh_small_books_1216_0068_mesh_0_00.usd`
- `mesh_small_books_1217_0069_mesh_0_00.usd`
- `mesh_small_books_1218_0070_mesh_0_00.usd`
- `mesh_small_books_1219_0071_mesh_0_00.usd`
- `mesh_small_books_1220_0072_mesh_0_00.usd`
- `mesh_small_books_1221_0073_mesh_0_00.usd`
- `mesh_small_books_1222_0074_mesh_0_00.usd`
- `mesh_small_books_1223_0075_mesh_0_00.usd`
- `mesh_small_books_1224_0076_mesh_0_00.usd`
- `mesh_small_books_1225_0077_mesh_0_00.usd`
- `mesh_small_books_1226_0078_mesh_0_00.usd`
- `mesh_small_box_0_0000_mesh_0_00.usd`
- `mesh_small_box_1110_0019_mesh_0_00.usd`
- `mesh_small_box_1204_0056_mesh_0_00.usd`
- `mesh_small_box_1230_0082_mesh_0_00.usd`
- `mesh_small_box_1231_0083_mesh_0_00.usd`
- `mesh_small_cable_pouch_3_0008_mesh_0_00.usd`
- `mesh_small_cereal_boxes_1232_0084_mesh_0_00.usd`
- `mesh_small_charging_cable_2_0007_mesh_0_00.usd`
- `mesh_small_cone_0_0002_mesh_0_00.usd`
- `mesh_small_cone_1108_0017_mesh_0_00.usd`
- `mesh_small_cone_1112_0021_mesh_0_00.usd`
- `mesh_small_cone_1135_0044_mesh_0_00.usd`
- `mesh_small_cone_1140_0049_mesh_0_00.usd`
- `mesh_small_cone_1141_0050_mesh_0_00.usd`
- `mesh_small_cone_1208_0060_mesh_0_00.usd`
- `mesh_small_cone_1233_0085_mesh_0_00.usd`
- `mesh_small_cone_1234_0086_mesh_0_00.usd`
- ... 其余 `72` 项已省略，避免这份证据文档被目录清单淹没。

### Prim 树

| Prim 路径 | 类型 | 启用 | 已定义 | 已应用 API schemas | References 引用 | Payload 负载 |
| --- | --- | --- | --- | --- | --- | --- |
| `/Asset` | Xform | 是 | 是 | - | - | - |
| &nbsp;&nbsp;`/Asset/Geometry` | Mesh | 是 | 是 | MaterialBindingAPI, PhysicsCollisionAPI | - | - |

### 物理 / 材质 / 碰撞标记

| Prim 路径 | 类型 | 刚体 API | 质量 API | 碰撞 API | Mesh 碰撞 API | 物理材质 API | 材质绑定 API |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `/Asset` | Xform | - | - | - | - | - | - |
| `/Asset/Geometry` | Mesh | - | - | 是 | - | - | 是 |

### Mesh 几何摘要

| Mesh prim | 点数 | 面数 | 局部包围盒中心 | 局部包围盒尺寸 | 世界包围盒 min | 世界包围盒 max | 材质目标 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `/Asset/Geometry` | 8 | 12 | (5.35, 5.956, 0.461) | (0.3, 0.2, 0.12) | (5.2, 5.856, 0.401) | (5.5, 6.056, 0.521) | /World/Looks/small_mesh_small_box_0_mat（当前 stage 中找不到目标 prim） |

### Prim 细节

#### `/Asset`

- 类型: `Xform`
- 已应用 API schemas: `-`
- Prim stack（这个 prim 的来源 layer）:
  - `InternDataAssets/benchmark1.0/scene/file_4/assets/small_usd/mesh_small_box_0_0000_mesh_0_00.usd:/Asset`

#### `/Asset/Geometry`

- 类型: `Mesh`
- 已应用 API schemas: `MaterialBindingAPI, PhysicsCollisionAPI`
- Prim stack（这个 prim 的来源 layer）:
  - `InternDataAssets/benchmark1.0/scene/file_4/assets/small_usd/mesh_small_box_0_0000_mesh_0_00.usd:/Asset/Geometry`
- 已写入 metadata:
  - `apiSchemas`: `explicitItems=['MaterialBindingAPI', 'PhysicsCollisionAPI']`
- 已写入 relationships:
  - `material:binding` -> `/World/Looks/small_mesh_small_box_0_mat（当前 stage 中找不到目标 prim）`
- 已写入 attributes:
  - `faceVertexCounts` `int[]` = `[3, 3, 3, 3, 3, 3, 3, 3, ... （共 12 项）]`
  - `faceVertexIndices` `int[]` = `[1, 3, 0, 4, 1, 0, 0, 3, ... （共 36 项）]`
  - `points` `point3f[]` = `[[5.199999809265137, 6.056000232696533, 0.4009999930858612], [5.199999809265137, 5.855999946594238, 0.4009999930858612], [5.199999809265137, 6.056000232696533, 0.5210000276565552], [5.199999809265137, 5.855999946594238, 0.5210000276565552], [5.5, 6.056000232696533, 0.4009999930858612], [5.5, 5.855999946594238, 0.4009999930858612], [5.5, 6.056000232696533, 0.5210000276565552], [5.5, 5.855999946594238, 0.5210000276565552]]`
