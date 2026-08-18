# USD 结构对比：为什么 benchmark1.0 小物体不能直接当 SimBox pickable 用

本文对比两个真实 USD 文件：

- 标准 SimBox pickable:
  `InternDataAssets/assets/art/heat_the_food_in_the_microwave/pick_objs/omniobject3d-bread_090/Aligned_obj.usd`
- benchmark1.0 小物体:
  `InternDataAssets/benchmark1.0/scene/file_4/assets/small_usd/mesh_small_box_0_0000_mesh_0_00.usd`

先给一个总判断：这两个文件都能被 `pxr.Usd` 打开，也都包含 mesh，但它们不是同一种“资产产品”。标准 SimBox pickable 是为了机器人任务重新整理过的单物体资产；benchmark1.0 这个小物体更像场景包里的一个 mesh 零件。前者已经把“对象主体、物理刚体、碰撞、材质、抓取标注、YAML 引用路径”整理到同一套约定里，后者还停留在“能被场景引用和显示”的层级。

所以这次 `make_rigid.py` 报错，不是因为 USD 坏了，也不是因为 Python 环境坏了，而是因为我们拿一个场景数据集里的 mesh 片段，去跑一个默认面向 SimBox pickable 结构的半自动脚本。

## 先从顶层看

理解 USD 时，最好先不要一上来盯着某个属性，比如 `physics:mass` 或 `MaterialBindingAPI`。更稳的顺序是：

1. 这个 USD 的默认入口是谁，也就是 `default prim`。
2. 默认入口下面第一层有哪些 child。
3. 哪一层代表“整个对象主体”，哪一层只是“几何网格”。
4. 物理刚体、质量、碰撞、材质分别挂在哪个 prim 上。
5. 这个文件旁边有没有 SimBox 下游代码会按约定寻找的旁路文件，例如 `Aligned_obj.obj` 和 `Aligned_grasp_sparse.npy`。

这五步像是先看地图，再看街道。USD 里的 prim 很多，属性也很多；如果先从属性开始看，很容易迷路。对 SimBox 来说，最重要的不是“文件里有没有一个 mesh”，而是“运行时 loader 能不能按约定找到对象主体和相关标注”。

## 标准 pickable 的顶层结构

标准文件的顶层长这样：

```text
default prim: /World

/World
├── /World/Looks
├── /World/Aligned
└── /World/Physics_Materials
```

这三个 child 分工很清楚：

- `/World/Looks` 管视觉材质。它告诉渲染系统 mesh 应该用什么贴图、什么 shader。
- `/World/Aligned` 是物体主体。SimBox 的 YAML 里常见 `prim_path_child: Aligned`，说的就是这一层。
- `/World/Physics_Materials` 管物理材质，比如摩擦系数。

继续往下展开，真正的 mesh 在 `Aligned` 下面：

```text
/World
└── /World/Aligned
    └── /World/Aligned/cake_cake_001
```

这件事很关键。SimBox 不是直接把 mesh 当作任务对象，而是把 `/World/Aligned` 当作“可移动物体的主体”，再把 mesh 放在主体下面。这样做有一个好处：物理刚体可以挂在对象主体上，碰撞可以挂在 mesh 上，材质可以独立挂在 `Looks` 上，运行时代码仍然能通过一个稳定名字 `Aligned` 把它们串起来。

可以粗略理解成：

```text
/World/Aligned
  是“这个物体”的外壳和运动主体

/World/Aligned/<mesh>
  是“这个物体长什么样、怎么碰撞”的几何细节
```

这个标准文件同目录还有：

```text
Aligned.mtl
Aligned.obj
Aligned_grasp_sparse.npy
Aligned_obj.usd
Aligned_sim.png
textures/
```

这些旁路文件不是摆设。`Aligned_obj.usd` 负责在仿真里加载物体，`Aligned_obj.obj` 常用于几何处理或抓取生成流程，`Aligned_grasp_sparse.npy` 是 pick 技能要用的抓取候选。也就是说，标准 pickable 不是一个孤立 USD，而是一组互相约定好名字的文件包。

## benchmark 小物体的顶层结构

benchmark 文件的顶层长这样：

```text
default prim: /Asset

/Asset
└── /Asset/Geometry
```

它非常简洁：一个 `/Asset`，下面一个 `/Asset/Geometry` mesh。这个结构对场景数据集来说并不奇怪。场景里的 `scene.usd` 可以引用很多这样的局部资产，把桌子、盒子、书、杯子等对象拼成一个完整场景。

但它和 SimBox pickable 的差别也正是在这里：

```text
标准 pickable:
  /World
  ├── /World/Looks
  ├── /World/Aligned
  │   └── mesh
  └── /World/Physics_Materials

benchmark 小物体:
  /Asset
  └── /Asset/Geometry
```

benchmark 不是“完全没有物理信息”。它的 `/Asset/Geometry` 上已经能看到 `PhysicsCollisionAPI`。这说明它至少有一些碰撞相关声明。可是，对 SimBox pickable 来说，这还不够，因为它缺少几层更高层的契约：

- 没有 `/World` 作为标准入口。
- 没有 `/World/Aligned` 作为对象主体。
- 没有把 `RigidBodyAPI` 和 `MassAPI` 放到对象主体上。
- 没有 `/World/Physics_Materials` 这类物理材质 prim。
- 没有标准目录里的 `Aligned_obj.obj`。
- 没有 `Aligned_grasp_sparse.npy` 抓取标注。

另外，它的 mesh 材质绑定目标指向 `/World/Looks/...`，但在单独打开这个小 USD 时，这个目标在当前 stage 里并不存在。这通常意味着它原本更依赖完整场景 package 的组合关系，而不是作为一个完整的、独立的 pickable 资产使用。

## 为什么 `make_rigid.py` 会报 `IndexError`

`make_rigid.py` 的关键逻辑是：

```python
root_prim = stage.GetDefaultPrim()
aligned_prim = root_prim.GetAllChildren()[1]
editor.RenamePrim(aligned_prim, "Aligned")
aligned_prim = stage.GetPrimAtPath("/World/Aligned")
UsdPhysics.RigidBodyAPI.Apply(aligned_prim)
UsdPhysics.MassAPI.Apply(aligned_prim)
```

这段代码背后有一个很强的隐含假设：它以为文件大概长这样：

```text
/World
├── /World/Looks
└── /World/<某个物体主体>
```

所以它直接取：

```python
root_prim.GetAllChildren()[1]
```

也就是默认入口下面的第 2 个 child，然后把它改名成 `Aligned`。

标准 pickable 里，`/World` 下面确实至少有多个 child，第 2 个通常就是物体主体：

```text
/World
├── child[0] = /World/Looks
├── child[1] = /World/Aligned
└── child[2] = /World/Physics_Materials
```

但 benchmark 文件里只有一个 child：

```text
/Asset
└── child[0] = /Asset/Geometry
```

所以 Python 在执行下面这句时：

```python
root_prim.GetAllChildren()[1]
```

实际是在问：“请给我第 2 个 child。”
可是列表里只有第 1 个 child，于是就报：

```text
IndexError: list index out of range
```

这就是根因。脚本没有先判断 child 数量，也没有搜索 mesh，更没有适配 `/Asset/Geometry` 这种 benchmark 结构。它不是通用 USD 修复器，而是针对某种预期结构写的工具。

## 这个差异和 YAML 有什么关系

SimBox 任务 YAML 里常见这样的对象配置：

```yaml
objects:
  -
    name: pick_object_left
    path: pick_and_place/pre-train-pick/assets/omniobject3d-banana/omniobject3d-banana_001/Aligned_obj.usd
    target_class: RigidObject
    prim_path_child: Aligned
```

这里的 `path` 和 `prim_path_child` 是配套的。

`RigidObject` 会先把 `Aligned_obj.usd` 引用到任务场景里的某个对象根路径下，例如：

```text
/World/envs/env_0/pick_object_left
```

然后它会继续找：

```text
/World/envs/env_0/pick_object_left/Aligned
```

因为 YAML 写了：

```yaml
prim_path_child: Aligned
```

所以，如果 USD 内部没有 `Aligned` 这一层，loader 就算成功引用了 USD，也找不到它认为的“刚体主体”。这就是为什么 `/World/Aligned` 不只是一个名字好看的目录层级，而是 YAML、loader、物理对象之间的连接点。

反过来看 benchmark 文件，它现在的自然结构是：

```text
/Asset
└── /Asset/Geometry
```

如果不做重包装，YAML 里直接写 `prim_path_child: Aligned` 就对不上；如果改成 `prim_path_child: Geometry`，又绕开了现有 pickable 工具链的很多默认假设，后续碰撞、质量、抓取标注、文件命名都会继续出现不一致。短期也许能 hack 过去，长期维护会很别扭。

## 为什么不建议直接改 benchmark 原文件

benchmark1.0 的 `scene/file_x` 更像一个场景包。里面的 `scene.usd`、`assets/*.usd`、`assets/small_usd/*.usd`、`textures/*.png` 是互相配合的。原始小物体 USD 可能还被场景引用，材质路径和对象位置也可能依赖场景 package 的组织方式。

如果直接在原文件上跑转换脚本，有几个风险：

- 会污染原始数据，后面不好判断问题来自数据本身还是我们的改动。
- 可能破坏 `scene.usd` 对这个 asset 的引用预期。
- 即使加上刚体，也仍然没有 `Aligned_obj.obj` 和 `Aligned_grasp_sparse.npy`。
- 如果脚本按 `/World/Aligned` 写死路径，而文件实际是 `/Asset/Geometry`，可能改到错误位置或直接失败。

更稳妥的思路是：保留 benchmark 原文件，把它复制或引用到一个新的 SimBox pickable 目录里，再在那里做重包装。

## 推荐的转换目标

第一轮目标可以是新建一个标准资产目录：

```text
InternDataAssets/assets/benchmark1.0/pickables/box/file_4_box_0000/
├── Aligned_obj.usd
├── Aligned_obj.obj
├── Aligned_grasp_sparse.npy
└── source.txt
```

其中 `source.txt` 记录原始来源，例如：

```text
InternDataAssets/benchmark1.0/scene/file_4/assets/small_usd/mesh_small_box_0_0000_mesh_0_00.usd
```

USD 内部目标结构应该逐步靠近：

```text
/World
├── /World/Looks
├── /World/Aligned
│   └── /World/Aligned/Geometry
└── /World/Physics_Materials
```

第一步可以只做最小重包装：

```text
/Asset/Geometry
```

变成：

```text
/World
└── /World/Aligned
    └── /World/Aligned/Geometry
```

等这个结构能被 loader 找到之后，再补：

- `RigidBodyAPI` 和 `MassAPI`，让 `/World/Aligned` 成为刚体主体。
- `CollisionAPI`、`MeshCollisionAPI` 和 convex decomposition 参数，让 mesh 接触更稳定。
- `PhysicsMaterialAPI`，补摩擦等物理材质。
- `Aligned_obj.obj`，给后续几何处理和抓取生成工具使用。
- `Aligned_grasp_sparse.npy`，让 pick 技能能找到抓取候选。

这个顺序的好处是每一步都能解释清楚：先让结构对上，再让物理对上，最后让抓取标注对上。

## 两份文档如何配合看

本文是主解释文档，重点是讲清楚来龙去脉。

详细证据在：

```text
docs/benchmark_vs_standard_usd_stage_dump.md
```

那份文档是用 `pxr.Usd` 从真实文件里导出的 stage 摘要。它更像证据附录：里面有 default prim、Prim 树、API schema、mesh 点面数、材质绑定、已写入属性等信息。阅读时建议先看每个文件的“Stage 顶层总览”和“本文件的快速读法”，再看下面的表格。

## 一句话总结

`make_rigid.py` 报错的直接原因是 benchmark 文件的 `/Asset` 下面只有一个 child，脚本却硬取第 2 个 child。更深层原因是：benchmark 小物体是场景包里的 mesh 片段，而 SimBox pickable 需要的是围绕 `/World/Aligned`、物理属性、碰撞、OBJ 和抓取标注整理好的任务资产包。
