# Benchmark 1.0 资产统计文档

本文档基于本地目录 `InternDataAssets/benchmark1.0` 生成，统计时间为 2026-05-21。

## 统计范围

当前统计的是本地的 Benchmark 1.0 资产包：

```text
InternDataAssets/benchmark1.0/
  render_image/
  render_params/
  scene/
    file_2/
    file_3/
    ...
    file_11/
```

目前一共有 10 个场景目录：`file_2` 到 `file_11`。

## 场景结构

每个场景目录大致是这个形态：

```text
scene/file_x/
  scene.usd
  scene.json
  manifest.json
  repair_conversion_manifest.json
  assets/
    *.usd
    *_texture.png
    small_usd/
      *.usd
  textures/
    *.png
```

`scene.usd` 是可以直接加载的场景入口。它是一个固定布局的完整 USD stage，里面包含 `/World`、地板、墙体、墙体碰撞、灯光、相机、大物体实例和小物体实例。

但它不是“单文件自包含”。`scene.usd` 会通过 USD reference 组合当前场景目录下的其他资源：

```text
/World/Objects      -> 引用 assets/*.usd
/World/SmallAssets  -> 引用 assets/small_usd/*.usd
/World/Looks        -> 引用 textures/*.png 和本地材质
```

所以移动或拷贝场景时，应该保留整个 `file_x/` 目录，而不是只拷贝一个 `scene.usd`。

## 总体统计

| 指标 | 数量 |
|---|---:|
| 场景数 | 10 |
| 被 `scene.usd` 实际引用的大物体实例 | 373 |
| 被 `scene.usd` 实际引用的小物体实例 | 1184 |
| 被实际引用的物体实例总数 | 1557 |
| 目录中存在的大物体 USD 文件 | 411 |
| 目录中存在的小物体 USD 文件 | 1403 |
| 目录中存在的 USD 资产文件总数 | 1814 |
| 粗略大物体类别数 | 296 |
| 粗略小物体类别数 | 350 |

类别数是根据文件名粗略归并出来的，例如去掉 instance/index/id 这类后缀。它适合用来快速了解资产分布，但不能当作严格 ontology。

## 每个场景的统计

说明：

- `大物体实例` 和 `小物体实例`：`scene.usd` 里实际挂在 `/World/Objects` 和 `/World/SmallAssets` 下的对象数量。
- `大物体文件` 和 `小物体文件`：场景目录下实际存在的 USD 文件数量。
- 有些 USD 文件存在于 `assets/` 中，但当前 `scene.usd` 未必实例化它们。

| 场景 | 大物体实例 | 大物体文件 | 大物体类别 | 小物体实例 | 小物体文件 | 小物体类别 | USD references | prims | 地板 | 墙体 | 墙体碰撞 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| file_2 | 40 | 47 | 42 | 175 | 215 | 44 | 215 | 1604 | 7 | 94 | 91 |
| file_3 | 38 | 42 | 41 | 89 | 100 | 30 | 127 | 1068 | 7 | 92 | 69 |
| file_4 | 28 | 35 | 32 | 93 | 112 | 39 | 121 | 943 | 6 | 76 | 70 |
| file_5 | 32 | 35 | 34 | 55 | 60 | 29 | 87 | 888 | 6 | 101 | 90 |
| file_6 | 32 | 36 | 35 | 104 | 125 | 61 | 136 | 1011 | 6 | 83 | 66 |
| file_7 | 38 | 40 | 40 | 259 | 277 | 45 | 297 | 1773 | 7 | 84 | 63 |
| file_8 | 48 | 50 | 46 | 162 | 192 | 70 | 210 | 1603 | 8 | 123 | 109 |
| file_9 | 39 | 42 | 33 | 94 | 116 | 32 | 133 | 1259 | 7 | 81 | 78 |
| file_10 | 37 | 40 | 37 | 97 | 141 | 60 | 134 | 1259 | 8 | 117 | 96 |
| file_11 | 41 | 44 | 40 | 56 | 65 | 61 | 97 | 1009 | 6 | 91 | 71 |

## 转换/修复统计

`scene/conversion_summary.json` 中记录：

```text
total_done: 154
total_fail: 0
```

每个场景的修复/转换条目如下：

| 场景 | 完成 | 失败 |
|---|---:|---:|
| file_2 | 7 | 0 |
| file_3 | 11 | 0 |
| file_4 | 4 | 0 |
| file_5 | 9 | 0 |
| file_6 | 27 | 0 |
| file_7 | 26 | 0 |
| file_8 | 31 | 0 |
| file_9 | 19 | 0 |
| file_10 | 9 | 0 |
| file_11 | 11 | 0 |

## 高频类别

出现频率较高的大物体类别：

| 类别 | 数量 |
|---|---:|
| nightstand | 8 |
| nightstand_01 | 7 |
| dining_chair | 6 |
| ergonomic_task_chair | 6 |
| meeting_chair | 6 |
| desk_chair | 5 |
| floor_lamp | 5 |
| full_length_mirror | 4 |
| nightstand_hinged | 4 |
| floor_lamp_01 | 4 |
| conference_chair | 4 |
| bookcase_low | 3 |
| coffee_table | 3 |
| compact_writing_desk | 3 |
| entry_console_table | 3 |
| study_desk | 3 |
| kitchen_base_cabinet_run | 3 |
| side_table | 3 |
| sideboard | 3 |
| umbrella_stand_01 | 3 |
| meeting_table | 3 |
| upholstered_dining_chair_01 | 3 |

出现频率较高的小物体类别：

| 类别 | 数量 |
|---|---:|
| book | 75 |
| tray | 48 |
| books_assorted | 46 |
| pen | 46 |
| paper_clips | 45 |
| cone | 30 |
| cup | 28 |
| paper_clip | 24 |
| tea_bag | 22 |
| sugar_packets | 20 |
| napkins | 20 |
| mug | 19 |
| notebook | 18 |
| books | 18 |
| catchall_tray | 17 |
| remote | 17 |
| jar | 17 |
| thread_spools | 17 |
| sample_cards | 15 |
| bowl | 14 |
| sugar_packet | 14 |
| marker | 13 |
| clothes_hangers | 12 |
| board_game_box | 11 |
| serving_tray | 11 |
| shoes | 10 |
| thread_spool | 10 |
| reading_lamp | 9 |
| small_cup | 9 |
| table_lamp | 9 |
| teapot | 9 |
| board_games | 9 |
| box | 9 |
| tea_bags | 9 |
| knife | 9 |

## 推荐先尝试的资产

建议先从形状简单、接近盒状、不太薄的小物体开始。不要一上来就选杯子、碗、笔、纸夹、电线这类薄、细、凹、多接触面的对象。

推荐首批 pickable 资产：

```text
InternDataAssets/benchmark1.0/scene/file_4/assets/small_usd/mesh_small_box_0_0000_mesh_0_00.usd
InternDataAssets/benchmark1.0/scene/file_2/assets/small_usd/mesh_small_book_0_0001_mesh_0_00.usd
InternDataAssets/benchmark1.0/scene/file_11/assets/small_usd/mesh_small_notebook_0_0030_mesh_0_00.usd
```

可以作为支撑面或背景的大物体：

```text
InternDataAssets/benchmark1.0/scene/file_2/assets/file_2__dining_table_0_id76.usd
InternDataAssets/benchmark1.0/scene/file_5/assets/file_5__long_worktable_oak_0_id62.usd
InternDataAssets/benchmark1.0/scene/file_11/assets/file_11__primary_lab_desk_0_id39.usd
```

## 推荐先尝试的场景

| 推荐顺序 | 场景 | 原因 |
|---:|---|---|
| 1 | file_5 | 最轻量，888 个 prim，87 个 reference，适合作为第一次加载测试。 |
| 2 | file_11 | workspace/lab 风格，小物体实例较少，适合后续机器人加载测试。 |
| 3 | file_2 | 内容更丰富，有 dining/kitchen/study，适合基础流程跑通后再试。 |

## 接入 InternData/SimBox 的注意点

Benchmark 1.0 资产目前还不是 InternData 标准 pickable 结构。这个包里有 USD、PNG、JSON，但没有 `.obj`、`.mtl`、`.npy`。

当前 SimBox 的 pick 技能期望可抓取资产长这样：

```text
some_asset/
  Aligned_obj.usd
  Aligned_obj.obj
  Aligned_grasp_sparse.npy
```

而且代码会通过文件名约定自动找抓取标注：

```text
Aligned_obj.usd -> Aligned_grasp_sparse.npy
```

所以文件名和目录结构很重要。

建议第一个转换目标：

```text
InternDataAssets/assets/benchmark1.0/pickables/box/file_4_box_0000/
  Aligned_obj.usd
  Aligned_obj.obj
  Aligned_grasp_sparse.npy
  source.txt
```

建议第一轮实验顺序：

1. 把 `file_4` 的 `mesh_small_box_0_0000_mesh_0_00.usd` 重新包装成标准 `Aligned_obj.usd`。
2. 从 USD mesh 导出或重建 `Aligned_obj.obj`。
3. 用现有 grasp pipeline 生成 `Aligned_grasp_sparse.npy`。
4. 替换一个最简单的 pick 任务 YAML，让它指向这个新资产。
5. 单独加载 `file_5/scene.usd` 作为背景场景。
6. 尝试把机器人和转换后的 pickable 一起加载进这个场景。
