# Bench 2.1 SimBox 观察录制使用说明

本文说明如何加载 `Bench_2.1_isaacsim/scene_4` 场景，并录制机器人视角与全局俯视视频。

当前能力：场景加载、地面碰撞、基于操作工作区的机器人初始位姿规划、四路相机录制，
以及使用 `RUN_MODE=skill` 保留任务 Skill 链进行验证。

当前不包含：Nav2、`navigate` skill、底盘移动控制。

## 1. 进入仓库

以下命令都要从仓库根目录执行：

```bash
cd /home/bld/ykqin/InternDataEngine
```

## 2. 归一化路径与修复材质绑定

运行：

```bash
python3 scripts/simbox/normalize_bench_paths.py
python3 scripts/simbox/repair_bench21_material_bindings.py
python3 scripts/simbox/refresh_bench21_plane_textures.py
```

脚本会处理 `Bench_2.1_isaacsim/scene_4` 下的 20 个 `simbox_task.yaml`，主要修正：

- `asset_root`
- `arena_file`
- `envmap_lib`
- `robots[0].path`
- 引擎不支持的 region 参数
- 四面墙朝向室内的法线方向，避免背向墙面及其碰撞薄盒出现在房间一侧

材质修复脚本只处理 Bench 交付资产中的一种明确异常：规范共享 USD 仍绑定
`semantic_*.png`，但同目录存在用于 RGB 渲染的 `texture.png`。修复后遵循 SimBox
官方规则：

- arena YAML 中有 `texture`：显式覆盖 USD 材质，例如当前的地板和墙面。
- arena YAML 中没有 `texture`：保留 USD 内部已经绑定的材质，例如家具。

因此，不需要给每件家具都在 `simbox_arena.yaml` 中补 `texture`。需要修的是被所有
任务引用的规范共享 `Aligned_obj.usd`；脚本会先把原文件备份到
`output/.bench21_material_backups/`。如果某个资产没有 `texture.png`，脚本会跳过，
不会臆造材质。

`refresh_bench21_plane_textures.py` 会把四个房间的 24 张地板/墙面 PNG 无损重编码为
标准 RGB PNG，并备份原图到 `output/.bench21_texture_backups/`。这一步用于刷新
Isaac/Hydra 的旧纹理缓存；它不改变纹理图案，也不关闭 YAML 中已有的随机抽取。

脚本可以重复执行。配置已归一化时，最后会显示：

```text
normalized 20 task files; changed 0; changed arenas 0
```

## 3. 地面碰撞配置

`PlaneObject` 只有在 arena 中配置以下字段时才创建碰撞体：

```yaml
- name: floor
  target_class: PlaneObject
  size: [5.0, 4.0]
  collision_enabled: true
  collision_thickness: 0.02
```

字段含义：

- `collision_enabled: true`：创建静态薄盒碰撞体。
- `collision_thickness`：碰撞体厚度，单位为米，默认值为 `0.02`。
- `collision_visible: true`：调试时显示碰撞体；默认隐藏。

隐藏碰撞薄盒会标记为 USD `guide` purpose，并与可见平面留出 1 mm 以上间隙，避免
碰撞面和渲染面共面。墙面法线统一朝向房间内部，因此碰撞薄盒位于墙外侧。

碰撞实现位于：

```text
workflows/simbox/core/objects/plane_object.py
```

## 4. 启动观察录制

脚本中的默认验证任务：

```text
InternDataAssets/Bench_2.1_isaacsim/scene_4/04_bedroom/assets/basic/bedroom_phone_placement/simbox_task.yaml
```

推荐直接运行脚本：

```bash
./scripts/simbox/run_bench21_observe.sh
```

脚本会自动激活 `interndata` Conda 环境，并默认使用 GPU 1、上述卧室 phone 任务和
`random_num=1`。运行名和输出目录会自动从任务路径中提取：

```text
场景名：04_bedroom
任务名：bedroom_phone_placement
运行名：04_bedroom/bedroom_phone_placement
输出目录：output/04_bedroom/bedroom_phone_placement/
```

启动前，脚本会根据原任务生成一份观察专用配置：

```text
output/.observe_configs/04_bedroom/bedroom_phone_placement/simbox_task.yaml
```

这份临时配置只用于本次启动，会自动完成三件事：

- 根据任务的操作支撑面和 `interaction_edge` 生成无明显基座碰撞的机器人候选位姿，
  并写出同目录的 `simbox_task.workspace_report.json`。
- 把任务原有技能替换为不会操作物体的 `wait`，避免 Bench 交付配置中的空技能列表触发 `IndexError`。
- 将相机名改为官方的 `${tasks.0.robots.0.name}_...` 动态写法，保证记录器能够识别当前机器人。

原始 `simbox_task.yaml` 不会被修改。

需要修改参数时，在命令前设置环境变量：

```bash
GPU_ID=0 RANDOM_NUM=1 \
TASK_CONFIG=InternDataAssets/Bench_2.1_isaacsim/scene_4/03_livingroom/assets/basic/livingroom_phone_to_cabinet/simbox_task.yaml \
./scripts/simbox/run_bench21_observe.sh
```

查看全部参数：

```bash
./scripts/simbox/run_bench21_observe.sh --help
```

需要执行任务原有 Skill 时使用：

```bash
RUN_MODE=skill \
TASK_CONFIG=InternDataAssets/Bench_2.1_isaacsim/scene_4/04_bedroom/assets/basic/bedroom_hand_cream_to_organizer/simbox_task.yaml \
./scripts/simbox/run_bench21_observe.sh
```

机器人工作区、对象混用、验证矩阵和 failure code 的完整说明见
`docs/Bench2.1_机器人工作区与Skill可达性规划实施说明.md`。

如果需要隔离运行环境，也可以使用 Docker：

```bash
INTERNDATA_ISAAC_GPU_DEVICE_IDS=1 \
docker compose -f docker/docker-compose.simbox.yml \
  -p bench21-observe run --rm --no-deps \
  -v /data1/yikai/InternDataEngine/output:/data1/yikai/InternDataEngine/output \
  isaac bash -lc 'cd /workspace && /isaac-sim/python.sh launcher.py \
    --config configs/simbox/de_plan_with_render_template.yaml \
    --name=bench21_observe \
    --load_stage.scene_loader.args.cfg_path=InternDataAssets/Bench_2.1_isaacsim/scene_4/03_livingroom/assets/basic/livingroom_coffee_table_cleanup/simbox_task.yaml \
    --load_stage.layout_random_generator.args.random_num=1 \
    --store_stage.writer.args.output_dir=output/bench21_observe/'
```

说明：

- Docker 命令中的 `INTERNDATA_ISAAC_GPU_DEVICE_IDS=1` 表示只向容器暴露物理 GPU 1。
- 直接运行脚本时，`GPU_ID` 始终表示物理 GPU 编号。脚本只向子进程暴露这一张卡，并将
  PhysX、PyTorch、CuRobo 和 Warp 映射到进程内的逻辑 `cuda:0`；Isaac 的 Vulkan renderer
  继续使用对应的物理 GPU 编号。两种编号最终指向同一张卡。
- 因此，选择 `GPU_ID=1` 后日志显示 `cuda:0` 是正常现象：这里的 `cuda:0` 实际对应物理
  GPU 1，不是错误使用了物理 GPU 0。脚本会覆盖外部已有的 `CUDA_VISIBLE_DEVICES`。
- `random_num=1` 表示生成一个样本。
- `--name` 和 `output_dir` 建议每次使用新名称，避免混淆输出。
- 当前仓库的 `output` 是外部目录软链接，因此命令中额外挂载了真实输出目录。

运行结束时应看到：

```text
Task is successful
```

## 5. 查看输出

输出位于：

```text
output/04_bedroom/bedroom_phone_placement/BananaBaseTask/split_aloha/runs/...
```

主要文件：

```text
images.rgb.global/demo.mp4
images.rgb.head/demo.mp4
images.rgb.hand_left/demo.mp4
images.rgb.hand_right/demo.mp4
lmdb/data.mdb
meta_info.pkl
```

其中：

- `global`：世界坐标系下的固定俯视相机。
- `head`：机器人头部相机。
- `hand_left`、`hand_right`：左右机械臂相机。

全局相机名称采用动态写法：

```yaml
name: ${tasks.0.robots.0.name}_global
```

使用 `split_aloha` 时解析为 `split_aloha_global`；记录器保存时会去掉机器人名前缀，因此输出目录为 `images.rgb.global`。

## 6. 调整录制长度

观察脚本会在临时配置中注入 `wait` skill 来产生渲染帧，无需修改原任务。通过环境变量调整长度：

```bash
WAIT_STEPS=600 ./scripts/simbox/run_bench21_observe.sh
```

默认 `WAIT_STEPS=300`。当前视频编码为 15 FPS，实测生成 310 帧、约 20.7 秒；设置为 `600` 时，视频长度大约翻倍。

## 7. 常见问题

### 全局视频中没有机器人

检查：

1. `robots[0].path` 是否指向 `../shared_assets/split_aloha_mid_360/robot.usd`。
2. arena 的 floor 是否设置了 `collision_enabled: true`。
3. LMDB 中的 `T_world_base.z` 是否持续下降。持续下降说明地面碰撞没有生效。

### 相机已加载但没有视频

相机名称必须包含机器人名。推荐始终使用：

```yaml
name: ${tasks.0.robots.0.name}_global
```

### 报错提示数组和 Kernel 位于不同 CUDA 设备

典型错误包含：

```text
trying to launch on device='cuda:X', but input array ... is on device=cuda:Y
```

这表示 Isaac、CuRobo 或 Warp 看到了多张 GPU，并分别选择了不同设备。请始终通过
`GPU_ID=<物理编号> ./scripts/simbox/run_bench21_observe.sh` 启动，不要绕过脚本直接把物理
GPU 编号同时传给 `active_gpu` 和 `physics_gpu`。脚本会把指定物理卡隔离为进程内唯一的
逻辑 `cuda:0`。启动信息应显示：

```text
physical GPU N -> process cuda:0
```

### Conda 环境导入了其他工作区的 CuRobo

Conda 环境不只保存依赖版本；`pip install -e <目录>` 还会在 `site-packages` 中保存源码目录
指针。因此即使当前工作目录已经切到本仓库，Python 仍可能从旧工作区导入 CuRobo。观察脚本
会检查实际导入路径，并要求它位于当前仓库的 `InternDataAssets/curobo` 下。可以这样确认：

```bash
conda activate interndata
python -c 'import curobo; print(curobo.__file__)'
```

如果路径不是当前仓库，重新绑定：

```bash
python -m pip install --no-deps -e \
  /home/bld/ykqin/InternDataEngine/InternDataAssets/curobo
```

### Bedroom 报 `Found zero norm quaternions in quat`

这通常不是相机或编码问题，而是机器人已经发生 PhysX 失稳。Bench 交付配置中的
`robot_initial_region.center` 是绝对房间坐标，而 `A_on_B_region_sampler.pos_range` 是相对
floor 中心的偏移；直接写成零会把机器人放到房间中心，可能与床等家具发生碰撞。
默认情况下，`prepare_bench_observe_config.py` 会先由工作区规划器生成新的 floor-relative
robot shift。只有设置 `SKIP_WORKSPACE_PLANNING=1` 对照交付起点时，才会执行交付绝对起点到
floor-relative shift 的转换。两种方式都不修改原始任务资产。

### 机器人能站住但不会移动

这是当前预期行为。地面碰撞已经接入，但 Nav2、`navigate` skill 和底盘 bridge 尚未合并。

### 同一面墙每次都固定变黑

这不是 HDR 随机选暗了。实际定位时把四面墙固定到不同 `texture_id`，黑面会跟随
`wall_textures/2.png` 移动；将同一张图无损重编码后再次固定测试，黑面消失。源图的
RGB 像素正常，因此根因是 Isaac/Hydra 对旧路径内容使用了失效或损坏的纹理缓存，
不是图片本身画成了黑色。运行 `refresh_bench21_plane_textures.py` 可刷新四个房间的
地板/墙面纹理。`normalize_bench_paths.py` 同时会修正墙面法线，但这是碰撞薄盒方向的
正确性修复，不是本次黑纹理的直接根因。

## 8. 后续待办：视觉随机化

当前先建立“材质和光照都正确”的可复现基线，再扩展随机化，避免把资产错误误当成
随机效果。

### TODO：纹理随机化

- 保留当前官方语义：有 YAML `texture` 才显式覆盖；没有则保留 USD 原材质。
- 对地板、墙面和允许变化的家具建立经过检查的 RGB 材质库，禁止把
  `semantic_*.png` 混入 RGB 录制。
- 增加固定 seed 与每个物体实际选中纹理的运行记录，保证结果可复现、可追查。
- 增加启动前材质审计和录制后黑块/缺失纹理检查。

### TODO：光照随机化

- 当前先使用 Bench 自带的 `scene_4/shared_assets/envmap_lib` 作为正确路径基线。
- 后续引入经过亮度校准的 HDR 集合，并联合随机化 HDR、rotation 和 intensity，避免
  不同 HDR 的曝光差异过大。
- 记录随机 seed、HDR 文件名、rotation、intensity，支持精确复现。
- 增加全黑帧、过曝帧和亮度异常的自动拒绝或重采样机制。
