# SimBox 配置参考

本文把 YAML 配置项映射到实际消费它们的代码路径。

## 1. Task YAML

Task YAML 由 `workflows/simbox/utils/task_config_parser.py` 解析，并由 `workflows/simbox_dual_workflow.py` 和 `workflows/simbox/core/tasks/banana.py` 执行。

### 顶层键

| Key | 消费位置 | 说明 |
| --- | --- | --- |
| `name` | `BananaBaseTask`、日志、输出命名 | 必填。 |
| `asset_root` | 物体/相机加载、资源查找、distractor | 必填。 |
| `task` | `get_task_cls(task)` | 必填，必须存在于任务注册表中。 |
| `task_id` | root prim path、场景命名空间 | 必填。 |
| `offset` | `BaseTask.__init__` | 当前任务类要求必填。 |
| `render` | `BananaBaseTask._render` | 是否启用相机观测。 |
| `arena_file` | `SimBoxDualWorkFlow.reset()` | 必填，会加载到 `cfg["arena"]`。 |
| `neglect_collision_names` | `SimBoxDualWorkFlow.reset()` | 可选，碰撞过滤。 |
| `env_map` | `BananaBaseTask._set_envmap()` | 可选。 |
| `robots` | `SimBoxDualWorkFlow._initialize_controllers()`、`BananaBaseTask._load_robot()` | 必填。 |
| `objects` | `update_rigid_objs()`、`update_articulated_objs()`、`_load_obj()` | 必填。 |
| `regions` | `BananaBaseTask._set_regions()`、`optimize_2d_manip_layout()` | 操作类任务通常必填。 |
| `cameras` | `BananaBaseTask._load_camera()`、`_set_camera_poses()` | 可选，但在渲染/记录时会用到。 |
| `distractors` | `BananaBaseTask._create_distractor_cfg()`、`set_distractors()` | 可选。 |
| `fluid` | `BananaBaseTask._set_fluid()`、`SimBoxDualWorkFlow` 重置钩子 | 可选。 |
| `skills` | `SimBoxDualWorkFlow._initialize_skills()` | 必填。 |
| `data` | logger、语言、episode 长度、collect-info | 必填。 |

### `env_map`

由 `BananaBaseTask._set_envmap()` 消费。

| Key | 含义 |
| --- | --- |
| `envmap_lib` | `asset_root` 下的 HDR 目录。 |
| `apply_randomization` | 为 true 时随机采样 HDR / intensity / rotation，否则走确定性默认值。 |
| `intensity_range` | 随机强度范围。 |
| `rotation_range` | 随机欧拉角范围。 |
| `light_type` | 可选，默认 `DomeLight`。 |

### `robots`

由 `SimBoxDualWorkFlow._initialize_controllers()` 和 `BananaBaseTask._load_robot()` 消费。

| Key | 含义 |
| --- | --- |
| `name` | 机器人实例名。 |
| `target_class` | 注册表中的机器人类 key。 |
| `robot_config_file` | 外部机器人 YAML，会合并进任务 robot 配置。 |
| `robot_file` | 每个 controller 的机器人文件，可以是字符串或列表。 |
| `path` | 可选，覆盖机器人 USD 资源。 |
| `translation` | 初始位姿平移。 |
| `euler` / `quaternion` | 初始朝向。 |
| `scale` | 初始缩放。 |
| `constrain_grasp_approach` | 传给 controller 构造函数。 |
| `collision_activation_distance` | 传给 controller 构造函数。 |
| `ignore_substring` | 传给 controller 构造函数。 |
| `use_batch` | 传给 controller 构造函数。 |
| `left_joint_home` / `right_joint_home` | `TemplateRobot` 使用。 |
| `left_joint_home_std` / `right_joint_home_std` | 关节初始噪声。 |

### `objects`

由 `update_rigid_objs()`、`update_articulated_objs()`、`_load_obj()`、`optimize_2d_manip_layout()` 和 contact-view 构建逻辑消费。

通用字段：

| Key | 含义 |
| --- | --- |
| `name` | 物体实例名。 |
| `path` | 相对于 `asset_root` 的资源路径。 |
| `target_class` | `RigidObject` / `GeometryObject` / `ArticulatedObject` 等。 |
| `dataset` | 仅元数据，除非下游工具额外使用。 |
| `category` | 会被随机化工具和语言替换逻辑更新。 |
| `prim_path_child` | 刚体资源的子 prim path。 |
| `translation` | 初始位姿平移。 |
| `euler` / `quaternion` | 初始朝向。 |
| `scale` | 初始缩放。 |
| `visible` | 初始可见性。 |
| `texture` | 传给 `apply_texture()`。 |
| `apply_randomization` | 启动物体类型对应的随机化逻辑。 |

随机化相关字段：

| Key | 含义 |
| --- | --- |
| `orientation_mode` | 刚体的 `keep` / `suggested` / `random`。 |
| `scale_mode` | 刚体的 `keep` / `suggested`。 |
| `randomization_scope` | 刚体的 `category` / `full` / 列表。 |
| `gap` | 可选 gap 元数据，从 `gap.yaml` 读取。 |
| `info_name` | articulation 信息查询必填。 |
| `obj_info_path` | articulation 物体会自动设置。 |
| `joint_position_range` | 若存在，`ArticulatedObject.initialize()` 会采样关节位置。 |
| `fix_base` | 固定 articulation 基座。 |

### `cameras`

由 `BananaBaseTask._load_camera()`、`CustomCamera`、`_perturb_camera()`、`_set_camera_poses()` 消费。

| Key | 含义 |
| --- | --- |
| `name` | 相机实例名。 |
| `camera_file` | 外部相机参数 YAML，包含内参和分辨率。 |
| `parent` | 可选父 prim path。 |
| `translation` | 安装位姿平移。 |
| `orientation` | 安装位姿四元数。 |
| `camera_axes` | 传给 Isaac camera API。 |
| `apply_randomization` | 为 true 时对相机位姿做噪声扰动。 |
| `max_translation_noise` | 最大平移噪声幅度。 |
| `max_orientation_noise` | 最大角度噪声，单位度。 |

注意：`camera_file` 提供内参和渲染分辨率；`apply_randomization` 只扰动外参。

### `regions`

由 `BananaBaseTask._set_regions()` 和 `optimize_2d_manip_layout()` 消费。

| Key | 含义 |
| --- | --- |
| `object` | 目标物体名。 |
| `target` | 参考物体名。 |
| `random_type` | `RandomRegionSampler` 上的方法名。 |
| `random_config` | 采样器参数。 |
| `priority` | 可选，从 `random_region_list` 里选索引。 |
| `container` | 容器类任务的特殊放置分支。 |
| `target2` | 需要双支撑时的第二目标。 |
| `sub_tgt_prim` | 目标上的可选子 prim path。 |

### `distractors`

由 `_create_distractor_cfg()` 和 `_rebuild_distractors()` 消费。

| Key | 含义 |
| --- | --- |
| `path` | 资源搜索根目录。 |
| `min_num` / `max_num` | 采样数量上下限。 |
| `scale` | distractor 默认缩放。 |
| `target` | 放置目标。 |
| `pos_range` | 放置范围。 |
| `min_object_distance` | XY 间距约束。 |
| `distractor_buffer` | distractor 间缓冲距离。 |
| `exclude_categories` | 可选排除分类。 |
| `exclude_keywords` | 可选排除子串。 |
| `target_class` | 默认 `RigidObject`。 |
| `prim_path_child` | 默认 `Aligned`。 |

### `fluid`

由 `_set_fluid()` 消费。

| Key | 含义 |
| --- | --- |
| `container_name` | 必填，用于放置粒子体积。 |
| `particleContactOffset` | 粒子间距基值。 |
| `spacing_scale` | 粒子间距倍率。 |
| `numParticlesX/Y/Z` | 粒子网格维度。 |
| `center_z` | 是否在 Z 方向居中。 |
| `z_offset` | Z 偏移。 |
| `max_velocity` | 粒子系统速度上限。 |
| `mass` | 粒子质量。 |
| `density` | 粒子密度。 |
| `color` / `emissiveColor` / `opacity` | 视觉材质。 |

### `skills`

由 `SimBoxDualWorkFlow._initialize_skills()` 和技能执行循环消费。

| Key | 含义 |
| --- | --- |
| robot name | 外层 key，必须匹配某个 robot 的 `name`。 |
| controller name (`left` / `right` / etc.) | 下一层嵌套。 |
| `name` | 注册表中的技能名。 |
| `objects` | 技能对象参数。 |
| `depends_on` | DAG 模式依赖列表。 |
| `id` | DAG 节点 id / 旧版内部 id。 |
| skill-specific fields | 原样传给技能构造函数。 |

### `data`

由 logger / language / episode 控制逻辑消费。

| Key | 含义 |
| --- | --- |
| `task_dir` | 输出子目录。 |
| `language_instruction` | 高层语言指令模板。 |
| `detailed_language_instruction` | 详细语言指令模板。 |
| `collect_info` | articulation 随机化时自动填充。 |
| `version` | 记录的数据集版本字符串。 |
| `update` | 下游 workflow 使用的任务开关。 |
| `max_episode_length` | episode 截断长度。 |
| `log_motion_vectors` | 是否记录 motion vector。 |

## 2. Arena YAML

Arena 文件由 `task_cfg["arena_file"]` 读取，并存入 `cfg["arena"]`。

### 顶层键

| Key | 含义 |
| --- | --- |
| `name` | arena 名称。 |
| `fixtures` | 必填，静态场景元素列表。 |
| `update_freq` | `BananaBaseTask.individual_reset()` 用于 scene-pair 刷新。 |
| `involved_scenes` | 当两个 fixture 都随机化时，`update_scene_pair()` 使用。 |

### `fixtures`

由 `BananaBaseTask.set_up_scene()`、`_set_fixture_textures()`、`update_scenes()`、`update_hearths()`、`update_scene_pair()` 消费。

通用字段：

| Key | 含义 |
| --- | --- |
| `name` | fixture 名称。 |
| `target_class` | `GeometryObject` / `PlaneObject` / 等。 |
| `path` | 相对于 `asset_root` 的资源路径。 |
| `translation` | 初始位姿平移。 |
| `euler` / `quaternion` | 初始朝向。 |
| `scale` | 初始缩放。 |
| `size` | `PlaneObject` 的平面尺寸。 |
| `texture` | 可选纹理规格。 |
| `apply_randomization` | 启用纹理或 scene 切换逻辑。 |
| `linear_velocity` | 传送带类 fixture 使用。 |
| `collision_enabled` | 可选平面碰撞开关。 |
| `collision_thickness` | 可选平面碰撞厚度。 |

纹理块：

| Key | 含义 |
| --- | --- |
| `texture_lib` | 资源库目录。 |
| `apply_randomization` | 随机纹理选择。 |
| `texture_id` | 不随机时的确定性纹理索引。 |
| `texture_scale` | 传给材质。 |

## 3. `configs/de*.yaml`

这些是 Nimbus pipeline 配置，由 `launcher.py` -> `ConfigProcessor` -> `run_data_engine()` 解析。

### 常见顶层键

| Key | 含义 |
| --- | --- |
| `name` | 实验名。 |
| `load_stage` | 创建 world 和 workflow 的 stage。 |
| `layout_random_generator` | 随机化或加载 layout 的 stage。 |
| `plan_stage` | 规划 stage。 |
| `plan_with_render_stage` | 规划 + 渲染合并 stage。 |
| `dump_stage` | 序列化 stage。 |
| `dedump_stage` | 反序列化 stage。 |
| `render_stage` | 仅渲染 stage。 |
| `store_stage` | 输出写盘 stage。 |
| `stage_pipe` | 仅 `de_pipe` 使用的并行管线配置。 |

### `load_stage.scene_loader.args.simulator`

由 `nimbus_extension/components/load/env_loader.py` 消费。

| Key | 含义 |
| --- | --- |
| `physics_dt` | 物理时间步。 |
| `rendering_dt` | 渲染时间步。 |
| `stage_units_in_meters` | Isaac stage 单位尺度。 |
| `headless` | 是否无头启动 SimulationApp。 |
| `experience` | 可选 Kit experience 路径。 |
| `anti_aliasing` | SimulationApp 抗锯齿模式。 |
| `multi_gpu` | SimulationApp 多 GPU 开关。 |
| `active_gpu` | CUDA 设备索引。 |
| `physics_gpu` | 物理 GPU 索引。 |
| `max_gpu_count` | SimulationApp 限制。 |
| `renderer` | `RayTracedLighting` / `PathTracing`。 |
| `width` / `height` | 渲染分辨率。 |
| `resolution` | `env_loader` 不消费，应该用 `width` / `height`。 |
| `denoiser` | Path tracing 降噪开关。 |
| `samples_per_pixel_per_frame` | 每帧采样数。 |
| `max_bounces` | 光线反弹上限。 |
| `max_specular_transmission_bounces` | 光线反弹上限。 |
| `max_volume_bounces` | 光线反弹上限。 |
| `subdiv_refinement_level` | 网格细分等级。 |
| `planning_step_render` | 传给 `SimBoxDualWorkFlow`。 |
| `rendering_interval` | 渲染进程里的渲染节奏。 |
| `total_spp` | RTX pathtracing 设置。 |
| `adaptive_sampling` | Path tracing 设置。 |
| `adaptive_sampling_target_error` | Path tracing 设置。 |
| `optix_denoiser` | Path tracing 设置。 |
| `optix_temporal_denoiser` | Path tracing 设置。 |
| `denoise_aovs` | Path tracing 设置。 |
| `firefly_filter` | Path tracing 设置。 |
| `firefly_max_intensity_glossy` | Path tracing 设置。 |
| `firefly_max_intensity_diffuse` | Path tracing 设置。 |
| `reset_pt_accum_on_time_change` | Path tracing 设置。 |
| `fractional_cutout_opacity` | Path tracing 设置。 |

重要：`physics_dt = 1/30` 和 `rendering_dt = 1/30` 表示仿真按 30 Hz 前进，但导出视频 FPS 由别处控制：

| 导出路径 | FPS 来源 |
| --- | --- |
| `nimbus/components/data/observation.py` | `flush_to_disk(..., video_fps=10)` 默认值 |
| `workflows/simbox/core/loggers/lmdb_logger.py` | 当前默认由 `physics_dt` 推导，`1/30 -> 30fps` |
| `scripts/simbox/record_collaborate_topdown_mp4.py` | CLI `--fps`，默认 `20` |

因此，导出的 MP4 不是天然 30fps，除非调用方传入 30，或者默认链路按 `physics_dt` 推导得到 30。

### `layout_random_generator.args`

由 `nimbus_extension` 的 layout randomizer 消费。

| Key | 含义 |
| --- | --- |
| `random_num` | 生成样本数。 |
| `strict_mode` | 强制输出条数等于 `random_num`。 |
| `input_dir` | 仅渲染模板使用，用于读取 plan 输出。 |

### `plan_with_render_stage.plan_with_render.args`

| Key | 含义 |
| --- | --- |
| `emit_obs_on_failure` | 规划失败时是否输出兜底观测。 |
| `failure_obs_length` | 兜底观测长度。 |

### `store_stage.writer.args`

| Key | 含义 |
| --- | --- |
| `batch_async` | 异步批量写入。 |
| `output_dir` | 输出根目录。 |

### `stage_pipe`

由 `nimbus/scheduler/sches.py` 和 `nimbus/data_engine.py` 消费。

| Key | 含义 |
| --- | --- |
| `stage_num` | 每个 pipe 段里的 stage 数量。 |
| `stage_dev` | 每个 pipe 段的设备。 |
| `worker_num` | 每个 pipe 段的 worker 数。 |
| `worker_schedule` | 是否启用动态 worker 调度。 |
| `safe_threshold` | 队列限流阈值。 |
| `status_timeouts.idle` | Idle 超时。 |
| `status_timeouts.ready` | Ready 超时。 |
| `status_timeouts.running` | Running 超时。 |
| `monitor_check_interval` | 健康检查间隔。 |

## 4. 说明

- 即使某些字段不会被代码直接使用，只要它们出现在发布版配置里，这里也保留。
- 有些字段只对特定任务或物体类生效；未被支持的字段会被忽略，除非对应代码分支消费它们。
