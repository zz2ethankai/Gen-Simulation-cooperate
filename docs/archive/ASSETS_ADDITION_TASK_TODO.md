# assets_addition 接入现有 Task TODO

目标：将 `InternDataAssets/assets/assets_addition` 下的新增场景/物体资产接入现有 SimBox task，尤其是 `navigate_pick_place_boxed_beverage` 这类包含导航、抓取、放置的任务。

## 1. Arena / Scene 接入

- [ ] 为目标场景新增 arena yaml，例如 `workflows/simbox/core/configs/arenas/addition_file_4_arena.yaml`。
- [ ] 在 arena yaml 中添加整体场景 fixture，`path` 使用相对 `asset_root` 的路径，例如 `assets_addition/file_4/scene.usd` 或 `assets_addition/file_4/empty_room.usd`。
- [ ] 确认 `target_class` 使用 `GeometryObject` 时，USD 可被 Isaac 正常加载并能计算 bbox。
- [ ] 保留或补充 task `regions` 引用的 fixture 名称，例如 `floor`、`table`；如果新增场景没有 `table`，同步修改 `regions.target`。
- [ ] 检查整体场景和 `PlaneObject floor` 是否重复产生碰撞或高度冲突。
- [ ] 确认场景资产内部贴图引用在 Isaac 环境里可解析。
- [ ] 确认整体场景的尺度、原点、朝向与机器人/任务坐标系一致。

## 2. Task YAML 接入

- [ ] 将目标 task 的 `arena_file` 指向新增 arena yaml。
- [ ] 保持 `asset_root: workflows/simbox/assets`；该路径已 symlink 到 `InternDataAssets/assets`。
- [ ] 按新增场景重新设置 `regions` 中机器人、可抓物、放置目标的初始位置。
- [ ] 按新增场景重新设置 `positions` 中 `nav_to_pick`、`nav_to_place` 的 `x/y/yaw`。
- [ ] 确认 `positions` 中每个 `goal` 名称都与 `navigate` skill 的 `goal` 字段一致。
- [ ] 检查 `neglect_collision_names` 是否仍然合理；新增场景可能需要忽略不同 fixture 或对象。
- [ ] 更新 `data.language_instruction` / `detailed_language_instruction` / `collect_info` / `task_dir`，避免旧任务语义和输出目录混用。

## 3. 可抓物资产要求

- [ ] 为每个要执行 `pick` 的新增物体确定最终 USD 路径。
- [ ] 确认该物体能作为 `RigidObject` 加载，而不是只能作为 `GeometryObject` 静态加载。
- [ ] 确认 USD 中存在可作为刚体根的 `prim_path_child`，并在 task yaml 中填写。
- [ ] 确认 `prim_path_child` 指向的 prim 下至少有一个 mesh child；`RigidObject` 当前会读取第一个 child 作为 `mesh_prim_path`。
- [ ] 确认 USD 具备合理 collision / rigid body 属性，用于接触检测、attach、规划避障。
- [ ] 为每个可抓物生成或补齐 `Aligned_grasp_sparse.npy`。
- [ ] 如果继续使用现有 `pick` 逻辑，将资产整理成 `Aligned_obj.usd` + `Aligned_grasp_sparse.npy` 的目录格式。
- [ ] 如果不整理成 `Aligned_obj.usd` 文件名，修改 `pick` skill 支持显式 `grasp_pose_path` 字段。
- [ ] 验证 `Aligned_grasp_sparse.npy` 的 pose 格式能被 `robot.pose_post_process_fn()` 正常处理。
- [ ] 为每个可抓物确定 `scale`、`euler` / `quaternion`、`orientation_mode`，并确认 grasp pose 与物体坐标系一致。
- [ ] 可选：保留或生成 `Aligned_grasp_dense.npz`、预览图、obj/mtl/texture，便于后续调试和可视化。

## 4. 放置目标要求

- [ ] 确定新增任务的 `place_target` 使用现有 tray，还是使用 `assets_addition` 中的新 receptacle / table / surface。
- [ ] 如果使用新增放置目标，确认它能用 `GeometryObject` 加载并能计算 bbox。
- [ ] 如只希望放到放置目标的某个子区域，补充 `place_part_prim_path`。
- [ ] 根据放置目标尺寸重新调整 `x_ratio_range`、`y_ratio_range`、`pre_place_z_offset`、`place_z_offset`。
- [ ] 确认 `success_mode: xybbox` 对新增放置目标合理；bbox 过大或包含无关子物体时需要换子 prim 或换判定方式。
- [ ] 当前 task 没用 `dexplace`，无需 `place_range.yaml`；如果切换到 `dexplace`，再补该文件。

## 5. Skill / 控制器依赖

- [ ] 确认 `navigate` skill 能从 `task.cfg["positions"]` 解析所有 `goal`。
- [ ] 确认机器人配置仍使用 `workflows/simbox/core/configs/robots/split_aloha.yaml`，并能找到 `base_config_file` 和 `nav_config_file`。
- [ ] 确认 Nav2 配置 `nav2/config/default_nav.yaml` 的 map 输出目录、ROS bridge 目录可写。
- [ ] 根据新场景高度调整 `navigate` skill 的 `map_z_min` / `map_z_max`，默认是 `0.0` 到 `0.35`。
- [ ] 检查新增场景障碍物是否会被 controller 的 `ignore_substring` 误忽略，尤其是 `scene`、`table`、机器人名。
- [ ] 检查 `pick` 的 `filter_x_dir` / `filter_y_dir` / `filter_z_dir` 是否适合新增物体的 grasp pose 分布。
- [ ] 检查 `pre_grasp_offset`、`post_grasp_offset_min/max` 是否适合新增物体尺度。
- [ ] `heuristic__skill` 不需要额外资产文件，但要确认左右臂 home pose 在新增场景中不碰撞。

## 6. assets_addition 随机化支持

- [ ] 不要直接对 `assets_addition` 资产启用现有 `apply_randomization: True`，当前随机化逻辑主要适配旧的 category/object/instance.usd 结构。
- [ ] 如需要随机选择 `file_2` 到 `file_11` 场景，新增专门的 scene manifest 读取逻辑。
- [ ] 如需要随机选择 `assets/task_obj/*.usd`，新增专门的资产索引逻辑，并补齐每个可抓物的 grasp 标注。
- [ ] 明确 `package_manifest.json` 与现有 `update_scene_pair()` 所需 JSON 格式之间的映射关系。

## 7. 验证清单

- [ ] 在 Isaac 环境中加载新增 arena，确认 USD stage 无缺失引用和贴图错误。
- [ ] 打印或可视化新增 USD 的 prim tree，记录每个可抓物的 `prim_path_child`。
- [ ] 单独验证 `GeometryObject` / `RigidObject` 加载，不跑完整 skill。
- [ ] 运行 `_set_regions()` 后检查所有对象最终 pose、高度、bbox。
- [ ] 生成 Nav2 map，确认机器人初始点和目标点在可通行区域。
- [ ] 单独跑 `nav_to_pick`，确认导航成功。
- [ ] 单独跑左/右 `pick`，确认能加载 grasp npy、采样候选、IK/forward 通过。
- [ ] 单独跑 `place`，确认放置 bbox、成功判定和碰撞都合理。
- [ ] 跑完整 DAG：`nav_to_pick -> pick_object_1/pick_object_2 -> nav_to_place -> place_object_1/place_object_2 -> home`。
- [ ] 检查输出目录 `output/ros_bridge/skills` 下的 pick debug snapshot，确认失败原因可追踪。
