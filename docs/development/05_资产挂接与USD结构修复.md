# 资产挂接与 USD 结构修复

> 覆盖时间：2026-06-07 ~ 2026-08-08
> 涉及提交：0643d0c, 46ae176, 48f1c44, baa37d2, b2c943f
> 涉及代码：workflows/simbox/core/utils/attach_collision_utils.py, workflows/simbox/core/objects/rigid_object.py, workflows/simbox/core/planning/collision_scene_manager.py, workflows/simbox/core/controllers/curobo/controller.py, workflows/simbox/core/execution/safety_monitor.py, InternDataAssets/

## 背景

挂接路径分两个层次：asset_root+path 负责在磁盘上找到 Aligned_obj.usd 文件；attach_prim_path_child 负责在已加载的 USD Stage 内定位 Prim。多数 Bench2.1 任务没有配置 attach_prim_path_child，RigidObject 在字段缺失时回退到刚体根的第一个 child（children[0]），该回退对官方扁平资产（/Aligned/Scan）有效，但 Bench2.1 资产是嵌套结构（/Aligned/Normalize/Source/base_link/...），回退选中中间 Xform /Aligned/Normalize，CuRobo world 里没有以该路径注册的 obstacle，报 ATTACH_PRIM_NOT_IN_CUROBO_WORLD。此外铰链资产存在 purpose=guide 隔离碰撞、Cube 尺寸写在父 Xform 上的结构问题，导致 Nav2 地图和 cuRobo 解析失败。修复按三阶段推进：逐资产显式补配置、加载与挂接接口重构为多 Prim 契约、源资产统一交付结构。

## 时间线

| 日期 | 提交 | 内容 |
|------|------|------|
| 2026-06-07 | -（记录于 NEW_ASSET_IMPORT_FIX_LOG） | Issue 1：robot.usd 路径 duplicated workflows 段；Issue 2：空 skill 列表崩溃 |
| 2026-06-08 | 0643d0c | checkpoint before curobo attach fix：导航侧状态固化，为 attach 修复做基线 |
| 2026-06-08 | 46ae176 | fix curobo attach path resolution：按 CuRobo world 实际 object name 解析 attach 路径 |
| 2026-07-27 | 48f1c44 | merge：RigidObject 拆分 rigid_prim_path/attach_collision_prim_paths，新增 attach_collision_utils.py，废除 children[0] 猜测 |
| 2026-08-02 | baa37d2 | attach_prim_paths 契约化：显式挂接路径必须是已注册 Physics collider 的子集 |
| 2026-08-08 | b2c943f | refresh_after_task_reset 重发现 collider；USD scale 混入位姿导致伪旋转漂移的判定调整 |

## 修改记录

### 2026-06-07

#### 2026-06-07 · robot.usd 路径 duplicated workflows 段（Issue 1）
- 改动：20 个导入的 assets/basic/*/simbox_task.yaml 中 robot 路径从 ../../../../../workflows/simbox/example_assets/split_aloha_mid_360/robot.usd 改为 ../../../../example_assets/split_aloha_mid_360/robot.usd；后续 envmap 路径同样改为资产树内相对路径 assets/envmap_lib。
- 原因：USD reference 解析把相对串从 asset_root 规范化出 /workspace/workflows/workflows/... 重复段；后续 Empty typeName（physxArticulation:solverPositionIterationCount）是 robot prim 加载失败的次生错误。
- 文件：InternDataAssets/Bench_2.1_isaacsim/scene_4/*/assets/basic/*/simbox_task.yaml
- 验证：20 个任务文件的静态路径校验 task_files=20 errors=0。

#### 2026-06-07 · 空 skill 列表崩溃（Issue 2）
- 改动：20 个导入 YAML 的首个 skill group（base/left/right 全空）替换为最小 right 臂序列 pick → place → heuristic__skill home；pick 目标选自带 Aligned_grasp_sparse.npy 的对象，place 目标选自同任务内非墙/非地板 fixture。
- 原因：plan_first_skill 直接取 lr_skill_list[0].simple_generate_manip_cmds()，空列表触发 IndexError: list index out of range。
- 文件：InternDataAssets/Bench_2.1_isaacsim/scene_4/*/assets/basic/*/simbox_task.yaml
- 验证：skill 结构校验 skill_lists=20 empty=0，静态任务校验 task_files=20 errors=0。

#### 2026-06-07 · 导航侧 Issue 3~10（不在资产/USD 修复范围）
- 改动：无。后续 Issue 3~10 全部是 Nav2 侧问题（起始点 lethal space、footprint/inflation 覆盖、positions 候选、cmd_vel 收敛），不属于资产挂接与 USD 结构修复，由导航系统文档记录。
- 原因：这些 Issue 不涉及 USD 内部 Prim 或挂接路径。
- 文件：nav2/
- 验证：-。

### 2026-06-08

#### 2026-06-08 · checkpoint before curobo attach fix（0643d0c）
- 改动：nav2/runtime/config.py 重构（367 行）、base_bridge.py 41 行等导航侧改动，作为随后 attach 修复前的基线；不涉及 attach 逻辑。
- 原因：在改动 CuRobo attach 路径前固化导航运行状态。
- 文件：nav2/runtime/config.py, nav2/runtime/runtime.py, workflows/simbox/core/mobile/bridge/base_bridge.py
- 验证：checkpoint 提交。

#### 2026-06-08 · fix curobo attach path resolution（46ae176）
- 改动：curobo/controller.py 新增 _get_curobo_world_object_names()、_select_attach_descendants()、_resolve_attach_object_names()；attach_obj() 先把请求路径映射到 CuRobo world 中实际注册的 object name（候选按 /visual、/collisions/、原列表的顺序选择），再传给 attach_objects_to_robot，并 disable 同实体其余 descendant obstacle。
- 原因：请求的 Prim path（如 /Aligned/Normalize）未在 CuRobo world 注册，需要解析到实际注册的碰撞 descendant。
- 文件：workflows/simbox/core/controllers/curobo/controller.py
- 验证：盐瓶 A/B 实验：同候选点 annulus_004 左右臂轨迹数保持 14/17 不变，attach_prim_valid 从 false 变 true，结果变 feasible。

### 2026-07-27

#### 2026-07-27 · attach 契约拆分与多 Prim 接口（48f1c44）
- 改动：RigidObject 拆分 rigid_prim_path 与 attach_collision_prim_paths；新增 attach_collision_utils.py：join_prim_path() 用 Sdf.Path.AppendPath 组合（不再用 os.path.join 表达 USD path）、has_nonempty_collision_bound()、collision_candidate_paths()、resolve_attach_collision_prims()；配置接口规范为复数 attach_prim_path_children，旧单数字段只作迁移兼容；缺失配置时禁止 children[0] 静默猜测，仅当唯一无歧义碰撞候选时自动采用，多候选或无候选时结构化失败。
- 原因：children[0] 只表示结构上第一个子节点，不表示碰撞语义；多数 Bench2.1 资产选中 /Aligned/Normalize 中间 Xform。
- 文件：workflows/simbox/core/objects/rigid_object.py, workflows/simbox/core/utils/attach_collision_utils.py, workflows/simbox/core/planning/grasp_plan_evaluator.py
- 验证：test/unit/test_attach_collision_utils.py；手 cream 资产显式配置 attach_prim_path_child: Aligned/Normalize/Source/base_link/collisions 后通过 attach 检查。

### 2026-08-02

#### 2026-08-02 · attach_prim_paths 契约化（baa37d2）
- 改动：collision_scene_manager.py 的 _discover()（第 370 行）把 entity.attach_collision_prim_paths 与已发现 Physics collider 清单对照（第 485 行），配置路径不在启用 collider 集合内直接抛 CollisionSceneError；attach_target()（第 1206 行）只把显式 attach_prim_paths 传给 controller.attach_objects()，不再回退到 children[0]；attach_collision_utils.py 扩展单数/复数配置互斥检查与路径必须位于刚体根之下的校验。
- 原因：attach 几何与 world collider 必须是一份精确且互为子集的契约。
- 文件：workflows/simbox/core/planning/collision_scene_manager.py, workflows/simbox/core/utils/attach_collision_utils.py
- 验证：test/unit/test_attach_collision_utils.py 增 64 行，test/unit/test_collision_scene_manager.py 增 186 行。

### 2026-08-08

#### 2026-08-08 · USD 重载后的 collider 重发现与 scale 误报修复（b2c943f）
- 改动：collision_scene_manager.py 新增 refresh_after_task_reset()（第 1451 行）：任务随机化重建刚体 USD 后先清除 attached CuRobo 状态，再清空全部按精确 Prim path 键控的结构并重新 _discover()，为每个 physics controller 重建 world（build_world_config + update_world + audit_controller）；safety_monitor.py 把 attached_slip_rotation_deg 从 abort 条件改为仅记录。
- 原因：重建的 USD 可能暴露不同的精确 collider path，旧记录残留会复现 ATTACH_COLLISION_PRIM_NOT_IN_CUROBO_WORLD（第 1504 行注释）；当前 collision-scene pose 路径在 3x3 块中包含 USD scale，旋转角公式对 0.001 缩放的 wine-glass 资产产生约 120° 伪漂移。
- 文件：workflows/simbox/core/planning/collision_scene_manager.py, workflows/simbox/core/execution/safety_monitor.py
- 验证：test/unit/test_collision_scene_manager.py 增 31 行。

## 资产结构约定（铰链资产，来自铰链修复规范）

### 层级与关节约定
- 目标层级：/root（Xform，defaultPrim）→ /root/instance（ArticulationRootAPI）→ /root/instance/base（RigidBodyAPI，固定部分）+ /root/instance/contact_link（RigidBodyAPI，目标运动部件）+ /root/instance/contact_link_revolute（PhysicsRevoluteJoint，body0=base、body1=contact_link、jointEnabled=true、collisionEnabled=false，axis/lowerLimit/upperLimit 按真实关节）。base 收编所有不随目标部件运动的部分（外壳、隔板、其他非目标门/抽屉）；contact_link 只含当前 skill 操作的运动部件及上面的把手/装饰/显示面板。多关节资产按 skill 目标拆成多个单 contact_link 资产（asset_left_door / asset_right_door / asset_drawer），保持与 microwave 单 contact_link 的 skill/Kps 约定一致。
- 违反时后果：base/contact_link/joint 无法被 skill 与 Kps 定位，open/close 类 skill 直接失败；forbid_collision_paths 与 contact sensor pattern 匹配不到目标部件。

### 碰撞几何约定
- 碰撞 prim 自身带 UsdPhysics.CollisionAPI，purpose=default，所有父级也不能用 purpose=guide 隔离；要求 UsdGeom.BBoxCache(TimeCode.Default(), [default_]) 下 bbox 非空。碰撞几何路径位于 base 或 contact_link 之下，Mesh 或 Cube 均可。
- 违反时后果：Nav2 静态地图导出器用 default-purpose bbox cache 栅格化 collision geometry，碰撞 prim 在 guide 下 bbox 为空，障碍物不进 map.pgm；CuRobo/contact view 按 default purpose 解析同样跳过该几何。

### Cube scale 约定
- Cube prim 自身必须有有效 xformOp:scale，不允许只写在父 Xform 上（错误示例：父 Xform 写 scale=(0.4,0.2,0.8)、box 自身无 scale）。
- 违反时后果：cuRobo 的 USD obstacle parser 直接读 Cube prim 自身的 xformOp:scale，缺失报 TypeError: 'NoneType' object is not iterable。

### purpose=guide 的处理
- 铰链资产的导航/碰撞几何一律不用 guide 隔离，default purpose 可见。
- 代码侧挂接自动发现（见下节 _preferred_discovery_candidates）反而优先选择 guide-purpose 碰撞代理，这只是过渡兼容，最终资产应 default purpose 可见、无 guide 依赖。

### 代码侧临时修复
- scripts/simbox/generate_hinge_articulation_kps.py 的 normalize_cube_scales() 补齐 Cube 自身 xformOp:scale；workflows/simbox/core/tasks/banana.py 读取 articulated object 的 forbid_collision_paths 并合并 contact sensor pattern；workflows/simbox/core/objects/articulated_object.py 保存 forbid_collision_paths。
- 长期目标是从资产侧统一为 microwave 风格结构，标注文件由工具生成，不作为资产主要交付项；不要用 task YAML 改路径来规避资产结构问题。

### 验收标准（来自铰链修复规范）
- defaultPrim=/root；instance/base/contact_link/contact_link_revolute 四 prim 存在；关节 body 关系正确（body0→base、body1→contact_link）；所有参与导航障碍的 collision prim 在 default-purpose bbox cache 下非空；所有 Cube prim 自身有 xformOp:scale；Nav2 导出的 map.pgm 中可见该资产占用区域；目标部件的把手/接触区域几何稳定，可由工具从 contact_link bbox 或 handle 命名推断接触点。

## 挂接路径契约（attach_collision_utils.py）

- 两段路径职责不同：asset_root+path 在磁盘上定位 Aligned_obj.usd；prim_path_child 指定刚体根（RigidBody 语义）；attach_prim_path_children 指定抓取后挂接的碰撞 Prim。prim_path_child 不能改成深层 collision Mesh，否则改变刚体根语义。
- join_prim_path()（第 22 行）：用 Sdf.Path.AppendPath 组合 USD path，空路径与非法路径抛 ValueError。
- has_nonempty_collision_bound()（第 34 行）：Gprim + BBoxCache（default/render/proxy/guide 四 purpose、useExtentsHint=False）计算局部 bbox，三轴尺寸必须都 > 0。
- collision_candidate_paths()（第 55 行）：枚举刚体根下带 CollisionAPI、collisionEnabled 不为 false、bbox 非空的全部候选。
- _configured_children()（第 83 行）：单数/复数互斥；复数必须是去重非空列表；任何冲突返回结构化失败码。
- resolve_attach_collision_prims()（第 107 行）：显式路径逐项校验存在性、位于刚体根之下、可碰撞、bbox 非空；缺配置时走自动发现：唯一候选自动采用，多候选或无候选结构化失败。失败码：ATTACH_COLLISION_CONFIG_CONFLICT、ATTACH_COLLISION_CONFIG_INVALID、ATTACH_COLLISION_PRIM_OUTSIDE_RIGID_ROOT、ATTACH_COLLISION_PRIM_NOT_FOUND、ATTACH_COLLISION_PRIM_NOT_COLLIDABLE、ATTACH_COLLISION_CONFIG_MISSING、ATTACH_COLLISION_PRIM_AMBIGUOUS；attach 契约检查（collision_scene_manager）还会报 ATTACH_COLLISION_PRIM_NOT_IN_CUROBO_WORLD。
- 自动发现优先级：_preferred_discovery_candidates()（第 70 行）先取 guide-purpose 碰撞代理，再退回全部候选。
- RigidObject 仍可加载作为渲染或干扰物，Pick/Probe 在规划前拒绝未解析的 attach 契约。

## 三阶段修复计划

### 阶段一 · 逐资产显式补配置
- 输入：待验证任务的 simbox_task.yaml、Aligned_obj.usd、初始 CuRobo Probe 结果。
- 输出：资产级 manifest（对象名、USD、刚体根、挂接路径、碰撞覆盖方式、Probe/Pick 结果）；经审计的显式 attach_prim_path_child。审计脚本独立（audit_bench21_attach_prims.py），不扩展 normalize_bench_paths.py。
- 验收标准：Probe 不再报 ATTACH_PRIM_NOT_IN_CUROBO_WORLD；attach_prim_valid=true 且至少一条 joint-success 轨迹；Probe 成功与 Pick 成功分开记录；源 USD 不修改，临时结果写入独立 output 目录。

### 阶段二 · 加载与挂接接口重构（多 Prim 契约）
- 输入：单 collision leaf 与多 collision leaf 资产、现有 RigidObject 配置与历史失败码。
- 输出：rigid_prim_path + attach_collision_prim_paths 契约；attach_collision_utils.py；collision_scene_manager 的 _discover()/attach_target()/refresh_after_task_reset()；清晰失败码与候选清单。
- 验收标准：运行结果不依赖 USD child 顺序；每个被挂接 Prim 通过存在性、碰撞语义、CuRobo world 三层检查；单 Prim 与多 Prim 资产都能表达；错误在对象初始化或 Probe 阶段暴露，不拖到真实 Pick 才失败。

### 阶段三 · 源资产统一交付结构
- 输入：制造方/转换流水线交付规范（rigid_body_prim 唯一稳定、attach collision proxy 覆盖完整物体、visual 与 collision 分离、USD 内部引用用交付包内相对路径、metadata 声明）。
- 输出：统一 /Aligned + /AttachCollisionProxy（或 metadata 声明的明确复数列表）结构；任务 YAML 只引用 metadata 稳定字段或由转换器自动写入；删除 children[0] 回退与资产名特判。
- 验收标准：同类别资产 Prim 语义一致，不要求按对象写特殊规则；任意安装路径加载后 rigid root、collision proxy、抓取标注可自动解析；CuRobo world 能查询全部声明 attach collision Prim；资产通过 Probe 后剩余失败可明确归因于真实抓取物理。

### 阶段执行顺序
- 先执行阶段一恢复验证，同步设计阶段二接口；把阶段一的审计结果整理成阶段三的制造方修复清单；阶段三完成后删除临时 YAML 特例与旧回退逻辑。
