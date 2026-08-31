# Bench2.1 可抓取资产挂接路径问题与三阶段修复计划

## 1. 全局判断

这个问题位于 SimBox 的“资产加载”和“Pick 执行”之间。系统不仅需要知道物体的 USD 文件在哪里，还要在 USD Stage 内明确区分：

1. 哪个 Prim 是物体的刚体根节点；
2. 哪个 Prim 或哪些 Prim 是 CuRobo 使用的碰撞几何；
3. 抓住物体后，应把哪些碰撞几何从世界环境挂接到机械臂末端。

当前根因可以概括为：

> 大多数 Bench2.1 任务没有配置 `attach_prim_path_child`；`RigidObject` 在字段缺失时直接选择刚体根节点的第一个 child；这个回退规则适用于部分官方扁平资产，却不符合多数 Bench2.1 资产的嵌套 USD 结构，最终把普通 `Xform` 节点当成了可挂接碰撞体。

因此，用户当前的理解基本正确，但需要补充两个边界：

- **缺少 `attach_prim_path_child` 本身不一定是错误。**如果资产满足官方旧假设，即刚体根节点的第一个 child 就是完整碰撞 Mesh，回退逻辑可以工作。
- **真正的错误是“配置没有明确说明”与“代码默认假设不成立”同时发生。**只责怪 YAML 或只责怪资产都不完整。

这也不是普通的磁盘文件路径归一化错误。`asset_root` 和 `path` 负责找到 `Aligned_obj.usd`；`attach_prim_path_child` 负责进入已经加载的 USD Stage，定位内部 Prim。两者是不同层级的“路径”。

## 2. 当前失败是怎样发生的

### 2.1 配置层：只声明了刚体根，没有声明挂接碰撞体

多数 Bench2.1 任务对象类似下面这样：

```yaml
- name: bedroom_phone_0_id9003
  target_class: RigidObject
  path: .../Aligned_obj.usd
  prim_path_child: Aligned
```

其中：

- `path`：磁盘上的 USD 文件路径；
- `prim_path_child: Aligned`：USD 加载后用作 `RigidPrim` 的刚体根；
- `attach_prim_path_child`：应该交给 CuRobo 检查并在抓取后挂到机器人上的碰撞 Prim，目前未填写。

Phone 配置的实际位置见 [`bedroom_phone_placement/simbox_task.yaml` 第 88–100 行](../InternDataAssets/Bench_2.1_isaacsim/scene_4/04_bedroom/assets/basic/bedroom_phone_placement/simbox_task.yaml#L88-L100)。

`prim_path_child` 和 `attach_prim_path_child` 不能混为一个字段。前者回答“哪个节点代表刚体”，后者回答“抓取时挂接哪块碰撞几何”。不能为了修复挂接路径，把 `prim_path_child` 直接改成很深的 collision Mesh 路径，否则会改变物理刚体的根节点语义。

### 2.2 历史加载层问题：缺少字段时选择 `children[0]`

以下是迁移前的历史逻辑，现已不再作为标准路径：

```python
rigid_prim_path = os.path.join(self.base_prim_path, cfg["prim_path_child"])
attach_prim_path_child = cfg.get("attach_prim_path_child")
if attach_prim_path_child:
    self.mesh_prim_path = os.path.join(self.base_prim_path, str(attach_prim_path_child))
else:
    children = get_prim_at_path(rigid_prim_path).GetChildren()
    self.mesh_prim_path = str(children[0].GetPrimPath())
```

历史代码位置仅用于说明根因；当前 [`rigid_object.py`](../workflows/simbox/core/objects/rigid_object.py) 使用 `rigid_prim_path` 与 `attach_collision_prim_paths`，并通过碰撞语义解析显式或唯一候选。

`GetChildren()` 返回 USD 层级中直接子节点的列表。`children[0]` 只表示“结构上的第一个子节点”，不表示：

- 它带有 `CollisionAPI`；
- 它覆盖整个物体；
- 它已经进入 CuRobo collision world；
- 它适合在抓取后挂到机器人上。

官方部分旧资产的结构较扁平，所以历史回退时第一个 child 恰好是碰撞 Mesh：

```text
/World/Aligned                 # RigidBody
└── /Scan                      # Mesh + Collision
```

官方任务也只配置 `prim_path_child: Aligned`，见 [`omniobject3d-bottle.yaml` 第 34–44 行](../workflows/simbox/core/configs/tasks/pick_and_place/split_aloha/single_pick/right/omniobject3d-bottle.yaml#L34-L44)。这解释了旧回退逻辑为何曾经对部分资产有效。

多数 Bench2.1 资产则类似：

```text
/World/Aligned                 # RigidBody
└── /Normalize                 # Xform，只负责尺度/坐标归一化
    └── /Source
        └── /base_link
            ├── /visuals
            └── /collisions
                └── /collision_mesh_*
```

于是代码选中 `/Aligned/Normalize`。它是中间变换节点，不是可挂接的完整碰撞障碍物。

### 2.3 规划层：CuRobo 按完整 Prim path 精确查询

`GraspPlanEvaluator` 不会根据名称猜测子孙碰撞体，而是执行精确查询：

```python
world.get_obstacle(prim_path)
```

代码位置：[`grasp_plan_evaluator.py` 第 41–58 行](../workflows/simbox/core/planning/grasp_plan_evaluator.py#L41-L58)。

当传入 `/World/task_0/<object>/Aligned/Normalize` 时，CuRobo world 中没有以这个 Xform path 注册的 obstacle，因而返回：

```text
ATTACH_PRIM_NOT_IN_CUROBO_WORLD
```

这不等于“没有加载物体”，也不等于“没有抓取标注”。它只表示：系统无法把当前给出的 Prim path 映射到 CuRobo 已知的碰撞障碍物。

### 2.4 Pick 层：同一个错误路径继续传给 attach

当前 Pick 使用 `self.pick_obj.attach_collision_prim_paths` 做两件事：

1. 在 Probe 阶段验证它是否存在于 CuRobo world；
2. 真正闭合夹爪后，把显式路径列表传给 `attach_objects`。

代码位置：[`pick.py` 第 106–120 行](../workflows/simbox/core/skills/pick.py#L106-L120)和 [`pick.py` 第 149–170 行](../workflows/simbox/core/skills/pick.py#L149-L170)。

当前主路径不再调用旧 CuRobo attach wrapper。`CollisionSceneManager.attach_target()` 将显式路径列表传给 [`TemplateController.attach_objects()`](../workflows/simbox/core/controllers/template_controller.py#L1911)，由 controller 调用 native v2 attachment manager：

```python
self.planner.attachment_manager.attach(
    cu_js, attachment_meshes, link_name="attached_object",
    disable_obstacle_names=attach_prim_paths,
)
```

普通执行使用 native `MotionPlanner`；抓取候选使用 native `BatchMotionPlanner`。两者消费同一组显式 attach collision paths。

因此，如果一个资产需要多个 collision leaf Prim 才能覆盖完整物体，当前单路径接口也无法完整表达。这是比 `children[0]` 更深一层的接口限制。

## 3. 现有证据说明了什么

### 3.1 七个 Bench2.1 样例停在 attach gate

8 个运行样例中，7 个在 Geometry 通过后，以 `ATTACH_PRIM_NOT_IN_CUROBO_WORLD` 停止；只有已经显式填写挂接路径的 hand cream 进入了真实 Pick。汇总见 [`workspace_annulus_v3_validation_summary.md` 第 12–25 行](../output/workspace_annulus_v3_validation_summary.md#L12-L25)。

这说明工作点生成、抓取标注、CuRobo 轨迹和真实夹取应分层判断，不能把 attach 错误笼统称为“点位不可达”。

### 3.2 盐瓶 A/B 实验直接定位了根因

同一个盐瓶、同一个候选点 `annulus_004`：

- 原配置已经得到左臂 14 条、右臂 17 条联合成功的 pre-grasp/grasp 轨迹；
- 但自动回退选中了 `/Aligned/Normalize`，`attach_prim_valid: false`，因此最终 `feasible: false`。

原始结果见 [`kitchen_salt_bottle_placement/candidates.json` 第 403–434 行](../output/workspace_annulus_v3_cases/kitchen_salt_bottle_placement/candidates.json#L403-L434)。

临时配置只增加：

```yaml
attach_prim_path_child: Aligned/Normalize/Source/base_link/visuals
```

配置位置见 [`salt_visuals/source_task.yaml` 第 540–549 行](../output/workspace_attach_prim_ab/salt_visuals/source_task.yaml#L540-L549)。在该盐瓶资产中，这个 Prim 实际被注册为 collision obstacle；修改后左右臂的轨迹数量保持为 14 和 17，但 `attach_prim_valid` 变为 `true`，结果变为 `feasible: true`：

- [`annulus_004.left.json` 第 1–14 行](../output/workspace_attach_prim_ab/salt_visuals/probes/annulus_004/results/annulus_004.left.json#L1-L14)
- [`annulus_004.right.json` 第 1–14 行](../output/workspace_attach_prim_ab/salt_visuals/probes/annulus_004/results/annulus_004.right.json#L1-L14)

这个 A/B 只证明“挂接 Prim 是原 Probe 阻塞点”，不代表应该给所有资产统一填写 `/visuals`。不同 Bench 资产的碰撞层级并不一致，必须逐资产审计。

### 3.3 Hand cream 说明 attach 通过也不等于 Pick 成功

Hand cream 已显式配置：

```yaml
attach_prim_path_child: Aligned/Normalize/Source/base_link/collisions
```

位置见 [`bedroom_hand_cream_to_organizer/simbox_task.yaml` 第 88–101 行](../InternDataAssets/Bench_2.1_isaacsim/scene_4/04_bedroom/assets/basic/bedroom_hand_cream_to_organizer/simbox_task.yaml#L88-L101)。它能够通过 attach 检查和 CuRobo 规划，但 seed 0 的真实 Pick 仍未通过内置接触/过程成功条件。

所以当前有两个不同问题：

1. 多数资产先被错误 attach Prim 阻塞；
2. 少数通过 attach 的资产，还要继续验证真实夹爪接触、摩擦、质量、抓取姿态和抬升过程。

## 4. 各环节责任边界

| 环节 | 当前问题 | 责任判断 |
|---|---|---|
| `normalize_bench_paths.py` | 只修复 `asset_root`、`arena_file`、机器人/HDR 路径和墙面姿态，不分析 USD 内部 Prim | 不是本次直接根因；不应把资产语义推断硬塞进普通文件路径归一化脚本 |
| 任务转换/配置生成 | 大多数 `RigidObject` 没有明确的 attach collision 信息 | 主要配置缺口；转换阶段没有把 USD 碰撞结构转成 SimBox 可执行契约 |
| `RigidObject` | 缺少字段时盲选 `children[0]`，只检查 Prim 是否存在，不检查碰撞语义 | 直接代码原因；默认规则过强，错误发现过晚 |
| CuRobo evaluator | 按精确路径检查 obstacle | 行为正确；它把上游错误暴露出来，而不是制造错误 |
| Controller | 只接受一个 attach path | 工程能力不足；无法自然覆盖多 collision leaf 资产 |
| Bench2.1 资产 | 碰撞层级不统一，缺少稳定 attach proxy 和机器可读元数据 | 资产契约不足；使用方只能逐个猜测或扫描 |
| 环形点位规划器 | 生成底盘候选并做几何过滤 | 不是 attach 错误根因；同点位 A/B 已经证明这一点 |

当前路径归一化脚本的实际职责见 [`normalize_bench_paths.py` 第 35–82 行](../scripts/simbox/normalize_bench_paths.py#L35-L82)。它只重写 YAML 中的磁盘/运行配置字段，没有打开 USD Stage，也没有检查 `CollisionAPI`。

## 5. 三阶段修复计划

三个方案不是互斥选择，而是从“尽快恢复验证”到“彻底消除结构歧义”的递进关系：

```text
方案一：逐资产显式补配置，尽快解除当前阻塞
  ↓
方案二：重构加载和挂接接口，让错误可检测、可表达
  ↓
方案三：统一源资产交付契约，从资产侧消除猜测
```

### 5.1 方案一：显式配置修复与逐资产验证

#### 目标

在不大改运行时代码的前提下，让当前选定的 Bench2.1 Pick 目标尽快进入 CuRobo 和真实 Pick 验证。

#### 实施步骤

1. 新增独立审计脚本 `scripts/simbox/audit_bench21_attach_prims.py`，不要扩展 `normalize_bench_paths.py`。
2. 对每个 Pick 目标打开 `Aligned_obj.usd`，记录：
   - `prim_path_child` 对应的刚体根是否存在；
   - 其子孙中所有带碰撞语义的 Prim；
   - 候选是单个完整 proxy、父级 collision Prim，还是多个 leaf Mesh；
   - 每个候选能否在初始化后的 CuRobo world 中按完整 path 查询到。
3. 只有审计确认后，才在对应 `simbox_task.yaml` 中显式增加：

   ```yaml
   prim_path_child: Aligned
   attach_prim_path_child: Aligned/<audited-collision-prim>
   ```

4. 先运行 planning-only Probe。要求 `attach_prim_valid: true`，并至少存在一条 joint-success grasp。
5. Probe 通过后，再运行 seed 0 Pick；成功后再运行 seed 1、2。
6. 输出一份资产级 manifest，记录“对象名、USD、刚体根、挂接路径、碰撞覆盖方式、Probe/Pick 结果”，避免以后重复人工判断。

#### 禁止做法

- 不能把盐瓶实验中的 `/visuals` 复制给所有物体；
- 不能把 `prim_path_child` 改成深层 Mesh；
- 不能仅凭 Prim 名字包含 `collision` 就认定它覆盖完整物体；
- 不能为了让 Probe 通过而选择只覆盖局部几何的路径。

#### 验收标准

- 每个待验证目标都有明确、存在且可被 CuRobo 查询的 attach Prim；
- Probe 报告不再出现 `ATTACH_PRIM_NOT_IN_CUROBO_WORLD`；
- Probe 成功和 Pick 成功分别记录，不能混为一个状态；
- 源 USD 不修改，所有临时结果写入独立 output 目录。

#### 优缺点

- 优点：修改小、见效快，适合先恢复 8 个样例的验证。
- 缺点：配置容易重复；多 Mesh 资产仍然无法由当前单路径接口完整表达；不能作为最终架构。

### 5.2 方案二：加载器与多 Prim 挂接接口重构

#### 目标

让运行时代码按碰撞语义选择和验证 Prim，而不是依赖 USD 子节点顺序；同时支持一个刚体对应多个碰撞 Prim。

#### 接口设计

当前运行时代码已经完成以下字段拆分：

```python
self.rigid_prim_path: str
self.attach_collision_prim_paths: list[str]
```

配置接口改为复数形式：

```yaml
prim_path_child: Aligned
attach_prim_path_children:
  - Aligned/AttachCollisionProxy
```

当前任务编译器以 `attach_prim_path_children` 作为规范配置；旧单数字段只属于任务配置迁移，不会作为 CuRobo API 传入 planner。CuRobo v1 兼容层已经删除，运行时不存在双 planner 分支。

#### `RigidObject` 修改

修改 [`rigid_object.py`](../workflows/simbox/core/objects/rigid_object.py)：

1. 使用 USD `Sdf.Path` 组合 Prim path，不再用面向磁盘路径的 `os.path.join` 表达 USD path；
2. 显式配置优先，但必须逐项验证：
   - Prim 存在；
   - Prim 具有碰撞语义或能映射到 CuRobo obstacle；
   - 路径位于当前物体刚体根之下；
3. 缺少配置时可以做受限自动发现：
   - 只有一个无歧义、能覆盖整个物体的 collision candidate 时自动采用；
   - 找到多个 leaf 或没有候选时立即失败，并列出候选路径；
   - 禁止继续使用 `children[0]` 静默猜测；
4. 使用清晰失败码区分：
   - `ATTACH_COLLISION_CONFIG_MISSING`
   - `ATTACH_COLLISION_PRIM_NOT_FOUND`
   - `ATTACH_COLLISION_PRIM_NOT_COLLIDABLE`
   - `ATTACH_COLLISION_PRIM_AMBIGUOUS`
   - `ATTACH_COLLISION_PRIM_NOT_IN_CUROBO_WORLD`

#### CuRobo 与 Controller 修改

修改 [`grasp_plan_evaluator.py`](../workflows/simbox/core/planning/grasp_plan_evaluator.py)和 [`template_controller.py`](../workflows/simbox/core/controllers/template_controller.py)：

1. evaluator 接收 `list[str]`，逐项检查，并把缺失路径完整写入结果 JSON；
2. `attach_obj` 改为 `attach_objects(obj_prim_paths: list[str])`，原样传给 CuRobo；
3. Probe 与 Pick 继续共用同一个 evaluator，禁止两套判断逻辑；
4. attach 前后记录 collision world 中被移除/挂接的 obstacle，方便定位残留碰撞或重复挂接。

#### 测试

至少增加以下测试：

- 官方扁平单 Mesh 资产；
- Bench 嵌套、单一 collision container 资产；
- Bench 多 collision leaf 资产；
- 缺字段但只有一个无歧义候选；
- 多候选时明确失败而不是选择第一个；
- 配置路径存在但没有碰撞语义；
- Probe 和 Pick 接收到完全相同的 attach path 列表；
- 多 Prim attach 后，世界碰撞和机器人附着模型一致。

#### 验收标准

- 运行结果不再依赖 USD child 顺序；
- 任何被挂接的 Prim 都已通过存在性、碰撞语义和 CuRobo world 三层检查；
- 单 Prim 和多 Prim 资产都能表达；
- 错误在对象初始化或 Probe 阶段明确暴露，不拖到真实 Pick 中才失败。

#### 优缺点

- 优点：解决当前代码的结构性缺陷，错误信息清晰，可兼容不同资产结构。
- 缺点：仍需面对不规范源资产；自动发现只能处理无歧义情况，不能替代资产语义标注。

### 5.3 方案三：源资产与制造方交付规范统一

#### 目标

从资产源头提供一个稳定、完整、唯一的抓取挂接碰撞代理，使任务配置和运行时代码都不需要猜测 USD 层级。

#### 推荐 USD 结构

```text
/World/Aligned                         # 唯一刚体根，RigidBodyAPI
├── /Visual                            # 只负责渲染
└── /AttachCollisionProxy              # 完整物体碰撞代理，CollisionAPI
```

如果单个代理无法满足精度要求，则允许明确的复数列表：

```text
/World/Aligned
├── /Visual
└── /AttachCollisions
    ├── /part_00
    ├── /part_01
    └── /part_02
```

但必须在 metadata 中声明，而不能让使用方遍历名称猜测：

```yaml
rigid_body_prim: Aligned
attach_collision_prims:
  - Aligned/AttachCollisionProxy
```

#### 制造方交付要求

1. `rigid_body_prim` 唯一且稳定；
2. attach collision proxy 覆盖完整物体，不能只覆盖一个局部；
3. Visual 与 Collision 分离，不能依赖名为 `visuals` 的 Prim 同时承担碰撞语义；
4. 所有 collision Prim 具有明确 approximation、质量、摩擦和尺度信息；
5. USD 内部引用使用交付包内相对路径；
6. 每个可抓取对象交付：
   - USD Stage 树摘要；
   - rigid/visual/collision Prim 清单；
   - 抓取标注文件；
   - Isaac Sim 加载检查；
   - CuRobo obstacle/attach 检查；
   - 至少一次官方机器人 Pick smoke test。

#### 引擎收敛方式

当 Bench 资产全部迁移到统一契约后：

1. 任务 YAML 只引用 metadata 中的稳定字段，或由转换器自动写入；
2. 删除 `children[0]` 回退逻辑；
3. 对声明为可 Pick 的对象，把 attach collision metadata 设为必填；
4. 将资产契约检查放入交付验收和 CI，而不是等到任务运行时再发现。

#### 验收标准

- 同一类别资产具有一致的 Prim 语义，不要求使用方按对象写特殊规则；
- 从任意安装路径加载后，rigid root、collision proxy、抓取标注均可自动解析；
- CuRobo world 能查询全部声明的 attach collision Prim；
- 资产通过 Probe 后，剩余失败可以明确归因于真实抓取物理，而不是路径或层级歧义。

#### 优缺点

- 优点：从根源消除猜测，配置最简洁，最适合规模化任务生成和 Agent 自动布置。
- 缺点：需要资产制造方或资产转换流水线重新导出并回归验证，周期最长。

## 6. 推荐执行顺序

建议按以下顺序推进：

1. **立即执行方案一**：先审计当前 8 个验证目标，显式补齐经过验证的 attach path，恢复 Probe 与 Pick 测试。
2. **同步设计方案二接口**：不要继续给 `RigidObject` 增加更多名称特判；直接把单路径和 `children[0]` 假设改成明确的碰撞 Prim 列表契约。
3. **把方案一的审计结果交给方案三**：将所有特殊层级、visual/collision 混用、多 leaf 情况整理成制造方修复清单。
4. **方案三完成后做代码收敛**：删除临时 YAML 特例和旧回退逻辑，只保留统一资产契约。

阶段性成功定义如下：

| 阶段 | 成功定义 |
|---|---|
| 方案一 | 当前目标不再被 attach path 阻塞，能够进入真实 Pick |
| 方案二 | 引擎不依赖 child 顺序，单/多碰撞 Prim 都可验证和挂接 |
| 方案三 | 新资产无需手工猜路径，交付时即可自动通过契约检查 |

## 7. 最终结论

当前问题不是“机器人看不到物体”，也不主要是“推荐点位不对”。更准确的描述是：

> 系统已经能加载物体、读取抓取标注，并且在部分点位上计算出可行抓取轨迹；但对于多数 Bench2.1 资产，任务配置没有声明应该挂接哪个碰撞 Prim，加载器又错误地用第一个 child 代替这个语义，导致 CuRobo 无法确认和挂接目标碰撞体。

短期应逐资产显式补配置，中期应重构为可验证的多 Prim 接口，长期应由资产交付提供统一的 attach collision proxy 和 metadata。三层同时推进，才能避免今天修一个 YAML、明天换一个资产又重复失败。
