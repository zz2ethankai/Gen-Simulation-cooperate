# SCENE_7 / SCENE_8 新资产自动适配 Agent 指南

本文面向后续 agent，用于把 `InternDataAssets/assets/custom/scene_7/<scene>` 或 `InternDataAssets/assets/custom/scene_8/<scene>` 下的新资产组适配到当前 SimBox 项目，至少完成静态闭环检查，并在条件允许时跑过 scene loading / reset 阶段。内容来自 `scene_7/01_kitchen` 和 `scene_8` 最新适配结果以及当前代码状态。

关键结构约定：scene_7 和 scene_8 的 basic task 结构同构，唯一直接发现和读取的任务入口都是该任务目录下的 `simbox_task.yaml`。以 scene_8 厨房参考任务为例，结构样板和可读取入口是 `InternDataAssets/assets/custom/scene_8/01_kitchen/assets/basic/kitchen_apple_to_tray/simbox_task.yaml`；`asset_root`、`arena_file`、`env_map`、`robots`、`objects`、`regions`、`source_regions`、`positions`、`skills` 等可读配置都在这个 YAML 里。arena、HDR、USD、texture lib 只能作为这些字段引用到的后续资源解析，不能另建一套按目录猜测的资产清单，也不能把外层 `task.yaml` / `arena.yaml` 当作运行结构来源。

## 适用范围

- 当前已验证参考组：
  - `InternDataAssets/assets/custom/scene_7/01_kitchen/assets/basic/*`
  - `InternDataAssets/assets/custom/scene_8/{01_kitchen,02_bookroom,03_livingroom,04_bedroom}/assets/basic/*`
- 参考任务：
  - `kitchen_apple_to_tray`
  - `kitchen_breakfast_setup`
  - `kitchen_cup_transfer`
  - `kitchen_prep_assembly`
  - `kitchen_salt_bottle_placement`
- 唯一任务入口文件：
  - 每个任务目录下的 `simbox_task.yaml`
- 入口 YAML 字段引用后才解析的外部资源：
  - `arena_file` 指向的 arena YAML，通常是同目录 `simbox_arena.yaml`
  - `env_map.envmap_lib` 指向的 HDR 库
  - `robots[].path`、`objects[].path`、arena `fixtures[].path` 指向的 USD
  - scene family 共享资产，例如 `InternDataAssets/assets/custom/scene_8/shared_assets/*`，但只能在入口 YAML 字段引用后读取
  - 共享运行代码 `workflows/simbox/core/tasks/banana.py`
  - pick/contact 相关 skill 文件

不要把本文直接套到 scene_4 的 floor-center-relative positions 规则；scene_4 有单独约束。

## 参数约定

后续 agent 不要把 `01_kitchen` 写死在适配逻辑里。除非用户明确指定别的目录，统一用下面变量枚举 inner entry；变量指向目录时只表示枚举范围，不表示这些目录本身携带可读取任务结构：

```bash
SCENE_ROOT=InternDataAssets/assets/custom/scene_8/01_kitchen
BASIC_ROOT="$SCENE_ROOT/assets/basic"
```

对新 scene，例如 `02_bookroom` 或 scene_7 的 `01_kitchen`，只需要把 `SCENE_ROOT` 改成对应目录；结构读取规则不变。所有检查都从 `$BASIC_ROOT/*/simbox_task.yaml` 发现任务。不要从 room 目录直接推断“可读取文件集合”，也不要扫描外层 `task.yaml` / `arena.yaml` 来补结构；必须先读取每个 inner `simbox_task.yaml`，并以其中的 `arena_file`、`env_map`、`robots`、`objects`、`regions`、`skills` 等字段作为唯一结构来源来解析后续资源。

scene_8 这类一组 4 个房间、20 个 basic task 的资产包，可以从 family root 批量检查；这里的 family root 只用于枚举 `*/assets/basic/*/simbox_task.yaml`，不会读取 family/root 目录下的其它 YAML 作为结构来源：

```bash
SCENE_FAMILY_ROOT=InternDataAssets/assets/custom/scene_8
/home/dyf/miniconda3/envs/anygrasp/bin/python \
  scripts/simbox/validate_custom_scene_assets.py "$SCENE_FAMILY_ROOT"
```

## 总体原则

1. 先以每个 `simbox_task.yaml` 为入口做静态闭环，再启动 Isaac。
2. 路径都必须按运行时 `asset_root` 解析，不按当前 shell 工作目录猜。
3. `source_name` 只当元数据保留，运行时字段必须引用实际 `name`。
4. `random_config` 只能传 sampler 函数签名支持的参数。
5. contact sensor 不是 scene loading 的硬依赖；初始化失败时必须可诊断、可降级。
6. 资产 YAML 当前可能不被 git 跟踪，不能只看 `git status` 判断是否改到。
7. 不要把所有双下划线都当错误；scene_8 中 `*_candidate_region` 和 `*__support_plane_*` 可以是真实 region/fixture 名。判断标准是运行时引用是否能在 `objects`、`fixtures`、`robots` 或 `regions` 中闭环。

## 一、发现任务集合

从 scene 根目录枚举 basic 任务入口；枚举到的每个文件都必须是 `*/assets/basic/<task>/simbox_task.yaml`：

```bash
find "$BASIC_ROOT" -maxdepth 2 -name simbox_task.yaml -print | sort
```

期望 `01_kitchen` 当前有 5 个 `simbox_task.yaml`。

scene_8 当前期望有 20 个 `simbox_task.yaml`：

```bash
find InternDataAssets/assets/custom/scene_8 \
  -path '*/assets/basic/*/simbox_task.yaml' -print | sort
```

检查每个任务入口声明的 arena 是否存在。不要硬编码 `simbox_arena.yaml`；先从 `simbox_task.yaml` 读取 `arena_file`，再按 task YAML 所在目录解析该引用：

```bash
BASIC_ROOT="$BASIC_ROOT" /home/dyf/miniconda3/envs/anygrasp/bin/python - <<'PY'
from pathlib import Path
import os
import yaml

for path in sorted(Path(os.environ["BASIC_ROOT"]).glob("*/simbox_task.yaml")):
    task = yaml.safe_load(path.read_text())["tasks"][0]
    arena_ref = Path(task.get("arena_file", "simbox_arena.yaml"))
    arena = arena_ref if arena_ref.is_absolute() else path.parent / arena_ref
    if not arena.exists():
        print(f"missing arena: {path} -> {arena}")
PY
```

失败处理：
- 缺 `arena_file` 指向的 YAML 时，不要先改 task；先确认资产包是否完整。
- `arena_file: simbox_arena.yaml` 在当前运行链路是可用的，因为该字段来自 task YAML，`simbox_dual_workflow.py` 会按 task YAML 所在目录做 fallback。

## 二、修正路径字段

每个 inner `simbox_task.yaml` 的关键路径必须从运行进程 cwd 可解析。`BananaBaseTask` 使用 `os.path.abspath(self.cfg["asset_root"])`，所以不要依赖 task 文件所在目录来解释 `../../..`。scene_8 的结构应与参考入口 `InternDataAssets/assets/custom/scene_8/01_kitchen/assets/basic/kitchen_apple_to_tray/simbox_task.yaml` 一致：可读取结构都在 task YAML 中，`asset_root`、`arena_file`、`env_map`、`robots`、`objects`、`regions`、`source_regions`、`positions`、`skills` 都从该 YAML 读取。

scene_8 当前可行路径字段值如下。注意这只是字段取值，不改变“结构全部从内层 `simbox_task.yaml` 读取”的规则：

```yaml
asset_root: InternDataAssets/assets/custom/scene_8/01_kitchen
env_map:
  envmap_lib: ../shared_assets/envmap_lib
robots:
- name: split_aloha
  path: ../shared_assets/split_aloha_mid_360/robot.usd
```

原因：
- `BananaBaseTask._set_envmap()` 使用 `os.path.join(self.asset_root, envmap_lib, "*.hdr")`。
- 历史错误 `empty range for randrange() (0, 0, 0)` 是 envmap glob 为空造成的。
- scene_8 每个房间使用 repo-relative room root，并通过 family root 下的 `shared_assets` 找共享 HDR 和 robot。

scene_7 历史资产组也使用同样的内层 `simbox_task.yaml` 结构；只有路径字段值可能不同，例如旧参考值是 `envmap_lib: ../../../../../workflows/simbox/example_assets/envmap_lib` 和 `robots[].path: ../../../../../workflows/simbox/example_assets/split_aloha_mid_360/robot.usd`。

对 `02_bookroom`、`03_livingroom`、`04_bedroom`，只替换 `asset_root` 中的房间名；`envmap_lib` 和 robot `path` 保持相同。

单 scene 静态检查：

```bash
/home/dyf/miniconda3/envs/anygrasp/bin/python - <<'PY'
from pathlib import Path
import glob, os, yaml

root = Path("InternDataAssets/assets/custom/scene_8/01_kitchen/assets/basic")
failed = False
for path in sorted(root.glob("*/simbox_task.yaml")):
    task = yaml.safe_load(path.read_text())["tasks"][0]
    asset_root = os.path.abspath(task["asset_root"])
    hdrs = glob.glob(os.path.join(asset_root, task["env_map"]["envmap_lib"], "*.hdr"))
    arena_ref = Path(task.get("arena_file", "simbox_arena.yaml"))
    arena = arena_ref if arena_ref.is_absolute() else path.parent / arena_ref
    missing_robots = []
    for robot in task.get("robots", []):
        full = os.path.abspath(os.path.join(asset_root, robot.get("path", "")))
        if not os.path.exists(full):
            missing_robots.append(full)
    missing_objects = []
    for obj in task.get("objects", []):
        full = os.path.abspath(os.path.join(asset_root, obj.get("path", "")))
        if obj.get("path") and not os.path.exists(full):
            missing_objects.append(obj["path"])
    print(path.parent.name, "hdrs=", len(hdrs), "arena=", arena.exists(),
          "robots_missing=", len(missing_robots),
          "objects_missing=", len(missing_objects))
    if not hdrs or not arena.exists() or missing_robots or missing_objects:
        failed = True
if failed:
    raise SystemExit(1)
PY
```

通过标准：
- 每个任务 `hdrs=1`
- `arena=True`
- `robots_missing=0`
- `objects_missing=0`

scene_8 批量检查优先使用仓库脚本：

```bash
/home/dyf/miniconda3/envs/anygrasp/bin/python \
  scripts/simbox/validate_custom_scene_assets.py \
  InternDataAssets/assets/custom/scene_8
```

2026-07-02 复核后的 scene_8 状态：
- 路径层已经闭环：20 个内层 `simbox_task.yaml` 的 `asset_root` 指向各自 room root，`envmap_lib: ../shared_assets/envmap_lib`，robot path 指向 `../shared_assets/split_aloha_mid_360/robot.usd`。
- object path、fixture path、texture lib、`regions` 引用、`spawn_region` 引用已按内层入口检查。
- 当前 `kitchen_apple_to_tray/simbox_task.yaml` 已有 5 个 DAG skill，可通过静态检查；其余 19 个入口 YAML 仍是空 `skills` 队列，例如 `base: [] / left: [] / right: []`。
- 空 legacy 队列不是可运行任务；`plan_first_skill()` 会对队列取 `[0]` 并触发 `IndexError: list index out of range`。因此 scene_8 family-root 静态检查应当失败，直到每个入口 YAML 写入可执行 skill graph，或运行时代码明确支持 scene-only 空 skill 任务。
- `sampler_extra` 仍会存在，因为 YAML 保留了 `support_surface_z` 等生成器元数据；只有在 `banana.py` 没有签名过滤时才是失败。

代码侧兜底：
- `workflows/simbox/core/tasks/banana.py::_set_envmap()` 必须在 HDR 列表为空时抛清晰 `FileNotFoundError`，包含 `task name / asset_root / envmap_lib / searched path`。

## 三、修正 runtime name 引用

历史错误：

```text
KeyError: 'white_mug_a__0__id9000'
```

根因：
- `objects[].name` 是运行时对象名，如 `white_mug_a_0_id9000`。
- `objects[].source_name` 是来源元数据，如 `white_mug_a__0__id9000`。
- `_set_regions()` 只按 `self._task_objects[cfg["object"]]` 查运行时 `name`，不会查 `source_name`。

需要检查和必要时替换的运行时字段：
- `regions[].object`
- `regions[].target`
- `regions[].container`
- `regions[].target2`
- `regions[].A`
- `regions[].B`
- `source_regions[].A`
- `source_regions[].B`
- `objects[].spawn_region`
- `objects[].placement.spawn_region`
- 由对象/fixture 名派生的 `*_candidate_region`
- `A: robot` 这类旧别名应改成实际 robot name `split_aloha`

不要替换：
- `objects[].source_name`
- `arena.fixtures[].source_name`

批量替换策略：
1. 从 task `objects[].source_name -> objects[].name` 建映射。
2. 从 arena `fixtures[].source_name -> fixtures[].name` 建映射。
3. 同时加入 `source_candidate_region -> name_candidate_region`。
4. 替换 `objects[].spawn_region` 和 `objects[].placement.spawn_region`，使其能在 `regions[].name` 中找到。
5. 文本替换时跳过 `source_name:` 行。

scene_8 注意事项：
- `regions[].target` / `regions[].B` 中的 `sink_counter_base_0_id1__support_plane_z0900` 这类值是 arena 里的真实 fixture 名，不要因为含 `__` 就替换。
- `objects[].spawn_region` 使用 `white_mug_a__0__id9000_candidate_region` 这类值是可接受的，只要 `regions[].name` 里存在同名 region。
- 当前运行时代码只消费 `task["regions"]`；`source_regions` 是来源记录。适配检查应优先保证 `regions` 和 `spawn_region` 闭环，不要为了清理元数据破坏来源追踪。只有当后续代码开始消费 `source_regions`，或人工排查明确要求来源字段同步时，才清理 `source_regions` 的旧别名。

单 room 运行时引用闭环检查示例：

```bash
/home/dyf/miniconda3/envs/anygrasp/bin/python - <<'PY'
from pathlib import Path
import yaml

root = Path("InternDataAssets/assets/custom/scene_8/01_kitchen/assets/basic")
failed = False
for path in sorted(root.glob("*/simbox_task.yaml")):
    task = yaml.safe_load(path.read_text())["tasks"][0]
    arena_ref = Path(task.get("arena_file", "simbox_arena.yaml"))
    arena_path = arena_ref if arena_ref.is_absolute() else path.parent / arena_ref
    arena = yaml.safe_load(arena_path.read_text())
    names = (
        {obj["name"] for obj in task.get("objects", [])}
        | {fixture["name"] for fixture in arena.get("fixtures", [])}
        | {robot["name"] for robot in task.get("robots", [])}
    )
    region_names = {
        region.get("name")
        for region in (task.get("regions", []) or [])
        if isinstance(region, dict) and region.get("name")
    }
    missing = []
    spawn_missing = []
    for idx, region in enumerate(task.get("regions", []) or []):
        if not isinstance(region, dict):
            continue
        for key in ("object", "target", "container", "target2", "A", "B"):
            value = region.get(key)
            if isinstance(value, str) and value not in names:
                missing.append(("regions", idx, key, value))
    for obj in task.get("objects", []) or []:
        for key, value in (
            ("spawn_region", obj.get("spawn_region")),
            ("placement.spawn_region", (obj.get("placement") or {}).get("spawn_region")),
        ):
            if isinstance(value, str) and value not in region_names:
                spawn_missing.append((obj.get("name"), key, value))
    print(path.parent.name, "refs_missing=", len(missing),
          "spawn_region_missing=", len(spawn_missing))
    if missing or spawn_missing:
        print("  missing_sample=", missing[:5])
        print("  spawn_missing_sample=", spawn_missing[:5])
        failed = True
if failed:
    raise SystemExit(1)
PY
```

通过标准：
- `refs_missing=0`
- `spawn_region_missing=0`
- 对运行时字段，不按是否包含 `__` 判定；只按是否能解析到实际 object / fixture / robot / region 判定

## 四、处理 random_config 多余字段

历史错误：

```text
RandomRegionSampler.A_on_B_region_sampler() got an unexpected keyword argument 'support_surface_z'
```

根因：
- `_set_regions()` 会调用 `sampler_fn(obj, tgt, **cfg["random_config"])`。
- 失败重置时 `reset_fixed_rigid_objects()` -> `_get_deterministic_region_pose()` 也会从同一个 region `random_config` 生成固定 pose。
- `A_on_B_region_sampler` 只接受 `pos_range` 和 `yaw_rotation`。
- 新 YAML 把 `support_surface_z` 放进 `random_config`，导致 Python 传入不支持的 keyword。

推荐代码策略：
- 不建议批量删除 YAML 里的 `support_surface_z`。
- 在 `banana.py` 统一用 `_filter_sampler_random_config()` 按 sampler 签名过滤 kwargs。
- 当前代码应覆盖四条路径：
  - 普通 region
  - `target2` region
  - `priority/random_region_list` region
  - `reset_fixed_rigid_objects()` 的 deterministic region pose 路径

检查代码：

```bash
rg -n "def _filter_sampler_random_config|_filter_sampler_random_config\\(" workflows/simbox/core/tasks/banana.py
```

通过标准：
- 至少能看到四处 `_filter_sampler_random_config(` 调用。
- 其中一处必须在 `_get_deterministic_region_pose()` / `reset_fixed_rigid_objects()` 路径中，避免失败重置时继续把 `support_surface_z` 传给 `A_on_B_region_sampler`。
- 如果 run log 里仍有 `Failed to reset workflow state after failed generation` 和 `unexpected keyword argument 'support_surface_z'`，说明运行进程没有加载最新 `banana.py`，或 reset 路径仍未走签名过滤；先重启 Isaac/launcher 并确认容器内代码，再继续判断任务本身失败。

行为验证：

```bash
PYTHONPATH=workflows/simbox /home/dyf/miniconda3/envs/anygrasp/bin/python - <<'PY'
import inspect
from core.utils.region_sampler import RandomRegionSampler

fn = RandomRegionSampler.A_on_B_region_sampler
cfg = {
    "pos_range": [[0, 0, 0], [0, 0, 0]],
    "yaw_rotation": [0, 0],
    "support_surface_z": 0.9,
}
accepted = set(inspect.signature(fn).parameters)
filtered = {key: value for key, value in cfg.items() if key in accepted}
print("accepted=", sorted(accepted))
print("filtered=", sorted(filtered))
assert sorted(filtered) == ["pos_range", "yaw_rotation"]
PY
```

注意：
- 普通 Python 环境直接 import `banana.py` 可能被 Isaac `omni` 依赖挡住；这不是失败，只要 `py_compile` 和签名过滤逻辑通过。

## 五、处理 pick contact view

历史错误：

```text
AttributeError: 'NoneType' object has no attribute 'sensor_count'
```

触发链路：

```text
world.reset()
  -> task.post_reset()
  -> view.initialize()
  -> RigidContactView.initialize()
  -> self._physics_view.sensor_count
```

风险来源：
- `RigidContactView` 的 `prim_paths_expr` 必须匹配真实 physics/collision prim。
- `filter_paths_expr` 也必须匹配 gripper 有效 collision prim。
- USD 还必须已经进入 PhysX tensor backend。
- Isaac 内部 `create_rigid_contact_view()` 返回 `None` 时，错误信息只有 `NoneType.sensor_count`，不会说明哪个 prim 匹配失败。

当前代码必须满足：
- `_set_pickcontact_view()` 中 pick object 优先使用 `obj.mesh_prim_path`，fallback `obj.prim_path`。
- 每个 contact view 带 `_simbox_metadata`，至少包括：
  - `view_kind`
  - `robot`
  - `arm`
  - `object`
  - `prim_paths_expr`
  - `filter_paths_expr`
- `post_reset()` 捕获 `"'NoneType' object has no attribute 'sensor_count'"`，标记 `_simbox_contact_available=False`，打印 metadata，并继续 scene loading。
- pick/scan 类 `get_contact()` 发现 contact unavailable 时返回空接触，不再调用 `get_contact_force_matrix()`。

检查命令：

```bash
rg -n "_simbox_contact_available|contact view initialization failed|mesh_prim_path|get_contact\\(" \
  workflows/simbox/core/tasks/banana.py \
  workflows/simbox/core/skills/pick.py \
  workflows/simbox/core/skills/dexpick.py \
  workflows/simbox/core/skills/manualpick.py \
  workflows/simbox/core/skills/dynamicpick.py \
  workflows/simbox/core/skills/scan.py
```

编译检查：

```bash
/home/dyf/miniconda3/envs/anygrasp/bin/python -m py_compile \
  workflows/simbox/core/tasks/banana.py \
  workflows/simbox/core/skills/pick.py \
  workflows/simbox/core/skills/dexpick.py \
  workflows/simbox/core/skills/manualpick.py \
  workflows/simbox/core/skills/dynamicpick.py \
  workflows/simbox/core/skills/scan.py
```

通过标准：
- 编译无错误。
- grep 能看到 `_simbox_contact_available` 在每个 pick/scan contact 读取点生效。

重要说明：
- 这是 mitigation，不是证明 contact sensor 的 PhysX 根因已修复。
- 如果日志出现 `[simbox] contact view initialization failed...`，agent 下一步必须分析打印出的 `prim_paths_expr/filter_paths_expr`，判断是对象 prim、gripper filter path，还是 USD physics schema 问题。

## 六、完整静态适配检查脚本

适配后至少跑一次：

```bash
/home/dyf/miniconda3/envs/anygrasp/bin/python \
  scripts/simbox/validate_custom_scene_assets.py \
  InternDataAssets/assets/custom/scene_8
```

这个脚本会离线检查。它的唯一发现入口和结构来源是 inner `simbox_task.yaml`；传入 scene/room/family 目录时只是递归枚举这些入口。arena、HDR、USD、texture lib 都只能按 task YAML 字段解析后再读取：
- `asset_root` 是否等于 task 所在 room root。
- `arena_file` 指向的 arena YAML 是否存在。
- `env_map.envmap_lib` 是否能解析到 HDR。
- task 内 robot/object 和 arena 内 fixture USD 路径是否存在。
- floor/wall 等 texture lib 是否有文件，固定 `texture_id` 是否越界。
- `regions` 里 `object/target/container/target2/A/B` 是否能解析到实际 object / fixture / robot。
- `objects[].spawn_region` 和 `objects[].placement.spawn_region` 是否能在 `regions[].name` 中找到。
- `skills` 是否包含可执行 skill 条目；空 `base/left/right` 队列不是可运行配置。
- skill object、navigate goal、DAG `depends_on` 是否闭环。
- object-relative navigate 的可选 `approach_arm` / `approach_object_armbase_xy` 是否形状正确；`approach_object_armbase_xy` 必须与 `approach_arm: left|right` 一起使用。
- `random_config` 多余字段是否由 `banana.py::_filter_sampler_random_config()` 兜底。

脚本不 import Isaac、`banana.py` 或 `core.utils.region_sampler`；它直接解析 `region_sampler.py` 的 AST 来读 sampler 签名，所以适合在普通 Python 环境先跑。

通过标准：
- `task_count` 符合目标场景任务数。
- 每个任务输出 `OK`。
- 每个任务 `hdrs >= 1`、`robots >= 1`。
- 每个任务 `skills > 0`；如果 `skill_dag=false`，不允许任何 legacy controller queue 为空。
- 不出现 `error:` 行。
- `sampler_extra` 可非零，但前提是 `banana.py` 已有签名过滤；否则脚本会失败。

scene_8 当前复核结果应为 `task_count 20`；`kitchen_apple_to_tray` 应输出 `OK` 且 `skills=5 skill_dag=true`，剩余 19 个原始空 `skills` 任务应失败并报告空 skill。路径类指标仍应显示 `hdrs=1`、`robots=1`；厨房 5 个任务每个 `objects=17 / fixtures=23 / regions=18 / sampler_extra=18`，其它房间按任务物体数不同输出不同计数。

## 七、自动适配算法

如果 agent 要自动修改一个新 scene，按这个顺序执行，且每一步后立即跑对应检查：

1. 设置 `SCENE_ROOT` 和 `BASIC_ROOT`，枚举所有 inner `simbox_task.yaml`；不要读取外层 `task.yaml` / `arena.yaml` 来补任务结构。
2. 对每个 task，把 `tasks[0].asset_root` 统一成 `SCENE_ROOT`。
3. 把 `env_map.envmap_lib` 指到 repo 内真实 HDR 库；scene_7 历史参考值是 `../../../../../workflows/simbox/example_assets/envmap_lib`，scene_8 当前可行值是 `../shared_assets/envmap_lib`。修改后必须能从 `asset_root/envmap_lib/*.hdr` 找到文件。
4. 把 `robots[].path` 统一到可从 `asset_root` 解析的 robot USD；scene_7 历史参考机器人是 `../../../../../workflows/simbox/example_assets/split_aloha_mid_360/robot.usd`，scene_8 当前可行值是 `../shared_assets/split_aloha_mid_360/robot.usd`。
5. 从每个 `simbox_task.yaml` 读取 task，并按 `arena_file` 读取对应 arena，构造映射：`objects[].source_name -> objects[].name`、`arena.fixtures[].source_name -> arena.fixtures[].name`，再派生 `old_candidate_region -> new_candidate_region`。
6. 只替换运行时引用字段，保留所有 `source_name` 元数据字段。
7. 对 `regions`、`objects[].spawn_region`、`objects[].placement.spawn_region` 做闭环检查；不要把真实存在的 `__support_plane` fixture 或 source-style candidate region 误判为失败。
8. 检查 `skills`。对于 plan_with_render 任务，入口 YAML 不能保留 `base: [] / left: [] / right: []` 这种空 legacy 队列；需要写入可执行 DAG skill，或显式实现 scene-only 空 skill 支持。
9. 确认代码侧已经有三类兜底：envmap 空列表清晰报错、sampler kwargs 签名过滤、contact view 初始化失败可诊断降级。
10. 跑综合静态检查和 `py_compile`。
11. 再启动 Isaac/Nav2 做真实 scene loading / reset 验证。

## 八、处理空 skills

scene_8 的内层入口目前可能包含：

```yaml
skills:
- split_aloha:
  - base: []
    left: []
    right: []
```

这类配置只能说明资产场景可加载，不能用于 `plan_with_render` 执行。当前 `SimBoxDualWorkFlow` 的 legacy 分支会在 `plan_first_skill()` 中执行：

```python
lr_skill_list[0].simple_generate_manip_cmds()
```

所以空队列会直接产生：

```text
IndexError: list index out of range
```

修复路径二选一：
- 如果任务是 mobile manipulation，按任务意图写入可执行 DAG skill graph。基本形状通常是 `nav_to_pick -> pick_* -> nav_to_place -> place_* -> home_*`，但对象和放置目标必须来自当前入口 `simbox_task.yaml` 中真实存在的 `objects/fixtures/positions`，不要从外层 `task.yaml` 或目录名猜。
- 如果任务只是 scene loading / reset smoke test，则不要走 `plan_with_render`，或先在 workflow 中增加明确的 scene-only 空 skill 支持，并让 writer/成功判定知道这是无操作任务。

写 skill graph 时必须满足：
- `id` 全局唯一，`depends_on` 全部可解析。
- `pick/place` 的 `objects` 全部能在 task objects 或 arena fixtures 中找到。
- `navigate.goal` 必须在当前入口 YAML 的 `positions` 中找到；如果用 `approach`，也要确认对象/fixture 名能在运行时解析。
- navigate 固定使用 `RotationShimController -> MPPIController`，任务 YAML 不应写 controller 启停字段。
- 修改后先跑 `scripts/simbox/validate_custom_scene_assets.py`，再启动 Isaac。

## 九、运行验证流程

静态检查通过后，再启动真实运行。必须走 `scripts/docker/` 下的项目启动脚本，不要直接进入已有 `interndata-engine` 容器手动跑 `launcher.py`：

```bash
scripts/docker/up_nav2_stack.sh \
  --stack-id scene8val \
  --gpu 0 \
  --keep-nav2 \
  --launcher-config configs/de_plan_with_render_scene8_validation.yaml \
  isaac nav2
```

注意：
- 当前 checkout 的 `scripts/docker/up_nav2_stack.sh` 是轻量 wrapper，默认委托 `scripts/docker/up_nav2_stack_single_gpu.sh`，`--multi` 时委托 `scripts/docker/up_nav2_stack_multi_gpu.sh`。
- scene_8 最小验证配置保存在 `configs/de_plan_with_render_scene8_validation.yaml`，避免在启动命令里手拼长 `INTERNDATA_LAUNCHER_EXTRA_ARGS`。
- `INTERNDATA_STACK_ID` 应使用独立值，例如 `scene8val`，避免覆盖已有 `isaac` / `nav2` 容器。
- `INTERNDATA_STOP_NAV2_WHEN_ISAAC_EXITS=0` 便于失败后保留 `nav2-scene8val` 日志；验证结束后可用 `docker stop isaac-scene8val nav2-scene8val` 清理。
- 不要把已运行的 `interndata-engine` 调试容器作为 scene_8 验证证据；它绕过了 `scripts/docker/` 的 launcher/env/ROS2 启动路径。

运行后检查日志，至少确认不再出现以下旧错误：

```text
empty range for randrange()
KeyError: '<double underscore object name>'
unexpected keyword argument 'support_surface_z'
'NoneType' object has no attribute 'sensor_count'
IndexError: list index out of range
```

如果出现：

```text
[simbox] contact view initialization failed; continuing without contact sensing: {...}
```

这代表 scene loading 没被 contact sensor 阻断，但 contact 根因仍在。继续检查日志中的：

- `prim_paths_expr`
- `filter_paths_expr`
- `robot`
- `arm`
- `object`

下一步应进入 USD / robot gripper path / collision schema 调查，不要继续盲改 YAML。

如果日志只显示：

```text
plan_with_render returned 0
generate failed but partial output was kept
```

不要把它当作 startup/reset 问题。按最新 mtime 查 `output/ros_bridge/skills/*`：

```bash
find output/ros_bridge/skills -maxdepth 2 -type f \
  \( -name 'failure_snapshot.json' -o -name 'pick_plan_snapshot.json' \
     -o -name 'pick_runtime_failure_snapshot.json' -o -name 'place_success_check_snapshot.json' \
     -o -name 'dynamic_goal_candidates.json' \) \
  -printf '%T@ %p\n' | sort -n | tail -n 80
```

对 pick failure，优先读：
- `pick_plan_snapshot.json`: `success_found`、`candidate_results[*].pregrasp_success`、`candidate_results[*].grasp_success`
- `geometry_debug.object_armbase_pose`
- `geometry_debug.mobile_base_world_pose`
- `selected_candidate`

2026-07-02 scene_8 `kitchen_apple_to_tray` 的最新失败经验：
- `support_surface_z` reset 异常修复后，失败不再是 reset traceback，而是 pick planning 不可行。
- 最新 run 中导航多次能到达 object-relative approach goal，`world_dist` 通常约 `0.01-0.03m`。
- 但所有 pick snapshot 都是 `success_found: false`，20 个候选的 `pregrasp_success=false` 且 `grasp_success=false`。
- 失败几何中 apple 在左臂 base 下约为 `x=0.66-0.69, y=-0.29~-0.39, z=-0.10`；历史成功样本约为 `x=0.57, y=-0.11, z=-0.196`。这说明当前 approach 点只满足 Nav2 可达，不保证左臂抓取姿态可达。
- 原始 object-relative approach 采样默认让底盘 yaw 正对目标。对 split_aloha 左臂，这会把目标固定到左臂 base 的侧向偏置附近，导致 apple 的 `object_armbase_pose.y` 接近 `-0.306`，而历史成功抓取更接近 `-0.11~-0.13`。
- 如果 dynamic goal 的 `selected.index` 稳定是 `242`，`goal=(2.5437, 1.1163, yaw=-0.4035)`，且 pick snapshot 继续显示 `success_found: false`，不要再查 `support_surface_z`；应转向 approach 选点和手臂可达性。
- 当前代码支持在 navigate skill 上 opt-in 写入 `approach_arm` 和 `approach_object_armbase_xy`，让采样 yaw 和 preflight 排序优先把目标物放到指定手臂 base 下的历史成功区域。例如 scene_8 `kitchen_apple_to_tray` 的 `nav_to_pick` 使用：

```yaml
approach: apple_0_id9008
approach_arm: left
approach_object_armbase_xy: [0.575, -0.12]
```

- 这个字段只影响配置了它的 object-relative approach；没有配置时仍保持旧的正对目标采样。添加后必须检查 `dynamic_goal_candidates.json`：`approach.arm` 应为 `left`，候选中应出现 `approach_yaw_strategy: object_armbase_xy`、`approach_score` 和 `approach_armbase_prediction.object_armbase_xy`。
- 离线复核显示，新字段会把第一批静态可行候选从旧的 index `242` 切到 index `137/158/192...` 这类点，预测 apple 在左臂 base 下约为 `x=0.59-0.63, y=-0.12`，更接近历史成功区间。但这只证明静态 footprint 和几何排序，不能替代真实 Nav2 path / pick 运行验证。
- 2026-07-02 后续真实运行显示，新 approach 选点已经生效：`dynamic_goal_candidates.json` 选中 index `137`，goal 约为 `(3.0332, 1.5681, -1.2669)`，`approach_yaw_strategy=object_armbase_xy`，预测 apple 在左臂 base 下约为 `(0.593, -0.116)`，且 `ComputePathToPose` 返回 `path_ok=true`。这说明当前轮失败不再是 approach 采样或 global plan 失败。
- 第一轮执行层失败：`failure_snapshot.json` 为 `reason=bridge_aborted`、`status_code=6`；Nav2 日志反复出现 `RotationShimController detected collision ahead`，最终 `Resulting plan has 0 poses in it`。同时 `bridge_command_history.json` 显示移动底盘姿态从正常 `z≈0.32` 演变到数米甚至十几米，实际速度出现几十到上百 m/s 量级，属于底盘物理失稳/弹飞后的 abort，不是 `support_surface_z`、不是 static footprint，也不是 `ComputePathToPose`。
- 该轮还有一个重要配置生命周期陷阱：goal 目录下冻结的 debug params 与 resident Nav2 bootstrap params 使用了不同 controller。只看 goal debug params 会误判 live Nav2 的实际 controller；排查时必须同时看 resident bootstrap params 和 Nav2 容器日志。当前实现已固定使用 `RotationShimController -> MPPIController`。
- 因此 scene_8 验证前要保证 resident bootstrap 默认也不会启用 Rotation Shim。当前默认 controller profile 位于 `nav2/config/nav2_params.yaml`；重跑时必须重启 Nav2 容器并确认 `output/nav2_runtime/<session>/bootstrap/nav2_params.yaml` 里的 `controller_server.ros__parameters.FollowPath.plugin` 是 `nav2_mppi_controller::MPPIController`，否则本轮修复没有进入实际控制器。
- 2026-07-02 第二轮重启 resident Nav2 后，live bootstrap 已确认 `FollowPath.plugin: nav2_mppi_controller::MPPIController`，Nav2 日志也显示 `Configured MPPI Controller: FollowPath`，但仍在同一目标附近失败。这一轮 `actual_trajectory.json` 显示机器人一度到达目标 XY 附近，最近距离约 `0.002m`，但 yaw 误差仍约 `2.315rad`；随后 MPPI 为补目标朝向继续移动，最终停在约 `(2.428, 0.992)`，距目标约 `0.836m`。Nav2 直接 abort 信号是 planner 日志 `Starting point in lethal space! Cannot create feasible plan`，不是 RotationShim，也不是启动配置未生效。
- 这轮还暴露两个修复点：一是旧 `Navigate._nav2_skill_overrides()` 没有把 skill 级 `xy_goal_tolerance/yaw_goal_tolerance` 下推到 live goal checker；后续 agent 应确认这些 tolerance 已写进 `controller_server.general_goal_checker`。二是 `approach_footprint_padding: 0.02` 小于 live global/local costmap padding；该 index `137` 目标点在 `0.02m` 静态检查下可行，但用 `0.03m` padding 已有 blocked padding cells，用 local `0.10m` padding 更不安全。参考任务应把 approach padding 调到和 local costmap 一致，或至少不小于 global footprint padding。
- 2026-07-02 第三轮把 `approach_footprint_padding` 提到 `0.1` 后，selected 从 index `137` 切到 index `336`，goal 约为 `(2.9107, 1.6313, yaw=-1.1963)`；`dynamic_goal_candidates.json` 中 768 个候选里 31 个 `static_ok`，只有 1 个 `path_ok=true`。这说明 padding 确实进入了候选 endpoint 检查，但失败仍没有消失。
- 这一轮不再是 `Starting point in lethal space`，而是 Nav2 日志里的 `Optimizer fail to compute path`，最终 `failure_snapshot.json` 为 `reason=bridge_aborted`、`status_code=6`，final `nav_dist≈0.452m`、`yaw_err≈2.72rad`。`actual_trajectory.json` 曾在 sample 79 到达 goal XY 约 `0.007m` 内，但 yaw 仍差约 `2.05rad`，随后控制器继续修正姿态并漂离。
- 关键根因是 dynamic preflight 只检查候选 endpoint 的静态 footprint，并只要求 `ComputePathToPose` 返回有限长度 path；它没有用同一套 full footprint + `approach_footprint_padding` 复核返回 path 的每个 pose。离线复核 `planned_path.json` 显示 46 个 path pose 中 23 个会被同一静态 footprint+padding 检查拒绝，首个失败 pose 约为 `(2.1587, 1.5536, yaw=0.9523)`，尾部 final path pose 也有 padding blocked cells。实际轨迹从 sample 51 左右开始进入同类静态碰撞区域。
- 因此修复方向不是继续调 `support_surface_z`、不是回退 RotationShim、也不是只增加 endpoint padding；dynamic goal 接受候选前必须把 Nav2 返回的 path poses 全程跑一遍同样的 static footprint + padding 检查。只有 endpoint、`ComputePathToPose` 状态和 path-level footprint 都通过，才能发布 navigation goal；否则应把候选标记为 `path_static_rejected` 并继续尝试下一个 static candidate。
- 2026-07-02 第四轮加入 path-level 静态检查后，修复已生效：旧 index `336` 被标记为 `path_static_rejected`，随后又拒绝 15 条完整 path 不安全的候选，最后选中 index `577`，其 `path_static_ok=true`、`path_static_blocked_pose_count=0`。这说明上一轮 `Optimizer fail to compute path` 的直接根因已被挡在 preflight 阶段。
- 第四轮仍失败，但失败原因变成 `goal_tolerance_not_met`：Nav2 返回 `state=succeeded/status_code=4`，最终 XY 误差约 `0.016m`，但 SimBox 复核相对原始 candidate yaw 的误差约 `0.161rad`，超过 skill 的 `yaw_goal_tolerance: 0.1`。这不是 bridge abort，也不是静态 path 碰撞。
- 该轮暴露出第二个口径差异：candidate 原始 yaw 是约 `-0.8582`，而 `planned_path.json` 的末端 yaw 是约 `-0.7854`；最终车体 yaw 约 `-0.6967`，相对 path 末端误差约 `0.089rad`，但相对原始 candidate yaw 误差约 `0.161rad`。Nav2 成功是按实际 FollowPath 的路径末端姿态收敛，SimBox skill 却按原始 candidate yaw 复核，导致 Nav2 成功后仍被 skill 判失败。
- 后续修复应在 dynamic preflight 接受 candidate 时记录 `effective_goal`：如果 Nav2 返回 path 的末端 pose 与原始 candidate 的 XY/yaw 差异仍在配置 tolerance 内，则发布和复核都使用 path 末端 pose；如果末端 pose 已超出 tolerance，则把候选标记为 `path_terminal_rejected` 并继续尝试下一个候选。不要简单放宽 `yaw_goal_tolerance` 来掩盖两套目标口径不一致。
- 2026-07-02 第五轮加入 `effective_goal` 后，导航阶段已通过：`success_snapshot.json` 为 `goal_succeeded`，实际发布给 Nav2 的 goal 是 path 末端 `(2.5300, 1.4700, yaw=-0.7854)`，最终 `nav_dist≈0.014m`、`yaw_err≈0.081rad`，均在 `0.1` tolerance 内。
- 第五轮失败转移到 pick：`pick_plan_snapshot.json` 为 `success_found: false`，20 个候选全部 `pregrasp_success=false/grasp_success=false`，随后 fallback 执行在 `open_gripper` pregrasp 阶段累计 `num_plan_failed=6`，`pick_runtime_failure_snapshot.json` 报 `skill_not_feasible`。此时不要再回查 navigation abort；当前根因已经是左臂抓取几何/IK 不可行。
- 该轮 pick 几何中 apple 在左臂 base 下约为 `(x=0.843, y=-0.264, z=-0.078)`，selected fallback pregrasp 约为 `(0.710, -0.216, 0.117)`，grasp 约为 `(0.776, -0.239, 0.019)`。对比旧成功 snapshot，例如 `split_aloha_pick_apple_0_id9008_1781976139592` 的 object-armbase 约为 `(x=0.333, y=-0.247, z=-0.049)`，本轮最硬的差异是左臂前向 `x` 偏远；说明 dynamic approach 的 Nav2-safe endpoint 仍不能保证左臂 pick reachability。
- 进一步量化后，selected index `577` 的原始 candidate 已经不是最理想抓取几何：`approach_armbase_prediction.object_armbase_xy≈(0.847, -0.066)`，到配置目标 `[0.575, -0.12]` 的平面距离约 `0.277m`。这是因为更接近目标区间的 static candidates（例如 index `336/357/.../569`）在完整 Nav2 path footprint 复核中被 `path_static_rejected`，preflight 只能继续往后找第一个 path-safe 候选。
- `effective_goal` 和最终执行姿态会进一步改变手臂相对几何：index `577` 原始 yaw 为 `-0.8582`，path 末端 yaw 为 `-0.7854`，实际到达 yaw 约 `-0.7042`；用同一 object pose 计算时，预测从原始 candidate 的 `(0.847, -0.066)` 变为 path 末端的约 `(0.858, -0.158)`，实际 base 姿态下约为 `(0.852, -0.261)`。因此本轮失败的精确原因是 dynamic preflight 的可达性评分仍基于原始 candidate，而真实 pick 使用的是 path 末端/实际到达姿态。
- 后续修复不要优先放宽 pick filter。应先让 dynamic preflight 在 `ComputePathToPose` 返回后，用 `effective_goal` 重新计算并记录 `effective_approach_armbase_prediction` / `effective_approach_score`，再按 path-safe 后的真实手臂相对几何决定是否接受候选。否则系统会继续选到 Nav2 安全但左臂抓取不可达的点。
- 如果确认 live bootstrap 已经是 MPPI 但仍失败，再继续查 `dynamic_goal_candidates.json` 的 `selected.index/path_ok`、`failure_snapshot.json` 的 `nav_dist/yaw_err`、`actual_trajectory.json` 的最近目标点、Nav2 日志里的 `Starting point in lethal space`、以及 `bridge_command_history.json` 的底盘 `z/roll/pitch/actual_linear_velocity_body`、wheel/steering 命令和局部控制器输出；不要回退到旧 index `242`，也不要再把 `support_surface_z` 当当前根因。

## 十、agent 执行顺序建议

1. 找到目标 scene 下所有 `simbox_task.yaml`，并把这些 YAML 当作唯一结构入口。
2. 统一 `asset_root / envmap_lib / robot path`。
3. 按 `simbox_task.yaml` 字段验证 arena、HDR、robot USD、object USD。
4. 建立 `source_name -> name` 映射，替换运行时引用字段，保留 `source_name` 元数据。
5. 验证 `regions/spawn_region` 引用闭环；`source_regions` 作为来源记录保留，除非运行代码实际消费它。
6. 确认 `banana.py` 有 envmap 空检查和 sampler kwargs 签名过滤。
7. 确认 pick contact view 优先 `mesh_prim_path`，并有不可用降级。
8. 验证 `skills` 非空且对象、目标、依赖闭环。
9. 跑 `py_compile`。
10. 启动真实 Isaac/Nav2 验证。
11. 如果仍失败，按第一条 traceback 定位，不要一次改多个无关层。

## 十一、已知非完成项

- contact view 降级只保证 scene loading 不被非关键 sensor 阻断，不证明 PhysX contact view 已正确工作。
- 如果 pick 成功判定依赖 contact force，contact unavailable 会影响 pick success；应优先用 `lift_th` 或执行 trace 判断真实抓取结果。
- scene_8 当前路径闭环不等于任务可执行；空 `skills` 仍需按任务语义补齐或改成明确的 scene-only 验证模式。
- 运行级成功仍需看真实日志和输出，不以静态检查替代。
