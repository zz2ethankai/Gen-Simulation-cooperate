"""Dump useful USD stage structure details as Markdown.

This tool is intentionally read-only. It is meant for comparing SimBox asset
contracts, where the important facts are often hidden in prim paths, applied
API schemas, material bindings, physics attributes, and sidecar files.
"""

import argparse
from pathlib import Path
from typing import Iterable

from pxr import Gf, Sdf, Usd, UsdGeom


KEY_ATTR_PREFIXES = (
    "physics:",
    "physx",
    "material:",
    "inputs:",
    "outputs:",
    "xformOp:",
)

KEY_ATTR_NAMES = {
    "doubleSided",
    "extent",
    "faceVertexCounts",
    "faceVertexIndices",
    "points",
    "purpose",
    "subdivisionScheme",
    "visibility",
    "xformOpOrder",
}

KEY_METADATA = (
    "kind",
    "instanceable",
    "references",
    "payload",
    "apiSchemas",
    "material:binding",
)

MAX_SIDECARS = 40


def md_escape(value) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def yn(flag: bool) -> str:
    return "是" if flag else "-"


def summarize_value(value, max_items: int = 8) -> str:
    if value is None:
        return "无"

    if isinstance(value, Sdf.Path):
        return str(value)

    if isinstance(value, Sdf.AssetPath):
        return f"asset={value.path}, resolved={value.resolvedPath}"

    if hasattr(value, "explicitItems") and hasattr(value, "prependedItems"):
        chunks = []
        for label in ("explicitItems", "prependedItems", "appendedItems", "deletedItems"):
            items = getattr(value, label, None)
            if items:
                chunks.append(f"{label}={list(items)}")
        return "; ".join(chunks) if chunks else repr(value)

    if isinstance(value, (str, int, float, bool)):
        return repr(value) if isinstance(value, str) else str(value)

    if hasattr(value, "__len__") and not isinstance(value, (dict, bytes)):
        try:
            length = len(value)
        except TypeError:
            length = None
        if length is not None:
            if length == 0:
                return "[]"
            preview = []
            for i, item in enumerate(value):
                if i >= max_items:
                    break
                preview.append(summarize_value(item, max_items=3))
            suffix = "" if length <= max_items else f", ... （共 {length} 项）"
            return "[" + ", ".join(preview) + suffix + "]"

    return repr(value)


def is_key_attr(name: str) -> bool:
    return name in KEY_ATTR_NAMES or name.startswith(KEY_ATTR_PREFIXES)


def authored_attributes(prim: Usd.Prim, mode: str):
    attrs = []
    for attr in prim.GetAttributes():
        if not attr.HasAuthoredValueOpinion():
            continue
        if mode == "key" and not is_key_attr(attr.GetName()):
            continue
        attrs.append(attr)
    return attrs


def authored_relationships(prim: Usd.Prim):
    rels = []
    for rel in prim.GetRelationships():
        if rel.HasAuthoredTargets():
            rels.append(rel)
    return rels


def relationship_targets_with_validity(stage: Usd.Stage, rel: Usd.Relationship):
    targets = []
    for target in rel.GetTargets():
        suffix = "" if stage.GetPrimAtPath(target).IsValid() else "（当前 stage 中找不到目标 prim）"
        targets.append(f"{target}{suffix}")
    return targets


def metadata_rows(prim: Usd.Prim):
    rows = []
    for key in KEY_METADATA:
        if prim.HasAuthoredMetadata(key):
            rows.append((key, prim.GetMetadata(key)))
    return rows


def local_arcs(prim: Usd.Prim):
    refs = prim.GetMetadata("references") if prim.HasAuthoredMetadata("references") else None
    payload = prim.GetMetadata("payload") if prim.HasAuthoredMetadata("payload") else None
    return refs, payload


def prim_stack_layers(prim: Usd.Prim):
    layers = []
    for spec in prim.GetPrimStack():
        layer = spec.layer
        layers.append(f"{layer.identifier}:{spec.path}")
    return layers


def schema_flags(prim: Usd.Prim):
    schemas = [str(s) for s in prim.GetAppliedSchemas()]
    joined = " ".join(schemas)
    return {
        "rigid_body": "RigidBody" in joined,
        "mass": "Mass" in joined,
        "collision": "Collision" in joined,
        "mesh_collision": "MeshCollision" in joined,
        "physics_material": "PhysicsMaterial" in joined,
        "material_binding": "MaterialBinding" in joined,
    }


def vec_tuple(value):
    return tuple(round(float(x), 6) for x in value)


def mesh_summary_rows(stage: Usd.Stage):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
    rows = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        face_counts = mesh.GetFaceVertexCountsAttr().Get() or []

        if points:
            local_min = Gf.Vec3d(
                min(point[0] for point in points),
                min(point[1] for point in points),
                min(point[2] for point in points),
            )
            local_max = Gf.Vec3d(
                max(point[0] for point in points),
                max(point[1] for point in points),
                max(point[2] for point in points),
            )
            local_center = (local_min + local_max) / 2.0
            local_size = local_max - local_min
        else:
            local_center = local_size = "-"

        world_box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        material_targets = []
        for rel in prim.GetRelationships():
            if rel.GetName().startswith("material:binding") and rel.HasAuthoredTargets():
                material_targets.extend(relationship_targets_with_validity(stage, rel))

        rows.append(
            [
                f"`{prim.GetPath()}`",
                str(len(points)),
                str(len(face_counts)),
                str(vec_tuple(local_center)) if points else "-",
                str(vec_tuple(local_size)) if points else "-",
                str(vec_tuple(world_box.GetMin())),
                str(vec_tuple(world_box.GetMax())),
                ", ".join(material_targets) or "-",
            ]
        )
    return rows


def tree_rows(stage: Usd.Stage):
    rows = []
    for prim in stage.Traverse():
        depth = len(str(prim.GetPath()).strip("/").split("/")) - 1
        indent = "&nbsp;&nbsp;" * max(depth, 0)
        schemas = ", ".join(str(s) for s in prim.GetAppliedSchemas()) or "-"
        refs, payload = local_arcs(prim)
        rows.append(
            [
                f"{indent}`{prim.GetPath()}`",
                prim.GetTypeName() or "-",
                "是" if prim.IsActive() else "否",
                "是" if prim.IsDefined() else "否",
                schemas,
                summarize_value(refs) if refs else "-",
                summarize_value(payload) if payload else "-",
            ]
        )
    return rows


def write_table(lines: list[str], headers: Iterable[str], rows: Iterable[Iterable[str]]) -> None:
    headers = list(headers)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(md_escape(cell) for cell in row) + " |")


def root_child_paths(default_prim: Usd.Prim) -> list[str]:
    if not default_prim:
        return []
    return [str(child.GetPath()) for child in default_prim.GetAllChildren()]


def has_schema(stage: Usd.Stage, fragment: str) -> bool:
    for prim in stage.Traverse():
        if fragment in " ".join(str(schema) for schema in prim.GetAppliedSchemas()):
            return True
    return False


def stage_takeaways(stage: Usd.Stage) -> list[str]:
    default_prim = stage.GetDefaultPrim()
    default_path = str(default_prim.GetPath()) if default_prim else "无"
    children = root_child_paths(default_prim)
    child_text = ", ".join(f"`{path}`" for path in children) if children else "没有直接 child"

    notes = [
        f"从顶层看，默认入口是 `{default_path}`。运行时代码如果从 default prim 开始找对象，第一步看到的就是这一层。",
        f"`{default_path}` 下面的直接 child 是：{child_text}。这决定了硬编码脚本用 `GetAllChildren()[1]` 时能不能取到目标。",
    ]

    if stage.GetPrimAtPath("/World/Aligned").IsValid():
        notes.append(
            "存在 `/World/Aligned`：这更接近 SimBox 标准 pickable 的组织方式，`Aligned` 是物体主体，刚体、质量和下游 `prim_path_child: Aligned` 都围绕它建立。"
        )
    elif stage.GetPrimAtPath("/Asset/Geometry").IsValid():
        notes.append(
            "存在 `/Asset/Geometry` 但没有 `/World/Aligned`：这更像场景数据集里的单个 mesh 片段，能显示或被场景引用，但还不是 SimBox task loader 期待的 pickable 资产。"
        )
    else:
        notes.append(
            "没有看到 `/World/Aligned` 或 `/Asset/Geometry` 这种明确结构，需要结合下面的 Prim 树继续判断它是哪类资产。"
        )

    if has_schema(stage, "RigidBody"):
        notes.append("已经看到刚体相关 API schema，说明至少有某个 prim 被声明为物理刚体主体。")
    else:
        notes.append("没有看到刚体相关 API schema；即使 mesh 上有碰撞信息，也不等于整个对象已经会作为动态刚体参与仿真。")

    if has_schema(stage, "Collision"):
        notes.append("已经看到碰撞相关 API schema，但还要继续看它加在 mesh 上还是加在对象主体上，以及是否有 convex decomposition 等细节。")
    else:
        notes.append("没有看到碰撞相关 API schema，后续接触、抓取和放置仿真通常还需要补碰撞。")

    return notes


def dump_stage(path: Path, attr_mode: str, include_sidecars: bool) -> str:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {path}")

    default_prim = stage.GetDefaultPrim()
    root_layer = stage.GetRootLayer()
    lines = [f"## `{path}`", ""]

    lines.extend(
        [
            "### Stage 顶层总览",
            "",
            f"- Default prim（默认入口）: `{default_prim.GetPath() if default_prim else None}`",
            f"- Root layer（根 layer）: `{root_layer.identifier}`",
            f"- Up axis（上方向）: `{UsdGeom.GetStageUpAxis(stage)}`",
            f"- Meters per unit（单位换算）: `{UsdGeom.GetStageMetersPerUnit(stage)}`",
            f"- Start/end time code（时间范围）: `{stage.GetStartTimeCode()}` / `{stage.GetEndTimeCode()}`",
            f"- Root prims（最顶层 prim）: `{', '.join(str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren())}`",
            "",
            "### 本文件的快速读法",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in stage_takeaways(stage))
    lines.append("")

    if include_sidecars:
        sidecars = sorted(p.name for p in path.parent.iterdir())
        lines.append("### 同目录旁路文件")
        lines.append("")
        lines.append(
            "这些文件不是 USD stage 的内部 prim，但它们常常决定资产能不能被 SimBox 的抓取、材质、碰撞生成流程继续使用。"
        )
        lines.append("")
        lines.append(f"- 共 `{len(sidecars)}` 项。")
        visible_sidecars = sidecars[:MAX_SIDECARS]
        lines.extend(f"- `{name}`" for name in visible_sidecars)
        if len(sidecars) > MAX_SIDECARS:
            lines.append(f"- ... 其余 `{len(sidecars) - MAX_SIDECARS}` 项已省略，避免这份证据文档被目录清单淹没。")
        lines.append("")

    lines.append("### Prim 树")
    lines.append("")
    write_table(
        lines,
        ["Prim 路径", "类型", "启用", "已定义", "已应用 API schemas", "References 引用", "Payload 负载"],
        tree_rows(stage),
    )
    lines.append("")

    lines.append("### 物理 / 材质 / 碰撞标记")
    lines.append("")
    flag_rows = []
    for prim in stage.Traverse():
        flags = schema_flags(prim)
        flag_rows.append(
            [
                f"`{prim.GetPath()}`",
                prim.GetTypeName() or "-",
                yn(flags["rigid_body"]),
                yn(flags["mass"]),
                yn(flags["collision"]),
                yn(flags["mesh_collision"]),
                yn(flags["physics_material"]),
                yn(flags["material_binding"]),
            ]
        )
    write_table(
        lines,
        [
            "Prim 路径",
            "类型",
            "刚体 API",
            "质量 API",
            "碰撞 API",
            "Mesh 碰撞 API",
            "物理材质 API",
            "材质绑定 API",
        ],
        flag_rows,
    )
    lines.append("")

    mesh_rows = mesh_summary_rows(stage)
    if mesh_rows:
        lines.append("### Mesh 几何摘要")
        lines.append("")
        write_table(
            lines,
            [
                "Mesh prim",
                "点数",
                "面数",
                "局部包围盒中心",
                "局部包围盒尺寸",
                "世界包围盒 min",
                "世界包围盒 max",
                "材质目标",
            ],
            mesh_rows,
        )
        lines.append("")

    lines.append("### Prim 细节")
    lines.append("")
    for prim in stage.Traverse():
        lines.append(f"#### `{prim.GetPath()}`")
        lines.append("")
        lines.append(f"- 类型: `{prim.GetTypeName() or '-'}`")
        lines.append(f"- 已应用 API schemas: `{', '.join(str(s) for s in prim.GetAppliedSchemas()) or '-'}`")

        stack = prim_stack_layers(prim)
        if stack:
            lines.append("- Prim stack（这个 prim 的来源 layer）:")
            lines.extend(f"  - `{item}`" for item in stack)

        refs, payload = local_arcs(prim)
        if refs:
            lines.append(f"- References: `{summarize_value(refs)}`")
        if payload:
            lines.append(f"- Payload: `{summarize_value(payload)}`")

        metadata = metadata_rows(prim)
        if metadata:
            lines.append("- 已写入 metadata:")
            for key, value in metadata:
                lines.append(f"  - `{key}`: `{summarize_value(value)}`")

        rels = authored_relationships(prim)
        if rels:
            lines.append("- 已写入 relationships:")
            for rel in rels:
                targets = relationship_targets_with_validity(stage, rel)
                lines.append(f"  - `{rel.GetName()}` -> `{', '.join(targets)}`")

        attrs = authored_attributes(prim, attr_mode)
        if attrs:
            lines.append("- 已写入 attributes:")
            for attr in attrs:
                lines.append(f"  - `{attr.GetName()}` `{attr.GetTypeName()}` = `{summarize_value(attr.Get())}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump USD stage structure as Markdown")
    parser.add_argument("usd_paths", nargs="+", type=Path, help="USD files to inspect")
    parser.add_argument("-o", "--output", type=Path, help="Write Markdown to this file instead of stdout")
    parser.add_argument(
        "--attrs",
        choices=("key", "all"),
        default="key",
        help="Attribute verbosity. 'key' keeps physics/material/xform/mesh summary attrs; 'all' dumps all authored attrs.",
    )
    parser.add_argument(
        "--no-sidecars",
        action="store_true",
        help="Do not list sibling files in each USD directory.",
    )
    args = parser.parse_args()

    parts = [
        "# USD Stage 结构证据摘录：标准 SimBox pickable vs benchmark1.0 小物体",
        "",
        "> 由 `workflows/simbox/tools/rigid_obj/dump_usd_stage_md.py` 生成；脚本只读检查 USD，不会改写资产文件。",
        "",
        "这份文档是证据附录，不是主解释文档。阅读时建议先看每个文件的 `Stage 顶层总览` 和 `本文件的快速读法`，再往下看 Prim 树、物理标记和具体属性。",
        "",
    ]
    for usd_path in args.usd_paths:
        parts.append(dump_stage(usd_path, args.attrs, not args.no_sidecars))

    markdown = "\n".join(parts).rstrip() + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")


if __name__ == "__main__":
    main()
