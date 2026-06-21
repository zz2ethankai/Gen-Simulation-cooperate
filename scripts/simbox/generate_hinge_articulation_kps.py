#!/usr/bin/env python3
"""Generate SimBox articulated-object keypoint annotations for hinged USD assets."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pxr import Gf, Usd, UsdGeom, UsdPhysics


AXIS_TO_VEC = {
    "x": Gf.Vec3d(1.0, 0.0, 0.0),
    "-x": Gf.Vec3d(-1.0, 0.0, 0.0),
    "y": Gf.Vec3d(0.0, 1.0, 0.0),
    "-y": Gf.Vec3d(0.0, -1.0, 0.0),
    "z": Gf.Vec3d(0.0, 0.0, 1.0),
    "-z": Gf.Vec3d(0.0, 0.0, -1.0),
}


@dataclass(frozen=True)
class HingeAnnotation:
    output_name: str
    joint_index: int
    joint_path: str
    base_path: str
    link_path: str
    head: list[float]
    tail: list[float]
    rot_axis: str
    contact_axis: str
    base_front_axis: str


def _tokenize_path(path: str) -> str:
    text = path.rsplit("/", 1)[-1]
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").lower()
    return text or "joint"


def _axis_token_from_usd(value) -> str:
    if value is None:
        return "z"
    text = str(value).strip().upper()
    return {"X": "x", "Y": "y", "Z": "z"}.get(text, "z")


def _nearest_axis_token(vec: Gf.Vec3d, fallback: str = "y") -> str:
    length = vec.GetLength()
    if length <= 1.0e-9:
        return fallback
    unit = vec / length
    best = max(AXIS_TO_VEC.items(), key=lambda item: Gf.Dot(unit, item[1]))
    return best[0]


def _vec_from_token(token: str) -> Gf.Vec3d:
    return AXIS_TO_VEC[token]


def _format_vec(vec: Iterable[float]) -> list[float]:
    return [round(float(v), 8) for v in vec]


def _bbox_for_prim(cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> Gf.Range3d | None:
    if not prim or not prim.IsValid():
        return None
    bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if bbox.IsEmpty():
        return None
    return bbox


def _bbox_center(bbox: Gf.Range3d) -> Gf.Vec3d:
    return (bbox.GetMin() + bbox.GetMax()) * 0.5


def _combined_local_bbox(
    cache: UsdGeom.BBoxCache,
    parent_prim: Usd.Prim,
    child_prims: list[Usd.Prim],
) -> Gf.Range3d | None:
    parent_inv = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    local_range = Gf.Range3d()
    used = False
    for prim in child_prims:
        bbox = _bbox_for_prim(cache, prim)
        if bbox is None:
            continue
        min_p = bbox.GetMin()
        max_p = bbox.GetMax()
        for x in (min_p[0], max_p[0]):
            for y in (min_p[1], max_p[1]):
                for z in (min_p[2], max_p[2]):
                    local_range.UnionWith(parent_inv.Transform(Gf.Vec3d(x, y, z)))
                    used = True
    return local_range if used else None


def _descendants_matching(root: Usd.Prim, pattern: str) -> list[Usd.Prim]:
    pattern = pattern.lower()
    matches = []
    for prim in Usd.PrimRange(root):
        if prim == root:
            continue
        if pattern in prim.GetName().lower():
            matches.append(prim)
    return matches


def _default_head_local(cache: UsdGeom.BBoxCache, link_prim: Usd.Prim, base_front_axis: str) -> Gf.Vec3d:
    local_bbox = _combined_local_bbox(cache, link_prim, [link_prim])
    if local_bbox is None:
        return Gf.Vec3d(0.0, 0.0, 0.0)
    min_p = local_bbox.GetMin()
    max_p = local_bbox.GetMax()
    center = _bbox_center(local_bbox)
    front = _vec_from_token(base_front_axis)
    coords = [center[0], center[1], center[2]]
    axis = max(range(3), key=lambda i: abs(front[i]))
    coords[axis] = max_p[axis] if front[axis] >= 0 else min_p[axis]
    return Gf.Vec3d(*coords)


def _contact_point_local(cache: UsdGeom.BBoxCache, link_prim: Usd.Prim, base_front_axis: str) -> tuple[Gf.Vec3d, str]:
    handle_prims = _descendants_matching(link_prim, "handle")
    handle_bbox = _combined_local_bbox(cache, link_prim, handle_prims)
    if handle_bbox is None:
        return _default_head_local(cache, link_prim, base_front_axis), base_front_axis

    extents = handle_bbox.GetSize()
    longest_axis = max(range(3), key=lambda i: abs(extents[i]))
    contact_axis = ["x", "y", "z"][longest_axis]
    return _bbox_center(handle_bbox), contact_axis


def _tail_direction_local(cache: UsdGeom.BBoxCache, link_prim: Usd.Prim, head: Gf.Vec3d, rot_axis: str) -> Gf.Vec3d:
    local_bbox = _combined_local_bbox(cache, link_prim, [link_prim])
    if local_bbox is None:
        return Gf.Vec3d(1.0, 0.0, 0.0)

    center = _bbox_center(local_bbox)
    direction = center - head
    rot = _vec_from_token(rot_axis)
    direction = direction - rot * Gf.Dot(direction, rot)
    if direction.GetLength() > 1.0e-6:
        return direction.GetNormalized()

    sizes = local_bbox.GetSize()
    candidates = [(i, abs(sizes[i])) for i in range(3) if abs(rot[i]) < 0.5]
    axis = max(candidates, key=lambda item: item[1])[0] if candidates else 0
    values = [0.0, 0.0, 0.0]
    values[axis] = 1.0
    return Gf.Vec3d(*values)


def _stage_root_name(stage: Usd.Stage) -> str:
    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        return default_prim.GetName()
    children = list(stage.GetPseudoRoot().GetChildren())
    if not children:
        raise ValueError("USD stage has no root prims")
    return children[0].GetName()


def _set_scale_op(prim: Usd.Prim, scale: Gf.Vec3f) -> None:
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            op.Set(scale)
            return
    xformable.AddScaleOp().Set(scale)


def normalize_cube_scales(stage: Usd.Stage, *, save: bool) -> int:
    """Move parent-only scale to Cube children for cuRobo obstacle parsing.

    cuRobo reads xformOp:scale from Cube prims directly. Some CAD exports put
    the scale on a parent Xform and leave the Cube unscaled, which preserves USD
    geometry but makes cuRobo fail while building cuboid obstacles.
    """
    changed = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Cube":
            continue
        scale_attr = prim.GetAttribute("xformOp:scale")
        if scale_attr and scale_attr.Get() is not None:
            continue

        parent = prim.GetParent()
        parent_scale_attr = parent.GetAttribute("xformOp:scale") if parent else None
        parent_scale = parent_scale_attr.Get() if parent_scale_attr else None
        if parent_scale is None:
            continue

        _set_scale_op(
            prim,
            Gf.Vec3f(float(parent_scale[0]), float(parent_scale[1]), float(parent_scale[2])),
        )
        parent_scale_attr.Set(Gf.Vec3f(1.0, 1.0, 1.0))
        changed += 1

    if changed and save:
        stage.GetRootLayer().Save()
    return changed


def _movable_joints(stage: Usd.Stage) -> list[Usd.Prim]:
    return [
        prim
        for prim in stage.Traverse()
        if prim.GetTypeName() in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}
    ]


def _body_target(joint_prim: Usd.Prim, rel_name: str) -> str:
    targets = joint_prim.GetRelationship(rel_name).GetTargets()
    if not targets:
        raise ValueError(f"{joint_prim.GetPath()} has no {rel_name} target")
    return str(targets[0])


def _classify_output_name(joint_prim: Usd.Prim, link_path: str, prefix: str, use_side_names: bool) -> str:
    if not use_side_names:
        return f"{prefix}_{_tokenize_path(str(joint_prim.GetPath()))}"
    name_text = f"{joint_prim.GetName()} {link_path}".lower()
    if "left" in name_text:
        return f"{prefix}_left"
    if "right" in name_text:
        return f"{prefix}_right"
    return f"{prefix}_{_tokenize_path(link_path)}"


def generate_annotations(
    stage: Usd.Stage,
    usd_path: Path,
    prefix: str,
    tail_offset: float,
    use_side_names: bool,
) -> list[HingeAnnotation]:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=True)
    movable = _movable_joints(stage)
    annotations = []

    for joint_index, joint_prim in enumerate(movable):
        if joint_prim.GetTypeName() != "PhysicsRevoluteJoint":
            continue

        base_path = _body_target(joint_prim, "physics:body0")
        link_path = _body_target(joint_prim, "physics:body1")
        base_prim = stage.GetPrimAtPath(base_path)
        link_prim = stage.GetPrimAtPath(link_path)
        if not base_prim or not base_prim.IsValid() or not link_prim or not link_prim.IsValid():
            raise ValueError(f"{joint_prim.GetPath()} has invalid body targets")

        base_bbox = _bbox_for_prim(cache, base_prim)
        link_bbox = _bbox_for_prim(cache, link_prim)
        base_front_axis = "y"
        if base_bbox is not None and link_bbox is not None:
            base_front_axis = _nearest_axis_token(_bbox_center(link_bbox) - _bbox_center(base_bbox), fallback="y")

        rot_axis = _axis_token_from_usd(joint_prim.GetAttribute("physics:axis").Get())
        head, contact_axis = _contact_point_local(cache, link_prim, base_front_axis)
        tail_dir = _tail_direction_local(cache, link_prim, head, rot_axis)
        tail = head + tail_dir * float(tail_offset)

        annotations.append(
            HingeAnnotation(
                output_name=_classify_output_name(joint_prim, link_path, prefix, use_side_names),
                joint_index=joint_index,
                joint_path=str(joint_prim.GetPath()),
                base_path=base_path,
                link_path=link_path,
                head=_format_vec(head),
                tail=_format_vec(tail),
                rot_axis=rot_axis,
                contact_axis=contact_axis,
                base_front_axis=base_front_axis,
            )
        )

    if not annotations:
        raise ValueError(f"No PhysicsRevoluteJoint prims found in {usd_path}")
    return annotations


def _object_relative_path(path: str, object_name: str) -> str:
    object_prefix = f"/{object_name}/"
    if path.startswith(object_prefix):
        return path[len(object_prefix) :]
    return path.lstrip("/")


def _forbid_collision_paths(stage: Usd.Stage, annotation: HingeAnnotation, object_name: str) -> list[str]:
    root_prim = stage.GetPrimAtPath(f"/{object_name}")
    paths = []
    if root_prim and root_prim.IsValid():
        for prim in Usd.PrimRange(root_prim):
            if prim == root_prim:
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                paths.append(_object_relative_path(str(prim.GetPath()), object_name))

    if not paths:
        paths = [
            _object_relative_path(annotation.base_path, object_name),
            _object_relative_path(annotation.link_path, object_name),
        ]

    return sorted(dict.fromkeys(paths))


def annotation_to_json(
    stage: Usd.Stage,
    annotation: HingeAnnotation,
    object_name: str,
    usd_path: Path,
    object_scale: float,
) -> dict:
    joint_key = (
        "object_revolute_joint_path"
        if "revolute" in annotation.joint_path.lower() or "door" in annotation.joint_path.lower()
        else "object_joint_path"
    )
    data = {
        "object_keypoints": {
            "articulated_object_head": annotation.head,
            "articulated_object_tail": annotation.tail,
        },
        "object_scale": [object_scale, object_scale, object_scale, 1.0],
        "object_name": object_name,
        "object_usd": usd_path.name,
        "object_link0_rot_axis": annotation.rot_axis,
        "object_link0_contact_axis": annotation.contact_axis,
        "object_base_front_axis": annotation.base_front_axis,
        "joint_index": annotation.joint_index,
        "object_prim_path": f"/{object_name}",
        "object_link_path": annotation.link_path,
        "object_base_path": annotation.base_path,
        joint_key: annotation.joint_path,
        "object_joint_path": annotation.joint_path,
        "forbid_collision_paths": _forbid_collision_paths(stage, annotation, object_name),
    }
    return data


def find_usd_path(asset_dir: Path, requested: str | None) -> Path:
    if requested:
        path = Path(requested)
        return path if path.is_absolute() else asset_dir / path
    for name in ("model.usd", "instance.usd"):
        path = asset_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No model.usd or instance.usd found under {asset_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_dir", type=Path, help="Directory containing the hinged USD asset")
    parser.add_argument("--usd", help="USD file path or name. Defaults to model.usd, then instance.usd")
    parser.add_argument("--output-root", type=Path, help="Output Kps directory. Defaults to <asset_dir>/Kps")
    parser.add_argument("--prefix", default="open_v", help="Output annotation prefix, e.g. open_v")
    parser.add_argument("--tail-offset", type=float, default=0.2, help="Distance from head to tail keypoint in link frame")
    parser.add_argument("--object-scale", type=float, default=1.0, help="Uniform object_scale value to write")
    parser.add_argument("--no-side-names", action="store_true", help="Use joint/link names instead of left/right suffixes")
    parser.add_argument(
        "--no-normalize-cube-scale",
        action="store_true",
        help="Do not move parent Xform scale onto Cube prims for cuRobo compatibility",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing info.json files")
    parser.add_argument("--dry-run", action="store_true", help="Print generated annotations without writing files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asset_dir = args.asset_dir.resolve()
    usd_path = find_usd_path(asset_dir, args.usd).resolve()
    output_root = (args.output_root.resolve() if args.output_root else asset_dir / "Kps")

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    if not args.no_normalize_cube_scale:
        changed = normalize_cube_scales(stage, save=not args.dry_run)
        if changed:
            action = "Would normalize" if args.dry_run else "Normalized"
            print(f"{action} {changed} Cube scale ops in {usd_path}")

    object_name = _stage_root_name(stage)
    annotations = generate_annotations(
        stage=stage,
        usd_path=usd_path,
        prefix=args.prefix,
        tail_offset=args.tail_offset,
        use_side_names=not args.no_side_names,
    )

    for annotation in annotations:
        data = annotation_to_json(stage, annotation, object_name, usd_path, float(args.object_scale))
        out_path = output_root / annotation.output_name / "info.json"
        if args.dry_run:
            print(f"\n# {out_path}")
            print(json.dumps(data, indent=4))
            continue
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} already exists; pass --overwrite to replace it")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
