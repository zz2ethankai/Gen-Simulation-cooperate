#!/usr/bin/env python3
"""Draw a top-down SimBox task layout from simbox_task.yaml.

The plot is intended for task authoring before skill tuning. It visualizes the
runtime YAMLs, not the USD stage itself, so footprints are bbox/layout
approximations. Use Isaac Sim for final collision and IK validation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Circle, FancyArrowPatch, Polygon


@dataclass
class Entity:
    name: str
    kind: str
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    size_xy: tuple[float, float] = (0.08, 0.08)
    category: str = ""
    role: str = ""
    color: str = "#cccccc"
    alpha: float = 0.55
    priority: int = 50
    notes: list[str] = field(default_factory=list)
    cfg: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArmReach:
    robot_name: str
    arm_name: str
    x: float
    y: float
    z: float
    radius: float
    color: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, bases: list[Path]) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    for base in bases:
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate
    return (bases[0] / p).resolve()


def load_yaml(path: Path) -> Any:
    try:
        from omegaconf import OmegaConf  # type: ignore

        conf = OmegaConf.load(str(path))
        OmegaConf.resolve(conf)
        return OmegaConf.to_container(conf, resolve=True)
    except Exception:
        pass

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def task_from_doc(doc: dict[str, Any], task_index: int) -> dict[str, Any]:
    if "tasks" in doc:
        return doc["tasks"][task_index]
    return doc


def euler_yaw(cfg: dict[str, Any]) -> float:
    euler = cfg.get("euler") or cfg.get("rotation") or [0.0, 0.0, 0.0]
    return float(euler[2]) if len(euler) >= 3 else 0.0


def midpoint_range(values: list[list[float]]) -> list[float]:
    lo, hi = values
    return [(float(a) + float(b)) * 0.5 for a, b in zip(lo, hi)]


def clean_name(name: str) -> str:
    short = name
    for prefix in ("livingroom_", "round_", "split_"):
        short = short.replace(prefix, "")
    short = short.replace("_0_id", "#")
    short = short.replace("__0__id", "#")
    return short


def truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def load_metadata(cfg: dict[str, Any], asset_root: Path) -> dict[str, Any] | None:
    candidates: list[Path] = []
    for key in ("source_metadata", "metadata"):
        value = cfg.get(key)
        if isinstance(value, str):
            candidates.append(resolve_path(value, [asset_root, repo_root()]))

    path_value = cfg.get("path") or cfg.get("usd_path")
    if isinstance(path_value, str):
        asset_path = resolve_path(path_value, [asset_root, repo_root()])
        candidates.append(asset_path.parent / "metadata.json")

    for path in candidates:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return None
    return None


@lru_cache(maxsize=256)
def usd_bbox_size_xyz(usd_path: str) -> tuple[float, float, float] | None:
    try:
        from pxr import Usd, UsdGeom  # type: ignore
    except Exception:
        return None

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        return None
    prim = stage.GetDefaultPrim()
    if not prim or not prim.IsValid():
        prim = stage.GetPseudoRoot()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    mn, mx = bbox.GetMin(), bbox.GetMax()
    return abs(float(mx[0] - mn[0])), abs(float(mx[1] - mn[1])), abs(float(mx[2] - mn[2]))


def scaled_usd_size(cfg: dict[str, Any], asset_root: Path) -> tuple[float, float, float] | None:
    path_value = cfg.get("path") or cfg.get("usd_path")
    if not isinstance(path_value, str):
        return None
    usd_path = resolve_path(path_value, [asset_root, repo_root()])
    if not usd_path.exists():
        return None
    size = usd_bbox_size_xyz(str(usd_path))
    if size is None:
        return None
    scale = cfg.get("scale") or [1.0, 1.0, 1.0]
    if not isinstance(scale, list):
        scale = [float(scale), float(scale), float(scale)]
    scale = list(scale) + [1.0, 1.0, 1.0]
    return tuple(max(size[i] * abs(float(scale[i])), 0.01) for i in range(3))


def footprint_xy(cfg: dict[str, Any], asset_root: Path, default: tuple[float, float] = (0.08, 0.08)) -> tuple[float, float]:
    size = cfg.get("size")
    if isinstance(size, list) and len(size) >= 2 and cfg.get("target_class") == "PlaneObject":
        return abs(float(size[0])), abs(float(size[1]))

    meta = load_metadata(cfg, asset_root)
    if meta:
        usd_size = meta.get("geometry_alignment", {}).get("usd_size_xyz_m")
        if isinstance(usd_size, list) and len(usd_size) >= 2:
            return max(abs(float(usd_size[0])), 0.01), max(abs(float(usd_size[1])), 0.01)

        layout_size = meta.get("layout_pose", {}).get("size_xyz_m")
        if isinstance(layout_size, list) and len(layout_size) >= 3:
            return max(abs(float(layout_size[0])), 0.01), max(abs(float(layout_size[2])), 0.01)

    if isinstance(size, list) and len(size) >= 3:
        # Source task.yaml uses layout [x, height, y]. This is a fallback only.
        return max(abs(float(size[0])), 0.01), max(abs(float(size[2])), 0.01)

    usd_size = scaled_usd_size(cfg, asset_root)
    if usd_size is not None:
        return usd_size[0], usd_size[1]

    return default


def rectangle_corners(x: float, y: float, width: float, height: float, yaw_deg: float) -> list[tuple[float, float]]:
    theta = math.radians(yaw_deg)
    c, s = math.cos(theta), math.sin(theta)
    local = [(-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2), (-width / 2, height / 2)]
    return [(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local]


def world_from_region(region: dict[str, Any], centers: dict[str, tuple[float, float, float]]) -> tuple[float, float, float, float]:
    target = region["target"]
    base = centers.get(target)
    if base is None:
        raise KeyError(f"Unknown region target: {target}")
    shift = midpoint_range(region["random_config"]["pos_range"])
    yaw = midpoint_range([region["random_config"].get("yaw_rotation", [0.0, 0.0])] * 2)[0]
    return base[0] + shift[0], base[1] + shift[1], base[2] + shift[2], yaw


def collect_skill_roles(skills: Any) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}

    def add(obj: str, role: str) -> None:
        if obj:
            roles.setdefault(obj, [])
            if role not in roles[obj]:
                roles[obj].append(role)

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            name = node.get("name")
            objects = node.get("objects") or []
            if name == "pick" and objects:
                add(str(objects[0]), "pick source")
            elif name == "place" and objects:
                add(str(objects[0]), "place moving")
                if len(objects) > 1:
                    add(str(objects[1]), "place target")
            for value in node.values():
                visit(value)

    visit(skills)
    return roles


def build_entities(task: dict[str, Any], arena: dict[str, Any], asset_root: Path) -> tuple[list[Entity], dict[str, Any]]:
    fixtures: dict[str, Entity] = {}
    entities: list[Entity] = []
    centers: dict[str, tuple[float, float, float]] = {}

    for cfg in arena.get("fixtures", []):
        trans = cfg.get("translation", [0.0, 0.0, 0.0])
        name = cfg["name"]
        category = cfg.get("asset_category") or cfg.get("category") or cfg.get("role", "") or name
        role = cfg.get("role", "")
        target_class = cfg.get("target_class", "")
        size_xy = footprint_xy(cfg, asset_root, default=(0.10, 0.10))
        kind = "fixture"
        color, alpha, priority = "#c8b28c", 0.40, 70
        if name == "floor":
            kind, color, alpha, priority = "floor", "#f7f3df", 0.55, 95
        elif target_class == "PlaneObject":
            kind, color, alpha, priority = "wall", "#eeeeee", 0.25, 99
        elif role == "wall":
            kind, color, alpha, priority = "wall", "#eeeeee", 0.25, 99
        elif cfg.get("support_surface") or any(token in f"{name} {category}".lower() for token in ("table", "counter", "cabinet", "shelf", "desk")):
            kind, color, alpha, priority = "support", "#e8c78f", 0.55, 35
        elif "/small_objects/" in str(cfg.get("path", "")) or (max(size_xy) <= 0.22 and role != "wall" and not cfg.get("support_surface")):
            kind, color, alpha, priority = "static small", "#9db6c7", 0.65, 45
        elif "rug" in category:
            color, alpha = "#b7b7b7", 0.35
        elif "sofa" in category:
            color, alpha = "#c8c8c8", 0.45

        ent = Entity(
            name=name,
            kind=kind,
            x=float(trans[0]),
            y=float(trans[1]),
            z=float(trans[2]) if len(trans) > 2 else 0.0,
            yaw=euler_yaw(cfg),
            size_xy=size_xy,
            category=category,
            role=role or target_class,
            color=color,
            alpha=alpha,
            priority=priority,
            cfg=cfg,
        )
        fixtures[name] = ent
        centers[name] = (ent.x, ent.y, ent.z)
        entities.append(ent)

    skill_roles = collect_skill_roles(task.get("skills", []))
    regions = {cfg["object"]: cfg for cfg in task.get("regions", [])}

    for cfg in task.get("objects", []):
        name = cfg["name"]
        if name not in regions:
            continue
        x, y, z, region_yaw = world_from_region(regions[name], centers)
        size_xy = footprint_xy(cfg, asset_root, default=(0.08, 0.08))
        notes = skill_roles.get(name, ["task object"])
        category = cfg.get("asset_category") or cfg.get("category", "")
        color = "#ffffff"
        if "coaster" in category:
            color = "#f2d35f"
        elif "mug" in category:
            color = "#ffffff"
        ent = Entity(
            name=name,
            kind="task object",
            x=x,
            y=y,
            z=z,
            yaw=euler_yaw(cfg) + region_yaw,
            size_xy=size_xy,
            category=category,
            role=cfg.get("role", ""),
            color=color,
            alpha=0.95,
            priority=5,
            notes=notes,
            cfg=cfg,
        )
        centers[name] = (x, y, z)
        entities.append(ent)

    source_regions = {cfg.get("A"): cfg for cfg in task.get("source_regions", []) if isinstance(cfg, dict)}
    for cfg in task.get("robots", []):
        name = cfg["name"]
        if name not in regions:
            continue
        x, y, z, region_yaw = world_from_region(regions[name], centers)
        sr = source_regions.get("robot", {})
        footprint = sr.get("footprint_m") or sr.get("base_footprint_m") or [0.70, 0.40]
        ent = Entity(
            name=name,
            kind="robot",
            x=x,
            y=y,
            z=z,
            yaw=euler_yaw(cfg) + region_yaw,
            size_xy=(float(footprint[0]), float(footprint[1])),
            category="robot",
            role="mobile base",
            color="#6aaed6",
            alpha=0.85,
            priority=1,
            notes=["robot base"],
            cfg=cfg,
        )
        centers[name] = (x, y, z)
        entities.append(ent)

    summary = {
        "task": task.get("name"),
        "asset_root": str(asset_root),
        "regions": task.get("regions", []),
        "entities": [
            {
                "name": e.name,
                "kind": e.kind,
                "category": e.category,
                "xy": [round(e.x, 4), round(e.y, 4)],
                "z": round(e.z, 4),
                "yaw_deg": round(e.yaw, 4),
                "size_xy": [round(e.size_xy[0], 4), round(e.size_xy[1], 4)],
                "notes": e.notes,
            }
            for e in entities
            if e.kind not in {"wall"}
        ],
    }
    return entities, summary


def merge_robot_cfg(robot_cfg: dict[str, Any], task_dir: Path, asset_root: Path) -> dict[str, Any]:
    cfg = dict(robot_cfg)
    robot_config_file = cfg.get("robot_config_file")
    if robot_config_file:
        path = resolve_path(robot_config_file, [repo_root(), task_dir, asset_root])
        if path.exists():
            base_cfg = load_yaml(path)
            merged = dict(base_cfg)
            if "path" in base_cfg:
                merged["_default_robot_path"] = base_cfg["path"]
            merged.update(cfg)
            cfg = merged
    return cfg


def resolve_robot_usd_path(path: str | Path, task_dir: Path, asset_root: Path) -> Path:
    root = repo_root()
    return resolve_path(
        path,
        [
            asset_root,
            task_dir,
            root,
            root / "workflows" / "simbox" / "example_assets",
            root / "InternDataAssets" / "assets",
        ],
    )


def usd_relative_translation(usd_path: Path, rel_prim_path: str) -> tuple[float, float, float] | None:
    try:
        from pxr import Usd, UsdGeom  # type: ignore
    except Exception:
        return None

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        return None
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        return None

    root_path = str(default_prim.GetPath())
    prim_path = f"{root_path}/{rel_prim_path.strip('/')}"
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None

    cache = UsdGeom.XformCache()
    root_mat = cache.GetLocalToWorldTransform(default_prim)
    prim_mat = cache.GetLocalToWorldTransform(prim)
    rel_mat = prim_mat * root_mat.GetInverse()
    t = rel_mat.ExtractTranslation()
    return float(t[0]), float(t[1]), float(t[2])


def rotate_xy(x: float, y: float, yaw_deg: float) -> tuple[float, float]:
    theta = math.radians(yaw_deg)
    c, s = math.cos(theta), math.sin(theta)
    return x * c - y * s, x * s + y * c


def collect_arm_reaches(
    task: dict[str, Any],
    task_dir: Path,
    asset_root: Path,
    robot_entities: list[Entity],
    arm_reach_radius: float,
) -> list[ArmReach]:
    by_name = {e.name: e for e in robot_entities}
    reaches: list[ArmReach] = []
    for raw_cfg in task.get("robots", []):
        robot_name = raw_cfg.get("name")
        robot = by_name.get(robot_name)
        if robot is None:
            continue
        cfg = merge_robot_cfg(raw_cfg, task_dir, asset_root)
        robot_paths = [cfg.get("path"), cfg.get("_default_robot_path")]
        usd_path = None
        for robot_path in robot_paths:
            if not robot_path:
                continue
            candidate = resolve_robot_usd_path(robot_path, task_dir, asset_root)
            if candidate.exists():
                usd_path = candidate
                break
        if usd_path is None:
            continue
        arm_specs = [
            ("left", cfg.get("fl_base_path"), "#2ca02c"),
            ("right", cfg.get("fr_base_path"), "#ff7f0e"),
        ]
        for arm_name, base_path, color in arm_specs:
            if not base_path:
                continue
            offset = usd_relative_translation(usd_path, str(base_path))
            if offset is None:
                continue
            dx, dy = rotate_xy(offset[0], offset[1], robot.yaw)
            reaches.append(
                ArmReach(
                    robot_name=robot_name,
                    arm_name=arm_name,
                    x=robot.x + dx,
                    y=robot.y + dy,
                    z=robot.z + offset[2],
                    radius=arm_reach_radius,
                    color=color,
                )
            )
    return reaches


class LabelPlacer:
    def __init__(self, ax, xlim: tuple[float, float], ylim: tuple[float, float]):
        self.ax = ax
        self.xlim = xlim
        self.ylim = ylim
        self.boxes: list[tuple[float, float, float, float]] = []

    @staticmethod
    def estimate_box(text: str, fontsize: int) -> tuple[float, float]:
        lines = text.splitlines() or [text]
        # Conservative data-space estimate. It intentionally overestimates so
        # labels stay readable in dense tabletop regions.
        width = max(len(line) for line in lines) * fontsize * 0.006
        height = len(lines) * fontsize * 0.025
        return width, height

    @staticmethod
    def overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
        return not (a[1] < b[0] or a[0] > b[1] or a[3] < b[2] or a[2] > b[3])

    def add(
        self,
        x: float,
        y: float,
        text: str,
        fontsize: int = 8,
        color: str = "#111",
        priority: int = 50,
        with_arrow: bool = True,
        bbox_facecolor: str = "white",
        bbox_alpha: float = 0.86,
        bbox_pad: float = 0.18,
    ) -> None:
        width, height = self.estimate_box(text, fontsize)
        base_offsets = [
            (0.08, 0.10), (-0.08 - width, 0.10), (0.08, -0.10 - height), (-0.08 - width, -0.10 - height),
            (0.18, 0.00), (-0.18 - width, 0.00), (0.00, 0.18), (0.00, -0.18 - height),
            (0.22, 0.18), (-0.22 - width, 0.18), (0.22, -0.18 - height), (-0.22 - width, -0.18 - height),
            (0.34, 0.25), (-0.34 - width, 0.25), (0.34, -0.25 - height), (-0.34 - width, -0.25 - height),
        ]
        chosen = None
        for scale in (1.0, 1.35, 1.8, 2.4, 3.0, 3.8):
            for ox, oy in base_offsets:
                tx, ty = x + ox * scale, y + oy * scale
                box = (tx, tx + width, ty, ty + height)
                in_bounds = box[0] >= self.xlim[0] and box[1] <= self.xlim[1] and box[2] >= self.ylim[0] and box[3] <= self.ylim[1]
                if in_bounds and not any(self.overlaps(box, old) for old in self.boxes):
                    chosen = (tx, ty, box)
                    break
            if chosen:
                break
        if chosen is None:
            tx, ty = x + 0.08, y + 0.08
            chosen = (tx, ty, (tx, tx + width, ty, ty + height))

        tx, ty, box = chosen
        self.boxes.append(box)
        arrowprops = {"arrowstyle": "-", "lw": 0.7, "color": "#555", "shrinkA": 0, "shrinkB": 2} if with_arrow else None
        bbox = {"boxstyle": f"round,pad={bbox_pad}", "facecolor": bbox_facecolor, "edgecolor": "#999", "alpha": bbox_alpha}
        self.ax.annotate(text, xy=(x, y), xytext=(tx, ty), fontsize=fontsize, color=color, bbox=bbox, arrowprops=arrowprops, zorder=100 - min(priority, 90))


def draw_frame(ax, x: float, y: float, yaw_deg: float, length: float, label: str | None = None) -> None:
    theta = math.radians(yaw_deg)
    x_dir = (math.cos(theta), math.sin(theta))
    y_dir = (math.cos(theta + math.pi / 2), math.sin(theta + math.pi / 2))
    ax.add_patch(FancyArrowPatch((x, y), (x + length * x_dir[0], y + length * x_dir[1]), arrowstyle="-|>", mutation_scale=9, lw=1.4, color="#d62728", zorder=20))
    ax.add_patch(FancyArrowPatch((x, y), (x + length * y_dir[0], y + length * y_dir[1]), arrowstyle="-|>", mutation_scale=9, lw=1.4, color="#2ca02c", zorder=20))
    if label:
        ax.text(x + length * x_dir[0], y + length * x_dir[1], "+x", color="#d62728", fontsize=7, zorder=21)
        ax.text(x + length * y_dir[0], y + length * y_dir[1], "+y", color="#2ca02c", fontsize=7, zorder=21)


def draw_code_tag(ax, ent: Entity, code: str, entities_by_name: dict[str, Entity]) -> None:
    parent_name = ent.cfg.get("parent_fixture") or ent.cfg.get("parent") or ""
    parent = entities_by_name.get(str(parent_name))
    if parent:
        dx = 0.025 if ent.x >= parent.x else -0.025
        dy = 0.025 if ent.y >= parent.y else -0.025
    else:
        dx, dy = 0.025, 0.025
    ha = "left" if dx >= 0 else "right"
    va = "bottom" if dy >= 0 else "top"
    ax.text(
        ent.x + dx,
        ent.y + dy,
        code,
        ha=ha,
        va=va,
        fontsize=6,
        color="#333",
        bbox={"boxstyle": "round,pad=0.06", "facecolor": "#f4fbff", "edgecolor": "#999", "alpha": 0.90},
        zorder=45,
    )


def panel_line(ax, y: float, text: str, fontsize: float = 6.9, weight: str = "normal", color: str = "#222") -> float:
    ax.text(
        0.02,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        family="monospace" if weight == "normal" else "sans-serif",
    )
    return y - (0.030 if fontsize <= 7.2 else 0.036)


def draw_side_panel(
    ax,
    task_objects: list[Entity],
    small_objects: list[Entity],
    small_label_indices: dict[str, str],
    arm_reaches: list[ArmReach],
    small_label_mode: str,
) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.0,
        1.0,
        "Object index",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#111",
    )
    y = 0.94

    y = panel_line(ax, y, "Core task objects", fontsize=8.3, weight="bold")
    for ent in task_objects:
        note = ", ".join(ent.notes[:2]) if ent.notes else ent.category or ent.kind
        row = f"T  {clean_name(ent.name)} ({ent.x:.2f},{ent.y:.2f}) {note}"
        y = panel_line(ax, y, truncate(row, 72))

    if small_label_mode != "none":
        y -= 0.012
        y = panel_line(ax, y, "Static small objects", fontsize=8.3, weight="bold")
        if not small_objects:
            y = panel_line(ax, y, "none")
        for ent in small_objects:
            code = small_label_indices.get(ent.name, "--")
            parent = ent.cfg.get("parent_fixture") or ent.cfg.get("parent") or ""
            suffix = f" on {clean_name(str(parent))}" if parent else ""
            row = f"{code} {clean_name(ent.name)} ({ent.x:.2f},{ent.y:.2f}){suffix}"
            y = panel_line(ax, y, truncate(row, 72))

    if arm_reaches:
        y -= 0.012
        y = panel_line(ax, y, "Arm reference centers", fontsize=8.3, weight="bold")
        for arm in arm_reaches:
            row = f"{arm.arm_name:<5} ({arm.x:.2f},{arm.y:.2f}) r~{arm.radius:.2f}m"
            y = panel_line(ax, y, row)

    y = max(y - 0.018, 0.17)
    y = panel_line(ax, y, "Legend", fontsize=8.3, weight="bold")
    legend = [
        "T labels: task-critical objects",
        "S labels: static small objects",
        "Tan: support surfaces/furniture",
        "Gray: non-support fixtures/rug",
        "Blue rectangle: robot base",
        "Dashed/dotted: base-distance refs",
        "Dash-dot: per-arm XY ref, not IK",
        "Red/green axes: local +x/+y",
    ]
    for row in legend:
        y = panel_line(ax, y, truncate(row, 48), fontsize=7.0, color="#333")


def draw_layout(
    task_path: Path,
    task_index: int,
    output: Path,
    reach_radii: list[float],
    label_distractors: bool = False,
    small_label_mode: str = "index",
    arm_reach_radius: float = 0.62,
    draw_arm_reach: bool = True,
) -> dict[str, Any]:
    root = repo_root()
    task_doc = load_yaml(task_path)
    task = task_from_doc(task_doc, task_index)
    task_dir = task_path.parent
    # Task-local relative roots such as "../../.." are common in delivered
    # assets. Resolve them from the task file first; resolving from repo root
    # can accidentally point at broad existing directories like /home.
    asset_root = resolve_path(task.get("asset_root", "."), [task_dir, root])
    arena_path = resolve_path(task["arena_file"], [task_dir, asset_root, root])
    arena = load_yaml(arena_path)

    entities, summary = build_entities(task, arena, asset_root)
    floor = next((e for e in entities if e.name == "floor"), None)
    if floor:
        xlim = (floor.x - floor.size_xy[0] / 2 - 0.25, floor.x + floor.size_xy[0] / 2 + 0.25)
        ylim = (floor.y - floor.size_xy[1] / 2 - 0.25, floor.y + floor.size_xy[1] / 2 + 0.25)
    else:
        xs = [e.x for e in entities]
        ys = [e.y for e in entities]
        xlim = (min(xs) - 0.5, max(xs) + 0.5)
        ylim = (min(ys) - 0.5, max(ys) + 0.5)

    if label_distractors and small_label_mode == "index":
        small_label_mode = "inline-near"

    fig = plt.figure(figsize=(17.2, 8.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.8, 1.85], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    panel_ax = fig.add_subplot(gs[0, 1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, color="#d0d0d0", linewidth=0.6)
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_title(f"{arena.get('name') or task.get('name')} - SimBox top-down layout")

    labeler = LabelPlacer(ax, xlim, ylim)

    for ent in sorted(entities, key=lambda e: e.priority, reverse=True):
        if ent.kind == "wall":
            continue
        if ent.kind == "floor":
            corners = rectangle_corners(ent.x, ent.y, ent.size_xy[0], ent.size_xy[1], ent.yaw)
            ax.add_patch(Polygon(corners, closed=True, facecolor=ent.color, edgecolor="#333", lw=1.4, alpha=ent.alpha, zorder=0))
            continue
        corners = rectangle_corners(ent.x, ent.y, ent.size_xy[0], ent.size_xy[1], ent.yaw)
        edge = "#003f66" if ent.kind == "robot" else "#333"
        lw = 1.6 if ent.kind in {"robot", "task object", "support"} else 0.9
        ax.add_patch(Polygon(corners, closed=True, facecolor=ent.color, edgecolor=edge, lw=lw, alpha=ent.alpha, zorder=10 if ent.kind != "fixture" else 3))
        ax.plot(ent.x, ent.y, marker="o", color=edge, markersize=2.8, zorder=15)

    robots = [e for e in entities if e.kind == "robot"]
    task_objects = [e for e in entities if e.kind == "task object"]
    task_object_names = {e.name for e in task_objects}
    static_small_objects = [e for e in entities if e.kind == "static small" and e.name not in task_object_names]
    arm_reaches = (
        collect_arm_reaches(task, task_dir, asset_root, robots, arm_reach_radius)
        if draw_arm_reach
        else []
    )

    for ent in robots:
        draw_frame(ax, ent.x, ent.y, ent.yaw, 0.32, label="robot")
        for idx, radius in enumerate(reach_radii):
            style = "--" if idx == 0 else ":"
            ax.add_patch(Circle((ent.x, ent.y), radius, fill=False, linestyle=style, lw=1.35, edgecolor="#005f99", alpha=0.82, zorder=7))
            ax.text(ent.x + radius * 0.72, ent.y + radius * 0.72, f"{radius:.2f}m ref", fontsize=7, color="#005f99", zorder=8)

    for arm in arm_reaches:
        ax.plot(arm.x, arm.y, marker="s", color=arm.color, markersize=5, zorder=25)
        ax.add_patch(
            Circle(
                (arm.x, arm.y),
                arm.radius,
                fill=False,
                linestyle="-.",
                lw=1.25,
                edgecolor=arm.color,
                alpha=0.70,
                zorder=6,
            )
        )
        labeler.add(
            arm.x,
            arm.y,
            f"{arm.arm_name} arm_base\n({arm.x:.2f}, {arm.y:.2f})\n~{arm.radius:.2f}m FK XY ref",
            fontsize=7,
            color="#111",
            priority=15,
        )

    for ent in task_objects:
        draw_frame(ax, ent.x, ent.y, ent.yaw, 0.13, label="object")

    if small_label_mode in {"index", "near-index", "inline-all", "inline-near"}:
        if small_label_mode in {"near-index", "inline-near"}:
            labelled_small_objects = [
                ent
                for ent in static_small_objects
                if any(math.hypot(ent.x - obj.x, ent.y - obj.y) < 0.90 for obj in task_objects)
            ]
        else:
            labelled_small_objects = list(static_small_objects)
    else:
        labelled_small_objects = []

    labelled_small_objects.sort(key=lambda ent: (str(ent.cfg.get("parent_fixture") or ""), ent.category, ent.name))
    small_label_indices = {ent.name: f"S{i:02d}" for i, ent in enumerate(labelled_small_objects, 1)}

    if small_label_mode in {"index", "near-index"}:
        entities_by_name = {ent.name: ent for ent in entities}
        for ent in labelled_small_objects:
            draw_code_tag(ax, ent, small_label_indices[ent.name], entities_by_name)

    # World origin and axes.
    ax.plot([0], [0], marker="o", color="#d62728", markersize=4.5, zorder=30)
    ax.add_patch(FancyArrowPatch((0, 0), (0.75, 0), arrowstyle="-|>", mutation_scale=14, lw=2, color="#d62728", zorder=30))
    ax.add_patch(FancyArrowPatch((0, 0), (0, 0.75), arrowstyle="-|>", mutation_scale=14, lw=2, color="#2ca02c", zorder=30))
    labeler.add(0, 0, "origin\n(0.00, 0.00)\nworld +X red, +Y green", fontsize=8, color="#111", priority=0)

    # Label important entities first. Avoid labeling every distant fixture by default.
    label_entities: list[Entity] = []
    label_entities.extend(robots)
    label_entities.extend(task_objects)
    label_entities.extend([e for e in entities if e.kind == "support"])
    if small_label_mode in {"inline-all", "inline-near"}:
        label_entities.extend(labelled_small_objects)
    for ent in entities:
        if ent.kind == "fixture" and ent.category in {"three_seat_sofa", "tv_cabinet", "side_cabinet", "open_storage_shelf", "central_gray_rug"}:
            label_entities.append(ent)

    seen = set()
    for ent in sorted(label_entities, key=lambda e: e.priority):
        if ent.name in seen:
            continue
        seen.add(ent.name)
        short = clean_name(ent.name)
        note = ", ".join(ent.notes[:2]) if ent.notes else ent.category or ent.kind
        text = f"{short}\n({ent.x:.2f}, {ent.y:.2f})"
        if note:
            text += f"\n{note}"
        labeler.add(ent.x, ent.y, text, fontsize=8, priority=ent.priority)

    draw_side_panel(
        panel_ax,
        task_objects,
        labelled_small_objects,
        small_label_indices,
        arm_reaches,
        small_label_mode,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary["task_path"] = str(task_path)
    summary["arena_path"] = str(arena_path)
    summary["output_png"] = str(output)
    summary["arm_reaches"] = [
        {
            "robot": arm.robot_name,
            "arm": arm.arm_name,
            "xy": [round(arm.x, 4), round(arm.y, 4)],
            "z": round(arm.z, 4),
            "radius": round(arm.radius, 4),
        }
        for arm in arm_reaches
    ]
    summary["small_label_mode"] = small_label_mode
    summary["small_label_indices"] = [
        {
            "code": small_label_indices[ent.name],
            "name": ent.name,
            "category": ent.category,
            "parent_fixture": ent.cfg.get("parent_fixture") or ent.cfg.get("parent") or "",
            "xy": [round(ent.x, 4), round(ent.y, 4)],
        }
        for ent in labelled_small_objects
    ]
    summary_path = output.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    summary["output_json"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cfg_path", type=Path, help="Path to simbox_task.yaml")
    parser.add_argument("--task-index", type=int, default=0, help="Task index if cfg_path contains tasks:")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--reach-radii", default="0.60,0.80", help="Comma-separated reference radii around robot base")
    parser.add_argument(
        "--small-label-mode",
        default="index",
        choices=["index", "near-index", "inline-all", "inline-near", "none"],
        help="How to label static small objects. index keeps the map readable by using S01/S02 tags plus the side panel.",
    )
    parser.add_argument("--label-distractors", action="store_true", help="Backward-compatible alias for --small-label-mode inline-near")
    parser.add_argument("--arm-reach-radius", type=float, default=0.62, help="Per-arm FK XY reference radius in meters")
    parser.add_argument("--no-arm-reach", action="store_true", help="Do not draw per-arm reference circles")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = args.cfg_path.resolve()
    task_doc = load_yaml(cfg_path)
    task = task_from_doc(task_doc, args.task_index)
    output = args.output
    if output is None:
        safe_name = str(task.get("name") or cfg_path.stem).replace("/", "_")
        output = repo_root() / "output" / "debug" / f"{safe_name}_layout.png"
    reach_radii = [float(x.strip()) for x in args.reach_radii.split(",") if x.strip()]
    summary = draw_layout(
        cfg_path,
        args.task_index,
        output.resolve(),
        reach_radii,
        label_distractors=args.label_distractors,
        small_label_mode=args.small_label_mode,
        arm_reach_radius=args.arm_reach_radius,
        draw_arm_reach=not args.no_arm_reach,
    )
    print(summary["output_png"])
    print(summary["output_json"])


if __name__ == "__main__":
    main()
