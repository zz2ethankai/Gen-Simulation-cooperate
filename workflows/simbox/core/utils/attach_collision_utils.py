"""Resolve and validate collision prims used when CuRobo attaches a rigid object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


@dataclass
class AttachCollisionResolution:
    """Result of resolving the runtime attach-collision contract for one object."""

    prim_paths: list[str] = field(default_factory=list)
    source: str = "unresolved"
    failure_code: str | None = None
    candidates: list[str] = field(default_factory=list)
    message: str | None = None


def join_prim_path(base_prim_path: str, child_prim_path: str) -> str:
    """Join a stage prim path with a relative child path using USD semantics."""

    child = str(child_prim_path).strip().strip("/")
    if not child:
        raise ValueError("attach collision prim path must not be empty")
    path = Sdf.Path(str(base_prim_path)).AppendPath(Sdf.Path(child))
    if path.isEmpty or not path.IsAbsolutePath() or not path.IsPrimPath():
        raise ValueError(f"invalid USD prim path: {base_prim_path!r} + {child_prim_path!r}")
    return str(path)


def has_nonempty_collision_bound(prim: Usd.Prim) -> bool:
    """Return whether a Prim is concrete collision geometry with volume."""

    if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Gprim):
        return False
    bound = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        useExtentsHint=False,
    ).ComputeLocalBound(prim).ComputeAlignedBox()
    if bound.IsEmpty():
        return False
    size = bound.GetSize()
    return all(float(size[index]) > 0.0 for index in range(3))


def collision_candidate_paths(rigid_prim: Usd.Prim) -> list[str]:
    """Return enabled, non-empty CollisionAPI geometry below a rigid body."""

    if not rigid_prim or not rigid_prim.IsValid():
        return []
    candidates: list[str] = []
    for prim in Usd.PrimRange(rigid_prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is not False and has_nonempty_collision_bound(prim):
            candidates.append(str(prim.GetPath()))
    return candidates


def _preferred_discovery_candidates(
    candidates: list[str], get_prim: Callable[[str], Usd.Prim]
) -> list[str]:
    """Prefer USD guide-purpose collision proxies over collidable visuals."""

    guide_candidates = [
        path
        for path in candidates
        if UsdGeom.Imageable(get_prim(path)).ComputePurpose() == UsdGeom.Tokens.guide
    ]
    return guide_candidates or candidates


def _configured_children(cfg: Any) -> tuple[list[str], str | None, str | None]:
    plural = cfg.get("attach_prim_path_children")
    singular = cfg.get("attach_prim_path_child")
    if plural is not None and singular is not None:
        return [], "ATTACH_COLLISION_CONFIG_CONFLICT", (
            "configure attach_prim_path_children or deprecated attach_prim_path_child, not both"
        )
    if plural is not None:
        if isinstance(plural, (str, bytes)) or not isinstance(plural, Iterable):
            return [], "ATTACH_COLLISION_CONFIG_INVALID", "attach_prim_path_children must be a list"
        values = [str(value).strip() for value in plural]
        if not values or any(not value for value in values):
            return [], "ATTACH_COLLISION_CONFIG_INVALID", "attach_prim_path_children must not be empty"
        if len(set(values)) != len(values):
            return [], "ATTACH_COLLISION_CONFIG_INVALID", "attach collision prim paths must be unique"
        return values, None, None
    if singular is not None:
        value = str(singular).strip()
        if not value:
            return [], "ATTACH_COLLISION_CONFIG_INVALID", "attach_prim_path_child must not be empty"
        return [value], None, None
    return [], None, None


def resolve_attach_collision_prims(
    base_prim_path: str,
    rigid_prim_path: str,
    cfg: Any,
    get_prim: Callable[[str], Usd.Prim],
) -> AttachCollisionResolution:
    """Resolve explicit or uniquely discoverable attach collision prims.

    Ambiguous discovery is deliberately returned as a structured failure.  A
    RigidObject may still load for rendering or as a distractor, while Pick and
    Probe reject it before motion planning.
    """

    children, failure_code, message = _configured_children(cfg)
    rigid_prim = get_prim(rigid_prim_path)
    candidates = collision_candidate_paths(rigid_prim)
    if failure_code is not None:
        return AttachCollisionResolution(
            failure_code=failure_code,
            candidates=candidates,
            message=message,
        )

    if children:
        full_paths: list[str] = []
        rigid_path = Sdf.Path(rigid_prim_path)
        for child in children:
            try:
                full_path = join_prim_path(base_prim_path, child)
            except ValueError as exc:
                return AttachCollisionResolution(
                    failure_code="ATTACH_COLLISION_CONFIG_INVALID",
                    candidates=candidates,
                    message=str(exc),
                )
            path = Sdf.Path(full_path)
            if not path.HasPrefix(rigid_path):
                return AttachCollisionResolution(
                    failure_code="ATTACH_COLLISION_PRIM_OUTSIDE_RIGID_ROOT",
                    candidates=candidates,
                    message=f"attach collision prim is outside rigid root: {full_path}",
                )
            prim = get_prim(full_path)
            if not prim or not prim.IsValid():
                return AttachCollisionResolution(
                    failure_code="ATTACH_COLLISION_PRIM_NOT_FOUND",
                    candidates=candidates,
                    message=f"attach collision prim does not exist: {full_path}",
                )
            collision = UsdPhysics.CollisionAPI(prim) if prim.HasAPI(UsdPhysics.CollisionAPI) else None
            enabled = collision.GetCollisionEnabledAttr().Get() if collision else None
            if (
                collision is None
                or enabled is False
                or not has_nonempty_collision_bound(prim)
            ):
                return AttachCollisionResolution(
                    failure_code="ATTACH_COLLISION_PRIM_NOT_COLLIDABLE",
                    candidates=candidates,
                    message=(
                        "attach prim has no enabled non-empty collision geometry: "
                        f"{full_path}"
                    ),
                )
            full_paths.append(full_path)
        return AttachCollisionResolution(
            prim_paths=full_paths,
            source="explicit_plural" if cfg.get("attach_prim_path_children") is not None else "explicit_legacy",
            candidates=candidates,
        )

    discovery_candidates = _preferred_discovery_candidates(candidates, get_prim)
    if len(discovery_candidates) == 1:
        return AttachCollisionResolution(
            prim_paths=discovery_candidates,
            source=(
                "auto_unique_guide_collision"
                if discovery_candidates != candidates
                else "auto_unique_collision"
            ),
            candidates=candidates,
        )
    if not candidates:
        return AttachCollisionResolution(
            failure_code="ATTACH_COLLISION_CONFIG_MISSING",
            candidates=[],
            message=f"no enabled collision prim was found below {rigid_prim_path}",
        )
    return AttachCollisionResolution(
        failure_code="ATTACH_COLLISION_PRIM_AMBIGUOUS",
        candidates=candidates,
        message=(
            f"found {len(discovery_candidates)} equally preferred collision prims "
            f"below {rigid_prim_path}; "
            "configure attach_prim_path_children explicitly"
        ),
    )
