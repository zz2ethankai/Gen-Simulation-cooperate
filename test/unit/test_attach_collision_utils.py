"""Offline tests for USD attach-collision path resolution."""

from __future__ import annotations

import sys
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.utils.attach_collision_utils import resolve_attach_collision_prims  # noqa: E402


def _stage(collision_names=("collision",)):
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/object", "Xform")
    rigid = stage.DefinePrim("/World/object/Aligned", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(rigid)
    for name in collision_names:
        mesh = UsdGeom.Mesh.Define(stage, f"/World/object/Aligned/{name}").GetPrim()
        UsdPhysics.CollisionAPI.Apply(mesh)
    return stage


def test_unique_collision_is_discovered_without_child_order_guessing():
    stage = _stage()
    result = resolve_attach_collision_prims(
        "/World/object",
        "/World/object/Aligned",
        {"prim_path_child": "Aligned"},
        stage.GetPrimAtPath,
    )
    assert result.failure_code is None
    assert result.source == "auto_unique_collision"
    assert result.prim_paths == ["/World/object/Aligned/collision"]


def test_multiple_collision_prims_require_explicit_configuration():
    stage = _stage(("part_0", "part_1"))
    result = resolve_attach_collision_prims(
        "/World/object",
        "/World/object/Aligned",
        {"prim_path_child": "Aligned"},
        stage.GetPrimAtPath,
    )
    assert result.failure_code == "ATTACH_COLLISION_PRIM_AMBIGUOUS"
    assert result.prim_paths == []
    assert result.candidates == [
        "/World/object/Aligned/part_0",
        "/World/object/Aligned/part_1",
    ]


def test_plural_configuration_preserves_all_paths():
    stage = _stage(("part_0", "part_1"))
    result = resolve_attach_collision_prims(
        "/World/object",
        "/World/object/Aligned",
        {
            "prim_path_child": "Aligned",
            "attach_prim_path_children": ["Aligned/part_0", "Aligned/part_1"],
        },
        stage.GetPrimAtPath,
    )
    assert result.failure_code is None
    assert result.source == "explicit_plural"
    assert result.prim_paths == [
        "/World/object/Aligned/part_0",
        "/World/object/Aligned/part_1",
    ]


def test_explicit_non_collision_prim_is_rejected():
    stage = _stage()
    stage.DefinePrim("/World/object/Aligned/visual_only", "Mesh")
    result = resolve_attach_collision_prims(
        "/World/object",
        "/World/object/Aligned",
        {
            "prim_path_child": "Aligned",
            "attach_prim_path_children": ["Aligned/visual_only"],
        },
        stage.GetPrimAtPath,
    )
    assert result.failure_code == "ATTACH_COLLISION_PRIM_NOT_COLLIDABLE"


def test_explicit_path_cannot_escape_rigid_root():
    stage = _stage()
    outside = UsdGeom.Mesh.Define(stage, "/World/object/outside").GetPrim()
    UsdPhysics.CollisionAPI.Apply(outside)
    result = resolve_attach_collision_prims(
        "/World/object",
        "/World/object/Aligned",
        {
            "prim_path_child": "Aligned",
            "attach_prim_path_children": ["outside"],
        },
        stage.GetPrimAtPath,
    )
    assert result.failure_code == "ATTACH_COLLISION_PRIM_OUTSIDE_RIGID_ROOT"
