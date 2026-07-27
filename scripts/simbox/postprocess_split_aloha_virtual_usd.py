#!/usr/bin/env python3
"""Build a SplitAloha asset with a supported virtual X/Y/yaw mobile base.

The source SplitAloha USD contains a dormant planar joint chain and a physical
4WIS wheel assembly.  The dormant joints are not used as-is: this script
deletes and recreates them as a canonical, world-anchored articulation chain,
fixes the physical steering/wheel joints, and disables physical base contact
collision that would otherwise lock the virtual planar joints to the floor.

The physical source asset is never modified.  The output is a separate robot
asset used by the ``SplitAloha`` virtual-base configuration; the original asset
remains available through ``SplitAlohaActual``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdPhysics


ROOT_PRIM = "/Root"
ROBOT_ROOT = f"{ROOT_PRIM}/split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper"

DUMMY_X_BODY = f"{ROBOT_ROOT}/dummy_base_x"
DUMMY_Y_BODY = f"{ROBOT_ROOT}/dummy_base_y"
DUMMY_YAW_BODY = f"{ROBOT_ROOT}/dummy_base_rotate"
BASE_BODY = f"{ROBOT_ROOT}/base_link"

ROOT_FIXED_JOINT = f"{DUMMY_X_BODY}/virtual_root_fixed_joint"
FORWARD_JOINT = f"{DUMMY_X_BODY}/mobile_translate_x"
SIDE_JOINT = f"{DUMMY_Y_BODY}/mobile_translate_y"
YAW_JOINT = f"{DUMMY_YAW_BODY}/mobile_rotate"

VIRTUAL_DOF_NAMES = ("mobile_translate_x", "mobile_translate_y", "mobile_rotate")

STEERING_JOINTS = tuple(f"{BASE_BODY}/{corner}_steering_joint" for corner in ("fl", "fr", "rl", "rr"))
WHEEL_JOINTS = tuple(
    f"{ROBOT_ROOT}/{corner}_steering_wheel_link/{corner}_wheel" for corner in ("fl", "fr", "rl", "rr")
)
PHYSICAL_BASE_JOINTS = STEERING_JOINTS + WHEEL_JOINTS

VIRTUAL_BASE_CONTACT_ROOTS = (f"{BASE_BODY}/collisions",) + tuple(
    path
    for corner in ("fl", "fr", "rl", "rr")
    for path in (
        f"{ROBOT_ROOT}/{corner}_steering_wheel_link",
        f"{ROBOT_ROOT}/{corner}_wheel_link",
    )
)

LINEAR_DRIVE_DAMPING = 1000.0
LINEAR_DRIVE_MAX_FORCE = 600.0
YAW_DRIVE_DAMPING = 1500.0
YAW_DRIVE_MAX_FORCE = 600.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_prim(stage: Usd.Stage, path: str) -> Usd.Prim:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Required SplitAloha prim does not exist: {path}")
    return prim


def _set_rigid_body_mass(prim: Usd.Prim) -> None:
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr().Set(True)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr().Set(0.1)
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(0.001, 0.001, 0.001))


def _configure_drive(prim: Usd.Prim, name: str, *, damping: float, max_force: float) -> None:
    drive = UsdPhysics.DriveAPI.Apply(prim, name)
    drive.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)
    drive.CreateStiffnessAttr().Set(0.0)
    drive.CreateDampingAttr().Set(float(damping))
    drive.CreateMaxForceAttr().Set(float(max_force))


def _define_fixed_joint(
    stage: Usd.Stage,
    path: str,
    *,
    body0: str | None,
    body1: str,
    local_pos0=None,
    local_pos1=None,
    local_rot0=None,
    local_rot1=None,
) -> None:
    stage.RemovePrim(path)
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    if body0:
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(local_pos0 or Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(local_pos1 or Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(local_rot0 or Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(local_rot1 or Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateJointEnabledAttr().Set(True)


def _replace_joint_with_fixed(stage: Usd.Stage, path: str) -> dict:
    prim = _require_prim(stage, path)
    source_joint = UsdPhysics.Joint(prim)
    body0_targets = source_joint.GetBody0Rel().GetTargets()
    body1_targets = source_joint.GetBody1Rel().GetTargets()
    if len(body0_targets) != 1 or len(body1_targets) != 1:
        raise RuntimeError(f"Expected one body0/body1 target for physical base joint: {path}")

    local_pos0 = source_joint.GetLocalPos0Attr().Get() or Gf.Vec3f(0.0, 0.0, 0.0)
    local_pos1 = source_joint.GetLocalPos1Attr().Get() or Gf.Vec3f(0.0, 0.0, 0.0)
    local_rot0 = source_joint.GetLocalRot0Attr().Get() or Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    local_rot1 = source_joint.GetLocalRot1Attr().Get() or Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    original_type = prim.GetTypeName()
    _define_fixed_joint(
        stage,
        path,
        body0=str(body0_targets[0]),
        body1=str(body1_targets[0]),
        local_pos0=local_pos0,
        local_pos1=local_pos1,
        local_rot0=local_rot0,
        local_rot1=local_rot1,
    )
    return {"path": path, "original_type": original_type, "replacement_type": "PhysicsFixedJoint"}


def _define_prismatic_joint(stage: Usd.Stage, path: str, *, body0: str, body1: str, axis) -> None:
    stage.RemovePrim(path)
    joint = UsdPhysics.PrismaticJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateAxisAttr().Set(axis)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateJointEnabledAttr().Set(True)
    _configure_drive(
        joint.GetPrim(),
        "linear",
        damping=LINEAR_DRIVE_DAMPING,
        max_force=LINEAR_DRIVE_MAX_FORCE,
    )


def _define_revolute_joint(stage: Usd.Stage, path: str, *, body0: str, body1: str, axis) -> None:
    stage.RemovePrim(path)
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateAxisAttr().Set(axis)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateJointEnabledAttr().Set(True)
    _configure_drive(
        joint.GetPrim(),
        "angular",
        damping=YAW_DRIVE_DAMPING,
        max_force=YAW_DRIVE_MAX_FORCE,
    )


def _disable_collision_tree(stage: Usd.Stage, root_path: str) -> list[str]:
    root = _require_prim(stage, root_path)
    disabled = []
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(False)
            disabled.append(str(prim.GetPath()))
    return disabled


def patch_stage(stage: Usd.Stage) -> dict:
    _require_prim(stage, ROOT_PRIM)
    for path in (ROBOT_ROOT, DUMMY_X_BODY, DUMMY_Y_BODY, DUMMY_YAW_BODY, BASE_BODY):
        _require_prim(stage, path)

    for path in (DUMMY_X_BODY, DUMMY_Y_BODY, DUMMY_YAW_BODY):
        _set_rigid_body_mass(_require_prim(stage, path))

    fixed_physical_joints = [_replace_joint_with_fixed(stage, path) for path in PHYSICAL_BASE_JOINTS]

    _define_fixed_joint(stage, ROOT_FIXED_JOINT, body0=None, body1=DUMMY_X_BODY)
    _define_prismatic_joint(
        stage,
        FORWARD_JOINT,
        body0=DUMMY_X_BODY,
        body1=DUMMY_Y_BODY,
        axis=UsdPhysics.Tokens.x,
    )
    _define_prismatic_joint(
        stage,
        SIDE_JOINT,
        body0=DUMMY_Y_BODY,
        body1=DUMMY_YAW_BODY,
        axis=UsdPhysics.Tokens.y,
    )
    _define_revolute_joint(
        stage,
        YAW_JOINT,
        body0=DUMMY_YAW_BODY,
        body1=BASE_BODY,
        axis=UsdPhysics.Tokens.z,
    )

    disabled_collisions = []
    for root_path in VIRTUAL_BASE_CONTACT_ROOTS:
        disabled_collisions.extend(_disable_collision_tree(stage, root_path))

    root = _require_prim(stage, ROOT_PRIM)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    return {
        "root_prim": ROOT_PRIM,
        "articulation_root": ROOT_PRIM,
        "virtual_root_fixed_joint": ROOT_FIXED_JOINT,
        "virtual_base_joint_chain": [FORWARD_JOINT, SIDE_JOINT, YAW_JOINT],
        "virtual_base_dof_names": list(VIRTUAL_DOF_NAMES),
        "fixed_physical_base_joints": fixed_physical_joints,
        "disabled_virtual_base_contact_collisions": disabled_collisions,
        "linear_drive_damping": LINEAR_DRIVE_DAMPING,
        "linear_drive_max_force": LINEAR_DRIVE_MAX_FORCE,
        "yaw_drive_damping": YAW_DRIVE_DAMPING,
        "yaw_drive_max_force": YAW_DRIVE_MAX_FORCE,
        "notes": [
            "The source dormant planar joints were deleted and recreated; they were not assumed usable.",
            "The virtual articulation is anchored to the world through a fixed root joint.",
            "Physical steering and wheel joints are fixed; wheel and base-link ground-contact collisions are disabled.",
            "The real chassis footprint remains in Nav2; lifting-body, upper-body, and arm collision remain enabled.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("InternDataAssets/assets/split_aloha_mid_360/robot.usd"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("InternDataAssets/assets/split_aloha_mid_360_virtual/robot.usd"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("InternDataAssets/assets/split_aloha_mid_360_virtual/virtual_postprocess_metadata.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    metadata_path = args.metadata_output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path == output_path:
        raise ValueError("Virtual SplitAloha output must be separate from the physical source asset")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    stage = Usd.Stage.Open(str(output_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {output_path}")
    metadata = patch_stage(stage)
    metadata.update(
        {
            "source_usd": str(input_path),
            "source_sha256": _sha256(input_path),
            "output_usd": str(output_path),
        }
    )
    stage.GetRootLayer().Save()
    metadata["output_sha256"] = _sha256(output_path)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
