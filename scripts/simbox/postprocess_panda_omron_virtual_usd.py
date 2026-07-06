#!/usr/bin/env python3
"""Patch PandaOmron USD into an Isaac-compatible virtual mobile-base model.

Isaac Sim 4.1 imports RoboSuite's three planar Omron base joints as one D6
joint.  PhysX articulations ignore linear drives on that D6 joint, so this
script replaces it with a simple supported joint chain:

    fixed root -> X prismatic -> Y prismatic -> Z revolute -> mobile base

The result is intentionally a virtual-base model.  It does not use wheel-ground
contact for motion and is kept separate from the physical differential-drive
PandaOmron asset.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from isaacsim import SimulationApp

SIMULATION_APP = SimulationApp({"renderer": "RayTracedLighting", "headless": True})

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


ROOT_PRIM = "/panda_omron_virtual"
ROBOT_ROOT = f"{ROOT_PRIM}/robot0_base"
JOINT_ROOT = f"{ROBOT_ROOT}/joints"
ROOT_BODY = f"{ROBOT_ROOT}/robot0_base"
BASE_BODY = f"{ROBOT_ROOT}/mobilebase0_base"
FIXED_SUPPORT_BODY = f"{ROBOT_ROOT}/mobilebase0_fixed_support"
SUPPORT_BODY = f"{ROBOT_ROOT}/mobilebase0_support"
EEF_BODY = f"{ROBOT_ROOT}/gripper0_right_eef"
LEFT_FINGER_TIP_BODY = f"{ROBOT_ROOT}/gripper0_right_finger_joint1_tip"
RIGHT_FINGER_TIP_BODY = f"{ROBOT_ROOT}/gripper0_right_finger_joint2_tip"

IMPORTED_D6_BASE_JOINT = f"{JOINT_ROOT}/mobilebase0_base"
TORSO_HEIGHT_JOINT = f"{JOINT_ROOT}/mobilebase0_joint_torso_height"

VIRTUAL_X_BODY = f"{ROBOT_ROOT}/virtual_mobile_base_x_link"
VIRTUAL_Y_BODY = f"{ROBOT_ROOT}/virtual_mobile_base_y_link"

FORWARD_JOINT = f"{JOINT_ROOT}/mobilebase0_joint_mobile_forward"
SIDE_JOINT = f"{JOINT_ROOT}/mobilebase0_joint_mobile_side"
YAW_JOINT = f"{JOINT_ROOT}/mobilebase0_joint_mobile_yaw"

LINEAR_DRIVE_DAMPING = 1000.0
LINEAR_DRIVE_MAX_FORCE = 600.0
YAW_DRIVE_DAMPING = 1500.0
YAW_DRIVE_MAX_FORCE = 600.0
ARM_DRIVE_STIFFNESS = 10000000.0
ARM_DRIVE_DAMPING = 100000.0
ARM_DRIVE_MAX_FORCE = 1000000.0
ARM_JOINT_NAMES = tuple(f"robot0_joint{i}" for i in range(1, 8))
GRIPPER_DRIVE_STIFFNESS = 100000.0
GRIPPER_DRIVE_DAMPING = 1000.0
GRIPPER_DRIVE_MAX_FORCE = 100.0
GRIPPER_JOINT_NAMES = (
    "gripper0_right_finger_joint1",
    "gripper0_right_finger_joint2",
)


def _remove_xform_op_properties(prim: Usd.Prim) -> None:
    for prop in list(prim.GetProperties()):
        name = prop.GetName()
        if name == "xformOpOrder" or name.startswith("xformOp:"):
            prim.RemoveProperty(name)


def _set_xform_identity(prim: Usd.Prim) -> None:
    _remove_xform_op_properties(prim)
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _normalize_rigid_body_xform_ops(stage: Usd.Stage) -> list[str]:
    normalized = []
    for prim in stage.Traverse():
        if not UsdPhysics.RigidBodyAPI(prim):
            continue
        xform = UsdGeom.Xformable(prim)
        local_transform = xform.GetLocalTransformation()
        translation = local_transform.ExtractTranslation()
        rotation = local_transform.ExtractRotationQuat()

        _remove_xform_op_properties(prim)
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(translation))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(
            Gf.Quatf(
                float(rotation.GetReal()),
                Gf.Vec3f(
                    float(rotation.GetImaginary()[0]),
                    float(rotation.GetImaginary()[1]),
                    float(rotation.GetImaginary()[2]),
                ),
            )
        )
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))
        normalized.append(str(prim.GetPath()))
    return normalized


def _set_rigid_body(prim: Usd.Prim) -> None:
    UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_api = UsdPhysics.RigidBodyAPI(prim)
    rigid_api.CreateRigidBodyEnabledAttr().Set(True)
    rigid_api.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    rigid_api.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))


def _apply_mass(
    prim: Usd.Prim,
    *,
    mass: float,
    center_of_mass: tuple[float, float, float],
    diagonal_inertia: tuple[float, float, float],
) -> None:
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(float(mass))
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*center_of_mass))
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*diagonal_inertia))


def _define_virtual_body(stage: Usd.Stage, path: str) -> None:
    stage.RemovePrim(path)
    prim = stage.DefinePrim(path, "Xform")
    _set_xform_identity(prim)
    _set_rigid_body(prim)
    _apply_mass(prim, mass=0.1, center_of_mass=(0.0, 0.0, 0.0), diagonal_inertia=(0.001, 0.001, 0.001))


def _configure_drive(prim: Usd.Prim, drive_name: str, *, damping: float, max_force: float) -> None:
    drive = UsdPhysics.DriveAPI.Apply(prim, drive_name)
    drive.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)
    drive.CreateStiffnessAttr().Set(0.0)
    drive.CreateDampingAttr().Set(float(damping))
    drive.CreateMaxForceAttr().Set(float(max_force))


def _remove_drive_properties(prim: Usd.Prim, drive_name: str) -> None:
    prim.RemoveAPI(UsdPhysics.DriveAPI, drive_name)
    for prop in list(prim.GetProperties()):
        if prop.GetName().startswith(f"drive:{drive_name}:"):
            prim.RemoveProperty(prop.GetName())


def _define_prismatic_joint(
    stage: Usd.Stage,
    path: str,
    *,
    body0: str,
    body1: str,
    axis,
) -> None:
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
    _configure_drive(joint.GetPrim(), "linear", damping=LINEAR_DRIVE_DAMPING, max_force=LINEAR_DRIVE_MAX_FORCE)


def _define_revolute_joint(
    stage: Usd.Stage,
    path: str,
    *,
    body0: str,
    body1: str,
    axis,
) -> None:
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
    _configure_drive(joint.GetPrim(), "angular", damping=YAW_DRIVE_DAMPING, max_force=YAW_DRIVE_MAX_FORCE)


def _define_fixed_joint(
    stage: Usd.Stage,
    path: str,
    body0: str,
    body1: str,
    *,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float],
    local_rot0: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    local_rot1: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    stage.RemovePrim(path)
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(*local_rot0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(*local_rot1))
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateJointEnabledAttr().Set(True)


def _fix_torso_height(stage: Usd.Stage) -> None:
    _define_fixed_joint(
        stage,
        TORSO_HEIGHT_JOINT,
        FIXED_SUPPORT_BODY,
        SUPPORT_BODY,
        local_pos0=(0.05, 0.0, 0.20),
        local_pos1=(0.0, 0.0, 0.0),
        local_rot0=(0.7071068, 0.0, -0.7071068, 0.0),
        local_rot1=(0.7071068, 0.0, -0.7071068, 0.0),
    )


def _repair_imported_body_masses(stage: Usd.Stage) -> None:
    body_specs = {
        ROOT_BODY: (1.0, (0.0, 0.0, 0.0), (0.01, 0.01, 0.01)),
        BASE_BODY: (80.0, (-0.20, 0.0, 0.04), (4.20, 3.80, 5.00)),
        FIXED_SUPPORT_BODY: (1.0, (0.0, 0.0, 0.0), (0.02, 0.02, 0.01)),
        SUPPORT_BODY: (1.0, (-0.05, 0.0, -0.20), (0.02, 0.02, 0.01)),
        EEF_BODY: (0.05, (0.0, 0.0, 0.0), (0.001, 0.001, 0.001)),
        LEFT_FINGER_TIP_BODY: (0.05, (0.0, 0.0, 0.0), (0.001, 0.001, 0.001)),
        RIGHT_FINGER_TIP_BODY: (0.05, (0.0, 0.0, 0.0), (0.001, 0.001, 0.001)),
    }
    for path, (mass, com, inertia) in body_specs.items():
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            _set_rigid_body(prim)
            _apply_mass(prim, mass=mass, center_of_mass=com, diagonal_inertia=inertia)


def _configure_arm_drives(stage: Usd.Stage) -> None:
    for joint_name in ARM_JOINT_NAMES:
        prim = stage.GetPrimAtPath(f"{JOINT_ROOT}/{joint_name}")
        if not prim.IsValid():
            continue
        _remove_drive_properties(prim, "X")
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
        drive.CreateStiffnessAttr().Set(ARM_DRIVE_STIFFNESS)
        drive.CreateDampingAttr().Set(ARM_DRIVE_DAMPING)
        drive.CreateMaxForceAttr().Set(ARM_DRIVE_MAX_FORCE)
        drive.CreateTargetVelocityAttr().Set(0.0)

    for joint_name in GRIPPER_JOINT_NAMES:
        prim = stage.GetPrimAtPath(f"{JOINT_ROOT}/{joint_name}")
        if not prim.IsValid():
            continue
        _remove_drive_properties(prim, "X")
        drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
        drive.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
        drive.CreateStiffnessAttr().Set(GRIPPER_DRIVE_STIFFNESS)
        drive.CreateDampingAttr().Set(GRIPPER_DRIVE_DAMPING)
        drive.CreateMaxForceAttr().Set(GRIPPER_DRIVE_MAX_FORCE)
        drive.CreateTargetVelocityAttr().Set(0.0)


def _demote_site_visuals(stage: Usd.Stage) -> None:
    sites_root = stage.GetPrimAtPath(f"{EEF_BODY}/sites")
    if not sites_root.IsValid():
        return
    for prim in Usd.PrimRange(sites_root):
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            purpose_attr = imageable.GetPurposeAttr()
            if not purpose_attr.IsValid():
                purpose_attr = imageable.CreatePurposeAttr()
            purpose_attr.Set(UsdGeom.Tokens.guide)


def patch_stage(stage: Usd.Stage) -> dict:
    if not stage.GetPrimAtPath(ROOT_PRIM).IsValid():
        raise RuntimeError(f"Expected PandaOmron virtual root prim at {ROOT_PRIM}")
    if not stage.GetPrimAtPath(ROOT_BODY).IsValid():
        raise RuntimeError(f"Expected imported dummy root body at {ROOT_BODY}")
    if not stage.GetPrimAtPath(BASE_BODY).IsValid():
        raise RuntimeError(f"Expected imported mobile base body at {BASE_BODY}")

    for path in (ROOT_PRIM, f"{ROOT_PRIM}/worldBody", ROOT_BODY, BASE_BODY):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

    stage.RemovePrim(IMPORTED_D6_BASE_JOINT)
    _define_virtual_body(stage, VIRTUAL_X_BODY)
    _define_virtual_body(stage, VIRTUAL_Y_BODY)
    _define_prismatic_joint(
        stage,
        FORWARD_JOINT,
        body0=ROOT_BODY,
        body1=VIRTUAL_X_BODY,
        axis=UsdPhysics.Tokens.x,
    )
    _define_prismatic_joint(
        stage,
        SIDE_JOINT,
        body0=VIRTUAL_X_BODY,
        body1=VIRTUAL_Y_BODY,
        axis=UsdPhysics.Tokens.y,
    )
    _define_revolute_joint(
        stage,
        YAW_JOINT,
        body0=VIRTUAL_Y_BODY,
        body1=BASE_BODY,
        axis=UsdPhysics.Tokens.z,
    )

    _fix_torso_height(stage)
    _repair_imported_body_masses(stage)
    _configure_arm_drives(stage)
    _demote_site_visuals(stage)

    articulation_prim = stage.GetPrimAtPath(ROBOT_ROOT)
    UsdPhysics.ArticulationRootAPI.Apply(articulation_prim)
    PhysxSchema.PhysxArticulationAPI.Apply(articulation_prim)
    normalized_xform_bodies = _normalize_rigid_body_xform_ops(stage)

    return {
        "root_prim": ROOT_PRIM,
        "articulation_root": ROBOT_ROOT,
        "imported_d6_base_joint_removed": IMPORTED_D6_BASE_JOINT,
        "virtual_base_joint_chain": [
            FORWARD_JOINT,
            SIDE_JOINT,
            YAW_JOINT,
        ],
        "virtual_base_dof_names": [
            "mobilebase0_joint_mobile_forward",
            "mobilebase0_joint_mobile_side",
            "mobilebase0_joint_mobile_yaw",
        ],
        "virtual_base_links": [
            VIRTUAL_X_BODY,
            VIRTUAL_Y_BODY,
            BASE_BODY,
        ],
        "torso_height_joint_fixed": TORSO_HEIGHT_JOINT,
        "linear_drive_damping": LINEAR_DRIVE_DAMPING,
        "linear_drive_max_force": LINEAR_DRIVE_MAX_FORCE,
        "yaw_drive_damping": YAW_DRIVE_DAMPING,
        "yaw_drive_max_force": YAW_DRIVE_MAX_FORCE,
        "arm_drive_stiffness": ARM_DRIVE_STIFFNESS,
        "arm_drive_damping": ARM_DRIVE_DAMPING,
        "arm_drive_max_force": ARM_DRIVE_MAX_FORCE,
        "gripper_drive_stiffness": GRIPPER_DRIVE_STIFFNESS,
        "gripper_drive_damping": GRIPPER_DRIVE_DAMPING,
        "gripper_drive_max_force": GRIPPER_DRIVE_MAX_FORCE,
        "normalized_xform_body_count": len(normalized_xform_bodies),
        "notes": [
            "This is a separate virtual mobile-base asset, not the physical differential-drive robot.",
            "The unsupported imported D6 planar base joint is replaced by X/Y prismatic joints plus a yaw revolute joint.",
            "The virtual base is driven by articulation joint velocity targets and does not depend on wheel-ground contact.",
            "Torso height is fixed so the only mobile-base DOFs are forward, side, and yaw.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("InternDataAssets/assets/panda_omron_virtual/robot.usd"),
        help="Input USD from Isaac MJCF importer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("InternDataAssets/assets/panda_omron_virtual/robot.usd"),
        help="Patched output USD path.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("InternDataAssets/assets/panda_omron_virtual/source/virtual_postprocess_metadata.json"),
        help="Output metadata JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path != output_path:
        shutil.copy2(input_path, output_path)

    stage = Usd.Stage.Open(str(output_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {output_path}")
    metadata = patch_stage(stage)
    stage.GetRootLayer().Save()

    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)
    print(f"Wrote {args.metadata_output}", flush=True)
    print(f"Virtual base joints: {metadata['virtual_base_dof_names']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        SIMULATION_APP.close()
