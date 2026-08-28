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

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


ROOT_PRIM = "/panda_omron_virtual"
ROBOT_ROOT = f"{ROOT_PRIM}/robot0_base"
JOINT_ROOT = f"{ROBOT_ROOT}/joints"
ROOT_BODY = f"{ROBOT_ROOT}/robot0_base"
BASE_BODY = f"{ROBOT_ROOT}/mobilebase0_base"
FIXED_SUPPORT_BODY = f"{ROBOT_ROOT}/mobilebase0_fixed_support"
SUPPORT_BODY = f"{ROBOT_ROOT}/mobilebase0_support"
FRANKA_SOURCE_USD = Path(__file__).resolve().parents[2] / "InternDataAssets/robots/franka/robot.usd"
FRANKA_SOURCE_ASSET_DIR = FRANKA_SOURCE_USD.parent
FRANKA_GRIPPER_ASSET_FILES = (
    "Materials/Materials.usd",
    "Props/panda_hand.usd",
    "Props/panda_leftfinger.usd",
    "Props/panda_rightfinger.usd",
)
FRANKA_SOURCE_ROOT = "/panda/fr3"
FRANKA_SOURCE_LINK7 = f"{FRANKA_SOURCE_ROOT}/panda_link7"
FRANKA_SOURCE_LINK8 = f"{FRANKA_SOURCE_ROOT}/panda_link8"
FRANKA_SOURCE_JOINT8 = f"{FRANKA_SOURCE_LINK7}/panda_joint8"
FRANKA_LINK8 = f"{ROBOT_ROOT}/panda_link8"
FRANKA_LINK8_JOINT = f"{JOINT_ROOT}/panda_joint8"
FRANKA_HAND = f"{ROBOT_ROOT}/panda_hand"
FRANKA_LEFT_FINGER = f"{ROBOT_ROOT}/panda_leftfinger"
FRANKA_RIGHT_FINGER = f"{ROBOT_ROOT}/panda_rightfinger"
FRANKA_HAND_JOINT = f"{FRANKA_LINK8}/panda_hand_joint"
STALE_FRANKA_HAND_JOINT = f"{JOINT_ROOT}/panda_hand_joint"
FRANKA_FINGER_JOINT1 = f"{FRANKA_HAND}/panda_finger_joint1"
FRANKA_FINGER_JOINT2 = f"{FRANKA_HAND}/panda_finger_joint2"
OLD_GRIPPER_PRIMS = (
    f"{ROBOT_ROOT}/robot0_right_hand",
    f"{ROBOT_ROOT}/gripper0_right_right_gripper",
    f"{ROBOT_ROOT}/gripper0_right_eef",
    f"{ROBOT_ROOT}/gripper0_right_leftfinger",
    f"{ROBOT_ROOT}/gripper0_right_rightfinger",
    f"{ROBOT_ROOT}/gripper0_right_finger_joint1_tip",
    f"{ROBOT_ROOT}/gripper0_right_finger_joint2_tip",
    f"{JOINT_ROOT}/robot0_right_hand",
    f"{JOINT_ROOT}/gripper0_right_right_gripper",
    f"{JOINT_ROOT}/gripper0_right_eef",
    f"{JOINT_ROOT}/gripper0_right_finger_joint1",
    f"{JOINT_ROOT}/gripper0_right_finger_joint2",
    f"{JOINT_ROOT}/gripper0_right_finger_joint1_tip",
    f"{JOINT_ROOT}/gripper0_right_finger_joint2_tip",
)

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
ARM_DRIVE_MAX_FORCE = 600000000.0
ARM_JOINT_NAMES = tuple(f"robot0_joint{i}" for i in range(1, 8))
GRIPPER_DRIVE_STIFFNESS = 10000.0
GRIPPER_DRIVE_DAMPING = 1000.0
GRIPPER_DRIVE_MAX_FORCE = 600000000.0
GRIPPER_STATIC_FRICTION = 10.0
GRIPPER_DYNAMIC_FRICTION = 10.0
ACTIVE_GRIPPER_JOINT_PATHS = (FRANKA_FINGER_JOINT1,)
PASSIVE_GRIPPER_JOINT_PATHS = (FRANKA_FINGER_JOINT2,)
GRIPPER_COLLISION_PRIMS = (
    f"{FRANKA_LEFT_FINGER}/geometry/panda_leftfinger",
    f"{FRANKA_RIGHT_FINGER}/geometry/panda_rightfinger",
)
DISABLED_VIRTUAL_BASE_CONTACT_COLLISION_ROOTS = (
    f"{FIXED_SUPPORT_BODY}/collisions",
    f"{SUPPORT_BODY}/collisions",
    f"{ROBOT_ROOT}/mobilebase0_wheeled_base/collisions/mobilebase0_pedestal_feet_col",
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


def _set_xform_from_transform(
    prim: Usd.Prim,
    *,
    translation: Gf.Vec3d,
    rotation: Gf.Quatd,
) -> None:
    _remove_xform_op_properties(prim)
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Vec3f(float(translation[0]), float(translation[1]), float(translation[2]))
    )
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


def _bind_physics_material(stage: Usd.Stage, prim: Usd.Prim, material: UsdShade.Material) -> None:
    del stage
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )


def _define_gripper_material(stage: Usd.Stage) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, f"{ROBOT_ROOT}/Looks/panda_omron_gripper_physics_material")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(GRIPPER_STATIC_FRICTION)
    physics_material.CreateDynamicFrictionAttr().Set(GRIPPER_DYNAMIC_FRICTION)
    physics_material.CreateRestitutionAttr().Set(0.0)
    return material


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
    }
    for path, (mass, com, inertia) in body_specs.items():
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            _set_rigid_body(prim)
            _apply_mass(prim, mass=mass, center_of_mass=com, diagonal_inertia=inertia)


def _clear_linear_drive(prim: Usd.Prim) -> None:
    for drive_name in ("X", "linear"):
        _remove_drive_properties(prim, drive_name)
    for prop in list(prim.GetProperties()):
        name = prop.GetName()
        if (
            name.startswith("drive:linear:physics:")
            or name.startswith("physics:drive:linear:")
            or name.startswith("physxMimicJoint:")
        ):
            prim.RemoveProperty(name)


def _copy_prim_spec(source_stage: Usd.Stage, source_path: str, stage: Usd.Stage, target_path: str) -> None:
    stage.RemovePrim(target_path)
    if not Sdf.CopySpec(source_stage.GetRootLayer(), Sdf.Path(source_path), stage.GetRootLayer(), Sdf.Path(target_path)):
        raise RuntimeError(f"Failed to copy USD prim {source_path} -> {target_path}")


def _offset_copied_xform(stage: Usd.Stage, source_stage: Usd.Stage, source_path: str, target_path: str, offset: Gf.Vec3d) -> None:
    source_prim = source_stage.GetPrimAtPath(source_path)
    target_prim = stage.GetPrimAtPath(target_path)
    if not source_prim.IsValid() or not target_prim.IsValid():
        raise RuntimeError(f"Cannot offset copied gripper prim {source_path} -> {target_path}")
    transform = UsdGeom.Xformable(source_prim).GetLocalTransformation()
    _set_xform_from_transform(
        target_prim,
        translation=transform.ExtractTranslation() + offset,
        rotation=transform.ExtractRotationQuat(),
    )


def _set_joint_bodies(stage: Usd.Stage, joint_path: str, body0: str, body1: str) -> None:
    prim = stage.GetPrimAtPath(joint_path)
    if not prim.IsValid():
        raise RuntimeError(f"Expected copied Franka gripper joint at {joint_path}")
    joint = UsdPhysics.Joint(prim)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])


def _replace_with_franka_gripper(stage: Usd.Stage) -> None:
    source_stage = Usd.Stage.Open(str(FRANKA_SOURCE_USD))
    if source_stage is None:
        raise RuntimeError(f"Failed to open Franka source USD: {FRANKA_SOURCE_USD}")

    target_link7 = f"{ROBOT_ROOT}/robot0_link7"
    source_link7 = source_stage.GetPrimAtPath(FRANKA_SOURCE_LINK7)
    source_link8 = source_stage.GetPrimAtPath(FRANKA_SOURCE_LINK8)
    target_link = stage.GetPrimAtPath(target_link7)
    if not source_link7.IsValid() or not source_link8.IsValid() or not target_link.IsValid():
        raise RuntimeError(f"Cannot align Franka gripper: source={FRANKA_SOURCE_LINK7}/{FRANKA_SOURCE_LINK8}, target={target_link7}")
    source_link7_t = UsdGeom.Xformable(source_link7).GetLocalTransformation().ExtractTranslation()
    target_link_t = UsdGeom.Xformable(target_link).GetLocalTransformation().ExtractTranslation()
    offset = target_link_t - source_link7_t

    for path in OLD_GRIPPER_PRIMS + (
        FRANKA_LINK8,
        FRANKA_LINK8_JOINT,
        STALE_FRANKA_HAND_JOINT,
        FRANKA_HAND,
        FRANKA_LEFT_FINGER,
        FRANKA_RIGHT_FINGER,
    ):
        stage.RemovePrim(path)

    _copy_prim_spec(source_stage, FRANKA_SOURCE_LINK8, stage, FRANKA_LINK8)
    _copy_prim_spec(source_stage, FRANKA_SOURCE_JOINT8, stage, FRANKA_LINK8_JOINT)
    _copy_prim_spec(source_stage, f"{FRANKA_SOURCE_ROOT}/panda_leftfinger", stage, FRANKA_LEFT_FINGER)
    _copy_prim_spec(source_stage, f"{FRANKA_SOURCE_ROOT}/panda_rightfinger", stage, FRANKA_RIGHT_FINGER)
    _copy_prim_spec(source_stage, f"{FRANKA_SOURCE_ROOT}/panda_hand", stage, FRANKA_HAND)

    _offset_copied_xform(stage, source_stage, FRANKA_SOURCE_LINK8, FRANKA_LINK8, offset)
    _offset_copied_xform(stage, source_stage, f"{FRANKA_SOURCE_ROOT}/panda_leftfinger", FRANKA_LEFT_FINGER, offset)
    _offset_copied_xform(stage, source_stage, f"{FRANKA_SOURCE_ROOT}/panda_rightfinger", FRANKA_RIGHT_FINGER, offset)
    _offset_copied_xform(stage, source_stage, f"{FRANKA_SOURCE_ROOT}/panda_hand", FRANKA_HAND, offset)

    _set_joint_bodies(stage, FRANKA_LINK8_JOINT, target_link7, FRANKA_LINK8)
    _set_joint_bodies(stage, FRANKA_HAND_JOINT, FRANKA_LINK8, FRANKA_HAND)
    _set_joint_bodies(stage, FRANKA_FINGER_JOINT1, FRANKA_HAND, FRANKA_LEFT_FINGER)
    _set_joint_bodies(stage, FRANKA_FINGER_JOINT2, FRANKA_HAND, FRANKA_RIGHT_FINGER)

    mimic_joint = stage.GetPrimAtPath(FRANKA_FINGER_JOINT2)
    mimic_joint.CreateRelationship("physxMimicJoint:rotX:referenceJoint").SetTargets([Sdf.Path(FRANKA_FINGER_JOINT1)])
    mimic_joint.CreateAttribute("physxMimicJoint:rotX:gearing", Sdf.ValueTypeNames.Float).Set(-1.0)


def _configure_franka_compatible_gripper(stage: Usd.Stage) -> None:
    _replace_with_franka_gripper(stage)

    for joint_path in ACTIVE_GRIPPER_JOINT_PATHS:
        prim = stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            continue
        _clear_linear_drive(prim)
        drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
        drive.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
        drive.CreateTargetPositionAttr().Set(0.04)
        drive.CreateStiffnessAttr().Set(GRIPPER_DRIVE_STIFFNESS)
        drive.CreateDampingAttr().Set(GRIPPER_DRIVE_DAMPING)
        drive.CreateMaxForceAttr().Set(GRIPPER_DRIVE_MAX_FORCE)
        drive.CreateTargetVelocityAttr().Set(0.0)

    for joint_path in PASSIVE_GRIPPER_JOINT_PATHS:
        prim = stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            continue
        _clear_linear_drive(prim)
        drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
        drive.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(0.0)
        drive.CreateMaxForceAttr().Set(float("inf"))
        drive.CreateTargetPositionAttr().Set(0.0)
        drive.CreateTargetVelocityAttr().Set(0.0)
        prim.CreateRelationship("physxMimicJoint:rotX:referenceJoint").SetTargets([Sdf.Path(FRANKA_FINGER_JOINT1)])
        prim.CreateAttribute("physxMimicJoint:rotX:gearing", Sdf.ValueTypeNames.Float).Set(-1.0)


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

    _configure_franka_compatible_gripper(stage)


def _bind_gripper_physics_material(stage: Usd.Stage) -> None:
    material = UsdShade.Material.Get(stage, f"{ROBOT_ROOT}/Looks/panda_omron_gripper_physics_material")
    if not material:
        material = _define_gripper_material(stage)
    for path in GRIPPER_COLLISION_PRIMS:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            _bind_physics_material(stage, prim, material)


def _disable_collision_tree(stage: Usd.Stage, root_path: str) -> list[str]:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []
    disabled = []
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(False)
            disabled.append(str(prim.GetPath()))
    return disabled


def _remove_invalid_source_material_bindings(stage: Usd.Stage) -> None:
    for root_path in (FRANKA_HAND, FRANKA_LEFT_FINGER, FRANKA_RIGHT_FINGER):
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            continue
        for prim in Usd.PrimRange(root):
            for rel in list(prim.GetRelationships()):
                if "material" not in rel.GetName().lower():
                    continue
                targets = rel.GetTargets()
                if any(str(target).startswith("/panda/fr3/") for target in targets):
                    prim.RemoveProperty(rel.GetName())


def _demote_site_visuals(stage: Usd.Stage) -> None:
    return


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
    _define_gripper_material(stage)
    _bind_gripper_physics_material(stage)
    disabled_virtual_base_contact_collisions = []
    for root_path in DISABLED_VIRTUAL_BASE_CONTACT_COLLISION_ROOTS:
        disabled_virtual_base_contact_collisions.extend(_disable_collision_tree(stage, root_path))
    _remove_invalid_source_material_bindings(stage)
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
        "gripper_source_usd": str(FRANKA_SOURCE_USD),
        "gripper_copied_asset_files": list(FRANKA_GRIPPER_ASSET_FILES),
        "gripper_hand_path": FRANKA_HAND,
        "gripper_left_finger_path": FRANKA_LEFT_FINGER,
        "gripper_right_finger_path": FRANKA_RIGHT_FINGER,
        "gripper_active_joint_paths": list(ACTIVE_GRIPPER_JOINT_PATHS),
        "gripper_passive_joint_paths": list(PASSIVE_GRIPPER_JOINT_PATHS),
        "removed_imported_gripper_prims": list(OLD_GRIPPER_PRIMS),
        "gripper_static_friction": GRIPPER_STATIC_FRICTION,
        "gripper_dynamic_friction": GRIPPER_DYNAMIC_FRICTION,
        "disabled_virtual_base_contact_collisions": disabled_virtual_base_contact_collisions,
        "normalized_xform_body_count": len(normalized_xform_bodies),
        "notes": [
            "This is a separate virtual mobile-base asset, not the physical differential-drive robot.",
            "The unsupported imported D6 planar base joint is replaced by X/Y prismatic joints plus a yaw revolute joint.",
            "The virtual base is driven by articulation joint velocity targets and does not depend on wheel-ground contact.",
            "Imported Omron support and pedestal contact collisions are disabled in the virtual model so scene contacts do not block the virtual base joints.",
            "Torso height is fixed so the only mobile-base DOFs are forward, side, and yaw.",
        ],
    }


def _copy_franka_gripper_asset_files(output_path: Path) -> None:
    target_asset_dir = output_path.parent
    for relative_path in FRANKA_GRIPPER_ASSET_FILES:
        source_path = FRANKA_SOURCE_ASSET_DIR / relative_path
        target_path = target_asset_dir / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("InternDataAssets/robots/panda_omron_virtual/robot.usd"),
        help="Input USD from Isaac MJCF importer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("InternDataAssets/robots/panda_omron_virtual/robot.usd"),
        help="Patched output USD path.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("InternDataAssets/robots/panda_omron_virtual/source/virtual_postprocess_metadata.json"),
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
    _copy_franka_gripper_asset_files(output_path)

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
