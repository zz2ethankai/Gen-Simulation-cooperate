#!/usr/bin/env python3
"""Patch the RoboCasa PandaOmron USD for Isaac wheel-speed control.

The RoboSuite/RoboCasa Omron mobile base is exported with virtual planar
translation/yaw joints.  SimBox navigation must drive physical wheel joints
instead, so this script converts the imported USD into an Isaac-specific asset:

* remove the world fixed joint on the imported root body;
* replace the virtual planar base joint and torso-height prismatic joint with
  fixed joints;
* add two real differential-drive wheel joints plus low-friction passive ball
  supports.  The base is still controlled only through real wheel angular
  velocity targets; the original planar translation/yaw joints are removed.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from isaacsim import SimulationApp

SIMULATION_APP = SimulationApp({"renderer": "RayTracedLighting", "headless": True})

from omni.physx.scripts import physicsUtils  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402


ROOT_PRIM = "/panda_omron"
ROBOT_ROOT = f"{ROOT_PRIM}/robot0_base"
JOINT_ROOT = f"{ROBOT_ROOT}/joints"
ROOT_BODY = f"{ROBOT_ROOT}/robot0_base"
BASE_BODY = f"{ROBOT_ROOT}/mobilebase0_base"
ARTICULATION_ROOT = BASE_BODY
FIXED_SUPPORT_BODY = f"{ROBOT_ROOT}/mobilebase0_fixed_support"
SUPPORT_BODY = f"{ROBOT_ROOT}/mobilebase0_support"
WHEELED_BASE_BODY = f"{ROBOT_ROOT}/mobilebase0_wheeled_base"
MANIPULATOR_MOUNT_BODY = f"{ROBOT_ROOT}/manipulator_mount"
FRANKA_SOURCE_USD = Path(__file__).resolve().parents[2] / "InternDataAssets/assets/franka/robot.usd"
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

ROOT_FIXED_JOINT = f"{JOINT_ROOT}/rootJoint_robot0_base"
VIRTUAL_BASE_JOINT = f"{JOINT_ROOT}/mobilebase0_base"
TORSO_HEIGHT_JOINT = f"{JOINT_ROOT}/mobilebase0_joint_torso_height"
EXISTING_BASE_COLLISION = f"{WHEELED_BASE_BODY}/collisions/mobilebase0_pedestal_feet_col"
IMPORTED_SUPPORT_COLLISIONS = (
    f"{FIXED_SUPPORT_BODY}/collisions/mobilebase0_g0_col",
    f"{FIXED_SUPPORT_BODY}/collisions/mobilebase0_fixed_support",
    f"{SUPPORT_BODY}/collisions/mobilebase0_g1_col",
    f"{SUPPORT_BODY}/collisions/mobilebase0_support",
)

WHEEL_RADIUS = 0.085
WHEEL_WIDTH = 0.045
TRACK_WIDTH = 0.56
SUPPORT_BASE = 0.64
FRONT_SUPPORT_X = 0.12
REAR_SUPPORT_X = FRONT_SUPPORT_X - SUPPORT_BASE
DRIVE_WHEEL_X_OFFSET = -0.20
WHEEL_Z = WHEEL_RADIUS
WHEEL_STATIC_FRICTION = 0.95
WHEEL_DYNAMIC_FRICTION = 0.75
WHEEL_DRIVE_DAMPING = 600.0
WHEEL_DRIVE_MAX_FORCE = 300000.0
PASSIVE_SUPPORT_RADIUS = 0.025
PASSIVE_SUPPORT_CLEARANCE = 0.01
PASSIVE_SUPPORT_MASS = 0.25
PASSIVE_SUPPORT_STATIC_FRICTION = 0.02
PASSIVE_SUPPORT_DYNAMIC_FRICTION = 0.01
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

WHEELS = (
    ("left_wheel_joint", "left_wheel_link", DRIVE_WHEEL_X_OFFSET, 0.5 * TRACK_WIDTH),
    ("right_wheel_joint", "right_wheel_link", DRIVE_WHEEL_X_OFFSET, -0.5 * TRACK_WIDTH),
)
STALE_WHEEL_MODULES = (
    ("left_front_steering_joint", "left_front_steering_link", "left_front_wheel_joint", "left_front_wheel_link"),
    ("right_front_steering_joint", "right_front_steering_link", "right_front_wheel_joint", "right_front_wheel_link"),
    ("left_rear_steering_joint", "left_rear_steering_link", "left_rear_wheel_joint", "left_rear_wheel_link"),
    ("right_rear_steering_joint", "right_rear_steering_link", "right_rear_wheel_joint", "right_rear_wheel_link"),
)
PASSIVE_SUPPORTS = (
    ("front_left_support", FRONT_SUPPORT_X, 0.40 * TRACK_WIDTH),
    ("front_right_support", FRONT_SUPPORT_X, -0.40 * TRACK_WIDTH),
    ("rear_left_support", REAR_SUPPORT_X, 0.40 * TRACK_WIDTH),
    ("rear_right_support", REAR_SUPPORT_X, -0.40 * TRACK_WIDTH),
)


def _set_xform_translate(prim: Usd.Prim, xyz: tuple[float, float, float]) -> None:
    _remove_xform_op_properties(prim)
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*xyz))


def _set_rigid_body_xform_identity(prim: Usd.Prim, xyz: tuple[float, float, float]) -> None:
    _remove_xform_op_properties(prim)
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*xyz))
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


def _remove_xform_op_properties(prim: Usd.Prim) -> None:
    for prop in list(prim.GetProperties()):
        name = prop.GetName()
        if name == "xformOpOrder" or name.startswith("xformOp:"):
            prim.RemoveProperty(name)


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


def _apply_mass(
    prim: Usd.Prim,
    *,
    mass: float,
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0),
    diagonal_inertia: tuple[float, float, float] = (0.01, 0.01, 0.01),
) -> None:
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(float(mass))
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*center_of_mass))
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*diagonal_inertia))
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def _set_rigid_body(prim: Usd.Prim, *, enabled: bool = True) -> None:
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_api.CreateRigidBodyEnabledAttr().Set(bool(enabled))
    rigid_api.CreateKinematicEnabledAttr().Set(False)
    rigid_api.CreateStartsAsleepAttr().Set(False)
    rigid_api.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    rigid_api.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))


def _define_fixed_joint(
    stage: Usd.Stage,
    path: str,
    body0: str,
    body1: str,
    *,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float] = (0.0, 0.0, 0.0),
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


def _define_wheel_material(stage: Usd.Stage) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, f"{ROBOT_ROOT}/Looks/panda_omron_wheel_physics_material")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(WHEEL_STATIC_FRICTION)
    physics_material.CreateDynamicFrictionAttr().Set(WHEEL_DYNAMIC_FRICTION)
    physics_material.CreateRestitutionAttr().Set(0.0)
    return material


def _define_visual_material(stage: Usd.Stage) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, f"{ROBOT_ROOT}/Looks/panda_omron_wheel_visual_material")
    shader = UsdShade.Shader.Define(stage, f"{ROBOT_ROOT}/Looks/panda_omron_wheel_visual_material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.02, 0.025))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _define_support_material(stage: Usd.Stage) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, f"{ROBOT_ROOT}/Looks/panda_omron_passive_support_material")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(PASSIVE_SUPPORT_STATIC_FRICTION)
    physics_material.CreateDynamicFrictionAttr().Set(PASSIVE_SUPPORT_DYNAMIC_FRICTION)
    physics_material.CreateRestitutionAttr().Set(0.0)
    return material


def _define_gripper_material(stage: Usd.Stage) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, f"{ROBOT_ROOT}/Looks/panda_omron_gripper_physics_material")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(GRIPPER_STATIC_FRICTION)
    physics_material.CreateDynamicFrictionAttr().Set(GRIPPER_DYNAMIC_FRICTION)
    physics_material.CreateRestitutionAttr().Set(0.0)
    return material


def _bind_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _bind_physics_material(stage: Usd.Stage, prim: Usd.Prim, material: UsdShade.Material) -> None:
    del stage
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )


def _remove_drive_properties(prim: Usd.Prim, drive_name: str) -> None:
    prim.RemoveAPI(UsdPhysics.DriveAPI, drive_name)
    for prop in list(prim.GetProperties()):
        if prop.GetName().startswith(f"drive:{drive_name}:"):
            prim.RemoveProperty(prop.GetName())


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
    source_xform = UsdGeom.Xformable(source_prim)
    transform = source_xform.GetLocalTransformation()
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
    del stage


def _define_wheel(stage: Usd.Stage, *, joint_name: str, link_name: str, x: float, y: float) -> None:
    link_path = f"{ROBOT_ROOT}/{link_name}"
    joint_path = f"{JOINT_ROOT}/{joint_name}"
    stage.RemovePrim(link_path)
    stage.RemovePrim(joint_path)

    link = stage.DefinePrim(link_path, "Xform")
    _set_rigid_body_xform_identity(link, (x, y, WHEEL_Z))
    _set_rigid_body(link)
    _apply_mass(
        link,
        mass=1.2,
        center_of_mass=(0.0, 0.0, 0.0),
        diagonal_inertia=(0.006, 0.003, 0.006),
    )

    collision_path = f"{link_path}/collisions/{link_name}_collision"
    collision = UsdGeom.Cylinder.Define(stage, collision_path)
    collision.CreateAxisAttr().Set(UsdGeom.Tokens.y)
    collision.CreateRadiusAttr().Set(WHEEL_RADIUS)
    collision.CreateHeightAttr().Set(WHEEL_WIDTH)
    collision.CreateExtentAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(-WHEEL_RADIUS, -0.5 * WHEEL_WIDTH, -WHEEL_RADIUS),
                Gf.Vec3f(WHEEL_RADIUS, 0.5 * WHEEL_WIDTH, WHEEL_RADIUS),
            ]
        )
    )
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr().Set(True)

    visual_path = f"{link_path}/visuals/{link_name}_visual"
    visual = UsdGeom.Cylinder.Define(stage, visual_path)
    visual.CreateAxisAttr().Set(UsdGeom.Tokens.y)
    visual.CreateRadiusAttr().Set(WHEEL_RADIUS)
    visual.CreateHeightAttr().Set(WHEEL_WIDTH)
    visual.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.02, 0.02, 0.025)]))

    wheel_material = UsdShade.Material.Get(stage, f"{ROBOT_ROOT}/Looks/panda_omron_wheel_physics_material")
    visual_material = UsdShade.Material.Get(stage, f"{ROBOT_ROOT}/Looks/panda_omron_wheel_visual_material")
    if wheel_material:
        _bind_physics_material(stage, collision.GetPrim(), wheel_material)
    if visual_material:
        _bind_material(visual.GetPrim(), visual_material)

    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(BASE_BODY)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(link_path)])
    joint.CreateAxisAttr().Set(UsdPhysics.Tokens.y)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(x, y, WHEEL_Z))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateJointEnabledAttr().Set(True)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)
    drive.CreateStiffnessAttr().Set(0.0)
    drive.CreateDampingAttr().Set(WHEEL_DRIVE_DAMPING)
    drive.CreateMaxForceAttr().Set(WHEEL_DRIVE_MAX_FORCE)


def _define_rolling_passive_support(stage: Usd.Stage, *, name: str, x: float, y: float) -> None:
    legacy_support_path = f"{BASE_BODY}/collisions/{name}"
    link_path = f"{ROBOT_ROOT}/{name}_link"
    joint_path = f"{JOINT_ROOT}/{name}_joint"
    collision_path = f"{link_path}/collisions/{name}_collision"
    stage.RemovePrim(legacy_support_path)
    stage.RemovePrim(link_path)
    stage.RemovePrim(joint_path)

    link = stage.DefinePrim(link_path, "Xform")
    _set_rigid_body_xform_identity(
        link,
        (x, y, PASSIVE_SUPPORT_RADIUS + PASSIVE_SUPPORT_CLEARANCE),
    )
    _set_rigid_body(link)
    _apply_mass(
        link,
        mass=PASSIVE_SUPPORT_MASS,
        center_of_mass=(0.0, 0.0, 0.0),
        diagonal_inertia=(0.0001, 0.0001, 0.0001),
    )

    sphere = UsdGeom.Sphere.Define(stage, collision_path)
    sphere.CreateRadiusAttr().Set(PASSIVE_SUPPORT_RADIUS)
    sphere.CreateExtentAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(-PASSIVE_SUPPORT_RADIUS, -PASSIVE_SUPPORT_RADIUS, -PASSIVE_SUPPORT_RADIUS),
                Gf.Vec3f(PASSIVE_SUPPORT_RADIUS, PASSIVE_SUPPORT_RADIUS, PASSIVE_SUPPORT_RADIUS),
            ]
        )
    )
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    material = UsdShade.Material.Get(stage, f"{ROBOT_ROOT}/Looks/panda_omron_passive_support_material")
    if material:
        _bind_physics_material(stage, sphere.GetPrim(), material)

    joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(BASE_BODY)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(link_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(x, y, PASSIVE_SUPPORT_RADIUS + PASSIVE_SUPPORT_CLEARANCE))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateJointEnabledAttr().Set(True)


def _remove_obsolete_wheels_and_supports(stage: Usd.Stage) -> None:
    stage.RemovePrim(ROOT_BODY)
    stage.RemovePrim(VIRTUAL_BASE_JOINT)
    for joint_name, link_name, _x, _y in WHEELS:
        stage.RemovePrim(f"{JOINT_ROOT}/{joint_name}")
        stage.RemovePrim(f"{ROBOT_ROOT}/{link_name}")
    for steering_joint_name, steering_link_name, wheel_joint_name, wheel_link_name in STALE_WHEEL_MODULES:
        stage.RemovePrim(f"{JOINT_ROOT}/{steering_joint_name}")
        stage.RemovePrim(f"{ROBOT_ROOT}/{steering_link_name}")
        stage.RemovePrim(f"{JOINT_ROOT}/{wheel_joint_name}")
        stage.RemovePrim(f"{ROBOT_ROOT}/{wheel_link_name}")
    for name, _x, _y in PASSIVE_SUPPORTS:
        stage.RemovePrim(f"{BASE_BODY}/collisions/{name}")
        stage.RemovePrim(f"{JOINT_ROOT}/{name}_joint")
        stage.RemovePrim(f"{ROBOT_ROOT}/{name}_link")


def _repair_imported_body_masses(stage: Usd.Stage) -> None:
    body_specs = {
        BASE_BODY: (80.0, (-0.20, 0.0, 0.04), (4.20, 3.80, 5.00)),
        FIXED_SUPPORT_BODY: (1.0, (0.0, 0.0, 0.0), (0.02, 0.02, 0.01)),
        SUPPORT_BODY: (1.0, (-0.05, 0.0, -0.20), (0.02, 0.02, 0.01)),
        WHEELED_BASE_BODY: (35.0, (-0.20, 0.0, 0.04), (2.20, 1.80, 2.60)),
        MANIPULATOR_MOUNT_BODY: (1.0, (0.0, 0.0, 0.0), (0.01, 0.01, 0.01)),
    }
    for path, (mass, com, inertia) in body_specs.items():
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            _set_rigid_body(prim)
            _apply_mass(prim, mass=mass, center_of_mass=com, diagonal_inertia=inertia)


def _disable_grounding_chassis_collision(stage: Usd.Stage) -> None:
    prim = stage.GetPrimAtPath(EXISTING_BASE_COLLISION)
    if prim.IsValid():
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(False)
    for collision_path in IMPORTED_SUPPORT_COLLISIONS:
        prim = stage.GetPrimAtPath(collision_path)
        if prim.IsValid():
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(False)

    chassis_path = f"{BASE_BODY}/collisions/panda_omron_chassis_collision"
    chassis = UsdGeom.Cube.Define(stage, chassis_path)
    _set_xform_translate(chassis.GetPrim(), (-0.20, 0.0, 0.25))
    chassis.CreateSizeAttr().Set(1.0)
    chassis.AddScaleOp().Set(Gf.Vec3f(0.70, 0.50, 0.24))
    UsdPhysics.CollisionAPI.Apply(chassis.GetPrim()).CreateCollisionEnabledAttr().Set(True)


def patch_stage(stage: Usd.Stage) -> dict:
    if not stage.GetPrimAtPath(ROOT_PRIM).IsValid():
        raise RuntimeError(f"Expected PandaOmron root prim at {ROOT_PRIM}")
    if not stage.GetPrimAtPath(BASE_BODY).IsValid():
        raise RuntimeError(f"Expected imported mobile base body at {BASE_BODY}")

    # Remove the fixed-to-world joint so the articulation is free-floating.
    if stage.GetPrimAtPath(ROOT_FIXED_JOINT).IsValid():
        stage.RemovePrim(ROOT_FIXED_JOINT)

    # Remove the imported virtual planar base joint entirely.  The physical
    # articulation root is the real mobile-base rigid body; wheel joints are
    # the only controllable base DOFs.
    stage.RemovePrim(VIRTUAL_BASE_JOINT)
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

    for articulation_prim_path in (ROOT_PRIM, f"{ROOT_PRIM}/worldBody", ROBOT_ROOT, ROOT_BODY):
        articulation_prim = stage.GetPrimAtPath(articulation_prim_path)
        if articulation_prim.IsValid():
            articulation_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            articulation_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

    _remove_obsolete_wheels_and_supports(stage)
    _repair_imported_body_masses(stage)
    _configure_arm_drives(stage)
    _demote_site_visuals(stage)
    _define_wheel_material(stage)
    _define_visual_material(stage)
    _define_support_material(stage)
    _define_gripper_material(stage)
    _bind_gripper_physics_material(stage)
    _remove_invalid_source_material_bindings(stage)
    _disable_grounding_chassis_collision(stage)
    for joint_name, link_name, x, y in WHEELS:
        _define_wheel(stage, joint_name=joint_name, link_name=link_name, x=x, y=y)
    for name, x, y in PASSIVE_SUPPORTS:
        _define_rolling_passive_support(stage, name=name, x=x, y=y)

    articulation_prim = stage.GetPrimAtPath(ARTICULATION_ROOT)
    if not articulation_prim.IsValid():
        raise RuntimeError(f"Expected PandaOmron articulation root body at {ARTICULATION_ROOT}")
    _set_rigid_body(articulation_prim)
    UsdPhysics.ArticulationRootAPI.Apply(articulation_prim)
    PhysxSchema.PhysxArticulationAPI.Apply(articulation_prim)
    normalized_xform_bodies = _normalize_rigid_body_xform_ops(stage)

    metadata = {
        "root_prim": ROOT_PRIM,
        "articulation_root": ARTICULATION_ROOT,
        "virtual_base_joint_removed": VIRTUAL_BASE_JOINT,
        "dummy_root_body_removed": ROOT_BODY,
        "root_world_fixed_joint_removed": ROOT_FIXED_JOINT,
        "torso_height_joint_fixed": TORSO_HEIGHT_JOINT,
        "wheel_radius": WHEEL_RADIUS,
        "wheel_width": WHEEL_WIDTH,
        "track_width": TRACK_WIDTH,
        "support_base": SUPPORT_BASE,
        "front_support_x": FRONT_SUPPORT_X,
        "rear_support_x": REAR_SUPPORT_X,
        "drive_wheel_x_offset": DRIVE_WHEEL_X_OFFSET,
        "wheel_static_friction": WHEEL_STATIC_FRICTION,
        "wheel_dynamic_friction": WHEEL_DYNAMIC_FRICTION,
        "passive_support_radius": PASSIVE_SUPPORT_RADIUS,
        "passive_support_clearance": PASSIVE_SUPPORT_CLEARANCE,
        "passive_support_mass": PASSIVE_SUPPORT_MASS,
        "passive_support_static_friction": PASSIVE_SUPPORT_STATIC_FRICTION,
        "passive_support_dynamic_friction": PASSIVE_SUPPORT_DYNAMIC_FRICTION,
        "wheel_drive_damping": WHEEL_DRIVE_DAMPING,
        "wheel_drive_max_force": WHEEL_DRIVE_MAX_FORCE,
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
        "wheel_joints": [joint_name for joint_name, _, _, _ in WHEELS],
        "wheel_links": [link_name for _, link_name, _, _ in WHEELS],
        "passive_supports": [name for name, _, _ in PASSIVE_SUPPORTS],
        "passive_support_type": "rolling_spherical_joint",
        "passive_support_joints": [f"{name}_joint" for name, _, _ in PASSIVE_SUPPORTS],
        "passive_support_links": [f"{name}_link" for name, _, _ in PASSIVE_SUPPORTS],
        "disabled_imported_support_collisions": list(IMPORTED_SUPPORT_COLLISIONS),
        "normalized_xform_body_count": len(normalized_xform_bodies),
        "notes": [
            "The generated wheel joints are real revolute joints driven by angular velocity targets.",
            "The original RoboSuite planar transX/transY/rotZ joint is removed; it is not a controllable DOF.",
            "The chassis ground collision is raised above the floor so propulsion and yaw come from the two real differential drive wheels.",
            "Imported fixed-support visual collisions are disabled; four passive rolling ball supports use real unactuated spherical joints for stable differential-drive contact.",
            "The Panda arm joints have position-drive stiffness so navigation waits do not leave the arm free under gravity.",
            "Imported near-massless fixed-frame bodies are assigned explicit positive mass/inertia for stable free-base articulation solving.",
            "Imported site visuals are marked as guide purpose so they do not pollute runtime bounding boxes.",
            "The articulation root is authored on the real mobile-base rigid body so Isaac Robot binds a physical root link.",
            "Rigid-body xform ops are normalized to translate/orient/scale so PhysX can publish simulated body poses.",
            "The mobile base mass and center of mass are authored as a heavy low-slung real base, avoiding virtual stabilization while keeping wheel-speed drive.",
        ],
    }
    return metadata


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
        default=Path("InternDataAssets/assets/panda_omron/robot.usd"),
        help="Input USD from Isaac MJCF importer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("InternDataAssets/assets/panda_omron/robot.usd"),
        help="Patched output USD path.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("InternDataAssets/assets/panda_omron/source/postprocess_metadata.json"),
        help="Metadata JSON path describing the wheel patch.",
    )
    return parser.parse_args()


def main() -> None:
    try:
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
        if not stage.GetRootLayer().Save():
            raise RuntimeError(f"Failed to save USD stage: {output_path}")

        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Patched {output_path}")
        print(f"Wrote {args.metadata_output}")
        print(f"Wheel joints: {metadata['wheel_joints']}")
    finally:
        SIMULATION_APP.close()


if __name__ == "__main__":
    main()
