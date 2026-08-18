"""Persistent RGB axis markers for comparing the robot EE frame with target
grasp frames in the Isaac Sim viewport / recorded frames.

Each queried pose gets three thin cylinders (X=red, Y=green, Z=blue) rooted at
the pose origin.  Markers are visual-only: no collision, no physics, and they
persist for the rest of the run so they also show up in recorded RGB frames.
All drawing is best-effort and must never break planning.
"""
import logging
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.usd
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)
LOGGER = logging.getLogger("simbox.debug_marker")
_AXIS_COLORS = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}

def _pq_str(T):
    p, q = pose_from_tf_matrix(T)
    return f"p=[{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}] q=[{q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {q[3]:.4f}]"


def _mat_str(T):
    return "\n".join("    [" + " ".join(f"{v:.4f}" for v in row) + "]" for row in np.asarray(T))


def get_stage():
    """Current USD stage via the Kit context — omni.isaac.core.utils.stage
    doesn't export get_stage consistently across versions."""
    return omni.usd.get_context().get_stage()
def _clear_markers(marker_root):
    stage = get_stage()
    prim = stage.GetPrimAtPath(marker_root)
    if prim.IsValid():
        stage.RemovePrim(marker_root)


def _make_material(stage, mat_path, color):
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    mat.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return mat


def _draw_axes_at(prim_path, T_world, length, radius):
    stage = get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    pos, quat = pose_from_tf_matrix(T_world)
    xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    )
    half = length / 2.0
    offsets = {"X": (half, 0.0, 0.0), "Y": (0.0, half, 0.0), "Z": (0.0, 0.0, half)}
    for axis, color in _AXIS_COLORS.items():
        cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/axis_{axis}")
        cyl.GetAxisAttr().Set(axis)
        cyl.GetHeightAttr().Set(length)
        cyl.GetRadiusAttr().Set(radius)
        UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*offsets[axis]))
        mat = _make_material(stage, f"{prim_path}/axis_{axis}_mat", color)
        UsdShade.MaterialBindingAPI(cyl).Bind(mat)


def _world_base_from_controller(controller, root_prim_path):
    base_path = getattr(controller, "robot_base_path", None)
    if base_path:
        return get_relative_transform(get_prim_at_path(base_path), get_prim_at_path(root_prim_path))
    return np.asarray(getattr(controller, "T_world_base_init", np.eye(4)), dtype=float)


def draw_grasp_debug(controller, root_prim_path, p_base_ee_cur, q_base_ee_cur, T_base_grasps, max_frames=8):
    """Draw the current EE frame (large sticks) and up to max_frames target
    grasp frames (small sticks), all in world coordinates."""
    marker_root = f"{root_prim_path}/DebugMarkers"
    _clear_markers(marker_root)
    T_world_base = _world_base_from_controller(controller, root_prim_path)
    T_world_ee = T_world_base @ tf_matrix_from_pose(p_base_ee_cur, q_base_ee_cur)
    _draw_axes_at(f"{marker_root}/ee_current", T_world_ee, length=0.15, radius=0.008)
    n = min(max_frames, len(T_base_grasps))
    for i in range(n):
        T_world_grasp = T_world_base @ T_base_grasps[i]
        _draw_axes_at(f"{marker_root}/grasp_{i:03d}", T_world_grasp, length=0.08, radius=0.004)
    
    T_base_ee = tf_matrix_from_pose(p_base_ee_cur, q_base_ee_cur)
    LOGGER.warning("[PickDebug] draw_grasp_debug root=%s n_grasps=%d", root_prim_path, n)
    LOGGER.warning("[PickDebug]   T_world_base (armbase->world):\n%s", _mat_str(T_world_base))
    LOGGER.warning("[PickDebug]   EE current base : %s", _pq_str(T_base_ee))
    LOGGER.warning("[PickDebug]   EE current world: %s", _pq_str(T_world_ee))
    for i in range(n):
        LOGGER.warning("[PickDebug]   grasp[%d] base : %s", i, _pq_str(T_base_grasps[i]))
        LOGGER.warning("[PickDebug]   grasp[%d] world: %s", i, _pq_str(T_world_base @ T_base_grasps[i]))
    return marker_root, n
