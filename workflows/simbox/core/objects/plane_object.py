import glob
import os
import random

from core.objects.base_object import register_object
from core.utils.asset_path_utils import resolve_texture_paths
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import get_current_stage

from isaacsim.core.api.materials.omni_pbr import OmniPBR

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt


@register_object
class PlaneObject(XFormPrim):
    def __init__(self, asset_root, root_prim_path, cfg, *args, **kwargs):
        """
        Args:
            asset_root: Asset root path
            root_prim_path: Root prim path in USD stage
            cfg: Config dict with required keys:
                - name: Object name
                - size: [width, length] of the plane
        """
        # ===== From cfg =====
        self.asset_root = asset_root
        prim_path = os.path.join(root_prim_path, cfg["name"])
        self.cfg = cfg

        # ===== Initialize =====
        stage = get_current_stage()
        # Use an explicit quad mesh instead of UsdGeom.Plane.  Isaac Sim 6's
        # RTX renderer does not reliably draw runtime-authored Plane schemas
        # after their material is rebound.  The old Plane was also missing an
        # explicit st primvar, so Isaac 6 could leave the floor/walls without
        # their visible texture even though their collision child existed.
        width = float(cfg["size"][0])
        length = float(cfg["size"][1])
        plane_geom = UsdGeom.Mesh.Define(stage, prim_path)
        plane_geom.CreatePointsAttr().Set(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(-0.5 * width, -0.5 * length, 0.0),
                    Gf.Vec3f(0.5 * width, -0.5 * length, 0.0),
                    Gf.Vec3f(0.5 * width, 0.5 * length, 0.0),
                    Gf.Vec3f(-0.5 * width, 0.5 * length, 0.0),
                ]
            )
        )
        plane_geom.CreateFaceVertexCountsAttr().Set(Vt.IntArray([4]))
        plane_geom.CreateFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
        plane_geom.CreateNormalsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
        )
        plane_geom.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        plane_geom.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        plane_geom.CreateDoubleSidedAttr().Set(True)
        super().__init__(prim_path=prim_path, name=cfg["name"], *args, **kwargs)

        self._create_collision_volume(stage, cfg)

    def _create_collision_volume(self, stage, cfg):
        if not bool(cfg.get("collision_enabled", False)):
            return

        thickness = float(cfg.get("collision_thickness", 0.02))
        if thickness <= 0.0:
            raise ValueError("collision_thickness must be positive when collision_enabled is true")

        collision_prim_path = f"{self.prim_path}/collision_volume"
        collision_geom = UsdGeom.Cube.Define(stage, collision_prim_path)
        collision_geom.CreateSizeAttr().Set(1.0)

        collision_prim = collision_geom.GetPrim()
        collision_xform = UsdGeom.Xformable(collision_prim)
        collision_xform.AddScaleOp().Set((float(cfg["size"][0]), float(cfg["size"][1]), thickness))
        collision_xform.AddTranslateOp().Set((0.0, 0.0, -0.5 * thickness))

        UsdPhysics.CollisionAPI.Apply(collision_prim)

        if not bool(cfg.get("collision_visible", False)):
            UsdGeom.Imageable(collision_prim).MakeInvisible()

    def get_observations(self):
        raise NotImplementedError

    def apply_texture(self, asset_root, cfg):
        texture_name = cfg["texture_lib"]
        texture_path_list = resolve_texture_paths(asset_root, texture_name)
        if cfg["apply_randomization"]:
            texture_id = random.randint(0, len(texture_path_list) - 1)
        else:
            texture_id = cfg["texture_id"]
        texture_path = texture_path_list[texture_id]

        # UsdGeom.Plane supplied implicit texture coordinates in the older
        # renderer.  Author them explicitly for the Mesh so OmniPBR has a
        # stable UV source on Isaac Sim 6 and on every retry/reset.
        st = UsdGeom.PrimvarsAPI(self.prim).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        st.Set(
            Vt.Vec2fArray(
                [
                    Gf.Vec2f(0.0, 0.0),
                    Gf.Vec2f(1.0, 0.0),
                    Gf.Vec2f(1.0, 1.0),
                    Gf.Vec2f(0.0, 1.0),
                ]
            )
        )
        mat_prim_path = f"{self.prim_path}/Looks/Material"
        if not is_prim_path_valid(mat_prim_path):
            self.mat = OmniPBR(
                prim_path=mat_prim_path,
                name="Material",
                texture_path=texture_path,
                texture_scale=cfg.get("texture_scale"),
            )
            self.apply_visual_material(self.mat)
        else:
            self.mat.set_texture(
                texture_path,
            )
