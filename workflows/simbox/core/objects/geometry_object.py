import glob
import os
import random

from core.objects.base_object import register_object
from omni.isaac.core.prims import GeometryPrim
from omni.isaac.core.utils.prims import (
    create_prim,
    get_prim_at_path,
    is_prim_path_valid,
)

try:
    from omni.isaac.core.materials.omni_pbr import OmniPBR  # Isaac Sim 4.1.0 / 4.2.0
except ImportError:
    from isaacsim.core.api.materials import OmniPBR  # Isaac Sim 4.5.0

from pxr import Usd, UsdGeom, UsdPhysics


@register_object
class GeometryObject(GeometryPrim):
    def __init__(self, asset_root, root_prim_path, cfg, *args, **kwargs):
        """
        Args:
            asset_root: Asset root path
            root_prim_path: Root prim path in USD stage
            cfg: Config dict with required keys:
                - name: Object name
                - path: USD file path relative to asset_root
                - prim_path_child (optional): Child prim path suffix
        """
        # ===== From cfg =====
        self.asset_root = asset_root
        prim_path = os.path.join(root_prim_path, cfg["name"])
        usd_path = os.path.join(asset_root, cfg["path"])
        if cfg.get("prim_path_child", None):
            prim_path = os.path.join(prim_path, cfg["prim_path_child"])
        self.cfg = cfg

        # ===== Initialize =====
        create_prim(prim_path=prim_path, usd_path=usd_path)
        super().__init__(prim_path=prim_path, name=cfg["name"], *args, **kwargs)
        self._create_collision_proxy()

    def _create_collision_proxy(self):
        if not bool(self.cfg.get("collision_enabled", False)):
            return

        approximation = self._normalize_collision_approximation(self.cfg.get("collision_approximation", "bbox"))
        if approximation != "bbox":
            self._apply_mesh_collision_approximation(approximation)
            return

        stage = self.prim.GetStage()
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        bbox = bbox_cache.ComputeLocalBound(self.prim).ComputeAlignedBox()
        size = bbox.GetSize()
        center = bbox.GetMidpoint()
        if min(size) <= 0.0:
            raise ValueError(f"Cannot create bbox collision for {self.name}: empty local bbox")

        collision_prim_path = f"{self.prim_path}/collision_proxy"
        collision_geom = UsdGeom.Cube.Define(stage, collision_prim_path)
        collision_geom.CreateSizeAttr().Set(1.0)

        collision_prim = collision_geom.GetPrim()
        collision_xform = UsdGeom.Xformable(collision_prim)
        collision_xform.AddTranslateOp().Set(center)
        collision_xform.AddScaleOp().Set(size)

        UsdPhysics.CollisionAPI.Apply(collision_prim)

        if not bool(self.cfg.get("collision_visible", False)):
            UsdGeom.Imageable(collision_prim).MakeInvisible()

    @staticmethod
    def _normalize_collision_approximation(approximation):
        value = str(approximation).strip()
        aliases = {
            "convex_decomposition": "convexDecomposition",
            "convexdecomposition": "convexDecomposition",
            "convex_hull": "convexHull",
            "convexhull": "convexHull",
            "mesh_simplification": "meshSimplification",
            "meshsimplification": "meshSimplification",
        }
        return aliases.get(value.replace("-", "_").lower(), value)

    def _iter_mesh_prims(self):
        def _walk(prim):
            if prim.IsA(UsdGeom.Mesh):
                yield prim
            for child in prim.GetChildren():
                yield from _walk(child)

        yield from _walk(self.prim)

    def _apply_mesh_collision_approximation(self, approximation):
        mesh_prims = list(self._iter_mesh_prims())
        if not mesh_prims:
            raise ValueError(f"Cannot create mesh collision for {self.name}: no mesh prims found")

        stage = self.prim.GetStage()
        for mesh_prim in mesh_prims:
            if not mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(mesh_prim)
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, mesh_prim.GetPath())
            if not mesh_collision_api:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            mesh_collision_api.CreateApproximationAttr().Set(approximation)

    def get_observations(self):
        translation, orientation = self.get_local_pose()
        scale = self.get_local_scale()
        obs = {
            "translation": translation,
            "orientation": orientation,
            "scale": scale,
        }
        return obs

    def apply_texture(self, asset_root, cfg):
        def _recursive_apply(prim, mat_prim_path):
            for child in prim.GetChildren():
                rel = child.GetRelationship("material:binding")
                if rel:
                    rel.SetTargets([mat_prim_path])
                    continue
                _recursive_apply(child, mat_prim_path)

        texture_name = cfg["texture_lib"]
        texture_path_list = glob.glob(os.path.join(asset_root, texture_name, "*"))
        texture_path_list.sort()
        if cfg["apply_randomization"]:
            texture_id = random.randint(0, len(texture_path_list) - 1)
        else:
            texture_id = cfg["texture_id"]
        texture_path = texture_path_list[texture_id]
        mat_prim_path = f"{self.prim_path}/Looks/Material"
        if not is_prim_path_valid(mat_prim_path):
            self.mat = OmniPBR(
                prim_path=mat_prim_path,
                name="Material",
                texture_path=texture_path,
                texture_scale=cfg.get("texture_scale"),
            )
            target_prim_path = self.prim_path
            if cfg.get("target_prim_path"):
                target_prim_path = os.path.join(self.prim_path, cfg["target_prim_path"])
                target_prim = get_prim_at_path(target_prim_path)
            else:
                target_prim = get_prim_at_path(target_prim_path)

            _recursive_apply(target_prim, mat_prim_path)
        else:
            self.mat.set_texture(
                texture_path,
            )
