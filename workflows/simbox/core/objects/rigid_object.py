import os
import random
import logging

from core.objects.base_object import register_object
from core.utils.asset_path_utils import resolve_texture_paths
from core.utils.attach_collision_utils import join_prim_path, resolve_attach_collision_prims
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import create_prim, get_prim_at_path

from isaacsim.core.api.materials.omni_pbr import OmniPBR
from pxr import Usd, UsdGeom, UsdPhysics


LOGGER = logging.getLogger(__name__)


@register_object
class RigidObject(SingleRigidPrim):
    def __init__(self, asset_root, root_prim_path, cfg, *args, **kwargs):
        """
        Args:
            asset_root: Asset root path
            root_prim_path: Root prim path in USD stage
            cfg: Config dict with required keys:
                - name: Object name
                - path: USD file path relative to asset_root
                - prim_path_child: Child prim path for rigid body
                - attach_prim_path_children (optional): Collision prim paths used by CuRobo attach
                - attach_prim_path_child (optional): Deprecated singular collision prim path
                - init_translation (optional): Initial translation
                - init_orientation (optional): Initial orientation
                - init_parent (optional): Initial parent prim
                - gap (optional): Gap parameter
                - mass (optional): Object mass
        """
        # ===== From cfg =====
        self.asset_root = asset_root
        cfg_name = cfg["name"]
        cfg_path = cfg["path"]
        prim_path = f"{root_prim_path}/{cfg_name}"
        if os.path.isabs(cfg_path) and os.path.exists(cfg_path):
            usd_path = cfg_path
        else:
            # Legacy distractor generation stores repo-relative asset paths
            # with a leading slash.  Treat a non-existent absolute-looking
            # path as relative to asset_root instead of discarding the root.
            usd_path = os.path.join(asset_root, cfg_path.lstrip(os.sep))
        self.usd_path = os.path.abspath(usd_path)
        self.init_translation = cfg.get("init_translation", None)
        self.init_orientation = cfg.get("init_orientation", None)
        self.init_parent = cfg.get("init_parent", None)
        self.gap = cfg.get("gap", None)
        self.mass = cfg.get("mass", None)
        self._physics_collision_approximation_paths = []
        kwargs["mass"] = cfg.get("mass", None)

        # ===== Initialize =====
        create_prim(prim_path=prim_path, usd_path=self.usd_path)
        self.base_prim_path = prim_path
        self.rigid_prim_path = join_prim_path(self.base_prim_path, cfg["prim_path_child"])
        if not get_prim_at_path(self.rigid_prim_path).IsValid():
            raise ValueError(f"rigid prim does not exist for {cfg_name}: {self.rigid_prim_path}")
        self._apply_small_scale_collision_fallback(cfg)
        resolution = resolve_attach_collision_prims(
            self.base_prim_path,
            self.rigid_prim_path,
            cfg,
            get_prim_at_path,
        )
        self.attach_collision_prim_paths = resolution.prim_paths
        self.attach_collision_resolution_source = resolution.source
        self.attach_collision_failure_code = resolution.failure_code
        self.attach_collision_candidates = resolution.candidates
        self.attach_collision_failure_message = resolution.message
        # Temporary compatibility for older skills.  New Pick code consumes
        # attach_collision_prim_paths and never guesses the first USD child.
        self.mesh_prim_path = (
            self.attach_collision_prim_paths[0] if len(self.attach_collision_prim_paths) == 1 else None
        )
        super().__init__(prim_path=self.rigid_prim_path, name=cfg["name"], *args, **kwargs)

    def _apply_small_scale_collision_fallback(self, cfg):
        """Use a native bounded collision approximation for tiny legacy assets.

        Isaac Sim 4 accepted the authored ``convexDecomposition`` collision
        on several assets whose USD geometry is stored in millimetres and is
        subsequently scaled by ``1e-3``.  Isaac Sim 6 can keep the rigid body
        alive while the cooked mesh produces no contacts at that scale.  The
        official USD collision API allows overriding the approximation in the
        active layer, so keep the source USD immutable and request a native
        bounding box only for those tiny, legacy-scaled rigid objects.

        An explicit per-object approximation remains authoritative.  The
        fallback is intentionally limited to uniformly tiny scales so normal
        scene assets and non-uniformly scaled containers keep their authored
        collision geometry.
        """

        if cfg.get("collision_approximation", None) is not None:
            return

        scale = cfg.get("scale", None)
        if scale is None:
            return
        try:
            scale_values = [abs(float(value)) for value in scale]
        except (TypeError, ValueError):
            return
        if len(scale_values) != 3 or not scale_values or max(scale_values) > 0.01:
            return
        if min(scale_values) <= 0.0:
            return

        rigid_prim = get_prim_at_path(self.rigid_prim_path)
        stage = rigid_prim.GetStage()
        changed_paths = []
        for mesh_prim in Usd.PrimRange(rigid_prim):
            if mesh_prim == rigid_prim or not mesh_prim.IsA(UsdGeom.Mesh):
                continue
            if not mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
                continue

            approximation_attr = mesh_prim.GetAttribute("physics:approximation")
            if approximation_attr and approximation_attr.IsValid():
                current = approximation_attr.Get()
                if current == "boundingCube":
                    continue

            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, mesh_prim.GetPath())
            if not mesh_collision_api:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            mesh_collision_api.CreateApproximationAttr().Set("boundingCube")
            changed_paths.append(str(mesh_prim.GetPath()))

        self._physics_collision_approximation_paths = changed_paths
        if changed_paths:
            LOGGER.info(
                "[Isaac6Physics] %s tiny-scale collision fallback: "
                "approximation=boundingCube scale=%s meshes=%s",
                self.rigid_prim_path,
                scale_values,
                changed_paths,
            )

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
        texture_name = cfg["texture_lib"]
        texture_path_list = resolve_texture_paths(asset_root, texture_name)
        if cfg["apply_randomization"]:
            texture_id = random.randint(0, len(texture_path_list) - 1)
        else:
            texture_id = cfg["texture_id"]
        texture_path = texture_path_list[texture_id]
        mat_prim_path = f"{self.base_prim_path}/Looks/Material"
        mat = OmniPBR(
            prim_path=mat_prim_path,
            name="Material",
            texture_path=texture_path,
            texture_scale=cfg.get("texture_scale"),
        )
        self.apply_visual_material(mat)
