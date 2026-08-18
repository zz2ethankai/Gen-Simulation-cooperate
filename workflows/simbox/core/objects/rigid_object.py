import glob
import os
import random

from core.objects.base_object import register_object
from core.utils.attach_collision_utils import join_prim_path, resolve_attach_collision_prims
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path

try:
    from omni.isaac.core.materials.omni_pbr import OmniPBR  # Isaac Sim 4.1.0 / 4.2.0
except ImportError:
    from isaacsim.core.api.materials import OmniPBR  # Isaac Sim 4.5.0


@register_object
class RigidObject(RigidPrim):
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
        kwargs["mass"] = cfg.get("mass", None)
        # Local rotational-symmetry axis ("x"/"y"/"z" or 3-vector).  The
        # collision-scene slip metric ignores roll about this axis because it is
        # a physically meaningless gauge DOF (e.g. a cup's free spin).
        self.attach_slip_ignore_axis = cfg.get("attach_slip_ignore_axis", None)

        # ===== Initialize =====
        create_prim(prim_path=prim_path, usd_path=self.usd_path)
        self.base_prim_path = prim_path
        self.rigid_prim_path = join_prim_path(self.base_prim_path, cfg["prim_path_child"])
        if not get_prim_at_path(self.rigid_prim_path).IsValid():
            raise ValueError(f"rigid prim does not exist for {cfg_name}: {self.rigid_prim_path}")
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
        texture_path_list = glob.glob(os.path.join(asset_root, texture_name, "*"))
        texture_path_list.sort()
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
