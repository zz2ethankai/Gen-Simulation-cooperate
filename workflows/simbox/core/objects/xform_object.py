import os

from core.objects.base_object import register_object
from core.utils.asset_path_utils import select_texture_path
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.prims import create_prim, is_prim_path_valid

from isaacsim.core.api.materials.omni_pbr import OmniPBR


@register_object
class XFormObject(XFormPrim):
    def __init__(self, asset_root, root_prim_path, cfg, *args, **kwargs):
        """
        Args:
            asset_root: Asset root path
            root_prim_path: Root prim path in USD stage
            cfg: Config dict with required keys:
                - name: Object name
                - path: USD file path relative to asset_root
        """
        # ===== From cfg =====
        self.asset_root = asset_root
        prim_path = os.path.join(root_prim_path, cfg["name"])
        usd_path = os.path.join(asset_root, cfg["path"])
        self.cfg = cfg

        # ===== Initialize =====
        create_prim(prim_path=prim_path, usd_path=usd_path)
        super().__init__(prim_path=prim_path, name=cfg["name"], *args, **kwargs)

    def get_observations(self):
        translation, orientation = self.get_local_pose()
        obs = {
            "translation": translation,
            "orientation": orientation,
        }
        return obs

    def apply_texture(self, asset_root, cfg):
        texture_path = select_texture_path(asset_root, cfg)
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
