import logging
import os

from core.objects.base_object import register_object
from core.utils.asset_path_utils import select_texture_path
from isaacsim.core.prims import SingleGeometryPrim as GeometryPrim
from isaacsim.core.utils.prims import (
    create_prim,
    get_prim_at_path,
    is_prim_path_valid,
)

from isaacsim.core.api.materials.omni_pbr import OmniPBR

from pxr import Gf, Usd, UsdGeom, UsdPhysics


LOGGER = logging.getLogger(__name__)


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
        self._collision_proxy_local_center = None
        self._collision_proxy_local_size = None

        # ===== Initialize =====
        create_prim(prim_path=prim_path, usd_path=usd_path)
        super().__init__(prim_path=prim_path, name=cfg["name"], *args, **kwargs)
        self._apply_configured_static_physics()
        self.collision_proxy_path = None
        self._create_collision_proxy()

    def _apply_configured_static_physics(self):
        """Honor static/kinematic flags on imported geometry bodies."""

        if not (
            bool(self.cfg.get("static", False))
            or bool(self.cfg.get("kinematic", False))
            or self.cfg.get("rigidbody", None) is False
        ):
            return

        kinematic = bool(self.cfg.get("static", False) or self.cfg.get("kinematic", False))
        rigid_body_enabled = kinematic or self.cfg.get("rigidbody", None) is not False
        rigid_bodies = [
            prim
            for prim in Usd.PrimRange(self.prim)
            if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        for prim in rigid_bodies:
            rigid_body = UsdPhysics.RigidBodyAPI(prim)
            rigid_body.CreateRigidBodyEnabledAttr().Set(rigid_body_enabled)
            rigid_body.CreateKinematicEnabledAttr().Set(kinematic and rigid_body_enabled)
            if kinematic:
                rigid_body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                rigid_body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        if rigid_bodies:
            LOGGER.info(
                "[GeometryObject] configured static physics name=%s bodies=%s kinematic=%s",
                self.name,
                [str(prim.GetPath()) for prim in rigid_bodies],
                kinematic,
            )

    def _create_collision_proxy(self):
        configured_collision = self.cfg.get("collision_enabled", None)
        configured_approximation = self.cfg.get("collision_approximation", None)
        # Older pick-and-place arenas rely on the collider authored inside
        # ``table0/instance.usd``.  Isaac Sim 4 tolerated that legacy
        # convex-decomposition mesh as the only support surface, but Isaac
        # Sim 6 can leave the referenced static mesh without a usable
        # contact shape after the non-uniform asset scale is applied.  Keep
        # explicit config authoritative and provide a bounded native USD
        # proxy only for the legacy table fixture when no policy was stated.
        legacy_table_fallback = (
            configured_collision is None
            and str(self.cfg.get("name", "")).strip().lower() == "table"
        )
        if configured_collision is not None and not bool(configured_collision):
            return
        if not legacy_table_fallback and configured_approximation is None:
            return

        if legacy_table_fallback:
            self._disable_authored_mesh_collisions()
            approximation = "bbox"
        else:
            approximation = self._normalize_collision_approximation(self.cfg.get("collision_approximation", "bbox"))

        if approximation == "none":
            return
        if approximation == "supportBodyBBox":
            self._create_bbox_collision_proxy(clip_top_world_z=self._support_body_clip_top_world_z())
            return
        if approximation == "bbox":
            self._create_bbox_collision_proxy()
            return
        if approximation == "openContainer":
            self._create_open_container_collision_proxy()
            return
        if approximation in {"convexDecomposition", "convexHull", "meshSimplification"}:
            self._apply_mesh_collision_approximation(approximation)
            return

        raise ValueError(f"Unsupported collision_approximation for {self.name}: {approximation}")

    def _disable_authored_mesh_collisions(self):
        """Disable legacy referenced mesh colliders before adding the proxy.

        This authors the override in the active task layer only; the source
        USD remains untouched.  A single native ``Cube`` collider avoids the
        expensive/fragile convex-decomposition cook while preserving the
        fixture's support footprint for pick/place.
        """

        for mesh_prim in self._iter_mesh_prims():
            if not mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collision_api = UsdPhysics.CollisionAPI(mesh_prim)
            collision_api.CreateCollisionEnabledAttr().Set(False)

    def _create_bbox_collision_proxy(self, clip_top_world_z=None):
        stage = self.prim.GetStage()
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        # The collision proxy is authored as a child of ``self.prim``.  The
        # local bound must therefore be expressed before the parent prim's
        # authored scale is applied.  ``ComputeLocalBound`` already includes
        # that transform for referenced assets in Isaac Sim 6, so applying
        # the parent's scale to the proxy would scale the collider twice.
        bbox = bbox_cache.ComputeUntransformedBound(self.prim).ComputeAlignedBox()
        min_point = bbox.GetMin()
        max_point = bbox.GetMax()
        min_x, min_y, min_z = float(min_point[0]), float(min_point[1]), float(min_point[2])
        max_x, max_y, max_z = float(max_point[0]), float(max_point[1]), float(max_point[2])
        if clip_top_world_z is not None:
            translation = self.cfg.get("translation", [0.0, 0.0, 0.0])
            local_clip_top_z = float(clip_top_world_z) - float(translation[2])
            max_z = min(max_z, local_clip_top_z)

        size = (max_x - min_x, max_y - min_y, max_z - min_z)
        center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5)
        if min(size) <= 0.0:
            raise ValueError(f"Cannot create bbox collision for {self.name}: empty local bbox")

        self._collision_proxy_local_center = center
        self._collision_proxy_local_size = size

        collision_prim_path = f"{self.prim_path}/collision_proxy"
        self.collision_proxy_path = collision_prim_path
        collision_geom = UsdGeom.Cube.Define(stage, collision_prim_path)
        collision_geom.CreateSizeAttr().Set(1.0)

        collision_prim = collision_geom.GetPrim()
        collision_xform = UsdGeom.Xformable(collision_prim)
        collision_xform.AddTranslateOp().Set(center)
        collision_xform.AddScaleOp().Set(size)

        UsdPhysics.CollisionAPI.Apply(collision_prim)

        if not bool(self.cfg.get("collision_visible", False)):
            UsdGeom.Imageable(collision_prim).MakeInvisible()

    def _create_open_container_collision_proxy(self):
        """Replace a closed container mesh with a native open-box collider."""

        self._disable_authored_mesh_collisions()

        stage = self.prim.GetStage()
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        bbox = bbox_cache.ComputeUntransformedBound(self.prim).ComputeAlignedBox()
        min_point = bbox.GetMin()
        max_point = bbox.GetMax()
        min_x, min_y, min_z = (float(min_point[index]) for index in range(3))
        max_x, max_y, max_z = (float(max_point[index]) for index in range(3))
        size_x = max_x - min_x
        size_y = max_y - min_y
        size_z = max_z - min_z
        if min(size_x, size_y, size_z) <= 0.0:
            raise ValueError(f"Cannot create open-container collision for {self.name}: empty local bbox")

        wall_thickness = float(self.cfg.get("open_container_wall_thickness", 0.02))
        wall_thickness = min(wall_thickness, size_x * 0.25, size_z * 0.25)
        if wall_thickness <= 0.0:
            raise ValueError(
                f"Cannot create open-container collision for {self.name}: invalid wall thickness"
            )

        center_x = (min_x + max_x) * 0.5
        center_y = (min_y + max_y) * 0.5
        center_z = (min_z + max_z) * 0.5
        wall_height = size_y
        inner_x = size_x - 2.0 * wall_thickness
        inner_z = size_z - 2.0 * wall_thickness
        bottom_at_max_y = str(self.cfg.get("open_container_bottom", "min_y")).strip().lower() == "max_y"
        bottom_y = max_y - wall_thickness * 0.5 if bottom_at_max_y else min_y + wall_thickness * 0.5
        boxes = (
            ("bottom", (center_x, bottom_y, center_z),
             (size_x, wall_thickness, size_z)),
            ("wall_x_min", (min_x + wall_thickness * 0.5, center_y, center_z),
             (wall_thickness, wall_height, size_z)),
            ("wall_x_max", (max_x - wall_thickness * 0.5, center_y, center_z),
             (wall_thickness, wall_height, size_z)),
            ("wall_z_min", (center_x, center_y, min_z + wall_thickness * 0.5),
             (inner_x, wall_height, wall_thickness)),
            ("wall_z_max", (center_x, center_y, max_z - wall_thickness * 0.5),
             (inner_x, wall_height, wall_thickness)),
        )

        self._collision_proxy_local_center = (center_x, center_y, center_z)
        self._collision_proxy_local_size = (size_x, size_y, size_z)
        collision_root_path = f"{self.prim_path}/open_container_collision"
        UsdGeom.Scope.Define(stage, collision_root_path)
        self.collision_proxy_path = f"{collision_root_path}/{boxes[0][0]}"

        for name, center, size in boxes:
            collision_geom = UsdGeom.Cube.Define(stage, f"{collision_root_path}/{name}")
            collision_geom.CreateSizeAttr().Set(1.0)
            collision_prim = collision_geom.GetPrim()
            collision_xform = UsdGeom.Xformable(collision_prim)
            collision_xform.AddTranslateOp().Set(center)
            collision_xform.AddScaleOp().Set(size)
            UsdPhysics.CollisionAPI.Apply(collision_prim)
            UsdGeom.Imageable(collision_prim).MakeInvisible()

    def create_native_support_collision_proxy(self, stage, collision_root_path: str, index: int):
        """Create a world-space support collider outside a referenced subtree.

        The legacy table proxy is authored below a referenced asset for
        rendering/debugging.  Isaac Sim 6 may compose that child after a
        collision collection has been expanded, so a support group must use a
        concrete native prim in an unreferenced scope.  Transform the cached
        local bounds through the table root once; this preserves the configured
        pose/scale without touching the source USD.
        """

        if self._collision_proxy_local_center is None or self._collision_proxy_local_size is None:
            LOGGER.warning(
                "Cannot create native support proxy for %s: cached local bounds are missing",
                self.name,
            )
            return None

        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        local_to_world = xform_cache.GetLocalToWorldTransform(self.prim)
        center = self._collision_proxy_local_center
        size = self._collision_proxy_local_size
        world_points = []
        for x_sign in (-0.5, 0.5):
            for y_sign in (-0.5, 0.5):
                for z_sign in (-0.5, 0.5):
                    local_point = Gf.Vec3d(
                        center[0] + x_sign * size[0],
                        center[1] + y_sign * size[1],
                        center[2] + z_sign * size[2],
                    )
                    world_points.append(local_to_world.Transform(local_point))

        min_values = [min(float(point[axis]) for point in world_points) for axis in range(3)]
        max_values = [max(float(point[axis]) for point in world_points) for axis in range(3)]
        world_size = [max_value - min_value for min_value, max_value in zip(min_values, max_values)]
        if any(value <= 1.0e-6 for value in world_size):
            LOGGER.warning(
                "Cannot create native support proxy for %s: min=%s max=%s",
                self.name,
                min_values,
                max_values,
            )
            return None

        UsdGeom.Scope.Define(stage, collision_root_path)
        proxy_path = f"{collision_root_path}/support_proxy_{index}"
        proxy_geom = UsdGeom.Cube.Define(stage, proxy_path)
        proxy_geom.CreateSizeAttr().Set(1.0)
        proxy_xform = UsdGeom.Xformable(proxy_geom.GetPrim())
        proxy_xform.AddTranslateOp().Set(
            tuple((min_value + max_value) * 0.5 for min_value, max_value in zip(min_values, max_values))
        )
        proxy_xform.AddScaleOp().Set(tuple(world_size))
        UsdPhysics.CollisionAPI.Apply(proxy_geom.GetPrim())
        UsdGeom.Imageable(proxy_geom.GetPrim()).MakeInvisible()
        LOGGER.info(
            "Created native support proxy=%s source=%s min=%s max=%s",
            proxy_path,
            self.prim_path,
            min_values,
            max_values,
        )
        return proxy_path

    def get_collision_debug_info(self):
        """Return USD-only collision diagnostics for Isaac 6 reset debugging.

        This method deliberately does not query or mutate the PhysX tensor view.
        It is used while a task is loading to distinguish an authored USD
        collision problem from a stale rigid-body view or collision-group
        problem. The caller gates logging with
        ``INTERNDATA_DEBUG_RESET_LIFECYCLE``.
        """

        stage = self.prim.GetStage()
        info = {
            "root": str(self.prim.GetPath()),
            "proxy": self.collision_proxy_path,
            "root_collision_meshes": [],
        }

        mesh_prims = list(self._iter_mesh_prims())
        info["root_collision_mesh_count"] = len(mesh_prims)
        for mesh_prim in mesh_prims[:32]:
            collision_attr = mesh_prim.GetAttribute("physics:collisionEnabled")
            approximation_attr = mesh_prim.GetAttribute("physics:approximation")
            info["root_collision_meshes"].append(
                {
                    "path": str(mesh_prim.GetPath()),
                    "has_collision_api": bool(mesh_prim.HasAPI(UsdPhysics.CollisionAPI)),
                    "collision_enabled": collision_attr.Get() if collision_attr else None,
                    "mesh_approximation": approximation_attr.Get() if approximation_attr else None,
                }
            )
        if len(mesh_prims) > 32:
            info["root_collision_meshes_truncated"] = True

        if not self.collision_proxy_path:
            return info

        proxy = stage.GetPrimAtPath(self.collision_proxy_path)
        proxy_info = {
            "valid": bool(proxy and proxy.IsValid()),
            "has_collision_api": bool(proxy and proxy.HasAPI(UsdPhysics.CollisionAPI)),
        }
        if proxy and proxy.IsValid():
            collision_attr = proxy.GetAttribute("physics:collisionEnabled")
            proxy_info["collision_enabled"] = collision_attr.Get() if collision_attr else None
            proxy_info["parent"] = str(proxy.GetParent().GetPath())
            proxy_info["type"] = str(proxy.GetTypeName())

            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
                useExtentsHint=False,
            )
            try:
                bbox = bbox_cache.ComputeWorldBound(proxy).ComputeAlignedBox()
                proxy_info["world_bbox_min"] = [float(v) for v in bbox.GetMin()]
                proxy_info["world_bbox_max"] = [float(v) for v in bbox.GetMax()]
            except Exception as exc:
                proxy_info["world_bbox_error"] = repr(exc)

        info["proxy_info"] = proxy_info
        return info

    def _support_body_clip_top_world_z(self):
        if "support_surface_z" not in self.cfg:
            raise ValueError(f"{self.name} uses supportBodyBBox but is missing support_surface_z")
        clearance = float(self.cfg.get("support_body_top_clearance", 0.0))
        if clearance < 0.0:
            raise ValueError(f"{self.name} support_body_top_clearance must be non-negative")
        return float(self.cfg["support_surface_z"]) - clearance

    @staticmethod
    def _normalize_collision_approximation(approximation):
        value = str(approximation).strip()
        aliases = {
            "none": "none",
            "no_collision": "none",
            "convex_decomposition": "convexDecomposition",
            "convexdecomposition": "convexDecomposition",
            "convex_hull": "convexHull",
            "convexhull": "convexHull",
            "mesh_simplification": "meshSimplification",
            "meshsimplification": "meshSimplification",
            "support_body_bbox": "supportBodyBBox",
            "supportbodybbox": "supportBodyBBox",
            "open_container": "openContainer",
            "opencontainer": "openContainer",
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

        texture_path = select_texture_path(asset_root, cfg)
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
