import os
import random
import logging
import math

from core.objects.base_object import register_object
from core.utils.asset_path_utils import resolve_texture_paths
from core.utils.attach_collision_utils import join_prim_path, resolve_attach_collision_prims
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import create_prim, get_prim_at_path

from isaacsim.core.api.materials.omni_pbr import OmniPBR
from pxr import Gf, Usd, UsdGeom, UsdPhysics


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
        self.asset_scale_correction = self._asset_scale_correction()
        configured_child = cfg.get("prim_path_child")
        configured_path = None
        configured_prim = None
        configured_error = None
        if configured_child:
            try:
                configured_path = join_prim_path(self.base_prim_path, configured_child)
                configured_prim = get_prim_at_path(configured_path)
                if not configured_prim.IsValid():
                    configured_error = "prim is invalid"
                elif not configured_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    configured_error = "prim has no UsdPhysics.RigidBodyAPI"
            except (TypeError, ValueError) as exc:
                configured_error = str(exc)
        else:
            configured_error = "prim_path_child is not configured"

        if configured_error is None:
            self.rigid_prim_path = configured_path
            self.rigid_prim_path_source = "configured"
        else:
            root_prim = get_prim_at_path(self.base_prim_path)
            fallback_prim = next(
                (
                    prim
                    for prim in Usd.PrimRange(root_prim)
                    if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI)
                ),
                None,
            )
            if fallback_prim is None:
                configured_display = configured_path or "<unset>"
                raise ValueError(
                    f"no rigid body prim found for {cfg_name}: "
                    f"configured={configured_display} ({configured_error}), "
                    f"root={self.base_prim_path}"
                )
            self.rigid_prim_path = str(fallback_prim.GetPath())
            self.rigid_prim_path_source = "fallback_first_rigid_body"
            LOGGER.warning(
                "[RigidObject] %s configured rigid child invalid (%s): %s; "
                "using first rigid body under %s: %s",
                cfg_name,
                configured_error,
                configured_path or "<unset>",
                self.base_prim_path,
                self.rigid_prim_path,
            )
        self._create_legacy_collision_proxy(cfg)
        self._author_initial_scale(cfg)
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
        self._apply_physics_material_overrides(cfg)

    def _apply_physics_material_overrides(self, cfg):
        """Apply an optional task-level friction coefficient to this asset.

        Pickable assets carry their physics material inside the referenced USD.
        Keep the source asset unchanged, but allow a task to use the same
        material contract as the grasp-evaluation pipeline when a legacy asset
        is too close to the friction limit for the simulator's default finger
        material.
        """

        friction = cfg.get("friction", None)
        if friction is None:
            return
        try:
            friction = float(friction)
        except (TypeError, ValueError):
            return
        if not math.isfinite(friction) or friction < 0.0:
            raise ValueError(
                f"RigidObject friction must be a finite non-negative number: {friction!r}"
            )

        root_prim = get_prim_at_path(self.base_prim_path)
        material_prims = []
        seen_paths = set()
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.MaterialAPI):
                material_prims.append(prim)
                seen_paths.add(str(prim.GetPath()))
            for relationship in prim.GetRelationships():
                if "material" not in relationship.GetName().lower():
                    continue
                for target_path in relationship.GetTargets():
                    material_prim = get_prim_at_path(str(target_path))
                    if not material_prim.IsValid() or str(material_prim.GetPath()) in seen_paths:
                        continue
                    material_prims.append(material_prim)
                    seen_paths.add(str(material_prim.GetPath()))
        for material_prim in material_prims:
            material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
            material_api.CreateStaticFrictionAttr().Set(friction)
            material_api.CreateDynamicFrictionAttr().Set(friction)
        if material_prims:
            LOGGER.info(
                "[RigidObject] %s physics friction=%s materials=%s",
                self.base_prim_path,
                friction,
                [str(prim.GetPath()) for prim in material_prims],
            )

    def _asset_scale_correction(self):
        """Undo the actual scale Isaac 6 authored on the reference parent.

        Legacy object geometry is already in scene-sized coordinates, while
        Isaac 6 can author more than one unit-resolution scale op on the
        divergent-unit reference parent.  Combine the source/stage unit
        ratio with the regular parent scale; the later ``unitsResolve`` op is
        then canceled without changing the object's pose axes.
        """

        try:
            source_stage = Usd.Stage.Open(self.usd_path)
            source_meters_per_unit = UsdGeom.GetStageMetersPerUnit(source_stage)
            scene_meters_per_unit = UsdGeom.GetStageMetersPerUnit(
                get_prim_at_path(self.base_prim_path).GetStage()
            )
            if source_meters_per_unit <= 0.0 or scene_meters_per_unit <= 0.0:
                return 1.0

            reference_prim = get_prim_at_path(self.base_prim_path)
            xformable = UsdGeom.Xformable(reference_prim)
            reference_scale = 1.0
            found_scale = False
            for op in xformable.GetOrderedXformOps():
                # The metrics assembler may author its ``unitsResolve`` op
                # on the next stage update.  Only count the regular op here;
                # the source/stage ratio accounts for that later op.
                if op.GetOpName() != "xformOp:scale":
                    continue
                if op.GetOpType() != UsdGeom.XformOp.TypeScale:
                    continue
                value = op.Get()
                components = [float(value[index]) for index in range(3)]
                if not all(abs(component - components[0]) <= 1e-6 for component in components):
                    return 1.0
                if abs(components[0]) <= 1e-12:
                    return 1.0
                reference_scale *= components[0]
                found_scale = True
            if not found_scale:
                reference_scale = 1.0
            return (
                float(scene_meters_per_unit)
                / float(source_meters_per_unit)
                / reference_scale
            )
        except Exception:
            LOGGER.debug(
                "Could not determine asset unit scale for %s",
                self.usd_path,
                exc_info=True,
            )
            return 1.0

    def _author_initial_scale(self, cfg):
        """Author the effective scale before Isaac binds the rigid view.

        Isaac 6 cooks a referenced rigid body's collision shape while
        ``SingleRigidPrim`` is initialized.  The task loader applies scale
        immediately after construction, which is too late for legacy assets
        whose reference-unit correction is not one.  Author the same local
        scale first so the initial physics cook and the later task-loader
        scale agree.
        """

        scale = cfg.get("scale", None)
        if scale is None:
            return
        try:
            scale_values = [float(value) for value in scale]
            correction = float(getattr(self, "asset_scale_correction", 1.0))
        except (TypeError, ValueError):
            return
        if len(scale_values) != 3 or not all(value > 0.0 for value in scale_values):
            return

        effective_scale = Gf.Vec3f(*(value * correction for value in scale_values))
        rigid_prim = get_prim_at_path(self.rigid_prim_path)
        if not rigid_prim.IsValid() or not rigid_prim.IsA(UsdGeom.Xformable):
            return

        xformable = UsdGeom.Xformable(rigid_prim)
        scale_ops = [
            op
            for op in xformable.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeScale
        ]
        if scale_ops:
            scale_ops[-1].Set(effective_scale)
        else:
            xformable.AddScaleOp().Set(effective_scale)

    def _create_legacy_collision_proxy(self, cfg):
        """Replace an unusable legacy mesh collider with a native bbox.

        Some legacy object USDs render correctly after the reference-unit
        correction but their authored dynamic mesh remains at the old scale
        in Isaac 6.  Use the same bounded proxy pattern as GeometryObject so
        the rigid body gets one cookable collider at its source-local size;
        the initial scale authored below then brings it into scene units.
        """

        if cfg.get("collision_approximation", None) is not None:
            return
        if abs(float(getattr(self, "asset_scale_correction", 1.0)) - 1.0) <= 1e-6:
            return

        rigid_prim = get_prim_at_path(self.rigid_prim_path)
        if not rigid_prim.IsValid() or not rigid_prim.IsA(UsdGeom.Xformable):
            return
        stage = rigid_prim.GetStage()
        bbox = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        ).ComputeUntransformedBound(rigid_prim).ComputeAlignedBox()
        min_point = bbox.GetMin()
        max_point = bbox.GetMax()
        size = tuple(float(max_point[index] - min_point[index]) for index in range(3))
        if any(value <= 0.0 for value in size):
            return
        center = tuple(float((min_point[index] + max_point[index]) * 0.5) for index in range(3))

        for mesh_prim in Usd.PrimRange(rigid_prim):
            if mesh_prim.IsA(UsdGeom.Mesh) and mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(mesh_prim).CreateCollisionEnabledAttr().Set(False)

        proxy_path = f"{self.rigid_prim_path}/collision_proxy"
        proxy_geom = UsdGeom.Cube.Define(stage, proxy_path)
        proxy_geom.CreateSizeAttr().Set(1.0)
        proxy_xform = UsdGeom.Xformable(proxy_geom.GetPrim())
        proxy_xform.AddTranslateOp().Set(center)
        proxy_xform.AddScaleOp().Set(size)
        UsdPhysics.CollisionAPI.Apply(proxy_geom.GetPrim())
        UsdGeom.Imageable(proxy_geom.GetPrim()).MakeInvisible()
        self._physics_collision_proxy_path = proxy_path
        LOGGER.info(
            "[Isaac6Physics] %s legacy bbox collision proxy=%s size=%s correction=%s",
            self.rigid_prim_path,
            proxy_path,
            size,
            self.asset_scale_correction,
        )

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
