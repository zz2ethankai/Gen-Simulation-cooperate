import os
import logging
import math

from core.objects.base_object import register_object
from core.utils.asset_path_utils import select_texture_path
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
        # Local rotational-symmetry axis ("x"/"y"/"z" or 3-vector).  The
        # collision-scene slip metric ignores roll about this axis because it is
        # a physically meaningless gauge DOF (e.g. a cup's free spin).
        self.attach_slip_ignore_axis = cfg.get("attach_slip_ignore_axis", None)

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
        self._sanitize_composed_physics_hierarchy()
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

    def _sanitize_composed_physics_hierarchy(self):
        """Keep a ``RigidObject`` as one rigid body with one collision source.

        Task-ready wrappers put the authoritative rigid body and CoACD shapes
        under ``/Aligned`` and ``/Aligned/Collision``.  Some source visual
        packages still carry their original RigidBody/Mass/Collision APIs
        below ``/Aligned/Geometry``.  Once composed, that creates a nested
        dynamic body: PhysX writes a compensating transform onto the visual
        child during reset while the wrapper body follows its dedicated
        collision shapes.  The collision object then remains present, but its
        rendered mesh sinks below the support surface.

        Remove descendant rigid bodies unconditionally because this class is
        explicitly a *single* rigid object.  When a dedicated Collision
        subtree exists, also strip collision APIs outside that subtree so the
        visual mesh cannot become a duplicate physics shape.  The opinions
        are authored only in the runtime stage; source USD files are not
        modified.
        """

        rigid_root = get_prim_at_path(self.rigid_prim_path)
        if not rigid_root.IsValid():
            return

        descendants = [prim for prim in Usd.PrimRange(rigid_root) if prim != rigid_root]
        dedicated_collision_roots = [
            prim
            for prim in descendants
            if prim.GetParent() == rigid_root
            and prim.GetName().lower() in {"collision", "collisions"}
        ]
        dedicated_prefixes = [f"{prim.GetPath()}/" for prim in dedicated_collision_roots]

        removed_rigid = []
        removed_visual_collision = []
        for prim in descendants:
            prim_path = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                removed_rigid.append(prim_path)
            if prim.HasAPI(UsdPhysics.MassAPI):
                prim.RemoveAPI(UsdPhysics.MassAPI)

            in_dedicated_collision = any(
                prim_path == prefix[:-1] or prim_path.startswith(prefix)
                for prefix in dedicated_prefixes
            )
            if dedicated_collision_roots and not in_dedicated_collision:
                removed = False
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)
                    removed = True
                if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
                    removed = True
                if removed:
                    removed_visual_collision.append(prim_path)

        remaining_nested = [
            str(prim.GetPath())
            for prim in descendants
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if remaining_nested:
            raise ValueError(
                f"RigidObject {self.base_prim_path} still has nested rigid bodies: "
                f"{remaining_nested}"
            )
        if removed_rigid or removed_visual_collision:
            LOGGER.warning(
                "[RigidObject] sanitized composed physics hierarchy root=%s "
                "nested_rigid=%s visual_collision=%s dedicated_collision=%s",
                self.rigid_prim_path,
                removed_rigid,
                removed_visual_collision,
                [str(prim.GetPath()) for prim in dedicated_collision_roots],
            )

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
        Isaac 6 composes a reference unit-resolution scale on divergent-unit
        assets.  Return only the source/stage ratio so that composed scale is
        canceled exactly once.
        """

        try:
            source_stage = Usd.Stage.Open(self.usd_path)
            source_meters_per_unit = UsdGeom.GetStageMetersPerUnit(source_stage)
            scene_meters_per_unit = UsdGeom.GetStageMetersPerUnit(
                get_prim_at_path(self.base_prim_path).GetStage()
            )
            if source_meters_per_unit <= 0.0 or scene_meters_per_unit <= 0.0:
                return 1.0

            # Isaac's metrics assembler has already authored the reference
            # unit-resolution scale on this referenced subtree.  Dividing by
            # that authored scale again double-counts the conversion (the
            # legacy Sort-the-Rubbish assets then reached 10000x).  The task
            # scale must only cancel the source/stage unit ratio; the
            # reference scale remains part of the composed transform.
            return float(scene_meters_per_unit) / float(source_meters_per_unit)
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
        texture_path = select_texture_path(asset_root, cfg)
        mat_prim_path = f"{self.base_prim_path}/Looks/Material"
        mat = OmniPBR(
            prim_path=mat_prim_path,
            name="Material",
            texture_path=texture_path,
            texture_scale=cfg.get("texture_scale"),
        )
        self.apply_visual_material(mat)
