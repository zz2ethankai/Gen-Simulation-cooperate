import os

from isaacsim import SimulationApp


app = SimulationApp({"headless": True})

from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import create_prim
from pxr import Usd, UsdGeom, UsdPhysics
import omni.usd


world = World(physics_dt=1.0 / 30.0, rendering_dt=1.0 / 30.0, stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

support = UsdGeom.Cube.Define(stage, "/World/support")
support.CreateSizeAttr().Set(1.0)
support.AddTranslateOp().Set((0.0, 0.0, 0.375))
support.AddScaleOp().Set((1.0, 0.6, 0.375))
UsdPhysics.CollisionAPI.Apply(support.GetPrim())

assets = {
    "handbag": "/workspace/workflows/simbox/assets/pick_and_place/pre-train-pick/assets/google_scan-handbag/google_scan-handbag_0325/Aligned_obj.usd",
    "basket": "/workspace/workflows/simbox/assets/pick_and_place/g1_pnp_basket/basket/basket_22/Aligned_obj.usd",
}
rigids = {}
print("SETUP_BEGIN", flush=True)
for name, usd_path in assets.items():
    root = f"/World/{name}"
    create_prim(prim_path=root, usd_path=usd_path)
    body_path = f"{root}/Aligned"
    body_prim = stage.GetPrimAtPath(body_path)
    collision_mode = os.environ.get("COLLISION_MODE", "none")
    if name == "handbag" and collision_mode != "none":
        for descendant in Usd.PrimRange(body_prim):
            if descendant == body_prim:
                continue
            if descendant.IsA(UsdGeom.Mesh):
                approximation = descendant.GetAttribute("physics:approximation")
                if approximation and approximation.IsValid():
                    approximation.Set(collision_mode)
                    print("APPROXIMATION", descendant.GetPath(), collision_mode, flush=True)
    if name == "handbag" and collision_mode == "proxy":
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeUntransformedBound(body_prim).ComputeAlignedRange()
        minimum = bbox.GetMin()
        maximum = bbox.GetMax()
        center = (minimum + maximum) * 0.5
        size = maximum - minimum
        proxy = UsdGeom.Cube.Define(stage, f"{body_path}/native_collision_proxy")
        proxy.CreateSizeAttr().Set(1.0)
        proxy.AddTranslateOp().Set(center)
        proxy.AddScaleOp().Set(size)
        UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
        proxy.GetImageable().MakeInvisible()
        print("PROXY", [float(value) for value in center], [float(value) for value in size], flush=True)
    body = SingleRigidPrim(prim_path=body_path, name=name)
    body.set_local_pose(translation=(0.3 if name == "handbag" else -0.3, 0.0, 0.752), orientation=(1.0, 0.0, 0.0, 0.0))
    body.set_local_scale((0.001, 0.001, 0.001) if name == "handbag" else (0.8, 0.25, 0.8))
    rigids[name] = body
    print("SETUP_DONE", name, flush=True)

print("RESET_BEGIN", flush=True)
world.reset()
print("RESET_DONE", flush=True)
for name, body in rigids.items():
    body.set_world_pose(
        position=(0.3 if name == "handbag" else -0.3, 0.0, 0.752),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )
    body.set_linear_velocity((0.0, 0.0, 0.0))
    body.set_angular_velocity((0.0, 0.0, 0.0))
for index in range(30):
    world.step(render=False)
    if index in (0, 1, 2, 5, 10, 20, 29):
        for name, body in rigids.items():
            position, _ = body.get_world_pose()
            velocity = body.get_linear_velocity()
            print("STATE", index, name, [float(value) for value in position], [float(value) for value in velocity], flush=True)

app.close()
