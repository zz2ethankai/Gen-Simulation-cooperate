from omni.isaac.kit import SimulationApp

# 必须先启动 Kit。
simulation_app = SimulationApp({
    "headless": True,
})

try:
    # 必须放在 SimulationApp 创建之后。
    from pxr import Usd, UsdPhysics, PhysxSchema

    USD_PATH = (
        "/home/bld/ykqin/InternDataEngine/"
        "InternDataAssets/robots/franka/robot.usd"
    )

    print(f"Opening USD: {USD_PATH}")

    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {USD_PATH}")

    default_prim = stage.GetDefaultPrim()

    print(f"Opened USD: {USD_PATH}")
    print(
        "Default prim:",
        default_prim.GetPath()
        if default_prim.IsValid()
        else "<invalid or not set>",
    )

    found_articulation_root = False
    found_physx_articulation = False

    print("\n=== Traversing stage ===")

    for prim in stage.Traverse():
        applied_schemas = list(prim.GetAppliedSchemas())

        has_root_api = prim.HasAPI(
            UsdPhysics.ArticulationRootAPI
        )
        has_physx_api = prim.HasAPI(
            PhysxSchema.PhysxArticulationAPI
        )

        if has_root_api or has_physx_api:
            print("\nPrim:", prim.GetPath())
            print("  Type:", prim.GetTypeName())
            print("  Applied schemas:", applied_schemas)
            print(
                "  ArticulationRootAPI:",
                has_root_api,
            )
            print(
                "  PhysxArticulationAPI:",
                has_physx_api,
            )

        if has_root_api:
            found_articulation_root = True

        if has_physx_api:
            found_physx_articulation = True

            attr_name = (
                "physxArticulation:"
                "solverPositionIterationCount"
            )
            attr = prim.GetAttribute(attr_name)

            print("  solver attr object valid:", bool(attr))
            print(
                "  solver attr type:",
                attr.GetTypeName(),
            )
            print(
                "  solver attr value:",
                attr.Get(),
            )

    print("\n=== Summary ===")
    print(
        "Found ArticulationRootAPI:",
        found_articulation_root,
    )
    print(
        "Found PhysxArticulationAPI:",
        found_physx_articulation,
    )

finally:
    simulation_app.close()
