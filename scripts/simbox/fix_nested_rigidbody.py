#!/usr/bin/env python3
"""Remove nested rigid-body APIs from a SimBox pickable USD asset.

The expected structure is one rigid body on /World/Aligned.  Geometry below
that prim, including /World/Aligned/Geometry/Canonical, should keep its
collision APIs but must not define another rigid body or mass.
"""

import argparse
from pathlib import Path

from pxr import Usd, UsdPhysics


def fix_asset(input_path: Path, output_path: Path) -> None:
    stage = Usd.Stage.Open(str(input_path))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {input_path}")

    aligned = stage.GetPrimAtPath("/World/Aligned")
    canonical = stage.GetPrimAtPath("/World/Aligned/Geometry/Canonical")
    if not aligned.IsValid():
        raise RuntimeError("missing /World/Aligned")
    if not canonical.IsValid():
        raise RuntimeError("missing /World/Aligned/Geometry/Canonical")

    # The wrapper/body remains the only physical rigid body.
    if not aligned.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(aligned)
        print("added RigidBodyAPI: /World/Aligned")

    # Remove only dynamic-body APIs from the nested geometry prim. Collision,
    # mesh-collision, and material APIs are intentionally left untouched.
    if canonical.HasAPI(UsdPhysics.RigidBodyAPI):
        canonical.RemoveAPI(UsdPhysics.RigidBodyAPI)
        print("removed RigidBodyAPI: /World/Aligned/Geometry/Canonical")
    if canonical.HasAPI(UsdPhysics.MassAPI):
        canonical.RemoveAPI(UsdPhysics.MassAPI)
        print("removed MassAPI: /World/Aligned/Geometry/Canonical")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Flatten so the removal is materialized even when Canonical came from a
    # referenced or sublayered source asset.
    stage.Flatten().Export(str(output_path))
    print(f"saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="source Aligned_obj.usd")
    parser.add_argument("output", type=Path, help="fixed output Aligned_obj.usd")
    args = parser.parse_args()
    fix_asset(args.input, args.output)


if __name__ == "__main__":
    main()
