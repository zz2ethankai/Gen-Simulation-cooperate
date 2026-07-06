#!/usr/bin/env python3
"""Import an MJCF file to USD with Isaac Sim's MJCF importer."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, required=True, help="Input MJCF XML path.")
    parser.add_argument("--usd", type=Path, required=True, help="Output USD path.")
    parser.add_argument("--prim-path", default="/panda_omron", help="Root prim path in the generated USD.")
    parser.add_argument("--fix-base", action="store_true", help="Import the root body as fixed.")
    return parser.parse_args()


def _call_if_present(obj, method_name: str, value) -> None:
    method = getattr(obj, method_name, None)
    if method is not None:
        method(value)


def main() -> None:
    args = parse_args()
    mjcf_path = args.mjcf.resolve()
    usd_path = args.usd.resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(mjcf_path)
    usd_path.parent.mkdir(parents=True, exist_ok=True)

    app = SimulationApp({"renderer": "RayTracedLighting", "headless": True})
    try:
        import omni.kit.commands  # pylint: disable=import-outside-toplevel
        import omni.usd  # pylint: disable=import-outside-toplevel
        from omni.isaac.core.utils.extensions import enable_extension  # pylint: disable=import-outside-toplevel

        enable_extension("omni.importer.mjcf")
        app.update()

        status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
        if not status:
            raise RuntimeError("MJCFCreateImportConfig failed")

        _call_if_present(import_config, "set_fix_base", bool(args.fix_base))
        _call_if_present(import_config, "set_import_inertia_tensor", True)
        _call_if_present(import_config, "set_self_collision", False)
        _call_if_present(import_config, "set_make_default_prim", True)

        status, imported_path = omni.kit.commands.execute(
            "MJCFCreateAsset",
            mjcf_path=str(mjcf_path),
            import_config=import_config,
            prim_path=args.prim_path,
            dest_path=str(usd_path),
        )
        if not status:
            raise RuntimeError(f"MJCFCreateAsset failed for {mjcf_path}")

        for _ in range(20):
            app.update()

        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            stage.Save()

        print(f"Imported {mjcf_path}", flush=True)
        print(f"Root prim: {imported_path}", flush=True)
        print(f"Wrote {usd_path}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
