#!/usr/bin/env python3
"""Standalone Docker/WebRTC viewer for existing Interndata scene YAMLs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any

from isaacsim import SimulationApp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open an existing task YAML in Isaac Sim WebRTC.")
    parser.add_argument("--task", required=True, help="Path to download task.yaml or simbox_task.yaml.")
    parser.add_argument("--include-robot", action="store_true", help="Also load robots from compatible task config.")
    parser.add_argument("--width", type=int, default=int(os.environ.get("WEBRTC_VIEWER_WIDTH", "1280")))
    parser.add_argument("--height", type=int, default=int(os.environ.get("WEBRTC_VIEWER_HEIGHT", "720")))
    parser.add_argument("--renderer", default=os.environ.get("WEBRTC_VIEWER_RENDERER", "RayTracedLighting"))
    parser.add_argument("--headless", action="store_true", default=True)
    args, kit_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *kit_args]
    return args


ARGS = _parse_args()
DEFAULT_EXPERIENCES = (
    "/isaac-sim/apps/omni.isaac.sim.headless.webrtc.kit",
    "/isaac-sim/apps/omni.isaac.sim.python.gym.livestream.kit",
    "/isaac-sim/apps/omni.isaac.sim.python.gym.headless.kit",
)
EXPERIENCE = os.environ.get("WEBRTC_VIEWER_EXPERIENCE", "")
LAUNCH_CONFIG = {
    "headless": bool(ARGS.headless),
    "width": int(ARGS.width),
    "height": int(ARGS.height),
    "renderer": ARGS.renderer,
}
if not EXPERIENCE:
    EXPERIENCE = next((item for item in DEFAULT_EXPERIENCES if Path(item).is_file()), "")
if EXPERIENCE and Path(EXPERIENCE).is_file():
    SIMULATION_APP = SimulationApp(LAUNCH_CONFIG, experience=EXPERIENCE)
else:
    SIMULATION_APP = SimulationApp(LAUNCH_CONFIG)


import carb  # noqa: E402  pylint: disable=wrong-import-position
import omni.usd  # noqa: E402  pylint: disable=wrong-import-position
from omni.isaac.core import World  # noqa: E402  pylint: disable=wrong-import-position
from omni.isaac.core.utils.extensions import enable_extension  # noqa: E402  pylint: disable=wrong-import-position
from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: E402  pylint: disable=wrong-import-position
from omni.isaac.core.utils.viewports import set_camera_view  # noqa: E402  pylint: disable=wrong-import-position
from pxr import Gf, Sdf, UsdGeom, UsdLux  # noqa: E402  pylint: disable=wrong-import-position

from webrtc.config_compat import load_scene_config  # noqa: E402  pylint: disable=wrong-import-position


REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWPORT_WINDOW = None


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "asset"


def _repo_path(path: str | Path, base: Path | None = None) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if base is not None:
        candidate = base / path_obj
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / path_obj).resolve()


def _vec3(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        return default
    return (float(value[0]), float(value[1]), float(value[2]))


def _scale3(value: Any) -> tuple[float, float, float]:
    if isinstance(value, list) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return (1.0, 1.0, 1.0)


def _apply_xform(prim, translation, euler_deg, scale=(1.0, 1.0, 1.0)) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*euler_deg))
    xform.AddScaleOp().Set(Gf.Vec3f(*scale))


def _define_plane(stage, root_path: str, cfg: dict[str, Any]) -> str:
    prim_path = f"{root_path}/{_safe_name(cfg.get('name', 'plane'))}"
    size = cfg.get("size") if isinstance(cfg.get("size"), list) else [1.0, 1.0]
    plane = UsdGeom.Plane.Define(stage, prim_path)
    plane.CreateWidthAttr().Set(float(size[0]))
    plane.CreateLengthAttr().Set(float(size[1]))
    translation = _vec3(cfg.get("translation"), (0.0, 0.0, 0.0))
    euler = _vec3(cfg.get("euler") or cfg.get("rotation"), (0.0, 0.0, 0.0))
    _apply_xform(plane.GetPrim(), translation, euler)
    return prim_path


def _load_reference(stage, root_path: str, cfg: dict[str, Any], asset_root: Path) -> str | None:
    usd_path = cfg.get("path") or cfg.get("usd_path")
    if not isinstance(usd_path, str) or not usd_path:
        return None
    abs_usd = _repo_path(usd_path, asset_root)
    if not abs_usd.exists():
        carb.log_warn(f"[webrtc_viewer] missing USD for {cfg.get('name')}: {abs_usd}")
        return None

    prim_path = f"{root_path}/{_safe_name(cfg.get('name', abs_usd.stem))}"
    add_reference_to_stage(str(abs_usd), prim_path)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        carb.log_warn(f"[webrtc_viewer] failed to create prim {prim_path}")
        return None
    translation = _vec3(cfg.get("translation"), (0.0, 0.0, 0.0))
    euler = _vec3(cfg.get("euler") or cfg.get("rotation"), (0.0, 0.0, 0.0))
    _apply_xform(prim, translation, euler, _scale3(cfg.get("scale")))
    return prim_path


def _load_items(stage, root_path: str, items: list[dict[str, Any]], asset_root: Path) -> list[str]:
    loaded = []
    for cfg in items:
        if not isinstance(cfg, dict):
            continue
        target_class = cfg.get("target_class")
        if target_class == "PlaneObject":
            loaded.append(_define_plane(stage, root_path, cfg))
            continue
        prim_path = _load_reference(stage, root_path, cfg, asset_root)
        if prim_path:
            loaded.append(prim_path)
    return loaded


def _bounds_from_payload(scene_cfg) -> tuple[float, float, float, float, float, float]:
    points = []
    for cfg in list(scene_cfg.arena.get("fixtures", [])) + list(scene_cfg.task.get("objects", [])):
        if not isinstance(cfg, dict):
            continue
        translation = _vec3(cfg.get("translation"), (0.0, 0.0, 0.0))
        size = cfg.get("size") if isinstance(cfg.get("size"), list) else [0.5, 0.5, 0.5]
        sx = float(size[0]) if len(size) > 0 else 0.5
        sy = float(size[1]) if len(size) > 1 else 0.5
        sz = float(size[2]) if len(size) > 2 else 0.5
        points.append((translation[0] - sx * 0.5, translation[1] - sy * 0.5, translation[2] - sz * 0.5))
        points.append((translation[0] + sx * 0.5, translation[1] + sy * 0.5, translation[2] + sz * 0.5))
    if not points:
        return (-2.0, -2.0, 0.0, 2.0, 2.0, 2.0)
    return (
        min(p[0] for p in points),
        min(p[1] for p in points),
        min(p[2] for p in points),
        max(p[0] for p in points),
        max(p[1] for p in points),
        max(p[2] for p in points),
    )


def _setup_lighting(stage) -> None:
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Lights/DomeLight"))
    dome.CreateIntensityAttr(600.0)
    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Lights/KeyLight"))
    distant.CreateIntensityAttr(3000.0)
    distant.CreateAngleAttr(0.6)
    _apply_xform(distant.GetPrim(), (0.0, 0.0, 3.0), (-45.0, 0.0, 35.0))


def _ensure_viewport():
    global VIEWPORT_WINDOW  # pylint: disable=global-statement

    from omni.kit.viewport.utility import create_viewport_window, get_active_viewport, get_active_viewport_window

    viewport_api = get_active_viewport()
    if viewport_api is None:
        VIEWPORT_WINDOW = get_active_viewport_window() or create_viewport_window(
            "Viewport",
            width=int(ARGS.width),
            height=int(ARGS.height),
            camera_path=Sdf.Path("/OmniverseKit_Persp"),
        )
        viewport_api = VIEWPORT_WINDOW.viewport_api if VIEWPORT_WINDOW else None

    if viewport_api is None:
        carb.log_warn("[webrtc_viewer] failed to create an active viewport")
        return None

    try:
        viewport_api.set_texture_resolution((int(ARGS.width), int(ARGS.height)))
    except Exception as exc:  # pragma: no cover - depends on viewport backend
        carb.log_warn(f"[webrtc_viewer] could not set viewport resolution: {exc}")
    viewport_api.camera_path = Sdf.Path("/OmniverseKit_Persp")
    render_product_path = ""
    for _ in range(30):
        SIMULATION_APP.update()
        try:
            render_product_path = viewport_api.get_render_product_path()
        except Exception as exc:  # pragma: no cover - depends on viewport backend
            carb.log_warn(f"[webrtc_viewer] could not read viewport render product path: {exc}")
            break
        if render_product_path:
            break
    print(
        f"[webrtc_viewer] viewport ready camera={viewport_api.camera_path} "
        f"render_product={render_product_path}",
        flush=True,
    )
    return viewport_api


def _setup_camera(bounds: tuple[float, float, float, float, float, float]) -> None:
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    center = [0.5 * (min_x + max_x), 0.5 * (min_y + max_y), 0.5 * (min_z + max_z)]
    span = max(max_x - min_x, max_y - min_y, 2.0)
    eye = [center[0] + 0.65 * span, center[1] - 1.15 * span, max(center[2] + 0.65 * span, 1.8)]
    target = [center[0], center[1], max(center[2], 0.8)]
    viewport_api = _ensure_viewport()
    if viewport_api is not None:
        set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp", viewport_api=viewport_api)


def _enable_livestream() -> None:
    for extension in (
        "omni.kit.viewport.window",
        "omni.kit.viewport.utility",
        "omni.services.streamclient.webrtc",
        "omni.kit.livestream.webrtc",
        "omni.kit.livestream.core",
        "omni.kit.streamsdk.plugins",
    ):
        try:
            enable_extension(extension)
        except Exception as exc:  # pragma: no cover - extension availability differs by kit build
            carb.log_warn(f"[webrtc_viewer] could not enable {extension}: {exc}")
    settings = carb.settings.get_settings()
    settings.set_bool("/app/window/drawMouse", True)
    settings.set_bool("/app/livestream/enabled", True)
    settings.set_string("/app/livestream/proto", "websocket")
    settings.set_int("/app/livestream/port", int(os.environ.get("WEBRTC_RTC_PORT", "49100")))
    settings.set_int(
        "/exts/omni.services.transport.server.http/port",
        int(os.environ.get("WEBRTC_HTTP_PORT", "8211")),
    )


def main() -> int:
    print(f"[webrtc_viewer] starting with task={ARGS.task}", flush=True)
    _enable_livestream()
    world = World(physics_dt=1.0 / 30.0, rendering_dt=1.0 / 30.0, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    print("[webrtc_viewer] loading compatible scene config", flush=True)
    scene_cfg = load_scene_config(ARGS.task, include_robot=ARGS.include_robot)

    _setup_lighting(stage)
    fixtures = scene_cfg.arena.get("fixtures", [])
    objects = scene_cfg.task.get("objects", [])
    robots = scene_cfg.task.get("robots", []) if ARGS.include_robot else []
    loaded = []
    loaded.extend(_load_items(stage, "/World/Arena", fixtures if isinstance(fixtures, list) else [], scene_cfg.asset_root))
    loaded.extend(_load_items(stage, "/World/Objects", objects if isinstance(objects, list) else [], scene_cfg.asset_root))
    loaded.extend(_load_items(stage, "/World/Robots", robots if isinstance(robots, list) else [], scene_cfg.asset_root))
    _setup_camera(_bounds_from_payload(scene_cfg))

    carb.log_info(
        f"[webrtc_viewer] loaded {len(loaded)} prims from {scene_cfg.task_path}; "
        "open the Isaac Sim WebRTC client and use viewport navigation to move freely."
    )
    print(f"[webrtc_viewer] loaded {len(loaded)} prims from {scene_cfg.task_path}", flush=True)
    for _ in range(3):
        SIMULATION_APP.update()
    print("[webrtc_viewer] entering render loop", flush=True)
    while SIMULATION_APP.is_running():
        world.step(render=True)
        time.sleep(0.001)
    return 0


try:
    raise SystemExit(main())
finally:
    SIMULATION_APP.close()
