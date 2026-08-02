"""Regression coverage for plural Pick attach collision paths."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
from typing import List, Tuple
import unittest

from pxr import Usd
import yaml


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.utils.attach_collision_utils import resolve_attach_collision_prims  # noqa: E402


PICK_PATH = SIMBOX_ROOT / "core" / "skills" / "pick.py"
CONTROLLER_PATH = SIMBOX_ROOT / "core" / "controllers" / "template_controller.py"
TASK_PATH = (
    ROOT
    / "InternDataAssets"
    / "assets"
    / "custom"
    / "scene_8"
    / "01_kitchen"
    / "assets"
    / "basic"
    / "kitchen_apple_orange_to_tray"
    / "simbox_task.yaml"
)


def _class_method(path: Path, class_name: str, method_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _load_method(path: Path, class_name: str, method_name: str):
    method_node = _class_method(path, class_name, method_name)
    namespace = {"List": List, "Tuple": Tuple}
    module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace[method_name]


class PickAttachPathTests(unittest.TestCase):
    def test_legacy_pick_uses_plural_attach_paths(self):
        method = _class_method(PICK_PATH, "Pick", "_legacy_simple_generate_manip_cmds")
        constants = {node.value for node in ast.walk(method) if isinstance(node, ast.Constant)}
        attributes = {node.attr for node in ast.walk(method) if isinstance(node, ast.Attribute)}

        self.assertIn("attach_objects", constants)
        self.assertIn("obj_prim_paths", constants)
        self.assertIn("attach_collision_prim_paths", attributes)
        self.assertNotIn("attach_obj", constants)
        self.assertNotIn("mesh_prim_path", attributes)

    def test_task_attach_paths_resolve_to_collidable_usd_prims(self):
        task = yaml.safe_load(TASK_PATH.read_text(encoding="utf-8"))["tasks"][0]
        for object_name in ("apple_0_id9008", "orange_0_id9009"):
            with self.subTest(object_name=object_name):
                cfg = next(item for item in task["objects"] if item["name"] == object_name)
                usd_path = TASK_PATH.parents[3] / cfg["path"]
                stage = Usd.Stage.Open(str(usd_path))

                resolution = resolve_attach_collision_prims(
                    "/World",
                    "/World/Aligned",
                    cfg,
                    stage.GetPrimAtPath,
                )

                self.assertIsNone(resolution.failure_code)
                self.assertEqual(resolution.source, "explicit_plural")
                self.assertEqual(
                    resolution.prim_paths,
                    ["/World/Aligned/Normalize/Source/base_link/collisions"],
                )

    def test_legacy_attach_rejects_none_before_path_operations(self):
        resolve = _load_method(
            CONTROLLER_PATH,
            "TemplateController",
            "_resolve_attach_object_names",
        )

        with self.assertRaisesRegex(ValueError, "non-empty CuRobo obstacle path strings"):
            resolve(object(), None)


if __name__ == "__main__":
    unittest.main()
