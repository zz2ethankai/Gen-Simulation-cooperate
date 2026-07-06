#!/usr/bin/env python3
"""Build the RoboCasa PandaOmron composite MJCF from a robosuite checkout."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path


def _install_minimal_robosuite_stubs(robosuite_root: Path) -> None:
    package_dir = robosuite_root / "robosuite"

    robosuite_pkg = types.ModuleType("robosuite")
    robosuite_pkg.__path__ = [str(package_dir)]
    robosuite_pkg.__file__ = str(package_dir / "__init__.py")
    sys.modules["robosuite"] = robosuite_pkg

    macros = types.ModuleType("robosuite.macros")
    macros.USING_INSTANCE_RANDOMIZATION = False
    macros.SIMULATION_TIMESTEP = 0.002
    macros.ENABLE_NUMBA = False
    macros.CACHE_NUMBA = False
    macros.IMAGE_CONVENTION = "opengl"
    macros.CONCATENATE_IMAGES = False
    sys.modules["robosuite.macros"] = macros

    mujoco = types.ModuleType("mujoco")
    mujoco.MjModel = object
    sys.modules["mujoco"] = mujoco

    grippers_pkg = types.ModuleType("robosuite.models.grippers")
    grippers_pkg.__path__ = [str(package_dir / "models" / "grippers")]
    grippers_pkg.__file__ = str(package_dir / "models" / "grippers" / "__init__.py")
    sys.modules["robosuite.models.grippers"] = grippers_pkg


def build_panda_omron_xml(robosuite_root: Path) -> ET.ElementTree:
    _install_minimal_robosuite_stubs(robosuite_root)

    from robosuite.models.bases import OmronMobileBase  # pylint: disable=import-outside-toplevel
    from robosuite.models.grippers.panda_gripper import PandaGripper  # pylint: disable=import-outside-toplevel
    from robosuite.models.robots import PandaOmron  # pylint: disable=import-outside-toplevel

    robot = PandaOmron(idn=0)
    robot.add_base(OmronMobileBase(idn=0))
    robot.add_gripper(PandaGripper(idn="0_right"), robot.eef_name["right"])
    return ET.ElementTree(robot.root)


def _localized_mesh_path(abs_path: Path, robosuite_root: Path) -> Path:
    asset_root = robosuite_root / "robosuite" / "models" / "assets"
    mapping_roots = [
        asset_root / "robots" / "panda",
        asset_root / "bases",
        asset_root / "grippers",
    ]
    for source_root in mapping_roots:
        try:
            return abs_path.relative_to(source_root)
        except ValueError:
            continue
    raise ValueError(f"Mesh path is outside known robosuite asset roots: {abs_path}")


def localize_mesh_assets(tree: ET.ElementTree, robosuite_root: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for node in tree.getroot().findall(".//asset/*[@file]"):
        abs_path = Path(node.get("file", "")).resolve()
        rel_path = _localized_mesh_path(abs_path, robosuite_root)
        dest_path = output_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(abs_path, dest_path)
        node.set("file", rel_path.as_posix())
        copied.append(rel_path.as_posix())
    return copied


def add_isaac_importer_placeholders(tree: ET.ElementTree) -> list[str]:
    """Add tiny invisible inertial geometry to empty MJCF bodies.

    Isaac Sim 4.1's MJCF importer can fail on bodies that only exist as
    kinematic grouping nodes. RoboSuite's PandaOmron composition creates a few
    such bodies, while MuJoCo accepts them. The placeholders are non-colliding
    and near-zero mass so the source hierarchy remains intact for import.
    """

    patched: list[str] = []
    for body in tree.getroot().findall(".//body"):
        if body.find("inertial") is not None or body.find("geom") is not None:
            continue

        name = body.get("name", "unnamed_body")
        body.insert(
            0,
            ET.Element(
                "inertial",
                {
                    "pos": "0 0 0",
                    "mass": "0.000001",
                    "diaginertia": "0.000001 0.000001 0.000001",
                },
            ),
        )
        body.insert(
            1,
            ET.Element(
                "geom",
                {
                    "name": f"{name}_isaac_importer_placeholder",
                    "type": "sphere",
                    "pos": "0 0 0",
                    "size": "0.001",
                    "rgba": "0 0 0 0",
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "5",
                    "mass": "0.000001",
                },
            ),
        )
        patched.append(name)
    return patched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robosuite-root",
        type=Path,
        default=Path("/tmp/robosuite"),
        help="Path to an ARISE-Initiative/robosuite checkout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("InternDataAssets/assets/panda_omron/source/panda_omron.xml"),
        help="Output MJCF XML path.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("InternDataAssets/assets/panda_omron/source/source_metadata.json"),
        help="Output metadata JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robosuite_root = args.robosuite_root.resolve()
    if not (robosuite_root / "robosuite" / "models").is_dir():
        raise FileNotFoundError(f"robosuite models directory not found under {robosuite_root}")

    tree = build_panda_omron_xml(robosuite_root)
    patched_empty_bodies = add_isaac_importer_placeholders(tree)
    copied_assets = localize_mesh_assets(tree, robosuite_root, args.output.parent)
    ET.indent(tree, space="  ")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)

    metadata = {
        "source": "RoboCasa365 PandaOmron via robosuite",
        "robosuite_root": str(robosuite_root),
        "components": {
            "robot": "robosuite.models.robots.PandaOmron",
            "arm_xml": str(robosuite_root / "robosuite/models/assets/robots/panda/robot.xml"),
            "base_xml": str(robosuite_root / "robosuite/models/assets/bases/omron_mobile_base.xml"),
            "gripper_xml": str(robosuite_root / "robosuite/models/assets/grippers/panda_gripper.xml"),
        },
        "notes": [
            "Uses robosuite RobotModel.add_mobile_base() to compose Panda and OmronMobileBase.",
            "Uses the default PandaGripper attached to robot0_right_hand.",
            "Mesh file attributes are localized relative to this source directory for Isaac MJCF import.",
            "Tiny invisible placeholders are added to empty grouping bodies for Isaac Sim 4.1 MJCF import.",
        ],
        "localized_mesh_count": len(copied_assets),
        "isaac_placeholder_bodies": patched_empty_bodies,
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    joints = [node.get("name") for node in tree.getroot().findall(".//joint")]
    actuators = [node.get("name") for node in tree.getroot().findall(".//actuator/*")]
    print(f"Wrote {args.output}")
    print(f"Wrote {args.metadata_output}")
    print(f"Localized meshes: {len(copied_assets)}")
    print(f"Joints ({len(joints)}): {joints}")
    print(f"Actuators ({len(actuators)}): {actuators}")


if __name__ == "__main__":
    main()
