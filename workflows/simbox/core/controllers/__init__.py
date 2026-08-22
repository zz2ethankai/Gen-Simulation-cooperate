"""Controller contracts with lazy subclass registration.

Narrow controller components should be importable without eagerly importing
Isaac/CuRobo-backed robot subclasses.  The concrete classes are loaded only
when a caller requests a subclass or the registry.
"""

from importlib import import_module

from core.controllers.controller_registry import ArmSpec, CONTROLLER_DICT
from core.controllers.curobo.phase_execution import ExecutionStatus
from core.controllers.curobo.skill_runtime import SkillRuntimePort
from core.planning.domain_types import CommandStatus

_LAZY_CONTROLLERS = {
    "TemplateController": ("core.controllers.curobo.controller", "TemplateController"),
    "FR3Controller": ("core.controllers.fr3_controller", "FR3Controller"),
    "FrankaRobotiq85Controller": (
        "core.controllers.frankarobotiq85_controller",
        "FrankaRobotiq85Controller",
    ),
    "Genie1Controller": ("core.controllers.genie1_controller", "Genie1Controller"),
    "Lift2Controller": ("core.controllers.lift2_controller", "Lift2Controller"),
    "PandaOmronController": (
        "core.controllers.panda_omron_controller",
        "PandaOmronController",
    ),
    "PandaOmronVirtualController": (
        "core.controllers.panda_omron_virtual_controller",
        "PandaOmronVirtualController",
    ),
    "SplitAlohaActualController": (
        "core.controllers.splitaloha_controller",
        "SplitAlohaActualController",
    ),
    "SplitAlohaController": (
        "core.controllers.splitaloha_controller",
        "SplitAlohaController",
    ),
}


def __getattr__(name):
    target = _LAZY_CONTROLLERS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, class_name = target
    value = getattr(import_module(module_name), class_name)
    globals()[name] = value
    return value


def _load_controllers():
    for name in _LAZY_CONTROLLERS:
        getattr(__import__(__name__, fromlist=[name]), name)

__all__ = [
    "TemplateController",
    "ArmSpec",
    "SkillRuntimePort",
    "CommandStatus",
    "ExecutionStatus",
    "FR3Controller",
    "FrankaRobotiq85Controller",
    "Genie1Controller",
    "Lift2Controller",
    "PandaOmronController",
    "PandaOmronVirtualController",
    "SplitAlohaActualController",
    "SplitAlohaController",
]


def get_controller_cls(category_name):
    """Get controller class by category name."""
    _load_controllers()
    return CONTROLLER_DICT[category_name]


def get_controller_dict():
    """Get controller dictionary."""
    _load_controllers()
    return CONTROLLER_DICT
