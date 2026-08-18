"""Lazy robot registry access that keeps profile loading Isaac-free."""

from __future__ import annotations

from importlib import import_module

_ROBOT_MODULES = {
    "TemplateRobot": "template_robot",
    "FR3": "fr3",
    "FrankaRobotiq85": "franka_robotiq85",
    "Genie1": "genie1",
    "Lift2": "lift2",
    "SplitAloha": "split_aloha",
    "SplitAlohaActual": "split_aloha",
    "PandaOmron": "panda_omron",
    "PandaOmronVirtual": "panda_omron_virtual",
}

__all__ = ["get_robot_cls"]


def get_robot_cls(category_name: str):
    """Load and return one registered robot implementation on demand."""

    from .base_robot import ROBOT_DICT

    if category_name not in ROBOT_DICT:
        try:
            module_name = _ROBOT_MODULES[category_name]
        except KeyError as exc:
            raise KeyError(f"unknown robot target_class: {category_name}") from exc
        import_module(f"{__package__}.{module_name}")
    return ROBOT_DICT[category_name]


def __getattr__(name: str):
    if name in _ROBOT_MODULES:
        return get_robot_cls(name)
    raise AttributeError(name)
