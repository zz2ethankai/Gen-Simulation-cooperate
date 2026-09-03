"""PandaOmron controller - template-based Panda arm controller."""

from core.controllers.controller_registry import ArmSpec, register_controller
from core.controllers.curobo.controller import TemplateController


# pylint: disable=unused-argument
@register_controller
class PandaOmronController(TemplateController):
    arm_spec = ArmSpec(
        planner_joints=(
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ),
        control_joints={"left": (
            "robot0_joint1", "robot0_joint2", "robot0_joint3", "robot0_joint4",
            "robot0_joint5", "robot0_joint6", "robot0_joint7",
        )},
        default_ignore_substring=("material", "Plane", "conveyor", "scene", "fluid"),
        supported_arms=("left",),
    )
