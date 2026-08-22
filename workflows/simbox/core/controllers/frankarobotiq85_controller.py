"""Franka Robotiq85 controller – template-based."""

from core.controllers.base_controller import ArmSpec, register_controller
from core.controllers.template_controller import TemplateController


# pylint: disable=unused-argument
@register_controller
class FrankaRobotiq85Controller(TemplateController):
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
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7",
        )},
        default_ignore_substring=("material", "Plane", "conveyor", "scene"),
        gripper_home=(5.0, 5.0),
        gripper_clip_max=5.0,
        supported_arms=("left",),
    )
