"""FR3 controller – template-based."""

from core.controllers.base_controller import ArmSpec, register_controller
from core.controllers.template_controller import TemplateController


# pylint: disable=unused-argument
@register_controller
class FR3Controller(TemplateController):
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
        default_ignore_substring=("material", "Plane", "conveyor", "scene", "table"),
        gripper_home=(1.0,),
        collision_cache={"cuboid": 1000, "mesh": 1000},
        supported_arms=("left",),
    )
