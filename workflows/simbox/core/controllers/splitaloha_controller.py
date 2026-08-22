"""SplitAloha dual-arm controller – template-based."""

from core.controllers.base_controller import ArmSpec, register_controller
from core.controllers.template_controller import TemplateController


# pylint: disable=unused-argument
@register_controller
class SplitAlohaController(TemplateController):
    arm_spec = ArmSpec(
        planner_joints=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        control_joints={
            "left": ("fl_joint1", "fl_joint2", "fl_joint3", "fl_joint4", "fl_joint5", "fl_joint6"),
            "right": ("fr_joint1", "fr_joint2", "fr_joint3", "fr_joint4", "fr_joint5", "fr_joint6"),
        },
        default_ignore_substring=("material", "Plane", "conveyor", "scene", "table", "fluid"),
        gripper_home=(1.0,),
        gripper_clip_max=0.1,
    )

@register_controller
class SplitAlohaActualController(SplitAlohaController):
    """Controller registry entry for the physical 4WIS SplitAlohaActual."""
