"""Genie1 dual-arm controller – template-based."""

from core.controllers.base_controller import ArmSpec, register_controller
from core.controllers.template_controller import TemplateController


# pylint: disable=unused-argument
@register_controller
class Genie1Controller(TemplateController):
    arm_spec = ArmSpec(
        planner_joints={
            "left": (
                "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
                "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6",
                "idx27_arm_l_joint7",
            ),
            "right": (
                "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
                "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6",
                "idx67_arm_r_joint7",
            ),
        },
        control_joints={
            "left": (
                "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
                "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6",
                "idx27_arm_l_joint7",
            ),
            "right": (
                "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
                "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6",
                "idx67_arm_r_joint7",
            ),
        },
        default_ignore_substring=("material", "Plane", "conveyor", "scene", "table", "fluid"),
        gripper_home=(1.0,),
        sort_path_weights=(1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 1.0),
    )
