"""PandaOmron controller - template-based Panda arm controller."""

import numpy as np
from core.controllers.base_controller import register_controller
from core.controllers.template_controller import TemplateController


# pylint: disable=unused-argument
@register_controller
class PandaOmronController(TemplateController):
    def _get_default_ignore_substring(self):
        return ["material", "Plane", "conveyor", "scene", "table", "fluid"]

    def _configure_joint_indices(self, robot_file: str) -> None:
        self.raw_js_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        if "left" not in robot_file:
            raise NotImplementedError("PandaOmron currently exposes the Panda arm as the left controller")

        self.cmd_js_names = [
            "robot0_joint1",
            "robot0_joint2",
            "robot0_joint3",
            "robot0_joint4",
            "robot0_joint5",
            "robot0_joint6",
            "robot0_joint7",
        ]
        self.arm_indices = np.array(self.robot.cfg["left_joint_indices"])
        self.gripper_indices = np.array(self.robot.cfg["left_gripper_indices"])
        self.reference_prim_path = self.task.robots[self.name].fl_base_path
        self.lr_name = "left"
        self._gripper_state = 1.0 if self.robot.left_gripper_state == 1.0 else -1.0
        self._gripper_joint_position = np.array([0.04, -0.04], dtype=float)

    def get_gripper_action(self):
        if self._gripper_state > 0.0:
            return self._gripper_joint_position.copy()
        return np.zeros(2, dtype=float)
