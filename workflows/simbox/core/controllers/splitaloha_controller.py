"""SplitAloha dual-arm controller – template-based."""

from core.controllers.base_controller import register_controller
from core.controllers.template_controller import TemplateController
from core.planning.motion_command import MotionPhaseCommand


# pylint: disable=unused-argument
@register_controller
class SplitAlohaController(TemplateController):
    def _get_default_ignore_substring(self):
        return ["material", "Plane", "conveyor", "scene", "table", "fluid"]

    def forward(self, manip_cmd, eps=5e-3):
        if isinstance(manip_cmd, MotionPhaseCommand):
            return super().forward(manip_cmd, eps=eps)
        ee_trans, ee_ori = manip_cmd[0:2]
        gripper_fn = manip_cmd[2]
        params = manip_cmd[3]
        self._last_command_name = gripper_fn
        assert hasattr(self, gripper_fn)
        method = getattr(self, gripper_fn)
        if gripper_fn in ["in_plane_rotation", "mobile_move", "dummy_forward", "joint_ctrl", "observe_hold"]:
            return method(**params)
        elif gripper_fn in ["update_pose_cost_metric", "update_specific"]:
            method(**params)
            return self.ee_forward(ee_trans, ee_ori, eps=eps, skip_plan=True)
        else:
            method(**params)
            return self.ee_forward(ee_trans, ee_ori, eps=eps)
