"""Observation-only Skill that holds measured joints without planning."""

from core.skills.base_skill import BaseSkill, register_skill


@register_skill
class ObserveHold(BaseSkill):
    """Keep one arm and its gripper at their measured positions for N steps."""

    def __init__(self, robot, controller, task, cfg, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg
        self.name = cfg["name"]
        self.hold_steps = int(cfg.get("hold_steps", cfg.get("wait_steps", 300)))
        if self.hold_steps <= 0:
            raise ValueError("ObserveHold requires hold_steps > 0")
        self.manip_list = []

    def simple_generate_manip_cmds(self):
        # Controller.forward dispatches this bookkeeping command directly to
        # observe_hold(); no FK, IK, CuRobo planning, or gripper state change.
        command = (None, None, "observe_hold", {})
        self.manip_list = [command] * self.hold_steps

    def is_done(self):
        if self.manip_list:
            self.manip_list.pop(0)
        return not self.manip_list

    def is_success(self):
        return not self.manip_list
