"""Observation-only Skill that holds measured joints without planning."""

from core.skills.base_skill import BaseSkill, register_skill


@register_skill
class ObserveHold(BaseSkill):
    """Keep one arm and its gripper at their measured positions for N steps."""

    def __init__(self, robot, skill_runtime, task, cfg, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.robot = robot
        self.bind_skill_runtime(skill_runtime)
        self.task = task
        self.skill_cfg = cfg
        self.name = cfg["name"]
        self.hold_steps = int(cfg.get("hold_steps", cfg.get("wait_steps", 300)))
        if self.hold_steps <= 0:
            raise ValueError("ObserveHold requires hold_steps > 0")
        # Keep a scheduler-visible tick budget without producing arm
        # commands.  Passthrough nodes are skipped by action collection, but
        # ``update`` still consumes one sentinel per workflow step.
        self.manip_list = [None] * self.hold_steps

    def simple_generate_manip_cmds(self):
        # Observation-only Skills stay outside the arm command path.  The
        # scheduler advances them on workflow ticks while the controller's
        # normal hold action preserves the measured articulation state.
        self.manip_list = [None] * self.hold_steps

    def is_ready(self):
        return False

    def is_done(self):
        if self.manip_list:
            self.manip_list.pop(0)
        return not self.manip_list

    def is_success(self):
        return not self.manip_list
