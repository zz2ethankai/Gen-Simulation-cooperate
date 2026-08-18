"""Franka Robotiq85 controller – template-based."""

from core.controllers.base_controller import register_controller
from core.controllers.template_controller import TemplateController


# pylint: disable=unused-argument
@register_controller
class FrankaRobotiq85Controller(TemplateController):
    def _get_default_ignore_substring(self):
        return ["material", "Plane", "conveyor", "scene"]
