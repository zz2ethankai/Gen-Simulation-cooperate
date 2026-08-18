"""FR3 controller – template-based."""

from core.controllers.base_controller import register_controller
from core.controllers.template_controller import TemplateController


# pylint: disable=unused-argument
@register_controller
class FR3Controller(TemplateController):
    def _get_default_ignore_substring(self):
        return ["material", "Plane", "conveyor", "scene", "table"]

    def _get_motion_gen_collision_cache(self):
        """FR3 uses larger collision cache (1000) for MotionGenConfig than template default (700)."""
        return {"obb": 1000, "mesh": 1000}
