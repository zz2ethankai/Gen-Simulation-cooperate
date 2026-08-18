"""Genie1 dual-arm controller – template-based."""

import numpy as np
from core.controllers.base_controller import register_controller
from core.controllers.template_controller import TemplateController


# pylint: disable=unused-argument
@register_controller
class Genie1Controller(TemplateController):
    def _get_default_ignore_substring(self):
        return ["material", "Plane", "conveyor", "scene", "table", "fluid"]

    def _get_sort_path_weights(self):
        """Genie1: weight joints 4 and 5 (index 4,5) by 3.0 for path selection."""
        return [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 1.0]

    def mobile_move(self, target: np.ndarray, joint_indices: np.ndarray = None, initial_position: np.ndarray = None):
        raise NotImplementedError
