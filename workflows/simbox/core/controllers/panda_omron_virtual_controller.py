"""Controller registration for PandaOmronVirtual."""

from __future__ import annotations

from core.controllers.base_controller import register_controller
from core.controllers.panda_omron_controller import PandaOmronController


@register_controller
class PandaOmronVirtualController(PandaOmronController):
    """Panda arm controller shared by the virtual-base PandaOmron model."""
