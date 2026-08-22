"""Concrete-class contract for every registered SimBox controller."""

from __future__ import annotations

import inspect

import pytest


@pytest.fixture(scope="module")
def controller_contract():
    """Load the real Isaac/CuRobo controller modules after Kit startup."""

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        from core.controllers import get_controller_dict
        from core.controllers.panda_omron_virtual_controller import (
            PandaOmronVirtualController,
        )
        from core.controllers.template_controller import TemplateController

        yield get_controller_dict(), PandaOmronVirtualController, TemplateController
    finally:
        simulation_app.close()


def test_registered_robot_controllers_are_concrete_typed_facades(controller_contract):
    """The Isaac ABC must not reject any configured robot at task startup."""

    registry, panda_omron_virtual, template_controller = controller_contract
    expected = {
        "FR3",
        "FrankaRobotiq85",
        "Genie1",
        "Lift2",
        "PandaOmron",
        "PandaOmronVirtual",
        "SplitAloha",
        "SplitAlohaActual",
    }

    assert expected <= set(registry)
    assert not inspect.isabstract(panda_omron_virtual)
    for category in expected:
        controller_cls = registry[category]
        assert issubclass(controller_cls, template_controller)
        assert not inspect.isabstract(controller_cls), category
        assert callable(controller_cls.execute)


def test_template_bridge_does_not_accept_legacy_forward_payloads(controller_contract):
    """Isaac's hook is only a typed bridge; planning uses ``execute``."""

    _registry, _virtual, template_controller = controller_contract
    forward = template_controller.forward
    annotation = inspect.signature(forward).parameters["command"].annotation
    assert annotation == "MotionPhaseCommand"
    assert "execute" in inspect.getsource(forward)
    assert "tuple" not in inspect.getsource(forward)
    assert "getattr" not in inspect.getsource(forward)
