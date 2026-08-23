"""CuRobo-backed controller implementation and its internal owners.

The parent :mod:`core.controllers` package contains the controller registry
and robot-specific subclasses.  This package owns the shared Isaac/CuRobo
implementation, split into setup, planning runtime, execution, and the
Skill-facing runtime port.
"""

from importlib import import_module


def __getattr__(name):
    if name != "TemplateController":
        raise AttributeError(name)
    value = getattr(import_module("core.controllers.curobo.controller"), name)
    globals()[name] = value
    return value


__all__ = ["TemplateController"]
