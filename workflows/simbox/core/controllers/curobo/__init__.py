"""CuRobo-backed controller implementation and its internal components.

The parent :mod:`core.controllers` package contains the controller registry
and robot-specific subclasses.  This package owns the shared Isaac/CuRobo
implementation, split by responsibility so that execution, scene setup,
planning queries, and phase handling are no longer hidden behind a collection
of similarly named ``controller_*`` modules.
"""

from importlib import import_module


def __getattr__(name):
    if name != "TemplateController":
        raise AttributeError(name)
    value = getattr(import_module("core.controllers.curobo.controller"), name)
    globals()[name] = value
    return value


__all__ = ["TemplateController"]
