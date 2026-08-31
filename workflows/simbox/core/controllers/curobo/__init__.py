from importlib import import_module
def __getattr__(name):
    if name != "TemplateController":
        raise AttributeError(name)
    value = getattr(import_module("core.controllers.curobo.controller"), name)
    globals()[name] = value
    return value
__all__ = ["TemplateController"]
