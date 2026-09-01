import logging
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

try:
    from core.planning.config_contract import canonicalize_planning_config
except ImportError:  # package import outside the Isaac ``simbox`` sys.path
    from workflows.simbox.core.planning.config_contract import canonicalize_planning_config


LOGGER = logging.getLogger("de_logger")


class AttrDict(dict):
    """A dict subclass that supports both task['key'] and task.key access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def _to_attr_dict(obj):
    if isinstance(obj, dict):
        return AttrDict({k: _to_attr_dict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attr_dict(i) for i in obj]
    return obj


class TaskConfigParser:
    """Shared utilities for workflow configuration parsing."""

    def __init__(self, task_cfg_path: str):
        self.task_cfg_path = str(task_cfg_path)
        self.config_path = Path(task_cfg_path).expanduser().resolve()

    def parse_tasks(self):
        yaml_conf = OmegaConf.load(self.task_cfg_path)
        task_cfgs = []
        if "tasks" not in yaml_conf:
            raise ValueError(
                f"Expected 'tasks' key in the task configuration file: {self.task_cfg_path}"
            )
        for task_index, task in enumerate(yaml_conf["tasks"]):
            if isinstance(task, (DictConfig, ListConfig)):
                cfg = OmegaConf.to_container(task, resolve=True)
            else:
                cfg = task
            if not isinstance(cfg, dict):
                raise ValueError(
                    f"Each task must resolve to a mapping in the task configuration file: "
                    f"{self.task_cfg_path}"
                )
            # Scene-format conversion outputs may intentionally omit
            # execution bindings.  Promote that scene-only document to the
            # minimal runtime identity expected by SimBoxDualWorkFlow while
            # preserving every explicit task implementation and id.
            if not cfg.get("task"):
                cfg["task"] = "BananaBaseTask"
                LOGGER.warning(
                    "Task config %s tasks[%d] has no task class; defaulting to BananaBaseTask",
                    self.config_path,
                    task_index,
                )
            if cfg.get("task_id") is None:
                cfg["task_id"] = task_index
                LOGGER.warning(
                    "Task config %s tasks[%d] has no task_id; defaulting to %d",
                    self.config_path,
                    task_index,
                    task_index,
                )
            cfg.setdefault("offset", None)
            # Normalize before the workflow sees the task.  This is the one
            # config-boundary call site that can provide the real path used by
            # warning de-duplication; runtime resets do not emit duplicate
            # warnings for the same ``(path, field)`` pair.
            cfg = canonicalize_planning_config(cfg, config_path=self.config_path)
            metadata = cfg.get("metadata")
            if isinstance(metadata, dict):
                metadata.setdefault("source_yaml", str(self.config_path))
            else:
                cfg["metadata"] = {"source_yaml": str(self.config_path)}
            task_cfgs.append(_to_attr_dict(cfg))
        return task_cfgs
