import logging
import re
from abc import ABC

SKILL_DICT = {}
LOGGER = logging.getLogger("de_logger")


def register_skill(target_class):
    key = "_".join(re.sub(r"([A-Z0-9])", r" \1", target_class.__name__).split()).lower()
    # key = target_class.__name__
    # assert key not in SKILL_DICT
    SKILL_DICT[key] = target_class
    return target_class


class BaseSkill(ABC):
    def __init__(self):
        self.plan_flag = False
        self._target_visualizer = None
        self._target_visualization_context = {}
        self._target_visualization_handle = None

    def bind_target_visualizer(self, visualizer, **context):
        """Bind optional observational target rendering without changing Skill APIs."""
        self._target_visualizer = visualizer
        self._target_visualization_context = dict(context)

    def publish_target_intent(self, descriptor: dict):
        if self._target_visualizer is None:
            return None
        try:
            self._target_visualization_handle = self._target_visualizer.record_target(
                self, descriptor
            )
        except Exception:
            LOGGER.exception(
                "[SkillTargetDebug] failed to publish skill=%s",
                self.__class__.__name__,
            )
        return self._target_visualization_handle

    def complete_target_intent(self, success: bool):
        if self._target_visualizer is None:
            return
        try:
            self._target_visualizer.finish_target(
                self._target_visualization_handle, bool(success)
            )
        except Exception:
            LOGGER.exception(
                "[SkillTargetDebug] failed to complete skill=%s",
                self.__class__.__name__,
            )

    def abort_target_intent(self, reason: str):
        if self._target_visualizer is None:
            return
        try:
            self._target_visualizer.finish_target(
                self._target_visualization_handle,
                False,
                reason=str(reason),
            )
        except Exception:
            LOGGER.exception(
                "[SkillTargetDebug] failed to abort skill=%s",
                self.__class__.__name__,
            )

    def is_ready(self):
        return True

    def is_done(self):
        raise NotImplementedError

    def is_success(self):
        raise NotImplementedError

    def update(self):
        pass

    def is_feasible(self):
        return True

    def is_record(self):
        return True
