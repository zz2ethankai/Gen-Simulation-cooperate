import importlib.util
from pathlib import Path
from types import SimpleNamespace


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "nimbus_extension"
    / "components"
    / "plan_with_render"
    / "plan_with_render.py"
)
_MODULE_SPEC = importlib.util.spec_from_file_location("plan_with_render_module", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(_MODULE)
_format_failure_context = _MODULE._format_failure_context


class _WorkflowWithFailure:
    def get_failure_context(self):
        return {
            "robot": "panda_omron",
            "failed_skill": "pick",
            "failed_skill_id": "pick_orange_0_id9009",
            "failure_reason": "skill_not_feasible",
            "failure_message": "Skill feasibility check failed before completion.",
        }


def test_format_failure_context_includes_skill_name_and_id():
    context = _format_failure_context(_WorkflowWithFailure())

    assert context == (
        "robot=panda_omron, failed_skill=pick, failed_skill_id=pick_orange_0_id9009, "
        "failure_reason=skill_not_feasible, "
        "failure_message=Skill feasibility check failed before completion."
    )


def test_format_failure_context_is_empty_when_workflow_does_not_expose_it():
    assert _format_failure_context(object()) == ""


def test_format_failure_context_does_not_break_failure_logging():
    class WorkflowWithBrokenFailureContext:
        def get_failure_context(self):
            raise RuntimeError("failure context unavailable")

    assert _format_failure_context(WorkflowWithBrokenFailureContext()) == ""


def test_plan_with_render_failure_log_includes_skill_id():
    class Workflow(_WorkflowWithFailure):
        def plan_with_render(self):
            return 0

    class Logger:
        def __init__(self):
            self.messages = []

        def info(self, message):
            self.messages.append(message)

    component = object.__new__(_MODULE.EnvPlanWithRender)
    component.scene = SimpleNamespace(name="scene", wf=Workflow())
    component.save_failed = False
    component.emit_obs_on_failure = False
    component.failure_obs_length = 1
    component.logger = Logger()

    assert component.plan_with_render() is None
    assert "failed_skill=pick" in component.logger.messages[-1]
    assert "failed_skill_id=pick_orange_0_id9009" in component.logger.messages[-1]


def test_save_failed_is_accepted_and_forwarded_to_workflow():
    class Workflow:
        save_failed = None

        def plan_with_render(self):
            return 1

    workflow = Workflow()
    component = _MODULE.EnvPlanWithRender(iter(()), save_failed=True)
    component.scene = SimpleNamespace(name="scene", wf=workflow)

    observation = component.plan_with_render()

    assert workflow.save_failed is True
    assert observation.get_length() == 1
