"""Runtime motion-planning services shared by Skills and validation probes.

Heavy CuRobo-backed services remain lazily imported so the schema and state
contracts can be tested with ordinary USD Python outside Isaac Sim.
"""

from .motion_command import MotionPhase, MotionPhaseCommand

__all__ = [
    "GraspPlanEvaluation",
    "GraspPlanEvaluator",
    "GraspPlanResult",
    "MotionPhase",
    "MotionPhaseCommand",
]


def __getattr__(name):
    if name in {"GraspPlanEvaluation", "GraspPlanEvaluator", "GraspPlanResult"}:
        from .grasp_plan_evaluator import (
            GraspPlanEvaluation,
            GraspPlanEvaluator,
            GraspPlanResult,
        )

        return {
            "GraspPlanEvaluation": GraspPlanEvaluation,
            "GraspPlanEvaluator": GraspPlanEvaluator,
            "GraspPlanResult": GraspPlanResult,
        }[name]
    raise AttributeError(name)
