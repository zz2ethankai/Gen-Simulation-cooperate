"""Skills module initialization."""

from core.skills.base_skill import SKILL_DICT

from .approach_rotate import Approach_Rotate
from .artpreplan import Artpreplan
from .close import Close
from .dexpick import Dexpick
from .dexplace import Dexplace
from .dynamicpick import Dynamicpick
from .failpick import FailPick
from .flip import Flip
from .goto_pose import Goto_Pose
from .gripper_action import Gripper_Action
from .heuristic_skill import Heuristic_Skill
from .home import Home
from .joint_ctrl import Joint_Ctrl
from .manualpick import Manualpick
from .mobile_rotate import Mobile_Rotate
from .mobile_translate import Mobile_Translate
from .move import Move
from .open import Open
from .observe_hold import ObserveHold
from .pick import Pick
from .pick_plan_probe import PickPlanProbe
from .place import Place
from .place_plan_probe import PlacePlanProbe
from .pour_water_succ import Pour_Water_Succ
from .rotate import Rotate
from .rotate_obj import Rotate_Obj
from .return_to_episode_initial import ReturnToEpisodeInitial
from .scan import Scan
from .track import Track
from .wait import Wait

# Explicitly declare the public interface
__all__ = [
    "Approach_Rotate",
    "Artpreplan",
    "Close",
    "Dexpick",
    "Dexplace",
    "Dynamicpick",
    "FailPick",
    "Flip",
    "Goto_Pose",
    "Gripper_Action",
    "Heuristic_Skill",
    "Home",
    "Joint_Ctrl",
    "Mobile_Rotate",
    "Mobile_Translate",
    "ReturnToEpisodeInitial",
    "Move",
    "Open",
    "ObserveHold",
    "Pick",
    "PickPlanProbe",
    "Manualpick",
    "Place",
    "PlacePlanProbe",
    "Pour_Water_Succ",
    "Rotate",
    "Rotate_Obj",
    "Scan",
    "Track",
    "Wait",
    "get_skill_cls",
    "get_skill_dict",
]


def get_skill_cls(category_name):
    """Get skill class by category name."""
    return SKILL_DICT[category_name]


def get_skill_dict():
    """Get skill dictionary."""
    return SKILL_DICT
