"""Robot-profile capability and Skill admission contract tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.contracts import RobotAdmission
from agent.robot_skills import (
    RobotSkillContractError,
    load_skill_contracts,
    validate_profile_skill_admission,
)


def _profile(**overrides):
    values = {
        "profile_id": "test_floor_standing_v1",
        "profile_hash": "profile-hash",
        "capabilities": frozenset({"pick", "place"}),
        "collision_world_modes": frozenset({"physics_schema"}),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _admissions(*, state="admitted", profile_hash="profile-hash"):
    return {
        ("test_floor_standing_v1", skill, "physics_schema"): RobotAdmission(
            profile_id="test_floor_standing_v1",
            skill=skill,
            collision_world_mode="physics_schema",
            state=state,
            profile_hash=profile_hash,
        )
        for skill in ("pick", "place")
    }


def test_admitted_profile_satisfies_capability_and_collision_contracts():
    required = validate_profile_skill_admission(
        _profile(),
        ["pick", "place"],
        "physics_schema",
        contracts=load_skill_contracts(),
        admissions=_admissions(),
    )

    assert required == frozenset({"pick", "place"})


def test_implemented_profile_cannot_execute_agent_skill():
    with pytest.raises(RobotSkillContractError, match="not admitted"):
        validate_profile_skill_admission(
            _profile(),
            ["pick"],
            "physics_schema",
            contracts=load_skill_contracts(),
            admissions=_admissions(state="implemented"),
        )


def test_profile_hash_change_invalidates_admission():
    with pytest.raises(RobotSkillContractError, match="profile hash is stale"):
        validate_profile_skill_admission(
            _profile(profile_hash="changed"),
            ["pick"],
            "physics_schema",
            contracts=load_skill_contracts(),
            admissions=_admissions(),
        )


def test_missing_capability_is_rejected_before_admission():
    with pytest.raises(RobotSkillContractError, match="lacks required capabilities"):
        validate_profile_skill_admission(
            _profile(capabilities=frozenset({"pick"})),
            ["pick", "place"],
            "physics_schema",
            contracts=load_skill_contracts(),
            admissions=_admissions(),
        )
