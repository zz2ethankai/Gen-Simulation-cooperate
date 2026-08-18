"""Robot Skill contracts and profile admission gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from ..contracts import RobotAdmission, RobotAdmissionState, SkillContract


ROBOT_SKILLS_DIR = Path(__file__).resolve().parent
DEFAULT_CONTRACTS_PATH = ROBOT_SKILLS_DIR / "contracts.yaml"
DEFAULT_ADMISSIONS_PATH = ROBOT_SKILLS_DIR.parent / "registry" / "robot_admissions.yaml"
EXECUTABLE_ADMISSION_STATES = frozenset(
    {RobotAdmissionState.ADMITTED.value, RobotAdmissionState.QUALIFIED.value}
)


class RobotSkillContractError(ValueError):
    """A Skill request is incompatible with its robot execution profile."""


def load_skill_contracts(path: Path | None = None) -> dict[str, SkillContract]:
    source = path or DEFAULT_CONTRACTS_PATH
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    contracts: dict[str, SkillContract] = {}
    for item in payload.get("skills", []):
        contract = SkillContract.from_dict(item)
        name = contract.name.lower()
        if name in contracts:
            raise RobotSkillContractError(f"duplicate Skill contract: {name}")
        contracts[name] = contract
    return contracts


def load_robot_admissions(path: Path | None = None) -> dict[tuple[str, str, str], RobotAdmission]:
    source = path or DEFAULT_ADMISSIONS_PATH
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    admissions: dict[tuple[str, str, str], RobotAdmission] = {}
    for item in payload.get("admissions", []):
        admission = RobotAdmission.from_dict(item)
        key = (
            admission.profile_id,
            admission.skill.lower(),
            admission.collision_world_mode,
        )
        if key in admissions:
            raise RobotSkillContractError(
                "duplicate robot admission: " + "/".join(key)
            )
        state = str(getattr(admission.state, "value", admission.state))
        if state != RobotAdmissionState.ABSENT.value and not admission.profile_hash:
            raise RobotSkillContractError(
                f"{state} robot admission requires profile_hash: {'/'.join(key)}"
            )
        if state == RobotAdmissionState.QUALIFIED.value and (
            not admission.evidence_run_ids or not admission.validated_seeds
        ):
            raise RobotSkillContractError(
                f"qualified robot admission requires evidence and validated seeds: {'/'.join(key)}"
            )
        admissions[key] = admission
    return admissions


def required_skill_capabilities(
    skill_names: Iterable[str],
    contracts: Mapping[str, SkillContract],
) -> frozenset[str]:
    required: set[str] = set()
    for raw_name in skill_names:
        name = raw_name.lower()
        contract = contracts.get(name)
        if contract is None:
            raise RobotSkillContractError(f"unsupported Skill: {name}")
        required.update(contract.required_capabilities)
    return frozenset(required)


def validate_profile_skill_admission(
    profile: Any,
    skill_names: Iterable[str],
    collision_world_mode: str,
    *,
    contracts: Mapping[str, SkillContract] | None = None,
    admissions: Mapping[tuple[str, str, str], RobotAdmission] | None = None,
    allowed_states: frozenset[str] = EXECUTABLE_ADMISSION_STATES,
) -> frozenset[str]:
    skill_contracts = contracts or load_skill_contracts()
    robot_admissions = admissions or load_robot_admissions()
    names = tuple(dict.fromkeys(name.lower() for name in skill_names))
    required = required_skill_capabilities(names, skill_contracts)
    profile_capabilities = frozenset(profile.capabilities)
    missing = sorted(required - profile_capabilities)
    if missing:
        raise RobotSkillContractError(
            f"robot profile {profile.profile_id!r} lacks required capabilities: {missing}"
        )
    if collision_world_mode not in profile.collision_world_modes:
        raise RobotSkillContractError(
            f"robot profile {profile.profile_id!r} does not support collision world "
            f"{collision_world_mode!r}"
        )
    for name in names:
        contract = skill_contracts[name]
        if collision_world_mode not in contract.collision_world_modes:
            raise RobotSkillContractError(
                f"Skill {name!r} does not support collision world {collision_world_mode!r}"
            )
        key = (profile.profile_id, name, collision_world_mode)
        admission = robot_admissions.get(key)
        if admission is None:
            raise RobotSkillContractError(
                "robot Skill has no admission record: " + "/".join(key)
            )
        state = str(getattr(admission.state, "value", admission.state))
        if state not in allowed_states:
            raise RobotSkillContractError(
                f"robot Skill is not admitted: {'/'.join(key)} state={state}"
            )
        if admission.profile_hash != profile.profile_hash:
            raise RobotSkillContractError(
                f"robot admission profile hash is stale: {'/'.join(key)}"
            )
    return required


__all__ = [
    "RobotSkillContractError",
    "load_robot_admissions",
    "load_skill_contracts",
    "required_skill_capabilities",
    "validate_profile_skill_admission",
]
