"""Stable data contracts shared by the task agent components."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ContractModel(BaseModel):
    """Pydantic v1/v2 compatible base with JSON-safe helpers."""

    class Config:
        extra = "forbid"
        use_enum_values = True

    def to_dict(self) -> dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")
        return json.loads(self.json())

    @classmethod
    def from_dict(cls, value: dict[str, Any]):
        if hasattr(cls, "model_validate"):
            return cls.model_validate(value)
        return cls.parse_obj(value)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        if hasattr(cls, "model_json_schema"):
            return cls.model_json_schema()
        return cls.schema()


class PlaceRelation(str, Enum):
    ON = "on"
    INSIDE = "inside"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    NEXT_TO = "next_to"
    HANG = "hang"
    INSERT = "insert"
    NONE = "none"


class ResolutionMode(str, Enum):
    REUSE_EXISTING = "reuse_existing"
    REUSE_SCENE_NEW_TASK = "reuse_scene_new_task"
    COMPOSE_REQUIRED = "compose_required"
    AMBIGUOUS = "ambiguous"


class RunStatus(str, Enum):
    CREATED = "created"
    PLANNED = "planned"
    WORKSPACE_READY = "workspace_ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


class RetentionKind(str, Enum):
    PLAYBOOK = "playbook"
    DEBUG_TOOL = "debug_tool"
    ROBOT_SKILL = "robot_skill"
    NONE = "none"


class StageExecutionMode(str, Enum):
    """How the Skill list of one planning stage is scheduled."""

    SINGLE_ARM_SINGLE_SKILL = "single_arm_single_skill"
    SINGLE_ARM_SEQUENTIAL = "single_arm_sequential"
    DUAL_ARM_SEQUENTIAL = "dual_arm_sequential"
    DUAL_ARM_SIMULTANEOUS = "dual_arm_simultaneous"


class TaskRequest(ContractModel):
    prompt: str
    source_object_terms: list[str] = Field(default_factory=list)
    target_object_terms: list[str] = Field(default_factory=list)
    relation: PlaceRelation = PlaceRelation.NONE
    constraints: list[str] = Field(default_factory=list)
    # None means that the user did not explicitly override the deterministic
    # default in agent/config.yaml.  Keeping that distinction prevents the LLM
    # from silently taking ownership of a global run policy.
    data_generation: bool | None = None


class AssetCapability(ContractModel):
    name: str
    category: str = "unknown"
    role: str = "unknown"
    target_class: str = "unknown"
    asset_path: str | None = None
    parent_fixture: str | None = None
    rigid_body: bool | None = None
    collision_enabled: bool | None = None
    attach_proxy_status: str = "not_required"
    attach_prim_path_children: list[str] = Field(default_factory=list)
    affordances: list[str] = Field(default_factory=list)


class SceneCapabilityManifest(ContractModel):
    task_id: str
    scene_id: str
    source_task: str
    task_class: str
    language: list[str] = Field(default_factory=list)
    robots: list[str] = Field(default_factory=list)
    robot_mounting: str = "floor"
    active_objects: list[str] = Field(default_factory=list)
    objects: list[AssetCapability] = Field(default_factory=list)
    container_regions: list[dict[str, Any]] = Field(default_factory=list)
    existing_skills: list[str] = Field(default_factory=list)
    physics_readiness: str = "unknown"


class SkillParameterContract(ContractModel):
    """Machine-readable constraints for one Agent-exposed Skill parameter."""

    value_type: str
    description: str
    owner: str = "agent"
    allowed_values: list[Any] = Field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    min_items: int | None = None
    max_items: int | None = None


class SkillContract(ContractModel):
    name: str
    category: str
    object_count: int
    supported_robots: list[str] = Field(default_factory=list)
    collision_world_modes: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)
    parameters: dict[str, SkillParameterContract] = Field(default_factory=dict)

    @property
    def allowed_params(self) -> list[str]:
        """Compatibility view used by diagnosis and deterministic validation."""

        return list(self.parameters)


class ResolutionDecision(ContractModel):
    mode: ResolutionMode
    selected_task_id: str | None = None
    selected_source_task: str | None = None
    selected_scene_id: str | None = None
    confidence: float = 0.0
    candidate_task_ids: list[str] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)
    object_role_overrides: dict[str, str] = Field(default_factory=dict)
    decision_basis: str


class SceneCompositionRequest(ContractModel):
    prompt: str
    suggested_base_scene: str | None = None
    required_assets: list[str] = Field(default_factory=list)
    required_relations: list[str] = Field(default_factory=list)
    required_robot_capabilities: list[str] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)
    validation_requirements: list[str] = Field(default_factory=list)
    status: str = "COMPOSITION_NOT_IMPLEMENTED_V1"


class RobotDecision(ContractModel):
    robot_type: str = "split_aloha"
    robot_profile: str = "split_aloha_tabletop_v1"
    decision_basis: str


class SkillStep(ContractModel):
    name: str
    objects: list[str]
    arm: str = "auto"
    params: dict[str, Any] = Field(default_factory=dict)
    decision_basis: str
    success_criteria: list[str] = Field(default_factory=list)
    evidence_required: list[str] = Field(default_factory=list)


class StagePlan(ContractModel):
    stage_id: str
    objective: str
    execution_mode: StageExecutionMode = StageExecutionMode.SINGLE_ARM_SEQUENTIAL
    skills: list[SkillStep]


class ObjectSubtask(ContractModel):
    subtask_id: str
    manipulated_object: str
    target_object: str | None = None
    relation: PlaceRelation = PlaceRelation.NONE
    arm: str
    stages: list[StagePlan]


class TaskPlan(ContractModel):
    prompt: str
    selected_task_id: str
    source_task: str
    task_request: TaskRequest
    robot: RobotDecision
    subtasks: list[ObjectSubtask]
    decision_basis: str
    unresolved: list[str] = Field(default_factory=list)


class PlanningResponse(ContractModel):
    task_request: TaskRequest
    resolution: ResolutionDecision
    task_plan: TaskPlan | None = None
    composition_request: SceneCompositionRequest | None = None


class ResolutionResponse(ContractModel):
    task_request: TaskRequest
    resolution: ResolutionDecision


class EvidenceBundle(ContractModel):
    attempt_id: str
    status: str
    task_success: bool = False
    event_status: str | None = None
    failure_reason: str | None = None
    episode_dir: str | None = None
    collision_world_ok: bool | None = None
    object_state_events: list[dict[str, Any]] = Field(default_factory=list)
    safety_events: list[dict[str, Any]] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    log_signals: list[str] = Field(default_factory=list)


class SkillParameterUpdate(ContractModel):
    subtask_id: str
    stage_id: str
    skill_index: int
    params: dict[str, Any] = Field(default_factory=dict)


class Diagnosis(ContractModel):
    stage: str
    failure_code: str
    category: str
    root_cause: str
    confidence: float = 1.0
    retryable: bool = False
    recommended_action: str
    workspace_action: str = "keep"
    skill_updates: list[SkillParameterUpdate] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class GeneratedFile(ContractModel):
    relative_path: str
    content: str


class RetentionDecision(ContractModel):
    kind: RetentionKind = RetentionKind.NONE
    name: str = ""
    category: str = ""
    summary: str = ""
    reusable_scope: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    proposed_interface: dict[str, Any] = Field(default_factory=dict)
    files: list[GeneratedFile] = Field(default_factory=list)


class RunState(ContractModel):
    run_id: str
    prompt: str
    status: RunStatus = RunStatus.CREATED
    run_dir: str
    current_subtask: int = 0
    attempt_index: int = 0
    max_revisions: int = 2
    task_plan_path: str | None = None
    workspace_manifest_path: str | None = None
    config_path: str | None = None
    last_evidence_path: str | None = None
    last_diagnosis_path: str | None = None
    message: str = ""


def dump_contract(value: ContractModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_contract(model, path: Path):
    return model.from_dict(json.loads(path.read_text(encoding="utf-8")))
