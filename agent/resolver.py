"""Semantic task resolution and structured Codex decisions."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

from workflows.simbox.core.robots.profile import load_robot_profile

from .contracts import (
    ContractModel,
    Diagnosis,
    EvidenceBundle,
    ExecutionVariant,
    ResolutionDecision,
    ResolutionMode,
    ResolutionResponse,
    RobotAdmission,
    SceneCapabilityManifest,
    SceneCompositionRequest,
    SkillContract,
    SkillParameterContract,
    StageExecutionMode,
    TaskPlan,
)
from .robot_skills import (
    RobotSkillContractError,
    load_robot_admissions,
    load_skill_contracts,
    required_skill_capabilities,
    validate_profile_skill_admission,
)
from .robot_skills.relation_admission import (
    RelationAdmissionError,
    validate_relation_admission,
)
from .settings import load_agent_settings


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = Path(__file__).resolve().parent
ROBOT_PROFILE_DIR = REPO_ROOT / "workflows" / "simbox" / "core" / "configs" / "robots"
T = TypeVar("T", bound=ContractModel)


class AgentDecisionError(RuntimeError):
    """Codex output or a semantic decision violated a deterministic contract."""


_CODEX_JSON_OBJECT_FIELDS = frozenset(
    {"object_role_overrides", "params", "proposed_interface"}
)


def _free_form_json_object(field_name: str | None, node: dict[str, Any]) -> dict[str, Any]:
    if field_name not in _CODEX_JSON_OBJECT_FIELDS:
        raise AgentDecisionError(
            f"Codex strict schema cannot encode unconstrained object field: {field_name or '<root>'}"
        )
    description = node.get("description")
    detail = "JSON-encoded object. Return a JSON string whose decoded value is an object."
    return {
        "type": "string",
        "description": f"{description} {detail}".strip() if description else detail,
    }


def _codex_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt a general Pydantic schema to Codex strict structured output.

    Internal contracts keep their defaults and optional-field semantics.  The
    provider schema instead requires every declared property, uses nullable
    value types where Pydantic already declared them, and forbids undeclared
    properties.  Free-form JSON objects cannot be represented safely by the
    strict schema subset, so Codex emits them as JSON strings and the backend
    decodes them before Pydantic validation.
    """

    def visit(value: Any, field_name: str | None = None) -> Any:
        if isinstance(value, list):
            return [visit(item, field_name) for item in value]
        if not isinstance(value, dict):
            return value

        node: dict[str, Any] = {}
        for key, item in value.items():
            if key == "properties" and isinstance(item, dict):
                node[key] = {
                    name: visit(property_schema, name)
                    for name, property_schema in item.items()
                }
            else:
                node[key] = visit(item, field_name)
        node.pop("default", None)

        properties = node.get("properties")
        if node.get("type") == "object" and not isinstance(properties, dict):
            return _free_form_json_object(field_name, node)

        if isinstance(properties, dict):
            node["required"] = list(properties)
            node["additionalProperties"] = False

        all_of = node.get("allOf")
        if isinstance(all_of, list) and len(all_of) == 1 and set(node) <= {"allOf", "title", "description"}:
            replacement = all_of[0]
            if isinstance(replacement, dict):
                replacement = deepcopy(replacement)
                if "title" in node:
                    replacement.setdefault("title", node["title"])
                if "description" in node:
                    replacement.setdefault("description", node["description"])
                return replacement
        return node

    return visit(deepcopy(schema))


def _decode_codex_json_objects(value: Any) -> Any:
    """Restore provider-encoded free-form objects before contract parsing."""

    if isinstance(value, list):
        return [_decode_codex_json_objects(item) for item in value]
    if not isinstance(value, dict):
        return value

    decoded: dict[str, Any] = {}
    for key, item in value.items():
        if key in _CODEX_JSON_OBJECT_FIELDS and isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError as exc:
                raise AgentDecisionError(f"Codex field {key} is not a valid JSON object string: {exc}") from exc
            if not isinstance(item, dict):
                raise AgentDecisionError(f"Codex field {key} must decode to an object")
        decoded[key] = _decode_codex_json_objects(item)
    return decoded

class OpenAIBackend:
    """Call an OpenAI-compatible chat completions API as a structured decision engine.

    Supports any OpenAI-compatible provider (DeepSeek, Groq, etc.) by setting
    *base_url* and *api_key*.  The backend writes the same artifacts as
    ``CodexBackend`` so that downstream debugging and retention are unchanged.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        try:
            from openai import OpenAI  # noqa: F401 — lazy import so codex_cli users are unaffected
        except ImportError as exc:
            raise AgentDecisionError(
                "openai package required for the API backend.  Install it with:  pip install openai"
            ) from exc
        self.model = model
        self._base_url = base_url
        self._api_key = api_key
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            kwargs: dict[str, Any] = {}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(
        self,
        output_model: type[T],
        prompt: str,
        artifact_dir: Path,
        name: str,
        images: list[Path] | None = None,
    ) -> T:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        schema = output_model.json_schema()

        schema_path = artifact_dir / f"{name}.schema.json"
        response_path = artifact_dir / f"{name}.response.json"
        prompt_path = artifact_dir / f"{name}.prompt.txt"
        schema_path.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        prompt_path.write_text(prompt, encoding="utf-8")

        system = (
            "You are a structured decision component of a configuration-first robot task agent. "
            "Return ONLY a valid JSON object. Do not include any explanatory text, markdown fences, "
            "or code blocks before or after the JSON.\n\n"
            "The JSON object must conform to this schema:\n\n"
            f"```json\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n```"
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if images:
            image_contents: list[dict[str, Any]] = []
            for image_path in images:
                import base64

                encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
                suffix = image_path.suffix.lower().lstrip(".")
                media_type = f"image/{suffix}" if suffix in {"png", "jpeg", "jpg", "gif", "webp"} else "image/png"
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                })
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *image_contents],
            })
        else:
            messages.append({"role": "user", "content": prompt})

        prompt_bytes = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
        (artifact_dir / f"{name}.prompt_size.txt").write_text(
            f"approx_chars: {prompt_bytes}\nmodel: {self.model}\n"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=16000,
                timeout=180.0,
            )
        except Exception as exc:
            (artifact_dir / f"{name}.api_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
            raise AgentDecisionError(
                f"API call failed for {name}: {exc}"
            ) from exc

        content = response.choices[0].message.content
        if content is None:
            raise AgentDecisionError(f"empty API response for {name}")

        response_path.write_text(content + "\n", encoding="utf-8")
        (artifact_dir / f"{name}.api_usage.json").write_text(
            json.dumps({
                "model": response.model,
                "usage": response.usage.to_dict() if response.usage else None,
                "finish_reason": response.choices[0].finish_reason,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        try:
            content = content.strip()
            if content.startswith("```"):
                lines = content.splitlines()
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            value = json.loads(content)
            return output_model.from_dict(value)
        except (json.JSONDecodeError, ValueError) as exc:
            raise AgentDecisionError(
                f"invalid structured response for {name}: {exc}"
            ) from exc

class CodexBackend:
    """Run Codex as a read-only structured decision engine."""

    def __init__(self, executable: str = "codex", model: str | None = None):
        self.executable = executable
        self.model = model

    def generate(
        self,
        output_model: type[T],
        prompt: str,
        artifact_dir: Path,
        name: str,
        images: list[Path] | None = None,
    ) -> T:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        schema_path = artifact_dir / f"{name}.schema.json"
        response_path = artifact_dir / f"{name}.response.json"
        schema_path.write_text(
            json.dumps(_codex_strict_schema(output_model.json_schema()), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        command = [
            self.executable,
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(response_path),
            "--cd",
            str(REPO_ROOT),
        ]
        if self.model:
            command.extend(["--model", self.model])
        for image in images or []:
            command.extend(["--image", str(image)])
        command.append("-")
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=REPO_ROOT,
            check=False,
        )
        (artifact_dir / f"{name}.codex_stdout.log").write_text(completed.stdout, encoding="utf-8")
        (artifact_dir / f"{name}.codex_stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0 or not response_path.is_file():
            raise AgentDecisionError(
                f"Codex decision failed ({completed.returncode}); see {name}.codex_stderr.log"
            )
        try:
            value = json.loads(response_path.read_text(encoding="utf-8"))
            value = _decode_codex_json_objects(value)
            return output_model.from_dict(value)
        except (json.JSONDecodeError, ValueError, AgentDecisionError) as exc:
            raise AgentDecisionError(f"invalid structured Codex response for {name}: {exc}") from exc


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_parameter_value(
    skill_name: str,
    parameter_name: str,
    value: Any,
    spec: SkillParameterContract,
) -> None:
    """Validate the small, explicit value vocabulary exposed to the planner."""

    label = f"{skill_name}.{parameter_name}"
    if spec.owner != "agent":
        raise AgentDecisionError(
            f"{label} is owned by {spec.owner}; the Agent must not set it"
        )

    value_type = spec.value_type
    if value_type == "string":
        valid_type = isinstance(value, str)
    elif value_type == "integer":
        valid_type = isinstance(value, int) and not isinstance(value, bool)
    elif value_type == "number":
        valid_type = _is_number(value)
    elif value_type == "boolean":
        valid_type = isinstance(value, bool)
    elif value_type in {"number_list", "ratio_range"}:
        valid_type = isinstance(value, list) and all(_is_number(item) for item in value)
    elif value_type == "direction_filter":
        valid_type = (
            isinstance(value, list)
            and len(value) in {2, 3}
            and isinstance(value[0], str)
            and all(_is_number(item) for item in value[1:])
        )
    elif value_type == "npy_filename":
        valid_type = (
            isinstance(value, str)
            and value.endswith(".npy")
            and "/" not in value
            and "\\" not in value
            and ".." not in value
        )
    else:
        raise AgentDecisionError(f"unsupported parameter contract type for {label}: {value_type}")
    if not valid_type:
        raise AgentDecisionError(f"invalid value type for {label}: expected {value_type}")

    if spec.allowed_values:
        checked_value = value[0] if value_type == "direction_filter" else value
        if checked_value not in spec.allowed_values:
            raise AgentDecisionError(
                f"invalid value for {label}: {checked_value!r}; "
                f"allowed={spec.allowed_values}"
            )
    if isinstance(value, list):
        if spec.min_items is not None and len(value) < spec.min_items:
            raise AgentDecisionError(f"{label} requires at least {spec.min_items} items")
        if spec.max_items is not None and len(value) > spec.max_items:
            raise AgentDecisionError(f"{label} allows at most {spec.max_items} items")
    if _is_number(value):
        if spec.minimum is not None and float(value) < spec.minimum:
            raise AgentDecisionError(f"{label} must be >= {spec.minimum}")
        if spec.maximum is not None and float(value) > spec.maximum:
            raise AgentDecisionError(f"{label} must be <= {spec.maximum}")
    if value_type == "direction_filter" and any(
        float(item) < 0.0 or float(item) > 180.0 for item in value[1:]
    ):
        raise AgentDecisionError(f"{label} angles must be within [0, 180] degrees")
    if value_type == "ratio_range" and float(value[0]) > float(value[1]):
        raise AgentDecisionError(f"{label} must be ordered as [min, max]")


def _effective_arm(skill_arm: str, plan_arm: str) -> str:
    return plan_arm if skill_arm in {"auto", "any_single_arm"} else skill_arm


def _validate_stage_execution(stage, subtask_arm: str) -> None:
    mode = str(getattr(stage.execution_mode, "value", stage.execution_mode))
    arms = [_effective_arm(skill.arm, subtask_arm) for skill in stage.skills]
    explicit_arms = {arm for arm in arms if arm in {"left", "right"}}

    if mode == StageExecutionMode.SINGLE_ARM_SINGLE_SKILL.value:
        if subtask_arm not in {"any_single_arm", "left", "right"} or len(stage.skills) != 1:
            raise AgentDecisionError(
                f"stage {stage.stage_id} single_arm_single_skill requires exactly one Skill on one arm"
            )
        if subtask_arm in {"left", "right"} and explicit_arms != {subtask_arm}:
            raise AgentDecisionError(
                f"stage {stage.stage_id} single_arm_single_skill has inconsistent arm constraints"
            )
        if subtask_arm == "any_single_arm" and len(explicit_arms) > 1:
            raise AgentDecisionError(
                f"stage {stage.stage_id} single_arm_single_skill has inconsistent arm constraints"
            )
        if subtask_arm == "any_single_arm" and explicit_arms and any(
            arm not in {"auto", "any_single_arm", *explicit_arms} for arm in arms
        ):
            raise AgentDecisionError(
                f"stage {stage.stage_id} single_arm_single_skill has inconsistent arm constraints"
            )
        return
    if mode == StageExecutionMode.SINGLE_ARM_SEQUENTIAL.value:
        if subtask_arm not in {"any_single_arm", "left", "right"} or len(stage.skills) != 2:
            raise AgentDecisionError(
                f"stage {stage.stage_id} single_arm_sequential requires two ordered Skills on one arm"
            )
        if subtask_arm in {"left", "right"} and explicit_arms != {subtask_arm}:
            raise AgentDecisionError(
                f"stage {stage.stage_id} single_arm_sequential has inconsistent arm constraints"
            )
        if subtask_arm == "any_single_arm" and len(explicit_arms) > 1:
            raise AgentDecisionError(
                f"stage {stage.stage_id} single_arm_sequential has inconsistent arm constraints"
            )
        if subtask_arm == "any_single_arm" and explicit_arms and any(
            arm not in {"auto", "any_single_arm", *explicit_arms} for arm in arms
        ):
            raise AgentDecisionError(
                f"stage {stage.stage_id} single_arm_sequential has inconsistent arm constraints"
            )
        return
    if mode not in {
        StageExecutionMode.DUAL_ARM_SEQUENTIAL.value,
        StageExecutionMode.DUAL_ARM_SIMULTANEOUS.value,
    }:
        raise AgentDecisionError(f"stage {stage.stage_id} has unsupported execution_mode: {mode}")
    if subtask_arm != "both" or "auto" in arms or set(arms) != {"left", "right"}:
        raise AgentDecisionError(
            f"stage {stage.stage_id} {mode} requires subtask.arm=both and explicit left/right Skills"
        )
    counts = {arm: arms.count(arm) for arm in ("left", "right")}
    if any(not 1 <= count <= 2 for count in counts.values()):
        raise AgentDecisionError(
            f"stage {stage.stage_id} {mode} requires 1-2 Skills per arm"
        )
    if mode == StageExecutionMode.DUAL_ARM_SEQUENTIAL.value:
        arm_switches = sum(left != right for left, right in zip(arms, arms[1:]))
        if arm_switches != 1:
            raise AgentDecisionError(
                f"stage {stage.stage_id} dual_arm_sequential must list one complete arm block, then the other"
            )


def _validate_strict_relation_chain(subtask, manifest: SceneCapabilityManifest) -> None:
    relation = str(getattr(subtask.relation, "value", subtask.relation))
    if len(subtask.stages) != 1:
        raise AgentDecisionError(
            f"subtask {subtask.subtask_id} relation={relation!r} requires one same-arm "
            "Pick followed by Place stage"
        )
    stage = subtask.stages[0]
    mode = str(getattr(stage.execution_mode, "value", stage.execution_mode))
    skill_names = [skill.name.lower() for skill in stage.skills]
    if (
        mode != StageExecutionMode.SINGLE_ARM_SEQUENTIAL.value
        or skill_names != ["pick", "place"]
    ):
        raise AgentDecisionError(
            f"subtask {subtask.subtask_id} relation={relation!r} requires one same-arm "
            "Pick followed by Place stage"
        )
    try:
        validate_relation_admission(
            relation,
            str(subtask.target_object or ""),
            manifest.container_regions,
        )
    except RelationAdmissionError as exc:
        raise AgentDecisionError(str(exc)) from exc


def _load_aliases(path: Path | None = None) -> dict[str, list[str]]:
    source = path or AGENT_DIR / "registry" / "semantic_aliases.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    return {
        str(key).lower(): [str(item).lower() for item in values]
        for key, values in payload.get("aliases", {}).items()
    }


def _manifest_haystack(manifest: SceneCapabilityManifest) -> str:
    values = [manifest.task_id, manifest.scene_id, *manifest.language, *manifest.active_objects]
    for obj in manifest.objects:
        values.extend([obj.name, obj.category, obj.role, *obj.affordances])
    for region in manifest.container_regions:
        values.extend(str(value) for value in region.values() if isinstance(value, (str, int, float)))
    return " ".join(values).lower().replace("_", " ")


def _plan_skill_names(plan: TaskPlan) -> list[str]:
    return [
        skill.name.lower()
        for subtask in plan.subtasks
        for stage in subtask.stages
        for skill in stage.skills
    ]


def _snake_case(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


def _resolved_source_task(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def shortlist_manifests(
    prompt: str,
    manifests: list[SceneCapabilityManifest],
    limit: int = 8,
) -> list[SceneCapabilityManifest]:
    aliases = _load_aliases()
    text = prompt.lower()
    canonical_terms = {key for key, values in aliases.items() if key in text or any(alias in text for alias in values)}
    tokens = set(re.findall(r"[a-z0-9]+", text.replace("_", " "))) | canonical_terms
    ranked: list[tuple[int, str, SceneCapabilityManifest]] = []
    for manifest in manifests:
        haystack = _manifest_haystack(manifest)
        score = sum(2 for token in tokens if token and token in haystack)
        task_text = manifest.task_id.lower().replace("_", " ")
        score += sum(4 for token in tokens if token and token in task_text)
        score += sum(
            3
            for obj in manifest.objects
            for token in tokens
            if token and (token in obj.name.lower() or token == obj.category.lower())
        )
        ranked.append((score, manifest.task_id, manifest))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    if ranked and ranked[0][0] == 0:
        return [item[2] for item in ranked]
    return [item[2] for item in ranked[:limit]]


def _compact_manifest(manifest: SceneCapabilityManifest) -> dict[str, Any]:
    return {
        "task_id": manifest.task_id,
        "scene_id": manifest.scene_id,
        "source_task": manifest.source_task,
        "robot_instances": [item.to_dict() for item in manifest.robot_instances],
        "active_objects": manifest.active_objects,
        "objects": [
            {
                "name": item.name,
                "category": item.category,
                "role": item.role,
                "rigid_body": item.rigid_body,
                "collision_enabled": item.collision_enabled,
                "attach_proxy_status": item.attach_proxy_status,
                "affordances": item.affordances,
                "parent_fixture": item.parent_fixture,
            }
            for item in manifest.objects
        ],
        "container_regions": manifest.container_regions,
        "physics_readiness": manifest.physics_readiness,
    }


class TaskResolver:
    def __init__(
        self,
        backend: CodexBackend|OpenAIBackend,
        skill_contracts: dict[str, SkillContract] | None = None,
        robot_admissions: Mapping[tuple[str, str, str], RobotAdmission] | None = None,
        settings: dict[str, Any] | None = None,
    ):
        self.backend = backend
        self.skill_contracts = skill_contracts or load_skill_contracts()
        self.robot_admissions = robot_admissions or load_robot_admissions()
        self.settings = settings or load_agent_settings()

    def resolve(
        self,
        prompt: str,
        manifests: list[SceneCapabilityManifest],
        artifact_dir: Path,
    ) -> tuple[ResolutionResponse, list[SceneCapabilityManifest]]:
        candidates = shortlist_manifests(prompt, manifests)
        template = (AGENT_DIR / "workflow" / "templates" / "plan.md").read_text(
            encoding="utf-8"
        )
        decision_prompt = template.replace("{{MODE}}", "RESOLVE").replace("{{USER_PROMPT}}", prompt).replace(
            "{{CANDIDATES}}",
            json.dumps([_compact_manifest(item) for item in candidates], ensure_ascii=False, indent=2),
        ).replace("{{SELECTED_MANIFEST}}", "null").replace("{{SKILL_CONTRACTS}}", "[]").replace(
            "{{PLANNING_RULES}}", "Not used in RESOLVE mode."
        ).replace("{{CENTER_OBJECT_RULES}}", "Not used in RESOLVE mode.").replace(
            "{{AGENT_DEFAULTS}}", json.dumps(self._prompt_defaults(), ensure_ascii=False, indent=2)
        )
        response = self.backend.generate(
            ResolutionResponse,
            decision_prompt,
            artifact_dir,
            "resolution",
        )
        if response.resolution.mode in {
            ResolutionMode.REUSE_EXISTING,
            ResolutionMode.REUSE_SCENE_NEW_TASK,
        }:
            self.select_source_manifest(response.resolution, candidates)
        return response, candidates

    def select_source_manifest(
        self,
        decision: ResolutionDecision,
        candidates: list[SceneCapabilityManifest],
    ) -> SceneCapabilityManifest:
        if decision.mode not in {
            ResolutionMode.REUSE_EXISTING,
            ResolutionMode.REUSE_SCENE_NEW_TASK,
        }:
            raise AgentDecisionError(
                f"resolution mode {decision.mode!r} does not select an existing scene source"
            )
        if not decision.selected_task_id or not decision.selected_source_task:
            raise AgentDecisionError(
                f"{decision.mode} requires selected_task_id and selected_source_task"
            )
        matches = [
            manifest
            for manifest in candidates
            if manifest.task_id == decision.selected_task_id
            and _resolved_source_task(manifest.source_task)
            == _resolved_source_task(decision.selected_source_task)
        ]
        if len(matches) != 1:
            raise AgentDecisionError(
                "selected_task_id and selected_source_task must identify exactly one candidate"
            )
        selected = matches[0]
        if decision.mode == ResolutionMode.REUSE_SCENE_NEW_TASK:
            if not decision.selected_scene_id:
                raise AgentDecisionError(
                    "REUSE_SCENE_NEW_TASK requires selected_scene_id"
                )
            if selected.scene_id != decision.selected_scene_id:
                raise AgentDecisionError(
                    "selected scene source does not belong to selected_scene_id"
                )
        elif decision.selected_scene_id and selected.scene_id != decision.selected_scene_id:
            raise AgentDecisionError(
                "selected scene source does not belong to selected_scene_id"
            )
        return selected

    def plan(
        self,
        prompt: str,
        resolution: ResolutionResponse,
        selected: SceneCapabilityManifest,
        artifact_dir: Path,
    ) -> TaskPlan:
        template = (AGENT_DIR / "workflow" / "templates" / "plan.md").read_text(
            encoding="utf-8"
        )
        planning_rules = (
            AGENT_DIR / "workflow" / "task_planning_policy.md"
        ).read_text(encoding="utf-8")
        center_object_rules = (
            AGENT_DIR / "workflow" / "object_role_policy.md"
        ).read_text(encoding="utf-8")
        skill_values = [item.to_dict() for item in self.skill_contracts.values()]
        defaults = self._prompt_defaults(selected)
        plan_prompt = template.replace("{{MODE}}", "PLAN").replace("{{USER_PROMPT}}", prompt).replace(
            "{{CANDIDATES}}", "[]"
        ).replace(
            "{{SELECTED_MANIFEST}}",
            json.dumps(_compact_manifest(selected), ensure_ascii=False, indent=2),
        ).replace(
            "{{SKILL_CONTRACTS}}",
            json.dumps(skill_values, ensure_ascii=False, indent=2),
        ).replace(
            "{{PLANNING_RULES}}",
            planning_rules,
        ).replace(
            "{{CENTER_OBJECT_RULES}}",
            center_object_rules,
        ).replace(
            "{{AGENT_DEFAULTS}}",
            json.dumps(defaults, ensure_ascii=False, indent=2),
        )
        plan = self.backend.generate(TaskPlan, plan_prompt, artifact_dir, "task_plan")
        plan = self.normalize_semantic_plan(plan)
        self.validate_plan(plan, selected)
        if not self.execution_variants(plan):
            raise AgentDecisionError(
                "no admitted robot execution variant satisfies the semantic TaskPlan"
            )
        return plan

    def _prompt_defaults(
        self,
        manifest: SceneCapabilityManifest | None = None,
    ) -> dict[str, Any]:
        generation = self.settings.get("generation", {})
        defaults: dict[str, Any] = {
            "generation": {"enabled": bool(generation.get("enabled", True))},
            "robot": {"selection": "choose one robot_instances entry from the selected manifest"},
        }
        if manifest is None:
            return defaults
        defaults["robot"] = {
            "source_instances": [
                {
                    "instance_name": item.instance_name,
                    "profile_id": item.profile_id,
                    "placement_family": item.placement_family,
                    "available_arms": item.available_arms,
                    "capabilities": item.capabilities,
                }
                for item in manifest.robot_instances
            ],
            "semantic_arm_constraint": "any_single_arm",
        }
        return defaults

    def normalize_semantic_plan(self, plan: TaskPlan) -> TaskPlan:
        data = plan.to_dict()
        skill_names = _plan_skill_names(plan)
        try:
            required = required_skill_capabilities(skill_names, self.skill_contracts)
        except RobotSkillContractError as exc:
            raise AgentDecisionError(str(exc)) from exc
        data["robot_requirement"]["required_capabilities"] = sorted(required)
        return TaskPlan.from_dict(data)

    def execution_variants(
        self,
        plan: TaskPlan,
        profile_paths: list[Path] | None = None,
        *,
        admission_states: frozenset[str] = frozenset({"admitted", "qualified"}),
    ) -> list[ExecutionVariant]:
        configured_planning = self.settings.get("planning", {})
        collision_world = (
            configured_planning.get("collision_world", {})
            if isinstance(configured_planning, dict)
            else {}
        )
        collision_world_mode = str(collision_world.get("mode", ""))
        paths = profile_paths or sorted(ROBOT_PROFILE_DIR.glob("*.yaml"))
        preferred = set(plan.robot_requirement.preferred_profile_ids)
        skill_names = _plan_skill_names(plan)
        variants: list[ExecutionVariant] = []
        for path in paths:
            profile = load_robot_profile(path)
            if preferred and profile.profile_id not in preferred:
                continue
            try:
                validate_profile_skill_admission(
                    profile,
                    skill_names,
                    collision_world_mode,
                    contracts=self.skill_contracts,
                    admissions=self.robot_admissions,
                    allowed_states=admission_states,
                )
            except RobotSkillContractError:
                continue

            arm_options: list[tuple[str, ...]] = []
            for subtask in plan.subtasks:
                if subtask.arm in {"left", "right"}:
                    choices = (subtask.arm,) if subtask.arm in profile.arms else ()
                elif subtask.arm == "any_single_arm":
                    explicit = {
                        skill.arm
                        for stage in subtask.stages
                        for skill in stage.skills
                        if skill.arm in {"left", "right"}
                    }
                    choices = (
                        tuple(sorted(explicit & set(profile.arms)))
                        if explicit
                        else tuple(sorted(profile.arms))
                    )
                else:
                    choices = ()
                if not choices:
                    arm_options = []
                    break
                arm_options.append(choices)
            if not arm_options:
                continue

            try:
                config_file = str(path.resolve().relative_to(REPO_ROOT))
            except ValueError:
                config_file = str(path.resolve())
            for arms in itertools.product(*arm_options):
                binding = {
                    subtask.subtask_id: arm
                    for subtask, arm in zip(plan.subtasks, arms, strict=True)
                }
                binding_id = "__".join(
                    f"{subtask_id}-{arm}" for subtask_id, arm in binding.items()
                )
                variants.append(
                    ExecutionVariant(
                        variant_id=f"{profile.profile_id}__{binding_id}",
                        instance_name=_snake_case(profile.target_class),
                        profile_id=profile.profile_id,
                        robot_config_file=config_file,
                        placement_family=str(
                            getattr(profile.placement.family, "value", profile.placement.family)
                        ),
                        profile_hash=profile.profile_hash,
                        collision_world_mode=collision_world_mode,
                        arm_binding=binding,
                    )
                )
        return sorted(variants, key=lambda item: item.variant_id)

    def composition_request(
        self,
        prompt: str,
        decision: ResolutionDecision,
        candidates: list[SceneCapabilityManifest],
    ) -> SceneCompositionRequest:
        return SceneCompositionRequest(
            prompt=prompt,
            suggested_base_scene=candidates[0].scene_id if candidates else None,
            required_assets=[],
            required_relations=[],
            required_robot_capabilities=["pick", "place"],
            missing_capabilities=decision.missing_capabilities,
            validation_requirements=[
                "Physics CollisionAPI audit",
                "RigidBody and attach proxy audit",
                "workspace geometry validation",
                "CuRobo planning-only validation",
            ],
        )

    def build_synthetic_manifest(
        self,
        decision: ResolutionDecision,
        template: SceneCapabilityManifest,
    ) -> SceneCapabilityManifest:
        if decision.mode != ResolutionMode.REUSE_SCENE_NEW_TASK:
            raise AgentDecisionError(
                "synthetic manifests require reuse_scene_new_task resolution"
            )
        if (
            decision.selected_task_id != template.task_id
            or not decision.selected_source_task
            or _resolved_source_task(decision.selected_source_task)
            != _resolved_source_task(template.source_task)
            or decision.selected_scene_id != template.scene_id
        ):
            raise AgentDecisionError(
                "synthetic manifest template does not match the selected scene source"
            )

        synthetic = SceneCapabilityManifest.from_dict(template.to_dict())
        synthetic_identity = json.dumps(
            {
                "source_task": _resolved_source_task(template.source_task).as_posix(),
                "scene_id": template.scene_id,
                "object_role_overrides": decision.object_role_overrides,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        synthetic.task_id = "synthetic_" + hashlib.sha256(
            synthetic_identity.encode("utf-8")
        ).hexdigest()

        overrides = decision.object_role_overrides
        for obj in synthetic.objects:
            if obj.name in overrides:
                obj.role = overrides[obj.name]

        manipulated_names = {
            name for name, role in overrides.items()
            if role in {"manipulated", "target"}
        }
        synthetic.active_objects = sorted(
            set(template.active_objects) | manipulated_names
        )

        for obj in synthetic.objects:
            if obj.role in {"target", "manipulated"}:
                if "pickable" not in obj.affordances:
                    obj.affordances = sorted(set(obj.affordances) | {"pickable"})
            if obj.role == "target":
                has_container = any(
                    str(r.get("object") or r.get("target")) == obj.name
                    for r in synthetic.container_regions
                )
                if has_container and "container" not in obj.affordances:
                    obj.affordances = sorted(set(obj.affordances) | {"container"})

        active_caps = [
            obj for obj in synthetic.objects
            if obj.name in synthetic.active_objects
        ]
        synthetic.physics_readiness = (
            "basic_ready"
            if all(o.rigid_body and o.collision_enabled for o in active_caps)
            else "runtime_audit_required"
        )

        return synthetic

    def diagnose_unknown(
        self,
        plan: TaskPlan,
        evidence: EvidenceBundle,
        artifact_dir: Path,
        images: list[Path] | None = None,
    ) -> Diagnosis:
        template = (AGENT_DIR / "workflow" / "templates" / "diagnose.md").read_text(
            encoding="utf-8"
        )
        prompt = template.replace(
            "{{TASK_PLAN}}", json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)
        ).replace(
            "{{EVIDENCE}}", json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2)
        ).replace(
            "{{ALLOWED_PARAMS}}",
            json.dumps(
                {name: item.to_dict() for name, item in self.skill_contracts.items()},
                ensure_ascii=False,
                indent=2,
            ),
        )
        return self.backend.generate(Diagnosis, prompt, artifact_dir, "diagnosis_llm", images=images)

    def validate_plan(self, plan: TaskPlan, manifest: SceneCapabilityManifest) -> None:
        if (
            plan.selected_task_id != manifest.task_id
            or _resolved_source_task(plan.source_task)
            != _resolved_source_task(manifest.source_task)
        ):
            raise AgentDecisionError("TaskPlan source does not match selected inventory manifest")
        skill_names = _plan_skill_names(plan)
        try:
            required_capabilities = required_skill_capabilities(
                skill_names, self.skill_contracts
            )
        except RobotSkillContractError as exc:
            raise AgentDecisionError(str(exc)) from exc
        if frozenset(plan.robot_requirement.required_capabilities) != required_capabilities:
            raise AgentDecisionError(
                "TaskPlan robot_requirement.required_capabilities must equal the deterministic "
                f"Skill requirements: {sorted(required_capabilities)}"
            )
        object_names = {item.name for item in manifest.objects}
        if not plan.subtasks:
            raise AgentDecisionError("TaskPlan must contain at least one object subtask")
        for subtask in plan.subtasks:
            if subtask.arm not in {"any_single_arm", "left", "right", "both"}:
                raise AgentDecisionError(f"invalid subtask arm: {subtask.arm}")
            if subtask.arm == "both" and not plan.unresolved:
                raise AgentDecisionError(
                    f"subtask {subtask.subtask_id} requires both arms but the current executable "
                    "workspace contract requires a fixed left/right arm; record it as unresolved"
                )
            if subtask.manipulated_object not in object_names:
                raise AgentDecisionError(f"unknown manipulated object: {subtask.manipulated_object}")
            if subtask.target_object and subtask.target_object not in object_names:
                raise AgentDecisionError(f"unknown target object: {subtask.target_object}")
            for stage in subtask.stages:
                _validate_stage_execution(stage, subtask.arm)
                seen_pick: dict[str, bool] = {
                    "any_single_arm": False,
                    "left": False,
                    "right": False,
                }
                for skill in stage.skills:
                    name = skill.name.lower()
                    contract = self.skill_contracts.get(name)
                    if contract is None:
                        raise AgentDecisionError(f"unsupported skill in v1: {name}")
                    if len(skill.objects) != contract.object_count:
                        raise AgentDecisionError(
                            f"{name} requires {contract.object_count} objects, got {len(skill.objects)}"
                        )
                    if any(item not in object_names for item in skill.objects):
                        raise AgentDecisionError(f"{name} references an unknown object")
                    if skill.arm not in {"auto", "any_single_arm", "left", "right"}:
                        raise AgentDecisionError(f"invalid arm: {skill.arm}")
                    unknown_params = set(skill.params) - set(contract.allowed_params)
                    if unknown_params:
                        raise AgentDecisionError(f"unsupported {name} params: {sorted(unknown_params)}")
                    for parameter_name, value in skill.params.items():
                        _validate_parameter_value(
                            name,
                            parameter_name,
                            value,
                            contract.parameters[parameter_name],
                        )

                    arm = _effective_arm(skill.arm, subtask.arm)
                    if name == "pick":
                        if skill.objects != [subtask.manipulated_object]:
                            raise AgentDecisionError(
                                f"Pick objects must be [{subtask.manipulated_object!r}]"
                            )
                        seen_pick[arm] = True
                        minimum = skill.params.get("post_grasp_offset_min")
                        maximum = skill.params.get("post_grasp_offset_max")
                        if minimum is not None and maximum is not None and minimum > maximum:
                            raise AgentDecisionError(
                                "pick.post_grasp_offset_min must not exceed post_grasp_offset_max"
                            )
                    if name == "place":
                        if subtask.target_object is None:
                            raise AgentDecisionError("Place requires a subtask target_object")
                        expected_objects = [subtask.manipulated_object, subtask.target_object]
                        if skill.objects != expected_objects:
                            raise AgentDecisionError(
                                f"Place objects must be {expected_objects} in that order"
                            )
                        if not seen_pick[arm]:
                            raise AgentDecisionError(
                                "Place must follow Pick on the same arm in the same object stage"
                            )
                        relation = str(getattr(subtask.relation, "value", subtask.relation))
                        success_mode = skill.params.get("success_mode")
                        if relation in {"on", "inside", "insert"} and success_mode is not None:
                            raise AgentDecisionError(
                                f"place.success_mode is compiler-owned for relation={relation!r}"
                            )
                        relation_success_modes = {
                            "left_of": {"left"},
                            "right_of": {"right"},
                            "next_to": {"left", "right"},
                        }
                        allowed_modes = relation_success_modes.get(relation)
                        if success_mode is not None and allowed_modes and success_mode not in allowed_modes:
                            raise AgentDecisionError(
                                f"place.success_mode={success_mode!r} does not match relation={relation!r}; "
                                f"allowed={sorted(allowed_modes)}"
                            )
                        direction = skill.params.get("place_direction")
                        if direction is None and relation in {"hang", "insert"}:
                            direction = "horizontal"
                        if direction == "horizontal":
                            required = {"align_place_obj_axis", "offset_place_obj_axis"}
                            missing = required - set(skill.params)
                            if missing:
                                raise AgentDecisionError(
                                    f"horizontal Place requires params: {sorted(missing)}"
                                )
            _validate_strict_relation_chain(subtask, manifest)
