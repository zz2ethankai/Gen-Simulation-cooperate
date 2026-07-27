"""Semantic task resolution and structured Codex decisions."""

from __future__ import annotations

import json
import re
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar
import uuid
import yaml

from .contracts import (
    ContractModel,
    Diagnosis,
    EvidenceBundle,
    ResolutionDecision,
    ResolutionMode,
    ResolutionResponse,
    SceneCapabilityManifest,
    SceneCompositionRequest,
    SkillContract,
    SkillParameterContract,
    StageExecutionMode,
    TaskPlan,
)
from .settings import load_agent_settings


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = Path(__file__).resolve().parent
T = TypeVar("T", bound=ContractModel)


class AgentDecisionError(RuntimeError):
    """Codex output or a semantic decision violated a deterministic contract."""


_CODEX_JSON_OBJECT_FIELDS = frozenset({"params", "proposed_interface"})


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


def load_skill_contracts(path: Path | None = None) -> dict[str, SkillContract]:
    source = path or AGENT_DIR / "registry" / "skill_contracts.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    result = {}
    for item in payload.get("skills", []):
        contract = SkillContract.from_dict(item)
        result[contract.name] = contract
    return result


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
    return plan_arm if skill_arm == "auto" else skill_arm


def _validate_stage_execution(stage, subtask_arm: str) -> None:
    mode = str(getattr(stage.execution_mode, "value", stage.execution_mode))
    arms = [_effective_arm(skill.arm, subtask_arm) for skill in stage.skills]
    explicit_arms = {arm for arm in arms if arm in {"left", "right"}}

    if mode == StageExecutionMode.SINGLE_ARM_SINGLE_SKILL.value:
        if subtask_arm not in {"left", "right"} or len(stage.skills) != 1 or explicit_arms != {subtask_arm}:
            raise AgentDecisionError(
                f"stage {stage.stage_id} single_arm_single_skill requires exactly one Skill on one arm"
            )
        return
    if mode == StageExecutionMode.SINGLE_ARM_SEQUENTIAL.value:
        if subtask_arm not in {"left", "right"} or len(stage.skills) != 2 or explicit_arms != {subtask_arm}:
            raise AgentDecisionError(
                f"stage {stage.stage_id} single_arm_sequential requires two ordered Skills on one arm"
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
        "robots": manifest.robots,
        "active_objects": manifest.active_objects,
        "robot_mounting": manifest.robot_mounting,
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
        settings: dict[str, Any] | None = None,
    ):
        self.backend = backend
        self.skill_contracts = skill_contracts or load_skill_contracts()
        self.settings = settings or load_agent_settings()

    def resolve(
        self,
        prompt: str,
        manifests: list[SceneCapabilityManifest],
        artifact_dir: Path,
    ) -> tuple[ResolutionResponse, list[SceneCapabilityManifest]]:
        candidates = shortlist_manifests(prompt, manifests)
        template = (AGENT_DIR / "prompts" / "plan.md").read_text(encoding="utf-8")
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
        candidate_by_id = {item.task_id: item for item in candidates}
        selected_id = response.resolution.selected_task_id
        if response.resolution.mode == ResolutionMode.REUSE_EXISTING:
            if selected_id not in candidate_by_id:
                raise AgentDecisionError(f"Codex selected a task outside the candidate set: {selected_id}")
            selected = candidate_by_id[selected_id]
            if response.resolution.selected_source_task != selected.source_task:
                raise AgentDecisionError("selected_source_task does not match inventory")
        elif response.resolution.mode == ResolutionMode.REUSE_SCENE_NEW_TASK:
            if not response.resolution.selected_scene_id:
                raise AgentDecisionError(
                    "REUSE_SCENE_NEW_TASK requires selected_scene_id"
                )
            scene_ids = {item.scene_id for item in candidates}
            if response.resolution.selected_scene_id not in scene_ids:
                raise AgentDecisionError(
                    f"selected_scene_id {response.resolution.selected_scene_id!r} "
                    f"is not in the candidate set"
                )
        return response, candidates

    def plan(
        self,
        prompt: str,
        resolution: ResolutionResponse,
        selected: SceneCapabilityManifest,
        artifact_dir: Path,
    ) -> TaskPlan:
        template = (AGENT_DIR / "prompts" / "plan.md").read_text(encoding="utf-8")
        planning_rules = (
            AGENT_DIR / "prompts" / "Agent任务规划与Skill编排规范.md"
        ).read_text(encoding="utf-8")
        center_object_rules = (
            AGENT_DIR / "prompts" / "Agent中心物品选择与机器人初始位姿生成规范.md"
        ).read_text(encoding="utf-8")
        skill_values = [item.to_dict() for item in self.skill_contracts.values()]
        defaults = self._prompt_defaults()
        if selected.robots:
            robot_type = selected.robots[0]
            defaults["robot"]["default_type"] = robot_type
            allowed = defaults["robot"].get("allowed_profiles", {}).get(robot_type, [])
            if allowed:
                defaults["robot"]["default_profile"] = allowed[0]
            arms = defaults["robot"].get("arms", {})
            if robot_type in arms:
                defaults["robot"]["default_arm"] = arms[robot_type]
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
        self.validate_plan(plan, selected)
        return plan

    def _prompt_defaults(self) -> dict[str, Any]:
        generation = self.settings.get("generation", {})
        robot = self.settings.get("robot", {})
        return {
            "generation": {"enabled": bool(generation.get("enabled", True))},
            "robot": {
                "default_type": robot.get("default_type", "split_aloha"),
                "default_profile": robot.get("default_profile", "split_aloha_tabletop_v1"),
                "default_arm": robot.get("default_arm", "right"),
                "allowed_profiles": robot.get("allowed_profiles", {}),
                "arms": robot.get("arms", {}),
            },
        }

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
            required_robot_capabilities=["split_aloha_manipulation"],
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
        candidates: list[SceneCapabilityManifest],
    ) -> SceneCapabilityManifest:
        scene_id = decision.selected_scene_id
        scene_manifests = [m for m in candidates if m.scene_id == scene_id]
        if not scene_manifests:
            raise AgentDecisionError(f"no manifest found for scene: {scene_id}")

        if decision.selected_task_id:
            template = next(
                (m for m in scene_manifests if m.task_id == decision.selected_task_id),
                scene_manifests[0],
            )
        else:
            template = scene_manifests[0]

        synthetic = SceneCapabilityManifest.from_dict(template.to_dict())
        synthetic.task_id = f"{template.task_id}__synthetic_{uuid.uuid4().hex[:8]}"

        overrides = decision.object_role_overrides
        for obj in synthetic.objects:
            if obj.name in overrides:
                obj.role = overrides[obj.name]

        manipulated_names = {
            name for name, role in overrides.items()
            if role in {"manipulated", "target"}
        }
        synthetic.active_objects = list(
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
        template = (AGENT_DIR / "prompts" / "diagnose.md").read_text(encoding="utf-8")
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
        if plan.selected_task_id != manifest.task_id or Path(plan.source_task) != Path(manifest.source_task):
            raise AgentDecisionError("TaskPlan source does not match selected inventory manifest")
        if plan.robot.robot_type not in manifest.robots:
            raise AgentDecisionError(f"robot {plan.robot.robot_type!r} is not available in the selected manifest")
        allowed_profiles = (
            self.settings.get("robot", {}).get("allowed_profiles", {}).get(plan.robot.robot_type, [])
        )
        if plan.robot.robot_profile not in allowed_profiles:
            raise AgentDecisionError(
                f"robot profile {plan.robot.robot_profile!r} is not allowed for {plan.robot.robot_type}; "
                f"allowed={allowed_profiles}"
            )
        object_names = {item.name for item in manifest.objects}
        if not plan.subtasks:
            raise AgentDecisionError("TaskPlan must contain at least one object subtask")
        for subtask in plan.subtasks:
            if subtask.arm not in {"left", "right", "both"}:
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
                seen_pick: dict[str, bool] = {"auto": False, "left": False, "right": False}
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
                    if skill.arm not in {"auto", "left", "right"}:
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
                        relation_success_modes = {
                            "on": {"xybbox", "3diou"},
                            "inside": {"xybbox", "3diou"},
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
