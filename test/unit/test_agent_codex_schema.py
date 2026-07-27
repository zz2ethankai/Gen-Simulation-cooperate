"""Offline tests for the provider-specific Codex structured-output adapter."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent.contracts import Diagnosis, ResolutionResponse, RetentionDecision, TaskPlan
from agent.resolver import AgentDecisionError, _codex_strict_schema, _decode_codex_json_objects


OUTPUT_MODELS = (ResolutionResponse, TaskPlan, Diagnosis, RetentionDecision)


def _object_nodes(value: Any):
    if isinstance(value, dict):
        if value.get("type") == "object" or isinstance(value.get("properties"), dict):
            yield value
        for item in value.values():
            yield from _object_nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _object_nodes(item)


@pytest.mark.parametrize("model", OUTPUT_MODELS)
def test_codex_schema_requires_every_declared_property(model):
    schema = _codex_strict_schema(model.json_schema())

    for node in _object_nodes(schema):
        properties = node.get("properties", {})
        assert node.get("required") == list(properties)
        assert node.get("additionalProperties") is False


@pytest.mark.parametrize("model", OUTPUT_MODELS)
def test_codex_schema_removes_pydantic_defaults(model):
    schema = _codex_strict_schema(model.json_schema())

    def visit(value: Any):
        if isinstance(value, dict):
            assert "default" not in value
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(schema)


def test_internal_contract_schema_keeps_its_optional_defaults():
    schema = ResolutionResponse.json_schema()
    decision = schema["$defs"]["ResolutionDecision"]

    assert "selected_task_id" not in decision["required"]
    assert decision["properties"]["selected_task_id"]["default"] is None


def test_free_form_objects_are_encoded_only_in_provider_schema():
    internal_schema = TaskPlan.json_schema()
    provider_schema = _codex_strict_schema(internal_schema)

    internal_params = internal_schema["$defs"]["SkillStep"]["properties"]["params"]
    provider_params = provider_schema["$defs"]["SkillStep"]["properties"]["params"]
    assert internal_params["type"] == "object"
    assert provider_params["type"] == "string"
    assert "JSON-encoded object" in provider_params["description"]


def test_provider_json_object_strings_are_restored_recursively():
    value = {
        "subtasks": [
            {
                "stages": [
                    {
                        "skills": [
                            {"name": "pick", "params": json.dumps({"test_mode": "forward"})}
                        ]
                    }
                ]
            }
        ],
        "proposed_interface": json.dumps({"argument": "value"}),
    }

    decoded = _decode_codex_json_objects(value)
    assert decoded["subtasks"][0]["stages"][0]["skills"][0]["params"] == {"test_mode": "forward"}
    assert decoded["proposed_interface"] == {"argument": "value"}


@pytest.mark.parametrize("encoded", ["not-json", "[]", "null", "1"])
def test_provider_json_object_strings_reject_invalid_or_non_object_values(encoded):
    with pytest.raises(AgentDecisionError):
        _decode_codex_json_objects({"params": encoded})
