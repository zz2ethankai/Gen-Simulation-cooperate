from __future__ import annotations

from eval.policies.http_policy import HttpJsonPolicyClient
from eval.policies.local import ConstantActionPolicy, ZeroActionPolicy
from eval.policies.starvla_msgpack import StarVLAMsgpackPolicyClient
from eval.policies.websocket_policy import JsonWebSocketPolicyClient, OpenPIWebSocketPolicyClient
from eval.specs import PolicySpec


def create_policy(spec: PolicySpec):
    if spec.policy_type == "zero_action":
        if spec.action_dim is None:
            raise ValueError("zero_action policy requires policy.action_dim.")
        return ZeroActionPolicy(action_dim=spec.action_dim)
    if spec.policy_type == "constant_action":
        return ConstantActionPolicy(action=list(spec.policy_args["action"]))
    if spec.policy_type == "http_json":
        if not spec.endpoint:
            raise ValueError("http_json policy requires policy.endpoint.")
        return HttpJsonPolicyClient(
            endpoint=spec.endpoint,
            timeout_s=float(spec.policy_args.get("timeout_s", 30.0)),
        )
    if spec.policy_type == "json_websocket":
        if not spec.endpoint:
            raise ValueError("json_websocket policy requires policy.endpoint.")
        return JsonWebSocketPolicyClient(
            endpoint=spec.endpoint,
            timeout_s=float(spec.policy_args.get("timeout_s", 30.0)),
        )
    if spec.policy_type == "openpi_websocket":
        if not spec.host or spec.port is None:
            raise ValueError("openpi_websocket policy requires policy.host and policy.port.")
        return OpenPIWebSocketPolicyClient(host=spec.host, port=spec.port)
    if spec.policy_type == "starvla_msgpack":
        if not spec.host or spec.port is None:
            raise ValueError("starvla_msgpack policy requires policy.host and policy.port.")
        return StarVLAMsgpackPolicyClient(
            host=spec.host,
            port=spec.port,
            timeout_s=float(spec.policy_args.get("timeout_s", 300.0)),
            unnorm_key=spec.policy_args.get("unnorm_key"),
            image_keys=list(spec.policy_args.get("image_keys", [])),
            state_key=spec.policy_args.get("state_key"),
            state_keys=list(spec.policy_args.get("state_keys", [])),
            prompt_key=spec.policy_args.get("prompt_key", "detailed_prompt"),
            request_args=dict(spec.policy_args.get("request_args", {})),
            action_slice=spec.policy_args.get("action_slice"),
        )
    raise ValueError(f"Unsupported policy_type: {spec.policy_type}")
