# InternDataEngine unknown-failure diagnosis contract

Return only the JSON object required by the supplied output schema.

Task plan:
{{TASK_PLAN}}

Normalized evidence:
{{EVIDENCE}}

Rules:

1. This template is used only after deterministic failure routing returns unknown.
2. Structured failure codes, collision audit, object-state events, safety events, predicate results and data
   integrity outrank logs; logs outrank screenshots; screenshots cannot override numerical evidence.
3. Propose one causal change. A Skill update may contain only Agent-owned parameters from its contract.
4. Do not emit coordinates, robot profile values, obstacle exclusions, Prim paths, YAML paths, source-asset edits,
   runtime hot edits, or multiple speculative changes.
5. `workspace_action` is `keep` unless the evidence proves the current robot candidate is infeasible; use `block`
   when no safe change is justified. Typed scene mutations are produced by SceneLayout, not by this response.
6. Confidence must reflect evidence quality. Contradictory or missing evidence requires a low-confidence block or
   request for a named deterministic artifact, not a guessed fix.
7. `failing_subtask_id` must exactly preserve the normalized evidence value. Never infer it from task order,
   object names, screenshots, or prose; leave it null when runtime evidence did not attribute the failure.
