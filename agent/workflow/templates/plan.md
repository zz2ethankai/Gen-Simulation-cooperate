# InternDataEngine semantic planning contract

Return only the JSON object required by the supplied output schema.

Mode: {{MODE}}

User request:
{{USER_PROMPT}}

Candidate existing tasks:
{{CANDIDATES}}

Selected task manifest:
{{SELECTED_MANIFEST}}

Executable robot Skill contracts:
{{SKILL_CONTRACTS}}

Available robot embodiments and deterministic defaults:
{{AGENT_DEFAULTS}}

Rules:

1. Treat the request as a goal. In RESOLVE mode choose `reuse_existing`, `reuse_scene_new_task`, or
   `compose_required` from inventory evidence. Missing task templates do not imply missing scene assets.
2. In PLAN mode copy the selected task and source paths exactly. Use only inventory entity names.
3. Produce an embodiment-independent semantic plan. `robot_requirement.required_capabilities` states what the
   Skills need; `preferred_profile_ids` is optional user intent, not an execution choice. Instance, profile, arm
   binding, placement family and collision mode belong to deterministic `ExecutionVariant`, not TaskPlan.
4. Create one subtask per manipulated object. Use `any_single_arm` unless the request gives a real left/right
   constraint. A normal transfer is a sequential Pick then Place on the same arm; the resolver binds it later.
5. Select only admitted Skills whose required capabilities are present. Do not expand motion phases, attachment,
   collision-world updates, base locking, or safety replanning.
6. Pick has exactly one object. Place has exactly `[manipulated_object, target_object]` and follows Pick.
7. Use only Agent-owned Skill parameters and obey the machine-readable Skill contract. Geometry, collision,
   robot, asset, camera, and compiler-owned values are not semantic planning outputs.
8. `inside` requires a declared container region. `hang` requires explicit affordance evidence. `insert` is unresolved in v1 until the scene declares an insertion axis, minimum depth and terminal orientation and the runtime has an insertion-specific evaluator.
9. Never hide an obstacle or add an arbitrary YAML edit to make planning pass. Record unsupported requirements
   in `unresolved`.
10. Set `task_request.data_generation` only when explicitly requested; otherwise return null.

Task and Skill planning policy:
{{PLANNING_RULES}}

Object-role policy:
{{CENTER_OBJECT_RULES}}
