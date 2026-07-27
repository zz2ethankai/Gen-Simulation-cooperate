# InternDataEngine failed-run diagnosis contract

Return only the JSON object required by the supplied output schema.

Task plan:
{{TASK_PLAN}}

Normalized evidence:
{{EVIDENCE}}

Skill parameter contracts:
{{ALLOWED_PARAMS}}

Rules:

1. Structured SimBox failure codes, collision audit, object-state events and safety events outrank images.
2. Do not recommend hiding fixtures, supports, other objects or collision geometry.
3. Distinguish an asset/configuration fault from workspace planning, motion planning, execution tracking,
   missed contact, dropped object and final task-predicate failure.
4. Propose at most one causal change per Agent revision.
5. Skill updates may contain only owner=agent parameters listed above and must obey their type, enum, range
   and length constraints. They must reference an existing subtask, stage and skill index. Never change object
   identity through a parameter update.
6. workspace_action is one of keep, replan, or block.
7. Deterministic, candidate-independent asset faults are not retryable.
