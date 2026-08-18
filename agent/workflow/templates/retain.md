# InternDataEngine evidence-retention contract

Return only the JSON object required by the supplied output schema.

Run summary:
{{RUN_SUMMARY}}

Rules:

1. One successful repair is only a candidate, never a qualified capability.
2. Keep a playbook only when structured evidence supports a reusable diagnostic rule across runs.
3. Keep a debug tool candidate only when its interface, inputs, outputs, and deterministic verification are clear.
4. Keep a robot Skill candidate only when it represents robot action rather than workflow or scene mutation.
5. Do not retain raw YAML pointers, robot-name branches, source-asset edits, prompt-only numerical rules, or
   conclusions supported only by screenshots, process exit, MP4 presence, or one seed.
6. Use `none` when evidence is incomplete, contradictory, or specific to one scene instance.
