# InternDataEngine experience-retention contract

Return only the JSON object required by the supplied output schema.

Run summary:
{{RUN_SUMMARY}}

Choose exactly one kind:

- playbook: a cross-task debugging rule or evidence-driven investigation method;
- debug_tool: a repeatable deterministic inspection that should become a script or module;
- robot_skill: a stable change to robot execution behavior, categorized under pick/place/perception/etc.;
- none: insufficient evidence, a one-off random event, or no reusable lesson.

Do not call an Agent/Codex system skill a robot_skill. A robot_skill must change runtime robot behavior and
must remain a candidate until cross-sample promotion gates pass.

For debug_tool or robot_skill, include a minimal reusable candidate implementation in files. Paths must be
relative, must not contain `..`, and should normally include implementation, README and a test proposal.
Robot Skill code must use the existing structured motion/collision/safety contracts and must not edit the
active Skill registry. Keep files empty for playbook or none.
