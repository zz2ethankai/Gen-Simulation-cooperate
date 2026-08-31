# Temp Agent Merge TODO

This merge keeps the local renderer and simulator behavior authoritative while
retaining code that the Agent directly calls.

`workflows/simbox/core/configs/robots/split_aloha_actual.yaml` is deprecated and
is not an active runtime configuration. Agent integration and contract checks
must use `split_aloha.yaml`; the old configuration remains only for historical
physical-4WIS probes.

- [x] Resolve the Agent runtime conflicts in `template_controller.py`,
  `template_robot.py`, `pick.py`, `place.py`, and `simbox_dual_workflow.py` by
  preserving local navigation, virtual-base, Pick/Place, reset, and logging
  behavior, then adding only the `physics_schema` and evidence hooks required by
  the Agent.
- [ ] Run the Agent workspace probe end to end with
  `configs/de_workspace_probe_template.yaml`; verify that the host GPU is assigned
  by Compose while container-local `active_gpu` and `physics_gpu` remain `0`.
- [ ] Runtime-validate SplitAloha `manipulation_base_hold` against the local
  virtual-base asset. Confirm the three base joints remain fixed during Agent
  Pick/Place and that Nav2 control resumes without a pose jump afterward.
- [ ] Run one Agent-generated task through `scripts/docker/up_simbox_isaac.sh` with `TASK_CONFIG`
  and verify strict success from the task log, episode events, LMDB metadata,
  and skill snapshots.
- [ ] Validate one Agent-generated task that uses an object-level `asset_root`;
  confirm object USD and texture paths resolve without changing the task-level
  HDR environment-map lookup.
- [x] Confirm that Agent execution does not require the remote `video_only`
  camera/logger path. The merge intentionally retains the local camera and LMDB
  logger implementations; no Agent runtime code calls `add_video_frame`.
- [ ] If pure-Python USD tests need the PyPI `usd-core` package, add it to a
  separate development environment. Do not install it through the main
  requirements because Isaac Sim supplies its own USD/PXR build.
- [ ] Treat the remote Docker parallel-generation stack and its sample configs
  as a separate integration. They are documented by the remote migration guide
  but are not called by the Agent orchestrator.
- [ ] Runtime-validate Docker-only failure cleanup: force one timeout and confirm
  the attempt's Isaac/Nav2 containers, Compose network, and ROS-domain lock are
  released without falling back to a host Isaac process.
