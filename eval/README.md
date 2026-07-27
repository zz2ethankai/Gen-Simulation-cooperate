# InternDataEngine Eval Module

This module runs closed-loop evaluation for external action policies.

The first target is VLA-style action policies served outside Isaac Sim. The
simulation process owns the environment loop; the policy process owns model
inference. This separation avoids dependency conflicts between Isaac Sim and
model-training stacks.

## Scope

- `mock` env: validates the runner/result loop without Isaac Sim.
- `simbox` env: thin adapter for one SimBox task YAML.
- `constant_action`, `http_json`, `json_websocket`, `openpi_websocket`, and
  `starvla_msgpack` policy clients.
- Sequential suite execution. Parallelism is intentionally left for a later
  runner backend.

## Run Smoke Test

```bash
python -m eval.cli.eval_policy --config eval/configs/smoke_mock.yaml
```

The run writes:

```text
outputs/eval/<eval_name>/<run_id>/
  config.json
  episodes.jsonl
  summary.json
```

## SimBox Template

`eval/configs/simbox_single_pick.yaml` points to an existing simple SimBox task:

```text
workflows/simbox/core/configs/tasks/pick_and_place/split_aloha/single_pick/left/omniobject3d-banana.yaml
```

The runner does not hard-code this task. Change `task.task_config` to evaluate a
different task YAML.

For real VLA deployment, fill in:

- `policy.endpoint`, `policy.host`, or `policy.port`
- `task.env_args.action.robot_name`
- `task.env_args.action.joint_indices`
- `task.env_args.success_predicate`

The first real success predicate is `object_lifted`, which compares the target
object's current z position against its reset-time z position.

## StarVLA Same-Distribution Smoke

StarVLA's native deployment server is msgpack-over-websocket, not JSON
websocket. Start it in a separate StarVLA environment:

```bash
CKPT_PATH=/path/to/starvla/checkpoint \
GPU_ID=0 \
PORT=5694 \
bash eval/scripts/start_starvla_server.sh
```

Then run the InternData eval loop from the Isaac/InternData environment:

```bash
python -m eval.cli.eval_policy --config eval/configs/starvla_same_dist.yaml
```

If the server metadata reports multiple `available_unnorm_keys`, set
`policy.policy_args.unnorm_key` in `eval/configs/starvla_same_dist.yaml`.

For the Franka banana pick checkpoint that uses
`video.base_view + video.ego_view + state.eef_position/eef_rotation` and returns
`delta_eef_position + delta_eef_rotation + gripper_close`, use:

```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.cli.eval_policy --config eval/configs/starvla_franka_single_pick_banana.yaml
```

This config maps:

- `video.base_view` to `raw.cameras.franka_head.color_image`
- `video.ego_view` to `raw.cameras.franka_hand.color_image`
- `state` to `raw.robots.franka.states.gripper.pose`
- action to `env_args.action.mode: franka_eef_delta`

When `run_args.record_video` is enabled, each configured camera uses the same
core layout as SimBox data generation:

```text
outputs/eval/<eval_name>/<run_id>/videos/seed_<seed>/<robot>/images.rgb.<camera_name>/demo.mp4
```

For StarVLA smoke runs, `task.max_episode_seconds` controls the simulated
episode horizon and is converted to steps using `task.simulator.rendering_dt`.
If you need exact step-level control, remove `max_episode_seconds` and set
`task.max_steps` directly.
