# SimBox Docker 并行生成使用说明

这套功能的核心思路是：**宿主机做调度，Docker 容器做隔离，每个容器内部只看见一张 GPU**。这样可以在 4 张 4090 上同时跑多个 Isaac/SimBox 生成任务，又不需要把 DataEngine 本身改成一个复杂的多 GPU 程序。

推荐主入口是 V2.6 配置化运行：

```bash
cd /home/bld/ykqin/InternDataEngine
python3 launcher.py --config configs/simbox/parallel_generate_v2.yaml
```

旧的 shell 入口 `scripts/simbox/simbox_parallel_generate.sh` 仍保留，适合临时命令；正式使用建议写 YAML 配置。

## 1. 最小 dry-run

dry-run 不启动 Docker/Isaac，只检查任务发现、GPU 分配、日志路径和输出路径。每次改配置后先跑这个。

```bash
python3 launcher.py \
  --config configs/simbox/parallel_generate_v2.yaml \
  --parallel.dry_run=true \
  --monitor.enabled=false \
  --parallel.run_id=dryrun_check
```

看结果：

```bash
cat output/_parallel_runs/dryrun_check/run_report.md
cat output/_parallel_runs/dryrun_check/manifest.jsonl
```

如果 dry-run 都不通过，通常是 task 路径、GPU 编号、配置字段写错。

## 2. 配置文件结构

一个 V2 配置一般分成四层：

```yaml
launcher_type: simbox_parallel_v2
name: my_parallel_run

parallel:
  backend: docker
  gpus: [0, 1, 2, 3]
  workers_per_gpu: 1
  task_timeout_sec: 0
  stats_after_run: true

startup_guard:
  enabled: false
  marker: "Simulation App Startup Complete"
  timeout_min: 5
  retry: 1

monitor:
  enabled: true
  mode: rich
  refresh_sec: 2
  theme: nvtop_like
  compact_paths: true
  show_gpu_panel: true
  show_data_panel: true

progress:
  enabled: true
  mode: event_hook_first
  dataset_scan_interval_sec: 30
  action_fps: 30
  video_fps: 15
  final_ffprobe_verify: true

failure_guard:
  enabled: true
  kill_on_fatal_log: true
  kill_on_suspect_hang: true
  suspect_hang_kill_sec: 1800

failed_episode_cleanup:
  enabled: true
  mode: conservative
  require_run_time_window: true
  delete_dirs: true

cache_cleanup:
  enabled: true
  scope: current_run
  cleanup_on_interrupt: true

gpu_sampling:
  enabled: true
  interval_sec: 10
  output: gpu_samples.csv

defaults:
  de_config: configs/simbox/de_plan_with_render_template.yaml
  random_num: 20
  dataset_root: null
  seed_base: 1000
  shard_random_num: false

jobs:
  - id: example_job_gpu0
    task: workflows/simbox/core/configs/tasks/pick_and_place/franka/single_pick/omniobject3d-shoe.yaml
    gpu: 0
```

关键字段：

- `parallel.gpus`：本次可用的物理 GPU。
- `parallel.workers_per_gpu`：每张 GPU 启几个 Docker 容器。
- `defaults.random_num`：每个 job 默认生成多少条。
- `jobs[].gpu`：这个 job 固定到哪张卡。
- `jobs[].allowed_gpus`：这个 job 可以被哪些 worker 抢到。
- `jobs[].shard_random_num`：把一个 job 的 `random_num` 拆成多个 job 分给多张卡。
- `defaults.dataset_root`：所有 job 的默认数据保存目录。可以写仓库内相对路径，也可以写 `/data1/yikai` 这种绝对路径。
- `jobs[].dataset_root`：单个 job 的数据保存目录，会覆盖 `defaults.dataset_root`。
- `startup_guard.enabled`：是否在 Isaac 启动超时后自动杀容器。关掉后容器不会自动退出，真卡死时需要手动停。
- `monitor.mode`：`rich` 是 V2.6 彩色 dashboard；`plain` 是普通文本。
- `progress.mode`：默认 `event_hook_first`，优先读内部 episode hook。
- `failed_episode_cleanup.enabled`：失败 episode 保存完成并记录后，是否删除 `fail_*` 目录。
- `cache_cleanup.enabled`：run 结束后是否删除当前 run 的 Isaac cache。

## 3. 一卡一个任务

适合“每张 GPU 跑不同 YAML”的正式生成。

```yaml
launcher_type: simbox_parallel_v2
name: four_jobs

parallel:
  backend: docker
  gpus: [0, 1, 2, 3]
  workers_per_gpu: 1
  task_timeout_sec: 0
  stats_after_run: true

startup_guard:
  enabled: false
  marker: "Simulation App Startup Complete"
  timeout_min: 5
  retry: 1

defaults:
  de_config: configs/simbox/de_plan_with_render_template.yaml
  random_num: 20
  seed_base: 1000

jobs:
  - id: open_microwave_part0
    task: workflows/simbox/core/configs/tasks/art/franka/open_the_microwave/open_the_microwave_part0.yaml
    gpu: 0

  - id: open_microwave_part1
    task: workflows/simbox/core/configs/tasks/art/franka/open_the_microwave/open_the_microwave_part1.yaml
    gpu: 1

  - id: pick_ball
    task: workflows/simbox/core/configs/tasks/pick_and_place/franka/single_pick/omniobject3d-ball.yaml
    gpu: 2

  - id: pick_toy_bus
    task: workflows/simbox/core/configs/tasks/pick_and_place/franka/single_pick/omniobject3d-toy_bus.yaml
    gpu: 3
```

运行：

```bash
python3 launcher.py --config configs/simbox/parallel_franka_4jobs_20.yaml
```

## 4. 一个 YAML 拆到多张 GPU

如果只有一个任务 YAML，但想把 `random_num` 拆给多张卡，使用 `allowed_gpus + shard_random_num`。

```yaml
launcher_type: simbox_parallel_v2
name: shoe_sharded

parallel:
  backend: docker
  gpus: [0, 1, 2, 3]
  workers_per_gpu: 1

defaults:
  de_config: configs/simbox/de_plan_with_render_template.yaml
  seed_base: 2000

jobs:
  - id: shoe_pick_sharded
    task: workflows/simbox/core/configs/tasks/pick_and_place/franka/single_pick/omniobject3d-shoe.yaml
    allowed_gpus: [0, 1, 2, 3]
    random_num: 40
    shard_random_num: true
```

含义：`40` 条会拆成多个 shard，例如 4 个 worker 时大致每张卡 10 条。每个 shard 有独立日志和 seed，数据仍写入同一个 dataset root。

## 5. 一个目录里多个 YAML

`task` 可以写目录。目录下每个 `.yaml` 会展开成独立 job。

```yaml
launcher_type: simbox_parallel_v2
name: open_microwave_dir

parallel:
  backend: docker
  gpus: [0, 1]
  workers_per_gpu: 1

defaults:
  de_config: configs/simbox/de_plan_with_render_template.yaml
  random_num: 10

jobs:
  - id: open_microwave_all_parts
    task: workflows/simbox/core/configs/tasks/art/franka/open_the_microwave
    allowed_gpus: [0, 1]
```

含义：目录下的 `open_the_microwave_part0.yaml`、`open_the_microwave_part1.yaml` 等会进入同一个队列，由 GPU0/1 的 worker 取任务。

## 6. 一张 GPU 多个 Docker

如果单容器显存占用低、GPU 利用率长期不高，可以探索一张卡多个容器。

```yaml
launcher_type: simbox_parallel_v2
name: two_workers_per_gpu

parallel:
  backend: docker
  gpus: [0, 1]
  workers_per_gpu: 2

defaults:
  de_config: configs/simbox/de_plan_with_render_template.yaml
  random_num: 5

jobs:
  - id: pick_batch
    task: workflows/simbox/core/configs/tasks/pick_and_place/franka/single_pick
    allowed_gpus: [0, 1]
```

含义：GPU0 会有 `gpu0_w0`、`gpu0_w1` 两个 worker，GPU1 同理。日志目录会区分 worker，例如：

```text
output/_parallel_runs/<run_id>/gpu0_w0/...
output/_parallel_runs/<run_id>/gpu0_w1/...
```

注意：一张卡多个 Docker 更容易触发显存压力、Isaac 启动慢、IO 抢占。建议先从 `workers_per_gpu: 1` 跑稳，再测试 `2`。

## 7. 运行中怎么看

V2.6 默认显示彩色 dashboard，分三块：

```text
SimBox Parallel V2.6 | run_id | dataset_root

[Run]
target  success  failed  generated  rate  traj  steps  data

[GPU]
gpu  util  mem  workers  run  ok  fail  current_jobs

[Jobs]
gpu  worker  state  job  task  done/target  S/F  rate  traj  data  peak  elapsed  silent  reason
```

状态含义：

- `pending`：还没被 worker 取走。
- `starting`：Docker 已启动，Isaac 还没到 startup marker。
- `running`：已经看到 `Simulation App Startup Complete`，任务正在生成。
- `success`：容器正常退出。
- `failed`：容器失败或重试用完。
- `startup_hang` / `failure_reason=isaac_startup_hang`：启动阶段超时。
- `suspect_hang`：长时间没有日志更新；V2.6 默认 30 分钟后自动杀掉容器，避免显存长期占死。
- `failure_reason=gpu_oom/gpu_crash/cuda_illegal_address/segfault`：日志中已经发现底层 GPU/Isaac 异常，V2.6 会主动终止容器并写报告。

更准确的状态看：

```bash
cat output/_parallel_runs/<run_id>/status.json
tail -f output/_parallel_runs/<run_id>/gpu0_w0/*/docker.log
tail -f output/_parallel_runs/<run_id>/manifest.jsonl
tail -f output/_parallel_runs/<run_id>/gpu_samples.csv
```

GPU 没负载可以作为辅助判断；最终以 `status.json` 和 `manifest.jsonl` 为准。

颜色含义：

- 蓝色/青色：正在启动或运行。
- 绿色：成功。
- 红色：失败或超时。
- 黄色：重试、静默过久、疑似卡住。
- 紫色：轨迹时长、数据量、cache/report 这类汇总信息。

如果终端显示太挤，可以临时切普通文本：

```bash
python3 launcher.py \
  --config configs/simbox/parallel_franka_12jobs_10h.yaml \
  --monitor.mode=plain
```

## 8. 输出在哪里

数据和运行记录是两套目录：

```text
output/simbox_plan_with_render/...        # 训练数据
output/_parallel_runs/<run_id>/...        # 日志、状态、报告
```

有效 episode 通常包含：

```text
meta_info.pkl
lmdb/data.mdb
images.rgb.*/demo.mp4
```

每个 Docker 的 stdout/stderr、`de_config.yaml`、`de_time_profile_*.log` 会按 run/gpu/worker/job 分开，避免互相覆盖。

V2.6 每个 run 还会生成：

```text
output/_parallel_runs/<run_id>/status.json                    # 实时状态，含完整路径
output/_parallel_runs/<run_id>/episode_events.jsonl          # 内部 hook 汇总事件
output/_parallel_runs/<run_id>/job_summary.csv               # 每个 job 的成功/失败/时长/数据量
output/_parallel_runs/<run_id>/gpu_samples.csv                # 每个容器的 GPU 显存历史
output/_parallel_runs/<run_id>/deleted_failed_episodes.jsonl # 被删除的失败 episode 记录
output/_parallel_runs/<run_id>/run_report.md                 # 人读报告
output/_parallel_runs/<run_id>/run_report.json               # 机器可读报告
```

如果不知道日志在哪里，先看：

```bash
cat output/_parallel_runs/<run_id>/status.json
```

其中 `paths` 字段会列出 `run_report.md`、`job_summary.csv`、`gpu_samples.csv`、`episode_events.jsonl` 的完整路径；每个 job 的 `log_path` 是对应 Docker 的完整日志。

如果要把数据放到 `/data1/yikai`，推荐在配置里写：

```yaml
defaults:
  de_config: configs/simbox/de_plan_with_render_template.yaml
  dataset_root: /data1/yikai
```

这只改变保存根目录，不改变 DataEngine 的内部数据结构。Docker 模式下，启动器会自动把 repo 外的绝对路径按同路径挂载进容器，例如 `/data1/yikai:/data1/yikai:rw`。

如果某个任务想单独放到另一个目录，也可以在 job 里覆盖：

```yaml
jobs:
  - id: pick_ball_gpu0
    task: workflows/simbox/core/configs/tasks/pick_and_place/franka/single_pick/omniobject3d-ball.yaml
    gpu: 0
    random_num: 20
    dataset_root: /data1/yikai/pick_ball
```

## 9. 数据统计

统计已有 dataset：

```bash
python3 scripts/simbox/simbox_dataset_stats.py \
  output/simbox_plan_with_render \
  --output-dir output/_parallel_runs/manual_stats
```

输出：

```text
dataset_stats.csv
dataset_stats.json
dataset_stats.md
```

主口径：

- episode 数：同时存在 `meta_info.pkl` 和 `lmdb/data.mdb` 的目录。
- 轨迹时长：每条 episode 取所有相机视频 duration 的最大值。
- action steps：来自 `meta_info.pkl` 的 `num_steps`。

运行中的 dashboard 先用内部 hook 的帧数/fps 估计轨迹时长；run 结束后会用 `ffprobe` 对成功 episode 的 MP4 做一次复核。

## 10. 失败数据和 cache 清理

默认策略是节省空间：

```yaml
failed_episode_cleanup:
  enabled: true
  mode: conservative
  require_run_time_window: true
  delete_dirs: true

cache_cleanup:
  enabled: true
  scope: current_run
  cleanup_on_success: true
  cleanup_on_failure: true
  cleanup_on_interrupt: true
```

含义：

- 失败 episode 默认保守删除：只删当前 run 时间窗内、当前 job 输出目录下的 `fail_*`。
- 运行中优先用内部 hook 统计；如果 hook 没写出来，V2.6 会用 dataset scan 兜底统计。
- 删除前会写入 `deleted_failed_episodes.jsonl`，失败数仍计入报告。
- run 正常结束、失败或 Ctrl-C 后都会尝试删除 `.docker/isaac-sim/<run_id>/`，不删除训练数据，也不删除 run 报告。
- 如果进程被 `kill -9`，进程内无法收尾，需要用 recover 命令补报告和清理。

异常退出后补报告/清残留：

```bash
python3 scripts/simbox/simbox_parallel_v2.py \
  --config configs/simbox/parallel_franka_12jobs_10h.yaml \
  --recover-run-id <run_id>
```

recover 会按 run id 停掉残留 `interdata-<run_id>-...` 容器，重新扫描本次 run 的数据，补写 `run_report.*`，并按配置清理当前 run cache。

如果想保留失败数据或 cache：

```bash
python3 launcher.py \
  --config configs/simbox/parallel_franka_12jobs_10h.yaml \
  --failed_episode_cleanup.enabled=false \
  --cache_cleanup.enabled=false
```

## 11. 常见用法速查

只看配置是否对：

```bash
python3 launcher.py --config configs/simbox/parallel_franka_4jobs_20.yaml --parallel.dry_run=true --monitor.enabled=false
```

正式运行：

```bash
python3 launcher.py --config configs/simbox/parallel_franka_4jobs_20.yaml
```

临时覆盖 run id：

```bash
python3 launcher.py \
  --config configs/simbox/parallel_franka_4jobs_20.yaml \
  --parallel.run_id=my_test_run
```

临时关闭统计：

```bash
python3 launcher.py \
  --config configs/simbox/parallel_franka_4jobs_20.yaml \
  --parallel.stats_after_run=false
```

临时开启 startup guard，10 分钟超时，重试 1 次：

```bash
python3 launcher.py \
  --config configs/simbox/parallel_franka_4jobs_20.yaml \
  --startup_guard.enabled=true \
  --startup_guard.timeout_min=10 \
  --startup_guard.retry=1
```

手动停止某次 run 的残留容器：

```bash
docker ps --format '{{.ID}} {{.Names}}' | rg '<run_id>'
docker rm -f <container_id>
```

## 12. 旧 shell 入口

V1 shell 入口仍可用，适合临时跑一个目录：

```bash
bash scripts/simbox/simbox_parallel_generate.sh \
  --backend docker \
  --gpus 0,1,2,3 \
  --workers-per-gpu 1 \
  --random-num 10 \
  workflows/simbox/core/configs/tasks/pick_and_place/franka/single_pick
```

单个 YAML 拆到多张 GPU：

```bash
bash scripts/simbox/simbox_parallel_generate.sh \
  --backend docker \
  --gpus 0,1,2,3 \
  --workers-per-gpu 1 \
  --random-num 40 \
  --split-random-num \
  --random-seed-base 1000 \
  workflows/simbox/core/configs/tasks/pick_and_place/franka/single_pick/omniobject3d-shoe.yaml
```

正式协作建议优先用 V2 YAML；shell 命令太长，容易把路径、GPU 或 `random_num` 写错。
