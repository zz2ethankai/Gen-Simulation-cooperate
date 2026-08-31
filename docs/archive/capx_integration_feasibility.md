# CaP-X 接入 InternDataEngine 的可行性调查

调查时间：2026-05-23
上游项目：<https://github.com/capgym/cap-x>
本地项目：`/home/bld/ykqin/InternDataEngine`

## 结论先行

CaP-X 值得接入，但不建议把整个 `cap-x` 直接塞进 `InternDataEngine` 的源码树里。更稳的做法是：

```text
/home/bld/ykqin/
  InternDataEngine/          # 我们的数据飞轮主工程
    docs/
    configs/
    workflows/
    nimbus/
    nimbus_extension/
    integrations/capx_bridge/  # 建议新增：薄 adapter 层

  cap-x/                    # 建议作为 sibling repo，保持上游独立
```

核心原因是：CaP-X 不是一个轻量 Python 工具包，而是一套完整的 Code-as-Policy 评测与训练框架。它有自己的 `uv` 环境、submodules、Robosuite/LIBERO/BEHAVIOR 分支依赖、LLM proxy、感知服务、Web UI 和 RL 训练栈。把它整体放进 `InternDataEngine` 里面，会让主工程的依赖、许可证、CI、资产下载和运行入口都变重。

但是，CaP-X 的思想和 InternDataEngine 是互补的。InternDataEngine 更像“数据生产工厂”：加载场景、随机化、规划技能轨迹、回放渲染、落盘成训练数据。CaP-X 更像“会写机器人代码的 agent 评测台”：给模型一个任务和一组 API，让模型生成 Python，再用环境 reward 评价这段程序有没有真的完成操作。

所以最合理的接入方式不是“合并两个工程”，而是在两者之间做一层 adapter：让 CaP-X 能调用 InternData/SimBox 的环境、技能、观测和 reward，同时让 InternDataEngine 能复用 CaP-X 的 agent 评测、多轮视觉反馈、代码日志和后续 RL 能力。

## 两个系统的顶层定位

### InternDataEngine 当前是什么

从本地代码看，InternDataEngine 的主线是 Nimbus + SimBox：

- `nimbus/` 定义数据流调度框架。
- `nimbus_extension/` 注册实际可用的 `env_loader`、`env_randomizer`、`env_planner`、`env_renderer`、`env_writer`。
- `workflows/simbox_dual_workflow.py` 把任务 YAML 转成 Isaac Sim 场景、机器人、对象、技能和 logger。
- `configs/simbox/de_pipe_template.yaml` 描述典型 pipeline：load -> randomize -> plan -> dump -> dedump -> render -> store。
- `workflows/simbox/core/configs/tasks/.../*.yaml` 是任务入口，里面的 `robots`、`objects`、`regions`、`skills`、`data` 共同定义一个可生产数据的任务。

也就是说，InternDataEngine 当前的核心不是“让 LLM 在线控制机器人”，而是“把已有任务和技能链批量生产成高质量轨迹和多模态数据”。

一个简化的数据流是：

```text
task YAML
  -> EnvLoader / SimBoxDualWorkFlow
  -> layout randomization
  -> skill sequence planning
  -> sequence replay and rendering
  -> LMDB / image / annotation storage
```

这里的 `generate_seq()` 更像一个确定性或半确定性的技能执行器：先根据 YAML 里的 `skills` 生成动作，再记录和回放。

### CaP-X 当前是什么

CaP-X 官方 README 把它拆成四块：

- `CaP-Gym`：机器人操作环境，agent 通过生成 Python 代码组合感知和控制 primitive。
- `CaP-Bench`：按 abstraction level、interaction mode、visual grounding modality 做系统评测。
- `CaP-Agent0`：多轮视觉差分、自动技能库、并行 ensemble reasoning。
- `CaP-RL`：用环境 reward 做 GRPO 后训练。

它的程序入口是 `capx/envs/launch.py`，典型运行方式是：

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
  --model "google/gemini-3.1-pro-preview"
```

CaP-X 的核心抽象是：

```text
BaseEnv
  reset()
  step()
  get_observation()
  compute_reward()
  task_completed()

CodeExecutionEnvBase
  给 LLM 拼 prompt
  执行 LLM 生成的 Python
  收集 stdout/stderr/reward/video

ApiBase
  暴露给 LLM 的函数集合
  函数 docstring 会进入 prompt，成为模型实际看到的“机器人说明书”
```

所以 CaP-X 的中心问题是：给模型哪些 API、给它怎样的视觉反馈、执行后如何判定成功。

## 为什么不能直接把 CaP-X 当普通依赖装进 `interndata`

本地 `interndata` conda 环境确实是 Python 3.10.20，而且已经有 `torch`、`isaacsim`、`omni`、`ray`、`gymnasium`、`pxr` 等关键组件。但版本边界不完全对齐：

```text
interndata:
  Python      3.10.20
  torch       2.6.0+cu124
  ray         2.54.1
  gymnasium   0.28.1
  numpy       1.26.4

CaP-X pyproject:
  Python      >=3.10,<3.13
  torch       >=2.8
  ray         ==2.48.0
  gymnasium   >=0.29
```

单看这些版本就能看出：把 CaP-X 直接 `uv sync` 到 `interndata` 里，会有比较高的破坏风险。`torch`、`ray`、`gymnasium` 这类包不是普通工具库，它们会影响 Isaac Sim、并行调度、GPU runtime 和仿真进程启动。一个小版本变化就可能把原来可跑的数据生产流程打断。

CaP-X 自己也明确要求按 simulator family 分环境。例如 Robosuite 和 LIBERO 依赖的 `robosuite` 版本冲突；LIBERO 要单独 venv；BEHAVIOR 走 OmniGibson + Isaac Sim + CUDA 12.x；RL 训练又需要 Python 3.12 和 flash-attn 相关栈。这个结构本身就在提醒我们：CaP-X 应该被当成一个独立运行栈，而不是主工程的普通子包。

## 可行的接入路线

### 路线 A：先把 CaP-X 作为外部 benchmark 使用

这是最低风险路线。

做法：

```text
InternDataEngine 继续生产数据和任务。
cap-x 作为 sibling repo 独立安装、独立运行。
两边先通过文件、配置和结果目录交互。
```

适合先验证的问题：

- CaP-X 原生任务在本服务器上能不能跑通。
- LLM proxy、感知服务、视频保存、parallel workers 是否稳定。
- CaP-X 的输出格式、代码日志、trial summary 是否适合纳入我们的评测报告。

这一步不需要改 InternDataEngine 的主流程。它的价值是建立一条“外部评测基线”，先把 CaP-X 自己跑明白。

可行性：高。
风险：主要是 CaP-X 自身安装、submodule 和数据许可。
对 InternDataEngine 的侵入：很低。

### 路线 B：给 SimBox 写 CaP-X adapter

这是最推荐的第一阶段真正接入路线。

思路是新增一组薄封装，而不是复制 CaP-X 源码：

```text
InternDataEngine/integrations/capx_bridge/
  simbox_env.py       # 把 SimBox 任务包装成 CaP-X BaseEnv
  simbox_api.py       # 把 pick/place/open/move/get_pose 等能力包装成 ApiBase
  simbox_task.py      # 定义 CodeExecutionEnvBase 的任务 prompt/oracle
  configs/
    single_pick_banana.yaml
```

概念映射如下：

| CaP-X 概念 | InternDataEngine/SimBox 对应物 |
|---|---|
| `BaseEnv.reset()` | 初始化 Isaac Sim `World`，加载一个 SimBox task YAML，执行 randomization |
| `BaseEnv.get_observation()` | 返回相机 RGB/depth、机器人状态、对象位姿、任务 prompt |
| `BaseEnv.compute_reward()` | 根据 task/skill 状态或对象位姿判断成功度 |
| `BaseEnv.task_completed()` | 对接 SimBox 任务成功条件 |
| `ApiBase.functions()` | 暴露 `pick()`、`place()`、`open()`、`close()`、`move_to_pose()`、`get_object_pose()` |
| LLM 生成的 Python | 在线组合这些 SimBox primitive |
| CaP-X trial output | code、stderr/stdout、reward、video、summary |
| Nimbus writer output | LMDB、图像、深度、分割、bbox、轨迹 annotation |

这条路线的关键收益是：CaP-X 负责“让模型生成代码并评测”，SimBox 继续负责“真实仿真、技能 primitive、相机和数据记录”。这样双方各做自己擅长的事。

建议第一个任务不要选长程任务，而选现成的简单 pick：

```text
workflows/simbox/core/configs/tasks/pick_and_place/franka/single_pick/omniobject3d-banana.yaml
```

原因是它已经有标准 pickable 资产、机器人、相机、region、技能配置和 `max_episode_length`。第一轮目标不是证明所有任务都能跑，而是证明“CaP-X 生成/执行代码 -> SimBox 状态改变 -> reward 判断 -> 输出日志视频”这条闭环成立。

可行性：中高。
风险：需要补齐 SimBox 的在线观测和 reward 接口；当前 SimBox 更偏批生产，不是天然 Gym step 环境。
对 InternDataEngine 的侵入：中等，但可限制在 adapter 目录。

### 路线 C：把 CaP-X agent 接进 Nimbus 的 `plan_stage`

这是第二阶段路线。

现在 Nimbus 的 plan 阶段大致是：

```text
scene -> EnvSeqPlanner -> wf.generate_seq() -> Sequence
```

如果接入 CaP-X，可以变成：

```text
scene -> CapXCodePlanner -> LLM 生成代码 -> 执行并得到 sequence/reward/log -> Sequence
```

这样做的意义更大：CaP-X 不只是一个外部 evaluator，而是成为数据飞轮里的“规划器候选”。模型写出的代码如果成功，可以转成轨迹、回放、渲染和存储；失败的代码和日志也能进入后续自我改进或数据筛选流程。

但这条路线比路线 B 更深，因为它要回答：

- LLM 代码执行产生的动作如何转换成 `WorldRecorder` 可回放的 sequence？
- 失败 trial 是否进入数据集，还是只进入评测日志？
- 多轮视觉反馈和 Nimbus 的 pipeline 并行调度如何共存？
- 同一个 worker 内 Isaac Sim、感知服务、LLM 请求是否会互相抢 GPU？

可行性：中。
风险：调度、可复现性和数据格式。
对 InternDataEngine 的侵入：中高。

### 路线 D：接入 CaP-RL

这是长期路线，不建议一开始做。

CaP-RL 需要 GRPO/VeRL、vLLM、flash-attn、W&B/HuggingFace 登录和多 GPU 训练环境。它的价值在于：如果我们已经有稳定的 InternData/SimBox CaP-X 环境，就可以用任务 reward 后训练 coding agent，让它更会调用我们的机器人 primitive。

但如果环境 adapter 和 reward 还没稳定，先上 RL 会变成“训练系统、仿真系统、代码执行系统、GPU 调度系统”一起调试，复杂度太高。

可行性：长期可行。
第一阶段优先级：低。

## 目录放置建议

### 不建议：放在 `InternDataAssets/` 下

`InternDataAssets/` 是资产和数据目录，本地已经有约 171G。CaP-X 是代码框架，不是资产包。放这里会混淆“数据资产”和“代码工具”的边界，也不利于 git 管理。

### 不建议：直接复制到 `InternDataEngine/cap-x/`

这么做看起来方便 import，但实际会带来几个问题：

- CaP-X 有多个 submodule，递归 clone 会拉 LIBERO、BEHAVIOR、cuRobo、SAM3、VeRL 等。
- CaP-X 的 `uv.lock`、`pyproject.toml`、`web-ui/`、`env_configs/` 会和主工程的依赖/配置并列，容易让入口混乱。
- 主工程当前 license 是 CC BY-NC-SA 4.0，CaP-X 是 MIT；MIT 可以被集成，但 vendoring 时需要保留上游 license 和 copyright notice。
- BEHAVIOR、Isaac Sim、SAM3、数据集等还有各自 EULA/模型权重访问要求，不应该被“默认为主仓依赖”。
- 后续跟上游同步会更麻烦。

### 可选但谨慎：作为 `deps/cap-x` git submodule

如果团队强烈希望所有源码在一个仓库入口下，可以用 submodule：

```text
InternDataEngine/
  deps/
    cap-x/       # git submodule, 默认不递归拉第三方 submodule
```

但这更适合“冻结一个已验证版本”，不适合早期探索。早期需要频繁试安装、切分支、删环境、重拉 submodule；放在主仓 `deps/` 里会让主工程变脏。

### 推荐：CaP-X sibling repo + InternDataEngine 内部 adapter

推荐结构：

```text
/home/bld/ykqin/
  InternDataEngine/
    integrations/
      capx_bridge/
        README.md
        simbox_env.py
        simbox_api.py
        simbox_task.py
    configs/
      capx/
        simbox_single_pick_banana.yaml
    scripts/
      capx/
        run_simbox_smoke.sh

  cap-x/
    capx/
    env_configs/
    docs/
    pyproject.toml
```

这种结构的意思是：

- 上游 `cap-x` 保持独立，可以按官方方式 `uv sync`。
- InternDataEngine 只保存我们自己的桥接代码、配置和文档。
- 运行时通过 `PYTHONPATH`、editable install 或小的 bridge package 把两边连起来。
- 哪一天要升级 CaP-X，只升级 sibling repo；哪一天要删掉实验，也不会污染主工程。

## 第一阶段建议拆解

### Step 1：外部跑通 CaP-X 原生 smoke test

目标不是跑全量 benchmark，而是确认安装边界：

```bash
cd /home/bld/ykqin/cap-x
uv sync
uv sync --extra robosuite
./scripts/regression_test.sh quick
```

如果只看 BEHAVIOR/Isaac Sim 分支，则不要和 Robosuite 混在同一个环境里，按 CaP-X 官方文档单独走 `capx/third_party/b1k/.venv`。

验收标准：

- 能启动 `capx/envs/launch.py`。
- oracle 或 quick regression 能跑。
- output 里有 trial summary、日志和视频。

### Step 2：做一个最小 SimBox adapter

最小闭环只需要一个任务：

```text
single_pick / omniobject3d-banana
```

adapter 先暴露少量 API：

```python
get_object_pose(name) -> np.ndarray
pick(name) -> None
lift(height=0.1) -> None
place_on(target) -> None
```

第一版 prompt 不追求泛化，只追求模型能理解这些 API 并生成可执行代码。oracle code 应该先能成功：

```python
pick("pick_object_left")
lift(0.1)
```

验收标准：

- `reset()` 能加载 task YAML。
- `get_observation()` 至少返回一个主相机 RGB 和对象/机器人状态。
- oracle code 能改变仿真状态。
- `compute_reward()` 能区分成功/失败。
- CaP-X 输出里能保存 code、stdout/stderr、reward 和视频。

### Step 3：把成功 trial 接回数据飞轮

当 CaP-X trial 成功后，有两种落盘方式：

1. 只保留 CaP-X 的评测结果，作为 planner/agent benchmark。
2. 把成功执行过程转成 Nimbus 可存储的数据，再走 `EnvWriter` 或新的 writer。

第一阶段建议先做 1。因为 2 会涉及轨迹回放、logger schema、LMDB 字段和多相机 annotation，对接成本更高。

### Step 4：再考虑多轮视觉反馈和技能库

CaP-X 的多轮视觉反馈和 skill library compilation 很适合数据飞轮后半段：失败 trial 可以被分析，成功代码片段可以沉淀成技能库，再反过来改善 prompt 或 API 设计。

但它依赖稳定的视频/图像反馈。应该等单轮 oracle 和单轮 LLM 都稳定后再做。

## 主要风险清单

### 1. 依赖风险

不要在 `interndata` 环境里直接安装完整 CaP-X。尤其是 `torch`、`ray`、`gymnasium`、`robosuite`、`libero`、`omnigibson`、`vllm` 这些包，任何一个版本改变都可能影响现有数据生产。

### 2. 仿真入口风险

InternDataEngine 当前通过 `SimulationApp`、`World`、`SimBoxDualWorkFlow` 管理 Isaac Sim 生命周期。CaP-X 的 BEHAVIOR 分支通过 OmniGibson 管理 Isaac Sim。两个系统都想当仿真主控时，不能在同一个 Python 进程里随意混用。

### 3. 数据格式风险

CaP-X 的天然产物是：

```text
generated code
stdout/stderr
reward
task_completed
video
trial summary
```

InternDataEngine 的天然产物是：

```text
trajectory
RGB/depth/seg/bbox/keypoint 等多模态 annotation
LMDB/package
task metadata
```

这两者相关，但不是同一种数据。不要默认 CaP-X output 可以直接当 InternData 训练数据。

### 4. reward 定义风险

CaP-X 的评测强依赖 `compute_reward()` 和 `task_completed()`。SimBox 任务目前更多通过技能成功、对象状态、轨迹长度和 logger 体现结果。要接 CaP-X，必须把每个任务的成功条件显式化。

### 5. 代码执行安全风险

CaP-X 的 `SimpleExecutor` 是本地 `exec()`，允许模型生成的代码 import 包并访问环境对象。它适合受控研究环境，但不适合执行不可信代码。接入时应该只在隔离环境、受控模型和受控 API 下运行。

### 6. 许可证和数据许可风险

CaP-X 本体是 MIT license，理论上可以集成到当前非商业主仓，但需要保留原 license。它依赖的 BEHAVIOR、Isaac Sim、SAM3 权重、OpenRouter/vLLM 模型服务等各自有许可、EULA 或访问条件。不要把这些隐含为 InternDataEngine 的默认依赖。

## 最推荐的近期决策

1. `cap-x` 放在 `/home/bld/ykqin/cap-x`，与 `InternDataEngine` 并行。
2. 不把完整 CaP-X 放进 `InternDataAssets/`，也不直接复制进主源码树。
3. 在 `InternDataEngine` 内新增一个小的 `integrations/capx_bridge/`，只放我们写的 adapter。
4. 第一阶段只做 `single_pick/omniobject3d-banana` 的 smoke adapter。
5. 先把 CaP-X 当 evaluation/agent harness，暂时不要直接接 CaP-RL。
6. 等 adapter 稳定后，再把成功 trial 转成 Nimbus 可存储的数据。

一句话概括：CaP-X 应该成为 InternData 数据飞轮旁边的“代码策略评测与改进引擎”，而不是被吞进数据资产目录或主工程依赖里。真正要合并的不是仓库，而是运行时契约：SimBox 提供环境和机器人 API，CaP-X 提供代码生成、评测、多轮反馈和后续 RL。

## 参考资料

- CaP-X GitHub: <https://github.com/capgym/cap-x>
- CaP-X README: <https://raw.githubusercontent.com/capgym/cap-x/main/README.md>
- CaP-X Adding Environments: <https://raw.githubusercontent.com/capgym/cap-x/main/docs/adding-environments.md>
- CaP-X Adding APIs: <https://raw.githubusercontent.com/capgym/cap-x/main/docs/adding-apis.md>
- CaP-X Configuration: <https://raw.githubusercontent.com/capgym/cap-x/main/docs/configuration.md>
- CaP-X BEHAVIOR Tasks: <https://raw.githubusercontent.com/capgym/cap-x/main/docs/behavior-tasks.md>
- CaP-X RL Training: <https://raw.githubusercontent.com/capgym/cap-x/main/docs/rl-training.md>
