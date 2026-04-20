# Nav2 Split Compose Architecture

当前项目里的导航主链路已经统一为两容器架构：

- `isaac` 容器负责仿真、场景导图、`navigate` skill 调度
- `nav2` 容器负责 Nav2 bringup、接收目标、执行路径规划与控制

## 当前目录结构

`nav2/` 现在只保留四组文件：

- `nav2/runtime/`
  - workflow 侧的导航会话管理
  - 入口：`nav2/runtime/__init__.py`
- `nav2/bridge/`
  - Isaac / Nav2 两端通过标准 ROS topic 交互的桥接层
  - 包含 `client.py`、`adapter.py`、`clock.py`
- `nav2/mapgen/`
  - 静态地图导出和 Nav2 bootstrap 参数生成
  - 包含 `exporter.py`、`prepare_stack.py`
- `nav2/container/`
  - Nav2 容器镜像和启动入口
  - 包含 `Dockerfile`、`entrypoint.sh`

移动底盘语义已经拆到平台层：

- `workflows/simbox/core/mobile/platforms/base_platform.py`
  - 抽象底盘接口，定义 Nav2 footprint / 转弯半径 / bridge 工厂
- `workflows/simbox/core/mobile/platforms/ranger_mini_v3_platform.py`
  - 当前唯一保留的平台实现，统一命名为 Ranger Mini V3

已删除的旧链路：

- 文件控制面协议 `protocol.py`
- `nav2_stack_watcher.py`
- `nav2_goal_executor.py`
- `nav2_navigator.py`
- 示例 `maps/`、`params/`

## 当前真实运行链路

1. `navigate` skill 调用 `nav2.runtime.PersistentNav2RuntimeManager`
2. `PersistentNav2RuntimeManager` 用 `nav2.mapgen.exporter.IsaacStaticMapExporter` 导出静态地图
3. `PersistentNav2RuntimeManager` 用 `nav2.bridge.client.Nav2BridgeClient` 发送：
   - map update
   - goal
   - cancel
4. `nav2` 容器通过 `nav2/container/entrypoint.sh` 启动：
   - `python3 -m nav2.mapgen.prepare_stack`
   - `python3 -m nav2.bridge.adapter`
   - `map_server`
   - `lifecycle_manager`
   - `nav2_bringup`
5. `nav2.bridge.adapter` 把 ROS topic 控制消息转成：
   - `LoadMap` service
   - `NavigateToPose` action
   - costmap clear service
6. 结果和状态再通过 ROS topic 回到 Isaac 侧 bridge client
7. `PersistentNav2RuntimeManager` 根据状态判定 skill 成功/失败，并输出调试产物

## 入口文件

- Isaac 侧会话管理：`nav2/runtime/__init__.py`
- Isaac 侧 bridge：`nav2/bridge/client.py`
- Nav2 侧 bridge：`nav2/bridge/adapter.py`
- 地图导出：`nav2/mapgen/exporter.py`
- 容器入口：`nav2/container/entrypoint.sh`
- 容器镜像：`nav2/container/Dockerfile`

## 说明

当前控制面不再使用共享目录轮询协议，而是统一走标准 ROS topic / action / service。
