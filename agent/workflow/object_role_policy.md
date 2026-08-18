# 操作对象与目标对象识别策略

中心物品是当前 subtask 中被机器人直接操作的精确实例，不是容器、桌面、地板或附近最显眼的资产。
选择时先用用户语义形成候选，再用 inventory 中的实例名、角色、刚体/碰撞状态、affordance 和现有任务
关系交叉验证。存在多个同类实例时必须保留歧义或引用可验证的关系，不能取列表首项。

目标物只描述任务关系：`inside` 需要容器区域，`on` 需要支撑面，`hang` 需要声明的结构和方向。`insert` 在 v1 中保持 unresolved，直到插入轴、最小深度、终端姿态和专用运行时谓词同时具备。
操作物与目标物确定后，语义层输出 capability 和可选 arm constraint；实际 profile、instance、机械臂、
机器人位姿、物体布局和可达性由 ExecutionVariant resolver、SceneLayout 与 CuRobo probe 决定。

多 subtask 可以共享一个已验证机器人位姿。没有共同位姿时应进入布局搜索或返回能力缺口；在闭环 Nav
被单独准入前，不生成开环底盘动作。
