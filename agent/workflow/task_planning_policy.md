# 任务规划与机器人 Skill 编排策略

本层只负责回答三个语义问题：操作哪个实例、目标关系是什么、需要按什么顺序调用哪些已准入的机器人
Skill。它不写 YAML，不选择机器人落点，不产生抓取姿态，也不决定碰撞距离、关节、相机或支撑高度。

一个被直接操作的物体对应一个 subtask。Pick 的对象是该物体；Place 的对象依次是被操作物和目标物。
同一传送任务使用同一机械臂顺序执行 Pick、Place。多物体任务拆成多个 subtask；若当前 profile 或执行器
无法满足并发、双臂或导航要求，必须写入 `unresolved`，不能改变任务语义来绕过能力缺口。

TaskPlan 只包含 `robot_requirement={required_capabilities, preferred_profile_ids, decision_basis}`。未被用户
明确限制时，subtask 使用 `any_single_arm`。确定性 embodiment resolver 随后产生一个或多个
`ExecutionVariant`，每个 variant 固定 instance、profile、placement family、collision mode 与全部 arm
binding。臂基座、CuRobo 配置、夹爪、相机和数据通道均由 profile 解析，不由 LLM 猜测。

Skill 参数以 `agent/robot_skills/contracts.yaml` 为唯一机器契约。Agent 只能填写 `owner=agent` 的字段；
没有任务、资产或证据依据的可选参数应省略。Skill 内部的接近、闭合、attach、collision-world rebuild、
tracking 和 safety recovery 不得展开成伪 Skill。

返回前必须确认：实例名来自 inventory；目标 affordance 真实存在；Skill 已准入且 capability 满足；Skill
顺序和对象数量合法；没有障碍物隐藏、任意 Prim/YAML 路径或编造的几何数值。
