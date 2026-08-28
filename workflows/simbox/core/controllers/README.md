# Arm Controllers

The CuRobo controller has one typed control path. `TemplateController` owns
Isaac lifecycle and assembly; `MotionPlannerRuntime` owns typed single/batch
pose and c-space requests, exact Physics-schema collision paths, attachments,
and execution delegation. `ControllerSetup` owns frame/joint setup and reset;
`ControllerExecution` consumes typed commands and named trajectories.

Pick and Place generate all YAML-filtered candidates, call native CuRobo in
capacity-sized batches, and intersect pre/terminal feasibility by original
candidate index. A batch failure is not retried through single planning.

Non-planning direct actions use `MotionPhaseCommand.direct_joint_action` with
`CollisionPolicy.PASSTHROUGH`. Debug snapshots and visual overlays are
external observational services and are not part of the planning chain.
