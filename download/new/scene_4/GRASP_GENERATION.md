# Grasp Label Scope

This delivery includes grasp labels only for task-level interactive small objects.

Required grasp files:

- `Aligned_grasp_sparse.npy` is required only for objects listed in `REQUIRED_GRASP_OBJECTS.json`.
- These are the objects that the task is expected to pick/grasp/manipulate.
- Room-level/background pools such as `*/assets/basic/01_kitchen/small_objects/` are not required to include grasp labels unless the same geometry is also used by a task-level small object.
- Static fixtures, support/background objects, and receptacles/targets such as trays, baskets, holders, boxes, and coasters do not require grasp labels unless explicitly listed.

Generation method:

- Labels are generated with InternDataEngine's official `workflows/simbox/tools/grasp/gen_sparse_label.py`.
- The generated sparse labels use `--unit m`, `--sparse_num 50`, and `--max_width 0.1`.
- Dense intermediate files such as `Aligned_grasp_dense.*` are intentionally excluded from delivery.
- OBJ hash deduplication is used during generation; identical required geometries share the same sparse label.

Current delivery counts:

- Required grasp objects: 30
- Missing task grasp sparse files at packaging time: 0
