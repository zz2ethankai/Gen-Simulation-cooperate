from nimbus.utils.types import STAGE_PIPE

__all__ = ["DataEngine", "DistPipeDataEngine", "run_data_engine"]


def __getattr__(name):
    if name in {"DataEngine", "DistPipeDataEngine"}:
        from .data_engine import DataEngine, DistPipeDataEngine

        globals()["DataEngine"] = DataEngine
        globals()["DistPipeDataEngine"] = DistPipeDataEngine
        return globals()[name]
    raise AttributeError(f"module 'nimbus' has no attribute {name!r}")


def run_data_engine(config, master_seed=None):
    import ray
    import nimbus_extension  # noqa: F401  pylint: disable=unused-import

    if STAGE_PIPE in config:
        import os
        from .data_engine import DistPipeDataEngine

        cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
        ray_init_kwargs = {}
        if cuda_devs:
            visible_devices = [device.strip() for device in cuda_devs.split(",") if device.strip()]
            ray_init_kwargs["num_gpus"] = len(visible_devices)
            ray_init_kwargs["runtime_env"] = {
                "env_vars": {
                    "CUDA_VISIBLE_DEVICES": cuda_devs,
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                }
            }
        ray.init(**ray_init_kwargs)
        data_engine = DistPipeDataEngine(config, master_seed=master_seed)
    else:
        from .data_engine import DataEngine

        data_engine = DataEngine(config, master_seed=master_seed)
    data_engine.run()
