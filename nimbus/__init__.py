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
        ray.init(num_gpus=1, runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": cuda_devs}} if cuda_devs else {})
        data_engine = DistPipeDataEngine(config, master_seed=master_seed)
    else:
        from .data_engine import DataEngine

        data_engine = DataEngine(config, master_seed=master_seed)
    data_engine.run()
