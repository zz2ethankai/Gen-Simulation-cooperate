import ray

from nimbus.utils.types import STAGE_PIPE

from .data_engine import DataEngine, DistPipeDataEngine


def run_data_engine(config, master_seed=None):
    import nimbus_extension  # noqa: F401  pylint: disable=unused-import

    if STAGE_PIPE in config:
        import os
        cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES")
        ray.init(num_gpus=1, runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": cuda_devs}} if cuda_devs else {})
        data_engine = DistPipeDataEngine(config, master_seed=master_seed)
    else:
        data_engine = DataEngine(config, master_seed=master_seed)
    data_engine.run()
