#!/usr/bin/env python3
# pylint: disable=C0413
# flake8: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path


def _read_launcher_type(config_path: str) -> str | None:
    """Read only enough YAML to decide which launcher path should run."""
    try:
        import yaml
    except ImportError:
        return None

    path = Path(config_path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    launcher_type = data.get("launcher_type")
    return str(launcher_type) if launcher_type is not None else None


def _run_parallel_v2(config_path: str, extras: list[str], random_seed: str | None) -> int:
    from scripts.simbox.simbox_parallel_v2 import run_parallel_config

    parallel_extras = list(extras)
    if random_seed is not None:
        parallel_extras.append(f"--defaults.seed_base={int(random_seed)}")
    return run_parallel_config(config_path, parallel_extras)


def _run_data_engine(config_path: str, extras: list[str], random_seed: str | None, debug: bool) -> int:
    from nimbus.utils.utils import init_env

    init_env()

    from nimbus import run_data_engine
    from nimbus.utils.config_processor import ConfigProcessor
    from nimbus.utils.flags import set_debug_mode, set_random_seed

    processor = ConfigProcessor()

    try:
        config = processor.process_config(config_path, cli_args=extras)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print(f"\n Available configuration paths can be found in: {config_path}")
        print("   Use dot notation to override nested values, e.g.:")
        print("   --stage_pipe.worker_num='[2,4]'")
        print("   --load_stage.layout_random_generator.args.random_num=500")
        return 1

    processor.print_final_config(config)

    if debug:
        set_debug_mode(True)

    if random_seed is not None:
        set_random_seed(int(random_seed))

    run_data_engine(config, random_seed)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--random_seed", help="random seed")
    parser.add_argument("--debug", action="store_true", help="enable debug mode: all errors raised immediately")
    args, extras = parser.parse_known_args()

    if _read_launcher_type(args.config) == "simbox_parallel_v2":
        return _run_parallel_v2(args.config, extras, args.random_seed)
    return _run_data_engine(args.config, extras, args.random_seed, args.debug)


if __name__ == "__main__":
    raise SystemExit(main())
