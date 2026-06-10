#!/usr/bin/env python3
import argparse
import os
import pickle

import lmdb
import numpy as np


DEFAULT_KEYS = [
    "states.base.pose",
    "states.base.twist_body",
    "states.base.steering_positions",
    "states.base.wheel_positions",
    "states.base.steering_velocities",
    "states.base.wheel_velocities",
    "base_actions.vx_body",
    "base_actions.vy_body",
    "base_actions.wz_body",
    "base_actions.requested_steering",
    "base_actions.requested_wheel_velocities",
    "base_actions.applied_wheel_velocities",
    "actions.base.steering_positions",
    "actions.base.wheel_velocities",
]


def key_to_npz_name(key: str) -> str:
    return key.replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lmdb_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--key", action="append", dest="keys")
    args = parser.parse_args()

    keys = args.keys or DEFAULT_KEYS
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    env = lmdb.open(args.lmdb_dir, readonly=True, lock=False, readahead=False, max_readers=1)
    arrays = {}
    with env.begin(write=False) as txn:
        for key in keys:
            raw = txn.get(key.encode("utf-8"))
            if raw is None:
                print(f"MISSING {key}")
                continue
            value = pickle.loads(raw)
            array = np.asarray(value)
            arrays[key_to_npz_name(key)] = array
            if array.size:
                min_value = float(np.nanmin(array))
                max_value = float(np.nanmax(array))
            else:
                min_value = None
                max_value = None
            print(f"{key} shape={array.shape} dtype={array.dtype} min={min_value} max={max_value}")
    env.close()

    np.savez_compressed(args.output, **arrays)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
