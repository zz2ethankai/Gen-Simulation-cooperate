# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from graspgenx.utils.logging_config import setup_logging

    setup_logging()
except ImportError:
    pass

try:
    import os

    from graspgenx._setup_dependencies import (
        ensure_checkpoints,
        ensure_gripper_descriptions,
        get_checkpoints_root,
        get_checkpoints_version_dir,
        get_gripper_descriptions_assets,
        get_gripper_descriptions_root,
    )

    # InterndataEngine keeps models and gripper descriptions outside the
    # vendored source tree.  Its generation CLI sets this flag before importing
    # graspgenx so an ordinary validation/import can never start a multi-GB git
    # clone as a side effect.  Upstream's auto-setup remains the default.
    if os.environ.get("GRASPGENX_DISABLE_AUTO_SETUP", "0") not in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }:
        ensure_gripper_descriptions()
        ensure_checkpoints()
except Exception:  # noqa: BLE001 - never let setup hook break imports
    import logging

    logging.getLogger(__name__).debug(
        "graspgenx dependency setup hook failed; continuing.", exc_info=True
    )

__version__ = "0.1.0"
