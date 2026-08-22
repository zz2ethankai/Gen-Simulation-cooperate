"""Quaternion helpers for rigid-object reset pose contracts."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation as R


def _quat_array(quaternion) -> np.ndarray:
    values = np.asarray(quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"invalid scalar-first quaternion: {quaternion!r}")
    return values / norm


def upright_world_orientation(quaternion) -> np.ndarray:
    """Return a scalar-first quaternion with the input world yaw only.

    Scene region sampling defines ``keep_upright`` as preserving the current
    XYZ-Euler yaw while removing roll and pitch.  Normalize and construct the
    quaternions through USD's ``Gf.Rotation`` when available, matching the
    Isaac/Usd scalar-first convention.  The scipy fallback is only for the
    lightweight test environment, which does not load the USD Python module.
    """

    values = _quat_array(quaternion)
    try:
        from pxr import Gf
    except ImportError:  # pragma: no cover - exercised by lightweight tests
        yaw = R.from_quat(values, scalar_first=True).as_euler("xyz", degrees=False)[2]
        return R.from_euler("z", yaw, degrees=False).as_quat(scalar_first=True)

    gf_quaternion = Gf.Quatd(
        float(values[0]),
        Gf.Vec3d(float(values[1]), float(values[2]), float(values[3])),
    )
    normalized = Gf.Rotation(gf_quaternion).GetQuat()
    normalized_values = np.asarray(
        [normalized.GetReal(), *normalized.GetImaginary()],
        dtype=float,
    )
    # Keep the exact Euler convention used by RandomRegionSampler, but source
    # the normalized quaternion and final construction from the USD helper.
    yaw = R.from_quat(normalized_values, scalar_first=True).as_euler("xyz", degrees=False)[2]
    upright = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), math.degrees(float(yaw))).GetQuat()
    return np.asarray([upright.GetReal(), *upright.GetImaginary()], dtype=float)
