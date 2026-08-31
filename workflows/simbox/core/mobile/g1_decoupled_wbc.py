"""Minimal NumPy/ONNX port of HUMANO's Python GEAR decoupled WBC."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np


@dataclass
class RobotState:
    """Minimal 29-DOF proprioceptive state consumed by the G1 WBC."""

    body_q: np.ndarray
    body_dq: np.ndarray
    base_quat: np.ndarray
    base_ang_vel: np.ndarray
    pelvis_z: float | None = None


@dataclass
class MotorCommand:
    """Joint targets produced by the G1 WBC for one control tick."""

    q_target: np.ndarray = field(
        default_factory=lambda: np.zeros(29, dtype=np.float64)
    )
    dq_target: np.ndarray = field(
        default_factory=lambda: np.zeros(29, dtype=np.float64)
    )
    kp: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float64))
    kd: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float64))
    tau_ff: np.ndarray = field(
        default_factory=lambda: np.zeros(29, dtype=np.float64)
    )


_NUM_BODY_JOINTS = 29
_NUM_LOWER_BODY_JOINTS = 15
_OBS_HISTORY_LENGTH = 6
_SINGLE_OBSERVATION_DIM = 86
_OBSERVATION_DIM = _OBS_HISTORY_LENGTH * _SINGLE_OBSERVATION_DIM


class G1DecoupledWbcPolicy:
    """Run GEAR-WBC at 50 Hz while holding HUMANO's stand upper-body pose."""

    default_lower_body_angles = np.asarray(
        [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    default_upper_body_angles = np.asarray(
        [
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        balance_model_path: str | Path | None = None,
        walk_model_path: str | Path | None = None,
        *,
        control_dt: float = 0.02,
        balance_session=None,
        walk_session=None,
    ):
        if (balance_session is None) != (walk_session is None):
            raise ValueError("balance_session and walk_session must be provided together")
        if balance_session is None:
            balance_session = self._load_session(balance_model_path, "balance")
            walk_session = self._load_session(walk_model_path, "walk")
        self._balance_session = balance_session
        self._walk_session = walk_session
        self._balance_input_name = self._input_name(balance_session, "balance")
        self._walk_input_name = self._input_name(walk_session, "walk")
        self._control_dt = float(control_dt)
        if not math.isfinite(self._control_dt) or self._control_dt <= 0.0:
            raise ValueError("decoupled WBC control_dt must be positive and finite")
        self._history = deque(maxlen=_OBS_HISTORY_LENGTH)
        self._previous_action = np.zeros(_NUM_LOWER_BODY_JOINTS, dtype=np.float32)
        self._cached_command: MotorCommand | None = None
        self._step_index = 0
        self._inference_count = 0
        self._last_mode = "balance"

    @staticmethod
    def _load_session(model_path: str | Path | None, label: str):
        if model_path is None:
            raise ValueError(f"{label}_model_path is required")
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"G1 {label} WBC model not found: {path}")
        import onnxruntime as ort

        available_providers = set(ort.get_available_providers())
        providers = (
            [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available_providers
            else ["CPUExecutionProvider"]
        )
        return ort.InferenceSession(str(path), providers=providers)

    @staticmethod
    def _input_name(session, label: str) -> str:
        inputs = session.get_inputs()
        if not inputs:
            raise RuntimeError(f"G1 {label} WBC model has no input")
        return str(inputs[0].name)

    @property
    def inference_count(self) -> int:
        return self._inference_count

    @property
    def history_length(self) -> int:
        return len(self._history)

    @property
    def last_mode(self) -> str:
        return self._last_mode

    def reset(self) -> None:
        self._history.clear()
        self._previous_action.fill(0.0)
        self._cached_command = None
        self._step_index = 0
        self._inference_count = 0
        self._last_mode = "balance"

    @staticmethod
    def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
        w, x, y, z = quaternion
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
        w, x, y, z = quaternion
        return np.asarray(
            [
                -2.0 * (x * z - w * y),
                -2.0 * (y * z + w * x),
                -(w * w - x * x - y * y + z * z),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _validate_state(state: RobotState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        body_q = np.asarray(state.body_q, dtype=np.float64).reshape(-1)
        body_dq = np.asarray(state.body_dq, dtype=np.float64).reshape(-1)
        base_quat = np.asarray(state.base_quat, dtype=np.float64).reshape(-1)
        base_ang_vel = np.asarray(state.base_ang_vel, dtype=np.float64).reshape(-1)
        if body_q.shape != (_NUM_BODY_JOINTS,) or body_dq.shape != (_NUM_BODY_JOINTS,):
            raise ValueError("G1 WBC state must contain 29 body positions and velocities")
        if base_quat.shape != (4,) or base_ang_vel.shape != (3,):
            raise ValueError("G1 WBC state must contain a 4D quaternion and 3D angular velocity")
        arrays = (body_q, body_dq, base_quat, base_ang_vel)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("G1 WBC state must be finite")
        norm = float(np.linalg.norm(base_quat))
        if norm <= 1.0e-8:
            raise ValueError("G1 WBC base quaternion must be non-zero")
        return body_q, body_dq, base_quat / norm, base_ang_vel

    def _policy_command(self, navigate_cmd: np.ndarray, current_yaw: float) -> np.ndarray:
        yaw_error = math.atan2(
            math.sin(float(navigate_cmd[3]) - current_yaw),
            math.cos(float(navigate_cmd[3]) - current_yaw),
        )
        yaw_rate = 0.0 if abs(yaw_error) < 0.01 else float(np.clip(yaw_error / 0.5, -1.0, 1.0))
        return np.asarray([navigate_cmd[0], navigate_cmd[1], yaw_rate], dtype=np.float32)

    def _build_observation(
        self,
        body_q: np.ndarray,
        body_dq: np.ndarray,
        base_quat: np.ndarray,
        base_ang_vel: np.ndarray,
        policy_command: np.ndarray,
    ) -> np.ndarray:
        defaults = np.zeros(_NUM_BODY_JOINTS, dtype=np.float64)
        defaults[:_NUM_LOWER_BODY_JOINTS] = self.default_lower_body_angles
        observation = np.zeros(_SINGLE_OBSERVATION_DIM, dtype=np.float32)
        observation[0:3] = policy_command * np.asarray([2.0, 2.0, 0.5], dtype=np.float32)
        observation[3] = 0.74
        observation[7:10] = base_ang_vel * 0.5
        observation[10:13] = self._gravity_orientation(base_quat)
        observation[13:42] = body_q - defaults
        observation[42:71] = body_dq * 0.05
        observation[71:86] = self._previous_action
        self._history.append(observation)
        padded = [np.zeros_like(observation)] * (_OBS_HISTORY_LENGTH - len(self._history))
        return np.concatenate([*padded, *self._history]).reshape(1, _OBSERVATION_DIM)

    @staticmethod
    def _run_session(session, input_name: str, observation: np.ndarray) -> np.ndarray:
        outputs = session.run(None, {input_name: observation.astype(np.float32, copy=False)})
        if not outputs:
            raise RuntimeError("G1 WBC model returned no outputs")
        action = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        if action.shape != (_NUM_LOWER_BODY_JOINTS,) or not np.all(np.isfinite(action)):
            raise RuntimeError(f"G1 WBC action must be 15 finite values, got shape={action.shape}")
        return action

    def step(self, state: RobotState, navigate_cmd, *, env_step_dt: float) -> MotorCommand:
        dt = float(env_step_dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("env_step_dt must be positive and finite")
        interval = max(1, int(round(self._control_dt / dt)))
        if not np.isclose(interval * dt, self._control_dt, atol=1.0e-6):
            raise ValueError(
                f"G1 WBC control_dt={self._control_dt} must be an integer multiple of env_step_dt={dt}"
            )
        command = np.asarray(navigate_cmd, dtype=np.float64).reshape(-1)
        if command.shape != (4,) or not np.all(np.isfinite(command)):
            raise ValueError("G1 WBC navigate_cmd must contain four finite values")
        body_q, body_dq, base_quat, base_ang_vel = self._validate_state(state)
        if self._cached_command is None or self._step_index % interval == 0:
            policy_command = self._policy_command(command, self._yaw_from_wxyz(base_quat))
            observation = self._build_observation(
                body_q, body_dq, base_quat, base_ang_vel, policy_command
            )
            moving = float(np.linalg.norm(policy_command)) >= 0.05
            session = self._walk_session if moving else self._balance_session
            input_name = self._walk_input_name if moving else self._balance_input_name
            self._last_mode = "walk" if moving else "balance"
            action = self._run_session(session, input_name, observation)
            self._previous_action = action.astype(np.float32)
            target_q = body_q.copy()
            target_q[:_NUM_LOWER_BODY_JOINTS] = (
                action * 0.25 + self.default_lower_body_angles
            )
            target_q[_NUM_LOWER_BODY_JOINTS:] = self.default_upper_body_angles
            self._cached_command = MotorCommand(
                q_target=target_q,
                dq_target=np.zeros(_NUM_BODY_JOINTS, dtype=np.float64),
            )
            self._inference_count += 1
        self._step_index += 1
        return self._cached_command
