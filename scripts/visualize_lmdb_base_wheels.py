#!/usr/bin/env python3
import argparse
import math
import os

import cv2
import numpy as np


WHEEL_NAMES = ["fl", "fr", "rl", "rr"]
WHEEL_MODULES = np.array(
    [
        [0.494 / 2.0, 0.364 / 2.0],
        [0.494 / 2.0, -0.364 / 2.0],
        [-0.494 / 2.0, 0.364 / 2.0],
        [-0.494 / 2.0, -0.364 / 2.0],
    ],
    dtype=np.float32,
)


def to_scalar_series(value, length):
    value = np.asarray(value)
    if value.ndim == 0:
        return np.full(length, float(value), dtype=np.float32)
    return value.reshape(length, -1)[:, 0].astype(np.float32)


def normalize_angle(values):
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def color_for_signed(value, scale):
    t = float(np.clip(abs(value) / max(scale, 1e-6), 0.0, 1.0))
    if value >= 0:
        return (int(70 * (1 - t)), int(130 + 90 * t), int(220 + 25 * t))
    return (int(220 + 25 * t), int(150 - 70 * t), int(60 * (1 - t)))


def draw_text(img, text, org, scale=0.55, color=(35, 35, 35), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_polyline_panel(
    img,
    rect,
    series,
    frame,
    labels,
    title,
    y_limit,
    colors,
    window=360,
    draw_zero=True,
):
    x0, y0, w, h = rect
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (245, 245, 245), -1)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (180, 180, 180), 1)
    draw_text(img, title, (x0 + 8, y0 + 22), 0.5, (30, 30, 30), 1)

    start = max(0, frame - window + 1)
    end = frame + 1
    if end - start < 2:
        return

    if draw_zero:
        zero_y = y0 + int(h * 0.5)
        cv2.line(img, (x0 + 45, zero_y), (x0 + w - 12, zero_y), (210, 210, 210), 1)
    draw_text(img, f"+{y_limit:g}", (x0 + 5, y0 + 40), 0.38, (80, 80, 80), 1)
    draw_text(img, f"-{y_limit:g}", (x0 + 5, y0 + h - 12), 0.38, (80, 80, 80), 1)

    plot_x0 = x0 + 45
    plot_y0 = y0 + 34
    plot_w = w - 58
    plot_h = h - 48
    xs = np.linspace(plot_x0, plot_x0 + plot_w, end - start).astype(np.int32)

    for idx, label in enumerate(labels):
        values = np.asarray(series)[:, idx]
        clipped = np.clip(values[start:end], -y_limit, y_limit)
        ys = plot_y0 + ((y_limit - clipped) / (2 * y_limit) * plot_h).astype(np.int32)
        pts = np.column_stack([xs, ys]).astype(np.int32)
        cv2.polylines(img, [pts], False, colors[idx], 1, cv2.LINE_AA)
        lx = x0 + w - 150 + (idx % 2) * 70
        ly = y0 + 22 + (idx // 2) * 16
        cv2.line(img, (lx, ly - 4), (lx + 18, ly - 4), colors[idx], 2)
        draw_text(img, label, (lx + 23, ly), 0.38, (60, 60, 60), 1)

    cursor_x = plot_x0 + int((end - start - 1) / max(window - 1, 1) * plot_w)
    cv2.line(img, (cursor_x, plot_y0), (cursor_x, plot_y0 + plot_h), (50, 50, 50), 1)


def draw_path_panel(img, rect, pose, frame):
    x0, y0, w, h = rect
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (248, 248, 248), -1)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (180, 180, 180), 1)
    draw_text(img, "base path in world xy", (x0 + 8, y0 + 22), 0.5)
    xy = pose[:, :2]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    pad = 0.12 * span.max()
    mins = mins - pad
    maxs = maxs + pad
    span = np.maximum(maxs - mins, 1e-3)
    plot_x0, plot_y0 = x0 + 35, y0 + 32
    plot_w, plot_h = w - 48, h - 48
    pts = xy[: frame + 1]
    px = plot_x0 + ((pts[:, 0] - mins[0]) / span[0] * plot_w)
    py = plot_y0 + plot_h - ((pts[:, 1] - mins[1]) / span[1] * plot_h)
    poly = np.column_stack([px, py]).astype(np.int32)
    if len(poly) > 1:
        cv2.polylines(img, [poly], False, (50, 115, 210), 2, cv2.LINE_AA)
    cx, cy = poly[-1]
    yaw = float(pose[frame, 3])
    cv2.circle(img, (cx, cy), 5, (20, 20, 20), -1)
    cv2.line(
        img,
        (cx, cy),
        (int(cx + 18 * math.cos(yaw)), int(cy - 18 * math.sin(yaw))),
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    draw_text(img, f"x={pose[frame,0]:.3f} y={pose[frame,1]:.3f}", (x0 + 8, y0 + h - 23), 0.43)
    draw_text(img, f"yaw={pose[frame,3]:.3f}", (x0 + 8, y0 + h - 7), 0.43)


def draw_wheel_panel(img, rect, steering, wheel_vel, requested_steering, requested_wheel_vel):
    x0, y0, w, h = rect
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (250, 250, 250), -1)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (180, 180, 180), 1)
    draw_text(img, "four wheel modules, body frame", (x0 + 8, y0 + 22), 0.5)
    center = np.array([x0 + w * 0.50, y0 + h * 0.54], dtype=np.float32)
    scale = min(w * 0.55 / 0.494, h * 0.55 / 0.364)
    body_w = int(0.494 * scale)
    body_h = int(0.364 * scale)
    cv2.rectangle(
        img,
        (int(center[0] - body_w / 2), int(center[1] - body_h / 2)),
        (int(center[0] + body_w / 2), int(center[1] + body_h / 2)),
        (235, 235, 235),
        -1,
    )
    cv2.rectangle(
        img,
        (int(center[0] - body_w / 2), int(center[1] - body_h / 2)),
        (int(center[0] + body_w / 2), int(center[1] + body_h / 2)),
        (90, 90, 90),
        2,
    )
    cv2.arrowedLine(
        img,
        (int(center[0]), int(center[1])),
        (int(center[0] + body_w * 0.38), int(center[1])),
        (80, 80, 80),
        2,
        tipLength=0.25,
    )
    draw_text(img, "+x", (int(center[0] + body_w * 0.4) + 4, int(center[1]) + 5), 0.42)

    vel_scale = max(8.0, float(np.nanmax(np.abs(requested_wheel_vel))) * 1.2)
    actual_scale = max(vel_scale, float(np.nanmax(np.abs(wheel_vel))) * 0.3)

    for i, name in enumerate(WHEEL_NAMES):
        module = WHEEL_MODULES[i]
        p = center + np.array([module[0] * scale, -module[1] * scale], dtype=np.float32)
        angle = float(steering[i])
        req_angle = float(requested_steering[i])
        vel = float(wheel_vel[i])
        req_vel = float(requested_wheel_vel[i])
        color = color_for_signed(vel, actual_scale)
        direction = np.array([math.cos(angle), -math.sin(angle)], dtype=np.float32)
        req_dir = np.array([math.cos(req_angle), -math.sin(req_angle)], dtype=np.float32)
        length = 30 + min(45, abs(vel) / actual_scale * 45)
        req_len = 22 + min(30, abs(req_vel) / vel_scale * 30)
        cv2.line(img, tuple((p - direction * length * 0.5).astype(int)), tuple((p + direction * length * 0.5).astype(int)), color, 5, cv2.LINE_AA)
        cv2.arrowedLine(
            img,
            tuple(p.astype(int)),
            tuple((p + direction * (length + 10) * np.sign(vel if abs(vel) > 1e-6 else 1.0)).astype(int)),
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.25,
        )
        cv2.line(
            img,
            tuple((p - req_dir * req_len * 0.5).astype(int)),
            tuple((p + req_dir * req_len * 0.5).astype(int)),
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(img, tuple(p.astype(int)), 4, (25, 25, 25), -1)
        draw_text(img, name, (int(p[0]) - 12, int(p[1]) - 32), 0.45)
        draw_text(img, f"a {angle:+.2f}", (int(p[0]) - 36, int(p[1]) + 48), 0.38)
        draw_text(img, f"v {vel:+.1f}", (int(p[0]) - 36, int(p[1]) + 63), 0.38, color, 1)

    draw_text(img, "thick/color = actual wheel velocity, thin gray = requested", (x0 + 8, y0 + h - 12), 0.42)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    data = np.load(args.npz)
    pose = np.asarray(data["states_base_pose"], dtype=np.float32)
    steering = np.asarray(data["states_base_steering_positions"], dtype=np.float32)
    wheel_vel = np.asarray(data["states_base_wheel_velocities"], dtype=np.float32)
    req_steering = np.asarray(data["base_actions_requested_steering"], dtype=np.float32)
    req_wheel_vel = np.asarray(data["base_actions_requested_wheel_velocities"], dtype=np.float32)
    vx = to_scalar_series(data["base_actions_vx_body"], len(pose))
    vy = to_scalar_series(data["base_actions_vy_body"], len(pose))
    wz = to_scalar_series(data["base_actions_wz_body"], len(pose))

    steering_error = normalize_angle(steering - req_steering)
    wheel_vel_error = wheel_vel - req_wheel_vel
    z = pose[:, 2]
    max_abs_actual_wheel = float(np.nanmax(np.abs(wheel_vel)))
    max_abs_requested_wheel = float(np.nanmax(np.abs(req_wheel_vel)))
    max_abs_steering_error = float(np.nanmax(np.abs(steering_error)))
    max_abs_wheel_error = float(np.nanmax(np.abs(wheel_vel_error)))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    width, height = 1280, 720
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {args.output}")

    colors = [(220, 80, 60), (60, 150, 220), (80, 180, 90), (190, 90, 190)]
    n = len(pose)
    frames = range(0, n, max(args.stride, 1))
    wheel_limit = max(8.0, math.ceil(max_abs_actual_wheel / 5.0) * 5.0)
    req_wheel_limit = max(8.0, math.ceil(max_abs_requested_wheel / 2.0) * 2.0)

    for frame in frames:
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        draw_text(canvas, "LMDB base wheel state visualization", (18, 32), 0.8, (20, 20, 20), 2)
        draw_text(
            canvas,
            f"frame {frame}/{n - 1}  z={z[frame]:.3f}m  cmd(vx,vy,wz)=({vx[frame]:+.2f},{vy[frame]:+.2f},{wz[frame]:+.2f})",
            (18, 60),
            0.52,
            (45, 45, 45),
            1,
        )
        draw_text(
            canvas,
            f"max |actual wheel|={max_abs_actual_wheel:.2f} rad/s, max |requested wheel|={max_abs_requested_wheel:.2f} rad/s, "
            f"max |steer err|={max_abs_steering_error:.2f} rad, max |wheel err|={max_abs_wheel_error:.2f} rad/s",
            (18, 84),
            0.48,
            (45, 45, 45),
            1,
        )

        draw_wheel_panel(
            canvas,
            (20, 105, 510, 330),
            steering[frame],
            wheel_vel[frame],
            req_steering[frame],
            req_wheel_vel[frame],
        )
        draw_path_panel(canvas, (550, 105, 705, 250), pose, frame)
        draw_polyline_panel(
            canvas,
            (550, 370, 705, 150),
            steering,
            frame,
            WHEEL_NAMES,
            "actual steering angle [rad]",
            2.2,
            colors,
        )
        draw_polyline_panel(
            canvas,
            (20, 455, 510, 115),
            req_wheel_vel,
            frame,
            WHEEL_NAMES,
            f"requested wheel velocity [rad/s], y-limit={req_wheel_limit:g}",
            req_wheel_limit,
            colors,
        )
        draw_polyline_panel(
            canvas,
            (550, 540, 705, 150),
            wheel_vel,
            frame,
            WHEEL_NAMES,
            f"actual wheel velocity [rad/s], y-limit={wheel_limit:g}",
            wheel_limit,
            colors,
        )
        draw_polyline_panel(
            canvas,
            (20, 585, 510, 105),
            np.column_stack([z, np.zeros_like(z), np.zeros_like(z), np.zeros_like(z)]),
            frame,
            ["z", "", "", ""],
            "base z height [m]",
            max(0.7, float(np.nanmax(np.abs(z))) * 1.05),
            colors,
            draw_zero=False,
        )
        writer.write(canvas)

    writer.release()
    print(args.output)
    print(f"frames_written={len(list(frames))} source_frames={n} stride={args.stride} fps={args.fps}")


if __name__ == "__main__":
    main()
