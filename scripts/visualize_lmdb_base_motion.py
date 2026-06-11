#!/usr/bin/env python3
"""
Visualize base wheel velocities and steering from LMDB data.
Outputs:
  - base_analysis.png : static composite figure
  - base_animation.mp4 : 2D top-down animation of base motion

Usage:
  .venv/bin/python3 scripts/visualize_lmdb_base_motion.py \
      --lmdb-path output/.../lmdb \
      --output-dir output/base_viz
"""

import argparse
import json
import os
import pickle
import shutil
import tempfile

import lmdb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle, FancyArrowPatch


def load_lmdb_data(lmdb_path):
    """Load relevant base-state keys from LMDB."""
    # copy to tmp to avoid locking issues
    tmpdir = tempfile.mkdtemp(prefix="lmdb_viz_")
    for fname in ("data.mdb", "info.json", "lock.mdb"):
        src = os.path.join(lmdb_path, fname)
        if os.path.exists(src):
            shutil.copy(src, tmpdir)

    env = lmdb.open(tmpdir, readonly=True, lock=False)
    data = {}
    with env.begin() as txn:
        for key in txn.cursor().iternext(keys=True, values=False):
            k = key.decode("utf-8")
            if k.startswith("states.base.") or k.startswith("base_actions."):
                data[k] = pickle.loads(txn.get(key))
    env.close()
    shutil.rmtree(tmpdir, ignore_errors=True)
    return data


def to_numpy(data_dict, key):
    """Convert list of arrays to a single numpy array."""
    val = data_dict.get(key, [])
    if not val:
        return np.array([])
    if isinstance(val[0], (list, tuple, np.ndarray)):
        return np.stack([np.asarray(v) for v in val])
    return np.array(val)


def plot_static_analysis(data, out_path):
    """Generate a static 2x2 figure."""
    pose = to_numpy(data, "states.base.pose")          # [N, 4] x,y,z,yaw
    wheel_vel = to_numpy(data, "states.base.wheel_velocities")   # [N, 4]
    steer_pos = to_numpy(data, "states.base.steering_positions") # [N, 4]
    twist = to_numpy(data, "states.base.twist_body")   # [N, 3]

    N = pose.shape[0]
    t = np.arange(N)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Base Motion Analysis ({} frames)".format(N), fontsize=14)

    # 1. XY trajectory with heading arrows
    ax = axes[0, 0]
    ax.plot(pose[:, 0], pose[:, 1], "b-", lw=1, alpha=0.6, label="Trajectory")
    # subsample arrows
    step = max(1, N // 30)
    for i in range(0, N, step):
        x, y, _, yaw = pose[i]
        dx, dy = 0.08 * np.cos(yaw), 0.08 * np.sin(yaw)
        ax.arrow(x, y, dx, dy, head_width=0.02, head_length=0.02, fc="red", ec="red", alpha=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory + Heading")
    ax.grid(True)
    ax.legend()

    # 2. Wheel velocities
    ax = axes[0, 1]
    labels = ["FL", "FR", "RL", "RR"]
    colors = ["C0", "C1", "C2", "C3"]
    for i in range(4):
        ax.plot(t, wheel_vel[:, i], color=colors[i], lw=1, label=labels[i])
    ax.set_xlabel("Frame")
    ax.set_ylabel("Wheel Velocity (rad/s)")
    ax.set_title("Wheel Velocities")
    ax.legend(loc="upper right")
    ax.grid(True)

    # 3. Steering positions
    ax = axes[1, 0]
    for i in range(4):
        ax.plot(t, steer_pos[:, i], color=colors[i], lw=1, label=labels[i])
    ax.set_xlabel("Frame")
    ax.set_ylabel("Steering Angle (rad)")
    ax.set_title("Steering Positions")
    ax.legend(loc="upper right")
    ax.grid(True)

    # 4. Base yaw
    ax = axes[1, 1]
    ax.plot(t, pose[:, 3], "k-", lw=1)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Yaw (rad)")
    ax.set_title("Base Heading (Yaw)")
    ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved static analysis -> {}".format(out_path))


def plot_animation(data, out_path, wheelbase=0.45, trackwidth=0.40):
    """
    Generate a top-down MP4 animation.
    Base drawn as a rectangle, wheels as small rectangles with steering angle,
    and wheel-velocity magnitude shown by arrow length.
    """
    pose = to_numpy(data, "states.base.pose")          # [N, 4]
    wheel_vel = to_numpy(data, "states.base.wheel_velocities")   # [N, 4]
    steer_pos = to_numpy(data, "states.base.steering_positions") # [N, 4]

    N = pose.shape[0]
    if N == 0:
        print("No pose data, skipping animation.")
        return

    # wheel offsets in body frame [front_left, front_right, rear_left, rear_right]
    half_wb = wheelbase / 2.0
    half_tw = trackwidth / 2.0
    wheel_offsets = np.array([
        [ half_wb,  half_tw],   # FL
        [ half_wb, -half_tw],   # FR
        [-half_wb,  half_tw],   # RL
        [-half_wb, -half_tw],   # RR
    ])
    wheel_colors = ["C0", "C1", "C2", "C3"]
    wheel_labels = ["FL", "FR", "RL", "RR"]

    # Determine plotting bounds with padding
    margin = 0.3
    x_min, x_max = pose[:, 0].min() - margin, pose[:, 0].max() + margin
    y_min, y_max = pose[:, 1].min() - margin, pose[:, 1].max() + margin

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Base Motion Animation")
    ax.grid(True, alpha=0.3)

    # Pre-draw full trajectory (faint)
    traj_line, = ax.plot(pose[:, 0], pose[:, 1], "lightgray", lw=1, alpha=0.5)

    # Dynamic artists
    base_rect = Rectangle((0, 0), wheelbase, trackwidth, fill=False, edgecolor="black", lw=2)
    ax.add_patch(base_rect)
    center_dot, = ax.plot([], [], "ko", markersize=6)
    heading_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="->", mutation_scale=15,
                                    color="red", lw=2)
    ax.add_patch(heading_arrow)

    wheel_arrows = []
    wheel_texts = []
    for i in range(4):
        arr = FancyArrowPatch((0, 0), (0, 0), arrowstyle="->", mutation_scale=12,
                              color=wheel_colors[i], lw=1.8, alpha=0.9)
        ax.add_patch(arr)
        wheel_arrows.append(arr)
        txt = ax.text(0, 0, wheel_labels[i], fontsize=7, color=wheel_colors[i],
                      ha="center", va="center", fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                edgecolor=wheel_colors[i], alpha=0.7))
        wheel_texts.append(txt)

    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=10,
                        verticalalignment="top", fontfamily="monospace",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    def init():
        center_dot.set_data([], [])
        heading_arrow.set_positions((0, 0), (0, 0))
        for arr in wheel_arrows:
            arr.set_positions((0, 0), (0, 0))
        for txt in wheel_texts:
            txt.set_position((0, 0))
        info_text.set_text("")
        base_rect.set_xy((0, 0))
        return [traj_line, center_dot, heading_arrow, base_rect, info_text] + wheel_arrows + wheel_texts

    def update(frame):
        x, y, z, yaw = pose[frame]
        cy, sy = np.cos(yaw), np.sin(yaw)
        R = np.array([[cy, -sy], [sy, cy]])

        # Center dot
        center_dot.set_data([x], [y])

        # Heading arrow (forward direction)
        ah_len = 0.12
        heading_arrow.set_positions(
            (x, y),
            (x + ah_len * cy, y + ah_len * sy)
        )

        # Base rectangle centered at (x,y), rotated by yaw
        base_rect.set_xy((x - half_wb, y - half_tw))
        base_rect.angle = np.degrees(yaw)

        # Wheels as vector arrows: direction = steering, length = speed magnitude
        for i in range(4):
            off = wheel_offsets[i]
            wx, wy = (R @ off) + np.array([x, y])
            steer = steer_pos[frame, i]
            wvel = wheel_vel[frame, i]
            # steering angle in world = yaw + steer
            theta = yaw + steer
            # arrow length proportional to velocity magnitude (scaled for visibility)
            length = 0.03 + 0.10 * min(abs(wvel) / 3.0, 1.0)
            # velocity sign determines arrow direction (forward/backward)
            signed_length = length if wvel >= 0 else -length
            x1 = wx + signed_length * np.cos(theta)
            y1 = wy + signed_length * np.sin(theta)
            wheel_arrows[i].set_positions((wx, wy), (x1, y1))
            # thicker line for higher speed
            wheel_arrows[i].set_linewidth(1.5 + 2.5 * min(abs(wvel) / 3.0, 1.0))
            wheel_texts[i].set_position((wx, wy))

        info_text.set_text(
            "Frame {:4d}/{:4d}\n"
            "X={:.3f}  Y={:.3f}\n"
            "Yaw={:+.3f} rad\n"
            "v_FL={:+.3f}  v_FR={:+.3f}\n"
            "v_RL={:+.3f}  v_RR={:+.3f}".format(
                frame, N - 1, x, y, yaw,
                wheel_vel[frame, 0], wheel_vel[frame, 1],
                wheel_vel[frame, 2], wheel_vel[frame, 3]
            )
        )

        # dynamic trajectory (faint tail)
        start = max(0, frame - 60)
        traj_line.set_data(pose[start:frame+1, 0], pose[start:frame+1, 1])

        return [traj_line, center_dot, heading_arrow, base_rect, info_text] + wheel_arrows + wheel_texts

    writer = FFMpegWriter(fps=30, metadata=dict(artist="base_viz"))
    with writer.saving(fig, out_path, dpi=150):
        for frame in range(N):
            update(frame)
            writer.grab_frame()
            if frame % 50 == 0:
                print("  Rendering frame {}/{}".format(frame, N))
    plt.close(fig)
    print("Saved animation -> {}".format(out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb-path", required=True,
                        help="Path to the LMDB directory (containing data.mdb)")
    parser.add_argument("--output-dir", default="output/base_viz",
                        help="Directory to save outputs")
    parser.add_argument("--wheelbase", type=float, default=0.45,
                        help="Base wheelbase in meters (front-rear distance)")
    parser.add_argument("--trackwidth", type=float, default=0.40,
                        help="Base track width in meters (left-right distance)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading LMDB from: {}".format(args.lmdb_path))
    data = load_lmdb_data(args.lmdb_path)

    if not data:
        print("No base data found in LMDB.")
        return

    # print available keys for user info
    print("Loaded keys:")
    for k in sorted(data.keys()):
        sample = data[k]
        shape = "len={}".format(len(sample)) if isinstance(sample, list) else str(type(sample))
        print("  {} -> {}".format(k, shape))

    png_path = os.path.join(args.output_dir, "base_analysis.png")
    mp4_path = os.path.join(args.output_dir, "base_animation.mp4")

    plot_static_analysis(data, png_path)
    plot_animation(data, mp4_path, wheelbase=args.wheelbase, trackwidth=args.trackwidth)
    print("Done. Outputs in: {}".format(args.output_dir))


if __name__ == "__main__":
    main()
