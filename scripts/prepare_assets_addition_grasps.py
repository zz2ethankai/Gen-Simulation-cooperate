#!/usr/bin/env python3
"""Prepare object USD files and generate grasp labels.

For legacy assets_addition inputs, this script normalizes each task object USD
into:

    <task_obj>/<name_without_file_prefix>/Aligned_obj.usd
    <task_obj>/<name_without_file_prefix>/Aligned_obj.obj
    <task_obj>/<name_without_file_prefix>/Aligned_grasp_sparse.npy

It handles both files directly under assets/task_obj and files under
assets/task_obj/small_usd.

For downloaded scenes, pass --source download-small-objects. That mode processes
existing download/**/small_objects/**/Aligned_obj.usd files in place and writes:

    <object_dir>/Aligned_obj.obj
    <object_dir>/Aligned_grasp_sparse.npy
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PREFIX_RE = re.compile(r"^file_\d+__")


def strip_prefix(path: Path) -> str:
    return PREFIX_RE.sub("", path.stem)


def is_source_usd(path: Path) -> bool:
    if path.name != f"{path.stem}.usd":
        return False
    if path.name == "Aligned_obj.usd":
        return False
    return path.is_file()


def iter_assets_addition_task_usds(root: Path) -> list[Path]:
    paths: list[Path] = []
    for scene_dir in sorted(root.glob("file_*")):
        task_dir = scene_dir / "assets" / "task_obj"
        if not task_dir.is_dir():
            continue
        paths.extend(sorted(p for p in task_dir.glob("*.usd") if is_source_usd(p)))
        small_dir = task_dir / "small_usd"
        if small_dir.is_dir():
            paths.extend(sorted(p for p in small_dir.glob("*.usd") if is_source_usd(p)))
    return paths


def iter_assets_addition_aligned_usds(root: Path, small_only: bool = False) -> list[Path]:
    paths: list[Path] = []
    for scene_dir in sorted(root.glob("file_*")):
        task_dir = scene_dir / "assets" / "task_obj"
        if not task_dir.is_dir():
            continue
        if not small_only:
            paths.extend(sorted(task_dir.glob("*/Aligned_obj.usd")))
        paths.extend(sorted((task_dir / "small_usd").glob("*/Aligned_obj.usd")))
    return paths


def iter_download_small_object_usds(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("**/small_objects/**/Aligned_obj.usd")
        if path.is_file() and "small_objects" in path.parts
    )


def get_full_transform(prim) -> "np.ndarray":
    import numpy as np
    from pxr import UsdGeom

    transform = np.identity(4)
    current = prim
    while current and current.IsValid():
        if current.IsA(UsdGeom.Xformable):
            xform = UsdGeom.Xformable(current)
            local = xform.GetLocalTransformation()
            if isinstance(local, tuple):
                local = local[0]
            transform = np.asarray(local.GetTranspose()) @ transform
        current = current.GetParent()
    return transform


def extract_meshes(usd_path: Path) -> list[tuple[Any, list[list[int]]]]:
    import numpy as np
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    meshes: list[tuple[Any, list[list[int]]]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or counts is None or indices is None:
            continue

        vertices = np.asarray([[p[0], p[1], p[2]] for p in points], dtype=np.float64)
        if vertices.size == 0:
            continue

        transform = get_full_transform(prim)
        homog = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
        vertices = (transform @ homog.T).T[:, :3]

        faces: list[list[int]] = []
        cursor = 0
        for count in counts:
            face = list(indices[cursor : cursor + count])
            cursor += count
            if len(face) == 3:
                faces.append(face)
            elif len(face) > 3:
                faces.extend([[face[0], face[i], face[i + 1]] for i in range(1, len(face) - 1)])

        if faces:
            meshes.append((vertices, faces))
    return meshes


def export_obj(usd_path: Path, obj_path: Path) -> None:
    meshes = extract_meshes(usd_path)
    if not meshes:
        raise RuntimeError(f"No mesh geometry found in {usd_path}")
    with obj_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Exported from {usd_path}\n")
        vertex_offset = 0
        for vertices, faces in meshes:
            for vertex in vertices:
                handle.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
            for face in faces:
                indices = [index + 1 + vertex_offset for index in face]
                handle.write("f " + " ".join(str(index) for index in indices) + "\n")
            vertex_offset += len(vertices)


def prepare_layout(usd_path: Path, move: bool) -> Path:
    name = strip_prefix(usd_path)
    dst_dir = usd_path.parent / name
    dst_dir.mkdir(exist_ok=True)
    dst_usd = dst_dir / "Aligned_obj.usd"
    if usd_path.resolve() != dst_usd.resolve():
        if dst_usd.exists():
            raise FileExistsError(f"Destination already exists: {dst_usd}")
        if move:
            shutil.move(str(usd_path), str(dst_usd))
        else:
            shutil.copy2(usd_path, dst_usd)
    return dst_usd


def run_grasp(
    script: Path,
    obj_path: Path,
    sparse_num: int,
    max_width: float,
    unit: str,
    *,
    gpu: str | None = None,
) -> None:
    cmd = [
        sys.executable,
        str(script),
        "--obj_path",
        str(obj_path),
        "--unit",
        unit,
        "--sparse_num",
        str(sparse_num),
        "--max_width",
        str(max_width),
    ]
    env = os.environ.copy()
    if gpu is not None and gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    subprocess.run(cmd, check=True, cwd=str(script.parent), env=env)


def parse_gpus(gpus: str | None, jobs: int) -> list[str]:
    if gpus:
        parsed = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    else:
        parsed = [str(index) for index in range(jobs)]
    if len(parsed) < jobs:
        raise ValueError(f"--gpus must provide at least {jobs} comma-separated GPU ids")
    return parsed[:jobs]


def is_valid_sparse_grasp(path: Path, sparse_num: int) -> bool:
    if not path.is_file():
        return False
    try:
        import numpy as np

        grasp = np.load(path, mmap_mode="r")
        return (
            grasp.ndim == 2
            and grasp.shape[1] == 17
            and grasp.shape[0] <= sparse_num
            and np.issubdtype(grasp.dtype, np.number)
        )
    except Exception:  # noqa: BLE001
        return False


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_grasp_lock(lock_dir: Path, source_usd: Path) -> bool:
    while True:
        try:
            lock_dir.mkdir()
        except FileExistsError:
            try:
                pid = int((lock_dir / "pid").read_text(encoding="utf-8").strip())
            except Exception:  # noqa: BLE001
                return False
            if pid_is_alive(pid):
                return False
            shutil.rmtree(lock_dir, ignore_errors=True)
            continue
        break

    (lock_dir / "pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    (lock_dir / "source").write_text(f"{source_usd}\n", encoding="utf-8")
    return True


def release_grasp_lock(lock_dir: Path | None) -> None:
    if lock_dir is not None:
        shutil.rmtree(lock_dir, ignore_errors=True)


def run_parallel(args: argparse.Namespace) -> int:
    gpus = parse_gpus(args.gpus, args.jobs)
    script_path = Path(__file__).resolve()
    processes: list[tuple[int, str, subprocess.Popen[Any]]] = []
    print(f"[PARENT] launching {args.jobs} workers on GPUs {','.join(gpus)}", flush=True)
    for worker_index, gpu in enumerate(gpus):
        cmd = [
            sys.executable,
            str(script_path),
            *sys.argv[1:],
            "--worker-index",
            str(worker_index),
            "--worker-count",
            str(args.jobs),
            "--worker-gpu",
            gpu,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        proc = subprocess.Popen(cmd, cwd=Path.cwd(), env=env)
        processes.append((worker_index, gpu, proc))
        print(f"[PARENT] worker {worker_index}/{args.jobs} pid={proc.pid} gpu={gpu}", flush=True)

    failed = False
    for worker_index, gpu, proc in processes:
        return_code = proc.wait()
        print(
            f"[PARENT] worker {worker_index}/{args.jobs} gpu={gpu} exited with {return_code}",
            flush=True,
        )
        if return_code != 0:
            failed = True
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["assets-addition", "download-small-objects"],
        default="assets-addition",
        help="Which asset layout to scan.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Asset root. Defaults to InternDataAssets/assets/assets_addition for "
            "--source assets-addition and download for --source download-small-objects."
        ),
    )
    parser.add_argument("--grasp-script", type=Path, default=Path("workflows/simbox/tools/grasp/gen_sparse_label.py"))
    parser.add_argument("--sparse-num", type=int, default=3000)
    parser.add_argument("--max-width", type=float, default=0.1)
    parser.add_argument("--unit", choices=["mm", "m"], default="m")
    parser.add_argument("--move", action="store_true", help="Move source USDs instead of copying them.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--layout-only", action="store_true", help="Only normalize USD locations and names.")
    parser.add_argument("--process-aligned", action="store_true", help="Process existing */Aligned_obj.usd files.")
    parser.add_argument("--small-only", action="store_true", help="Only process assets/task_obj/small_usd objects.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many assets.")
    parser.add_argument("--jobs", type=int, default=1, help="Run this many parallel worker processes.")
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids for parallel workers. Defaults to 0..jobs-1.",
    )
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-gpu", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    if args.jobs > 1 and args.worker_index is None:
        return run_parallel(args)

    root = (
        args.root
        or (
            Path("download")
            if args.source == "download-small-objects"
            else Path("InternDataAssets/assets/assets_addition")
        )
    ).resolve()
    grasp_script = args.grasp_script.resolve()
    if args.source == "download-small-objects":
        usds = iter_download_small_object_usds(root)
        if not args.process_aligned:
            print("[INFO] download-small-objects always processes existing Aligned_obj.usd files in place")
            args.process_aligned = True
    else:
        usds = (
            iter_assets_addition_aligned_usds(root, small_only=args.small_only)
            if args.process_aligned
            else iter_assets_addition_task_usds(root)
        )
    if args.limit is not None:
        usds = usds[: args.limit]
    discovered_count = len(usds)
    if args.skip_existing and not args.layout_only:
        pending_usds = [
            path
            for path in usds
            if not is_valid_sparse_grasp(path.with_name("Aligned_grasp_sparse.npy"), args.sparse_num)
        ]
        print(
            f"Found {discovered_count} USD files; skipping {discovered_count - len(pending_usds)} valid completed"
        )
        usds = pending_usds
    else:
        print(f"Found {discovered_count} USD files")

    dynamic_worker_queue = (
        args.worker_index is not None
        and args.process_aligned
        and args.skip_existing
        and not args.layout_only
        and not args.dry_run
    )
    if args.worker_index is not None:
        if args.worker_count is None or args.worker_count < 1:
            raise SystemExit("--worker-count must be set when --worker-index is used")
        if args.worker_index < 0 or args.worker_index >= args.worker_count:
            raise SystemExit("--worker-index must be in [0, --worker-count)")
        if not dynamic_worker_queue:
            usds = usds[args.worker_index :: args.worker_count]
        print(
            (
                f"[WORKER {args.worker_index}/{args.worker_count} gpu={args.worker_gpu}] "
                f"{'scanning' if dynamic_worker_queue else 'assigned'} {len(usds)} USD files"
            ),
            flush=True,
        )

    if args.dry_run:
        for path in usds[:20]:
            if args.process_aligned:
                print(path)
            else:
                name = strip_prefix(path)
                print(f"{path} -> {path.parent / name / 'Aligned_obj.usd'}")
        if len(usds) > 20:
            print(f"... {len(usds) - 20} more")
        return 0

    failures: list[tuple[Path, str]] = []
    for index, source_usd in enumerate(usds, start=1):
        lock_dir: Path | None = None
        try:
            dst_usd = source_usd if args.process_aligned else prepare_layout(source_usd, move=args.move)
            if args.layout_only:
                print(f"[{index}/{len(usds)}] {source_usd} -> {dst_usd}")
                continue
            obj_path = dst_usd.with_suffix(".obj")
            grasp_path = dst_usd.with_name("Aligned_grasp_sparse.npy")
            prefix = (
                f"[worker {args.worker_index}/{args.worker_count} gpu={args.worker_gpu}] "
                if args.worker_index is not None
                else ""
            )
            if args.skip_existing and is_valid_sparse_grasp(grasp_path, args.sparse_num):
                if not dynamic_worker_queue:
                    print(f"{prefix}SKIP valid existing {grasp_path}", flush=True)
                continue
            if args.skip_existing:
                lock_dir = grasp_path.with_name(f"{grasp_path.name}.lock")
                if not acquire_grasp_lock(lock_dir, dst_usd):
                    if not dynamic_worker_queue:
                        print(f"{prefix}SKIP locked {grasp_path}", flush=True)
                    lock_dir = None
                    continue
                if is_valid_sparse_grasp(grasp_path, args.sparse_num):
                    print(f"{prefix}SKIP valid existing {grasp_path}", flush=True)
                    continue
            print(f"{prefix}[{index}/{len(usds)}] {dst_usd}", flush=True)
            if grasp_path.exists() and not is_valid_sparse_grasp(grasp_path, args.sparse_num):
                print(f"{prefix}REBUILD invalid existing {grasp_path}", flush=True)
            if args.skip_existing and is_valid_sparse_grasp(grasp_path, args.sparse_num):
                print(f"{prefix}SKIP valid existing {grasp_path}", flush=True)
                continue
            if not obj_path.exists():
                export_obj(dst_usd, obj_path)
            run_grasp(grasp_script, obj_path, args.sparse_num, args.max_width, args.unit, gpu=args.worker_gpu)
        except Exception as exc:  # noqa: BLE001
            failures.append((source_usd, str(exc)))
            print(f"FAILED {source_usd}: {exc}", file=sys.stderr)
        finally:
            release_grasp_lock(lock_dir)

    if failures:
        print("\nFailures:")
        for path, message in failures:
            print(f"- {path}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
