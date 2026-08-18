"""Command-line entry point for the InternDataEngine task agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inventory import DEFAULT_INDEX_PATH
from .orchestrator import AgentOrchestrator, DEFAULT_RUN_ROOT
from .settings import load_agent_settings
from .tools.qualification import HeldOutVariantArtifact, qualify_heldout_variants
from .tools.scene_ingest import convert_all
from .tools.simbox_diagnostics import run_probe, run_view
from workflows.simbox.core.utils.camera_template import CAMERA_TEMPLATE_DEFAULTS


AGENT_DIR = Path(__file__).resolve().parent


def _settings() -> dict:
    return load_agent_settings(AGENT_DIR / "config.yaml")


def _common_kwargs(args: argparse.Namespace, settings: dict) -> dict:
    execution = settings.get("execution", {})
    backend = settings.get("backend", {})
    paths = settings.get("paths", {})
    return {
        "gpu": int(
            getattr(args, "gpu", None)
            if getattr(args, "gpu", None) is not None
            else execution.get("gpu", 0)
        ),
        "max_revisions": int(
            getattr(args, "max_revisions", None)
            if getattr(args, "max_revisions", None) is not None
            else execution.get("max_revisions", 2)
        ),
        "conda_env": str(
            getattr(args, "conda_env", None) or execution.get("conda_env", "interndata")
        ),
        "timeout_sec": int(
            getattr(args, "timeout_sec", None)
            if getattr(args, "timeout_sec", None) is not None
            else execution.get("timeout_sec", 1800)
        ),
        "model": getattr(args, "model", None) or backend.get("model"),
        "inventory_path": Path(paths.get("inventory", DEFAULT_INDEX_PATH)),
        "run_root": Path(paths.get("run_root", DEFAULT_RUN_ROOT)),
        "retain_experience": bool(execution.get("retain_experience", True))
        and not bool(getattr(args, "no_retain", False)),
        "settings": settings,
    }


def _add_camera_template_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--camera-template",
        choices=tuple(CAMERA_TEMPLATE_DEFAULTS),
        help="Independent world-camera template",
    )
    parser.add_argument("--camera-height-m", type=float)
    parser.add_argument("--camera-look-fraction", type=float)
    parser.add_argument("--camera-look-height-m", type=float)
    parser.add_argument("--camera-behind-m", type=float)
    parser.add_argument("--camera-side-m", type=float)


def build_parser(settings: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Optional Codex model override")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="Rebuild scene/task/asset inventory")
    index.add_argument("--scene-root", action="append", type=Path, default=[])

    for name in ("plan", "run"):
        command = subparsers.add_parser(
            name, help=f"{name} a natural-language robot task"
        )
        command.add_argument("--prompt", required=True)
        command.add_argument("--gpu", type=int)
        command.add_argument("--max-revisions", type=int)
        command.add_argument("--conda-env")
        command.add_argument("--timeout-sec", type=int)
        command.add_argument("--no-retain", action="store_true")
        if name == "plan":
            command.add_argument("--run-id")

    convert = subparsers.add_parser(
        "convert", help="Convert interdata scenes into SimBox task documents"
    )
    convert.add_argument(
        "--scene-dir", nargs="*", type=Path, default=[], dest="scene_dir"
    )
    resume = subparsers.add_parser(
        "resume", help="Resume a planned or interrupted Agent run"
    )

    resume.add_argument("--run-id", required=True)
    resume.add_argument("--gpu", type=int)
    resume.add_argument("--max-revisions", type=int)
    resume.add_argument("--conda-env")
    resume.add_argument("--timeout-sec", type=int)
    resume.add_argument("--no-retain", action="store_true")

    diagnose = subparsers.add_parser(
        "diagnose", help="Reclassify the latest evidence in an Agent run"
    )
    diagnose.add_argument("--run-dir", required=True, type=Path)

    qualify = subparsers.add_parser(
        "qualify", help="Evaluate a frozen list of held-out variant artifacts"
    )
    qualify.add_argument("--artifacts", required=True, type=Path)
    qualify.add_argument("--output-dir", required=True, type=Path)

    execution = settings.get("execution", {})
    generation = settings.get("generation", {})
    default_gpu = int(execution.get("gpu", 0)) if isinstance(execution, dict) else 0
    default_seed = int(generation.get("seed", 0)) if isinstance(generation, dict) else 0

    view = subparsers.add_parser(
        "view", help="Render a deterministic scene view without running a task"
    )
    view.add_argument("--task", required=True, type=Path)
    view.add_argument("--output-dir", required=True, type=Path)
    view.add_argument("--mode", choices=("layout", "physics"), default="physics")
    view.add_argument(
        "--view",
        help="Named built-in view; physics defaults to configured debug_overview",
    )
    view.add_argument("--eye", nargs=3, type=float, metavar=("X", "Y", "Z"))
    view.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"))
    view.add_argument("--focal-length-mm", type=float)
    view.add_argument("--width", type=int)
    view.add_argument("--height", type=int)
    _add_camera_template_arguments(view)
    view.add_argument("--gpu", type=int, default=default_gpu)
    view.add_argument(
        "--include-robot",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    probe = subparsers.add_parser(
        "probe", help="Run a planning-only CuRobo workspace diagnostic"
    )
    probe.add_argument("--manifest", required=True, type=Path)
    probe.add_argument("--output-dir", required=True, type=Path)
    probe.add_argument("--candidate-id")
    probe.add_argument("--gpu", type=int, default=default_gpu)
    probe.add_argument("--arm", choices=("left", "right"))
    probe.add_argument("--planning-config", type=Path)
    probe.add_argument("--conda-env")
    probe.add_argument("--gate", choices=("pick", "pick-place"), default="pick")
    probe.add_argument("--seed", type=int, default=default_seed)
    probe.add_argument("--timeout-sec", type=int, default=900)
    probe.add_argument(
        "--collision-world", choices=("full", "target-only", "empty"), default="full"
    )
    probe.add_argument("--attach-prim-path-child", action="append", default=[])
    probe.add_argument("--disable-collision-entity", action="append", default=[])
    probe.add_argument("--disable-curobo-obstacle-path", action="append", default=[])
    probe.add_argument(
        "--disable-physics-and-curobo-obstacle-path", action="append", default=[]
    )
    probe.add_argument(
        "--capture-overview",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    probe.add_argument(
        "--capture-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    probe.add_argument(
        "--camera-eye", nargs=3, type=float, metavar=("X", "Y", "Z")
    )
    probe.add_argument(
        "--camera-target", nargs=3, type=float, metavar=("X", "Y", "Z")
    )
    probe.add_argument("--resolution", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    probe.add_argument("--focal-length-mm", type=float)
    _add_camera_template_arguments(probe)
    probe.add_argument(
        "--stop-after-feasible",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    probe.add_argument("--dry-run", action="store_true")
    return parser


def _load_qualification_artifacts(path: Path) -> list[HeldOutVariantArtifact]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("qualification artifacts JSON must be a list")
    return [HeldOutVariantArtifact.from_dict(value) for value in payload]


def main() -> int:
    settings = _settings()
    parser = build_parser(settings)
    args = parser.parse_args()
    if args.command == "qualify":
        try:
            artifacts = _load_qualification_artifacts(args.artifacts.resolve())
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            parser.error(f"invalid qualification artifacts: {exc}")
        summary = qualify_heldout_variants(artifacts, args.output_dir.resolve())
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0 if summary.qualified else 2
    if args.command in {"view", "probe"}:
        try:
            return run_view(args, settings) if args.command == "view" else run_probe(args, settings)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            parser.error(str(exc))
    orchestrator = AgentOrchestrator(**_common_kwargs(args, settings))
    if args.command == "index":
        configured_roots = [Path(value) for value in settings.get("scene_roots", [])]
        path = orchestrator.build_index(args.scene_root or configured_roots or None)
        print(path.resolve())
        return 0
    if args.command == "convert":
        roots = args.scene_dir or [
            Path(value) for value in settings.get("scene_roots", [])
        ]
        reports = convert_all(roots, settings=settings)
        for report in reports:
            status = (
                report.status
                if report.status == "skipped"
                else (report.failure_code or report.status)
            )
            print(
                f"{report.status:>9}  {report.task_id:<24} {report.scene_dir}"
                f"{'  -> ' + str(report.out_task_path) if report.status == 'converted' else ''}"
            )
        failed = [report for report in reports if report.status == "failed"]
        if failed:
            for report in failed:
                print(f"  failure_code={report.failure_code} message={report.message}")
            return 2
        return 0
    if args.command == "plan":
        state = orchestrator.plan(args.prompt, run_id=args.run_id)
        payload = state.to_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["status"] not in {"blocked", "failed"} else 2
    if args.command == "run":
        state = orchestrator.run(args.prompt)
        payload = state.to_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["status"] == "succeeded" else 2
    if args.command == "resume":
        state = orchestrator.resume(args.run_id)
        payload = state.to_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["status"] == "succeeded" else 2
    if args.command == "diagnose":
        diagnosis = orchestrator.diagnose_path(args.run_dir.resolve())
        print(json.dumps(diagnosis.to_dict(), ensure_ascii=False, indent=2))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
