"""Command-line entry point for the InternDataEngine task agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inventory import DEFAULT_INDEX_PATH
from .orchestrator import AgentOrchestrator, DEFAULT_RUN_ROOT
from .settings import load_agent_settings


AGENT_DIR = Path(__file__).resolve().parent


def _settings() -> dict:
    return load_agent_settings(AGENT_DIR / "config.yaml")


def _common_kwargs(args: argparse.Namespace, settings: dict) -> dict:
    execution = settings.get("execution", {})
    backend = settings.get("backend", {})
    paths = settings.get("paths", {})
    return {
        "gpu": int(getattr(args, "gpu", None) if getattr(args, "gpu", None) is not None else execution.get("gpu", 0)),
        "max_revisions": int(
            getattr(args, "max_revisions", None)
            if getattr(args, "max_revisions", None) is not None
            else execution.get("max_revisions", 2)
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


def build_parser(settings: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Optional Codex model override")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="Rebuild scene/task/asset inventory")
    index.add_argument("--scene-root", action="append", type=Path, default=[])

    for name in ("plan", "run"):
        command = subparsers.add_parser(name, help=f"{name} a natural-language robot task")
        command.add_argument("--prompt", required=True)
        command.add_argument("--gpu", type=int)
        command.add_argument("--max-revisions", type=int)
        command.add_argument("--timeout-sec", type=int)
        command.add_argument("--no-retain", action="store_true")
        if name == "plan":
            command.add_argument("--run-id")

    resume = subparsers.add_parser("resume", help="Resume a planned or interrupted Agent run")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--gpu", type=int)
    resume.add_argument("--max-revisions", type=int)
    resume.add_argument("--timeout-sec", type=int)
    resume.add_argument("--no-retain", action="store_true")

    diagnose = subparsers.add_parser("diagnose", help="Reclassify the latest evidence in an Agent run")
    diagnose.add_argument("--run-dir", required=True, type=Path)
    return parser


def main() -> int:
    settings = _settings()
    parser = build_parser(settings)
    args = parser.parse_args()
    orchestrator = AgentOrchestrator(**_common_kwargs(args, settings))
    if args.command == "index":
        configured_roots = [Path(value) for value in settings.get("scene_roots", [])]
        path = orchestrator.build_index(args.scene_root or configured_roots or None)
        print(path.resolve())
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
