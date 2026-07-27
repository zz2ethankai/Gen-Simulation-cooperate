from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from eval.runners import SuiteRunner
from eval.specs import eval_spec_from_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a policy evaluation suite.")
    parser.add_argument("--config", required=True, help="Path to eval YAML/JSON config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    data = _load_config(config_path)
    spec = eval_spec_from_dict(data)
    summary = SuiteRunner(spec).run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


if __name__ == "__main__":
    raise SystemExit(main())
