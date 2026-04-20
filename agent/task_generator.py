"""
Simple LLM-driven task config generator.

Reads an existing task YAML as a reference example, then calls
the Claude API to generate a variant based on a text instruction.

所有运行参数从 agent/config.yaml 的 defaults 块读取，无需命令行传参。

Config files (in agent/):
    .env          - API keys and secrets (copy from .env.example, never commit)
    config.yaml   - non-secret settings (model, paths, defaults, etc.)
"""

import difflib
import os

import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv


# ── Config loading ────────────────────────────────────────────────────────────

AGENT_DIR = Path(__file__).parent  # always points to the agent/ folder


def load_config() -> dict:
    """Load agent/config.yaml (non-secret settings)."""
    config_path = AGENT_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts() -> tuple[str, str]:
    """Load system prompt and user prompt template from agent/prompts/."""
    system = (AGENT_DIR / "prompts" / "system.txt").read_text(encoding="utf-8")
    user_tmpl = (AGENT_DIR / "prompts" / "user_template.txt").read_text(encoding="utf-8")
    return system, user_tmpl


def load_secrets():
    """Load agent/.env into environment variables (if the file exists)."""
    env_path = AGENT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)  # override=False: real env vars win
        print(f"Loaded secrets from {env_path}")
    else:
        print(f"Note: {env_path} not found, relying on existing environment variables.")
        print(f"      Copy agent/.env.example to agent/.env and fill in your key.")


# ── Utility functions ─────────────────────────────────────────────────────────

def load_yaml_text(path: str) -> str:
    """Read a file and return its raw text (not parsed)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_yaml_block(text: str) -> str:
    """Extract the first YAML code block from LLM response, or return raw text."""
    match = re.search(r"```(?:yaml)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def validate_yaml(text: str) -> dict:
    """Parse YAML and check that required top-level fields are present."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or "tasks" not in data:
        raise ValueError("Generated output missing 'tasks' key")
    if not isinstance(data["tasks"], list) or len(data["tasks"]) == 0:
        raise ValueError("'tasks' must be a non-empty list")
    task = data["tasks"][0]
    for required in ("name", "task", "robots", "objects", "skills"):
        if required not in task:
            raise ValueError(f"Task config missing required field: '{required}'")
    return data


# ── Inline annotation ────────────────────────────────────────────────────────

def annotate_changes(ref_path: str, generated_text: str) -> str:
    """
    对比参考文件和生成结果，在生成 YAML 中改动的行末尾加上行内注释。
      修改的行：# [modified] was: <原始值>
      新增的行：# [added]
    返回注释后的完整文本。
    """
    ref_lines  = Path(ref_path).read_text(encoding="utf-8").splitlines()
    gen_lines  = generated_text.splitlines()

    matcher = difflib.SequenceMatcher(None, ref_lines, gen_lines, autojunk=False)
    result  = []
    added_count    = 0
    modified_count = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            result.extend(gen_lines[j1:j2])

        elif tag == "replace":
            # 逐行配对；多出来的行标 [added]，少了的行在 output 里不存在故忽略
            ref_chunk = ref_lines[i1:i2]
            gen_chunk = gen_lines[j1:j2]
            for k, gen_line in enumerate(gen_chunk):
                if k < len(ref_chunk):
                    old_val = ref_chunk[k].strip()
                    result.append(f"{gen_line}  # [modified] was: {old_val}")
                    modified_count += 1
                else:
                    result.append(f"{gen_line}  # [added]")
                    added_count += 1

        elif tag == "insert":
            for gen_line in gen_lines[j1:j2]:
                result.append(f"{gen_line}  # [added]")
                added_count += 1

        elif tag == "delete":
            pass  # 删除的行在生成结果里不存在，跳过

    print(f"Annotation: {modified_count} modified, {added_count} added")
    return "\n".join(result)


# ── Provider-specific API calls ───────────────────────────────────────────────

def _call_anthropic(system: str, user: str, model: str, max_tokens: int) -> tuple[str, str]:
    """Call the Anthropic API. Returns (response_text, actual_model_used)."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: `anthropic` package not installed. Run: pip install anthropic")
        sys.exit(1)

    client_kwargs = {}
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    url = os.environ.get("ANTHROPIC_BASE_URL")
    if key:
        client_kwargs["api_key"] = key
    if url:
        client_kwargs["base_url"] = url

    client = anthropic.Anthropic(**client_kwargs)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    # message.model is the model the server actually used (may differ from requested)
    return message.content[0].text, message.model


def _call_openai(system: str, user: str, model: str, max_tokens: int) -> tuple[str, str]:
    """Call any OpenAI-compatible API. Returns (response_text, actual_model_used)."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: `openai` package not installed. Run: pip install openai")
        sys.exit(1)

    client_kwargs = {}
    key = os.environ.get("OPENAI_API_KEY")
    url = os.environ.get("OPENAI_BASE_URL")
    if key:
        client_kwargs["api_key"] = key
    if url:
        client_kwargs["base_url"] = url

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    # response.model is the model the server actually used
    return response.choices[0].message.content, response.model


# provider name → call function
_PROVIDERS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
}


# ── Core generation ───────────────────────────────────────────────────────────

def generate_task(ref_yaml_path: str, instruction: str, model: str, provider: str, max_tokens: int) -> str:
    """Load prompts, fill template, dispatch to the right provider."""
    if provider not in _PROVIDERS:
        print(f"ERROR: unknown provider '{provider}'. Choose from: {list(_PROVIDERS)}")
        sys.exit(1)

    system_prompt, user_tmpl = load_prompts()
    ref_text = load_yaml_text(ref_yaml_path)
    user_prompt = user_tmpl.format(ref_text=ref_text, instruction=instruction)

    print(f"  requested : {provider} / {model}")
    raw_text, actual_model = _PROVIDERS[provider](system_prompt, user_prompt, model, max_tokens)
    print(f"  actual    : {actual_model}")
    if actual_model != model:
        print(f"  WARNING   : proxy substituted a different model — output may differ from expectations")
    return raw_text


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    load_secrets()

    # ── 从 config.yaml 读取所有运行参数 ──────────────────────────────────────
    d = cfg["defaults"]
    provider = cfg["active_provider"]
    provider_cfg = cfg[provider]
    model = provider_cfg["fast"] if d["fast"] else provider_cfg["name"]
    max_tokens = provider_cfg["max_tokens"]
    ref = d["ref"]
    instruction = d["instruction"]
    output = d["output"]

    ref_path = Path(ref)
    if not ref_path.exists():
        print(f"ERROR: ref not found: {ref_path}")
        sys.exit(1)
    if not instruction:
        print("ERROR: defaults.instruction is empty in config.yaml")
        sys.exit(1)

    print(f"ref        : {ref}")
    print(f"instruction: {instruction}")
    print(f"provider   : {provider}  model: {model}")

    # ── 调用 API ─────────────────────────────────────────────────────────────
    raw_response = generate_task(ref, instruction, model=model, provider=provider, max_tokens=max_tokens)

    yaml_text = extract_yaml_block(raw_response)

    parsed = None
    if cfg["generation"]["validate_schema"]:
        try:
            parsed = validate_yaml(yaml_text)
            print("Schema validation passed.")
        except (ValueError, yaml.YAMLError) as e:
            print(f"WARNING: Schema validation failed: {e}")
            print("Saving output anyway.")

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_dir = "generated_task"
    if parsed:
        task_dir = parsed["tasks"][0].get("data", {}).get("task_dir", "generated_task")

    out_path = output_dir / f"{task_dir}.yaml"
    out_path.write_text(yaml_text, encoding="utf-8")
    print(f"Saved: {out_path}")

    if cfg["generation"]["save_raw_response"]:
        raw_path = output_dir / f"{task_dir}_raw_response.txt"
        raw_path.write_text(raw_response, encoding="utf-8")
        print(f"Raw response: {raw_path}")

    # ── 行内注释标注改动 ──
    if cfg["generation"].get("annotate_changes", True):
        annotated = annotate_changes(ref, yaml_text)
        annotated_path = output_dir / f"{task_dir}_annotated.yaml"
        annotated_path.write_text(annotated, encoding="utf-8")
        print(f"Annotated  : {annotated_path}")


if __name__ == "__main__":
    main()
