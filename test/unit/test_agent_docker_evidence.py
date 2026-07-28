import json

from agent import evidence as evidence_module
from agent.evidence import classify_evidence, collect_evidence


def test_collect_evidence_maps_workspace_paths_back_to_host(monkeypatch, tmp_path):
    monkeypatch.setattr(evidence_module, "REPO_ROOT", tmp_path)
    episode_dir = tmp_path / "output" / "episode_000"
    episode_dir.mkdir(parents=True)
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    event_path = attempt_dir / "episode_events.jsonl"
    event_path.write_text(
        json.dumps(
            {
                "status": "success",
                "primary_episode_dir": "/workspace/output/episode_000",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    log_path = attempt_dir / "stdout.log"
    log_path.write_text("Task is successful\n", encoding="utf-8")

    evidence = collect_evidence("docker:00", attempt_dir, event_path, log_path, 0, False)

    assert evidence.task_success is True
    assert evidence.episode_dir == str(episode_dir)


def test_docker_runtime_failure_is_non_retryable(tmp_path):
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    event_path = attempt_dir / "episode_events.jsonl"
    log_path = attempt_dir / "stdout.log"
    log_path.write_text(
        "DOCKER_RUNTIME_UNAVAILABLE: Docker daemon is unavailable.\n",
        encoding="utf-8",
    )

    evidence = collect_evidence("docker:01", attempt_dir, event_path, log_path, 3, False)
    diagnosis = classify_evidence(evidence)

    assert diagnosis.failure_code == "DOCKER_RUNTIME_UNAVAILABLE"
    assert diagnosis.category == "infrastructure"
    assert diagnosis.retryable is False
