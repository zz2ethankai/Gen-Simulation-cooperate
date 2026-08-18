"""Deterministic workspace manifests produce an explicit static gate artifact."""

from pathlib import Path

from agent.compiler import _workspace_static_validation


def _manifest(source: Path) -> dict:
    return {
        "version": 4,
        "source_task": str(source.resolve()),
        "target": {"name": "cup", "world_xyz": [0.0, 0.0, 0.7]},
        "support": {"name": "table"},
        "required_arm": "left",
        "robot": {"placement_family": "support_mounted"},
        "geometry_candidates": [
            {"candidate_id": "candidate_000", "geometry_feasible": True}
        ],
        "asset_audit": [
            {
                "name": "cup",
                "usd_exists": True,
                "scale_valid": True,
                "grasp_exists": True,
                "grasp_shape_valid": True,
                "grasp_finite": True,
            },
            {"name": "tray", "usd_exists": True, "scale_valid": True},
        ],
    }


def test_workspace_static_validation_requires_geometry_assets_and_identity(
    tmp_path: Path,
):
    source = tmp_path / "task.yaml"
    source.write_text("tasks:\n  - metadata: {}\n", encoding="utf-8")

    result = _workspace_static_validation(
        _manifest(source),
        source_task=source,
        target="cup",
        arm="left",
        placement_family="support_mounted",
    )

    assert result["hard_ok"] is True
    assert result["scene_revision"] == "source"
    assert all(result["checks"].values())

    invalid = _manifest(source)
    invalid["asset_audit"][0]["grasp_finite"] = False
    result = _workspace_static_validation(
        invalid,
        source_task=source,
        target="cup",
        arm="left",
        placement_family="support_mounted",
    )

    assert result["hard_ok"] is False
    assert result["failed_checks"] == ["assets"]
