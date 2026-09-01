import pytest

from workflows.simbox.core.utils.region_metadata import (
    normalize_legacy_runtime_region_offsets,
    resolve_region_target_name,
)


@pytest.mark.parametrize(
    ("region", "expected"),
    [
        ({"target": "table", "B": "floor"}, "table"),
        ({"B": "table", "parent_fixture": "floor"}, "table"),
        ({"parent_fixture": "table"}, "table"),
        ({"support_target_fixture": "table"}, "table"),
    ],
)
def test_resolve_region_target_name_supports_runtime_schema_aliases(region, expected):
    assert resolve_region_target_name(region) == expected


def test_resolve_region_target_name_rejects_missing_support():
    with pytest.raises(KeyError, match="must define one of"):
        resolve_region_target_name({"name": "orphan", "object": "cube"})


def test_legacy_zero_centered_range_is_recentered_on_runtime_offset():
    cfg = {
        "regions": [
            {
                "object": "cover",
                "runtime_placement": {
                    "frame": "parent_world_xy_offset",
                    "offset_xy": [-0.32, 0.4268],
                },
                "random_config": {
                    "pos_range": [[-0.01, -0.02, 0.0], [0.01, 0.02, 0.0]],
                },
            }
        ]
    }

    normalize_legacy_runtime_region_offsets(cfg)

    actual = cfg["regions"][0]["random_config"]["pos_range"]
    assert actual[0] == pytest.approx([-0.33, 0.4068, 0.0])
    assert actual[1] == pytest.approx([-0.31, 0.4468, 0.0])


def test_modern_offset_centered_range_is_preserved():
    original = [[0.08, 0.18, 0.0], [0.12, 0.22, 0.0]]
    cfg = {
        "regions": [
            {
                "object": "cube",
                "runtime_placement": {
                    "frame": "parent_world_xy_offset",
                    "offset_xy": [0.1, 0.2],
                },
                "random_config": {"pos_range": [row.copy() for row in original]},
            }
        ]
    }

    normalize_legacy_runtime_region_offsets(cfg)

    assert cfg["regions"][0]["random_config"]["pos_range"] == original


def test_arbitrary_nonzero_conflict_is_not_silently_rewritten():
    original = [[0.29, 0.39, 0.0], [0.31, 0.41, 0.0]]
    cfg = {
        "regions": [
            {
                "object": "cube",
                "runtime_placement": {
                    "frame": "parent_world_xy_offset",
                    "offset_xy": [0.1, 0.2],
                },
                "random_config": {"pos_range": [row.copy() for row in original]},
            }
        ]
    }

    normalize_legacy_runtime_region_offsets(cfg)

    assert cfg["regions"][0]["random_config"]["pos_range"] == original
