import pytest

from workflows.simbox.core.utils.region_metadata import resolve_region_target_name


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
