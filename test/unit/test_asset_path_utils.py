from pathlib import Path

import pytest

from workflows.simbox.core.utils.asset_path_utils import (
    resolve_texture_paths,
    select_texture_path,
)


def test_explicit_interdata_texture_file_takes_precedence(tmp_path: Path):
    texture = tmp_path / "interdata" / "texture_libs" / "floor_textures" / "1.png"
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b"png")

    selected = select_texture_path(
        str(tmp_path),
        {
            "texture_file": "interdata/texture_libs/floor_textures/1.png",
            "texture_lib": "floor_textures",
            "apply_randomization": True,
            "texture_id": 99,
        },
    )

    assert selected == str(texture.resolve())


def test_scene_layout_texture_library_is_resolved(tmp_path: Path):
    texture = tmp_path / "interdata" / "texture_libs" / "wall_textures" / "0.jpg"
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b"jpg")

    resolved = resolve_texture_paths(str(tmp_path), "wall_textures")

    assert resolved == [str(texture.resolve())]


def test_missing_texture_has_actionable_error(tmp_path: Path):
    fallback = tmp_path / "interdata" / "texture_libs" / "fallback" / "0.png"
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"png")

    with pytest.raises(FileNotFoundError, match="texture_file='missing.png'"):
        select_texture_path(
            str(tmp_path),
            {
                "texture_file": "missing.png",
                "texture_lib": "fallback",
                "apply_randomization": True,
            },
        )
