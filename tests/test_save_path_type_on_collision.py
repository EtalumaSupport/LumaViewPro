# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A save path is a Path whether or not the name collided.

`generate_image_save_path` is annotated to return a Path, and does on a
fresh name. On a collision in 'increment' mode it handed back whatever the
collision helper returned, which was a str -- so a caller's `.name` or `/`
worked on the first capture of a session and failed on the second.
"""

import pathlib

import pytest


class _MinimalScope:
    """The collision helper keeps a scope parameter and never reads it."""


@pytest.mark.parametrize('output_format', ['TIFF', 'OME-TIFF', 'JPG'])
def test_increment_collision_returns_a_path(tmp_path, output_format):
    from modules.image_save import generate_image_save_path

    kwargs = {
        'scope': _MinimalScope(),
        'save_folder': tmp_path,
        'file_root': 'live_',
        'append': 'A1_Blue',
        'tail_id_mode': 'increment',
        'output_format': output_format,
    }
    first = generate_image_save_path(**kwargs)
    first.write_bytes(b'')
    second = generate_image_save_path(**kwargs)

    assert isinstance(second, pathlib.Path), type(second)
    assert second.parent == tmp_path
    assert second.name.startswith('live_A1_Blue_000002')


def test_next_save_path_returns_a_path(tmp_path):
    from modules.image_save import get_next_save_path

    nxt = get_next_save_path(_MinimalScope(), tmp_path / 'live_A1_Blue_000001.ome.tiff')

    assert nxt == tmp_path / 'live_A1_Blue_000002.ome.tiff'
