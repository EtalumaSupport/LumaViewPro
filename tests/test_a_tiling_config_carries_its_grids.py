# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The tiling grids an installation offers are the record's own data.

``TilingConfig`` crosses the wire as a record, and a record has no
address, so its two methods are in-process; a wire client holds its data.
Until now it had none: the grids were reachable only through the
methods. ``available`` and ``default`` are read from tiling.json once, at
construction, and the in-process methods answer from them.
"""

import json

from modules.tiling_config import TilingConfig


def _write(tmp_path, document):
    path = tmp_path / 'tiling.json'
    path.write_text(json.dumps(document), encoding='utf-8')
    return path


def test_the_grids_and_the_default_are_fields(tmp_path):
    path = _write(
        tmp_path,
        {
            'metadata': {'default': '1x1'},
            'data': {'1x1': {'m': 1, 'n': 1}, '2x2': {'m': 2, 'n': 2}},
        },
    )

    grids = TilingConfig(path)

    assert grids.available == ('1x1', '2x2')
    assert grids.default == '1x1'
    assert grids.available_configs() == ['1x1', '2x2']
    assert grids.default_config() == '1x1'


def test_a_file_that_names_no_default_says_none(tmp_path):
    grids = TilingConfig(_write(tmp_path, {'data': {'1x1': {'m': 1, 'n': 1}}}))

    assert grids.default is None
