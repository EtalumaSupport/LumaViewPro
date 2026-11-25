# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``path_utils.written_whole``: a file is written whole or not at all.

The one routine every whole-file write in LVP goes through. A block that
ends moves its temporary onto the target; one that raises leaves the
target as it was and nothing beside it; the temporary's name is the
target's whole name, 8 hex characters and ``.tmp``.
"""

from __future__ import annotations

import os
import re

import pytest

from _lumascope.modules import image_utils, path_utils


def test_a_block_that_ends_replaces_the_target(tmp_path):
    target = tmp_path / 'protocol.tsv'
    target.write_text('old', encoding='utf-8')

    with path_utils.written_whole(target) as temporary:
        temporary.write_text('new', encoding='utf-8')

    assert target.read_text(encoding='utf-8') == 'new'
    assert sorted(tmp_path.iterdir()) == [target]


# parametrize-justified: an Exception and a KeyboardInterrupt are the two
# kinds a block can leave by, and the routine removes the temporary on both.
@pytest.mark.parametrize('stop', [OSError(28, 'No space left on device'), KeyboardInterrupt()])
def test_a_block_that_raises_partway_leaves_the_earlier_file_and_nothing_beside_it(tmp_path, stop):
    target = tmp_path / 'protocol.tsv'
    target.write_text('old', encoding='utf-8')

    with pytest.raises(type(stop)) as raised, path_utils.written_whole(target) as temporary:
        temporary.write_text('ne', encoding='utf-8')
        raise stop

    assert raised.value is stop
    assert target.read_text(encoding='utf-8') == 'old'
    assert sorted(tmp_path.iterdir()) == [target]


def test_a_failing_move_raises_its_os_error_and_leaves_nothing_beside_it(tmp_path):
    # A folder at the target's name: the move onto it fails on every platform.
    target = tmp_path / 'results.csv'
    target.mkdir()

    with pytest.raises(OSError), path_utils.written_whole(target) as temporary:
        temporary.write_text('whole', encoding='utf-8')

    assert sorted(tmp_path.iterdir()) == [target]
    assert list(target.iterdir()) == []


def test_each_temporary_is_named_from_its_whole_target_and_no_image_search_takes_it(tmp_path):
    # The support report blanks the API key in every name beginning
    # current.json; a leftover ending in its target's kind would be counted
    # as one.
    names = []
    for target in (tmp_path / 'current.json', tmp_path / 'image_enhanced.ome.tiff'):
        for _ in range(2):
            with path_utils.written_whole(target) as temporary:
                names.append(temporary.name)
                assert temporary.parent == tmp_path
                assert not temporary.exists()
                temporary.write_bytes(b'')

    for name in names[:2]:
        assert re.fullmatch(r'current\.json\.[0-9a-f]{8}\.tmp', name), name
    for name in names[2:]:
        assert re.fullmatch(r'image_enhanced\.ome\.tiff\.[0-9a-f]{8}\.tmp', name), name
        assert not image_utils.is_image(name)
    assert names[0] != names[1] and names[2] != names[3]


def test_a_target_of_242_characters_is_written(tmp_path):
    # The temporary adds 13 characters; NAME_MAX is 255 on APFS, NTFS and ext4.
    target = tmp_path / ('a' * 238 + '.tsv')
    assert len(target.name) == 242

    with path_utils.written_whole(target) as temporary:
        temporary.write_text('whole', encoding='utf-8')

    assert target.read_text(encoding='utf-8') == 'whole'


def test_the_json_form_writes_the_whole_file_through_the_routine(tmp_path):
    target = tmp_path / 'current.json'
    target.write_text('{"old": 1}', encoding='utf-8')

    path_utils.write_whole(target, {'new': 2}, indent=4)

    assert target.read_text(encoding='utf-8') == '{\n    "new": 2\n}'
    assert sorted(os.listdir(tmp_path)) == ['current.json']
