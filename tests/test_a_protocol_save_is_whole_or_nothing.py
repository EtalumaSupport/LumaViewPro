# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol save writes the whole file or leaves the previous one, and says so.

The writer opened its target for writing, which empties it, and then turned
every failure into a guessed sentence that its callers were free to ignore:
a save that failed part-way left a stub that no longer loaded, and a run
whose own copy of its protocol could not be written ran without it. A save
now raises with the operating system's reason, and the file that was there
is untouched until the new one is complete.
"""

import pathlib

import pandas as pd
import pytest

from modules.exceptions import ProtocolNotSavedError
from modules.protocol import Protocol
from tests.test_protocol_roundtrip import TILING_CONFIGS, _build_protocol, _make_step


def _protocol(label='first'):
    return _build_protocol([_make_step(name='A1_BF', label=label)])


def _disk_full(*args, **kwargs):
    raise OSError(28, 'No space left on device')


class TestAFailedSave:
    def test_a_save_failing_part_way_leaves_the_previous_file_whole(self, tmp_path, monkeypatch):
        target = tmp_path / 'plate.tsv'
        _protocol(label='first').to_file(file_path=target)
        before = target.read_bytes()

        # The header is written, then the step table's write fails.
        monkeypatch.setattr(pd.DataFrame, 'to_csv', _disk_full)
        with pytest.raises(ProtocolNotSavedError) as raised:
            _protocol(label='second').to_file(file_path=target)
        monkeypatch.undo()

        assert isinstance(raised.value.__cause__, OSError)
        assert 'No space left on device' in str(raised.value)
        assert raised.value.file == target
        assert target.read_bytes() == before
        reloaded = Protocol.from_file(file_path=target, tiling_configs_file_loc=TILING_CONFIGS)
        assert reloaded.steps()['Label'].to_list() == ['first']
        assert sorted(p.name for p in tmp_path.iterdir()) == ['plate.tsv'], (
            'a failed save leaves nothing beside the file'
        )

    def test_a_save_into_a_missing_folder_raises(self, tmp_path):
        target = tmp_path / 'no_such_folder' / 'plate.tsv'

        with pytest.raises(ProtocolNotSavedError) as raised:
            _protocol().to_file(file_path=target)

        assert isinstance(raised.value.__cause__, FileNotFoundError)
        assert not target.parent.exists()


def test_a_save_replaces_the_previous_file(tmp_path):
    target = tmp_path / 'plate.tsv'
    _protocol(label='first').to_file(file_path=target)

    _protocol(label='second').to_file(file_path=str(target))

    reloaded = Protocol.from_file(file_path=target, tiling_configs_file_loc=TILING_CONFIGS)
    assert reloaded.steps()['Label'].to_list() == ['second']
    assert [p.name for p in tmp_path.iterdir()] == ['plate.tsv']


def test_the_words_are_on_the_type():
    error = ProtocolNotSavedError(
        file=pathlib.Path('/data/plate.tsv'),
        cause=PermissionError(13, 'Permission denied'),
    )

    assert ProtocolNotSavedError.title == 'Protocol Not Saved'
    assert str(error) == (
        'The protocol was not saved to /data/plate.tsv (Permission denied). '
        'A file already there is unchanged.'
    )
