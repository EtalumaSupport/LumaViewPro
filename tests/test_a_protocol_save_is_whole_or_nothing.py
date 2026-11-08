# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol save writes the whole file or leaves the previous one, and says so.

The writer opened its target for writing, which empties it, and then turned
every failure into a guessed sentence that its callers were free to ignore:
a save that failed part-way left a stub that no longer loaded, and a run
whose own copy of its protocol could not be written ran without it. A save
now raises with the operating system's reason, and the file that was there
is untouched until the new one is complete. A missing folder is made where
the save was sent.
"""

import pathlib

import pytest

import _lumascope.modules.protocol as protocol_module
from _lumascope.modules.exceptions import ProtocolNotSavedError
from tests.test_protocol_roundtrip import _build_protocol, _make_step
from tests.protocol_fixtures import protocol_from_file


def _protocol(label='first'):
    return _build_protocol([_make_step(name='A1_BF', label=label)])


def _filling_the_disk(real_open):
    """``open`` whose file takes half of what it is written, then reports the disk full."""

    def opened(*args, **kwargs):
        file = real_open(*args, **kwargs)
        real_write = file.write

        def write(text):
            real_write(text[: len(text) // 2])
            raise OSError(28, 'No space left on device')

        file.write = write
        return file

    return opened


class TestAFailedSave:
    def test_a_save_failing_part_way_leaves_the_previous_file_whole(self, tmp_path, monkeypatch):
        target = tmp_path / 'plate.tsv'
        _protocol(label='first').to_file(file_path=target)
        before = target.read_bytes()

        # Half the file is written, then the disk is full.
        monkeypatch.setattr(protocol_module, 'open', _filling_the_disk(open), raising=False)
        with pytest.raises(ProtocolNotSavedError) as raised:
            _protocol(label='second').to_file(file_path=target)
        monkeypatch.undo()

        assert isinstance(raised.value.__cause__, OSError)
        assert 'No space left on device' in str(raised.value)
        assert raised.value.file == target
        assert target.read_bytes() == before
        reloaded = protocol_from_file(
            file_path=target,
        )
        assert reloaded.steps()['Label'].to_list() == ['first']
        assert sorted(p.name for p in tmp_path.iterdir()) == ['plate.tsv'], (
            'a failed save leaves nothing beside the file'
        )


def test_a_save_into_a_missing_folder_makes_it(tmp_path):
    target = tmp_path / 'no_such_folder' / 'deeper' / 'plate.tsv'

    _protocol(label='made').to_file(file_path=target)

    assert target.is_file()


def test_a_save_replaces_the_previous_file(tmp_path):
    target = tmp_path / 'plate.tsv'
    _protocol(label='first').to_file(file_path=target)

    _protocol(label='second').to_file(file_path=str(target))

    reloaded = protocol_from_file(
        file_path=target,
    )
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
